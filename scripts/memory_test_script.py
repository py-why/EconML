
import argparse
import datetime
import joblib
import logging
import numpy as np
import os
import sys
import time
import tracemalloc
import pandas as pd
import gc
import types
import weakref
import warnings


from econml.dml import CausalForestDML
from memory_profiler import memory_usage
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# More aggressive warning suppression
# Filter all warnings from scipy
warnings.filterwarnings("ignore", module="scipy")
# Also try to catch other possible warning categories
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Please import")

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def get_size_of_object(obj):
    """Get memory size of an object in bytes"""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    elif isinstance(obj, pd.DataFrame):
        return obj.memory_usage(deep=True).sum()
    elif hasattr(obj, 'nbytes'):  # Other objects with nbytes attribute
        return obj.nbytes
    else:
        # Approximate size for other objects
        return sys.getsizeof(obj)


def analyze_object_memory(obj, name="object", max_depth=100):
    """
    Analyze a specific object for large arrays and DataFrames

    Args:
        obj: The object to analyze
        name: Name to use for the object's root path
        max_depth: Maximum recursion depth

    Returns
    -------
        List of large arrays/DataFrames found in the object
    """
    # Force garbage collection
    gc.collect()
    print(f"Analyzing memory usage of {name}...")

    # Get results using the find_arrays_in_object function
    results = find_arrays_in_object(obj, name, max_depth=max_depth)

    # Sort by size
    results.sort(key=lambda x: x["size_mb"], reverse=True)

    # Deduplicate by object id
    unique_results = []
    seen_ids = set()
    for result in results:
        obj_id = id(result["object"])
        if obj_id not in seen_ids:
            seen_ids.add(obj_id)
            unique_results.append(result)

    sums = {}
    counts = {}
    for item in unique_results:
        # Extract the last component of the path (after the last period)
        last_component = item["path"].split(".")[-1]

        # Add the size_mb to the sum for this last component
        if last_component in sums:
            sums[last_component] += item["size_mb"]
            counts[last_component] += 1
        else:
            sums[last_component] = item["size_mb"]
            counts[last_component] = 1

    sums = dict(sorted(sums.items(), key=lambda x: x[1], reverse=True))

    df = pd.DataFrame({
        "obj_name": list(sums.keys()),
        "total_size_mb": list(sums.values()),
        "count": [counts[key] for key in sums.keys()]
    })

    # Sort the DataFrame by total_size_mb in descending order
    df = df.sort_values(by="total_size_mb", ascending=False).reset_index(drop=True)
    print(df)

    return df


def find_arrays_in_object(obj, path="", visited=None, results=None, max_depth=10):
    """
    Recursively search an object and its attributes for NumPy arrays and DataFrames

    Args:
        obj: The object to search
        path: Current attribute path (for tracking)
        visited: Set of already visited object IDs
        results: List to collect results
        max_depth: Maximum recursion depth
    """
    if visited is None:
        visited = set()
    if results is None:
        results = []

    # Stop if we've seen this object before to prevent infinite recursion
    obj_id = id(obj)
    if obj_id in visited:
        return results

    # Add to visited set
    visited.add(obj_id)

    # Check if this object is a numpy array or DataFrame
    if isinstance(obj, np.ndarray):
        size_mb = float(obj.nbytes) / float(1024 * 1024)
        results.append({
            "type": "ndarray",
            "path": path,
            "shape": obj.shape,
            "dtype": obj.dtype,
            "size_mb": size_mb,
            "object": obj
        })
        return results
    elif isinstance(obj, pd.DataFrame):
        size_mb = float(obj.memory_usage(deep=True).sum()) / float(1024 * 1024)
        results.append({
            "type": "DataFrame",
            "path": path,
            "shape": obj.shape,
            "size_mb": size_mb,
            "object": obj,
            "columns": list(obj.columns)[:5] + ['...'] if len(obj.columns) > 5 else list(
                obj.columns)
        })
        return results
        # Skip certain types that don't contain arrays
    elif (isinstance(obj, (str, int, float, bool, type, types.FunctionType, types.MethodType, weakref.ref)) or obj is None):
        return results

    # For dictionaries
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, (str, int, float, bool, tuple)):
                new_path = f"{path}['{key}']" if path else f"['{key}']"
                find_arrays_in_object(value, new_path, visited, results,
                                      max_depth)
        return results

    # For lists and tuples
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]" if path else f"[{i}]"
            find_arrays_in_object(item, new_path, visited, results, max_depth)
        return results

    else:
        for attr_name in sorted(list(dir(obj))):
            if not (attr_name.startswith('__') or attr_name.startswith('_')):  # Skip special methods
                try:
                    attr_value = getattr(obj, attr_name)
                    if not callable(attr_value):
                        new_path = f"{path}.{attr_name}" if path else attr_name
                        find_arrays_in_object(attr_value, new_path, visited, results,
                                          max_depth)
                except Exception:
                    continue
        return results


def balanced_downsample(X:pd.DataFrame, y:np.array, T_feat:pd.DataFrame, t_id:np.array, downsample_ratio:float):
    """
    Balanced downsampling based on treatment identity

    :param X:
    :param y:
    :param T_feat: features of the treatments
    :param t_id: which treatment was applied
    :return:
    """
    # Keep all the positive experiences
    pos_ex = y > 0
    positive_df = X[pos_ex].reset_index(drop=True)
    positive_actions = t_id[pos_ex]
    act_feature_positve = T_feat[pos_ex].reset_index(drop=True)
    y_positive = y[pos_ex]
    positive_df['action_id'] = positive_actions
    positive_df['y'] = y_positive
    positive_df = pd.concat([positive_df,act_feature_positve],axis=1).reset_index(drop=True)

    negative_df = X[~pos_ex].reset_index(drop=True)
    negative_actions = t_id[~pos_ex]
    act_feature_negative = T_feat[~pos_ex].reset_index(drop=True)
    y_negative = y[~pos_ex]

    negative_df['action_id'] = negative_actions
    negative_df['y'] = y_negative
    negative_df = pd.concat([negative_df,act_feature_negative],axis=1).reset_index(drop=True)

    df_downsampled = negative_df.groupby('action_id', group_keys=False).apply(lambda x: x.sample(frac=downsample_ratio))

    logger.info(f'Dowsampled negative examples by {downsample_ratio}')
    actions_before = negative_df['action_id'].value_counts()
    actions_after = df_downsampled['action_id'].value_counts()
    # logger.info(f'Before: {actions_before}')
    # logger.info(f'After: {actions_after}')

    # Final sample
    combo_df   = pd.concat([positive_df,df_downsampled],axis=0).reset_index(drop=True)
    # this shuffles it
    combo_df = combo_df.sample(frac=1).reset_index(drop=True)
    X = combo_df[X.columns.values]
    T_feat= combo_df[T_feat.columns.values]
    y = combo_df['y'].to_numpy()
    t_id = combo_df['action_id'].to_numpy()

    n_pos_ex2 = y.sum()
    logger.info(f"After downsampling non-conversion, Conversion rate  now {n_pos_ex2/len(y)}")
    logger.info(f"X {X.shape}")
    logger.info(f"T {T_feat.shape}")
    logger.info(f"y {y.shape}")

    return X, y, T_feat, t_id


def causalforestdml_memory_test(
    file_name: str,
    estimator: str,
    n_est_y: int,
    n_est_t: int,
    n_est_2: int,
    n_jobs: int,
    use_memmap: bool,
    downsample:float|None,
):

    if estimator.lower() not in ('causalforest','catboost'):
        raise NotImplementedError(f"Estimator must be in 'causalforest','catboost', got {estimator}")

    root_dir = os.environ.get("LOCAL_ANALYSIS_DIR", ".")
    logger.info(f"Using root dir {root_dir} loading data file {file_name}")
    data_file = os.path.join(os.path.abspath(root_dir), file_name)
    data = joblib.load(data_file)

    # This is specific to Offerfit - names of the data are derived from contextual bandits
    X = data['X']
    T = data['T']
    y = data['Y']
    i = data['i']
    if downsample is not None:
        if downsample > 1:
            downsample = float(downsample)/float(len(y))
        elif downsample <=0 or downsample==1:
            raise ValueError("Downsample needs to be >1 or 0 < downsample < 1")
        X, y, T, i = balanced_downsample(X,y,T,i,downsample_ratio=downsample)

    logger.info(f"X has a shape of: {X.shape}")
    logger.info(f"T has a shape of: {T.shape}")
    logger.info(f"y has a shape of: {y.shape}, total={np.sum(y)}")

    if estimator == 'causalforest':
        est = CausalForestDML(
            model_t=XGBRegressor(n_estimators=n_est_t),
            model_y=XGBRegressor(n_estimators=n_est_y),
            discrete_outcome=False,
            n_jobs=n_jobs,
            n_estimators=n_est_2,
            use_memmap=use_memmap
        )
        logger.info(f"Calling CausalForestDML fit: njobs={n_jobs}, MemMap={use_memmap},"
                    f"N estimators y={n_est_y}, t={n_est_t}, 2={n_est_2}")

    elif estimator == 'catboost':
        est = CatBoostRegressor(n_estimators=n_est_2, allow_writing_files=False)
        logger.info(f"Calling CatBoostRegressor fit: n_estimators={n_est_2}")

    tracemalloc.start()
    start_time = time.time()
    if estimator == 'causalforest':
        mem_usage = memory_usage(
            (est.fit, [y,T], {"X":X}),
            interval=0.1,  # Sample every 0.1 seconds
            timeout=None,  # No timeout
            max_usage=True,  # Get maximum memory used
            retval=True,  # Return the fitted model too
            include_children=True  # Include joblib's child processes
        )
    else:
        mem_usage = memory_usage(
            (est.fit, [X], {"y": y, "silent": True}),
            interval=0.1,  # Sample every 0.1 seconds
            timeout=None,  # No timeout
            max_usage=True,  # Get maximum memory used
            retval=True,  # Return the fitted model too
            include_children=True  # Include joblib's child processes
        )

    end_time = time.time()

    # mem_usage section
    max_memory, fitted_model = mem_usage
    elapsed_time = end_time-start_time
    logger.info(f"Maximum memory usage during fit: {max_memory} MiB")
    logger.info(f"Time to fit: {elapsed_time} seconds")

    mem_df = analyze_object_memory(est,name=estimator, max_depth=100)

    array_file_name = os.path.join(root_dir,"mem_test_data_arrays.csv")
    mem_df.to_csv(array_file_name)

    result_file_name = os.path.join(root_dir,"mem_test_results.csv")
    if not os.path.isfile(result_file_name):
        with open(result_file_name,"w") as output:
            output.write("data,estimator,run_time,N_examples,N_nuisances,N_treatments,max_memory_mb,fit_time_secs,top_obj,top_obj_mem\n")

    run_time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(result_file_name,"a") as output:
        output.write(f"{file_name},{estimator},{run_time_string},{X.shape[0]},{X.shape[1]},{T.shape[1]},{max_memory:.1f},{elapsed_time:.1f},{mem_df.iloc[0][0]},{mem_df.iloc[0][1]}\n")


    logger.info('Done')

    # # malloc section
    # malloc_snapshot = tracemalloc.take_snapshot()
    # top_stats = malloc_snapshot.statistics('lineno')
    # print("\nTop 10 *tracemalloc* memory-consuming locations:")
    # for i, stat in enumerate(top_stats[:10], 1):
    #     print(f"{i}. {stat}")
    #
    # # Get statistics grouped by object type (more useful for identifying large objects)
    # top_types = malloc_snapshot.statistics('traceback')
    # print("\nTop 10  *tracemalloc* memory-consuming object types:")
    # for i, stat in enumerate(top_types[:10], 1):
    #     print(f"{i}. {stat}")
    #     # Print the traceback to see where the object was created
    #     for line in stat.traceback.format():
    #         print(f"    {line}")
    #
    #
    # # muppy section
    # all_objects = muppy.get_objects()
    # # Get a summary of memory usage by type
    # mem_summary = summary.summarize(all_objects)
    # print("*muppy* Memory usage summary:")
    # summary.print_(mem_summary)
    # # Look at NumPy arrays
    # numpy_arrays = [obj for obj in all_objects if isinstance(obj, np.ndarray)]
    # numpy_arrays.sort(key=lambda arr: arr.nbytes, reverse=True)
    #
    # print("\nTop 10 NumPy arrays by memory usage from *muppy*:")
    # for i, arr in enumerate(numpy_arrays[:10], 1):
    #     memory_mb = arr.nbytes / (1024 * 1024)
    #     print(f"{i}. Shape: {arr.shape}, Dtype: {arr.dtype}, Memory: {memory_mb:.2f} MB")
    #
    # pandas_dfs = [obj for obj in all_objects if isinstance(obj, pd.DataFrame)]
    # pandas_dfs.sort(key=lambda df: df.memory_usage(deep=True).sum(), reverse=True)
    #
    # print("\nTop 10 Pandas DataFrames by memory usage *muppy*:")
    # for i, df in enumerate(pandas_dfs[:10], 1):
    #     memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    #     print(f"{i}. Shape: {df.shape}, Memory: {memory_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="", default="offerfit.joblib")
    parser.add_argument("--n_est_y", type=int, help="", default=None)
    parser.add_argument("--n_est_t", type=int, help="", default=None)
    parser.add_argument("--n_est_2", type=int, help="", default=500)
    parser.add_argument("--n_jobs", type=int, help="", default=-1)
    parser.add_argument("--memmap", type=bool, help="", default=False)
    parser.add_argument("--estimator", type=str, help="", default=None)
    parser.add_argument("--downsample", type=float, help="", default=None)
    args = parser.parse_args(sys.argv[1:])
    data_file = args.data_file

    causalforestdml_memory_test(
        file_name=data_file,
        estimator=args.estimator,
        n_est_y=args.n_est_y,
        n_est_t=args.n_est_t,
        n_est_2=args.n_est_2,
        n_jobs=args.n_jobs,
        use_memmap=args.memmap,
        downsample=args.downsample
    )
