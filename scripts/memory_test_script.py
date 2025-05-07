
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

from econml.dml import CausalForestDML
from memory_profiler import memory_usage
from pympler import muppy, summary
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


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
    logger.info(f'Before: {actions_before}')
    logger.info(f'After: {actions_after}')

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

    root_dir = os.environ.get("LOCAL_ANALYSIS_DIR", ".")
    logger.info(f"Using root dir {root_dir} loading data file {file_name}")
    data_file = os.path.join(os.path.abspath(root_dir), file_name)
    data = joblib.load(data_file)

    # This is specific to Offerfit - names of the data are derived from contextual bandits
    X = data['X']
    T = data['T']
    y = data['Y']
    i = data['i']
    if downsample:
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
    logger.info(f"Maximum memory usage using memory_usage: {max_memory} MiB")
    logger.info(f"Time to fit: {elapsed_time} seconds")

    result_file_name = os.path.join(root_dir,"mem_test_results.csv")
    if not os.path.isfile(result_file_name):
        with open(result_file_name,"w") as output:
            output.write("data,estimator,run_time,N_examples,N_nuisances,N_treatments,max_memory_mb,fit_time_secs\n")

    run_time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(result_file_name,"a") as output:
        output.write(f"{file_name},{estimator},{run_time_string},{X.shape[0]},{X.shape[1]},{T.shape[1]},{max_memory:.1f},{elapsed_time:.1f}\n")


    # malloc section
    malloc_snapshot = tracemalloc.take_snapshot()
    top_stats = malloc_snapshot.statistics('lineno')
    print("\nTop 10 *tracemalloc* memory-consuming locations:")
    for i, stat in enumerate(top_stats[:10], 1):
        print(f"{i}. {stat}")

    # Get statistics grouped by object type (more useful for identifying large objects)
    top_types = malloc_snapshot.statistics('traceback')
    print("\nTop 10  *tracemalloc* memory-consuming object types:")
    for i, stat in enumerate(top_types[:10], 1):
        print(f"{i}. {stat}")
        # Print the traceback to see where the object was created
        for line in stat.traceback.format():
            print(f"    {line}")


    # muppy section
    all_objects = muppy.get_objects()
    # Get a summary of memory usage by type
    mem_summary = summary.summarize(all_objects)
    print("*muppy* Memory usage summary:")
    summary.print_(mem_summary)
    # Look at NumPy arrays
    numpy_arrays = [obj for obj in all_objects if isinstance(obj, np.ndarray)]
    numpy_arrays.sort(key=lambda arr: arr.nbytes, reverse=True)

    print("\nTop 10 NumPy arrays by memory usage from *muppy*:")
    for i, arr in enumerate(numpy_arrays[:10], 1):
        memory_mb = arr.nbytes / (1024 * 1024)
        print(f"{i}. Shape: {arr.shape}, Dtype: {arr.dtype}, Memory: {memory_mb:.2f} MB")

    pandas_dfs = [obj for obj in all_objects if isinstance(obj, pd.DataFrame)]
    pandas_dfs.sort(key=lambda df: df.memory_usage(deep=True).sum(), reverse=True)

    print("\nTop 10 Pandas DataFrames by memory usage *muppy*:")
    for i, df in enumerate(pandas_dfs[:10], 1):
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(f"{i}. Shape: {df.shape}, Memory: {memory_mb:.2f} MB")


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
