
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


def causalforestdml_memory_test(
    test_type:str,
    file_name: str,
    estimator: str,
    n_est_y: int,
    n_est_t: int,
    n_est_2: int,
    n_jobs: int,
    use_memmap: bool
):
    if test_type not in ("profile","malloc","object"):
        raise ValueError(f"Test type should be 'profile'|'malloc'|'object' but got {test_type}")
    if not estimator.lower() in ('causalforest','catboost'):
        raise NotImplementedError(f"Estimator must be in 'causalforest','catboost', got {estimator}")

    root_dir = os.environ.get("LOCAL_ANALYSIS_DIR", ".")
    logger.info(f"Using root dir {root_dir} loading data file {file_name}")
    data_file = os.path.join(os.path.abspath(root_dir), file_name)
    data = joblib.load(data_file)

    # This is specific to Offerfit - names of the data are derived from contextual bandits
    X = data['X']
    T = data['T']
    y = data['Y']

    logger.info(f"X has a shape of: {X.shape}")
    logger.info(f"T has a shape of: {T.shape}")
    logger.info(f"y has a shape of: {y.shape}")

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

        start_time = time.time()
        if test_type=='profile':
            mem_usage = memory_usage(
                (est.fit, [y,T], {"X":X}),
                interval=0.1,  # Sample every 0.1 seconds
                timeout=None,  # No timeout
                max_usage=True,  # Get maximum memory used
                retval=True,  # Return the fitted model too
                include_children=True  # Include joblib's child processes
            )
        elif test_type=='malloc':
            tracemalloc.start()
            est.fit(y,T, X=X)
            malloc_snapshot = tracemalloc.take_snapshot()
        else:
            est.fit(y,T, X=X)

        end_time = time.time()
    elif estimator == 'catboost':
        est = CatBoostRegressor(n_estimators=n_est_2, allow_writing_files=False)

        logger.info(f"Calling CatBoostRegressor fit: n_estimators={n_est_2}")

        start_time = time.time()
        if test_type=='profile':
            mem_usage = memory_usage(
                (est.fit, [X], {"y": y, "silent": True}),
                interval=0.1,  # Sample every 0.1 seconds
                timeout=None,  # No timeout
                max_usage=True,  # Get maximum memory used
                retval=True,  # Return the fitted model too
                include_children=True  # Include joblib's child processes
            )
        elif test_type=='malloc':
            tracemalloc.start()
            est.fit(X,y=y,silent=True)
            malloc_snapshot = tracemalloc.take_snapshot()
        else:
            est.fit(X,y=y,silent=True)

        end_time = time.time()

    # Extract results
    if test_type == 'malloc':
        # Get statistics grouped by filename and line number
        top_stats = malloc_snapshot.statistics('lineno')
        print("\nTop 10 memory-consuming locations:")
        for i, stat in enumerate(top_stats[:10], 1):
            print(f"{i}. {stat}")

        # Get statistics grouped by object type (more useful for identifying large objects)
        top_types = malloc_snapshot.statistics('traceback')
        print("\nTop 10 memory-consuming object types:")
        for i, stat in enumerate(top_types[:10], 1):
            print(f"{i}. {stat}")
            # Print the traceback to see where the object was created
            for line in stat.traceback.format():
                print(f"    {line}")
    elif test_type == 'profile':
        max_memory, fitted_model = mem_usage
        elapsed_time = end_time-start_time
        logger.info(f"Maximum memory usage: {max_memory} MiB")
        logger.info(f"Time to fit: {elapsed_time} seconds")

        result_file_name = os.path.join(root_dir,"mem_test_results.csv")
        if not os.path.isfile(result_file_name):
            with open(result_file_name,"w") as output:
                output.write("data,estimator,run_time,N_examples,N_nuisances,N_treatments,max_memory_mb,fit_time_secs\n")

        run_time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        with open(result_file_name,"a") as output:
            output.write(f"{file_name},{estimator},{run_time_string},{X.shape[0]},{X.shape[1]},{T.shape[1]},{max_memory:.1f},{elapsed_time:.1f}\n")
    elif test_type=='object':
        # Collect all objects
        all_objects = muppy.get_objects()
        # Get a summary of memory usage by type
        mem_summary = summary.summarize(all_objects)
        summary.print_(mem_summary)
        # Look at NumPy arrays
        numpy_arrays = [obj for obj in all_objects if isinstance(obj, np.ndarray)]
        numpy_arrays.sort(key=lambda arr: arr.nbytes, reverse=True)

        print("\nTop 10 NumPy arrays by memory usage:")
        for i, arr in enumerate(numpy_arrays[:10], 1):
            memory_mb = arr.nbytes / (1024 * 1024)
            print(f"{i}. Shape: {arr.shape}, Dtype: {arr.dtype}, Memory: {memory_mb:.2f} MB")

        pandas_dfs = [obj for obj in all_objects if isinstance(obj, pd.DataFrame)]
        pandas_dfs.sort(key=lambda df: df.memory_usage(deep=True).sum(), reverse=True)

        print("\nTop 10 Pandas DataFrames by memory usage:")
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
    parser.add_argument("--test", type=str, help="profile or malloc, meaning use memory_profiler or tracemalloc", default="profile")
    args = parser.parse_args(sys.argv[1:])
    data_file = args.data_file

    causalforestdml_memory_test(
        test_type=args.test,
        file_name=data_file,
        estimator=args.estimator,
        n_est_y=args.n_est_y,
        n_est_t=args.n_est_t,
        n_est_2=args.n_est_2,
        n_jobs=args.n_jobs,
        use_memmap=args.memmap
    )
