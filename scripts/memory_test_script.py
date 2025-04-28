
import argparse
from econml.dml import CausalForestDML
from memory_profiler import memory_usage
import joblib
import os
from catboost import CatBoostRegressor
from xgboost import XGBRegressor, XGBClassifier
import sys
import time
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def causalforestdml_memory_test(
    file_name: str,
    n_est_y: int,
    n_est_t: int,
    n_est_2: int,
    n_jobs: int,
    use_memmap: bool
):


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
    mem_usage = memory_usage(
        (est.fit, [y,T], {"X":X}),
        interval=0.1,  # Sample every 0.1 seconds
        timeout=None,  # No timeout
        max_usage=True,  # Get maximum memory used
        retval=True,  # Return the fitted model too
        include_children=True  # Include joblib's child processes
    )
    end_time = time.time()

    # Extract results
    max_memory, fitted_model = mem_usage

    logger.info(f"Maximum memory usage: {max_memory} MiB")
    elapsed_time = end_time-start_time
    logger.info(f"Time to fit: {elapsed_time} seconds")


    est2 = CatBoostRegressor(n_estimators=n_est_2,allow_writing_files=False)


    logger.info(f"Calling CatBoostRegressor fit: n_estimators={n_est_2}")

    start_time2 = time.time()
    mem_usage2 = memory_usage(
        (est2.fit, [X],{"y":y, "silent": True}),
        interval=0.1,  # Sample every 0.1 seconds
        timeout=None,  # No timeout
        max_usage=True,  # Get maximum memory used
        retval=True,  # Return the fitted model too
        include_children=True  # Include joblib's child processes
    )
    end_time2 = time.time()

    # Extract results
    max_memory2, fitted_model2 = mem_usage2
    elapsed_time2 = end_time2-start_time2

    logger.info(f"Maximum memory usage: {max_memory2} MiB")
    logger.info(f"Time to fit: {elapsed_time2} seconds")

    file_name = os.path.join(root_dir,"mem_test_results.csv")
    if not os.path.isfile(file_name):
        with open(file_name,"w") as output:
            output.write("data,N_examples,N_nuisances,N_treatments,CFDML_max_memory,CFDML_fit_time,CB_max_memory,CB_fit_time\n")

    with open(file_name,"a") as output:
        output.write(f"{file_name},{X.shape[0]},{X.shape[1]},{T.shape[1]},{max_memory},{elapsed_time},{max_memory2},{elapsed_time2}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="", default="offerfit.joblib")
    parser.add_argument("--n_est_y", type=int, help="", default=None)
    parser.add_argument("--n_est_t", type=int, help="", default=None)
    parser.add_argument("--n_est_2", type=int, help="", default=None)
    parser.add_argument("--n_jobs", type=int, help="", default=-1)
    parser.add_argument("--memmap", type=bool, help="", default=False)
    args = parser.parse_args(sys.argv[1:])
    data_file = args.data_file

    causalforestdml_memory_test(
        file_name=data_file,
        n_est_y=args.n_est_y,
        n_est_t=args.n_est_t,
        n_est_2=args.n_est_2,
        n_jobs=args.n_jobs,
        use_memmap=args.memmap
    )
