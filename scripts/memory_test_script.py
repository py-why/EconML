import json
from datetime import datetime

import sklearn.metrics
import argparse
from econml.dml import SparseLinearDML, CausalForestDML
from econml.validate import DRTester
import collinearity
from itertools import product
import joblib
import numpy as np
import os
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor, XGBClassifier
import sys
import logging
from sklearn.model_selection import KFold

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
        model_y=XGBClassifier(n_estimators=n_est_y),
        discrete_outcome=True,
        n_jobs=n_jobs,
        n_estimators=n_est_2,
        use_memmap=use_memmap
    )
    logger.info(f"Calling fit: njobs={n_jobs}, MemMap={use_memmap}")
    est.fit(y,T,X=X)
    print(est.summary())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="", default="offerfit.joblib")
    parser.add_argument("--n_est_y", type=int, help="", default=100)
    parser.add_argument("--n_est_t", type=int, help="", default=200)
    parser.add_argument("--n_est_2", type=int, help="", default=500)
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
