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
    file_name: str
):


    root_dir = os.environ.get("LOCAL_ANALYSIS_DIR", ".")
    logger.info(f"Using root dir {root_dir} loading data file {file_name}")
    data_file = os.path.join(os.path.abspath(root_dir), file_name)
    data = joblib.load(data_file)

    # This is specific to Offerfit - names of the data are derived from contextual bandits
    X = data['X_no_act']
    T = data['a_processed']
    y = data['real_reward']

    logger.info(f"X has a shape of: {X.shape}")
    logger.info(f"T has a shape of: {T.shape}")
    logger.info(f"y has a shape of: {y.shape}")


    est = CausalForestDML(
        model_t=XGBRegressor(n_estimators=10),
        model_y=XGBClassifier(n_estimators=10),
        discrete_outcome=True,
        n_jobs=2,
        n_estimators=20
    )
    est.fit(y,T,X=X)
    print(est.summary())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, help="", default="offerfit.joblib")
    args = parser.parse_args(sys.argv[1:])
    data_file = args.data_file

    causalforestdml_memory_test(
        file_name=data_file
    )
