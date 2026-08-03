# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import numpy as np
import dml_te_functions
from mcpy import metrics
from mcpy import plotting
from mcpy import utils
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV,LinearRegression,MultiTaskElasticNet,MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

CONFIG = {
        "type": 'single_parameter',
        "dgps": {
            "dgp1": dml_te_functions.gen_data
        },
        "dgp_instance_fns": {
            'dgp1': dml_te_functions.instance_params
        },
        "dgp_opts": {
            'dgp1': {
                'n_samples': 2000,
                'n_features': 1,
                'n_controls': 30,
                'support_size': 5
            },
        },
        "methods": {
            "LinearDMLCate": dml_te_functions.linear_dml_fit,
            "SparseLinearDMLCate": dml_te_functions.sparse_linear_dml_poly_fit,
            # "DMLCate": dml_te_functions.dml_poly_fit,
            # "ForestDMLCate": dml_te_functions.forest_dml_fit
        },
        "method_opts": {
                'LinearDMLCate': {
                    'model_y': RandomForestRegressor(),
                    'model_t': RandomForestRegressor(),
                    'inference': 'statsmodels'
                },
                'SparseLinearDMLCate': {
                    'model_y': RandomForestRegressor(),
                    'model_t': RandomForestRegressor(),
                    'featurizer': PolynomialFeatures(degree=3),
                    'inference': 'debiasedlasso'
                },
            'DMLCate': {
                    'model_y': RandomForestRegressor(),
                    'model_t': RandomForestRegressor(),
                    'model_final': Lasso(alpha=0.1, fit_intercept=False),
                    'featurizer': PolynomialFeatures(degree=10),
                    'inference': 'bootstrap'
                },
                'ForestDMLCate': {
                    'model_y': RandomForestRegressor(),
                    'model_t': RandomForestRegressor(),
                    'discrete_treatment': False,
                    'n_estimators': 1000,
                    'subsample_fr': 0.8,
                    'min_samples_leaf': 10,
                    'min_impurity_decrease': 0.001,
                    'verbose': 0,
                    'min_weight_fraction_leaf': 0.01,
                    'inference': 'bootstrap'
                }
        },
        "metrics": {
            'rmse': metrics.rmse,
            'conf_length': metrics.conf_length,
            'coverage': metrics.coverage,
            'std': metrics.std,
            'coverage_band': metrics.coverage_band
        },
        "plots": {
            'plot1': plotting.plot_metrics,
            'plot2': plotting.plot_visualization,
            'plot3': plotting.plot_violin
        },
        # different metrics are plotted differnetly
        # single summary metrics are a single value per dgp and method
        "single_summary_metrics": ['coverage_band'],
        "sweep_plots": {
        },
        "mc_opts": {
            'n_experiments': 5, # number of monte carlo experiments
            "seed": 123
        },
        "proposed_method": "CrossOrtho",
        "target_dir": "dml_te_test",
        "reload_results": False
    }
