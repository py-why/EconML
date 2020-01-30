import os
import numpy as np
import econml_dml_te
from mcpy import metrics
from mcpy import plotting
from mcpy import utils
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV,LinearRegression,MultiTaskElasticNet,MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures

CONFIG = {
        "dgps": {
            "dgp1": econml_dml_te.gen_data
        },
        "dgp_instance_fns": {
            'dgp1': econml_dml_te.instance_params
        },
        "dgp_opts": {
            'dgp1': {
                'n_samples': [100, 2000],
                'n_features': 1,
                'n_controls': 30,
                'support_size': 5
            },
        },
        "methods": {
            "LinearDMLCate": econml_dml_te.linear_dml_fit,
            "SparseLinearDMLCate": econml_dml_te.sparse_linear_dml_poly_fit,
            # "DMLCate": econml_dml_te.dml_poly_fit,
            # "ForestDMLCate": econml_dml_te.forest_dml_fit
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
            # 'DMLCate': {
            #         'model_y': RandomForestRegressor(),
            #         'model_t': RandomForestRegressor(),
            #         'model_final': Lasso(alpha=0.1, fit_intercept=False),
            #         'featurizer': PolynomialFeatures(degree=10),
            #         'inference': 'bootstrap'
            #     },
            #     'ForestDMLCate': {
            #         'model_y': RandomForestRegressor(),
            #         'model_t': RandomForestRegressor(),
            #         'discrete_treatment': False,
            #         'n_estimators': 1000,
            #         'subsample_fr': 0.8,
            #         'min_samples_leaf': 10,
            #         'min_impurity_decrease': 0.001,
            #         'verbose': 0,
            #         'min_weight_fraction_leaf': 0.01,
            #         'inference': 'bootstrap'
            #     }
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
        "per_plots": ['coverage_band'],
        "sweep_plots": {
            # 'plot4': plotting.plot_sweep
            # "plot1": {'varying_params': ['$k_g$'], 'select_vals': {'$\\sigma_\\epsilon$': [1.0], '$\\sigma_\\eta$': [1.0]}},
        #     "plot2": {'varying_params': [n('$k_g$', '$\\sigma_\\eta$')], 'select_vals': {'$\\sigma_\\epsilon$':[1.0]}, 'methods': ["Direct"], 'metric_transforms': {'% decrease': metrics.transform_ratio}},
        #     "plot3": {'varying_params': [('$k_g$', '$\\sigma_\\epsilon$')], 'select_vals': {'$\\sigma_\\eta$':[1.0]}, 'methods': ["Direct"], 'metric_transforms': {'% decrease': metrics.transform_ratio}}
        },
        "mc_opts": {
            'n_experiments': 5, # number of monte carlo experiments
            "seed": 123
        },
        "proposed_method": "CrossOrtho",
        "target_dir": "sweep_econml_test",
        "reload_results": False
    }
