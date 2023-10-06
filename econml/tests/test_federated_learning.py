# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from econml.dml import LinearDML
from econml.inference import StatsModelsInference
from econml.federated_learning import FederatedEstimator


class FunctionRegressor:
    """A simple model that ignores the data it is fitted on, always just using the specified function to predict"""
    def __init__(self, func):
        self.func = func

    def fit(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        return self.func(X)


class TestFederatedLearning(unittest.TestCase):
    """
    A set of unit tests for the FederatedLearner class.

    These tests check various scenarios of splitting, aggregation, and comparison
    between FederatedLearner and individual LinearDML estimators.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    def test_splitting_works(self):

        num_samples = 1000
        n_w = 5
        for d_x in [None, 5]:
            for d_t in [(), (2,)]:
                for d_y in [(), (2,)]:
                    for weights in [None, np.random.uniform(size=num_samples)]:
                        for sample_var in [None, np.random.uniform(size=(num_samples,) + d_y)]:

                            n_x = d_x if d_x is not None else 0
                            X = np.random.normal(size=(num_samples, n_x)) if d_x is not None else None
                            W = np.random.normal(size=(num_samples, n_w))
                            T = np.random.normal(size=(num_samples,) + d_t)
                            Y = np.random.normal(size=(num_samples,) + d_y)
                            freq_weights = np.random.choice(
                                [2, 3, 4], size=num_samples) if sample_var is not None else None

                            (Y1, T1, X1, W1,
                             weights1, freq_weights1, sample_var1) = [A[:num_samples // 2] if A is not None else None
                                                                      for A in [Y, T, X, W,
                                                                                weights, freq_weights, sample_var]]
                            (Y2, T2, X2, W2,
                             weights2, freq_weights2, sample_var2) = [A[num_samples // 2:] if A is not None else None
                                                                      for A in [Y, T, X, W,
                                                                                weights, freq_weights, sample_var]]

                            # fixed functions as first stage models
                            # they can be anything as long as fitting doesn't modify the predictions
                            # that way, it doesn't matter if they are trained on different subsets of the data
                            a = np.random.normal(size=(n_x + n_w,) + d_t)
                            b = np.random.normal(size=(n_x + n_w,) + d_y)

                            t_model = FunctionRegressor(lambda XW: XW @ a)
                            y_model = FunctionRegressor(lambda XW: XW @ b)

                            for cov_type in ['HC0', 'HC1', 'nonrobust']:
                                with self.subTest(d_x=d_x, d_t=d_t, d_y=d_y,
                                                  weights=(weights is not None), fw=(freq_weights is not None),
                                                  cov_type=cov_type):
                                    est_all = LinearDML(model_y=y_model, model_t=t_model, linear_first_stages=False)
                                    est_h1 = LinearDML(model_y=y_model, model_t=t_model, linear_first_stages=False)
                                    est_h2 = LinearDML(model_y=y_model, model_t=t_model, linear_first_stages=False)

                                    est_all.fit(Y, T, X=X, W=W,
                                                sample_weight=weights, freq_weight=freq_weights, sample_var=sample_var,
                                                inference=StatsModelsInference(cov_type=cov_type))
                                    est_h1.fit(Y1, T1, X=X1, W=W1,
                                               sample_weight=weights1,
                                               freq_weight=freq_weights1,
                                               sample_var=sample_var1,
                                               inference=StatsModelsInference(cov_type=cov_type))
                                    est_h2.fit(Y2, T2, X=X2, W=W2,
                                               sample_weight=weights2,
                                               freq_weight=freq_weights2,
                                               sample_var=sample_var2,
                                               inference=StatsModelsInference(cov_type=cov_type))

                                    est_fed1 = FederatedEstimator([est_all])

                                    est_fed2 = FederatedEstimator([est_h1, est_h2])

                                    np.testing.assert_allclose(est_fed1.model_final_._param,
                                                               est_fed2.model_final_._param)
                                    np.testing.assert_allclose(est_fed1.model_final_._param,
                                                               est_all.model_final_._param)
                                    np.testing.assert_allclose(est_fed1.model_final_._param_var,
                                                               est_fed2.model_final_._param_var)
                                    np.testing.assert_allclose(est_fed1.model_final_._param_var,
                                                               est_all.model_final_._param_var)
