# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from econml.dml import LinearDML
from econml.inference import StatsModelsInference
from econml.sklearn_extensions.federated_learning import FederatedLearner


class FunctionRegressor:
    def __init__(self, func):
        self.func = func

    def fit(self, X, y):
        pass

    def predict(self, X):
        return self.func(X)


class TestFederatedLearning(unittest.TestCase):

    def test_splitting_works(self):

        num_samples = 1000
        n_w = 5
        for d_x in [None, 5]:
            for d_t in [(), (2,)]:
                for d_y in [(), (2,)]:

                    n_x = d_x if d_x is not None else 0
                    X = np.random.normal(size=(num_samples, n_x)) if d_x is not None else None
                    W = np.random.normal(size=(num_samples, n_w))
                    T = np.random.normal(size=(num_samples,) + d_t)
                    Y = np.random.normal(size=(num_samples,) + d_y)

                    X1 = X[:num_samples // 2] if d_x is not None else None
                    X2 = X[num_samples // 2:] if d_x is not None else None

                    [Y1, T1, W1] = [Y[:num_samples // 2], T[:num_samples // 2], W[:num_samples // 2]]
                    [Y2, T2, W2] = [Y[num_samples // 2:], T[num_samples // 2:], W[num_samples // 2:]]

                    # fixed functions as first stage models - can be anything as long as fitting doesn't modify them
                    # that way, it doesn't matter if they are trained on different subsets of the data
                    a = np.random.normal(size=(n_x + n_w,) + d_t)
                    b = np.random.normal(size=(n_x + n_w,) + d_y)

                    t_model = FunctionRegressor(lambda XW: XW @ a)
                    y_model = FunctionRegressor(lambda XW: XW @ b)

                    for cov_type in ['HC0', 'HC1', 'nonrobust']:
                        with self.subTest(d_x=d_x, d_t=d_t, d_y=d_y, cov_type=cov_type):
                            est_all = LinearDML(model_y=y_model, model_t=t_model, linear_first_stages=False)
                            est_h1 = LinearDML(model_y=y_model, model_t=t_model, linear_first_stages=False)
                            est_h2 = LinearDML(model_y=y_model, model_t=t_model, linear_first_stages=False)

                            est_all.fit(Y, T, X=X, W=W, inference=StatsModelsInference(cov_type=cov_type))
                            est_h1.fit(Y1, T1, X=X1, W=W1, inference=StatsModelsInference(cov_type=cov_type))
                            est_h2.fit(Y2, T2, X=X2, W=W2, inference=StatsModelsInference(cov_type=cov_type))

                            est_fed1 = FederatedLearner()
                            est_fed1.initialize_from_existing([est_all])
                            est_fed1.solve_linear_equation()

                            est_fed2 = FederatedLearner()
                            est_fed2.initialize_from_existing([est_h1, est_h2])
                            est_fed2.solve_linear_equation()

                            np.testing.assert_allclose(est_fed1.theta_hat, est_fed2.theta_hat)
                            np.testing.assert_allclose(est_fed1.variance_matrix, est_fed2.variance_matrix)
                            np.testing.assert_allclose(est_fed1.theta_hat, est_all.model_final_._param)
                            np.testing.assert_allclose(est_fed1.variance_matrix, est_all.model_final_._param_var)
