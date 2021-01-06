# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.dml import LinearDML
from econml.inference import BootstrapInference


class TestATEInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        # DGP constants
        cls.n = 1000
        cls.d_w = 3
        cls.d_x = 3
        # Generate data
        cls.X = np.random.uniform(0, 1, size=(cls.n, cls.d_x))
        cls.W = np.random.normal(0, 1, size=(cls.n, cls.d_w))
        cls.T = np.random.binomial(1, .5, size=(cls.n, 2))
        cls.Y = np.random.normal(0, 1, size=(cls.n, 3))

    def test_ate_inference(self):
        """Tests the ate inference results."""
        Y, T, X, W = TestATEInference.Y, TestATEInference.T, TestATEInference.X, TestATEInference.W
        for inference in [BootstrapInference(n_bootstrap_samples=5), 'auto']:
            cate_est = LinearDML(model_t=LinearRegression(), model_y=LinearRegression(),
                                 featurizer=PolynomialFeatures(degree=2,
                                                               include_bias=False))
            cate_est.fit(Y, T, X=X, W=W, inference=inference)
            cate_est.ate(X)
            cate_est.ate_inference(X)
            cate_est.ate_interval(X, alpha=.01)
            lb, _ = cate_est.ate_inference(X).conf_int_mean()
            np.testing.assert_array_equal(lb.shape, Y.shape[1:])

            cate_est.marginal_ate(T, X)
            cate_est.marginal_ate_interval(T, X, alpha=.01)
            cate_est.marginal_ate_inference(T, X)
            lb, _ = cate_est.marginal_ate_inference(T, X).conf_int_mean()
            np.testing.assert_array_equal(lb.shape, Y.shape[1:] + T.shape[1:])

            cate_est.const_marginal_ate(X)
            cate_est.const_marginal_ate_interval(X, alpha=.01)
            cate_est.const_marginal_ate_inference(X)
            lb, _ = cate_est.const_marginal_ate_inference(X).conf_int_mean()
            np.testing.assert_array_equal(lb.shape, Y.shape[1:] + T.shape[1:])

            summary = cate_est.ate_inference(X).summary(value=10)
            for i in range(Y.shape[1]):
                assert summary.tables[0].data[1 + i][4] < 1e-5

            summary = cate_est.ate_inference(X).summary(value=np.mean(cate_est.effect(X), axis=0))
            for i in range(Y.shape[1]):
                np.testing.assert_almost_equal(summary.tables[0].data[1 + i][4], 1.0)

            summary = cate_est.marginal_ate_inference(T, X).summary(value=10)
            for i in range(Y.shape[1]):
                for j in range(T.shape[1]):
                    assert summary.tables[0].data[2 + i][1 + 3 * T.shape[1] + j] < 1e-5

            summary = cate_est.marginal_ate_inference(T, X).summary(
                value=np.mean(cate_est.marginal_effect(T, X), axis=0))
            for i in range(Y.shape[1]):
                for j in range(T.shape[1]):
                    np.testing.assert_almost_equal(summary.tables[0].data[2 + i][1 + 3 * T.shape[1] + j], 1.0)

            summary = cate_est.const_marginal_ate_inference(X).summary(value=10)
            for i in range(Y.shape[1]):
                for j in range(T.shape[1]):
                    assert summary.tables[0].data[2 + i][1 + 3 * T.shape[1] + j] < 1e-5

            summary = cate_est.const_marginal_ate_inference(X).summary(
                value=np.mean(cate_est.const_marginal_effect(X), axis=0))
            for i in range(Y.shape[1]):
                for j in range(T.shape[1]):
                    np.testing.assert_almost_equal(summary.tables[0].data[2 + i][1 + 3 * T.shape[1] + j], 1.0)
