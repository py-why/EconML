# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from sklearn.linear_model import Lasso, LassoCV, LinearRegression
import econml.dml
import econml.dgp
import unittest


########################################
# Core DML Tests
########################################


class TestDMLMethods(unittest.TestCase):

    def test_dml_effect(self):
        """Testing dml.effect."""
        np.random.seed(123)
        # How many samples
        n_samples = 200
        # How many control features
        n_cov = 5
        # How many treatment variables
        n_treatments = 10
        for exp in range(100):
            # Coefficients of how controls affect treatments
            Alpha = 20 * np.random.rand(n_cov, n_treatments) - 10
            # Coefficients of how controls affect outcome
            beta = 20 * np.random.rand(n_cov) - 10
            # Treatment effects that we want to estimate
            effect = 20 * np.random.rand(n_treatments) - 10
            y, T, X, epsilon = dgp.dgp_perfect_data_multiple_treatments(
                n_samples, n_cov, n_treatments, Alpha, beta, effect)

            # Run dml estimation
            reg = dml.LinearDML(np.arange(X.shape[1]), [], np.arange(X.shape[1], X.shape[1] + T.shape[1]))
            reg.fit(np.concatenate((X, T), axis=1), y)

            T0 = np.zeros((1, T.shape[1]))
            T1 = np.zeros((1, T.shape[1]))
            dml_coef = np.zeros(T.shape[1])
            for t in range(T.shape[1]):
                T1[:, t] = 1
                dml_coef[t] = reg.effect([], T0, T1)
                T1[:, t] = 0
            self.assertTrue(np.max(np.abs(dml_coef - effect)) < 0.0000000001, "core.double_ml() wrong")

    def test_dml_predict(self):
        """Testing dml.predict."""
        np.random.seed(123)
        # How many samples
        n_samples = 200
        # How many control features
        n_cov = 5
        # How many treatment variables
        n_treatments = 10
        for exp in range(100):
            # Coefficients of how controls affect treatments
            Alpha = 20 * np.random.rand(n_cov, n_treatments) - 10
            # Coefficients of how controls affect outcome
            beta = 20 * np.random.rand(n_cov) - 10
            # Treatment effects that we want to estimate
            effect = 20 * np.random.rand(n_treatments) - 10
            y, T, X, epsilon = dgp.dgp_perfect_data_multiple_treatments(
                n_samples, n_cov, n_treatments, Alpha, beta, effect)

            # Run dml estimation
            reg = dml.LinearDML(np.arange(X.shape[1]), [], np.arange(X.shape[1], X.shape[1] + T.shape[1]))
            reg.fit(np.concatenate((X, T), axis=1), y)

            y, T, X = dgp.dgp_perfect_counterfactual_data_multiple_treatments(
                n_samples, n_cov, beta, effect, np.ones(n_treatments))

            r2score = reg.score(np.concatenate((X, T), axis=1), y)
            self.assertTrue(r2score > 0.99, "core.double_ml() wrong")


if __name__ == '__main__':
    unittest.main(verbosity=2)
