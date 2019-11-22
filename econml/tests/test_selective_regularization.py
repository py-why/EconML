# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np
import econml.dgp
from econml.selective_regularization import SelectiveRegularization
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import unittest


class TestSelectiveRegularization(unittest.TestCase):

    # selective ridge has a simple implementation that we can test against
    # see https://stats.stackexchange.com/questions/69205/how-to-derive-the-ridge-regression-solution/164546#164546
    def test_against_ridge_ground_truth(self):
        for _ in range(10):
            n = 100 + np.random.choice(500)
            d = 5 + np.random.choice(20)
            n_inds = np.random.choice(np.arange(2, d - 2))
            inds = np.random.choice(d, n_inds, replace=False)
            alpha = np.random.uniform(0.5, 1.5)
            X = np.random.normal(size=(n, d))
            y = np.random.normal(size=(n,))
            coef = SelectiveRegularization(unpenalized_inds=np.delete(np.arange(d), inds),
                                           penalized_model=Ridge(alpha=alpha, fit_intercept=False),
                                           fit_intercept=False).fit(X, y).coef_
            X_aug = np.zeros((n_inds, d))
            X_aug[np.arange(n_inds), inds] = np.sqrt(alpha)
            y_aug = np.zeros((n_inds,))
            coefs = LinearRegression(fit_intercept=False).fit(np.vstack((X, X_aug)),
                                                              np.concatenate((y, y_aug))).coef_
            np.testing.assert_allclose(coef, coefs)

    # it should be the case that when we set fit_intercept to true,
    # it doesn't matter whether the penalized model also fits an intercept or not
    def test_intercept(self):
        for _ in range(10):
            n = 100 + np.random.choice(500)
            d = 5 + np.random.choice(20)
            n_inds = np.random.choice(np.arange(2, d - 2))
            inds = np.random.choice(d, n_inds, replace=False)
            unpenalized_inds = np.delete(np.arange(d), inds)
            alpha = np.random.uniform(0.5, 1.5)
            X = np.random.normal(size=(n, d))
            y = np.random.normal(size=(n,))
            models = [SelectiveRegularization(unpenalized_inds=unpenalized_inds,
                                              penalized_model=Lasso(alpha=alpha,
                                                                    fit_intercept=inner_intercept),
                                              fit_intercept=True)
                      for inner_intercept in [False, True]]

            for model in models:
                model.fit(X, y)

            np.testing.assert_allclose(models[0].coef_, models[1].coef_)
            np.testing.assert_allclose(models[0].intercept_, models[1].intercept_)

    def test_vectors_and_arrays(self):
        X = np.random.normal(size=(10, 3))
        Y = np.random.normal(size=(10, 2))
        model = SelectiveRegularization(unpenalized_inds=[0],
                                        penalized_model=Ridge(),
                                        fit_intercept=True)
        self.assertEqual(model.fit(X, Y).coef_.shape, (2, 3))
        self.assertEqual(model.fit(X, Y[:, 0]).coef_.shape, (3,))

    def test_can_use_sample_weights(self):
        n = 100 + np.random.choice(500)
        d = 5 + np.random.choice(20)
        n_inds = np.random.choice(np.arange(2, d - 2))
        inds = np.random.choice(d, n_inds, replace=False)
        alpha = np.random.uniform(0.5, 1.5)
        X = np.random.normal(size=(n, d))
        y = np.random.normal(size=(n,))
        sample_weight = np.random.choice([1, 2], n)
        # create an extra copy of rows with weight 2
        X_aug = X[sample_weight == 2, :]
        y_aug = y[sample_weight == 2]
        model = SelectiveRegularization(unpenalized_inds=inds,
                                        penalized_model=Ridge(),
                                        fit_intercept=True)
        coef = model.fit(X, y, sample_weight=sample_weight).coef_
        coef2 = model.fit(np.vstack((X, X_aug)),
                          np.concatenate((y, y_aug))).coef_
        np.testing.assert_allclose(coef, coef2)
