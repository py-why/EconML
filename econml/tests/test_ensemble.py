# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for linear_model extensions."""

import numpy as np
import pytest
import unittest
import warnings
from econml.sklearn_extensions.ensemble import SubsampledHonestForest


class TestSubsampledHonestForest(unittest.TestCase):
    """Test SubsampledHonestForest."""

    def test_y1d(self):
        np.random.seed(123)
        n = 5000
        d = 5
        x_grid = np.linspace(-1, 1, 10)
        X_test = np.hstack([x_grid.reshape(-1, 1), np.random.normal(size=(10, d - 1))])
        for _ in range(3):
            for criterion in ['mse', 'mae']:
                X = np.random.normal(0, 1, size=(n, d))
                y = X[:, 0] + np.random.normal(0, .1, size=(n,))
                est = SubsampledHonestForest(n_estimators=100, max_depth=5, criterion=criterion,
                                             min_samples_leaf=10, verbose=0)
                est.fit(X, y)
                point = est.predict(X_test)
                lb, ub = est.predict_interval(X_test, alpha=0.01)
                np.testing.assert_allclose(point, X_test[:, 0], rtol=0, atol=.2)
                np.testing.assert_array_less(lb, X_test[:, 0] + .05)
                np.testing.assert_array_less(X_test[:, 0], ub + .05)

    def test_nonauto_subsample_fr(self):
        np.random.seed(123)
        n = 5000
        d = 5
        x_grid = np.linspace(-1, 1, 10)
        X_test = np.hstack([x_grid.reshape(-1, 1), np.random.normal(size=(10, d - 1))])
        X = np.random.normal(0, 1, size=(n, d))
        y = X[:, 0] + np.random.normal(0, .1, size=(n,))
        est = SubsampledHonestForest(n_estimators=100, subsample_fr=.8, max_depth=5, min_samples_leaf=10, verbose=0)
        est.fit(X, y)
        point = est.predict(X_test)
        lb, ub = est.predict_interval(X_test, alpha=0.01)
        np.testing.assert_allclose(point, X_test[:, 0], rtol=0, atol=.2)
        np.testing.assert_array_less(lb, X_test[:, 0] + .05)
        np.testing.assert_array_less(X_test[:, 0], ub + .05)

    def test_y2d(self):
        np.random.seed(123)
        n = 5000
        d = 5
        x_grid = np.linspace(-1, 1, 10)
        X_test = np.hstack([x_grid.reshape(-1, 1), np.random.normal(size=(10, d - 1))])
        for _ in range(3):
            for criterion in ['mse', 'mae']:
                X = np.random.normal(0, 1, size=(n, d))
                y = X[:, [0, 0]] + np.random.normal(0, .1, size=(n, 2))
                est = SubsampledHonestForest(n_estimators=100, max_depth=5, criterion=criterion,
                                             min_samples_leaf=10, verbose=0)
                est.fit(X, y)
                point = est.predict(X_test)
                lb, ub = est.predict_interval(X_test, alpha=0.01)
                np.testing.assert_allclose(point, X_test[:, [0, 0]], rtol=0, atol=.2)
                np.testing.assert_array_less(lb, X_test[:, [0, 0]] + .05)
                np.testing.assert_array_less(X_test[:, [0, 0]], ub + .05)

    def test_dishonest_y1d(self):
        np.random.seed(123)
        n = 5000
        d = 1
        x_grid = np.linspace(-1, 1, 10)
        X_test = np.hstack([x_grid.reshape(-1, 1), np.random.normal(size=(10, d - 1))])
        for _ in range(3):
            X = np.random.normal(0, 1, size=(n, d))
            y = 1. * (X[:, 0] > 0) + np.random.normal(0, .1, size=(n,))
            est = SubsampledHonestForest(n_estimators=100, honest=False, max_depth=3,
                                         min_samples_leaf=10, verbose=0)
            est.fit(X, y)
            point = est.predict(X_test)
            lb, ub = est.predict_interval(X_test, alpha=0.01)
            np.testing.assert_allclose(point, 1 * (X_test[:, 0] > 0), rtol=0, atol=.2)
            np.testing.assert_array_less(lb, 1 * (X_test[:, 0] > 0) + .05)
            np.testing.assert_array_less(1 * (X_test[:, 0] > 0), ub + .05)

    def test_dishonest_y2d(self):
        np.random.seed(123)
        n = 5000
        d = 1
        x_grid = np.linspace(-1, 1, 10)
        X_test = np.hstack([x_grid.reshape(-1, 1), np.random.normal(size=(10, d - 1))])
        for _ in range(3):
            X = np.random.normal(0, 1, size=(n, d))
            y = 1. * (X[:, [0, 0]] > 0) + np.random.normal(0, .1, size=(n, 2))
            est = SubsampledHonestForest(n_estimators=100, honest=False, max_depth=3,
                                         min_samples_leaf=10, verbose=0)
            est.fit(X, y)
            point = est.predict(X_test)
            lb, ub = est.predict_interval(X_test, alpha=0.01)
            np.testing.assert_allclose(point, 1. * (X_test[:, [0, 0]] > 0), rtol=0, atol=.2)
            np.testing.assert_array_less(lb, 1. * (X_test[:, [0, 0]] > 0) + .05)
            np.testing.assert_array_less(1. * (X_test[:, [0, 0]] > 0), ub + .05)

    def test_random_state(self):
        np.random.seed(123)
        n = 5000
        d = 5
        x_grid = np.linspace(-1, 1, 10)
        X_test = np.hstack([x_grid.reshape(-1, 1), np.random.normal(size=(10, d - 1))])
        X = np.random.normal(0, 1, size=(n, d))
        y = X[:, 0] + np.random.normal(0, .1, size=(n,))
        est = SubsampledHonestForest(n_estimators=100, max_depth=5, min_samples_leaf=10, verbose=0, random_state=12345)
        est.fit(X, y)
        point1 = est.predict(X_test)
        est = SubsampledHonestForest(n_estimators=100, max_depth=5,
                                     min_samples_leaf=10, verbose=0, random_state=12345)
        est.fit(X, y)
        point2 = est.predict(X_test)
        # Check that the point estimates are the same
        np.testing.assert_equal(point1, point2)
