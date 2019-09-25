# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for lasso extensions."""

import numpy as np
import pytest
import unittest
import warnings
from econml.utilities import WeightedLasso, WeightedLassoCV, WeightedMultiTaskLassoCV, WeightedKFold
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, MultiTaskLassoCV
from sklearn.model_selection import KFold


class TestWeightedLasso(unittest.TestCase):
    """Test WeightedLasso."""

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        np.random.seed(123)
        # DGP constants
        cls.n_samples = 1000
        cls.n_dim = 50
        cls.X = np.random.normal(size=(cls.n_samples, cls.n_dim))
        # DGP coefficients
        cls.coefs1 = np.zeros(cls.n_dim)
        nonzero_idx1 = np.random.choice(cls.n_dim, replace=False, size=5)
        cls.coefs2 = np.zeros(cls.n_dim)
        nonzero_idx2 = np.random.choice(cls.n_dim, replace=False, size=5)
        cls.coefs1[nonzero_idx1] = 1
        cls.coefs2[nonzero_idx2] = 1
        cls.intercept = 3
        cls.intercept1 = 2
        cls.intercept2 = 0
        cls.error_sd = 0.2
        # Generated outcomes
        cls.y1 = cls.intercept1 + np.dot(cls.X[:cls.n_samples // 2], cls.coefs1) + \
            np.random.normal(scale=cls.error_sd, size=cls.n_samples // 2)
        cls.y2 = cls.intercept2 + np.dot(cls.X[cls.n_samples // 2:], cls.coefs2) + \
            np.random.normal(scale=cls.error_sd, size=cls.n_samples // 2)
        cls.y = np.concatenate((cls.y1, cls.y2))
        cls.y_simple = np.dot(cls.X, cls.coefs1) + np.random.normal(scale=cls.error_sd, size=cls.n_samples)
        cls.y_2D = np.concatenate((TestWeightedLasso.y_simple.reshape(-1, 1),
                                   TestWeightedLasso.y.reshape(-1, 1)), axis=1)

    #################
    # WeightedLasso #
    #################
    def test_one_DGP(self):
        """Test WeightedLasso with one set of coefficients."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.ones(TestWeightedLasso.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate((TestWeightedLasso.X, TestWeightedLasso.X[TestWeightedLasso.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestWeightedLasso.y_simple, TestWeightedLasso.y_simple[TestWeightedLasso.n_samples // 2:]))
        # Range of alphas
        alpha_range = [0.001, 0.01, 0.1]
        # Compare with Lasso
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)
        # --> With intercept
        params = {'fit_intercept': True}
        # When DGP has no intercept
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)
        # When DGP has intercept
        self._compare_with_lasso(X_expanded, y_expanded + TestWeightedLasso.intercept, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple + TestWeightedLasso.intercept,
                                 sample_weight, alpha_range, params)
        # --> Coerce coefficients to be positive
        params = {'positive': True}
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)
        # --> Toggle max_iter & tol
        params = {'max_iter': 100, 'tol': 1e-3}
        self._compare_with_lasso(X_expanded, y_expanded, TestWeightedLasso.X,
                                 TestWeightedLasso.y_simple, sample_weight, alpha_range, params)

    def test_mixed_DGP(self):
        """Test WeightedLasso with two sets of coefficients."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.zeros(TestWeightedLasso.n_samples // 2)))
        # Data from one DGP has weight 0. Check that we recover correct coefficients
        self._compare_with_lasso(TestWeightedLasso.X[:TestWeightedLasso.n_samples // 2], TestWeightedLasso.y1,
                                 TestWeightedLasso.X, TestWeightedLasso.y, sample_weight)
        # Mixed DGP scenario.
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.ones(TestWeightedLasso.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate((TestWeightedLasso.X, TestWeightedLasso.X[TestWeightedLasso.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestWeightedLasso.y1, TestWeightedLasso.y2, TestWeightedLasso.y2))
        self._compare_with_lasso(X_expanded, y_expanded,
                                 TestWeightedLasso.X, TestWeightedLasso.y, sample_weight)

    def test_multiple_outputs(self):
        """Test multiple outputs."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.zeros(TestWeightedLasso.n_samples // 2)))
        # Define multioutput
        self._compare_with_lasso(TestWeightedLasso.X[:TestWeightedLasso.n_samples // 2],
                                 TestWeightedLasso.y_2D[:TestWeightedLasso.n_samples // 2],
                                 TestWeightedLasso.X, TestWeightedLasso.y_2D, sample_weight)

    ###################
    # WeightedLassoCV #
    ###################
    def test_no_weights_cv(self):
        """Test whether WeightedLassoCV with no weights returns LassoCV results."""
        # Define alphas to test
        alphas = np.logspace(-4, -1, num=10)
        # Compare with LassoCV
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso_cv(TestWeightedLasso.X, TestWeightedLasso.y_simple, TestWeightedLasso.X,
                                    TestWeightedLasso.y_simple, sample_weight=None, alphas=alphas, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        y_intercept = TestWeightedLasso.y_simple + TestWeightedLasso.intercept
        self._compare_with_lasso_cv(TestWeightedLasso.X, y_intercept, TestWeightedLasso.X,
                                    y_intercept, sample_weight=None, alphas=alphas, params=params)
        # --> Force parameters to be positive
        params = {'positive': True}
        self._compare_with_lasso_cv(TestWeightedLasso.X, y_intercept, TestWeightedLasso.X,
                                    y_intercept, sample_weight=None, alphas=alphas, params=params)

    def test_weighted_KFold(self):
        """Test WeightedKFold used in WeightedLassoCV."""
        # Choose a smaller n to speed-up process
        n = 100
        sample_weight = np.random.choice(10, size=n)
        n_splits = 3
        wkf = WeightedKFold(n_splits=n_splits)
        total_weight = np.sum(sample_weight)
        for _, test_index in wkf.split(TestWeightedLasso.X[:n],
                                       TestWeightedLasso.y_simple[:n], sample_weight=sample_weight):
            # Compare fold weights
            self.assertAlmostEqual(
                np.sum(sample_weight[test_index]) / total_weight, 1 / n_splits,
                delta=5e-2)

    def test_balanced_weights_cv(self):
        """Test whether WeightedLassoCV with balanced weights."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.ones(TestWeightedLasso.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate((TestWeightedLasso.X, TestWeightedLasso.X[TestWeightedLasso.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestWeightedLasso.y_simple, TestWeightedLasso.y_simple[TestWeightedLasso.n_samples // 2:]))
        # Define splitters
        # WeightedKFold splitter
        cv_splitter = WeightedKFold(n_splits=3)
        wlasso_cv = list(cv_splitter.split(TestWeightedLasso.X, TestWeightedLasso.y_simple,
                                           sample_weight=sample_weight))
        # Map weighted splitter to an extended splitter
        index_mapper = {}
        for i in range(TestWeightedLasso.n_samples):
            if i < TestWeightedLasso.n_samples // 2:
                index_mapper[i] = [i]
            else:
                index_mapper[i] = [i, i + TestWeightedLasso.n_samples // 2]
        lasso_cv = self._map_splitter(wlasso_cv, TestWeightedLasso.n_samples +
                                      TestWeightedLasso.n_samples // 2, index_mapper)
        # Define alphas to test
        alphas = np.logspace(-4, -1, num=10)
        # Compare with LassoCV
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso_cv(X_expanded, y_expanded, TestWeightedLasso.X, TestWeightedLasso.y_simple,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        y_intercept = TestWeightedLasso.y_simple + TestWeightedLasso.intercept
        self._compare_with_lasso_cv(X_expanded, y_expanded + TestWeightedLasso.intercept,
                                    TestWeightedLasso.X, y_intercept,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)
        # --> Force parameters to be positive
        params = {'positive': True}
        self._compare_with_lasso_cv(X_expanded, y_expanded + TestWeightedLasso.intercept,
                                    TestWeightedLasso.X, y_intercept,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)

    ############################
    # MultiTaskWeightedLassoCV #
    ############################
    def test_multiple_outputs_no_weights_cv(self):
        """Test MultiTaskWeightedLassoCV with no weights."""
        # Define alphas to test
        alphas = np.logspace(-4, -1, num=10)
        # Define splitter
        cv = WeightedKFold(n_splits=3)
        # Compare with MultiTaskLassoCV
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso_cv(TestWeightedLasso.X, TestWeightedLasso.y_2D,
                                    TestWeightedLasso.X, TestWeightedLasso.y_2D,
                                    sample_weight=None, alphas=alphas,
                                    lasso_cv=cv, wlasso_cv=cv, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        self._compare_with_lasso_cv(TestWeightedLasso.X, TestWeightedLasso.y_2D,
                                    TestWeightedLasso.X, TestWeightedLasso.y_2D,
                                    sample_weight=None, alphas=alphas,
                                    lasso_cv=cv, wlasso_cv=cv, params=params)

    # @unittest.skip("Failing")
    def test_multiple_outputs_balanced_weights_cv(self):
        """Test MultiTaskWeightedLassoCV with weights."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestWeightedLasso.n_samples // 2),
                                        np.ones(TestWeightedLasso.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate((TestWeightedLasso.X, TestWeightedLasso.X[TestWeightedLasso.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestWeightedLasso.y_2D, TestWeightedLasso.y_2D[TestWeightedLasso.n_samples // 2:]))
        # Define splitters
        # WeightedKFold splitter
        cv_splitter = WeightedKFold(n_splits=3)
        wlasso_cv = list(cv_splitter.split(TestWeightedLasso.X, TestWeightedLasso.y_2D,
                                           sample_weight=sample_weight))
        # Map weighted splitter to an extended splitter
        index_mapper = {}
        for i in range(TestWeightedLasso.n_samples):
            if i < TestWeightedLasso.n_samples // 2:
                index_mapper[i] = [i]
            else:
                index_mapper[i] = [i, i + TestWeightedLasso.n_samples // 2]
        lasso_cv = self._map_splitter(wlasso_cv, TestWeightedLasso.n_samples +
                                      TestWeightedLasso.n_samples // 2, index_mapper)
        # Define alphas to test
        alphas = np.logspace(-4, -1, num=10)
        # Compare with LassoCV
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso_cv(X_expanded, y_expanded, TestWeightedLasso.X, TestWeightedLasso.y_2D,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        self._compare_with_lasso_cv(X_expanded, y_expanded,
                                    TestWeightedLasso.X, TestWeightedLasso.y_2D,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)

    def _compare_with_lasso(self, lasso_X, lasso_y, wlasso_X, wlasso_y, sample_weight, alpha_range=[0.01], params={}):
        for alpha in alpha_range:
            lasso = Lasso(alpha=alpha)
            lasso.set_params(**params)
            lasso.fit(lasso_X, lasso_y)
            wlasso = WeightedLasso(alpha=alpha)
            wlasso.set_params(**params)
            wlasso.fit(wlasso_X, wlasso_y, sample_weight=sample_weight)
            # Check results are similar with tolerance 1e-6
            if np.ndim(lasso_y) > 1:
                for i in range(lasso_y.shape[1]):
                    self.assertTrue(np.allclose(lasso.coef_[i], wlasso.coef_[i]))
                    if lasso.get_params()["fit_intercept"]:
                        self.assertAlmostEqual(lasso.intercept_[i], wlasso.intercept_[i])
            else:
                self.assertTrue(np.allclose(lasso.coef_, wlasso.coef_))
                self.assertAlmostEqual(lasso.intercept_, wlasso.intercept_)

    def _compare_with_lasso_cv(self, lasso_X, lasso_y, wlasso_X, wlasso_y,
                               sample_weight, alphas, lasso_cv=3, wlasso_cv=3, params={}, tol=1e-8):
        # Check if multitask
        if np.ndim(lasso_y) > 1:
            lassoCV = MultiTaskLassoCV(alphas=alphas, cv=lasso_cv)
            wlassoCV = WeightedMultiTaskLassoCV(alphas=alphas, cv=wlasso_cv)
        else:
            lassoCV = LassoCV(alphas=alphas, cv=lasso_cv)
            wlassoCV = WeightedLassoCV(alphas=alphas, cv=wlasso_cv)
        lassoCV.set_params(**params)
        lassoCV.fit(lasso_X, lasso_y)
        wlassoCV.set_params(**params)
        wlassoCV.fit(wlasso_X, wlasso_y, sample_weight)
        # Check that same alpha is chosen
        self.assertEqual(lassoCV.alpha_, wlassoCV.alpha_)
        # Check that the coefficients are similar
        if np.ndim(lasso_y) > 1:
            for i in range(lasso_y.shape[1]):
                self.assertTrue(np.allclose(lassoCV.coef_[i], wlassoCV.coef_[i], atol=tol))
                if lassoCV.get_params()["fit_intercept"]:
                    self.assertAlmostEqual(lassoCV.intercept_[i], wlassoCV.intercept_[i])
        else:
            self.assertTrue(np.allclose(lassoCV.coef_, wlassoCV.coef_, atol=tol))
            self.assertAlmostEqual(lassoCV.intercept_, wlassoCV.intercept_)

    def _map_splitter(self, weighted_splits, n_expanded, index_mapper):
        unweighted_splits = []
        all_idx = np.arange(n_expanded)
        for _, test_idx in weighted_splits:
            unweighted_test_idx = []
            for idx in test_idx:
                unweighted_test_idx += index_mapper[idx]
            unweighted_test_idx = np.asarray(unweighted_test_idx)
            unweighted_train_idx = np.setdiff1d(all_idx, unweighted_test_idx, assume_unique=True)
            unweighted_splits.append((unweighted_train_idx, unweighted_test_idx))
        return unweighted_splits
