# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for linear_model extensions."""

import numpy as np
import pytest
import unittest
import warnings
from econml.sklearn_extensions.linear_model import (WeightedLasso, WeightedLassoCV, WeightedMultiTaskLassoCV,
                                                    WeightedLassoCVWrapper, DebiasedLasso, MultiOutputDebiasedLasso,
                                                    SelectiveRegularization)
from econml.sklearn_extensions.model_selection import WeightedKFold
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, MultiTaskLassoCV, Ridge
from sklearn.model_selection import KFold
from sklearn.base import clone


class TestLassoExtensions(unittest.TestCase):
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
        cls.y_2D = np.concatenate((TestLassoExtensions.y_simple.reshape(-1, 1),
                                   TestLassoExtensions.y.reshape(-1, 1)), axis=1)
        cls.y2_full = np.dot(cls.X, cls.coefs2) + np.random.normal(scale=cls.error_sd, size=cls.n_samples)
        cls.y_2D_consistent = np.concatenate((TestLassoExtensions.y_simple.reshape(-1, 1),
                                              TestLassoExtensions.y2_full.reshape(-1, 1)), axis=1)

    #################
    # WeightedLasso #
    #################
    def test_one_DGP(self):
        """Test WeightedLasso with one set of coefficients.

        To test the correctness of the weighted lasso, we compare the weight lasso with integer weights
        with the standard lasso where the data entries have been replicated a number of times given by the
        integer weights.
        """
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.ones(TestLassoExtensions.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate(
            (TestLassoExtensions.X, TestLassoExtensions.X[TestLassoExtensions.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestLassoExtensions.y_simple, TestLassoExtensions.y_simple[TestLassoExtensions.n_samples // 2:]))
        # Range of alphas
        alpha_range = [0.001, 0.01, 0.1]
        # Compare with Lasso
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso(X_expanded, y_expanded, TestLassoExtensions.X,
                                 TestLassoExtensions.y_simple, sample_weight, alpha_range, params)
        # --> With intercept
        params = {'fit_intercept': True}
        # When DGP has no intercept
        self._compare_with_lasso(X_expanded, y_expanded, TestLassoExtensions.X,
                                 TestLassoExtensions.y_simple, sample_weight, alpha_range, params)
        # When DGP has intercept
        self._compare_with_lasso(X_expanded, y_expanded + TestLassoExtensions.intercept, TestLassoExtensions.X,
                                 TestLassoExtensions.y_simple + TestLassoExtensions.intercept,
                                 sample_weight, alpha_range, params)
        # --> Coerce coefficients to be positive
        params = {'positive': True}
        self._compare_with_lasso(X_expanded, y_expanded, TestLassoExtensions.X,
                                 TestLassoExtensions.y_simple, sample_weight, alpha_range, params)
        # --> Toggle max_iter & tol
        params = {'max_iter': 100, 'tol': 1e-3}
        self._compare_with_lasso(X_expanded, y_expanded, TestLassoExtensions.X,
                                 TestLassoExtensions.y_simple, sample_weight, alpha_range, params)

    def test_mixed_DGP(self):
        """Test WeightedLasso with two sets of coefficients."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.zeros(TestLassoExtensions.n_samples // 2)))
        # Data from one DGP has weight 0. Check that we recover correct coefficients
        self._compare_with_lasso(TestLassoExtensions.X[:TestLassoExtensions.n_samples // 2], TestLassoExtensions.y1,
                                 TestLassoExtensions.X, TestLassoExtensions.y, sample_weight)
        # Mixed DGP scenario.
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.ones(TestLassoExtensions.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate(
            (TestLassoExtensions.X, TestLassoExtensions.X[TestLassoExtensions.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestLassoExtensions.y1, TestLassoExtensions.y2, TestLassoExtensions.y2))
        self._compare_with_lasso(X_expanded, y_expanded,
                                 TestLassoExtensions.X, TestLassoExtensions.y, sample_weight)

    def test_multiple_outputs(self):
        """Test multiple outputs."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.zeros(TestLassoExtensions.n_samples // 2)))
        # Define multioutput
        self._compare_with_lasso(TestLassoExtensions.X[:TestLassoExtensions.n_samples // 2],
                                 TestLassoExtensions.y_2D[:TestLassoExtensions.n_samples // 2],
                                 TestLassoExtensions.X, TestLassoExtensions.y_2D, sample_weight)

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
        self._compare_with_lasso_cv(TestLassoExtensions.X, TestLassoExtensions.y_simple, TestLassoExtensions.X,
                                    TestLassoExtensions.y_simple, sample_weight=None, alphas=alphas, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        y_intercept = TestLassoExtensions.y_simple + TestLassoExtensions.intercept
        self._compare_with_lasso_cv(TestLassoExtensions.X, y_intercept, TestLassoExtensions.X,
                                    y_intercept, sample_weight=None, alphas=alphas, params=params)
        # --> Force parameters to be positive
        params = {'positive': True}
        self._compare_with_lasso_cv(TestLassoExtensions.X, y_intercept, TestLassoExtensions.X,
                                    y_intercept, sample_weight=None, alphas=alphas, params=params)

    def test_weighted_KFold(self):
        """Test WeightedKFold used in WeightedLassoCV."""
        # Choose a smaller n to speed-up process
        n = 100
        sample_weight = np.random.choice(10, size=n)
        n_splits = 3
        wkf = WeightedKFold(n_splits=n_splits)
        total_weight = np.sum(sample_weight)
        for _, test_index in wkf.split(TestLassoExtensions.X[:n],
                                       TestLassoExtensions.y_simple[:n], sample_weight=sample_weight):
            # Compare fold weights
            self.assertAlmostEqual(
                np.sum(sample_weight[test_index]) / total_weight, 1 / n_splits,
                delta=5e-2)

    def test_balanced_weights_cv(self):
        """Test whether WeightedLassoCV with balanced weights."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.ones(TestLassoExtensions.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate(
            (TestLassoExtensions.X, TestLassoExtensions.X[TestLassoExtensions.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestLassoExtensions.y_simple, TestLassoExtensions.y_simple[TestLassoExtensions.n_samples // 2:]))
        # Define splitters
        # WeightedKFold splitter
        cv_splitter = WeightedKFold(n_splits=3)
        wlasso_cv = list(cv_splitter.split(TestLassoExtensions.X, TestLassoExtensions.y_simple,
                                           sample_weight=sample_weight))
        # Map weighted splitter to an extended splitter
        index_mapper = {}
        for i in range(TestLassoExtensions.n_samples):
            if i < TestLassoExtensions.n_samples // 2:
                index_mapper[i] = [i]
            else:
                index_mapper[i] = [i, i + TestLassoExtensions.n_samples // 2]
        lasso_cv = self._map_splitter(wlasso_cv, TestLassoExtensions.n_samples +
                                      TestLassoExtensions.n_samples // 2, index_mapper)
        # Define alphas to test
        alphas = np.logspace(-4, -1, num=10)
        # Compare with LassoCV
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso_cv(X_expanded, y_expanded, TestLassoExtensions.X, TestLassoExtensions.y_simple,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        y_intercept = TestLassoExtensions.y_simple + TestLassoExtensions.intercept
        self._compare_with_lasso_cv(X_expanded, y_expanded + TestLassoExtensions.intercept,
                                    TestLassoExtensions.X, y_intercept,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)
        # --> Force parameters to be positive
        params = {'positive': True}
        self._compare_with_lasso_cv(X_expanded, y_expanded + TestLassoExtensions.intercept,
                                    TestLassoExtensions.X, y_intercept,
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
        self._compare_with_lasso_cv(TestLassoExtensions.X, TestLassoExtensions.y_2D,
                                    TestLassoExtensions.X, TestLassoExtensions.y_2D,
                                    sample_weight=None, alphas=alphas,
                                    lasso_cv=cv, wlasso_cv=cv, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        self._compare_with_lasso_cv(TestLassoExtensions.X, TestLassoExtensions.y_2D,
                                    TestLassoExtensions.X, TestLassoExtensions.y_2D,
                                    sample_weight=None, alphas=alphas,
                                    lasso_cv=cv, wlasso_cv=cv, params=params)

    def test_multiple_outputs_balanced_weights_cv(self):
        """Test MultiTaskWeightedLassoCV with weights."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.ones(TestLassoExtensions.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate(
            (TestLassoExtensions.X, TestLassoExtensions.X[TestLassoExtensions.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestLassoExtensions.y_2D, TestLassoExtensions.y_2D[TestLassoExtensions.n_samples // 2:]))
        # Define splitters
        # WeightedKFold splitter
        cv_splitter = WeightedKFold(n_splits=3)
        wlasso_cv = list(cv_splitter.split(TestLassoExtensions.X, TestLassoExtensions.y_2D,
                                           sample_weight=sample_weight))
        # Map weighted splitter to an extended splitter
        index_mapper = {}
        for i in range(TestLassoExtensions.n_samples):
            if i < TestLassoExtensions.n_samples // 2:
                index_mapper[i] = [i]
            else:
                index_mapper[i] = [i, i + TestLassoExtensions.n_samples // 2]
        lasso_cv = self._map_splitter(wlasso_cv, TestLassoExtensions.n_samples +
                                      TestLassoExtensions.n_samples // 2, index_mapper)
        # Define alphas to test
        alphas = np.logspace(-4, -1, num=10)
        # Compare with LassoCV
        # --> No intercept
        params = {'fit_intercept': False}
        self._compare_with_lasso_cv(X_expanded, y_expanded, TestLassoExtensions.X, TestLassoExtensions.y_2D,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)
        # --> With intercept
        params = {'fit_intercept': True}
        self._compare_with_lasso_cv(X_expanded, y_expanded,
                                    TestLassoExtensions.X, TestLassoExtensions.y_2D,
                                    sample_weight=sample_weight, alphas=alphas,
                                    lasso_cv=lasso_cv, wlasso_cv=wlasso_cv, params=params)

    ##########################
    # WeightedLassoCVWrapper #
    ##########################
    def test_wrapper_attributes(self):
        """Test that attributes are properly maintained across calls to fit that switch between 1- and 2-D"""
        wrapper = WeightedLassoCVWrapper(alphas=[5, 10], max_iter=100)
        wrapper.tol = 0.01  # set an attribute manually as well

        assert wrapper.alphas == [5, 10]
        assert wrapper.max_iter == 100
        assert wrapper.tol == 0.01

        # perform 1D fit
        wrapper.fit(np.random.normal(size=(100, 3)), np.random.normal(size=100))

        assert wrapper.alphas == [5, 10]
        assert wrapper.max_iter == 100
        assert wrapper.tol == 0.01

        # perform 2D fit
        wrapper.fit(np.random.normal(size=(100, 3)), np.random.normal(size=(100, 2)))

        assert wrapper.alphas == [5, 10]
        assert wrapper.max_iter == 100
        assert wrapper.tol == 0.01

    #################
    # DebiasedLasso #
    #################

    def test_debiased_lasso_one_DGP(self):
        """Test DebiasedLasso with one set of coefficients."""
        # Test DebiasedLasso without weights
        # --> Check debiased coeffcients without intercept
        params = {'fit_intercept': False}
        self._check_debiased_coefs(TestLassoExtensions.X, TestLassoExtensions.y_simple, sample_weight=None,
                                   expected_coefs=TestLassoExtensions.coefs1,
                                   expected_intercept=TestLassoExtensions.intercept,
                                   params=params)
        # --> Check debiased coeffcients with intercept
        self._check_debiased_coefs(TestLassoExtensions.X,
                                   TestLassoExtensions.y_simple + TestLassoExtensions.intercept,
                                   sample_weight=None,
                                   expected_coefs=TestLassoExtensions.coefs1,
                                   expected_intercept=TestLassoExtensions.intercept)
        # --> Check 5-95 CI coverage for unit vectors
        self._check_debiased_CI(TestLassoExtensions.X, TestLassoExtensions.y_simple + TestLassoExtensions.intercept,
                                sample_weight=None,
                                expected_coefs=TestLassoExtensions.coefs1,
                                expected_intercept=TestLassoExtensions.intercept)
        # Test DebiasedLasso with weights for one DGP
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.ones(TestLassoExtensions.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate(
            (TestLassoExtensions.X, TestLassoExtensions.X[TestLassoExtensions.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestLassoExtensions.y_simple, TestLassoExtensions.y_simple[TestLassoExtensions.n_samples // 2:]))
        # --> Check debiased coefficients
        weighted_debiased_coefs = self._check_debiased_coefs(TestLassoExtensions.X, TestLassoExtensions.y_simple,
                                                             sample_weight=sample_weight,
                                                             expected_coefs=TestLassoExtensions.coefs1)
        expanded_debiased_coefs = self._check_debiased_coefs(X_expanded, y_expanded, sample_weight=None,
                                                             expected_coefs=TestLassoExtensions.coefs1)
        np.testing.assert_allclose(weighted_debiased_coefs, expanded_debiased_coefs, atol=5e-2)

    def test_debiased_lasso_mixed_DGP(self):
        """Test WeightedLasso with two sets of coefficients."""
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.zeros(TestLassoExtensions.n_samples // 2)))
        # Data from one DGP has weight 0. Check that we recover correct coefficients
        # --> Check debiased coeffcients
        self._check_debiased_coefs(TestLassoExtensions.X, TestLassoExtensions.y,
                                   sample_weight=sample_weight,
                                   expected_coefs=TestLassoExtensions.coefs1,
                                   expected_intercept=TestLassoExtensions.intercept1)

    def test_multi_output_debiased_lasso(self):
        """Test MultiOutputDebiasedLasso."""
        # Test that attributes propagate correctly
        est = MultiOutputDebiasedLasso()
        multioutput_attrs = est.get_params()
        debiased_attrs = DebiasedLasso().get_params()
        for attr in debiased_attrs:
            self.assertTrue(attr in multioutput_attrs)
        # Test MultiOutputDebiasedLasso without weights
        # --> Check debiased coeffcients without intercept
        params = {'fit_intercept': False}
        self._check_debiased_coefs(TestLassoExtensions.X, TestLassoExtensions.y_2D_consistent,
                                   sample_weight=None,
                                   expected_coefs=[TestLassoExtensions.coefs1, TestLassoExtensions.coefs2],
                                   params=params)
        # --> Check debiased coeffcients with intercept
        intercept_2D = np.array([TestLassoExtensions.intercept1, TestLassoExtensions.intercept2])
        self._check_debiased_coefs(TestLassoExtensions.X,
                                   TestLassoExtensions.y_2D_consistent + intercept_2D,
                                   sample_weight=None,
                                   expected_coefs=[TestLassoExtensions.coefs1, TestLassoExtensions.coefs2],
                                   expected_intercept=intercept_2D)
        # --> Check CI coverage
        self._check_debiased_CI_2D(TestLassoExtensions.X,
                                   TestLassoExtensions.y_2D_consistent + intercept_2D,
                                   sample_weight=None,
                                   expected_coefs=np.array([TestLassoExtensions.coefs1, TestLassoExtensions.coefs2]),
                                   expected_intercept=intercept_2D)
        # Test MultiOutputDebiasedLasso with weights
        # Define weights
        sample_weight = np.concatenate((np.ones(TestLassoExtensions.n_samples // 2),
                                        np.ones(TestLassoExtensions.n_samples // 2) * 2))
        # Define extended datasets
        X_expanded = np.concatenate(
            (TestLassoExtensions.X, TestLassoExtensions.X[TestLassoExtensions.n_samples // 2:]))
        y_expanded = np.concatenate(
            (TestLassoExtensions.y_2D_consistent,
             TestLassoExtensions.y_2D_consistent[TestLassoExtensions.n_samples // 2:]))
        # --> Check debiased coefficients
        weighted_debiased_coefs = self._check_debiased_coefs(
            TestLassoExtensions.X,
            TestLassoExtensions.y_2D_consistent,
            sample_weight=sample_weight,
            expected_coefs=[TestLassoExtensions.coefs1, TestLassoExtensions.coefs2],
            params=params)
        expanded_debiased_coefs = self._check_debiased_coefs(
            X_expanded, y_expanded, sample_weight=None,
            expected_coefs=[TestLassoExtensions.coefs1, TestLassoExtensions.coefs2],
            params=params)
        for i in range(2):
            np.testing.assert_allclose(weighted_debiased_coefs[i], expanded_debiased_coefs[i], atol=5e-2)

    def _check_debiased_CI(self,
                           X, y, sample_weight, expected_coefs,
                           expected_intercept=0, n_experiments=200, params={}):
        # Unit vectors
        X_test = np.eye(TestLassoExtensions.n_dim)
        y_test_mean = expected_intercept + expected_coefs
        is_in_interval = np.zeros((n_experiments, TestLassoExtensions.n_dim))
        for i in range(n_experiments):
            np.random.seed(i)
            X_exp = np.random.normal(size=X.shape)
            err = np.random.normal(scale=TestLassoExtensions.error_sd, size=X.shape[0])
            y_exp = expected_intercept + np.dot(X_exp, expected_coefs) + err
            debiased_lasso = DebiasedLasso()
            debiased_lasso.set_params(**params)
            debiased_lasso.fit(X_exp, y_exp, sample_weight)
            y_lower, y_upper = debiased_lasso.predict_interval(X_test, alpha=0.1)
            is_in_interval[i] = ((y_test_mean >= y_lower) & (y_test_mean <= y_upper))
        CI_coverage = np.mean(is_in_interval, axis=0)
        self.assertTrue(all(CI_coverage >= 0.85))
        self.assertTrue(all(CI_coverage <= 0.95))

    def _check_debiased_CI_2D(self,
                              X, y, sample_weight, expected_coefs,
                              expected_intercept=0, n_experiments=200, params={}):
        # Unit vectors
        X_test = np.eye(TestLassoExtensions.n_dim)
        y_test_mean = expected_intercept + expected_coefs.T
        is_in_interval = np.zeros((y.shape[1], n_experiments, TestLassoExtensions.n_dim))
        for i in range(n_experiments):
            np.random.seed(i)
            X_exp = np.random.normal(size=X.shape)
            err = np.random.normal(scale=TestLassoExtensions.error_sd, size=(X.shape[0], y.shape[1]))
            y_exp = expected_intercept + np.dot(X_exp, expected_coefs.T) + err
            debiased_lasso = MultiOutputDebiasedLasso()
            debiased_lasso.set_params(**params)
            debiased_lasso.fit(X_exp, y_exp, sample_weight)
            y_lower, y_upper = debiased_lasso.predict_interval(X_test, alpha=0.1)
            for j in range(y.shape[1]):
                is_in_interval[j, i, :] = ((y_test_mean[:, j] >= y_lower[:, j]) & (y_test_mean[:, j] <= y_upper[:, j]))
        for i in range(y.shape[1]):
            CI_coverage = np.mean(is_in_interval[i], axis=0)
            self.assertTrue(all(CI_coverage >= 0.85))

    def _check_debiased_coefs(self, X, y, sample_weight, expected_coefs, expected_intercept=0, params={}):
        debiased_lasso = MultiOutputDebiasedLasso() if np.ndim(y) > 1 else DebiasedLasso()
        debiased_lasso.set_params(**params)
        debiased_lasso.fit(X, y, sample_weight)
        all_params = debiased_lasso.get_params()
        # Check coeffcients and intercept are the same within tolerance
        if np.ndim(y) > 1:
            for i in range(y.shape[1]):
                np.testing.assert_allclose(debiased_lasso.coef_[i], expected_coefs[i], atol=5e-2)
                if all_params["fit_intercept"]:
                    self.assertAlmostEqual(debiased_lasso.intercept_[i], expected_intercept[i], delta=1e-2)
        else:
            np.testing.assert_allclose(debiased_lasso.coef_, expected_coefs, atol=5e-2)
            if all_params["fit_intercept"]:
                self.assertAlmostEqual(debiased_lasso.intercept_, expected_intercept, delta=1e-2)
        return debiased_lasso.coef_

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
                    np.testing.assert_allclose(lasso.coef_[i], wlasso.coef_[i])
                    if lasso.get_params()["fit_intercept"]:
                        self.assertAlmostEqual(lasso.intercept_[i], wlasso.intercept_[i])
            else:
                np.testing.assert_allclose(lasso.coef_, wlasso.coef_)
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
                np.testing.assert_allclose(lassoCV.coef_[i], wlassoCV.coef_[i], atol=tol)
                if lassoCV.get_params()["fit_intercept"]:
                    self.assertAlmostEqual(lassoCV.intercept_[i], wlassoCV.intercept_[i])
        else:
            np.testing.assert_allclose(lassoCV.coef_, wlassoCV.coef_, atol=tol)
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

    def test_can_slice(self):
        n = 100 + np.random.choice(500)
        d = 5 + np.random.choice(20)
        alpha = np.random.uniform(0.5, 1.5)
        X = np.random.normal(size=(n, d))
        y = np.random.normal(size=(n,))
        coef = SelectiveRegularization(unpenalized_inds=slice(2, None),
                                       penalized_model=Lasso(),
                                       fit_intercept=True).fit(X, y).coef_
        X_perm = np.hstack((X[:, 1:],
                            X[:, :1]))
        coef2 = SelectiveRegularization(unpenalized_inds=slice(1, -1),
                                        penalized_model=Lasso(),
                                        fit_intercept=True).fit(X_perm, y).coef_
        np.testing.assert_allclose(coef2, np.hstack((coef[1:],
                                                     coef[:1])))

    def test_can_use_index_lambda(self):
        n = 100 + np.random.choice(500)
        d = 5 + np.random.choice(20)
        alpha = np.random.uniform(0.5, 1.5)
        X = np.random.normal(size=(n, d))
        y = np.random.normal(size=(n,))
        coef = SelectiveRegularization(unpenalized_inds=slice(2, None),
                                       penalized_model=Lasso(),
                                       fit_intercept=True).fit(X, y).coef_

        def index_lambda(X, y):
            # instead of a slice, explicitly return an array of indices
            return np.arange(2, X.shape[1])
        coef2 = SelectiveRegularization(unpenalized_inds=index_lambda,
                                        penalized_model=Lasso(),
                                        fit_intercept=True).fit(X, y).coef_
        np.testing.assert_allclose(coef, coef2)

    def test_can_pass_through_attributes(self):
        X = np.random.normal(size=(10, 3))
        y = np.random.normal(size=(10,))
        model = SelectiveRegularization(unpenalized_inds=[0],
                                        penalized_model=LassoCV(),
                                        fit_intercept=True)

        # _penalized_inds is only set during fitting
        with self.assertRaises(AttributeError):
            inds = model._penalized_inds

        # cv exists on penalized model
        old_cv = model.cv
        model.cv = 2

        model.fit(X, y)

        # now we can access _penalized_inds
        assert np.array_equal(model._penalized_inds, [1, 2])

        # check that we can read the cv attribute back out from the underlying model
        assert model.cv == 2

    def test_can_clone_selective_regularization(self):
        X = np.random.normal(size=(10, 3))
        y = np.random.normal(size=(10,))
        model = SelectiveRegularization(unpenalized_inds=[0],
                                        penalized_model=LassoCV(),
                                        fit_intercept=True)
        model.cv = 2
        model2 = clone(model, safe=False)
        assert model2.cv == 2
