# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np
import pytest
from sklearn.linear_model import LassoCV, LogisticRegression
import statsmodels.api as sm
from econml.dml import DML
from econml.iv.dml import OrthoIV
from econml.utilities import shape
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression


@pytest.mark.cate_api
class TestClusteredSE(unittest.TestCase):

    def test_clustered_se_dml(self):
        """Test that LinearDML works with clustered standard errors."""
        np.random.seed(123)
        n = 500
        n_groups = 25

        # Generate data with clustering structure
        X = np.random.normal(0, 1, (n, 3))
        W = np.random.normal(0, 1, (n, 2))
        groups = np.random.randint(0, n_groups, n)
        T = np.random.binomial(1, 0.5, n)

        # Add group-level effects to create clustering
        group_effects = np.random.normal(0, 1, n_groups)
        Y = X[:, 0] + 2 * T + group_effects[groups] + np.random.normal(0, 0.5, n)

        # Test DML with clustered standard errors via custom model_final
        est = DML(model_y=LassoCV(), model_t=LogisticRegression(),
                 model_final=StatsModelsLinearRegression(fit_intercept=False, cov_type='clustered'),
                 discrete_treatment=True)
        est.fit(Y, T, X=X, W=W, groups=groups)

        # Test basic functionality
        effects = est.effect(X[:10])
        self.assertEqual(shape(effects), (10,))

        # Test confidence intervals
        lb, ub = est.effect_interval(X[:10], alpha=0.05)
        self.assertEqual(shape(lb), (10,))
        self.assertEqual(shape(ub), (10,))
        self.assertTrue(np.all(lb <= ub))

        # Test that clustered SEs are different from non-clustered
        est_regular = DML(model_y=LassoCV(), model_t=LogisticRegression(),
                         model_final=StatsModelsLinearRegression(fit_intercept=False, cov_type='nonrobust'),
                         discrete_treatment=True)
        est_regular.fit(Y, T, X=X, W=W)

        lb_regular, ub_regular = est_regular.effect_interval(X[:10], alpha=0.05)

        # Confidence intervals should be different (not identical)
        self.assertFalse(np.allclose(lb, lb_regular, atol=1e-10))
        self.assertFalse(np.allclose(ub, ub_regular, atol=1e-10))

    def test_clustered_se_iv(self):
        """Test that OrthoIV works with clustered standard errors."""
        np.random.seed(123)
        n = 500
        n_groups = 25

        # Generate data with clustering structure
        X = np.random.normal(0, 1, (n, 3))
        W = np.random.normal(0, 1, (n, 2))
        groups = np.random.randint(0, n_groups, n)
        Z = np.random.binomial(1, 0.5, n)
        T = np.random.binomial(1, 0.5, n)

        # Add group-level effects to create clustering
        group_effects = np.random.normal(0, 1, n_groups)
        Y = X[:, 0] + 2 * T + group_effects[groups] + np.random.normal(0, 0.5, n)

        # Test OrthoIV with clustered standard errors
        est = OrthoIV(discrete_treatment=True, discrete_instrument=True,
                     cov_type='clustered')
        est.fit(Y, T, Z=Z, X=X, W=W, groups=groups)

        # Test basic functionality
        effects = est.effect(X[:10])
        self.assertEqual(shape(effects), (10,))

        # Test confidence intervals
        lb, ub = est.effect_interval(X[:10], alpha=0.05)
        self.assertEqual(shape(lb), (10,))
        self.assertEqual(shape(ub), (10,))
        self.assertTrue(np.all(lb <= ub))

        # Test that clustered SEs are different from non-clustered
        est_regular = OrthoIV(discrete_treatment=True, discrete_instrument=True,
                             cov_type='nonrobust')
        est_regular.fit(Y, T, Z=Z, X=X, W=W)

        lb_regular, ub_regular = est_regular.effect_interval(X[:10], alpha=0.05)

        # Confidence intervals should be different (not identical)
        self.assertFalse(np.allclose(lb, lb_regular, atol=1e-10))
        self.assertFalse(np.allclose(ub, ub_regular, atol=1e-10))

    def test_clustered_se_without_groups_defaults_to_individual(self):
        """Test that clustered SE without groups matches HC0 with adjustment factor."""
        np.random.seed(123)
        n = 100
        X = np.random.normal(0, 1, (n, 2))
        T = np.random.binomial(1, 0.5, n)
        Y = np.random.normal(0, 1, n)

        # Clustered SE without groups (defaults to individual groups)
        np.random.seed(123)
        est_clustered = DML(model_y=LassoCV(), model_t=LogisticRegression(),
                           model_final=StatsModelsLinearRegression(fit_intercept=False, cov_type='clustered'),
                           discrete_treatment=True)
        est_clustered.fit(Y, T, X=X)

        # HC0 for comparison
        np.random.seed(123)
        est_hc0 = DML(model_y=LassoCV(), model_t=LogisticRegression(),
                     model_final=StatsModelsLinearRegression(fit_intercept=False, cov_type='HC0'),
                     discrete_treatment=True)
        est_hc0.fit(Y, T, X=X)

        # Get confidence intervals
        X_test = X[:5]
        lb_clustered, ub_clustered = est_clustered.effect_interval(X_test, alpha=0.05)
        lb_hc0, ub_hc0 = est_hc0.effect_interval(X_test, alpha=0.05)

        # Clustered SE should be HC0 SE * sqrt(n/(n-1)) when each obs is its own cluster
        # Width of confidence intervals should differ by the adjustment factor
        width_clustered = ub_clustered - lb_clustered
        width_hc0 = ub_hc0 - lb_hc0

        # When each observation is its own cluster, clustered SE should equal HC0 * sqrt(n/(n-1))
        # due to the finite sample correction factor
        correction_factor = np.sqrt(n / (n - 1))
        expected_width = width_hc0 * correction_factor
        np.testing.assert_allclose(width_clustered, expected_width, rtol=1e-10)

        # Test basic functionality still works
        effects = est_clustered.effect(X_test)
        self.assertEqual(shape(effects), (5,))
        self.assertTrue(np.all(np.isfinite(effects)))
        self.assertTrue(np.all(np.isfinite(lb_clustered)))
        self.assertTrue(np.all(np.isfinite(ub_clustered)))

    def test_clustered_se_matches_statsmodels(self):
        """Test that our final stage clustered SE matches statsmodels exactly."""
        np.random.seed(42)
        n = 200
        n_groups = 20

        # Generate simple data for direct comparison with clustering
        X = np.random.normal(0, 1, (n, 2))
        groups = np.random.randint(0, n_groups, n)
        group_effects = np.random.normal(0, 0.5, n_groups)
        Y = 1 + 2 * X[:, 0] + 3 * X[:, 1] + group_effects[groups] + np.random.normal(0, 0.5, n)

        # Fit with our StatsModelsLinearRegression directly
        X_with_intercept = np.column_stack([np.ones(n), X])
        econml_model = StatsModelsLinearRegression(cov_type='clustered')
        econml_model.fit(X, Y, groups=groups)
        econml_se = econml_model.coef_stderr_[0]  # SE for X[:, 0] coefficient

        # Fit equivalent model with statsmodels
        sm_model = sm.OLS(Y, X_with_intercept).fit(cov_type='cluster', cov_kwds={'groups': groups})
        sm_se = sm_model.bse[1]  # SE for X[:, 0] coefficient

        # Account for statsmodels' additional n/(n-k) adjustment
        k = X_with_intercept.shape[1]  # Number of parameters
        sm_adjustment = np.sqrt((n - 1) / (n - k))
        adjusted_sm_se = sm_se / sm_adjustment

        # Should match very closely
        relative_diff = abs(econml_se - adjusted_sm_se) / adjusted_sm_se
        self.assertLess(relative_diff, 1e-4,
                       f"EconML SE ({econml_se:.8f}) differs from adjusted statsmodels SE ({adjusted_sm_se:.8f})")
