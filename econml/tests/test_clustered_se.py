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

    def test_clustered_micro_equals_aggregated(self):
        """Test that clustered SE matches for summarized and non-summarized data."""

        def _generate_micro_and_aggregated(rng, *, n_groups=12, cells_per_group=6, d=4, p=1):
            """Build a micro dataset and aggregated counterpart with many freq > 1."""
            G = n_groups
            K = cells_per_group
            N = G * K

            # Design
            X = rng.normal(size=(N, d))
            # True coefficients used just to generate data; intercept will be fit by the model
            beta_true = rng.normal(size=(d + 1, p))

            # Positive sample weights and integer freq weights with many freq > 1
            sw = np.exp(rng.normal(scale=0.3, size=N))
            freq = rng.integers(1, 6, size=N)  # values in {1,2,3,4,5}

            # Group labels
            groups = np.repeat(np.arange(G), K)

            # Build micro outcomes y_{ij}
            ybar = np.zeros((N, p), dtype=float)
            svar = np.zeros((N, p), dtype=float)

            X_micro, y_micro, sw_micro, groups_micro = [], [], [], []

            for i in range(N):
                f = int(freq[i])
                x_i = X[i]
                mu_i = np.concatenate(([1.0], x_i)) @ beta_true  # shape (p,)
                eps = rng.normal(scale=1.0, size=(f, p))
                y_ij = mu_i + eps  # shape (f, p)

                X_micro.append(np.repeat(x_i[None, :], f, axis=0))
                y_micro.append(y_ij)
                sw_micro.append(np.repeat(sw[i], f))
                groups_micro.append(np.repeat(groups[i], f))

                ybar[i, :] = y_ij.mean(axis=0)
                svar[i, :] = y_ij.var(axis=0, ddof=0)

            X_micro = np.vstack(X_micro)
            y_micro = np.vstack(y_micro)
            sw_micro = np.concatenate(sw_micro)
            groups_micro = np.concatenate(groups_micro)

            if p == 1:
                ybar = ybar.ravel()
                svar = svar.ravel()
                y_micro = y_micro.ravel()

            return (X, ybar, sw, freq, svar, groups), (X_micro, y_micro, sw_micro, groups_micro)

        rng = np.random.default_rng(7)
        for p in [1, 3]:
            (X, ybar, sw, freq, svar, groups), (X_micro, y_micro, sw_micro, groups_micro) = \
                _generate_micro_and_aggregated(rng, n_groups=10, cells_per_group=7, d=5, p=p)

            m_agg = StatsModelsLinearRegression(fit_intercept=True, cov_type="clustered", enable_federation=False)
            m_agg.fit(X, ybar, sample_weight=sw, freq_weight=freq, sample_var=svar, groups=groups)

            m_micro = StatsModelsLinearRegression(fit_intercept=True, cov_type="clustered",
                                                 enable_federation=False)
            m_micro.fit(
                X_micro,
                y_micro,
                sample_weight=sw_micro,
                freq_weight=None,
                sample_var=None,
                groups=groups_micro
            )

            np.testing.assert_allclose(m_agg._param, m_micro._param, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(np.array(m_agg._param_var), np.array(m_micro._param_var),
                                       rtol=1e-10, atol=1e-12)
