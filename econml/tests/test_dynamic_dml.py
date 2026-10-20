# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
import unittest
import pytest
import pickle
import numpy as np
from contextlib import ExitStack
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (LassoCV, Lasso, LinearRegression, MultiTaskLasso,
                                  LogisticRegression)
from econml.panel.dml import DynamicDML
from econml.panel.dml._dml import _get_groups_period_filter
from econml.inference import BootstrapInference
from econml.utilities import shape
from econml.tests.dgp import DynamicPanelDGP


@pytest.mark.parametrize("n_periods", [1, 3])
@pytest.mark.parametrize("d_t", [1, 2])
@pytest.mark.parametrize("d_x", [0, 2])
@pytest.mark.parametrize("d_y", [None, 1, 2])
def test_dynamic_covariance_score_derivative(n_periods, d_t, d_x, d_y):
    """Compare fitted covariance and effect inference to differentiated panel scores."""
    rng = np.random.default_rng(71)
    n_units = 160
    groups = np.repeat(np.arange(n_units), n_periods)
    W = rng.normal(size=(n_units * n_periods, 2))
    X = rng.normal(size=(n_units * n_periods, d_x)) if d_x else None
    T = rng.normal(size=(n_units, n_periods, d_t))
    # Nonsymmetric coupling exposes transposed cross-period blocks that scalar or symmetric cases hide.
    coupling = np.array([[0.4, 1.3], [-0.7, 0.2]])[:d_t, :d_t]
    for t in range(1, n_periods):
        T[:, t] += T[:, t - 1] @ coupling
    Y = rng.normal(size=(n_units * n_periods,) + (() if d_y is None else (d_y,)))
    T = T.reshape(-1, d_t)
    if d_t == 1:
        T = T.ravel()
    est = DynamicDML(model_y=LinearRegression(), model_t=LinearRegression(), cv=2, random_state=123)
    est.fit(Y, T, X=X, W=W, groups=groups, cache_values=True)

    # Reuse fitted nuisance residuals to isolate final-stage correctness: this is an independent
    # estimating-equation reference, not primarily an invariance test or end-to-end DGP calibration.
    Y_res, T_res, X_res, _ = est.residuals_
    filters = _get_groups_period_filter(est._cached_values.groups, n_periods)
    features = np.ones((n_units, 1))
    if X_res is not None:
        features = np.column_stack([features, X_res[filters[0]]])
    T_res = T_res.reshape(-1, d_t, n_periods)
    Y_res = Y_res.reshape(-1, d_y or 1)
    # design[j, t] uses period-j treatment residuals relative to period-t information, interacted
    # with baseline features. Explicit treatment-major columns keep the reference independent of _combine.
    design = {
        (j, t): np.column_stack([T_res[filters[j], k, t] * features[:, f]
                                for k in range(d_t) for f in range(d_x + 1)])
        for t in range(n_periods) for j in range(t, n_periods)
    }
    width = d_t * (d_x + 1)
    beta = np.zeros((n_periods, width, d_y or 1))
    # Independent backward regressions verify unchanged point coefficients before checking uncertainty.
    for t in reversed(range(n_periods)):
        adjusted = Y_res[filters[t]].copy()
        for j in range(t + 1, n_periods):
            adjusted -= design[j, t] @ beta[j]
        beta[t] = np.linalg.lstsq(design[t, t], adjusted, rcond=None)[0]
    expected_coef = beta.reshape(n_periods * width, d_y or 1).T
    np.testing.assert_allclose(est.model_final_.coef_,
                               expected_coef[0] if d_y is None else expected_coef, atol=1e-12)

    expected_cov = []
    for outcome in range(d_y or 1):
        # Each row is one panel unit's stacked estimating equations, keeping its periods together.
        def scores(params):
            coefs = params.reshape(n_periods, width)
            return np.column_stack([
                design[t, t] * (Y_res[filters[t], outcome] -
                                sum(design[j, t] @ coefs[j] for j in range(t, n_periods)))[:, None]
                for t in range(n_periods)
            ])

        params = expected_coef[outcome]
        steps = 1e-5 * np.eye(params.size)
        # Negative finite differences put score coordinates on rows and coefficient coordinates
        # on columns, without reproducing the production outer-product formula.
        jacobian = np.column_stack([
            -(scores(params + step).mean(axis=0) - scores(params - step).mean(axis=0)) / 2e-5
            for step in steps
        ])
        influence = np.linalg.solve(jacobian, scores(params).T).T
        # Coefficient sampling covariance, not outcome variance: independent units contribute
        # influence vectors, and their outer-product sum / n_units**2 is covariance of the average.
        expected_cov.append(influence.T @ influence / n_units**2)
    expected_cov = np.array(expected_cov)
    np.testing.assert_allclose(est.model_final_._param_var,
                               expected_cov[0] if d_y is None else expected_cov, rtol=1e-8, atol=1e-10)

    # A treatment contrast checks that public effect inference receives the covariance in the
    # correct period/treatment/feature layout.
    X_test = rng.normal(size=(4, d_x)) if d_x else None
    test_features = np.column_stack([np.ones(4), X_test]) if d_x else np.ones((1, 1))
    contrast = rng.normal(size=n_periods * d_t)
    effect_design = np.column_stack([contrast[k] * test_features[:, f]
                                    for k in range(contrast.size) for f in range(d_x + 1)])
    expected_stderr = np.column_stack([
        np.sqrt(np.einsum('ij,jk,ik->i', effect_design, cov, effect_design)) for cov in expected_cov
    ])
    T1 = np.tile(contrast, (test_features.shape[0], 1))
    inference = est.effect_inference(X_test, T0=np.zeros_like(T1), T1=T1)
    np.testing.assert_allclose(inference.stderr,
                               expected_stderr[:, 0] if d_y is None else expected_stderr,
                               rtol=1e-8, atol=1e-10)


@pytest.mark.cate_api
class TestDynamicDML(unittest.TestCase):

    def test_cate_api(self):
        """Test that we correctly implement the CATE API."""
        n_panels = 100  # number of panels
        n_periods = 3  # number of time periods per panel
        n = n_panels * n_periods
        groups = np.repeat(a=np.arange(n_panels), repeats=n_periods, axis=0)

        def make_random(n, is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d >= 0 else (n,)
            if is_discrete:
                return np.random.choice(['a', 'b', 'c'], size=sz)
            else:
                return np.random.normal(size=sz)

        for d_t in [2, 1, -1]:
            for is_discrete in [True, False] if d_t <= 1 else [False]:
                # for is_discrete in [False]:
                for d_y in [3, 1, -1]:
                    for d_x in [2, None]:
                        for d_w in [2, None]:
                            W, X, Y, T = [make_random(n, is_discrete, d)
                                          for is_discrete, d in [(False, d_w),
                                                                 (False, d_x),
                                                                 (False, d_y),
                                                                 (is_discrete, d_t)]]
                            T_test = np.hstack([(T.reshape(-1, 1) if d_t == -1 else T) for i in range(n_periods)])
                            for featurizer, fit_cate_intercept in\
                                [(None, True),
                                 (PolynomialFeatures(degree=2, include_bias=False), True),
                                 (PolynomialFeatures(degree=2, include_bias=True), False)]:

                                d_t_final = (2 if is_discrete else max(d_t, 1)) * n_periods

                                effect_shape = (n,) + ((d_y,) if d_y > 0 else ())
                                effect_summaryframe_shape = (n * (d_y if d_y > 0 else 1), 6)
                                marginal_effect_shape = ((n,) +
                                                         ((d_y,) if d_y > 0 else ()) +
                                                         ((d_t_final,) if d_t_final > 0 else ()))
                                marginal_effect_summaryframe_shape = (n * (d_y if d_y > 0 else 1) *
                                                                      (d_t_final if d_t_final > 0 else 1), 6)

                                # since T isn't passed to const_marginal_effect, defaults to one row if X is None
                                const_marginal_effect_shape = ((n if d_x else 1,) +
                                                               ((d_y,) if d_y > 0 else ()) +
                                                               ((d_t_final,) if d_t_final > 0 else ()))
                                const_marginal_effect_summaryframe_shape = (
                                    (n if d_x else 1) * (d_y if d_y > 0 else 1) *
                                    (d_t_final if d_t_final > 0 else 1), 6)

                                fd_x = featurizer.fit_transform(X).shape[1:] if featurizer and d_x\
                                    else ((d_x,) if d_x else (0,))
                                coef_shape = Y.shape[1:] + (d_t_final, ) + fd_x

                                coef_summaryframe_shape = (
                                    (d_y if d_y > 0 else 1) * (fd_x[0] if fd_x[0] >
                                                               0 else 1) * (d_t_final), 6)
                                intercept_shape = Y.shape[1:] + (d_t_final, )
                                intercept_summaryframe_shape = (
                                    (d_y if d_y > 0 else 1) * (d_t_final if d_t_final > 0 else 1), 6)

                                all_infs = [None, 'auto', BootstrapInference(2)]
                                est = DynamicDML(model_y=Lasso() if d_y < 1 else MultiTaskLasso(),
                                                 model_t=LogisticRegression() if is_discrete else
                                                 (Lasso() if d_t < 1 else MultiTaskLasso()),
                                                 featurizer=featurizer,
                                                 fit_cate_intercept=fit_cate_intercept,
                                                 discrete_treatment=is_discrete)

                                # ensure we can serialize the unfit estimator
                                pickle.dumps(est)

                                for inf in all_infs:
                                    with self.subTest(d_w=d_w, d_x=d_x, d_y=d_y, d_t=d_t,
                                                      is_discrete=is_discrete, est=type(est).__name__, inf=repr(inf)):

                                        if X is None and (not fit_cate_intercept):
                                            with pytest.raises(AttributeError):
                                                est.fit(Y, T, X=X, W=W, groups=groups, inference=inf)
                                            continue

                                        est.fit(Y, T, X=X, W=W, groups=groups, inference=inf)

                                        # ensure we can pickle the fit estimator
                                        pickle.dumps(est)

                                        # make sure we can call the marginal_effect and effect methods
                                        const_marg_eff = est.const_marginal_effect(X)
                                        marg_eff = est.marginal_effect(T_test, X)
                                        self.assertEqual(shape(marg_eff), marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_eff), const_marginal_effect_shape)

                                        np.testing.assert_allclose(
                                            marg_eff if d_x else marg_eff[0:1], const_marg_eff)

                                        assert len(est.score_) == n_periods
                                        for score in est.nuisance_scores_y[0]:
                                            assert score.shape == (n_periods, )
                                        for score in est.nuisance_scores_t[0]:
                                            assert score.shape == (n_periods, n_periods)

                                        T0 = np.full_like(T_test, 'a') if is_discrete else np.zeros_like(T_test)
                                        eff = est.effect(X, T0=T0, T1=T_test)
                                        self.assertEqual(shape(eff), effect_shape)

                                        self.assertEqual(shape(est.coef_), coef_shape)
                                        if fit_cate_intercept:
                                            self.assertEqual(shape(est.intercept_), intercept_shape)
                                        else:
                                            with pytest.raises(AttributeError):
                                                self.assertEqual(shape(est.intercept_), intercept_shape)

                                        if inf is not None:
                                            const_marg_eff_int = est.const_marginal_effect_interval(X)
                                            marg_eff_int = est.marginal_effect_interval(T_test, X)
                                            self.assertEqual(shape(marg_eff_int),
                                                             (2,) + marginal_effect_shape)
                                            self.assertEqual(shape(const_marg_eff_int),
                                                             (2,) + const_marginal_effect_shape)
                                            self.assertEqual(shape(est.effect_interval(X, T0=T0, T1=T_test)),
                                                             (2,) + effect_shape)
                                            self.assertEqual(shape(est.coef__interval()),
                                                             (2,) + coef_shape)
                                            if fit_cate_intercept:
                                                self.assertEqual(shape(est.intercept__interval()),
                                                                 (2,) + intercept_shape)
                                            else:
                                                with pytest.raises(AttributeError):
                                                    self.assertEqual(shape(est.intercept__interval()),
                                                                     (2,) + intercept_shape)

                                            const_marg_effect_inf = est.const_marginal_effect_inference(X)
                                            T1 = np.full_like(T_test, 'b') if is_discrete else T_test
                                            effect_inf = est.effect_inference(X, T0=T0, T1=T1)
                                            marg_effect_inf = est.marginal_effect_inference(T_test, X)
                                            # test const marginal inference
                                            self.assertEqual(shape(const_marg_effect_inf.summary_frame()),
                                                             const_marginal_effect_summaryframe_shape)
                                            self.assertEqual(shape(const_marg_effect_inf.point_estimate),
                                                             const_marginal_effect_shape)
                                            self.assertEqual(shape(const_marg_effect_inf.stderr),
                                                             const_marginal_effect_shape)
                                            self.assertEqual(shape(const_marg_effect_inf.var),
                                                             const_marginal_effect_shape)
                                            self.assertEqual(shape(const_marg_effect_inf.pvalue()),
                                                             const_marginal_effect_shape)
                                            self.assertEqual(shape(const_marg_effect_inf.zstat()),
                                                             const_marginal_effect_shape)
                                            self.assertEqual(shape(const_marg_effect_inf.conf_int()),
                                                             (2,) + const_marginal_effect_shape)
                                            np.testing.assert_array_almost_equal(
                                                const_marg_effect_inf.conf_int()[0],
                                                const_marg_eff_int[0], decimal=5)
                                            const_marg_effect_inf.population_summary()._repr_html_()

                                            # test effect inference
                                            self.assertEqual(shape(effect_inf.summary_frame()),
                                                             effect_summaryframe_shape)
                                            self.assertEqual(shape(effect_inf.point_estimate),
                                                             effect_shape)
                                            self.assertEqual(shape(effect_inf.stderr),
                                                             effect_shape)
                                            self.assertEqual(shape(effect_inf.var),
                                                             effect_shape)
                                            self.assertEqual(shape(effect_inf.pvalue()),
                                                             effect_shape)
                                            self.assertEqual(shape(effect_inf.zstat()),
                                                             effect_shape)
                                            self.assertEqual(shape(effect_inf.conf_int()),
                                                             (2,) + effect_shape)
                                            np.testing.assert_array_almost_equal(
                                                effect_inf.conf_int()[0],
                                                est.effect_interval(X, T0=T0, T1=T1)[0], decimal=5)
                                            effect_inf.population_summary()._repr_html_()

                                            # test marginal effect inference
                                            self.assertEqual(shape(marg_effect_inf.summary_frame()),
                                                             marginal_effect_summaryframe_shape)
                                            self.assertEqual(shape(marg_effect_inf.point_estimate),
                                                             marginal_effect_shape)
                                            self.assertEqual(shape(marg_effect_inf.stderr),
                                                             marginal_effect_shape)
                                            self.assertEqual(shape(marg_effect_inf.var),
                                                             marginal_effect_shape)
                                            self.assertEqual(shape(marg_effect_inf.pvalue()),
                                                             marginal_effect_shape)
                                            self.assertEqual(shape(marg_effect_inf.zstat()),
                                                             marginal_effect_shape)
                                            self.assertEqual(shape(marg_effect_inf.conf_int()),
                                                             (2,) + marginal_effect_shape)
                                            np.testing.assert_array_almost_equal(
                                                marg_effect_inf.conf_int()[0], marg_eff_int[0], decimal=5)
                                            marg_effect_inf.population_summary()._repr_html_()

                                            # test coef__inference and intercept__inference
                                            if X is not None:
                                                self.assertEqual(
                                                    shape(est.coef__inference().summary_frame()),
                                                    coef_summaryframe_shape)
                                                np.testing.assert_array_almost_equal(
                                                    est.coef__inference().conf_int()
                                                    [0], est.coef__interval()[0], decimal=5)

                                            if fit_cate_intercept:
                                                cm = ExitStack()
                                                # ExitStack can be used as a "do nothing" ContextManager
                                            else:
                                                cm = pytest.raises(AttributeError)
                                            with cm:
                                                self.assertEqual(shape(est.intercept__inference().
                                                                       summary_frame()),
                                                                 intercept_summaryframe_shape)
                                                np.testing.assert_array_almost_equal(
                                                    est.intercept__inference().conf_int()
                                                    [0], est.intercept__interval()[0], decimal=5)

                                            est.summary()
                                        est.score(Y, T, X, W, groups=groups)
                                        # make sure we can call effect with implied scalar treatments,
                                        # no matter the dimensions of T, and also that we warn when there
                                        # are multiple treatments
                                        if d_t > 1:
                                            cm = self.assertWarns(Warning)
                                        else:
                                            # ExitStack can be used as a "do nothing" ContextManager
                                            cm = ExitStack()
                                        with cm:
                                            effect_shape2 = (n if d_x else 1,) + ((d_y,) if d_y > 0 else ())
                                            eff = est.effect(X) if not is_discrete else est.effect(
                                                X, T0='a', T1='b')
                                            self.assertEqual(shape(eff), effect_shape2)

    def test_perf(self):
        np.random.seed(123)
        n_units = 1000
        n_periods = 3
        n_treatments = 1
        n_x = 100
        s_x = 10
        s_t = 10
        hetero_strength = .5
        hetero_inds = np.arange(n_x - n_treatments, n_x)

        def lasso_model():
            return LassoCV(cv=3)

        # No heterogeneity
        dgp = DynamicPanelDGP(n_periods, n_treatments, n_x).create_instance(
            s_x, random_seed=12345)
        Y, T, X, W, groups = dgp.observational_data(n_units, s_t=s_t, random_seed=12345)
        est = DynamicDML(model_y=lasso_model(), model_t=lasso_model(), cv=3)
        # Define indices to test
        groups_filter = _get_groups_period_filter(groups, 3)
        shuffled_idx = np.array([groups_filter[i] for i in range(n_periods)]).flatten()
        test_indices = [np.arange(n_units * n_periods), shuffled_idx]
        for test_idx in test_indices:
            est.fit(Y[test_idx], T[test_idx], X=X[test_idx] if X is not None else None, W=W[test_idx],
                    groups=groups[test_idx], inference="auto")
            np.testing.assert_allclose(est.intercept_, dgp.true_effect.flatten(), atol=0.2)
            np.testing.assert_array_less(est.intercept__interval()[0], dgp.true_effect.flatten())
            np.testing.assert_array_less(dgp.true_effect.flatten(), est.intercept__interval()[1])

        # Heterogeneous effects
        dgp = DynamicPanelDGP(n_periods, n_treatments, n_x).create_instance(
            s_x, hetero_strength=hetero_strength, hetero_inds=hetero_inds, random_seed=12)
        Y, T, X, W, groups = dgp.observational_data(n_units, s_t=s_t, random_seed=1)
        hetero_strength = .5
        hetero_inds = np.arange(n_x - n_treatments, n_x)
        for test_idx in test_indices:
            est.fit(Y[test_idx], T[test_idx], X=X[test_idx], W=W[test_idx], groups=groups[test_idx], inference="auto")
            np.testing.assert_allclose(est.intercept_, dgp.true_effect.flatten(), atol=0.2)
            np.testing.assert_allclose(est.coef_, dgp.true_hetero_effect[:, hetero_inds + 1], atol=0.2)
            np.testing.assert_array_less(est.intercept__interval()[0], dgp.true_effect.flatten())
            np.testing.assert_array_less(dgp.true_effect.flatten(), est.intercept__interval()[1])
            np.testing.assert_array_less(est.coef__interval()[0] - .05, dgp.true_hetero_effect[:, hetero_inds + 1])
            np.testing.assert_array_less(dgp.true_hetero_effect[:, hetero_inds + 1] - .05, est.coef__interval()[1])
