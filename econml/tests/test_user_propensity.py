# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for user-supplied propensities in DML and DRLearner estimators."""

import unittest

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from econml.dml import CausalForestDML, LinearDML, NonParamDML
from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner
from econml.inference import BootstrapInference
from econml.iv.dml import OrthoIV


class _FailOnFitClassifier(BaseEstimator, ClassifierMixin):
    """A classifier whose fit always raises, to prove that the model is bypassed."""

    def fit(self, X, y, **kwargs):
        raise AssertionError("The first stage treatment/propensity model should not be fitted "
                             "when user-supplied propensities are provided!")

    def predict_proba(self, X):
        raise AssertionError("The first stage treatment/propensity model should not be used "
                             "when user-supplied propensities are provided!")


def _binary_dgp(n=1000, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    e = 0.2 + 0.6 / (1 + np.exp(-X[:, 0]))  # known propensity, varies with X
    T = rng.binomial(1, e)
    tau = 1 + X[:, 0]
    Y = tau * T + X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n)
    return Y, T, X, e, tau


def _multivalued_dgp(n=2000, seed=456):
    # per-unit-varying assignment probabilities, so that a row-misaligned or
    # column-misaligned implementation could not pass the tests below
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = np.column_stack([np.zeros(n), 0.5 * X[:, 0], -0.5 * X[:, 0]])
    propensity = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    u = rng.random(n)
    T = (u[:, None] > np.cumsum(propensity, axis=1)).sum(axis=1)
    Y = (T == 1) * 1.0 + (T == 2) * 2.0 + X[:, 0] + rng.normal(size=n)
    return Y, T, X, propensity


def _block_rct_dgp(n=8000, seed=789):
    # block-randomized experiment where the block is NOT observable from X,
    # so the assignment probabilities cannot be recovered by a propensity model
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    block = rng.choice(5, size=n)
    e = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[block]
    T = rng.binomial(1, e)
    tau = 1 + X[:, 0]
    Y = tau * T + 4 * block + X[:, 0] + rng.normal(size=n)
    return Y, T, X, e, tau


class TestUserPropensity(unittest.TestCase):

    def test_dml_binary_accuracy_and_residuals(self):
        """With known propensities, DML should recover the ATE and use T - e as treatment residuals."""
        Y, T, X, e, tau = _binary_dgp()
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=1, random_state=0)
        # cv=1 keeps samples in input order, so we can check residuals elementwise;
        # _FailOnFitClassifier proves the treatment model is completely bypassed
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e, cache_values=True)
        np.testing.assert_allclose(est.ate(X[:, :1]), np.mean(tau), atol=0.15)
        _, T_res, _, _ = est.residuals_
        np.testing.assert_allclose(T_res.flatten(), T - e)

    def test_dml_binary_crossfit(self):
        """Cross-fitting (cv>1) should also work, with residuals matching T - e up to reordering."""
        Y, T, X, e, tau = _binary_dgp()
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=3, random_state=0)
        est.fit(Y, T, X=None, W=X, propensity=e, cache_values=True)
        _, T_res, _, _ = est.residuals_
        np.testing.assert_allclose(np.sort(T_res.flatten()), np.sort(T - e))

    def test_dml_single_and_two_column_propensity_equivalent(self):
        """A (n,) propensity vector and the equivalent (n, 2) matrix should give identical results."""
        Y, T, X, e, _ = _binary_dgp()
        results = []
        for prop in [e, np.column_stack([1 - e, e])]:
            est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                            discrete_treatment=True, cv=2, random_state=0)
            est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=prop)
            results.append(est.ate(X[:, :1]))
        np.testing.assert_allclose(results[0], results[1])

    def test_dr_binary_accuracy_and_bypass(self):
        """With known propensities, DRLearner should recover the ATE without fitting model_propensity."""
        Y, T, X, e, tau = _binary_dgp()
        est = LinearDRLearner(model_regression=RandomForestRegressor(n_estimators=50, random_state=0),
                              model_propensity=_FailOnFitClassifier(), cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        np.testing.assert_allclose(est.ate(X[:, :1]), np.mean(tau), atol=0.2)
        # no propensity model was fitted, so its nuisance scores should be None
        for scores in est.nuisance_scores_propensity:
            for score in scores:
                self.assertIsNone(score)

    def test_dr_multivalued_treatment(self):
        """User-supplied propensities should work with more than two treatment categories."""
        Y, T, X, propensity = _multivalued_dgp()
        est = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=_FailOnFitClassifier(), cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=propensity)
        np.testing.assert_allclose(est.ate(X[:, :1], T0=0, T1=1), 1.0, atol=0.2)
        np.testing.assert_allclose(est.ate(X[:, :1], T0=0, T1=2), 2.0, atol=0.2)

    def test_dml_multivalued_treatment_residuals(self):
        """Multi-valued DML treatment residuals should equal the one-hot encoding minus the propensities."""
        Y, T, X, propensity = _multivalued_dgp()
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=1, random_state=0)
        est.fit(Y, T, X=None, W=X, propensity=propensity, cache_values=True)
        _, T_res, _, _ = est.residuals_
        T_onehot = np.column_stack([(T == 1), (T == 2)]).astype(float)
        np.testing.assert_allclose(T_res, T_onehot - propensity[:, 1:])

    def test_hidden_block_rct(self):
        """Known propensities recover the ATE in a block-randomized RCT where the blocks are not observed.

        This is the motivating use case for the feature: the assignment probabilities are known by
        design but cannot be estimated from the observed covariates, so a fitted propensity model
        produces badly confounded estimates while the supplied probabilities give consistent ones
        with valid confidence intervals.
        """
        Y, T, X, e, tau = _block_rct_dgp()
        true_ate = np.mean(tau)
        for cls, kwargs in [(LinearDML, {'model_y': LinearRegression(), 'model_t': LogisticRegression(),
                                         'discrete_treatment': True}),
                            (LinearDRLearner, {'model_regression': LinearRegression(),
                                               'model_propensity': LogisticRegression()})]:
            with self.subTest(estimator=cls.__name__):
                known = cls(cv=2, random_state=0, **kwargs)
                known.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
                lb, ub = known.ate_interval(X[:, :1], alpha=0.05)
                self.assertTrue(lb <= true_ate <= ub,
                                f"CI ({lb}, {ub}) does not cover the true ATE {true_ate}")
                naive = cls(cv=2, random_state=0, **kwargs)
                naive.fit(Y, T, X=X[:, :1], W=X[:, 1:])
                known_bias = abs(known.ate(X[:, :1]) - true_ate)
                naive_bias = abs(naive.ate(X[:, :1]) - true_ate)
                self.assertLess(known_bias, 0.3)
                self.assertGreater(naive_bias, 5 * known_bias)

    def test_categories_ordering(self):
        """Propensity columns must follow the `categories` initializer argument when it is set."""
        rng = np.random.default_rng(0)
        n = 4000
        X = rng.normal(size=(n, 2))
        e_a = 0.1 + 0.8 / (1 + np.exp(-2 * X[:, 0]))  # probability of treatment 'a'
        T = np.where(rng.random(n) < e_a, 'a', 'b')
        Y = 2.0 * (T == 'a') + X[:, 0] + rng.normal(size=n)
        # with categories=['b', 'a'], 'b' is the control, so the first propensity
        # column is P(T='b') and the effect of 'a' relative to 'b' is 2
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, categories=['b', 'a'], cv=2, random_state=0)
        est.fit(Y, T, X=None, W=X, propensity=np.column_stack([1 - e_a, e_a]))
        np.testing.assert_allclose(est.ate(T0='b', T1='a'), 2.0, atol=0.2)
        # negative control: swapping the columns must give a substantially wrong answer,
        # proving the column ordering is actually consumed
        swapped = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                            discrete_treatment=True, categories=['b', 'a'], cv=2, random_state=0)
        swapped.fit(Y, T, X=None, W=X, propensity=np.column_stack([e_a, 1 - e_a]))
        self.assertGreater(abs(swapped.ate(T0='b', T1='a') - 2.0), 0.5)

    def test_no_X_no_W(self):
        """DML with X=None and W=None (a pure experiment) should accept known propensities."""
        rng = np.random.default_rng(1)
        n = 2000
        block = rng.choice(2, size=n)
        e = np.array([0.2, 0.8])[block]
        T = rng.binomial(1, e)
        Y = 1.0 * T + 2 * block + rng.normal(size=n)
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=None, W=None, propensity=e)
        np.testing.assert_allclose(est.ate(), 1.0, atol=0.25)

    def test_refit_resets_state(self):
        """Re-fitting the same estimator instance with/without propensities should fully reset state."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        est = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                        discrete_treatment=True, cv=2, random_state=0)
        # fit with user propensities, then re-fit without: score must work without them again
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
        est.score(Y, T, X=X[:, :1], W=X[:, 1:])
        # and the reverse: re-fitting with propensities must re-impose the score requirement
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        with pytest.raises(ValueError, match="propensity"):
            est.score(Y, T, X=X[:, :1], W=X[:, 1:])

    def test_other_estimators_smoke(self):
        """CausalForestDML, NonParamDML, DRLearner and ForestDRLearner should all accept propensities."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        for est in [CausalForestDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                                    discrete_treatment=True, n_estimators=100, cv=2, random_state=0),
                    NonParamDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                                model_final=RandomForestRegressor(n_estimators=20, random_state=0),
                                discrete_treatment=True, cv=2, random_state=0),
                    DRLearner(model_regression=LinearRegression(), model_propensity=_FailOnFitClassifier(),
                              model_final=LinearRegression(), cv=2, random_state=0),
                    ForestDRLearner(model_regression=LinearRegression(), model_propensity=_FailOnFitClassifier(),
                                    n_estimators=100, cv=2, random_state=0)]:
            with self.subTest(estimator=type(est).__name__):
                est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
                est.effect(X[:5, :1])

    def test_score_requires_propensity_when_fit_with_it(self):
        """If fit used user-supplied propensities, score must also receive them."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        with pytest.raises(ValueError, match="propensity"):
            est.score(Y, T, X=X[:, :1], W=X[:, 1:])
        self.assertIsInstance(est.score(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e), float)

        dr = LinearDRLearner(model_regression=LinearRegression(),
                             model_propensity=_FailOnFitClassifier(), cv=2, random_state=0)
        dr.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        with pytest.raises(ValueError, match="propensity"):
            dr.score(Y, T, X=X[:, :1], W=X[:, 1:])
        dr.score(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)

    def test_score_optional_when_fit_without_it(self):
        """An estimator fit normally can score with or without user-supplied propensities."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        est = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
        est.score(Y, T, X=X[:, :1], W=X[:, 1:])
        est.score(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)

    def test_validation_errors(self):
        Y, T, X, e, _ = _binary_dgp(n=500)
        Y3, T3, X3, propensity3 = _multivalued_dgp(n=500)

        # continuous treatment is not supported
        with pytest.raises(ValueError, match="discrete"):
            LinearDML(random_state=0).fit(Y, X[:, 0], X=X[:, :1], W=X[:, 1:], propensity=e)
        # single column propensity requires binary treatment
        with pytest.raises(ValueError, match="column"):
            LinearDRLearner(cv=2, random_state=0).fit(Y3, T3, X=X3[:, :1], W=X3[:, 1:], propensity=e)
        # column count must match the number of categories
        with pytest.raises(ValueError, match="column"):
            LinearDRLearner(cv=2, random_state=0).fit(
                Y3, T3, X=X3[:, :1], W=X3[:, 1:], propensity=propensity3[:, :2])
        # values must be valid probabilities
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            LinearDML(discrete_treatment=True, random_state=0).fit(
                Y, T, X=X[:, :1], W=X[:, 1:], propensity=2 * e)
        # rows must sum to 1
        with pytest.raises(ValueError, match="sum"):
            LinearDML(discrete_treatment=True, random_state=0).fit(
                Y, T, X=X[:, :1], W=X[:, 1:], propensity=np.column_stack([e, e]))
        # row count must match the data
        with pytest.raises(AssertionError):
            LinearDML(discrete_treatment=True, random_state=0).fit(
                Y, T, X=X[:, :1], W=X[:, 1:], propensity=e[:-1])
        # estimators that don't model propensities don't accept the argument
        Z = np.random.default_rng(0).binomial(1, 0.5, size=Y.shape[0])
        with pytest.raises(TypeError):
            OrthoIV(discrete_treatment=True, discrete_instrument=True).fit(
                Y, T, Z=Z, X=X[:, :1], W=X[:, 1:], propensity=e)

    def test_mc_iters_and_refit_final(self):
        """User-supplied propensities should compose with monte carlo iterations and refit_final."""
        Y, T, X, e, tau = _binary_dgp()
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=2, mc_iters=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e, cache_values=True)
        ate_before = est.ate(X[:, :1])
        est.refit_final()
        np.testing.assert_allclose(est.ate(X[:, :1]), ate_before)
        np.testing.assert_allclose(ate_before, np.mean(tau), atol=0.15)

    def test_bootstrap_inference(self):
        """Bootstrap inference resamples the supplied propensities along with the rest of the data."""
        Y, T, X, e, tau = _binary_dgp(n=500)
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e,
                inference=BootstrapInference(n_bootstrap_samples=5, n_jobs=1))
        lb, ub = est.ate_interval(X[:, :1])
        self.assertLess(lb, ub)

    def test_clipping_and_trimming(self):
        """Supplied propensities remain subject to min_propensity clipping and trimming_threshold trimming."""
        Y, T, X, e, _ = _binary_dgp(n=1000)
        # push some propensities into the trimmable/clippable region
        e_extreme = np.clip(e, 0.02, 0.98)
        e_extreme[:100] = 0.02
        est = DRLearner(model_regression=LinearRegression(), model_propensity=_FailOnFitClassifier(),
                        model_final=LinearRegression(), trimming_threshold=0.1, min_propensity=0.05,
                        cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e_extreme)
        self.assertGreater(est.n_samples_trimmed_, 0)
        # without trimming, the same input fits on all samples
        est2 = DRLearner(model_regression=LinearRegression(), model_propensity=_FailOnFitClassifier(),
                        model_final=LinearRegression(), cv=2, random_state=0)
        est2.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e_extreme)
        self.assertEqual(est2.n_samples_trimmed_, 0)

    def test_composition_with_groups_and_weights(self):
        """The propensity array must survive fold slicing alongside groups and frequency weights."""
        Y, T, X, e, _ = _binary_dgp(n=600)
        groups = np.repeat(np.arange(300), 2)
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e, groups=groups)
        est.effect(X[:5, :1])
        est2 = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                         discrete_treatment=True, cv=2, random_state=0)
        est2.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e, sample_weight=np.ones(600),
                 freq_weight=np.ones(600, dtype=int), sample_var=np.ones(600))
        est2.effect(X[:5, :1])

    def test_accessors_after_bypassed_fit(self):
        """Accessors for the bypassed model should raise an informative error, and score_nuisances works."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        with pytest.raises(AttributeError, match="user-supplied propensities"):
            est.models_t
        scores = est.score_nuisances(Y, T, X=X[:, :1], W=X[:, 1:])
        self.assertTrue(all(s is None for s in scores['T_default_score']))
        self.assertTrue(all(s is not None for s in scores['Y_default_score']))

        dr = LinearDRLearner(model_regression=LinearRegression(),
                             model_propensity=_FailOnFitClassifier(), cv=2, random_state=0)
        dr.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        with pytest.raises(AttributeError, match="user-supplied propensities"):
            dr.models_propensity

    def test_degenerate_propensity_warns(self):
        """Propensities of exactly 0 or 1 should produce a warning since they imply no overlap."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        e_degenerate = e.copy()
        e_degenerate[T == 1] = np.maximum(e_degenerate[T == 1], 0.5)
        e_degenerate[:5] = np.where(T[:5] == 1, 1.0, 0.0)
        est = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=_FailOnFitClassifier(), cv=2, random_state=0)
        with pytest.warns(UserWarning, match="exactly 0 or 1"):
            est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e_degenerate)

    def test_score_positional_compatibility(self):
        """Existing positional score() callers must be unaffected (propensity is the last parameter)."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        est = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:])
        # the historical positional order is (Y, T, X, W, sample_weight, scoring)
        s = est.score(Y, T, X[:, :1], X[:, 1:], None, 'mean_squared_error')
        self.assertIsInstance(s, float)

    def test_nuisance_scores_t_none(self):
        """When the treatment model is bypassed, its nuisance scores should be None."""
        Y, T, X, e, _ = _binary_dgp(n=500)
        est = LinearDML(model_y=LinearRegression(), model_t=_FailOnFitClassifier(),
                        discrete_treatment=True, cv=2, random_state=0)
        est.fit(Y, T, X=X[:, :1], W=X[:, 1:], propensity=e)
        for scores in est.nuisance_scores_t:
            for score in scores:
                self.assertIsNone(score)
        # the outcome model is still fitted and scored normally
        for scores in est.nuisance_scores_y:
            for score in scores:
                self.assertIsNotNone(score)


if __name__ == '__main__':
    unittest.main()
