"""Smoke tests for CausalSurvivalForest.

DGP mirrors original/simulation/survival/survival_hte_simulation_case1.R
(log-logistic / log-normal event times, treatment-dependent censoring).
"""

import unittest
import warnings
import numpy as np

from econml.grf import CausalSurvivalForest, SurvivalForest, survival_forest
from econml.grf._causal_survival_forest import (
    _CausalSurvivalForestTrainer,
    _ClusteredTreatmentForest,
)
from .dgp import make_survival_data


class _FailingSurvivalModel:
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    def fit(self, X, y):
        raise ValueError("intentional fold failure")


class TestCausalSurvivalForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_survival_data(n=300, tau=2.0, seed=42,
                                  compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.time = data['time']
        cls.event = data['event']

    def _make_est(self, **kwargs):
        return CausalSurvivalForest(
            horizon=2.0,
            n_estimators=52,  # must be divisible by subforest_size=4 → 52 = 4*13
            random_state=0,
            **kwargs,
        )

    def test_fit_no_error(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)

    def test_predict_shape(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        pred = est.predict(self.X)
        self.assertEqual(pred.shape[0], self.X.shape[0])

    def test_predict_finite(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        pred = est.predict(self.X)
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_tau_alias_matches_horizon(self):
        est_tau = CausalSurvivalForest(
            tau=2.0,
            n_estimators=52,
            random_state=0,
        )
        est_tau.fit(self.X, self.T, self.time, self.event)

        est_horizon = CausalSurvivalForest(
            horizon=2.0,
            n_estimators=52,
            random_state=0,
        )
        est_horizon.fit(self.X, self.T, self.time, self.event)

        np.testing.assert_allclose(est_tau.effect(self.X), est_horizon.effect(self.X))

    def test_nuisance_cv_two(self):
        est = CausalSurvivalForest(
            horizon=2.0,
            nuisance_cv=2,
            n_estimators=52,
            random_state=0,
        )
        est.fit(self.X, self.T, self.time, self.event)
        self.assertEqual(est.effect(self.X).shape[0], self.X.shape[0])

    def test_effect_uses_oob_on_training_data(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        effect = est.effect(self.X)
        oob = np.asarray(est.oob_predict(self.X)).ravel()
        np.testing.assert_allclose(effect, oob)

    def test_nuisance_stored(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        self.assertTrue(hasattr(est, 'W_hat_'))
        self.assertTrue(hasattr(est, 'Y_hat_'))
        self.assertTrue(hasattr(est, 'S_hat_'))
        self.assertTrue(hasattr(est, 'C_hat_'))
        self.assertTrue(hasattr(est, 'S1_hat_'))
        self.assertTrue(hasattr(est, 'S0_hat_'))
        self.assertEqual(est.W_hat_.shape, (self.X.shape[0],))
        self.assertEqual(est.Y_hat_.shape, (self.X.shape[0],))
        self.assertEqual(est.S_hat_.shape[0], self.X.shape[0])
        self.assertEqual(est.C_hat_.shape, est.S_hat_.shape)
        self.assertEqual(est.S1_hat_.shape, est.S_hat_.shape)
        self.assertEqual(est.S0_hat_.shape, est.S_hat_.shape)
        self.assertTrue(np.all(np.isfinite(est.S_hat_)))
        self.assertTrue(np.all(np.isfinite(est.C_hat_)))

    def test_propensity_in_unit_interval(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        self.assertTrue(np.all(est.W_hat_ >= 0))
        self.assertTrue(np.all(est.W_hat_ <= 1))

    def test_default_propensity_uses_oob_predictions(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        in_sample = np.asarray(est.model_t_nuisance_.predict(self.X)).ravel()
        in_sample = np.clip(in_sample, 1e-3, 1 - 1e-3)
        self.assertFalse(np.allclose(est.W_hat_, in_sample))

    def test_survival_probability_target(self):
        est = CausalSurvivalForest(
            horizon=2.0,
            target="survival.probability",
            n_estimators=52,
            random_state=0,
        )
        est.fit(self.X, self.T, self.time, self.event)
        pred = est.predict(self.X)
        self.assertEqual(pred.shape[0], self.X.shape[0])
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_survival_nuisance_failure_falls_back(self):
        est = CausalSurvivalForest(
            horizon=2.0,
            model_event=_FailingSurvivalModel(),
            model_cens=_FailingSurvivalModel(),
            n_estimators=52,
            random_state=0,
        )
        est.fit(self.X, self.T, self.time, self.event)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))
        self.assertTrue(np.allclose(est.S_hat_, 1.0))
        self.assertTrue(np.allclose(est.C_hat_, 1.0))

    def test_psi_stored(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        self.assertTrue(hasattr(est, 'psi_numerator_'))
        self.assertTrue(hasattr(est, 'psi_denominator_'))
        self.assertTrue(np.all(np.isfinite(est.psi_numerator_)))
        self.assertTrue(np.all(est.psi_denominator_ >= 0))

    def test_predict_and_var_shape(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        point, var = est.predict_and_var(self.X[:17])
        self.assertEqual(point.shape, (17, 1))
        self.assertEqual(var.shape, (17, 1, 1))
        self.assertTrue(np.all(np.isfinite(point)))
        self.assertTrue(np.all(np.isfinite(var)))

    def test_dedicated_final_trainer_used(self):
        est = self._make_est()
        est.fit(self.X, self.T, self.time, self.event)
        self.assertIsInstance(est.final_trainer_, _CausalSurvivalForestTrainer)

    def test_invalid_horizon_raises(self):
        data = make_survival_data(n=100, tau=2.0, seed=0, compute_true_cate=False)
        X, T, time, event = data['X'], data['T'], data['time'], data['event']
        # horizon before first event → should raise
        with self.assertRaises(ValueError):
            CausalSurvivalForest(horizon=-1.0).fit(X, T, time, event)

    def test_conflicting_tau_and_horizon_raise(self):
        with self.assertRaises(ValueError):
            CausalSurvivalForest(horizon=2.0, tau=3.0)

    def test_all_censored_raises(self):
        X = self.X.copy()
        T = self.T.copy()
        time = self.time.copy()
        event = np.zeros_like(self.event)
        with self.assertRaises(ValueError):
            self._make_est().fit(X, T, time, event)

    def test_binary_treatment_only(self):
        T = self.T.astype(float) + 0.1
        with self.assertRaises(NotImplementedError):
            self._make_est().fit(self.X, T, self.time, self.event)

    def test_failure_times_arg(self):
        grid = np.linspace(self.time.min(), self.time.max(), 12)
        est = self._make_est(failure_times=grid)
        est.fit(self.X, self.T, self.time, self.event)
        self.assertEqual(est.S_hat_.shape[1], len(grid))

    def test_cluster_arg_supported(self):
        clusters = np.arange(self.X.shape[0]) % 11
        est = self._make_est(clusters=clusters)
        est.fit(self.X, self.T, self.time, self.event)
        self.assertEqual(est.effect(self.X).shape[0], self.X.shape[0])

    def test_equalize_cluster_weights_supported(self):
        clusters = np.arange(self.X.shape[0]) % 13
        est = self._make_est(clusters=clusters, equalize_cluster_weights=True)
        est.fit(self.X, self.T, self.time, self.event)
        self.assertEqual(est.effect(self.X).shape[0], self.X.shape[0])

    def test_clustered_default_treatment_nuisance_used(self):
        clusters = np.arange(self.X.shape[0]) % 11
        est = self._make_est(clusters=clusters)
        est.fit(self.X, self.T, self.time, self.event)
        self.assertIsInstance(est.model_t_nuisance_, _ClusteredTreatmentForest)

    def test_equalized_clusters_do_not_reweight_final_trainer_samples(self):
        base_clusters = np.repeat(np.arange(15), np.array([5, 9, 11, 7, 13, 6, 10, 8, 12, 14, 5, 9, 7, 11, 13]))
        clusters = np.resize(base_clusters, self.X.shape[0])
        est = self._make_est(clusters=clusters, equalize_cluster_weights=True)
        est.fit(self.X, self.T, self.time, self.event)
        self.assertIsNone(est.effective_sample_weight_)
        np.testing.assert_allclose(est.final_trainer_.sample_weight_, est.psi_denominator_)

    def test_honesty_fraction_supported(self):
        est = self._make_est(honesty_fraction=0.3)
        est.fit(self.X, self.T, self.time, self.event)
        self.assertAlmostEqual(est.final_trainer_.honesty_fraction, 0.3)
        self.assertEqual(est.effect(self.X).shape[0], self.X.shape[0])

    def test_honesty_prune_leaves_false_supported(self):
        est = self._make_est(honesty_prune_leaves=False)
        est.fit(self.X, self.T, self.time, self.event)
        self.assertFalse(est.final_trainer_.honesty_prune_leaves)
        self.assertEqual(est.effect(self.X).shape[0], self.X.shape[0])

    def test_grf_control_args_affect_fit(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = self._make_est(
                imbalance_penalty=0.5,
                stabilize_splits=False,
                fast_logrank=True,
                tune_parameters="all",
                tune_num_draws=4,
                tune_num_reps=2,
                tune_num_trees=16,
            )
            est.fit(self.X, self.T, self.time, self.event)
        self.assertEqual(len(caught), 0)
        self.assertIsNotNone(est.tuned_model_t_params_)
        self.assertIn("sample.fraction", est.tuned_model_t_params_)
        self.assertIsNotNone(est.tuning_output_)
        self.assertEqual(est.tuning_output_["metric"], "oob_mse")
        self.assertEqual(est.tuning_output_["num.reps"], 2)
        self.assertEqual(est.tuning_output_["num.trees"], 16)
        self.assertGreaterEqual(len(est.tuning_output_["results"]), 1)
        self.assertEqual(est.final_trainer_.imbalance_penalty, 0.5)
        self.assertFalse(est.final_trainer_.stabilize_splits)

    def test_heavy_censoring_sparse_events_remains_finite(self):
        rng = np.random.RandomState(123)
        time = np.minimum(self.time, 0.35 + 0.05 * rng.rand(self.time.shape[0]))
        event = ((self.event == 1) & (self.time <= 0.2)).astype(int)
        event[:5] = 1
        est = self._make_est(tune_parameters="all", tune_num_draws=3, tune_num_trees=12)
        est.fit(self.X, self.T, time, event)
        pred = est.effect(self.X[:25])
        point, var = est.predict_and_var(self.X[:25])
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertTrue(np.all(np.isfinite(point)))
        self.assertTrue(np.all(np.isfinite(var)))


class TestSurvivalForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_survival_data(n=200, tau=2.0, seed=7, compute_true_cate=False)
        cls.X = data["X"]
        cls.time = data["time"]
        cls.event = data["event"]

    def test_fit_and_oob_predict(self):
        est = SurvivalForest(num_trees=40, seed=0)
        est.fit(self.X, self.time, self.event)
        oob = est.oob_predict(self.X)
        self.assertEqual(oob.shape[0], self.X.shape[0])
        self.assertTrue(np.all(np.isfinite(oob)))

    def test_predict_returns_result(self):
        est = SurvivalForest(num_trees=20, seed=0)
        est.fit(self.X, self.time, self.event)
        pred = est.predict(self.X[:10])
        self.assertEqual(pred.predictions.shape[0], 10)
        self.assertTrue(np.all(np.isfinite(pred.predictions)))

    def test_survival_forest_function(self):
        est = survival_forest(self.X, self.time, self.event, num_trees=20, seed=0)
        self.assertIsInstance(est, SurvivalForest)
        self.assertEqual(est.oob_predict(self.X).shape[0], self.X.shape[0])

    def test_kaplan_meier_prediction_type(self):
        est = SurvivalForest(num_trees=20, seed=0, prediction_type="Kaplan-Meier")
        est.fit(self.X, self.time, self.event)
        pred = est.predict(self.X[:5])
        self.assertEqual(pred.predictions.shape[0], 5)
        self.assertTrue(np.all(np.isfinite(pred.predictions)))

    def test_cluster_weighting_supported(self):
        clusters = np.arange(self.X.shape[0]) % 9
        est = SurvivalForest(num_trees=20, seed=0, clusters=clusters, equalize_cluster_weights=True)
        est.fit(self.X, self.time, self.event)
        self.assertEqual(est.oob_predict(self.X).shape[0], self.X.shape[0])

    def test_honesty_fraction_supported(self):
        est = SurvivalForest(num_trees=20, seed=0, honesty=True, honesty_fraction=0.3)
        est.fit(self.X, self.time, self.event)
        self.assertEqual(est.predict(self.X[:5]).predictions.shape[0], 5)

    def test_honesty_prune_leaves_false_supported(self):
        est = SurvivalForest(num_trees=20, seed=0, honesty=True, honesty_prune_leaves=False)
        est.fit(self.X, self.time, self.event)
        self.assertEqual(est.oob_predict(self.X).shape[0], self.X.shape[0])

    def test_fast_logrank_supported(self):
        est = SurvivalForest(num_trees=20, seed=0, fast_logrank=True)
        est.fit(self.X, self.time, self.event)
        self.assertEqual(est.predict(self.X[:5]).predictions.shape[0], 5)


if __name__ == '__main__':
    unittest.main()
