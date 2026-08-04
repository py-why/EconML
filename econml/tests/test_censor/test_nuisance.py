"""Tests for nuisance survival model fitting utility.

Tests verify:
  - fit_nuisance_survival runs without error with CoxPH models
  - Returns correct matrix shapes (n, ns)
  - Survival matrices are in valid range [1e-3, 1]
  - Only requested models are fitted (others are None)
  - Works for both survival (single-failure) and competing risks data
  - End-to-end: nuisance → CUT transform pipeline works
  - Public nuisance helpers are OOS-only
"""

import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis

from econml.censor._nuisance import (
    fit_nuisance_survival,
    fit_nuisance_survival_crossfit,
    fit_nuisance_competing_crossfit,
    CrossFitNuisanceResult,
    _make_sksurv_y,
)
from .dgp import make_survival_data, make_competing_data


class TestMakeSksuvY(unittest.TestCase):

    def test_dtype(self):
        y = _make_sksurv_y(np.array([1.0, 2.0]), np.array([True, False]))
        self.assertEqual(y.dtype.names, ('event', 'time'))
        self.assertTrue(y['event'][0])
        self.assertFalse(y['event'][1])

    def test_values(self):
        y = _make_sksurv_y(np.array([3.5, 7.2]), np.array([False, True]))
        self.assertAlmostEqual(y['time'][0], 3.5)
        self.assertAlmostEqual(y['time'][1], 7.2)


class TestFitNuisanceSurvival(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_survival_data(n=300, tau=4.0, seed=42,
                                  compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.time = data['time']
        cls.event = data['event'].astype(int)
        cls.n = len(cls.T)

    def test_censoring_only(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=None,
            model_cause=None,
            model_competing=None,
            propensity_model=None)

        self.assertIsNotNone(result.G_a0)
        self.assertIsNotNone(result.G_a1)
        self.assertIsNone(result.S_a0)
        self.assertIsNone(result.Sj_a0)

        ns = len(result.time_grid)
        self.assertEqual(result.G_a0.shape, (self.n, ns))
        self.assertEqual(result.G_a1.shape, (self.n, ns))

    def test_event_only(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_censoring=None,
            model_event=CoxPHSurvivalAnalysis(),
            model_cause=None,
            model_competing=None,
            propensity_model=None)

        self.assertIsNotNone(result.S_a0)
        self.assertIsNotNone(result.S_a1)
        self.assertIsNone(result.G_a0)

    def test_both_censoring_and_event(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis())

        self.assertIsNotNone(result.G_a0)
        self.assertIsNotNone(result.S_a0)

    def test_survival_values_in_range(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis())

        for mat in [result.G_a0, result.G_a1, result.S_a0, result.S_a1]:
            self.assertTrue(np.all(mat >= 1e-3))
            self.assertTrue(np.all(mat <= 1.0))
            self.assertTrue(np.all(np.isfinite(mat)))

    def test_custom_time_grid(self):
        grid = np.linspace(0.5, 8.0, 50)
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_event=CoxPHSurvivalAnalysis(),
            time_grid=grid)

        np.testing.assert_array_equal(result.time_grid, grid)
        self.assertEqual(result.S_a0.shape[1], 50)

    def test_x_pred(self):
        with self.assertRaisesRegex(ValueError, "OOS-only"):
            fit_nuisance_survival(
                self.time, self.event, self.T, self.X,
                model_event=CoxPHSurvivalAnalysis(),
                X_pred=self.X[:10])

    def test_matches_crossfit_wrapper(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            cv=2,
            random_state=123)
        result_cf = fit_nuisance_survival_crossfit(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            cv=2,
            random_state=123)

        np.testing.assert_allclose(result.G_a0, result_cf.G_a0)
        np.testing.assert_allclose(result.G_a1, result_cf.G_a1)
        np.testing.assert_allclose(result.S_a0, result_cf.S_a0)
        np.testing.assert_allclose(result.S_a1, result_cf.S_a1)
        np.testing.assert_array_equal(result.time_grid, result_cf.time_grid)

    def test_wrapper_propagates_propensity_outputs(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            propensity_model=LogisticRegression(max_iter=1000),
            cv=2,
            random_state=123)

        self.assertIsInstance(result, CrossFitNuisanceResult)
        self.assertEqual(result.ps.shape, (self.n,))
        self.assertEqual(result.iptw.shape, (self.n,))
        self.assertEqual(result.naive.shape, (self.n,))
        self.assertTrue(np.all(np.isfinite(result.ps)))
        self.assertTrue(np.all(np.isfinite(result.iptw)))

    def test_default_auto_models_fit(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_cause=None,
            model_competing=None,
            cv=2,
            random_state=123)

        self.assertIsNotNone(result.G_a0)
        self.assertIsNotNone(result.S_a0)
        self.assertIsNotNone(result.ps)
        self.assertIsNone(result.Sj_a0)
        self.assertIsNone(result.Sjbar_a0)


class TestFitNuisanceCompetingRisks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_competing_data(n=300, tau=4.0, seed=42,
                                   compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.time = data['time']
        cls.event = data['event']
        cls.n = len(cls.T)

    def test_all_four_models(self):
        cox = CoxPHSurvivalAnalysis()
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_censoring=cox,
            model_event=cox,
            model_cause=cox,
            model_competing=cox,
            cause=1)

        ns = len(result.time_grid)
        for attr in ['G_a0', 'G_a1', 'S_a0', 'S_a1',
                      'Sj_a0', 'Sj_a1', 'Sjbar_a0', 'Sjbar_a1']:
            mat = getattr(result, attr)
            self.assertIsNotNone(mat, f"{attr} should not be None")
            self.assertEqual(mat.shape, (self.n, ns), f"{attr} shape mismatch")
            self.assertTrue(np.all(np.isfinite(mat)), f"{attr} has non-finite values")

    def test_cause_only(self):
        result = fit_nuisance_survival(
            self.time, self.event, self.T, self.X,
            model_cause=CoxPHSurvivalAnalysis(),
            model_censoring=None,
            model_event=None,
            model_competing=None,
            propensity_model=None,
            cause=1)

        self.assertIsNotNone(result.Sj_a0)
        self.assertIsNotNone(result.Sj_a1)
        self.assertIsNone(result.G_a0)
        self.assertIsNone(result.S_a0)
        self.assertIsNone(result.Sjbar_a0)

    def test_competing_defaults_fit(self):
        result = fit_nuisance_competing_crossfit(
            self.time, self.event, self.T, self.X,
            cv=2,
            random_state=123)

        self.assertIsNotNone(result.G_a0)
        self.assertIsNotNone(result.S_a0)
        self.assertIsNotNone(result.Sj_a0)
        self.assertIsNotNone(result.Sjbar_a0)
        self.assertIsNotNone(result.ps)


class TestEndToEnd(unittest.TestCase):
    """Test the full pipeline: nuisance → CUT transform."""

    def test_nuisance_to_ipcw(self):
        from econml.censor import ipcw_cut_rmst

        data = make_survival_data(n=200, tau=4.0, seed=99,
                                  compute_true_cate=False)
        result = fit_nuisance_survival(
            data['time'], data['event'].astype(int), data['T'], data['X'],
            model_censoring=CoxPHSurvivalAnalysis())

        pseudo_y = ipcw_cut_rmst(
            data['T'], data['time'], data['event'], 4.0,
            result.G_a0, result.G_a1,
            time_grid=result.time_grid)

        self.assertEqual(pseudo_y.shape, (200,))
        self.assertTrue(np.all(np.isfinite(pseudo_y)))

    def test_nuisance_to_aipcw(self):
        from econml.censor import aipcw_cut_rmst

        data = make_survival_data(n=200, tau=4.0, seed=99,
                                  compute_true_cate=False)
        result = fit_nuisance_survival(
            data['time'], data['event'].astype(int), data['T'], data['X'],
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis())

        pseudo_y = aipcw_cut_rmst(
            data['T'], data['time'], data['event'], 4.0,
            result.G_a0, result.G_a1,
            result.S_a0, result.S_a1,
            time_grid=result.time_grid)

        self.assertEqual(pseudo_y.shape, (200,))
        self.assertTrue(np.all(np.isfinite(pseudo_y)))


class TestCrossFitNuisanceSurvival(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_survival_data(n=240, tau=4.0, seed=314,
                                  compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.time = data['time']
        cls.event = data['event'].astype(int)
        cls.tau = 4.0

    def test_crossfit_shapes_and_weights(self):
        result = fit_nuisance_survival_crossfit(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            propensity_model=LogisticRegression(max_iter=1000),
            cv=3,
            random_state=123)

        self.assertIsInstance(result, CrossFitNuisanceResult)
        ns = len(result.time_grid)
        self.assertEqual(result.G_a0.shape, (len(self.T), ns))
        self.assertEqual(result.G_a1.shape, (len(self.T), ns))
        self.assertEqual(result.S_a0.shape, (len(self.T), ns))
        self.assertEqual(result.S_a1.shape, (len(self.T), ns))
        self.assertEqual(result.ps.shape, (len(self.T),))
        self.assertTrue(np.all(np.isfinite(result.ps)))
        self.assertTrue(np.all(result.ps >= 1e-3))
        self.assertTrue(np.all(result.ps <= 1 - 1e-3))
        self.assertTrue(np.all(np.isfinite(result.iptw)))
        self.assertTrue(np.all(np.isfinite(result.ow)))
        self.assertTrue(np.all(result.naive == 1.0))

    def test_crossfit_pipeline_to_aipcw_and_uif(self):
        from econml.censor import aipcw_cut_rmst, uif_diff_rmst

        result = fit_nuisance_survival_crossfit(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            propensity_model=LogisticRegression(max_iter=1000),
            cv=3,
            random_state=123)

        y_aipcw = aipcw_cut_rmst(
            self.T, self.time, self.event, self.tau,
            result.G_a0, result.G_a1, result.S_a0, result.S_a1,
            time_grid=result.time_grid)
        y_uif = uif_diff_rmst(
            self.T, self.time, self.event, self.tau,
            bw=result.iptw, tilt=result.naive,
            G_a0=result.G_a0, G_a1=result.G_a1,
            S_a0=result.S_a0, S_a1=result.S_a1,
            time_grid=result.time_grid)

        self.assertEqual(y_aipcw.shape, (len(self.T),))
        self.assertEqual(y_uif.shape, (len(self.T),))
        self.assertTrue(np.all(np.isfinite(y_aipcw)))
        self.assertTrue(np.all(np.isfinite(y_uif)))


class TestCrossFitNuisanceCompetingRisks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_competing_data(n=240, tau=4.0, seed=2718,
                                   compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.time = data['time']
        cls.event = data['event'].astype(int)
        cls.tau = 4.0

    def test_crossfit_all_matrices_and_propensity(self):
        result = fit_nuisance_competing_crossfit(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            model_cause=CoxPHSurvivalAnalysis(),
            model_competing=CoxPHSurvivalAnalysis(),
            propensity_model=LogisticRegression(max_iter=1000),
            cause=1,
            cv=3,
            random_state=123)

        ns = len(result.time_grid)
        for attr in ['G_a0', 'G_a1', 'S_a0', 'S_a1',
                     'Sj_a0', 'Sj_a1', 'Sjbar_a0', 'Sjbar_a1']:
            mat = getattr(result, attr)
            self.assertEqual(mat.shape, (len(self.T), ns))
            self.assertTrue(np.all(np.isfinite(mat)))
        self.assertTrue(np.all(np.isfinite(result.ps)))
        self.assertTrue(np.all(np.isfinite(result.iptw)))

    def test_crossfit_pipeline_to_competing_aipcw_and_uif(self):
        from econml.censor import aipcw_cut_rmtlj, uif_diff_rmtlj

        result = fit_nuisance_competing_crossfit(
            self.time, self.event, self.T, self.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            model_cause=CoxPHSurvivalAnalysis(),
            propensity_model=LogisticRegression(max_iter=1000),
            cause=1,
            cv=3,
            random_state=123)

        y_aipcw = aipcw_cut_rmtlj(
            self.T, self.time, self.event, self.tau,
            result.G_a0, result.G_a1, result.S_a0, result.S_a1,
            result.Sj_a0, result.Sj_a1,
            cause=1,
            time_grid=result.time_grid)
        y_uif = uif_diff_rmtlj(
            self.T, self.time, self.event, self.tau,
            bw=result.iptw, tilt=result.naive,
            G_a0=result.G_a0, G_a1=result.G_a1,
            S_a0=result.S_a0, S_a1=result.S_a1,
            Sj_a0=result.Sj_a0, Sj_a1=result.Sj_a1,
            cause=1,
            time_grid=result.time_grid)

        self.assertEqual(y_aipcw.shape, (len(self.T),))
        self.assertEqual(y_uif.shape, (len(self.T),))
        self.assertTrue(np.all(np.isfinite(y_aipcw)))
        self.assertTrue(np.all(np.isfinite(y_uif)))


if __name__ == '__main__':
    unittest.main()
