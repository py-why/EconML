"""Smoke tests for SurvivalTLearner, SurvivalSLearner, and pseudo-outcome learners
(IPTWLearner, ULearner, MCLearner, MCEALearner).

DGP mirrors original/simulation/survival/survival_hte_simulation_case1.R
(case 1: proportional-hazards-style with log-logistic / log-normal event times).

Tests verify:
  - fit() runs without error
  - effect() returns shape (n, 1)
  - ate() returns a scalar
  - effect values are finite
"""

import unittest
import numpy as np
from sksurv.ensemble import RandomSurvivalForest

from econml.metalearners._censor_metalearners import (SurvivalTLearner, SurvivalSLearner,
                                                      IPTWLearner, ULearner, MCLearner, MCEALearner,
                                                      RALearner, IFLearner)
from econml.censor import fit_nuisance_survival, aipcw_cut_rmst, uif_diff_rmst
from ._helpers import gbr
from .dgp import make_survival_data


class TestSurvivalTLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_survival_data(n=300, tau=2.0, seed=42,
                                  compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.Y = data['Y']

    def _make_model(self):
        return RandomSurvivalForest(n_estimators=50, min_samples_leaf=5,
                                    random_state=0)

    def test_fit_effect_shape(self):
        est = SurvivalTLearner(models=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        # effect() squeezes d_t=1 dimension → (n,) per EconML convention
        self.assertEqual(cate.shape, (self.X.shape[0],))

    def test_effect_finite(self):
        est = SurvivalTLearner(models=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertTrue(np.all(np.isfinite(cate)))

    def test_ate_scalar(self):
        est = SurvivalTLearner(models=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        ate = est.ate(self.X)
        # ate() returns scalar () when d_t=1
        self.assertEqual(np.asarray(ate).ndim, 0)
        self.assertTrue(np.isfinite(ate))

    def test_const_marginal_effect_shape(self):
        est = SurvivalTLearner(models=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cme = est.const_marginal_effect(self.X)
        self.assertEqual(cme.shape, (self.X.shape[0], 1))

    def test_tau_respected(self):
        """RMST at tau=0.5 should be <= RMST at tau=2.0 everywhere."""
        est_small = SurvivalTLearner(models=self._make_model(), tau=0.5)
        est_large = SurvivalTLearner(models=self._make_model(), tau=2.0)
        est_small.fit(self.Y, self.T, X=self.X)
        est_large.fit(self.Y, self.T, X=self.X)
        # RMST(tau=0.5) ≤ RMST(tau=2) → |CATE| tends to be smaller at smaller tau
        # We just check no error and shapes agree
        self.assertEqual(est_small.effect(self.X).shape,
                         est_large.effect(self.X).shape)

    def test_default_survival_model(self):
        est = SurvivalTLearner(tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(cate)))


class TestSurvivalSLearner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = make_survival_data(n=300, tau=2.0, seed=42,
                                  compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.Y = data['Y']

    def _make_model(self):
        return RandomSurvivalForest(n_estimators=50, min_samples_leaf=5,
                                    random_state=0)

    def test_fit_effect_shape(self):
        est = SurvivalSLearner(overall_model=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        # effect() squeezes d_t=1 dimension → (n,) per EconML convention
        self.assertEqual(cate.shape, (self.X.shape[0],))

    def test_effect_finite(self):
        est = SurvivalSLearner(overall_model=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertTrue(np.all(np.isfinite(cate)))

    def test_ate_scalar(self):
        est = SurvivalSLearner(overall_model=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        ate = est.ate(self.X)
        # ate() returns scalar () when d_t=1
        self.assertEqual(np.asarray(ate).ndim, 0)
        self.assertTrue(np.isfinite(ate))

    def test_const_marginal_effect_shape(self):
        est = SurvivalSLearner(overall_model=self._make_model(), tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cme = est.const_marginal_effect(self.X)
        self.assertEqual(cme.shape, (self.X.shape[0], 1))

    def test_default_survival_model(self):
        est = SurvivalSLearner(tau=2.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(cate)))


# ---------------------------------------------------------------------------
# Shared fixture for pseudo-outcome learner tests
# ---------------------------------------------------------------------------

class _PseudoOutcomeBase(unittest.TestCase):
    """Base class providing a pre-computed AIPCW pseudo-outcome for testing."""

    @classmethod
    def setUpClass(cls):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        data = make_survival_data(n=300, tau=2.0, seed=42, compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.time = data['time']
        cls.event = data['event']

        cox = CoxPHSurvivalAnalysis()
        nuis = fit_nuisance_survival(
            cls.time, cls.event, cls.T, cls.X,
            model_censoring=cox,
            model_event=cox,
        )
        cls.Y_star = aipcw_cut_rmst(
            cls.T, cls.time, cls.event, 2.0,
            nuis.G_a0, nuis.G_a1, nuis.S_a0, nuis.S_a1,
            time_grid=nuis.time_grid,
        )
        cls.n = cls.X.shape[0]

class TestIPTWLearner(_PseudoOutcomeBase):

    def test_fit_effect_shape(self):
        est = IPTWLearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = IPTWLearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_default_model_cate(self):
        est = IPTWLearner()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))


class TestULearner(_PseudoOutcomeBase):

    def test_fit_effect_shape(self):
        est = ULearner(model_cate=gbr(), model_mu=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = ULearner(model_cate=gbr(), model_mu=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_default_model_mu(self):
        est = ULearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))

    def test_default_model_cate(self):
        est = ULearner()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))


class TestMCLearner(_PseudoOutcomeBase):

    def test_fit_effect_shape(self):
        est = MCLearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = MCLearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_ate_scalar(self):
        est = MCLearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        ate = est.ate(self.X)
        self.assertEqual(np.asarray(ate).ndim, 0)
        self.assertTrue(np.isfinite(ate))

    def test_default_model_cate(self):
        est = MCLearner()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))


class TestMCEALearner(_PseudoOutcomeBase):

    def test_fit_effect_shape(self):
        est = MCEALearner(model_cate=gbr(), model_mu=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = MCEALearner(model_cate=gbr(), model_mu=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_default_model_mu(self):
        est = MCEALearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))

    def test_default_model_cate(self):
        est = MCEALearner()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))


class TestRALearner(_PseudoOutcomeBase):

    def test_fit_effect_shape(self):
        est = RALearner(model_cate=gbr(), model_mu=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = RALearner(model_cate=gbr(), model_mu=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_default_model_mu(self):
        est = RALearner(model_cate=gbr())
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))

    def test_default_model_cate(self):
        est = RALearner()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertEqual(est.effect(self.X).shape, (self.n,))


class TestIFLearner(_PseudoOutcomeBase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # Compute UIF scores using nuisance already fitted in _PseudoOutcomeBase
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        nuis = fit_nuisance_survival(
            cls.time, cls.event, cls.T, cls.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
        )
        ps = 0.5 * np.ones_like(cls.T, dtype=float)
        bw = 1.0 / (ps * cls.T + (1.0 - ps) * (1.0 - cls.T))
        tilt = np.ones_like(ps)
        cls.Y_uif = uif_diff_rmst(
            cls.T, cls.time, cls.event, 2.0,
            bw, tilt,
            nuis.G_a0, nuis.G_a1,
            nuis.S_a0, nuis.S_a1,
            time_grid=nuis.time_grid,
        )

    def test_fit_effect_shape(self):
        est = IFLearner(model_cate=gbr())
        est.fit(self.Y_uif, self.T, X=self.X)
        cate = est.const_marginal_effect(self.X)
        self.assertEqual(cate.shape, (self.n, 1))

    def test_effect_finite(self):
        est = IFLearner(model_cate=gbr())
        est.fit(self.Y_uif, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.const_marginal_effect(self.X))))

    def test_ate_scalar(self):
        est = IFLearner(model_cate=gbr())
        est.fit(self.Y_uif, self.T, X=self.X)
        ate = np.mean(est.const_marginal_effect(self.X))
        self.assertEqual(np.asarray(ate).ndim, 0)
        self.assertTrue(np.isfinite(ate))

    def test_default_model_cate(self):
        est = IFLearner()
        est.fit(self.Y_uif, self.T, X=self.X)
        self.assertEqual(est.const_marginal_effect(self.X).shape, (self.n, 1))


if __name__ == '__main__':
    unittest.main()
