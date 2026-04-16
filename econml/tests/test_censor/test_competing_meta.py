"""Smoke tests for CompetingRisksTLearner and CompetingRisksSLearner.

DGP mirrors original/simulation/competing/competing_hte_simulation_case1.R
(exponential cause times, treatment-dependent censoring, 3-state event indicator).

Tests verify:
  - fit() runs without error
  - effect() returns shape (n,)
  - ate() returns a scalar
  - effect values are finite
  - const_marginal_effect() returns shape (n, 1)
"""

import unittest
import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis

from econml.metalearners._censor_metalearners import (
    CompetingRisksTLearner, CompetingRisksSLearner,
    SeparableDirectAstar1TLearner, SeparableDirectAstar1SLearner,
    SeparableIndirectAstar1TLearner, SeparableIndirectAstar1SLearner,
)
from .dgp import make_competing_data


class TestCompetingRisksTLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_competing_data(n=300, tau=4.0, seed=42,
                                   compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.Y = data['Y']

    def _make_model(self):
        return CoxPHSurvivalAnalysis()

    def _make_est(self):
        return CompetingRisksTLearner(
            models=self._make_model(),
            models_cause=self._make_model(),
            tau=4.0,
        )

    def test_fit_effect_shape(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        # effect() squeezes d_t=1 dimension → (n,) per EconML convention
        self.assertEqual(cate.shape, (self.X.shape[0],))

    def test_effect_finite(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertTrue(np.all(np.isfinite(cate)))

    def test_ate_scalar(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        ate = est.ate(self.X)
        self.assertEqual(np.asarray(ate).ndim, 0)
        self.assertTrue(np.isfinite(ate))

    def test_const_marginal_effect_shape(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        cme = est.const_marginal_effect(self.X)
        self.assertEqual(cme.shape, (self.X.shape[0], 1))

    def test_separable_direct_effect(self):
        est = SeparableDirectAstar1TLearner(
            models=self._make_model(),
            models_cause=self._make_model(),
            tau=4.0,
            models_competing=self._make_model(),
        )
        est.fit(self.Y, self.T, X=self.X)
        direct_astar1 = est.effect(self.X)
        self.assertEqual(direct_astar1.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(direct_astar1)))

    def test_separable_indirect_effect(self):
        est = SeparableIndirectAstar1TLearner(
            models=self._make_model(),
            models_cause=self._make_model(),
            tau=4.0,
            models_competing=self._make_model(),
        )
        est.fit(self.Y, self.T, X=self.X)
        indirect_astar1 = est.effect(self.X)
        self.assertEqual(indirect_astar1.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(indirect_astar1)))

    def test_default_models_competing(self):
        direct_est = SeparableDirectAstar1TLearner(tau=4.0)
        indirect_est = SeparableIndirectAstar1TLearner(tau=4.0)
        direct_est.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        self.assertEqual(direct_est.effect(self.X).shape, (self.X.shape[0],))
        self.assertEqual(indirect_est.effect(self.X).shape, (self.X.shape[0],))

    def test_default_survival_models(self):
        est = CompetingRisksTLearner(tau=4.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(cate)))


class TestCompetingRisksSLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_competing_data(n=300, tau=4.0, seed=42,
                                   compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.Y = data['Y']

    def _make_model(self):
        return CoxPHSurvivalAnalysis()

    def _make_est(self):
        return CompetingRisksSLearner(
            overall_model=self._make_model(),
            cause_model=self._make_model(),
            tau=4.0,
        )

    def test_fit_effect_shape(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.X.shape[0],))

    def test_effect_finite(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertTrue(np.all(np.isfinite(cate)))

    def test_ate_scalar(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        ate = est.ate(self.X)
        self.assertEqual(np.asarray(ate).ndim, 0)
        self.assertTrue(np.isfinite(ate))

    def test_const_marginal_effect_shape(self):
        est = self._make_est()
        est.fit(self.Y, self.T, X=self.X)
        cme = est.const_marginal_effect(self.X)
        self.assertEqual(cme.shape, (self.X.shape[0], 1))

    def test_separable_direct_effect(self):
        est = SeparableDirectAstar1SLearner(
            overall_model=self._make_model(),
            cause_model=self._make_model(),
            tau=4.0,
            competing_model=self._make_model(),
        )
        est.fit(self.Y, self.T, X=self.X)
        direct_astar1 = est.effect(self.X)
        self.assertEqual(direct_astar1.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(direct_astar1)))

    def test_separable_indirect_effect(self):
        est = SeparableIndirectAstar1SLearner(
            overall_model=self._make_model(),
            cause_model=self._make_model(),
            tau=4.0,
            competing_model=self._make_model(),
        )
        est.fit(self.Y, self.T, X=self.X)
        indirect_astar1 = est.effect(self.X)
        self.assertEqual(indirect_astar1.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(indirect_astar1)))

    def test_default_competing_model(self):
        direct_est = SeparableDirectAstar1SLearner(tau=4.0)
        indirect_est = SeparableIndirectAstar1SLearner(tau=4.0)
        direct_est.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        self.assertEqual(direct_est.effect(self.X).shape, (self.X.shape[0],))
        self.assertEqual(indirect_est.effect(self.X).shape, (self.X.shape[0],))

    def test_default_survival_models(self):
        est = CompetingRisksSLearner(tau=4.0)
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(cate)))


if __name__ == '__main__':
    unittest.main()
