# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Smoke tests for CrossFit* metalearners.

Tests verify:
  - fit() runs without error
  - effect() returns shape (n,)
  - effect values are finite
  - cv=2 and cv=3 both work
"""

import unittest
import numpy as np

from econml.metalearners._censor_metalearners import (
    TLearner,
    SLearner,
    XLearner,
    IPTWLearner,
    AIPTWLearner,
    MCLearner,
    MCEALearner,
    RLearner,
    RALearner,
    ULearner,
    IFLearner,
    SurvivalTLearner,
    SurvivalSLearner,
    CompetingRisksTLearner,
    CompetingRisksSLearner,
    SeparableDirectAstar1TLearner,
    SeparableIndirectAstar1TLearner,
    SeparableDirectAstar1SLearner,
    SeparableIndirectAstar1SLearner,
)
from econml.censor import fit_nuisance_survival, aipcw_cut_rmst, uif_diff_rmst
from ._helpers import gbr, lr, ridge
from .dgp import make_survival_data, make_competing_data


# ---------------------------------------------------------------------------
# Shared fixture: generate survival data + AIPCW pseudo-outcome
# ---------------------------------------------------------------------------

class _CrossFitBase(unittest.TestCase):
    """Base class: survival data + pre-computed AIPCW + UIF pseudo-outcomes."""

    @classmethod
    def setUpClass(cls):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        data = make_survival_data(n=360, tau=2.0, seed=42, compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.time = data['time']
        cls.event = data['event']
        cls.Y = data['Y']  # structured array for survival learners
        cls.n = cls.X.shape[0]

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
        # UIF scores for IFLearner test
        ps = data.get('ps', 0.5 * np.ones(cls.n))
        bw = 1.0 / (ps * cls.T + (1 - ps) * (1 - cls.T))
        tilt = np.ones(cls.n)
        cls.Y_uif = uif_diff_rmst(
            cls.T, cls.time, cls.event, 2.0, bw, tilt,
            nuis.G_a0, nuis.G_a1, nuis.S_a0, nuis.S_a1,
            time_grid=nuis.time_grid,
        )


# ---------------------------------------------------------------------------
# CrossFitTLearner
# ---------------------------------------------------------------------------

class TestCrossFitTLearner(_CrossFitBase):

    def _est(self, cv=2):
        return TLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)

    def test_cv_one_raises(self):
        est = self._est(cv=1)
        with self.assertRaisesRegex(ValueError, "cv >= 2"):
            est.fit(self.Y_star, self.T, X=self.X)


# ---------------------------------------------------------------------------
# CrossFitSLearner
# ---------------------------------------------------------------------------

class TestCrossFitSLearner(_CrossFitBase):

    def _est(self, cv=2):
        return SLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitXLearner
# ---------------------------------------------------------------------------

class TestCrossFitXLearner(_CrossFitBase):

    def _est(self, cv=2):
        return XLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))


# ---------------------------------------------------------------------------
# CrossFitIPTWLearner
# ---------------------------------------------------------------------------

class TestCrossFitIPTWLearner(_CrossFitBase):

    def _est(self, cv=2):
        return IPTWLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitAIPTWLearner
# ---------------------------------------------------------------------------

class TestCrossFitAIPTWLearner(_CrossFitBase):

    def _est(self, cv=2):
        return AIPTWLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitMCLearner
# ---------------------------------------------------------------------------

class TestCrossFitMCLearner(_CrossFitBase):

    def _est(self, cv=2):
        return MCLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitMCEALearner
# ---------------------------------------------------------------------------

class TestCrossFitMCEALearner(_CrossFitBase):

    def _est(self, cv=2):
        return MCEALearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitRALearner
# ---------------------------------------------------------------------------

class TestCrossFitRALearner(_CrossFitBase):

    def _est(self, cv=2):
        return RALearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitULearner
# ---------------------------------------------------------------------------

class TestCrossFitULearner(_CrossFitBase):

    def _est(self, cv=2):
        return ULearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitRLearner
# ---------------------------------------------------------------------------

class TestCrossFitRLearner(_CrossFitBase):

    def _est(self, cv=2):
        return RLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_star, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_star, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitIFLearner
# ---------------------------------------------------------------------------

class TestCrossFitIFLearner(_CrossFitBase):

    def _est(self, cv=2):
        return IFLearner(cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y_uif, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y_uif, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_fit_without_treatment(self):
        est = self._est()
        est.fit(self.Y_uif, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))
        self.assertTrue(np.all(np.isfinite(cate)))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y_uif, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y_uif, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitSurvivalTLearner
# ---------------------------------------------------------------------------

class TestCrossFitSurvivalTLearner(_CrossFitBase):

    def _make_surv(self):
        from sksurv.ensemble import RandomSurvivalForest
        return RandomSurvivalForest(n_estimators=20, min_samples_leaf=5, random_state=0)

    def _est(self, cv=2):
        return SurvivalTLearner(tau=2.0, cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitSurvivalSLearner
# ---------------------------------------------------------------------------

class TestCrossFitSurvivalSLearner(_CrossFitBase):

    def _make_surv(self):
        from sksurv.ensemble import RandomSurvivalForest
        return RandomSurvivalForest(n_estimators=20, min_samples_leaf=5, random_state=0)

    def _est(self, cv=2):
        return SurvivalSLearner(tau=2.0, cv=cv, random_state=0)

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)


# ---------------------------------------------------------------------------
# CrossFitCompetingRisks Learners
# ---------------------------------------------------------------------------

class _CrossFitCompetingBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = make_competing_data(n=360, tau=4.0, seed=123,
                                   compute_true_cate=False)
        cls.X = data['X']
        cls.T = data['T']
        cls.Y = data['Y']
        cls.n = cls.X.shape[0]
        cls.tau = 4.0

    def _cox(self):
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        return CoxPHSurvivalAnalysis()


class TestCrossFitCompetingRisksTLearner(_CrossFitCompetingBase):

    def _est(self, cv=2, separable=False):
        kwargs = {}
        if separable:
            kwargs.update(
                compute_separable=True,
                models_competing=self._cox(),
            )
        return CompetingRisksTLearner(
            tau=self.tau,
            cv=cv,
            random_state=0,
            **kwargs,
        )

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_separable_effects(self):
        direct_est = SeparableDirectAstar1TLearner(
            tau=self.tau, cv=2, random_state=0
        )
        indirect_est = SeparableIndirectAstar1TLearner(
            tau=self.tau, cv=2, random_state=0
        )
        direct_est.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        direct = direct_est.effect(self.X)
        indirect = indirect_est.effect(self.X)
        self.assertEqual(direct.shape, (self.n,))
        self.assertEqual(indirect.shape, (self.n,))
        self.assertTrue(np.all(np.isfinite(direct)))
        self.assertTrue(np.all(np.isfinite(indirect)))

    def test_separable_effects_training_is_oof(self):
        direct_est = SeparableDirectAstar1TLearner(
            tau=self.tau, cv=2, random_state=0
        )
        indirect_est = SeparableIndirectAstar1TLearner(
            tau=self.tau, cv=2, random_state=0
        )
        direct_est.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(direct_est.effect(self.X), direct_est._training_oof_separable_direct_)
        np.testing.assert_allclose(indirect_est.effect(self.X), indirect_est._training_oof_separable_indirect_)

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)

    def test_default_models_competing(self):
        est = CompetingRisksTLearner(
            tau=self.tau,
            cv=2,
            random_state=0,
            compute_separable=True,
        )
        est.fit(self.Y, self.T, X=self.X)
        self.assertEqual(est._training_oof_separable_direct_.shape, (self.n,))
        self.assertEqual(est._training_oof_separable_indirect_.shape, (self.n,))


class TestCrossFitCompetingRisksSLearner(_CrossFitCompetingBase):

    def _est(self, cv=2, separable=False):
        kwargs = {}
        if separable:
            kwargs.update(
                compute_separable=True,
                competing_model=self._cox(),
            )
        return CompetingRisksSLearner(
            tau=self.tau,
            cv=cv,
            random_state=0,
            **kwargs,
        )

    def test_fit_effect_shape(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        cate = est.effect(self.X)
        self.assertEqual(cate.shape, (self.n,))

    def test_effect_finite(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        self.assertTrue(np.all(np.isfinite(est.effect(self.X))))

    def test_cv_parameter(self):
        for cv in (2,):
            est = self._est(cv=cv)
            est.fit(self.Y, self.T, X=self.X)
            cate = est.effect(self.X)
            self.assertEqual(cate.shape, (self.n,))

    def test_separable_effects(self):
        direct_est = SeparableDirectAstar1SLearner(
            tau=self.tau, cv=2, random_state=0
        )
        indirect_est = SeparableIndirectAstar1SLearner(
            tau=self.tau, cv=2, random_state=0
        )
        direct_est.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        direct = direct_est.effect(self.X)
        indirect = indirect_est.effect(self.X)
        self.assertEqual(direct.shape, (self.n,))
        self.assertEqual(indirect.shape, (self.n,))


class TestSeparableAstar1TLearners(_CrossFitCompetingBase):

    def test_direct_effect_matches_separable_direct(self):
        base = CompetingRisksTLearner(
            tau=self.tau, cv=2, random_state=0, compute_separable=True
        )
        direct_est = SeparableDirectAstar1TLearner(
            tau=self.tau, cv=2, random_state=0
        )
        base.fit(self.Y, self.T, X=self.X)
        direct_est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(direct_est.effect(self.X), base._training_oof_separable_direct_)

    def test_indirect_effect_matches_separable_indirect(self):
        base = CompetingRisksTLearner(
            tau=self.tau, cv=2, random_state=0, compute_separable=True
        )
        indirect_est = SeparableIndirectAstar1TLearner(
            tau=self.tau, cv=2, random_state=0
        )
        base.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(indirect_est.effect(self.X), base._training_oof_separable_indirect_)


class TestSeparableAstar1SLearners(_CrossFitCompetingBase):

    def _est(self, cv=2, separable=True):
        return CompetingRisksSLearner(
            tau=self.tau, cv=cv, random_state=0, compute_separable=separable
        )

    def test_direct_effect_matches_separable_direct(self):
        base = CompetingRisksSLearner(
            tau=self.tau, cv=2, random_state=0, compute_separable=True
        )
        direct_est = SeparableDirectAstar1SLearner(
            tau=self.tau, cv=2, random_state=0
        )
        base.fit(self.Y, self.T, X=self.X)
        direct_est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(direct_est.effect(self.X), base._training_oof_separable_direct_)

    def test_indirect_effect_matches_separable_indirect(self):
        base = CompetingRisksSLearner(
            tau=self.tau, cv=2, random_state=0, compute_separable=True
        )
        indirect_est = SeparableIndirectAstar1SLearner(
            tau=self.tau, cv=2, random_state=0
        )
        base.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(indirect_est.effect(self.X), base._training_oof_separable_indirect_)
        self.assertTrue(np.all(np.isfinite(base._training_oof_separable_direct_)))
        self.assertTrue(np.all(np.isfinite(base._training_oof_separable_indirect_)))

    def test_separable_effects_training_is_oof(self):
        direct_est = SeparableDirectAstar1SLearner(
            tau=self.tau, cv=2, random_state=0
        )
        indirect_est = SeparableIndirectAstar1SLearner(
            tau=self.tau, cv=2, random_state=0
        )
        direct_est.fit(self.Y, self.T, X=self.X)
        indirect_est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(direct_est.effect(self.X), direct_est._training_oof_separable_direct_)
        np.testing.assert_allclose(indirect_est.effect(self.X), indirect_est._training_oof_separable_indirect_)

    def test_training_effect_is_oof(self):
        est = self._est()
        est.fit(self.Y, self.T, X=self.X)
        np.testing.assert_allclose(est.effect(self.X), est._training_oof_effect_)

    def test_default_competing_model(self):
        est = CompetingRisksSLearner(
            tau=self.tau,
            cv=2,
            random_state=0,
            compute_separable=True,
        )
        est.fit(self.Y, self.T, X=self.X)
        self.assertEqual(est._training_oof_separable_direct_.shape, (self.n,))
        self.assertEqual(est._training_oof_separable_indirect_.shape, (self.n,))
