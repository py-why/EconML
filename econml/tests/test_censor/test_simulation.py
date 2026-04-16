# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""End-to-end simulation tests for censored-outcome HTE workflows.

These tests sit above the smoke/unit tests and exercise the notebook-style
pipelines end to end:

  simulation DGP -> OOS nuisances -> CUT/UIF transforms -> learner fit/predict

The goal is not exact numerical replication of the original R simulations.
Instead, the tests verify that representative workflows:

  - run without error on the ported DGPs
  - return finite out-of-sample training predictions
  - retain non-trivial signal relative to the known simulation truth for a
    small set of stable estimators
"""

import unittest

import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

from econml.censor import (
    fit_nuisance_survival_crossfit,
    fit_nuisance_competing_crossfit,
    aipcw_cut_rmst,
    uif_diff_rmst,
    aipcw_cut_rmtlj,
    aipcw_cut_rmtlj_sep_direct_astar1,
    aipcw_cut_rmtlj_sep_indirect_astar1,
    uif_diff_rmtlj_sep_indirect_astar1,
)
from econml.grf import CausalSurvivalForest, GRFCausalForest
from econml.metalearners._censor_metalearners import (
    SurvivalTLearner,
    CompetingRisksTLearner,
    SeparableDirectAstar1TLearner,
    SeparableIndirectAstar1TLearner,
    TLearner,
    AIPTWLearner,
    RLearner,
    IFLearner,
)

from ._helpers import gbr, lr
from .dgp import make_survival_data, make_competing_data


def _rsf(seed=0):
    return RandomSurvivalForest(
        n_estimators=20,
        min_samples_leaf=5,
        random_state=seed,
    )


def _corr(pred, truth):
    pred = np.asarray(pred).ravel()
    truth = np.asarray(truth).ravel()
    return float(np.corrcoef(pred, truth)[0, 1])


class TestEndToEndSurvivalSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tau = 2.0
        data = make_survival_data(n=220, tau=tau, seed=123, compute_true_cate=True)

        cls.X = data["X"]
        cls.T = data["T"]
        cls.time = data["time"]
        cls.event = data["event"]
        cls.Y = data["Y"]
        cls.true_cate = data["true_cate"]
        cls.n = cls.X.shape[0]

        nuis = fit_nuisance_survival_crossfit(
            cls.time,
            cls.event.astype(int),
            cls.T,
            cls.X,
            model_censoring=CoxPHSurvivalAnalysis(),
            model_event=CoxPHSurvivalAnalysis(),
            propensity_model=lr(),
            cv=2,
            random_state=123,
        )
        cls.nuis = nuis
        cls.Y_aipcw = aipcw_cut_rmst(
            cls.T,
            cls.time,
            cls.event,
            tau,
            nuis.G_a0,
            nuis.G_a1,
            nuis.S_a0,
            nuis.S_a1,
            time_grid=nuis.time_grid,
        )
        cls.Y_uif = uif_diff_rmst(
            cls.T,
            cls.time,
            cls.event,
            tau,
            nuis.iptw,
            nuis.naive,
            nuis.G_a0,
            nuis.G_a1,
            nuis.S_a0,
            nuis.S_a1,
            time_grid=nuis.time_grid,
        )

        cls.est_survival_t = SurvivalTLearner(
            models=_rsf(0),
            tau=tau,
            cv=2,
            random_state=0,
        )
        cls.est_survival_t.fit(cls.Y, cls.T, X=cls.X)
        cls.pred_survival_t = cls.est_survival_t.effect(cls.X)

        cls.est_tl = TLearner(model_mu=gbr(), cv=2, random_state=0)
        cls.est_tl.fit(cls.Y_aipcw, cls.T, X=cls.X)
        cls.pred_tl = cls.est_tl.effect(cls.X)

        cls.est_aiptw = AIPTWLearner(
            propensity_model=lr(),
            model_mu=gbr(),
            model_cate=gbr(),
            cv=2,
            random_state=0,
        )
        cls.est_aiptw.fit(cls.Y_aipcw, cls.T, X=cls.X)
        cls.pred_aiptw = cls.est_aiptw.effect(cls.X)

        cls.est_r = RLearner(
            propensity_model=lr(),
            model_mu=gbr(),
            model_cate=gbr(),
            cv=2,
            random_state=0,
        )
        cls.est_r.fit(cls.Y_aipcw, cls.T, X=cls.X)
        cls.pred_r = cls.est_r.effect(cls.X)

        cls.est_if = IFLearner(
            propensity_model=lr(),
            model_cate=gbr(),
            cv=2,
            random_state=0,
        )
        cls.est_if.fit(cls.Y_uif, X=cls.X)
        cls.pred_if = cls.est_if.effect(cls.X)

        cls.est_csf = CausalSurvivalForest(
            horizon=tau,
            n_estimators=52,
            random_state=123,
        )
        cls.est_csf.fit(cls.X, cls.T, cls.time, cls.event.astype(bool))
        cls.pred_csf = cls.est_csf.effect(cls.X)

        cls.est_cf = GRFCausalForest(num_trees=52, seed=123)
        cls.est_cf.fit(cls.X, cls.Y_aipcw, cls.T)
        cls.pred_cf = cls.est_cf.predict().predictions

    def test_survival_workflow_outputs_are_finite(self):
        self.assertEqual(self.Y_aipcw.shape, (self.n,))
        self.assertEqual(self.Y_uif.shape, (self.n,))
        self.assertGreater(np.std(self.Y_aipcw), 0.0)
        self.assertGreater(np.std(self.Y_uif), 0.0)

        for pred in (
            self.pred_survival_t,
            self.pred_tl,
            self.pred_aiptw,
            self.pred_r,
            self.pred_if,
            self.pred_csf,
            self.pred_cf,
        ):
            self.assertEqual(np.asarray(pred).shape, (self.n,))
            self.assertTrue(np.all(np.isfinite(pred)))

    def test_survival_crossfit_and_oob_contracts_hold(self):
        np.testing.assert_allclose(
            self.est_survival_t.effect(self.X),
            self.est_survival_t._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_tl.effect(self.X),
            self.est_tl._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_aiptw.effect(self.X),
            self.est_aiptw._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_r.effect(self.X),
            self.est_r._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_if.effect(self.X),
            self.est_if._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_csf.effect(self.X),
            np.asarray(self.est_csf.oob_predict(self.X)).ravel(),
        )

    def test_survival_representative_learners_track_truth(self):
        self.assertGreater(_corr(self.pred_survival_t, self.true_cate), 0.15)
        self.assertGreater(_corr(self.pred_csf, self.true_cate), 0.15)


class TestEndToEndCompetingSimulation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        tau = 4.0
        data = make_competing_data(n=260, tau=tau, seed=456, compute_true_cate=True)

        cls.X = data["X"]
        cls.T = data["T"]
        cls.time = data["time"]
        cls.event = data["event"]
        cls.Y = data["Y"]
        cls.true_total = data["true_cate_total"]
        cls.true_direct = data["true_cate_direct"]
        cls.true_indirect = data["true_cate_indirect"]
        cls.n = cls.X.shape[0]

        nuis = fit_nuisance_competing_crossfit(
            cls.time,
            cls.event,
            cls.T,
            cls.X,
            model_censoring=_rsf(1),
            model_event=_rsf(2),
            model_cause=_rsf(3),
            model_competing=_rsf(4),
            propensity_model=lr(),
            cause=1,
            cv=2,
            random_state=456,
        )
        cls.nuis = nuis
        cls.Y_total = aipcw_cut_rmtlj(
            cls.T,
            cls.time,
            cls.event,
            tau,
            nuis.G_a0,
            nuis.G_a1,
            nuis.S_a0,
            nuis.S_a1,
            nuis.Sj_a0,
            nuis.Sj_a1,
            cause=1,
            time_grid=nuis.time_grid,
        )
        cls.Y_direct = aipcw_cut_rmtlj_sep_direct_astar1(
            cls.T,
            cls.time,
            cls.event,
            tau,
            nuis.G_a0,
            nuis.G_a1,
            nuis.S_a0,
            nuis.S_a1,
            nuis.Sj_a0,
            nuis.Sj_a1,
            nuis.Sjbar_a0,
            nuis.Sjbar_a1,
            cause=1,
            time_grid=nuis.time_grid,
        )
        cls.Y_indirect = aipcw_cut_rmtlj_sep_indirect_astar1(
            cls.T,
            cls.time,
            cls.event,
            tau,
            nuis.G_a0,
            nuis.G_a1,
            nuis.S_a0,
            nuis.S_a1,
            nuis.Sj_a0,
            nuis.Sj_a1,
            nuis.Sjbar_a0,
            nuis.Sjbar_a1,
            cause=1,
            time_grid=nuis.time_grid,
        )
        cls.Y_uif_indirect = uif_diff_rmtlj_sep_indirect_astar1(
            cls.T,
            nuis.ps,
            cls.time,
            cls.event,
            tau,
            nuis.iptw,
            nuis.naive,
            nuis.G_a0,
            nuis.G_a1,
            nuis.S_a0,
            nuis.S_a1,
            nuis.Sj_a0,
            nuis.Sj_a1,
            nuis.Sjbar_a0,
            nuis.Sjbar_a1,
            cause=1,
            time_grid=nuis.time_grid,
        )

        cls.est_competing_total = CompetingRisksTLearner(
            models=_rsf(5),
            models_cause=_rsf(6),
            tau=tau,
            cv=2,
            random_state=0,
        )
        cls.est_competing_total.fit(cls.Y, cls.T, X=cls.X)
        cls.pred_competing_total = cls.est_competing_total.effect(cls.X)

        cls.est_competing_direct = SeparableDirectAstar1TLearner(
            models=_rsf(7),
            models_cause=_rsf(8),
            models_competing=_rsf(9),
            tau=tau,
            cv=2,
            random_state=0,
        )
        cls.est_competing_direct.fit(cls.Y, cls.T, X=cls.X)
        cls.pred_competing_direct = cls.est_competing_direct.effect(cls.X)

        cls.est_competing_indirect = SeparableIndirectAstar1TLearner(
            models=_rsf(7),
            models_cause=_rsf(8),
            models_competing=_rsf(9),
            tau=tau,
            cv=2,
            random_state=0,
        )
        cls.est_competing_indirect.fit(cls.Y, cls.T, X=cls.X)
        cls.pred_competing_indirect = cls.est_competing_indirect.effect(cls.X)

        cls.est_tl_total = TLearner(model_mu=gbr(), cv=2, random_state=0)
        cls.est_tl_total.fit(cls.Y_total, cls.T, X=cls.X)
        cls.pred_tl_total = cls.est_tl_total.effect(cls.X)

        cls.est_tl_direct = TLearner(model_mu=gbr(), cv=2, random_state=0)
        cls.est_tl_direct.fit(cls.Y_direct, cls.T, X=cls.X)
        cls.pred_tl_direct = cls.est_tl_direct.effect(cls.X)

        cls.est_aiptw_total = AIPTWLearner(
            propensity_model=lr(),
            model_mu=gbr(),
            model_cate=gbr(),
            cv=2,
            random_state=0,
        )
        cls.est_aiptw_total.fit(cls.Y_total, cls.T, X=cls.X)
        cls.pred_aiptw_total = cls.est_aiptw_total.effect(cls.X)

        cls.est_r_total = RLearner(
            propensity_model=lr(),
            model_mu=gbr(),
            model_cate=gbr(),
            cv=2,
            random_state=0,
        )
        cls.est_r_total.fit(cls.Y_total, cls.T, X=cls.X)
        cls.pred_r_total = cls.est_r_total.effect(cls.X)

        cls.est_if_indirect = IFLearner(
            propensity_model=lr(),
            model_cate=gbr(),
            cv=2,
            random_state=0,
        )
        cls.est_if_indirect.fit(cls.Y_uif_indirect, X=cls.X)
        cls.pred_if_indirect = cls.est_if_indirect.effect(cls.X)

        cls.est_cf_total = GRFCausalForest(num_trees=52, seed=123)
        cls.est_cf_total.fit(cls.X, cls.Y_total, cls.T)
        cls.pred_cf_total = cls.est_cf_total.predict().predictions

    def test_competing_workflow_outputs_are_finite(self):
        for y in (self.Y_total, self.Y_direct, self.Y_indirect, self.Y_uif_indirect):
            self.assertEqual(np.asarray(y).shape, (self.n,))
            self.assertTrue(np.all(np.isfinite(y)))
            self.assertGreater(np.std(y), 0.0)

        for pred in (
            self.pred_competing_total,
            self.pred_competing_direct,
            self.pred_competing_indirect,
            self.pred_tl_total,
            self.pred_tl_direct,
            self.pred_aiptw_total,
            self.pred_r_total,
            self.pred_if_indirect,
            self.pred_cf_total,
        ):
            self.assertEqual(np.asarray(pred).shape, (self.n,))
            self.assertTrue(np.all(np.isfinite(pred)))

    def test_competing_crossfit_and_oob_contracts_hold(self):
        np.testing.assert_allclose(
            self.est_competing_total.effect(self.X),
            self.est_competing_total._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_competing_direct.effect(self.X),
            self.est_competing_direct._training_oof_separable_direct_,
        )
        np.testing.assert_allclose(
            self.pred_competing_direct,
            self.est_competing_direct._training_oof_separable_direct_,
        )
        np.testing.assert_allclose(
            self.pred_competing_indirect,
            self.est_competing_indirect._training_oof_separable_indirect_,
        )
        np.testing.assert_allclose(
            self.est_tl_total.effect(self.X),
            self.est_tl_total._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_tl_direct.effect(self.X),
            self.est_tl_direct._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_aiptw_total.effect(self.X),
            self.est_aiptw_total._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_r_total.effect(self.X),
            self.est_r_total._training_oof_effect_,
        )
        np.testing.assert_allclose(
            self.est_if_indirect.effect(self.X),
            self.est_if_indirect._training_oof_effect_,
        )

    def test_competing_representative_learners_track_truth(self):
        self.assertGreater(_corr(self.pred_competing_total, self.true_total), 0.20)
        self.assertGreater(_corr(self.pred_cf_total, self.true_total), 0.20)
        self.assertGreater(_corr(self.pred_competing_direct, self.true_direct), 0.05)


if __name__ == "__main__":
    unittest.main()
