# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
import pickle
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.model_selection import KFold
from econml.dml import DML, LinearDML, SparseLinearDML, KernelDML, CausalForestDML
from econml.dml import NonParamDML
import numpy as np
import pandas as pd
from econml.utilities import shape, hstack, vstack, reshape, cross_product
from econml.inference import BootstrapInference
from contextlib import ExitStack
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import itertools
from econml.sklearn_extensions.linear_model import WeightedLasso, StatsModelsRLM
from econml.tests.test_statsmodels import _summarize
from econml.grf import MultiOutputGRF
from econml.tests.utilities import (GroupingModel, NestedModel)


try:
    import ray
    ray_installed = True
except ImportError:
    ray_installed = False


# all solutions to underdetermined (or exactly determined) Ax=b are given by A⁺b+(I-A⁺A)w for some arbitrary w
# note that if Ax=b is overdetermined, this will raise an assertion error


def rand_sol(A, b):
    """Generate a random solution to the equation Ax=b."""
    assert np.linalg.matrix_rank(A) <= len(b)
    A_plus = np.linalg.pinv(A)
    x = A_plus @ b
    return x + (np.eye(x.shape[0]) - A_plus @ A) @ np.random.normal(size=x.shape)


@pytest.mark.dml
class TestDML(unittest.TestCase):

    def test_cate_api_without_ray(self):
        treatment_featurizations = [None]
        self._test_cate_api(treatment_featurizations, False)

    @pytest.mark.ray
    def test_cate_api_with_ray(self):
        try:
            ray.init(num_cpus=1)
            treatment_featurizations = [None]
            self._test_cate_api(treatment_featurizations, use_ray=True)
        finally:
            ray.shutdown()

    def _test_cate_api(self, treatment_featurizations, use_ray=False):
        """Test that we correctly implement the CATE API."""
        n_c = 20  # number of rows for continuous models
        n_d = 30  # number of rows for discrete models

        def make_random(n, is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d >= 0 else (n,)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b', 'c'], size=sz)
                    # ensure that we've got at least 6 of every element
                    # 2 outer splits, 3 inner splits when model_t is 'auto' and treatment is discrete
                    # NOTE: this number may need to change if the default number of folds in
                    #       WeightedStratifiedKFold changes
                    _, counts = np.unique(arr, return_counts=True)
                    if len(counts) == 3 and counts.min() > 5:
                        return arr
            else:
                return np.random.normal(size=sz)

        for d_t in [2, 1, -1]:
            for is_discrete in [True, False] if d_t <= 1 else [False]:
                for treatment_featurizer in treatment_featurizations:
                    for d_y in [3, 1, -1]:
                        for d_x in [2, None]:
                            for d_w in [2, None]:
                                n = n_d if is_discrete else n_c
                                W, X, Y, T = [make_random(n, is_discrete, d)
                                              for is_discrete, d in [(False, d_w),
                                                                     (False, d_x),
                                                                     (False, d_y),
                                                                     (is_discrete, d_t)]]

                                for featurizer, fit_cate_intercept in\
                                    [(None, True),
                                     (PolynomialFeatures(degree=2, include_bias=False), True),
                                     (PolynomialFeatures(degree=2, include_bias=True), False)]:

                                    if is_discrete and treatment_featurizer:
                                        continue

                                    d_t_final = 2 if is_discrete else d_t

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

                                    fd_x = featurizer.fit_transform(X).shape[1:] if featurizer and d_x \
                                        else ((d_x,) if d_x else (0,))
                                    coef_shape = Y.shape[1:] + (T.shape[1:] if not is_discrete else (2,)) + fd_x

                                    coef_summaryframe_shape = (
                                        (d_y if d_y > 0 else 1) * (fd_x[0] if fd_x[0] >
                                                                   0 else 1) * (d_t_final if d_t_final > 0 else 1), 6)
                                    intercept_shape = Y.shape[1:] + (T.shape[1:] if not is_discrete else (2,))
                                    intercept_summaryframe_shape = (
                                        (d_y if d_y > 0 else 1) * (d_t_final if d_t_final > 0 else 1), 6)

                                    model_t = LogisticRegression() if is_discrete else Lasso()

                                    all_infs = [None, 'auto', BootstrapInference(2)]

                                    for est, multi, infs in \
                                            [(DML(model_y=Lasso(),
                                                  model_t=model_t,
                                                  model_final=Lasso(alpha=0.1, fit_intercept=False),
                                                  featurizer=featurizer,
                                                  fit_cate_intercept=fit_cate_intercept,
                                                  discrete_treatment=is_discrete,
                                                  treatment_featurizer=treatment_featurizer,
                                                  use_ray=use_ray),
                                              True,
                                              [None] +
                                              ([BootstrapInference(
                                                  n_bootstrap_samples=20)] if not is_discrete else [])),
                                             (DML(model_y=Lasso(),
                                                  model_t=model_t,
                                                  model_final=StatsModelsRLM(fit_intercept=False),
                                                  featurizer=featurizer,
                                                  fit_cate_intercept=fit_cate_intercept,
                                                  discrete_treatment=is_discrete,
                                                  treatment_featurizer=treatment_featurizer,
                                                  use_ray=use_ray),
                                              True,
                                              ['auto']),
                                             (LinearDML(model_y=Lasso(),
                                                        model_t=model_t,
                                                        featurizer=featurizer,
                                                        fit_cate_intercept=fit_cate_intercept,
                                                        discrete_treatment=is_discrete,
                                                        treatment_featurizer=treatment_featurizer,
                                                        use_ray=use_ray),
                                              True,
                                              all_infs),
                                             (SparseLinearDML(model_y=WeightedLasso(),
                                                              model_t=model_t,
                                                              featurizer=featurizer,
                                                              fit_cate_intercept=fit_cate_intercept,
                                                              discrete_treatment=is_discrete,
                                                              treatment_featurizer=treatment_featurizer,
                                                              use_ray=use_ray),
                                              True,
                                              [None, 'auto'] +
                                              ([BootstrapInference(n_bootstrap_samples=20)]
                                              if not is_discrete else [])),
                                             (KernelDML(model_y=WeightedLasso(),
                                                        model_t=model_t,
                                                        fit_cate_intercept=fit_cate_intercept,
                                                        discrete_treatment=is_discrete,
                                                        treatment_featurizer=treatment_featurizer,
                                                        use_ray=use_ray),
                                              False,
                                              [None]),
                                             (CausalForestDML(model_y=WeightedLasso(),
                                                              model_t=model_t,
                                                              featurizer=featurizer,
                                                              n_estimators=4,
                                                              n_jobs=1,
                                                              discrete_treatment=is_discrete,
                                                              use_ray=use_ray),
                                              True,
                                              ['auto', 'blb'])]:

                                        if not (multi) and d_y > 1:
                                            continue

                                        if X is None and isinstance(est, CausalForestDML):
                                            continue

                                        # ensure we can serialize the unfit estimator
                                        pickle.dumps(est)

                                        for inf in infs:
                                            with self.subTest(d_w=d_w, d_x=d_x, d_y=d_y, d_t=d_t,
                                                              is_discrete=is_discrete, est=est, inf=inf):

                                                if X is None and (not fit_cate_intercept):
                                                    with pytest.raises(AttributeError):
                                                        est.fit(Y, T, X=X, W=W, inference=inf)
                                                    continue

                                                est.fit(Y, T, X=X, W=W, inference=inf)

                                                # ensure we can pickle the fit estimator
                                                pickle.dumps(est)

                                                # make sure we can call the marginal_effect and effect methods
                                                const_marg_eff = est.const_marginal_effect(X)
                                                marg_eff = est.marginal_effect(T, X)
                                                self.assertEqual(shape(marg_eff), marginal_effect_shape)
                                                self.assertEqual(shape(const_marg_eff), const_marginal_effect_shape)

                                                np.testing.assert_allclose(
                                                    marg_eff if d_x else marg_eff[0:1], const_marg_eff)

                                                assert isinstance(est.score_, float)
                                                for score_list in est.nuisance_scores_y:
                                                    for score in score_list:
                                                        assert isinstance(score, float)
                                                for score_list in est.nuisance_scores_t:
                                                    for score in score_list:
                                                        assert isinstance(score, float)

                                                T0 = np.full_like(T, 'a') if is_discrete else np.zeros_like(T)
                                                eff = est.effect(X, T0=T0, T1=T)
                                                self.assertEqual(shape(eff), effect_shape)

                                                if ((not isinstance(est, KernelDML)) and
                                                        (not isinstance(est, CausalForestDML))):
                                                    self.assertEqual(shape(est.coef_), coef_shape)
                                                    if fit_cate_intercept:
                                                        self.assertEqual(shape(est.intercept_), intercept_shape)
                                                    else:
                                                        with pytest.raises(AttributeError):
                                                            self.assertEqual(shape(est.intercept_), intercept_shape)

                                                if d_y > 1 or is_discrete or d_t > 1:
                                                    # sensitivity interval should not calculate
                                                    # when d_y > 1 or t is multi category discrete / multi dim cont
                                                    with pytest.raises(
                                                            ValueError,
                                                            match='Sensitivity analysis for DML is not supported'):
                                                        est.sensitivity_interval()

                                                    with pytest.raises(
                                                            ValueError,
                                                            match='Sensitivity analysis for DML is not supported'):
                                                        est.robustness_value()

                                                else:

                                                    # make sure sensitivity methods can be called.
                                                    # allow for data-dependent error with negative sigma, nu
                                                    for method in [est.sensitivity_interval,
                                                                   est.robustness_value,
                                                                   est.sensitivity_summary]:
                                                        try:
                                                            method()
                                                        except ValueError as e:
                                                            assert 'sigma and nu must be non-negative' in str(e)

                                                if inf is not None:
                                                    const_marg_eff_int = est.const_marginal_effect_interval(X)
                                                    marg_eff_int = est.marginal_effect_interval(T, X)
                                                    self.assertEqual(shape(marg_eff_int),
                                                                     (2,) + marginal_effect_shape)
                                                    self.assertEqual(shape(const_marg_eff_int),
                                                                     (2,) + const_marginal_effect_shape)
                                                    self.assertEqual(shape(est.effect_interval(X, T0=T0, T1=T)),
                                                                     (2,) + effect_shape)
                                                    if ((not isinstance(est, KernelDML)) and
                                                            (not isinstance(est, CausalForestDML))):
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
                                                    T1 = np.full_like(T, 'b') if is_discrete else T
                                                    effect_inf = est.effect_inference(X, T0=T0, T1=T1)
                                                    marg_effect_inf = est.marginal_effect_inference(T, X)
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
                                                    if ((not isinstance(est, KernelDML)) and
                                                            (not isinstance(est, CausalForestDML))):
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

                                                est.score(Y, T, X, W)

                                                if isinstance(est, CausalForestDML):
                                                    np.testing.assert_array_equal(est.feature_importances_.shape,
                                                                                  ((d_y,) if d_y > 0 else ()) + fd_x)

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

    def test_cate_api_nonparam_without_ray(self):
        self._test_cate_api_nonparam(use_ray=False)

    @pytest.mark.ray
    def test_cate_api_nonparam_with_ray(self):
        try:
            ray.init(num_cpus=1)
            self._test_cate_api_nonparam(use_ray=True)
        finally:
            ray.shutdown()

    def _test_cate_api_nonparam(self, use_ray=False):
        """Test that we correctly implement the CATE API."""
        n = 20

        def make_random(is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d >= 0 else (n,)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b'], size=sz)
                    # ensure that we've got at least two of every element
                    _, counts = np.unique(arr, return_counts=True)
                    if len(counts) == 2 and counts.min() > 2:
                        return arr
            else:
                return np.random.normal(size=sz)

        for d_t in [1, -1]:
            for is_discrete in [True, False] if d_t <= 1 else [False]:
                for d_y in [3, 1, -1]:
                    for d_x in [2, None]:
                        for d_w in [2, None]:
                            W, X, Y, T = [make_random(is_discrete, d)
                                          for is_discrete, d in [(False, d_w),
                                                                 (False, d_x),
                                                                 (False, d_y),
                                                                 (is_discrete, d_t)]]

                            d_t_final = 1 if is_discrete else d_t

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
                                (n if d_x else 1) * (d_y if d_y > 0 else 1) * (d_t_final if d_t_final > 0 else 1), 6)

                            model_t = LogisticRegression() if is_discrete else WeightedLasso()

                            base_infs = [None, BootstrapInference(2)]
                            for est, multi, infs in [(NonParamDML(model_y=WeightedLasso(),
                                                                  model_t=model_t,
                                                                  model_final=WeightedLasso(),
                                                                  featurizer=None,
                                                                  discrete_treatment=is_discrete,
                                                                  use_ray=use_ray),
                                                      True,
                                                      base_infs),
                                                     (NonParamDML(model_y=WeightedLasso(),
                                                                  model_t=model_t,
                                                                  model_final=WeightedLasso(),
                                                                  featurizer=FunctionTransformer(),
                                                                  discrete_treatment=is_discrete,
                                                                  use_ray=use_ray),
                                                      True,
                                                      base_infs), ]:

                                if not (multi) and d_y > 1:
                                    continue

                                for inf in infs:
                                    with self.subTest(d_w=d_w, d_x=d_x, d_y=d_y, d_t=d_t,
                                                      is_discrete=is_discrete, est=est, inf=inf):
                                        if X is None:
                                            with pytest.raises(AttributeError):
                                                est.fit(Y, T, X=X, W=W, inference=inf)
                                            continue

                                        est.fit(Y, T, X=X, W=W, inference=inf)
                                        # make sure we can call the marginal_effect and effect methods
                                        const_marg_eff = est.const_marginal_effect(X)
                                        marg_eff = est.marginal_effect(T, X)
                                        self.assertEqual(shape(marg_eff), marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_eff), const_marginal_effect_shape)

                                        np.testing.assert_array_equal(
                                            marg_eff if d_x else marg_eff[0:1], const_marg_eff)

                                        T0 = np.full_like(T, 'a') if is_discrete else np.zeros_like(T)
                                        eff = est.effect(X, T0=T0, T1=T)
                                        self.assertEqual(shape(eff), effect_shape)

                                        if inf is not None:
                                            const_marg_eff_int = est.const_marginal_effect_interval(X)
                                            marg_eff_int = est.marginal_effect_interval(T, X)
                                            self.assertEqual(shape(marg_eff_int),
                                                             (2,) + marginal_effect_shape)
                                            self.assertEqual(shape(const_marg_eff_int),
                                                             (2,) + const_marginal_effect_shape)
                                            self.assertEqual(shape(est.effect_interval(X, T0=T0, T1=T)),
                                                             (2,) + effect_shape)
                                            if inf in ['auto', 'statsmodels', 'debiasedlasso', 'blb']:
                                                const_marg_effect_inf = est.const_marginal_effect_inference(X)
                                                T1 = np.full_like(T, 'b') if is_discrete else T
                                                effect_inf = est.effect_inference(X, T0=T0, T1=T1)
                                                marg_effect_inf = est.marginal_effect_inference(T, X)
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

                                        est.score(Y, T, X, W)

                                        # make sure we can call effect with implied scalar treatments, no matter the
                                        # dimensions of T, and also that we warn when there are multiple treatments
                                        if d_t > 1:
                                            cm = self.assertWarns(Warning)
                                        else:
                                            cm = ExitStack()  # ExitStack can be used as a "do nothing" ContextManager
                                        with cm:
                                            effect_shape2 = (n if d_x else 1,) + ((d_y,) if d_y > 0 else ())
                                            eff = est.effect(X) if not is_discrete else est.effect(X, T0='a', T1='b')
                                            self.assertEqual(shape(eff), effect_shape2)

    def test_bad_splits_discrete(self):
        """Test that we raise an error when some training splits in a crossfit fold don't contain all treatments."""
        Y = np.array([2, 3, 1, 3, 2, 1, 1, 1])
        T = np.array([2, 2, 1, 2, 1, 1, 1, 1])
        X = np.ones((8, 1))
        est = LinearDML(cv=[(np.arange(4, 8), np.arange(4))], discrete_treatment=True)
        with pytest.raises(AttributeError):
            est.fit(Y, T, X=X)
        Y = np.array([2, 3, 1, 3, 2, 1, 1, 1])
        T = np.array([2, 2, 1, 2, 2, 2, 2, 2])
        X = np.ones((8, 1))
        est = LinearDML(cv=[(np.arange(4, 8), np.arange(4))], discrete_treatment=True)
        with pytest.raises(AttributeError):
            est.fit(Y, T, X=X)

    def test_bad_treatment_nonparam(self):
        """Test that the non-parametric dml raises errors when treatment is not binary or single dimensional."""
        Y = np.array([2, 3, 1, 3, 2, 1, 1, 1])
        T = np.array([3, 2, 1, 2, 1, 2, 1, 3])
        X = np.ones((8, 1))
        est = NonParamDML(model_y=WeightedLasso(),
                          model_t=LogisticRegression(),
                          model_final=WeightedLasso(),
                          discrete_treatment=True)
        with pytest.raises(AttributeError):
            est.fit(Y, T, X=X)
        T = np.ones((8, 2))
        est = NonParamDML(model_y=WeightedLasso(),
                          model_t=LinearRegression(),
                          model_final=WeightedLasso(),
                          discrete_treatment=False)
        with pytest.raises(AttributeError):
            est.fit(Y, T, X=X)

    def test_access_to_internal_models(self):
        """Test that API related to accessing the nuisance models, cate_model and featurizer is working."""
        Y = np.array([2, 3, 1, 3, 2, 1, 1, 1])
        T = np.array([3, 2, 1, 2, 1, 2, 1, 3])
        X = np.ones((8, 1))
        est = DML(model_y=WeightedLasso(),
                  model_t=LogisticRegression(),
                  model_final=WeightedLasso(),
                  featurizer=PolynomialFeatures(degree=2, include_bias=False),
                  fit_cate_intercept=True,
                  discrete_treatment=True)
        est.fit(Y, T, X=X)
        assert isinstance(est.original_featurizer, PolynomialFeatures)
        assert isinstance(est.featurizer_, Pipeline)
        assert isinstance(est.model_cate, WeightedLasso)
        for mdl in est.models_y[0]:
            assert isinstance(mdl, WeightedLasso)
        for mdl in est.models_t[0]:
            assert isinstance(mdl, LogisticRegression)
        np.testing.assert_array_equal(est.cate_feature_names(['A']), ['A', 'A^2'])
        np.testing.assert_array_equal(est.cate_feature_names(), ['X0', 'X0^2'])
        est = DML(model_y=WeightedLasso(),
                  model_t=LogisticRegression(),
                  model_final=WeightedLasso(),
                  featurizer=None,
                  fit_cate_intercept=True,
                  discrete_treatment=True)
        est.fit(Y, T, X=X)
        assert est.original_featurizer is None
        assert isinstance(est.featurizer_, FunctionTransformer)
        assert isinstance(est.model_cate, WeightedLasso)
        for mdl in est.models_y[0]:
            assert isinstance(mdl, WeightedLasso)
        for mdl in est.models_t[0]:
            assert isinstance(mdl, LogisticRegression)
        np.testing.assert_array_equal(est.cate_feature_names(['A']), ['A'])

    def test_forest_dml_perf(self):
        """Testing accuracy of forest DML is reasonable."""
        np.random.seed(1234)
        n = 20000  # number of raw samples
        d = 10
        for _ in range(2):
            X = np.random.binomial(1, .5, size=(n, d))
            T = np.random.binomial(1, .5, size=(n,))

            def true_fn(x):
                return -1 + 2 * x[:, 0] + x[:, 1] * x[:, 2]

            y = true_fn(X) * T + X[:, 0] + (1 * X[:, 0] + 1) * np.random.normal(0, 1, size=(n,))

            XT = np.hstack([T.reshape(-1, 1), X])
            _X1, _X2, _y1, _y2, X1_sum, X2_sum, y1_sum, y2_sum, n1_sum, n2_sum, _var1_sum, _var2_sum = _summarize(XT, y)
            # We concatenate the two copies data
            X_sum = np.vstack([np.array(X1_sum)[:, 1:], np.array(X2_sum)[:, 1:]])
            T_sum = np.concatenate((np.array(X1_sum)[:, 0], np.array(X2_sum)[:, 0]))
            y_sum = np.concatenate((y1_sum, y2_sum))  # outcome
            n_sum = np.concatenate((n1_sum, n2_sum))  # number of summarized points
            for summarized, min_samples_leaf, tune in [(False, 20, False), (True, 1, False), (False, 20, True)]:
                est = CausalForestDML(model_y=GradientBoostingRegressor(n_estimators=30, min_samples_leaf=30),
                                      model_t=GradientBoostingClassifier(n_estimators=30, min_samples_leaf=30),
                                      discrete_treatment=True,
                                      cv=2,
                                      n_jobs=None,
                                      n_estimators=1000,
                                      max_samples=.4,
                                      min_samples_leaf=min_samples_leaf,
                                      min_impurity_decrease=0.001,
                                      verbose=0, min_var_fraction_leaf=.1,
                                      fit_intercept=False,
                                      random_state=12345)
                if tune:
                    est.tune(y, T, X=X[:, :4], W=X[:, 4:])
                if summarized:
                    est.fit(y_sum, T_sum, X=X_sum[:, :4], W=X_sum[:, 4:],
                            sample_weight=n_sum)
                else:
                    est.fit(y, T, X=X[:, :4], W=X[:, 4:])
                X_test = np.array(list(itertools.product([0, 1], repeat=4)))
                point = est.effect(X_test)
                truth = true_fn(X_test)
                lb, ub = est.effect_interval(X_test, alpha=.01)
                np.testing.assert_allclose(point, truth, rtol=0, atol=.3)
                np.testing.assert_array_less(lb - .01, truth)
                np.testing.assert_array_less(truth, ub + .01)

                est = CausalForestDML(model_y=GradientBoostingRegressor(n_estimators=50, min_samples_leaf=100),
                                      model_t=GradientBoostingRegressor(n_estimators=50, min_samples_leaf=100),
                                      discrete_treatment=False,
                                      cv=2,
                                      n_jobs=None,
                                      n_estimators=1000,
                                      max_samples=.4,
                                      min_samples_leaf=min_samples_leaf,
                                      min_impurity_decrease=0.001,
                                      verbose=0, min_var_fraction_leaf=.1,
                                      fit_intercept=False,
                                      random_state=12345)
                if tune:
                    with np.testing.assert_raises(ValueError):
                        est.tune(y, T, X=X[:, :4], W=X[:, 4:], params={'discrete_treatment': [True, False]})
                    est.tune(y, T, X=X[:, :4], W=X[:, 4:], params={'max_samples': [.1, .3]})
                if summarized:
                    est.fit(y_sum, T_sum, X=X_sum[:, :4], W=X_sum[:, 4:],
                            sample_weight=n_sum)
                else:
                    est.fit(y, T, X=X[:, :4], W=X[:, 4:])
                X_test = np.array(list(itertools.product([0, 1], repeat=4)))
                point = est.effect(X_test)
                truth = true_fn(X_test)
                lb, ub = est.effect_interval(X_test, alpha=.01)
                np.testing.assert_allclose(point, truth, rtol=0, atol=.3)
                np.testing.assert_array_less(lb - .01, truth)
                np.testing.assert_array_less(truth, ub + .01)


    def test_forest_dml_score_fns(self):
        np.random.seed(1234)
        n = 20000  # number of raw samples
        d = 10

        Z = np.random.binomial(1, .5, size=(n, d))
        T = np.random.binomial(1, .5, size=(n,))

        def true_fn(x):
            return -1 + 2 * x[:, 0] + x[:, 1] * x[:, 2]

        y = true_fn(Z) * T + Z[:, 0] + (1 * Z[:, 0] + 1) * np.random.normal(0, 1, size=(n,))
        X = Z[:, :4]
        W = Z[:, 4:]

        est = CausalForestDML(model_y=GradientBoostingRegressor(n_estimators=30, min_samples_leaf=30),
                              model_t=GradientBoostingClassifier(n_estimators=30, min_samples_leaf=30),
                              discrete_treatment=True,
                              cv=2,
                              n_jobs=None,
                              n_estimators=1000,
                              max_samples=.4,
                              min_samples_leaf=10,
                              min_impurity_decrease=0.001,
                              verbose=0, min_var_fraction_leaf=.1,
                              fit_intercept=False,
                              random_state=12345)

        est.fit(y, T, X=X, W=W)

        s1 = est.score(Y=y,T=T,X=X, W=W, scoring='mean_squared_error')
        s2 = est.score(Y=y,T=T,X=X, W=W)
        assert s1 == s2
        np.testing.assert_allclose(s1, 2.50, rtol=0, atol=.01)
        s3 = est.score(Y=y, T=T, X=X, W=W, scoring='mean_absolute_error')
        np.testing.assert_allclose(s3, 1.19, rtol=0, atol=.01)
        s4 = est.score(Y=y, T=T, X=X, W=W, scoring='r2')
        np.testing.assert_allclose(s4, 0.113, rtol=0, atol=.001)
        s5 = est.score(Y=y, T=T, X=X, W=W, scoring=pearsonr)
        np.testing.assert_allclose(s5[0], 0.337, rtol=0, atol=0.005 )

        sn1 = est.score_nuisances(Y=y, T=T, X=X, W=W,
                                  t_scoring='mean_squared_error',
                                  y_scoring='mean_squared_error')
        np.testing.assert_allclose(sn1['Y_mean_squared_error'], [2.8,2.8], rtol=0, atol=.1)
        np.testing.assert_allclose(sn1['T_mean_squared_error'], [1.5,1.5], rtol=0, atol=.1)

        sn2 = est.score_nuisances(Y=y, T=T, X=X, W=W,
                                  t_scoring='mean_absolute_error',
                                  y_scoring='mean_absolute_error')
        np.testing.assert_allclose(sn2['Y_mean_absolute_error'], [1.3,1.3], rtol=0, atol=.1)
        np.testing.assert_allclose(sn2['T_mean_absolute_error'], [1.0,1.0], rtol=0, atol=.1)

        sn3 = est.score_nuisances(Y=y, T=T, X=X, W=W,
                                  t_scoring='r2',
                                  y_scoring='r2')
        np.testing.assert_allclose(sn3['Y_r2'], [0.27,0.27], rtol=0, atol=.005)
        np.testing.assert_allclose(sn3['T_r2'], [-5.1,-5.1], rtol=0, atol=0.25)

        sn4 = est.score_nuisances(Y=y, T=T, X=X, W=W,
                                  t_scoring=pearsonr,
                                  y_scoring=pearsonr)
        # Ignoring the p-values returned with the score
        y_pearsonr = [sn4[s][i][0] for s in sn4 if s.startswith('Y_') for i in range(len(sn4[s])) ]
        t_pearsonr = [sn4[s][i][0] for s in sn4 if s.startswith('T_') for i in range(len(sn4[s])) ]
        np.testing.assert_allclose(y_pearsonr, [0.52, 0.52], rtol=0, atol=.01)
        np.testing.assert_allclose(t_pearsonr, [.035, .035], rtol=0, atol=0.005)

        # T is binary, and can be used to check binary eval functions
        sn5 = est.score_nuisances(Y=y, T=T, X=X, W=W, t_scoring='roc_auc')
        np.testing.assert_allclose(sn5['T_roc_auc'], [0.52,0.52], rtol=0, atol=.01)

        sn6 = est.score_nuisances(Y=y, T=T, X=X, W=W, t_scoring='log_loss')
        np.testing.assert_allclose(sn6['T_log_loss'], [17.4,17.4], rtol=0, atol=0.1)

    def test_aaforest_pandas(self):
        """Test that we can use CausalForest with pandas inputs."""
        df = pd.DataFrame({'a': np.random.normal(size=500),
                           'b': np.random.normal(size=500),
                           'c': np.random.choice([0, 1], size=500),
                           'd': np.random.choice(['a', 'b', 'c'], size=500)})

        est = CausalForestDML(discrete_treatment=True)
        est.tune(Y=df['a'], T=df['d'], X=df[['b', 'c']])
        est.fit(Y=df['a'], T=df['d'], X=df[['b', 'c']])

        # make sure we can get out post-fit stuff
        est.ate_
        est.ate__inference()
        est.effect(df[['b', 'c']], T0=pd.Series(['b'] * 500), T1=pd.Series(['c'] * 500))

    def test_cfdml_ate_inference(self):
        np.random.seed(1234)
        n = 20000  # number of raw samples
        d = 10
        for it in range(2):
            X = np.random.binomial(1, .5, size=(n, d))
            T = np.random.binomial(2, .5, size=(n,))

            def true_fn(x):
                return -1 + 2 * x[:, 0] + x[:, 1] * x[:, 2]

            y = true_fn(X) * (T == 1) + true_fn(X) * (T == 2) + X[:, 0] + np.random.normal(0, 1, size=(n,))
            est = CausalForestDML(discrete_treatment=True,
                                  featurizer=PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                                  cv=2,
                                  n_estimators=400,
                                  max_samples=.45,
                                  max_depth=3,
                                  min_balancedness_tol=.2,
                                  n_jobs=None,
                                  min_samples_leaf=40,
                                  min_impurity_decrease=0.001,
                                  verbose=0, min_var_fraction_leaf=.1,
                                  fit_intercept=True,
                                  random_state=125)
            est.fit(np.hstack([y.reshape(-1, 1), y.reshape(-1, 1)]), T, X=X[:, :4], W=X[:, 4:])
            X_test = np.array(list(itertools.product([0, 1], repeat=4)))
            point = est.effect(X_test)
            truth = np.hstack([true_fn(X_test).reshape(-1, 1), true_fn(X_test).reshape(-1, 1)])
            lb, ub = est.effect_interval(X_test, alpha=.01)
            np.testing.assert_allclose(point, truth, rtol=0, atol=.4)
            np.testing.assert_array_less(lb - .04, truth)
            np.testing.assert_array_less(truth, ub + .04)
            mean_truth = np.mean(true_fn(X))
            np.testing.assert_allclose(est.ate_, mean_truth, rtol=0, atol=.06)
            np.testing.assert_allclose(est.att_(T=1), mean_truth, rtol=0, atol=.06)
            np.testing.assert_allclose(est.att_(T=2), mean_truth, rtol=0, atol=.06)
            inf_sum = est.ate__inference().summary_frame(treatment_names=['drugA', 'drugB'])
            np.testing.assert_allclose(inf_sum.point_estimate, mean_truth, rtol=0, atol=.06)
            np.testing.assert_array_less(inf_sum.ci_lower - .03, mean_truth)
            np.testing.assert_array_less(mean_truth - .03, inf_sum.ci_upper)
            inf_sum = est.att__inference(T=1).summary_frame(alpha=0.01, treatment_names=['drugA', 'drugB'])
            np.testing.assert_allclose(inf_sum.point_estimate, mean_truth, rtol=0, atol=.06)
            np.testing.assert_array_less(inf_sum.ci_lower - .03, mean_truth)
            np.testing.assert_array_less(mean_truth - .03, inf_sum.ci_upper)
            tables = est.summary().tables
            for t in range(4):
                np.testing.assert_allclose(np.array(tables[t].data[1:])[:, 1].astype(float),
                                           mean_truth, rtol=0, atol=.06)

            if it == 0:
                est.fit(y[:100], T[:100], X=X[:100, :4], W=X[:100, 4:], cache_values=True)
                np.testing.assert_equal(len(est.summary().tables), 7)
                np.testing.assert_equal(len(est[0][0].feature_importances_), 10)
                np.testing.assert_equal(len(est), est.n_estimators)
                np.testing.assert_equal(len([tree[0].feature_importances_ for tree in est]), est.n_estimators)
                with np.testing.assert_raises(ValueError):
                    est.model_final = LinearRegression()
                assert isinstance(est.model_final, MultiOutputGRF)
                est.featurizer = None
                est.fit(y[:100], T[:100], X=X[:100, :4], W=X[:100, 4:], cache_values=True)
                np.testing.assert_equal(est.shap_values(X[:2, :4])['Y0']['T0_1'].values.shape, (2, 4))
                with np.testing.assert_raises(TypeError):
                    est.fit(y[:100], T[:100], X=X[:100, :4], W=X[:100, 4:], sample_var=np.ones(100))
                with np.testing.assert_raises(ValueError):
                    est.fit(y[:100], T[:100], X=None, W=X[:100, 4:])
                for est in [CausalForestDML(discrete_treatment=False,
                                            n_estimators=16),
                            CausalForestDML(model_y=GradientBoostingRegressor(n_estimators=30, min_samples_leaf=30),
                                            model_t=GradientBoostingClassifier(n_estimators=30, min_samples_leaf=30),
                                            discrete_treatment=True,
                                            drate=False,
                                            n_estimators=16)]:
                    est.fit(y[:100], T[:100], X=X[:100, :4], W=X[:100, 4:], cache_values=True)
                    with np.testing.assert_raises(AttributeError):
                        est.ate_
                    with np.testing.assert_raises(AttributeError):
                        est.ate__inference()
                    with np.testing.assert_raises(AttributeError):
                        est.ate_stderr_
                    with np.testing.assert_raises(AttributeError):
                        est.att_(T=1)
                    with np.testing.assert_raises(AttributeError):
                        est.att__inference(T=1)
                    with np.testing.assert_raises(AttributeError):
                        est.att_stderr_(T=1)
                    np.testing.assert_equal(len(est.summary().tables), 3)

    def test_can_use_vectors(self):
        """Test that we can pass vectors for T and Y (not only 2-dimensional arrays)."""
        dmls = [
            LinearDML(model_y=LinearRegression(), model_t=LinearRegression(), fit_cate_intercept=False),
            SparseLinearDML(model_y=LinearRegression(), model_t=LinearRegression(), fit_cate_intercept=False)
        ]
        for dml in dmls:
            dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), X=np.ones((6, 1)))
            self.assertAlmostEqual(dml.coef_.reshape(())[()], 1)
            score = dml.score(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))
            self.assertAlmostEqual(score, 0)

    def test_can_use_sample_weights(self):
        """Test that we can pass sample weights to an estimator."""
        dmls = [
            LinearDML(model_y=LinearRegression(), model_t='auto', fit_cate_intercept=False),
            SparseLinearDML(model_y=LinearRegression(), model_t='auto', fit_cate_intercept=False)
        ]
        for dml in dmls:
            dml.fit(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]),
                    X=np.ones((12, 1)), sample_weight=np.ones((12, )))
            self.assertAlmostEqual(dml.coef_.reshape(())[()], 1)

    def test_discrete_treatments(self):
        """Test that we can use discrete treatments."""
        dmls = [
            LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(C=1000),
                      fit_cate_intercept=False, discrete_treatment=True),
            SparseLinearDML(model_y=LinearRegression(), model_t=LogisticRegression(C=1000),
                            fit_cate_intercept=False, discrete_treatment=True)
        ]
        for dml in dmls:
            # create a simple artificial setup where effect of moving from treatment
            #     1 -> 2 is 2,
            #     1 -> 3 is 1, and
            #     2 -> 3 is -1 (necessarily, by composing the previous two effects)
            # Using an uneven number of examples from different classes,
            # and having the treatments in non-lexicographic order,
            # Should rule out some basic issues.
            dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array([3, 2, 1, 2, 3, 1, 1, 1]), X=np.ones((8, 1)))
            np.testing.assert_almost_equal(
                dml.effect(
                    np.ones((9, 1)),
                    T0=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                    T1=np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
                ),
                [0, 2, 1, -2, 0, -1, -1, 1, 0],
                decimal=2)
            dml.score(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array([3, 2, 1, 2, 3, 1, 1, 1]), np.ones((8, 1)))

    def _test_can_custom_splitter(self, use_ray=False):
        # test that we can fit with a KFold instance
        dml = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(C=1000),
                        discrete_treatment=True, cv=KFold(), use_ray=use_ray)
        dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), X=np.ones((6, 1)))
        dml.score(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))

        # test that we can fit with a train/test iterable
        dml = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(C=1000),
                        discrete_treatment=True, cv=[([0, 1, 2], [3, 4, 5])], use_ray=use_ray)
        dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), X=np.ones((6, 1)))
        dml.score(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))

    @pytest.mark.ray
    def test_can_use_custom_splitter_with_ray(self):
        try:
            ray.init(num_cpus=1)
            self._test_can_custom_splitter(use_ray=True)
        finally:
            ray.shutdown()

    def test_can_use_custom_splitter_without_ray(self):
        self._test_can_custom_splitter(use_ray=False)

    def test_can_use_featurizer(self):
        """Test that we can use a featurizer, and that fit is only called during training."""
        # predetermined splits ensure that all features are seen in each split
        splits = ([0, 2, 3, 6, 8, 11, 13, 15, 16],
                  [1, 4, 5, 7, 9, 10, 12, 14, 17])

        dml = LinearDML(model_y=LinearRegression(), model_t=LinearRegression(),
                        fit_cate_intercept=False, featurizer=OneHotEncoder(sparse_output=False),
                        cv=[splits, splits[::-1]])

        T = np.tile([1, 2, 3], 6)
        Y = np.array([1, 2, 3, 1, 2, 3])
        Y = np.concatenate([Y, 0 * Y, -Y])
        X = np.repeat([[7, 8, 9]], 6, axis=1).T

        dml.fit(Y, T, X=X)

        # because there is one fewer unique element in the test set, fit_transform would return the wrong number of fts
        X_test = np.array([[7, 8]]).T

        np.testing.assert_equal(dml.effect(X_test)[::-1], dml.effect(X_test[::-1]))
        eff_int = np.array(dml.effect_interval(X_test))
        eff_int_rev = np.array(dml.effect_interval(X_test[::-1]))
        np.testing.assert_equal(eff_int[:, ::-1], eff_int_rev)

        eff_int = np.array(dml.const_marginal_effect_interval(X_test))
        eff_int_rev = np.array(dml.const_marginal_effect_interval(X_test[::-1]))
        np.testing.assert_equal(eff_int[:, ::-1], eff_int_rev)

    def test_can_use_statsmodel_inference(self):
        """Test that we can use statsmodels to generate confidence intervals."""
        dml = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(C=1000),
                        discrete_treatment=True)
        dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array(
            [3, 2, 1, 2, 3, 1, 1, 1]), X=np.ones((8, 1)))
        interval = dml.effect_interval(np.ones((9, 1)),
                                       T0=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                                       T1=np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
                                       alpha=0.05)
        point = dml.effect(np.ones((9, 1)),
                           T0=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                           T1=np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]))
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        assert (lo < hi).any()  # for at least some of the examples, the CI should have nonzero width

        interval = dml.const_marginal_effect_interval(np.ones((9, 1)), alpha=0.05)
        point = dml.const_marginal_effect(np.ones((9, 1)))
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        assert (lo < hi).any()  # for at least some of the examples, the CI should have nonzero width

        interval = dml.coef__interval(alpha=0.05)
        point = dml.coef_
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        assert (lo < hi).any()  # for at least some of the examples, the CI should have nonzero width

        interval = dml.intercept__interval(alpha=0.05)
        point = dml.intercept_
        assert len(interval) == 2
        lo, hi = interval
        assert (lo <= point).all()
        assert (point <= hi).all()
        assert (lo < hi).any()  # for at least some of the examples, the CI should have nonzero width

    def test_ignores_final_intercept(self):
        """Test that final model intercepts are ignored (with a warning)."""
        class InterceptModel:
            def fit(self, Y, X):
                pass

            def predict(self, X):
                return X + 1

            def prediction_stderr(self, X):
                return np.zeros(X.shape[0])

        # (incorrectly) use a final model with an intercept
        dml = DML(model_y=LinearRegression(), model_t=LinearRegression(),
                  model_final=InterceptModel())
        # Because final model is fixed, actual values of T and Y don't matter
        t = np.random.normal(size=100)
        y = np.random.normal(size=100)
        with self.assertWarns(Warning):  # we should warn whenever there's an intercept
            dml.fit(y, t)
        assert dml.const_marginal_effect() == 1  # coefficient on X in InterceptModel is 1
        assert dml.const_marginal_effect_inference().point_estimate == 1
        assert dml.const_marginal_effect_inference().conf_int() == (1, 1)
        assert dml.const_marginal_effect_interval() == (1, 1)
        assert dml.effect() == 1
        assert dml.effect_inference().point_estimate == 1
        assert dml.effect_inference().conf_int() == (1, 1)
        assert dml.effect_interval() == (1, 1)
        assert dml.marginal_effect(1) == 1  # coefficient on X in InterceptModel is 1
        assert dml.marginal_effect_inference(1).point_estimate == 1
        assert dml.marginal_effect_inference(1).conf_int() == (1, 1)
        assert dml.marginal_effect_interval(1) == (1, 1)

    def test_sparse(self):
        # Ensure reproducibility
        np.random.seed(123)
        n_p = np.random.randint(2, 5)  # 2 to 4 products
        d_w = np.random.randint(0, 5)  # random number of covariates
        min_n = np.ceil(2 + d_w * (1 + (d_w + 1) / n_p))  # minimum number of rows per product
        n_r = np.random.randint(min_n, min_n + 3)
        with self.subTest(n_p=n_p, d_w=d_w, n_r=n_r):
            TestDML._test_sparse(n_p, d_w, n_r)

    def test_linear_sparse(self):
        """SparseDML test with a sparse DGP."""
        # Sparse DGP
        np.random.seed(123)
        n_x = 50
        n_nonzero = 5
        n_w = 5
        n = 1000
        # Treatment effect coef
        a = np.zeros(n_x)
        nonzero_idx = np.random.choice(n_x, size=n_nonzero, replace=False)
        a[nonzero_idx] = 1
        # Other coefs
        b = np.zeros(n_x + n_w)
        g = np.zeros(n_x + n_w)
        b_nonzero = np.random.choice(n_x + n_w, size=n_nonzero, replace=False)
        g_nonzero = np.random.choice(n_x + n_w, size=n_nonzero, replace=False)
        b[b_nonzero] = 1
        g[g_nonzero] = 1
        # Features and controls
        x = np.random.normal(size=(n, n_x))
        w = np.random.normal(size=(n, n_w))
        xw = np.hstack([x, w])
        err_T = np.random.normal(size=n)
        T = xw @ b + err_T
        err_Y = np.random.normal(size=n, scale=0.5)
        Y = T * (x @ a) + xw @ g + err_Y
        # Test sparse estimator
        # --> test coef_, intercept_
        # with this DGP, since T depends linearly on X, Y depends on X quadratically
        # so we should use a quadratic featurizer
        sparse_dml = SparseLinearDML(fit_cate_intercept=False, model_y=Pipeline([('poly', PolynomialFeatures(2)),
                                                                                 ('lr', LassoCV())]))
        sparse_dml.fit(Y, T, X=x, W=w)
        np.testing.assert_allclose(a, sparse_dml.coef_, atol=2e-1)
        with pytest.raises(AttributeError):
            sparse_dml.intercept_
        # --> test treatment effects
        # Restrict x_test to vectors of norm < 1
        x_test = np.random.uniform(size=(10, n_x))
        true_eff = (x_test @ a)
        eff = sparse_dml.effect(x_test, T0=0, T1=1)
        np.testing.assert_allclose(true_eff, eff, atol=0.5)
        # --> check inference
        y_lower, y_upper = sparse_dml.effect_interval(x_test, T0=0, T1=1)
        in_CI = ((y_lower < true_eff) & (true_eff < y_upper))
        # Check that a majority of true effects lie in the 5-95% CI
        self.assertTrue(in_CI.mean() > 0.8)

    @staticmethod
    def _generate_recoverable_errors(a_X, X, a_W=None, W=None, featurizer=None):
        """Return error vectors e_t and e_y such that OLS can recover the true coefficients from both stages."""
        if W is None:
            W = np.empty((shape(X)[0], 0))
        if a_W is None:
            a_W = np.zeros((shape(W)[1],))
        # to correctly recover coefficients for T via OLS, we need e_t to be orthogonal to [W;X]
        WX = hstack([W, X])
        e_t = rand_sol(WX.T, np.zeros((shape(WX)[1],)))

        # to correctly recover coefficients for Y via OLS, we need ([X; W]⊗[1; ϕ(X); W])⁺ e_y =
        #                                                          -([X; W]⊗[1; ϕ(X); W])⁺ ((ϕ(X)⊗e_t)a_X+(W⊗e_t)a_W)
        # then, to correctly recover a in the third stage, we additionally need (ϕ(X)⊗e_t)ᵀ e_y = 0

        ϕ = featurizer.fit_transform(X) if featurizer is not None else X

        v_X = cross_product(ϕ, e_t)
        v_W = cross_product(W, e_t)

        M = np.linalg.pinv(cross_product(WX, hstack([np.ones((shape(WX)[0], 1)), ϕ, W])))
        e_y = rand_sol(vstack([M, v_X.T]), vstack([-M @ (v_X @ a_X + v_W @ a_W), np.zeros((shape(v_X)[1],))]))

        return e_t, e_y

    # sparse test case: heterogeneous effect by product
    @staticmethod
    def _test_sparse(n_p, d_w, n_r):
        # need at least as many rows in e_y as there are distinct columns
        # in [X;X⊗W;W⊗W;X⊗e_t] to find a solution for e_t
        assert n_p * n_r >= 2 * n_p + n_p * d_w + d_w * (d_w + 1) / 2
        a = np.random.normal(size=(n_p,))  # one effect per product
        n = n_p * n_r * 100
        p = np.tile(range(n_p), n_r * 100)  # product id

        b = np.random.normal(size=(d_w + n_p,))
        g = np.random.normal(size=(d_w + n_p,))

        x = np.empty((2 * n, n_p))  # product dummies
        w = np.empty((2 * n, d_w))
        y = np.empty(2 * n)
        t = np.empty(2 * n)

        for fold in range(0, 2):
            x_f = OneHotEncoder().fit_transform(np.reshape(p, (-1, 1))).toarray()
            w_f = np.random.normal(size=(n, d_w))
            xw_f = hstack([x_f, w_f])
            e_t_f, e_y_f = TestDML._generate_recoverable_errors(a, x_f, W=w_f)

            t_f = xw_f @ b + e_t_f
            y_f = t_f * np.choose(p, a) + xw_f @ g + e_y_f

            x[fold * n:(fold + 1) * n, :] = x_f
            w[fold * n:(fold + 1) * n, :] = w_f
            y[fold * n:(fold + 1) * n] = y_f
            t[fold * n:(fold + 1) * n] = t_f

        # we have quadratic terms in y, so we need to pipeline with a quadratic featurizer
        dml = SparseLinearDML(model_y=Pipeline([('poly', PolynomialFeatures(2)),
                                                ('lr', LinearRegression(fit_intercept=False))]),
                              model_t=LinearRegression(fit_intercept=False),
                              fit_cate_intercept=False)
        dml.fit(y, t, X=x, W=w)
        np.testing.assert_allclose(a, dml.coef_.reshape(-1), atol=1e-1)
        eff = reshape(t * np.choose(np.tile(p, 2), a), (-1,))
        np.testing.assert_allclose(eff, dml.effect(x, T0=0, T1=t), atol=1e-1)

    def _test_nuisance_scores(self, use_ray=False):
        X = np.random.choice(np.arange(5), size=(100, 3))
        y = np.random.normal(size=(100,))
        T = np.random.choice(np.arange(3), size=(100, 2))
        W = np.random.normal(size=(100, 2))
        for mc_iters in [1, 2, 3]:
            for cv in [1, 2, 3]:
                est = LinearDML(cv=cv, mc_iters=mc_iters, use_ray=use_ray)
                est.fit(y, T, X=X, W=W)
                assert len(est.nuisance_scores_t) == len(est.nuisance_scores_y) == mc_iters
                assert len(est.nuisance_scores_t[0]) == len(est.nuisance_scores_y[0]) == cv
                est.score(y, T, X=X, W=W)

    @pytest.mark.ray
    def test_nuisance_scores_with_ray(self):
        try:
            ray.init(num_cpus=1)
            self._test_nuisance_scores(use_ray=True)
        finally:
            ray.shutdown()

    def test_nuisance_scores_without_ray(self):
        self._test_nuisance_scores(use_ray=False)

    @pytest.mark.ray
    def test_compare_nuisance_with_ray_vs_without_ray(self):
        X = np.random.choice(np.arange(5), size=(100, 3))
        y = np.random.normal(size=(100,))
        T = np.random.choice(np.arange(3), size=(100, 2))
        W = np.random.normal(size=(100, 2))
        try:
            ray.init(num_cpus=1)
            for mc_iters in [1, 2, 3]:
                for cv in [1, 2, 3]:
                    est_with_ray = LinearDML(cv=cv, mc_iters=mc_iters, use_ray=True)
                    est_without_ray = LinearDML(cv=cv, mc_iters=mc_iters, use_ray=False)

                    est_with_ray.fit(y, T, X=X, W=W)
                    est_without_ray.fit(y, T, X=X, W=W)

                    # Compare results with and without Ray
                    assert len(est_with_ray.nuisance_scores_t) == len(est_without_ray.nuisance_scores_t) == mc_iters
                    assert len(est_with_ray.nuisance_scores_y) == len(est_without_ray.nuisance_scores_y) == mc_iters
                    assert len(est_with_ray.nuisance_scores_t[0]) == len(est_without_ray.nuisance_scores_t[0]) == cv
                    assert len(est_with_ray.nuisance_scores_y[0]) == len(est_without_ray.nuisance_scores_y[0]) == cv
        finally:
            ray.shutdown()

    def test_categories(self):
        dmls = [LinearDML, SparseLinearDML]
        for ctor in dmls:
            dml1 = ctor(model_y=LinearRegression(), model_t=LogisticRegression(C=1000),
                        fit_cate_intercept=False, discrete_treatment=True, random_state=123)
            dml2 = ctor(model_y=LinearRegression(), model_t=LogisticRegression(C=1000),
                        fit_cate_intercept=False, discrete_treatment=True, categories=['c', 'b', 'a'],
                        random_state=123)

            # create a simple artificial setup where effect of moving from treatment
            #     a -> b is 2,
            #     a -> c is 1, and
            #     b -> c is -1 (necessarily, by composing the previous two effects)
            # Using an uneven number of examples from different classes,
            # and having the treatments in non-lexicographic order,
            # should rule out some basic issues.

            # Note that explicitly specifying the dtype as object is necessary until
            # there's a fix for https://github.com/scikit-learn/scikit-learn/issues/15616

            for dml in [dml1, dml2]:
                dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]),
                        np.array(['c', 'b', 'a', 'b', 'c', 'a', 'a', 'a'], dtype='object'), X=np.ones((8, 1)))

            # estimated effects should be identical when treatment is explicitly given
            np.testing.assert_almost_equal(
                dml1.effect(
                    np.ones((9, 1)),
                    T0=np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], dtype='object'),
                    T1=np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'], dtype='object')
                ),
                dml2.effect(
                    np.ones((9, 1)),
                    T0=np.array(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'], dtype='object'),
                    T1=np.array(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'], dtype='object')
                ),
                decimal=4)

            # but const_marginal_effect should be reordered based on the explicit cagetories
            cme1 = dml1.const_marginal_effect(np.ones((1, 1))).reshape(-1)
            cme2 = dml2.const_marginal_effect(np.ones((1, 1))).reshape(-1)
            self.assertAlmostEqual(cme1[1], -cme2[1], places=3)  # 1->3 in original ordering; 3->1 in new ordering
            # 1-> 2 in original ordering; combination of 3->1 and 3->2
            self.assertAlmostEqual(cme1[0], -cme2[1] + cme2[0], places=3)

    def test_groups(self):

        groups = [1, 2, 3, 4, 5, 6] * 10
        t = groups
        y = groups

        n_copies = {i: 10 for i in [1, 2, 3, 4, 5, 6]}

        # test outer grouping
        # with 2 folds, we should get exactly 3 groups per split, each with 10 copies of the y or t value
        est = LinearDML(model_y=GroupingModel(LinearRegression(), 60, (3, 3), n_copies),
                        model_t=GroupingModel(LinearRegression(), 60, (3, 3), n_copies))
        est.fit(y, t, groups=groups)

        # test nested grouping

        # with 2-fold outer and 2-fold inner grouping, and six total groups,
        # should get 1 or 2 groups per split

        est = LinearDML(model_y=NestedModel(LassoCV(cv=2), 60, (1, 2), n_copies),
                        model_t=NestedModel(LassoCV(cv=2), 60, (1, 2), n_copies))
        est.fit(y, t, groups=groups)

    def test_treatment_names(self):
        Y = np.random.normal(size=(100, 1))
        T = np.random.binomial(n=1, p=0.5, size=(100, 1))
        X = np.random.normal(size=(100, 3))

        Ts = [
            T,
            pd.DataFrame(T, columns=[0])
        ]

        init_args_list = [
            {'discrete_treatment': True},
            {'treatment_featurizer': PolynomialFeatures(degree=2, include_bias=False)},
            {'treatment_featurizer': FunctionTransformer(lambda x: np.hstack([x, np.sqrt(x)]))},
        ]

        for T in Ts:
            for init_args in init_args_list:
                est = LinearDML(**init_args).fit(Y=Y, T=T, X=X)
                t_name = '0' if isinstance(T, pd.DataFrame) else 'T0'  # default treatment name
                postfixes = ['_1'] if 'discrete_treatment' in init_args else ['', '^2']  # transformer postfixes

                # Try default, integer, and new user-passed treatment name
                for new_treatment_name in [None, [999], ['NewTreatmentName']]:

                    # FunctionTransformers are agnostic to passed treatment names
                    if isinstance(init_args.get('treatment_featurizer'), FunctionTransformer):
                        assert (est.cate_treatment_names(new_treatment_name) == ['feat(T)0', 'feat(T)1'])

                    # Expected treatment names are the sums of user-passed prefixes and transformer-specific postfixes
                    else:
                        expected_prefix = str(new_treatment_name[0]) if new_treatment_name is not None else t_name
                        assert (est.cate_treatment_names(new_treatment_name) == [
                                expected_prefix + postfix for postfix in postfixes])

    def test_causal_forest_tune_with_discrete_outcome_and_treatment(self):
        np.random.seed(1234)
        n = 1000
        treatment = np.repeat([0, 1], n // 2)
        covariate = np.resize([0, 1], n)
        outcome = ((treatment == 1) & (covariate == 1)).astype(int)
        X = covariate.reshape(-1, 1)
        Y = outcome
        T = treatment

        est = CausalForestDML(
            model_y=GradientBoostingClassifier(),
            model_t=GradientBoostingClassifier(),
            discrete_outcome=True,
            discrete_treatment=True
        )
        est.tune(Y=Y, T=T, X=X)
        est.fit(Y=Y, T=T, X=X)
