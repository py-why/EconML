# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

from econml.iv.dr import (DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV, IntentToTreatDRIV, LinearIntentToTreatDRIV,)
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.utilities import shape
from econml.tests.utilities import GroupingModel

import itertools
import numpy as np
import pytest
import pickle
from scipy import special
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
import unittest

try:
    import ray

    ray_installed = True
except ImportError:
    ray_installed = False


@pytest.mark.cate_api
class TestDRIV(unittest.TestCase):
    def _test_cate_api(self, use_ray=False):
        def const_marg_eff_shape(n, d_x, binary_T):
            """Constant marginal effect shape."""
            return (n if d_x else 1,) + ((1,) if binary_T else ())

        def marg_eff_shape(n, binary_T):
            """Marginal effect shape."""
            return (n,) + ((1,) if binary_T else ())

        def eff_shape(n, d_x):
            """Effect shape."""
            return (n if d_x else 1,)

        n = 500
        y = np.random.normal(size=(n,))

        # parameter combinations to test
        for d_w, d_x, binary_T, binary_Z, projection, fit_cov_directly, featurizer\
            in itertools.product(
                [None, 10],     # d_w
                [None, 3],      # d_x
                [True, False],  # binary_T
                [True, False],  # binary_Z
                [True, False],  # projection
                [True, False],  # fit_cov_directly
                [None, PolynomialFeatures(degree=2, include_bias=False), ],):    # featurizer

            if d_w is None:
                W = None
            else:
                W = np.random.normal(size=(n, d_w))

            if d_x is None:
                X = None
            else:
                X = np.random.normal(size=(n, d_x))

            if binary_T:
                T = np.random.choice(["a", "b"], size=(n,))
            else:
                T = np.random.normal(size=(n,))

            if binary_Z:
                Z = np.random.choice(["c", "d"], size=(n,))
            else:
                Z = np.random.normal(size=(n,))

            est_list = [
                # we're running a lot of tests, so use fixed models instead of model selection
                DRIV(
                    model_y_xw=LinearRegression(),
                    model_t_xw=LogisticRegression() if binary_T else LinearRegression(),
                    model_tz_xw=LogisticRegression() if binary_T and binary_Z and not (
                        projection or fit_cov_directly) else LinearRegression(),
                    model_t_xwz="auto" if not projection else LogisticRegression() if binary_T else LinearRegression(),
                    model_z_xw="auto" if projection else LogisticRegression() if binary_Z else LinearRegression(),
                    flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
                    model_final=StatsModelsLinearRegression(
                        fit_intercept=False
                    ),
                    fit_cate_intercept=True,
                    projection=projection,
                    fit_cov_directly=fit_cov_directly,
                    discrete_instrument=binary_Z,
                    discrete_treatment=binary_T,
                    featurizer=featurizer,
                    use_ray=use_ray,
                ),
                LinearDRIV(
                    model_y_xw=LinearRegression(),
                    model_t_xw=LogisticRegression() if binary_T else LinearRegression(),
                    model_tz_xw=LogisticRegression() if binary_T and binary_Z and not (
                        projection or fit_cov_directly) else LinearRegression(),
                    model_t_xwz="auto" if not projection else LogisticRegression() if binary_T else LinearRegression(),
                    model_z_xw="auto" if projection else LogisticRegression() if binary_Z else LinearRegression(),
                    flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
                    fit_cate_intercept=True,
                    projection=projection,
                    fit_cov_directly=fit_cov_directly,
                    discrete_instrument=binary_Z,
                    discrete_treatment=binary_T,
                    featurizer=featurizer,
                    use_ray=use_ray,
                ),
                SparseLinearDRIV(
                    model_y_xw=LinearRegression(),
                    model_t_xw=LogisticRegression() if binary_T else LinearRegression(),
                    model_tz_xw=LogisticRegression() if binary_T and binary_Z and not (
                        projection or fit_cov_directly) else LinearRegression(),
                    model_t_xwz="auto" if not projection else LogisticRegression() if binary_T else LinearRegression(),
                    model_z_xw="auto" if projection else LogisticRegression() if binary_Z else LinearRegression(),
                    flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
                    fit_cate_intercept=True,
                    projection=projection,
                    fit_cov_directly=fit_cov_directly,
                    discrete_instrument=binary_Z,
                    discrete_treatment=binary_T,
                    featurizer=featurizer,
                    use_ray=use_ray,
                ),
                ForestDRIV(
                    model_y_xw=LinearRegression(),
                    model_t_xw=LogisticRegression() if binary_T else LinearRegression(),
                    model_tz_xw=LogisticRegression() if binary_T and binary_Z and not (
                        projection or fit_cov_directly) else LinearRegression(),
                    model_t_xwz="auto" if not projection else LogisticRegression() if binary_T else LinearRegression(),
                    model_z_xw="auto" if projection else LogisticRegression() if binary_Z else LinearRegression(),
                    flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
                    projection=projection,
                    fit_cov_directly=fit_cov_directly,
                    discrete_instrument=binary_Z,
                    discrete_treatment=binary_T,
                    featurizer=featurizer,
                    use_ray=use_ray,
                ),
            ]

            if X is None:
                est_list = est_list[:-1]  # ForestDRIV doesn't support X=None

            # IntentToTreatDRIV only supports binary treatments and instruments, and doesn't support fit_cov_directly
            if binary_T and binary_Z and not fit_cov_directly:
                est_list += [
                    IntentToTreatDRIV(
                        model_y_xw=LinearRegression(),
                        model_t_xwz=LogisticRegression(),
                        flexible_model_effect=StatsModelsLinearRegression(
                            fit_intercept=False
                        ),
                        fit_cate_intercept=True,
                        featurizer=featurizer,
                        use_ray=use_ray,
                    ),
                    LinearIntentToTreatDRIV(
                        model_y_xw=LinearRegression(),
                        model_t_xwz=LogisticRegression(),
                        flexible_model_effect=StatsModelsLinearRegression(
                            fit_intercept=False
                        ),
                        featurizer=featurizer,
                        use_ray=use_ray,
                    ),
                ]

            for est in est_list:
                with self.subTest(d_w=d_w, d_x=d_x, binary_T=binary_T, binary_Z=binary_Z,
                                  projection=projection, fit_cov_directly=fit_cov_directly,
                                  featurizer=featurizer,
                                  est=est):

                    # TODO: serializing/deserializing for every combination -- is this necessary?
                    # ensure we can serialize unfit estimator
                    pickle.dumps(est)

                    est.fit(y, T, Z=Z, X=X, W=W)

                    # ensure we can serialize fit estimator
                    pickle.dumps(est)

                    # expected effect size
                    exp_const_marginal_effect_shape = const_marg_eff_shape(n, d_x, binary_T)
                    marginal_effect_shape = marg_eff_shape(n, binary_T)
                    effect_shape = eff_shape(n, d_x)

                    # assert calculated constant marginal effect shape is expected
                    # const_marginal effect is defined in LinearCateEstimator class
                    const_marg_eff = est.const_marginal_effect(X)
                    self.assertEqual(shape(const_marg_eff), exp_const_marginal_effect_shape)

                    # assert calculated marginal effect shape is expected
                    marg_eff = est.marginal_effect(T, X)
                    self.assertEqual(shape(marg_eff), marginal_effect_shape)

                    T0 = "a" if binary_T else 0
                    T1 = "b" if binary_T else 1
                    eff = est.effect(X, T0=T0, T1=T1)
                    self.assertEqual(shape(eff), effect_shape)

                    # test inference
                    const_marg_eff_int = est.const_marginal_effect_interval(X)
                    marg_eff_int = est.marginal_effect_interval(T, X)
                    eff_int = est.effect_interval(X, T0=T0, T1=T1)
                    self.assertEqual(shape(const_marg_eff_int), (2,) + exp_const_marginal_effect_shape)
                    self.assertEqual(shape(marg_eff_int), (2,) + marginal_effect_shape)
                    self.assertEqual(shape(eff_int), (2,) + effect_shape)

                    # test can run score
                    est.score(y, T, Z=Z, X=X, W=W)

                    if X is not None:
                        # test cate_feature_names
                        expect_feat_len = featurizer.fit(
                            X).n_output_features_ if featurizer else d_x
                        self.assertEqual(len(est.cate_feature_names()), expect_feat_len)

                        # test can run shap values
                        _ = est.shap_values(X[:10])

    @pytest.mark.ray
    def test_cate_api_with_ray(self):
        try:
            ray.init(num_cpus=1)
            self._test_cate_api(use_ray=True)
        finally:
            ray.shutdown()

    def test_cate_api_without_ray(self):
        self._test_cate_api(use_ray=False)

    def _test_accuracy(self, use_ray=False):
        np.random.seed(0)

        # dgp (binary T, binary Z)

        def dgp(n, p, true_fn):
            X = np.random.normal(0, 1, size=(n, p))
            Z = np.random.binomial(1, 0.5, size=(n,))
            nu = np.random.uniform(0, 10, size=(n,))
            coef_Z = 0.8
            C = np.random.binomial(
                1, coef_Z * special.expit(0.4 * X[:, 0] + nu)
            )  # Compliers when recomended
            C0 = np.random.binomial(
                1, 0.06 * np.ones(X.shape[0])
            )  # Non-compliers when not recommended
            T = C * Z + C0 * (1 - Z)
            y = true_fn(X) * T + 2 * nu + 5 * (X[:, 3] > 0) + 0.1 * np.random.uniform(0, 1, size=(n,))
            return y, T, Z, X

        ests_list = [LinearIntentToTreatDRIV(
            flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False), fit_cate_intercept=True,
            use_ray=use_ray
        ), LinearDRIV(
            fit_cate_intercept=True,
            projection=False,
            discrete_instrument=True,
            discrete_treatment=True,
            flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
            use_ray=use_ray
        )]
        for est in ests_list:
            with self.subTest(est=est):
                # no heterogeneity
                n = 1000
                p = 10
                true_ate = 10

                def true_fn(X):
                    return true_ate
                y, T, Z, X = dgp(n, p, true_fn)

                est.fit(y, T, Z=Z, X=None, W=X, inference="auto")
                ate_lb, ate_ub = est.ate_interval()
                np.testing.assert_array_less(ate_lb, true_ate)
                np.testing.assert_array_less(true_ate, ate_ub)

                # with heterogeneity
                true_coef = 10

                def true_fn(X):
                    return true_coef * X[:, 0]
                y, T, Z, X = dgp(n, p, true_fn)

                est.fit(y, T, Z=Z, X=X[:, [0]], W=X[:, 1:], inference="auto")
                coef_lb, coef_ub = est.coef__interval()
                intercept_lb, intercept_ub = est.intercept__interval(alpha=0.05)
                np.testing.assert_array_less(coef_lb, true_coef)
                np.testing.assert_array_less(true_coef, coef_ub)
                np.testing.assert_array_less(intercept_lb, 0)
                np.testing.assert_array_less(0, intercept_ub)

    @pytest.mark.ray
    def test_accuracy_with_ray(self):
        try:
            ray.init(num_cpus=1)
            self._test_accuracy(use_ray=True)
        finally:
            ray.shutdown()

    def test_accuracy_without_ray(self):
        self._test_accuracy(use_ray=False)

    def test_fit_cov_directly(self):
        # fitting the covariance directly should be at least as good as computing the covariance from separate models

        # set the models so that model selection over random forests doesn't take too much time in the repeated trials
        est = LinearDRIV(model_y_xw=LinearRegression(), model_t_xw=LinearRegression(), model_z_xw=LinearRegression(),
                         model_tz_xw=LassoCV())

        n = 500
        p = 10
        true_coef = 10

        n_trials = 201
        wins = 0

        for _ in range(n_trials):

            X = np.random.normal(size=(n, p))
            Z = np.random.binomial(1, 0.5, size=(n,))
            u = np.random.normal(size=(n,))  # unmeasured confounder
            T = np.random.binomial(1, special.expit(X[:, 0] + Z + u))
            y = np.random.normal(size=(n,)) + true_coef * X[:, 0] * T + u

            est.fit_cov_directly = False
            est.fit(y, T, Z=Z, X=X[:, [0]], W=X[:, 1:])
            coef_indirect = est.coef_

            est.fit_cov_directly = True
            est.fit(y, T, Z=Z, X=X[:, [0]], W=X[:, 1:])
            coef_direct = est.coef_

            # directly fitting the covariance should be better than indirectly fitting it
            if (np.linalg.norm(coef_direct - true_coef) <
                    np.linalg.norm(coef_indirect - true_coef)):
                wins += 1
        print("wins: ", wins)
        self.assertGreater(wins, n_trials / 2)

    def test_groups(self):
        n = 500
        d_w = 10
        d_x = 3
        W = np.random.normal(size=(n, d_w))
        X = np.random.normal(size=(n, d_x))
        T = np.random.choice(["a", "b"], size=(n,))
        Z = np.random.choice(["c", "d"], size=(n,))
        groups = [i // 4 for i in range(n)]
        y = groups
        n_copies = {i: 4 for i in range(125)}

        def ceil(a, b):  # ceiling analog of //
            return -(a // -b)
        ct_lims_2 = (125 // 2, ceil(125, 2))
        ct_lims_3 = (125 - ceil(125, 3), 125 - 125 // 3)

        est_list = [
            DRIV(
                discrete_instrument=True,
                discrete_treatment=True,
                model_y_xw=GroupingModel(LinearRegression(), n, ct_lims_2, n_copies),
                model_z_xw=LogisticRegression(),
                model_t_xw=LogisticRegression(),
                model_tz_xw=LinearRegression(),
                model_t_xwz=LogisticRegression(),
                prel_cate_approach='dmliv'
            ),
            LinearDRIV(
                discrete_instrument=True,
                discrete_treatment=True,
                model_y_xw=GroupingModel(LinearRegression(), n, ct_lims_2, n_copies),
                model_z_xw=LogisticRegression(),
                model_t_xw=LogisticRegression(),
                model_tz_xw=LinearRegression(),
                model_t_xwz=LogisticRegression(),
                prel_cate_approach='dmliv'
            ),
            SparseLinearDRIV(
                discrete_instrument=True,
                discrete_treatment=True,
                model_y_xw=GroupingModel(LinearRegression(), n, ct_lims_2, n_copies),
                model_z_xw=LogisticRegression(),
                model_t_xw=LogisticRegression(),
                model_tz_xw=LinearRegression(),
                model_t_xwz=LogisticRegression(),
                prel_cate_approach='dmliv'
            ),
            ForestDRIV(
                discrete_instrument=True,
                discrete_treatment=True,
                model_y_xw=GroupingModel(LinearRegression(), n, ct_lims_2, n_copies),
                model_z_xw=LogisticRegression(),
                model_t_xw=LogisticRegression(),
                model_tz_xw=LinearRegression(),
                model_t_xwz=LogisticRegression(),
                prel_cate_approach='dmliv'
            ),
            IntentToTreatDRIV(
                model_y_xw=GroupingModel(LinearRegression(), n, ct_lims_3, n_copies),
                model_t_xwz=LogisticRegression(),
                prel_cate_approach='dmliv'
            ),
            LinearIntentToTreatDRIV(
                model_y_xw=GroupingModel(LinearRegression(), n, ct_lims_3, n_copies),
                model_t_xwz=LogisticRegression(),
                prel_cate_approach='dmliv'
            )
        ]

        for est in est_list:
            with self.subTest(est=est):
                est.fit(y, T, Z=Z, X=X, W=W, groups=groups)
                est.score(y, T, Z=Z, X=X, W=W)
                est.const_marginal_effect(X)
