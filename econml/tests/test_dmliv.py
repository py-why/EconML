# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pickle
import unittest

import numpy as np
import pytest
from scipy import special
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from econml.iv.dml import OrthoIV, DMLIV, NonParamDMLIV
from econml.iv.dr._dr import _DummyCATE
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.utilities import shape


@pytest.mark.cate_api
class TestDMLIV(unittest.TestCase):
    def test_cate_api(self):
        def const_marg_eff_shape(n, d_x, d_y, binary_T):
            return (n if d_x else 1,) + ((d_y,) if d_y > 1 else ()) + ((1,) if binary_T else ())

        def marg_eff_shape(n, d_y, binary_T):
            return (n,) + ((d_y,) if d_y > 1 else ()) + ((1,) if binary_T else ())

        def eff_shape(n, d_x, d_y):
            return (n if d_x else 1,) + ((d_y,) if d_y > 1 else ())

        n = 1000
        y = np.random.normal(size=(n,))

        for d_y in [1, 3]:
            if d_y == 1:
                y = np.random.normal(size=(n,))
            else:
                y = y = np.random.normal(size=(n, d_y))
            for d_w in [None, 10]:
                if d_w is None:
                    W = None
                else:
                    W = np.random.normal(size=(n, d_w))
                for d_x in [None, 3]:
                    if d_x is None:
                        X = None
                    else:
                        X = np.random.normal(size=(n, d_x))
                    for binary_Z in [True, False]:
                        if binary_Z:
                            Z = np.random.choice([3, 4], size=(n,))
                        else:
                            Z = np.random.normal(1, 3, size=(n,))
                        for binary_T in [True, False]:
                            if binary_T:
                                T = np.random.choice([0, 1], size=(n,))
                            else:
                                T = np.random.uniform(1, 3, size=(n,)) + 0.5 * Z
                            for featurizer in [
                                None,
                                PolynomialFeatures(degree=2, include_bias=False),
                            ]:
                                est_list = [
                                    OrthoIV(
                                        projection=False,
                                        featurizer=featurizer,
                                        discrete_treatment=binary_T,
                                        discrete_instrument=binary_Z,
                                    ),
                                    OrthoIV(
                                        projection=True,
                                        featurizer=featurizer,
                                        discrete_treatment=binary_T,
                                        discrete_instrument=binary_Z,
                                    ),
                                    DMLIV(
                                        model_final=LinearRegression(fit_intercept=False),
                                        featurizer=featurizer,
                                        discrete_treatment=binary_T,
                                        discrete_instrument=binary_Z,
                                    ),
                                    NonParamDMLIV(
                                        model_final=RandomForestRegressor(),
                                        featurizer=featurizer,
                                        discrete_treatment=binary_T,
                                        discrete_instrument=binary_Z,
                                    ),
                                ]

                                if X is None:
                                    est_list = est_list[:-1]

                                for est in est_list:
                                    with self.subTest(d_w=d_w, d_x=d_x, binary_T=binary_T, binary_Z=binary_Z,
                                                      featurizer=featurizer, est=est):

                                        # ensure we can serialize unfit estimator
                                        pickle.dumps(est)

                                        est.fit(y, T, Z=Z, X=X, W=W)

                                        # ensure we can serialize fit estimator
                                        pickle.dumps(est)

                                        # expected effect size
                                        const_marginal_effect_shape = const_marg_eff_shape(n, d_x, d_y, binary_T)
                                        marginal_effect_shape = marg_eff_shape(n, d_y, binary_T)
                                        effect_shape = eff_shape(n, d_x, d_y)
                                        # test effect
                                        const_marg_eff = est.const_marginal_effect(X)
                                        self.assertEqual(shape(const_marg_eff), const_marginal_effect_shape)
                                        marg_eff = est.marginal_effect(T, X)
                                        self.assertEqual(shape(marg_eff), marginal_effect_shape)
                                        eff = est.effect(X, T0=0, T1=1)
                                        self.assertEqual(shape(eff), effect_shape)

                                        # test inference
                                        # only OrthoIV support inference other than bootstrap
                                        if isinstance(est, OrthoIV):
                                            const_marg_eff_int = est.const_marginal_effect_interval(X)
                                            marg_eff_int = est.marginal_effect_interval(T, X)
                                            eff_int = est.effect_interval(X, T0=0, T1=1)
                                            self.assertEqual(shape(const_marg_eff_int), (2,) +
                                                             const_marginal_effect_shape)
                                            self.assertEqual(shape(marg_eff_int), (2,) + marginal_effect_shape)
                                            self.assertEqual(shape(eff_int), (2,) + effect_shape)

                                        # test summary
                                        if isinstance(est, (OrthoIV, DMLIV)):
                                            est.summary()

                                        # test can run score
                                        est.score(y, T, Z, X=X, W=W)

                                        if X is not None:
                                            # test cate_feature_names
                                            expect_feat_len = featurizer.fit(
                                                X).n_output_features_ if featurizer else d_x
                                            self.assertEqual(len(est.cate_feature_names()), expect_feat_len)

                                            # test can run shap values
                                            shap_values = est.shap_values(X[:10])

    def test_accuracy(self):
        np.random.seed(123)

        # dgp
        def dgp(n, p, true_fn):
            def epsilon_sample(n):
                return np.random.normal(-3, 3, size=(n,))
            X = np.random.normal(0, 1, size=(n, p))
            beta_z = np.random.uniform(3, 5, size=(p,))
            Z = np.dot(X, beta_z) + epsilon_sample(n)
            beta_t = np.random.uniform(0, 1, size=(p,))
            T = 3 * true_fn(X) * Z + np.dot(X, beta_t) + epsilon_sample(n)
            beta_y = np.random.uniform(1, 3, size=(p,))
            y = true_fn(X) * T + np.dot(X, beta_y) + epsilon_sample(n)
            return y, T, Z, X

        ests_list = [
            OrthoIV(
                projection=False,
                discrete_treatment=False,
                discrete_instrument=False,
                fit_cate_intercept=True,
            ),
            OrthoIV(
                projection=True,
                discrete_treatment=False,
                discrete_instrument=False,
                fit_cate_intercept=True,
            ),
        ]

        # no heterogeneity
        n = 2000
        p = 5
        true_ate = 1.3

        def true_fn(X):
            return true_ate
        y, T, Z, X = dgp(n, p, true_fn)
        for est in ests_list:
            with self.subTest(est=est):
                est.fit(y, T, Z=Z, X=None, W=X, inference="auto")
                ate_lb, ate_ub = est.effect_interval()
                np.testing.assert_array_less(ate_lb, true_ate)
                np.testing.assert_array_less(true_ate, ate_ub)
