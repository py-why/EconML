# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
import pickle
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.model_selection import KFold
from econml.iv.dml import (DMLATEIV, ProjectedDMLATEIV, DMLIV, NonParamDMLIV)
from econml.iv.dr import (IntentToTreatDRIV, LinearIntentToTreatDRIV)
import numpy as np
from econml.utilities import shape, hstack, vstack, reshape, cross_product
from econml.inference import BootstrapInference
from contextlib import ExitStack
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
import itertools
from econml.sklearn_extensions.linear_model import WeightedLasso, StatsModelsLinearRegression
from econml.tests.test_statsmodels import _summarize
import econml.tests.utilities  # bugfix for assertWarns


class TestOrthoIV(unittest.TestCase):

    def test_cate_api(self):
        """Test that we correctly implement the CATE API."""
        n = 30

        def size(n, d):
            return (n, d) if d >= 0 else (n,)

        def make_random(is_discrete, d):
            if d is None:
                return None
            sz = size(n, d)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b', 'c'], size=sz)
                    # ensure that we've got at least two of every row
                    _, counts = np.unique(arr, return_counts=True, axis=0)
                    if len(counts) == 3**(d if d > 0 else 1) and counts.min() > 1:
                        return arr
            else:
                return np.random.normal(size=sz)

        def eff_shape(n, d_y):
            return (n,) + ((d_y,) if d_y > 0 else())

        def marg_eff_shape(n, d_y, d_t_final):
            return ((n,) +
                    ((d_y,) if d_y > 0 else ()) +
                    ((d_t_final,) if d_t_final > 0 else()))

        # since T isn't passed to const_marginal_effect, defaults to one row if X is None
        def const_marg_eff_shape(n, d_x, d_y, d_t_final):
            return ((n if d_x else 1,) +
                    ((d_y,) if d_y > 0 else ()) +
                    ((d_t_final,) if d_t_final > 0 else()))

        for d_t in [2, 1, -1]:
            n_t = d_t if d_t > 0 else 1
            for discrete_t in [True, False] if n_t == 1 else [False]:
                for d_y in [3, 1, -1]:
                    for d_q in [2, None]:
                        for d_z in [2, 1]:
                            if d_z < n_t:
                                continue
                            for discrete_z in [True, False] if d_z == 1 else[False]:
                                Z1, Q, Y, T1 = [make_random(is_discrete, d)
                                                for is_discrete, d in [(discrete_z, d_z),
                                                                       (False, d_q),
                                                                       (False, d_y),
                                                                       (discrete_t, d_t)]]
                                if discrete_t and discrete_z:
                                    # need to make sure we get all *joint* combinations
                                    arr = make_random(True, 2)
                                    Z1 = arr[:, 0].reshape(size(n, d_z))
                                    T1 = arr[:, 0].reshape(size(n, d_t))

                                d_t_final1 = 2 if discrete_t else d_t

                                if discrete_t:
                                    # IntentToTreat only supports binary treatments/instruments
                                    T2 = T1.copy()
                                    T2[T1 == 'c'] = np.random.choice(['a', 'b'], size=np.count_nonzero(T1 == 'c'))
                                    d_t_final2 = 1
                                if discrete_z:
                                    # IntentToTreat only supports binary treatments/instruments
                                    Z2 = Z1.copy()
                                    Z2[Z1 == 'c'] = np.random.choice(['a', 'b'], size=np.count_nonzero(Z1 == 'c'))

                                effect_shape = eff_shape(n, d_y)

                                model_t = LogisticRegression() if discrete_t else Lasso()
                                model_z = LogisticRegression() if discrete_z else Lasso()

                                all_infs = [None, BootstrapInference(1)]

                                estimators = [(DMLATEIV(model_Y_W=Lasso(),
                                                        model_T_W=model_t,
                                                        model_Z_W=model_z,
                                                        discrete_treatment=discrete_t,
                                                        discrete_instrument=discrete_z),
                                               True,
                                               all_infs),
                                              (ProjectedDMLATEIV(model_Y_W=Lasso(),
                                                                 model_T_W=model_t,
                                                                 model_T_WZ=model_t,
                                                                 discrete_treatment=discrete_t,
                                                                 discrete_instrument=discrete_z),
                                               False,
                                               all_infs),
                                              (DMLIV(model_Y_X=Lasso(), model_T_X=model_t, model_T_XZ=model_t,
                                                     model_final=Lasso(),
                                                     discrete_treatment=discrete_t, discrete_instrument=discrete_z),
                                               False,
                                               all_infs)]

                                if d_q and discrete_t and discrete_z:
                                    # IntentToTreat requires X
                                    estimators.append((
                                        LinearIntentToTreatDRIV(model_Y_X=Lasso(), model_T_XZ=model_t,
                                                                flexible_model_effect=WeightedLasso(),
                                                                cv=2),
                                        False,
                                        all_infs + ['auto']))

                                for est, multi, infs in estimators:
                                    if not(multi) and d_y > 1 or d_t > 1 or d_z > 1:
                                        continue

                                    # ensure we can serialize unfit estimator
                                    pickle.dumps(est)

                                    d_ws = [None]
                                    if isinstance(est, LinearIntentToTreatDRIV):
                                        d_ws.append(2)

                                    for d_w in d_ws:
                                        W = make_random(False, d_w)

                                        for inf in infs:
                                            with self.subTest(d_z=d_z, d_x=d_q, d_y=d_y, d_t=d_t,
                                                              discrete_t=discrete_t, discrete_z=discrete_z,
                                                              est=est, inf=inf):
                                                Z = Z1
                                                T = T1
                                                d_t_final = d_t_final1
                                                X = Q
                                                d_x = d_q

                                                if isinstance(est, (DMLATEIV, ProjectedDMLATEIV)):
                                                    # these support only W but not X
                                                    W = Q
                                                    X = None
                                                    d_x = None

                                                    def fit():
                                                        return est.fit(Y, T, Z=Z, W=W, inference=inf)

                                                    def score():
                                                        return est.score(Y, T, Z=Z, W=W)
                                                else:
                                                    # these support only binary, not general discrete T and Z
                                                    if discrete_t:
                                                        T = T2
                                                        d_t_final = d_t_final2

                                                    if discrete_z:
                                                        Z = Z2

                                                    if isinstance(est, LinearIntentToTreatDRIV):
                                                        def fit():
                                                            return est.fit(Y, T, Z=Z, X=X, W=W, inference=inf)

                                                        def score():
                                                            return est.score(Y, T, Z=Z, X=X, W=W)
                                                    else:
                                                        def fit():
                                                            return est.fit(Y, T, Z=Z, X=X, inference=inf)

                                                        def score():
                                                            return est.score(Y, T, Z=Z, X=X)

                                                marginal_effect_shape = marg_eff_shape(n, d_y, d_t_final)
                                                const_marginal_effect_shape = const_marg_eff_shape(
                                                    n, d_x, d_y, d_t_final)

                                                fit()

                                                # ensure we can serialize fit estimator
                                                pickle.dumps(est)

                                                # make sure we can call the marginal_effect and effect methods
                                                const_marg_eff = est.const_marginal_effect(X)
                                                marg_eff = est.marginal_effect(T, X)
                                                self.assertEqual(shape(marg_eff), marginal_effect_shape)
                                                self.assertEqual(shape(const_marg_eff), const_marginal_effect_shape)

                                                np.testing.assert_array_equal(
                                                    marg_eff if d_x else marg_eff[0:1], const_marg_eff)

                                                T0 = np.full_like(T, 'a') if discrete_t else np.zeros_like(T)
                                                eff = est.effect(X, T0=T0, T1=T)
                                                self.assertEqual(shape(eff), effect_shape)

                                                # TODO: add tests for extra properties like coef_ where they exist

                                                if inf is not None:
                                                    const_marg_eff_int = est.const_marginal_effect_interval(X)
                                                    marg_eff_int = est.marginal_effect_interval(T, X)
                                                    self.assertEqual(shape(marg_eff_int),
                                                                     (2,) + marginal_effect_shape)
                                                    self.assertEqual(shape(const_marg_eff_int),
                                                                     (2,) + const_marginal_effect_shape)
                                                    self.assertEqual(shape(est.effect_interval(X, T0=T0, T1=T)),
                                                                     (2,) + effect_shape)

                                                # TODO: add tests for extra properties like coef_ where they exist

                                                score()

                                                # make sure we can call effect with implied scalar treatments,
                                                # no matter the dimensions of T, and also that we warn when there
                                                # are multiple treatments
                                                if d_t > 1:
                                                    cm = self.assertWarns(Warning)
                                                else:
                                                    # ExitStack can be used as a "do nothing" ContextManager
                                                    cm = ExitStack()
                                                with cm:
                                                    effect_shape2 = (n if d_x else 1,) + ((d_y,) if d_y > 0 else())
                                                    eff = est.effect(X) if not discrete_t else est.effect(
                                                        X, T0='a', T1='b')
                                                    self.assertEqual(shape(eff), effect_shape2)

    def test_bad_splits_discrete(self):
        """
        Tests that when some training splits in a crossfit fold don't contain all treatments then an error
        is raised.
        """
        Y = np.array([2, 3, 1, 3, 2, 1, 1, 1])
        bad = np.array([2, 2, 1, 2, 1, 1, 1, 1])
        W = np.ones((8, 1))
        ok = np.array([1, 2, 3, 1, 2, 3, 1, 2])
        est = DMLATEIV(model_Y_W=Lasso(), model_T_W=Lasso(), model_Z_W=Lasso(),
                       cv=[(np.arange(4, 8), np.arange(4))])
        est.fit(Y, T=bad, Z=bad, W=W)  # imbalance ok with continuous instrument/treatment

        est = DMLATEIV(model_Y_W=Lasso(), model_T_W=LogisticRegression(), model_Z_W=Lasso(),
                       cv=[(np.arange(4, 8), np.arange(4))], discrete_treatment=True)
        with pytest.raises(AttributeError):
            est.fit(Y, T=bad, Z=ok, W=W)

        est = DMLATEIV(model_Y_W=Lasso(), model_T_W=Lasso(), model_Z_W=LogisticRegression(),
                       cv=[(np.arange(4, 8), np.arange(4))], discrete_instrument=True)
        with pytest.raises(AttributeError):
            est.fit(Y, T=ok, Z=bad, W=W)

        # TODO: ideally we could also test whether Z and X are jointly okay when both discrete
        #       however, with custom splits the checking happens in the first stage wrapper
        #       where we don't have all of the required information to do this;
        #       we'd probably need to add it to _crossfit instead

    def test_multidim_arrays_fail(self):

        Y = np.array([2, 3, 1, 3, 2, 1, 1, 1])
        three_class = np.array([1, 2, 3, 1, 2, 3, 1, 2])
        two_class = np.array([1, 2, 1, 1, 2, 1, 1, 2])

        est = NonParamDMLIV(model_Y_X=Lasso(), model_T_X=LogisticRegression(), model_T_XZ=LogisticRegression(),
                            model_final=WeightedLasso(), discrete_treatment=True)

        with pytest.raises(AttributeError):
            est.fit(Y, T=three_class, Z=two_class)

        est = IntentToTreatDRIV(model_Y_X=Lasso(), model_T_XZ=LogisticRegression(),
                                flexible_model_effect=WeightedLasso())

        with pytest.raises(AttributeError):
            est.fit(Y, T=three_class, Z=two_class)

        with pytest.raises(AttributeError):
            est.fit(Y, T=two_class, Z=three_class)

    def test_access_to_internal_models(self):
        """
        Test that API related to accessing the nuisance models, cate_model and featurizer is working.
        """
        est = LinearIntentToTreatDRIV(model_Y_X=LinearRegression(),
                                      model_T_XZ=LogisticRegression(C=1000),
                                      flexible_model_effect=WeightedLasso(),
                                      featurizer=PolynomialFeatures(degree=2, include_bias=False))
        Y = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
        T = np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2])
        Z = np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2])
        X = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
        est.fit(Y, T, Z=Z, X=X)
        assert isinstance(est.original_featurizer, PolynomialFeatures)
        assert isinstance(est.featurizer_, Pipeline)
        assert isinstance(est.model_final_, StatsModelsLinearRegression)
        for mdl in est.models_Y_X:
            assert isinstance(mdl, LinearRegression)
        for mdl in est.models_T_XZ:
            assert isinstance(mdl, LogisticRegression)
        np.testing.assert_array_equal(est.cate_feature_names(['A']), ['A', 'A^2'])
        np.testing.assert_array_equal(est.cate_feature_names(), ['x0', 'x0^2'])

        est = LinearIntentToTreatDRIV(model_Y_X=LinearRegression(),
                                      model_T_XZ=LogisticRegression(C=1000),
                                      flexible_model_effect=WeightedLasso(),
                                      featurizer=None)
        est.fit(Y, T, Z=Z, X=X)
        assert est.original_featurizer is None
        assert isinstance(est.featurizer_, FunctionTransformer)
        assert isinstance(est.model_final_, StatsModelsLinearRegression)
        for mdl in est.models_Y_X:
            assert isinstance(mdl, LinearRegression)
        for mdl in est.models_T_XZ:
            assert isinstance(mdl, LogisticRegression)
        np.testing.assert_array_equal(est.cate_feature_names(['A']), ['A'])

    def test_can_use_statsmodel_inference(self):
        """Test that we can use statsmodels to generate confidence intervals"""
        est = LinearIntentToTreatDRIV(model_Y_X=LinearRegression(),
                                      model_T_XZ=LogisticRegression(C=1000),
                                      flexible_model_effect=WeightedLasso())
        est.fit(np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]),
                np.array([1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]),
                Z=np.array([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]),
                X=np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]).reshape(-1, 1))
        interval = est.effect_interval(np.ones((9, 1)),
                                       T0=np.array([1, 1, 1, 2, 2, 2, 1, 1, 1]),
                                       T1=np.array([1, 2, 1, 1, 2, 2, 2, 2, 1]),
                                       alpha=0.05)
        point = est.effect(np.ones((9, 1)),
                           T0=np.array([1, 1, 1, 2, 2, 2, 1, 1, 1]),
                           T1=np.array([1, 2, 1, 1, 2, 2, 2, 2, 1]))

        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert np.all(lo <= point)
        assert np.all(point <= hi)
        assert np.any(lo < hi)  # for at least some of the examples, the CI should have nonzero width

        interval = est.const_marginal_effect_interval(np.ones((9, 1)), alpha=0.05)
        point = est.const_marginal_effect(np.ones((9, 1)))
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert np.all(lo <= point)
        assert np.all(point <= hi)
        assert np.any(lo < hi)  # for at least some of the examples, the CI should have nonzero width

        interval = est.coef__interval(alpha=0.05)
        point = est.coef_
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert np.all(lo <= point)
        assert np.all(point <= hi)
        assert np.any(lo < hi)  # for at least some of the examples, the CI should have nonzero width

        interval = est.intercept__interval(alpha=0.05)
        point = est.intercept_
        assert len(interval) == 2
        lo, hi = interval
        assert np.all(lo <= point)
        assert np.all(point <= hi)
        assert np.any(lo < hi)  # for at least some of the examples, the CI should have nonzero width
