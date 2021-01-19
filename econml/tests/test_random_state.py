import unittest
import pytest
import pickle
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.model_selection import KFold, GroupKFold
from econml.dml import DML, LinearDML, SparseLinearDML, KernelDML
from econml.dml import NonParamDML, CausalForestDML
from econml.dr import DRLearner, SparseLinearDRLearner, LinearDRLearner, ForestDRLearner
from econml.iv.dml import (DMLATEIV, ProjectedDMLATEIV, DMLIV, NonParamDMLIV)
from econml.iv.dr import (IntentToTreatDRIV, LinearIntentToTreatDRIV)
import numpy as np
from econml.utilities import shape, hstack, vstack, reshape, cross_product
from econml.inference import BootstrapInference
from contextlib import ExitStack
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import itertools
from econml.sklearn_extensions.linear_model import WeightedLasso, StatsModelsRLM
from econml.tests.test_statsmodels import _summarize
import econml.tests.utilities  # bugfix for assertWarns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class TestRandomState(unittest.TestCase):

    @staticmethod
    def _make_data(n, p):
        np.random.seed(1283)
        X = np.random.uniform(-1, 1, size=(n, p))
        W = np.random.uniform(-1, 1, size=(n, p))

        def true_propensity(x):
            return .4 + .1 * (x[:, 0] > 0)

        def true_effect(x):
            return .4 + .2 * x[:, 0]

        def true_conf(x):
            return x[:, 1]

        T = np.random.binomial(1, true_propensity(X))
        Y = true_effect(X) * T + true_conf(X) + np.random.normal(size=(n,))
        X_test = np.zeros((100, p))
        X_test[:, 0] = np.linspace(-1, 1, 100)
        return Y, T, X, W, X_test

    @staticmethod
    def _test_random_state(est, X_test, Y, T, **kwargs):
        est.fit(Y, T, **kwargs)
        te1 = est.effect(X_test)
        est.fit(Y, T, **kwargs)
        te2 = est.effect(X_test)
        est.fit(Y, T, **kwargs)
        te3 = est.effect(X_test)
        np.testing.assert_allclose(te1, te2, err_msg='random state fixing does not work')
        np.testing.assert_allclose(te1, te3, err_msg='random state fixing does not work')

    def test_dml_random_state(self):
        Y, T, X, W, X_test = TestRandomState._make_data(500, 2)
        for est in [
                NonParamDML(model_y=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                            model_t=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                            model_final=RandomForestRegressor(max_depth=3, n_estimators=10, min_samples_leaf=100,
                                                              bootstrap=True, random_state=123),
                            discrete_treatment=True, cv=2, random_state=123),
                CausalForestDML(model_y=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                                model_t=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                                n_estimators=8,
                                discrete_treatment=True, cv=2, random_state=123),
                LinearDML(model_y=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                          model_t=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                          discrete_treatment=True, cv=2, random_state=123),
                SparseLinearDML(discrete_treatment=True, cv=2, random_state=123),
                KernelDML(discrete_treatment=True, cv=2, random_state=123)]:
            TestRandomState._test_random_state(est, X_test, Y, T, X=X, W=W)

    def test_dr_random_state(self):
        Y, T, X, W, X_test = self._make_data(500, 2)
        for est in [
                DRLearner(model_final=RandomForestRegressor(max_depth=3, n_estimators=10, min_samples_leaf=100,
                                                            bootstrap=True, random_state=123),
                          cv=2, random_state=123),
                LinearDRLearner(random_state=123),
                SparseLinearDRLearner(cv=2, random_state=123),
                ForestDRLearner(model_regression=RandomForestRegressor(n_estimators=10, max_depth=4,
                                                                       random_state=123),
                                model_propensity=RandomForestClassifier(
                    n_estimators=10, max_depth=4, random_state=123),
                    cv=2, random_state=123)]:
            TestRandomState._test_random_state(est, X_test, Y, T, X=X, W=W)

    def test_orthoiv_random_state(self):
        Y, T, X, W, X_test = self._make_data(500, 2)
        for est in [
            DMLATEIV(model_Y_W=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                     model_T_W=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                     model_Z_W=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                     discrete_treatment=True, discrete_instrument=True, cv=2, random_state=123),
            ProjectedDMLATEIV(model_Y_W=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                              model_T_W=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                              model_T_WZ=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                              discrete_treatment=True, discrete_instrument=True, cv=2, random_state=123)]:
            TestRandomState._test_random_state(est, None, Y, T, W=W, Z=T)
        for est in [
                DMLIV(model_Y_X=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                      model_T_X=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                      model_T_XZ=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                      model_final=LinearRegression(fit_intercept=False),
                      discrete_treatment=True, discrete_instrument=True, cv=2, random_state=123),
                NonParamDMLIV(model_Y_X=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                              model_T_X=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                              model_T_XZ=RandomForestClassifier(n_estimators=10, max_depth=4, random_state=123),
                              model_final=LinearRegression(),
                              discrete_treatment=True, discrete_instrument=True, cv=2, random_state=123)]:
            TestRandomState._test_random_state(est, X_test, Y, T, X=X, Z=T)
        for est in [IntentToTreatDRIV(model_Y_X=RandomForestRegressor(n_estimators=10, max_depth=4, random_state=123),
                                      model_T_XZ=RandomForestClassifier(n_estimators=10,
                                                                        max_depth=4, random_state=123),
                                      flexible_model_effect=RandomForestRegressor(n_estimators=10,
                                                                                  max_depth=4, random_state=123),
                                      cv=2, random_state=123),
                    LinearIntentToTreatDRIV(model_Y_X=RandomForestRegressor(n_estimators=10,
                                                                            max_depth=4, random_state=123),
                                            model_T_XZ=RandomForestClassifier(n_estimators=10,
                                                                              max_depth=4, random_state=123),
                                            flexible_model_effect=RandomForestRegressor(n_estimators=10,
                                                                                        max_depth=4,
                                                                                        random_state=123),
                                            cv=2, random_state=123)]:
            TestRandomState._test_random_state(est, X_test, Y, T, X=X, W=W, Z=T)
