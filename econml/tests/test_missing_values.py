# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
import pytest
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML, SparseLinearDML, KernelDML, CausalForestDML, NonParamDML, DML
from econml.iv.dml import OrthoIV, DMLIV, NonParamDMLIV
from econml.iv.dr import DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV, IntentToTreatDRIV, LinearIntentToTreatDRIV
from econml.orf import DMLOrthoForest
from econml.dr import DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.panel.dml import DynamicDML

import inspect
import scipy


class ModelNuisance:
    def __init__(self, model_t, model_y):
        self._model_t = model_t
        self._model_y = model_y

    def fit(self, Y, T, W=None):
        self._model_t.fit(W, T)
        self._model_y.fit(W, Y)
        return self

    def predict(self, Y, T, W=None):
        return Y - self._model_y.predict(W), T - self._model_t.predict(W)


class ModelFinal:

    def __init__(self):
        return

    def fit(self, Y, T, W=None, nuisances=None):
        Y_res, T_res = nuisances
        self.model = LinearRegression(fit_intercept=False).fit(T_res.reshape(-1, 1), Y_res)
        return self

    def predict(self):
        # theta needs to be of dimension (1, d_t) if T is (n, d_t)
        return np.array([[self.model.coef_[0]]])

    def score(self, Y, T, W=None, nuisances=None):
        Y_res, T_res = nuisances
        return np.mean((Y_res - self.model.predict(T_res.reshape(-1, 1)))**2)


class ParametricModelFinalForMissing:
    def __init__(self, model_final):
        self.model_final = model_final

    def fit(self, *args, **kwargs):
        self.model_final.fit(*args, **kwargs)

        self.coef_ = self.model_final[-1].coef_
        self.intercept_ = self.model_final[-1].intercept_
        self.intercept_stderr_ = self.model_final[-1].intercept_stderr_
        self.coef_stderr_ = self.model_final[-1].coef_stderr_

    def predict(self, *args, **kwargs):
        return self.model_final.predict(*args, **kwargs)


class NonParamModelFinal:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, *args, **kwargs):
        sample_weight = kwargs.pop('sample_weight')
        return self.pipeline.fit(*args, **kwargs, lassocv__sample_weight=sample_weight)

    def predict(self, *args, **kwargs):
        return self.pipeline.predict(*args, **kwargs)


def create_data_dict(Y, T, X, X_missing, W, W_missing, Z, X_has_missing=False, W_has_missing=False, include_Z=False):
    data_dict = {'Y': Y, 'T': T, 'X': X_missing if X_has_missing else X, 'W': W_missing if W_has_missing else W}
    if include_Z:
        data_dict['Z'] = Z

    return data_dict


class OrthoLearner(_OrthoLearner):
    def _gen_ortho_learner_model_nuisance(self):
        return ModelNuisance(
            make_pipeline(SimpleImputer(strategy='mean'), LinearRegression()),
            make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
        )

    def _gen_ortho_learner_model_final(self):
        return ModelFinal()


class TestMissing(unittest.TestCase):

    def test_missing(self):
        # create data with missing values
        np.random.seed(123)
        X = np.random.normal(size=(1000, 1))
        W = np.random.normal(size=(1000, 5))
        T = X[:, 0] + np.random.normal(size=(1000,))
        y = (1 + 0.5 * X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        mask = np.random.rand(*W.shape) < 0.05
        W_missing = W.copy()
        W_missing[mask] = np.nan
        groups = np.repeat(np.arange(500), 2)  # groups for dynamic dml

        # model that can handle missing values
        nuisance_model = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
        OrthoLearner(discrete_treatment=False, treatment_featurizer=None, discrete_instrument=None,
                     categories='auto', cv=3, random_state=1, enable_missing=True).fit(y, T, W=W_missing)

        CausalForestDML(model_y=nuisance_model, model_t=nuisance_model,
                        enable_missing=True).fit(y, T, X=X, W=W_missing)

        DynamicDML(model_y=nuisance_model, model_t=nuisance_model,
                   enable_missing=True).fit(y, T, W=W_missing, groups=groups)

        LinearDML(model_y=nuisance_model, model_t=nuisance_model,
                  enable_missing=True).dowhy.fit(y, T, X=X, W=W_missing)

    def test_missing2(self):
        n = 100
        np.random.seed(123)
        X = np.random.normal(size=(n, 1))
        W = np.random.normal(size=(n, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0] + np.random.normal(size=(n,))))
        y = (1 + 0.5 * X[:, 0]) * T + X[:, 0] + np.random.normal(size=(n,))
        mask = np.random.rand(*W.shape) < 0.05
        W_missing = W.copy()
        W_missing[mask] = np.nan
        mask = np.random.rand(*X.shape) < 0.05
        X_missing = X.copy()
        X_missing[mask] = np.nan
        Z = np.random.binomial(1, 0.5, size=(n,))

        model_t = make_pipeline(SimpleImputer(strategy='mean'), LogisticRegressionCV())
        model_y = make_pipeline(SimpleImputer(strategy='mean'), LassoCV())
        model_final = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
        param_model_final = ParametricModelFinalForMissing(make_pipeline(
            SimpleImputer(strategy='mean'), StatsModelsLinearRegression(fit_intercept=False)))
        non_param_model_final = NonParamModelFinal(make_pipeline(SimpleImputer(strategy='mean'), LassoCV()))
        discrete_treatment = True
        discrete_instrument = True

        # test X, W only
        x_w_missing_models = [
            NonParamDML(model_y=model_y, model_t=model_t, model_final=non_param_model_final,
                        discrete_treatment=discrete_treatment, enable_missing=True),
            DML(model_y=model_y, model_t=model_t, model_final=param_model_final, enable_missing=True),
            DMLIV(model_y_xw=model_y, model_t_xw=model_t, model_t_xwz=model_t,
                  model_final=param_model_final, discrete_treatment=discrete_treatment,
                  discrete_instrument=discrete_instrument, enable_missing=True),
            NonParamDMLIV(model_y_xw=model_y, model_t_xw=model_t, model_t_xwz=model_t,
                          model_final=non_param_model_final, discrete_treatment=discrete_treatment,
                          discrete_instrument=discrete_instrument, enable_missing=True),
            DRLearner(model_propensity=model_t, model_regression=model_y, model_final=model_final, enable_missing=True)
        ]

        # test W only
        w_missing_models = [
            DRIV(model_y_xw=model_y, model_t_xw=model_t, model_z_xw=model_t, model_tz_xw=model_t,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='driv', projection=False, enable_missing=True),
            DRIV(model_y_xw=model_y, model_t_xw=model_t, model_t_xwz=model_t, model_tz_xw=model_y,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='driv', projection=True, enable_missing=True),
            DRIV(model_y_xw=model_y, model_t_xw=model_t, model_z_xw=model_t, model_t_xwz=model_t, model_tz_xw=model_t,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='dmliv', projection=False, enable_missing=True),
            DRIV(model_y_xw=model_y, model_t_xw=model_t, model_t_xwz=model_t, model_tz_xw=model_y,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='dmliv', projection=True, enable_missing=True),
            IntentToTreatDRIV(model_y_xw=model_y, model_t_xwz=model_t, prel_cate_approach='driv',
                              model_final=model_final, enable_missing=True),
            IntentToTreatDRIV(model_y_xw=model_y, model_t_xwz=model_t, prel_cate_approach='dmliv',
                              model_final=model_final, enable_missing=True),
            LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, enable_missing=True),
            SparseLinearDML(model_y=model_y, model_t=model_t, discrete_treatment=True, enable_missing=True),
            KernelDML(model_y=model_y, model_t=model_t, discrete_treatment=True, enable_missing=True),
            CausalForestDML(model_y=model_y, model_t=model_t, discrete_treatment=True, enable_missing=True),
            LinearDRLearner(model_propensity=model_t, model_regression=model_y, enable_missing=True),
            SparseLinearDRLearner(model_propensity=model_t, model_regression=model_y, enable_missing=True),
            ForestDRLearner(model_propensity=model_t, model_regression=model_y, enable_missing=True),
            OrthoIV(model_y_xw=model_y, model_t_xw=model_t, model_z_xw=model_t,
                    discrete_treatment=True, discrete_instrument=True, enable_missing=True),
            LinearDRIV(model_y_xw=model_y, model_t_xw=model_t, model_z_xw=model_t, model_tz_xw=model_t,
                       prel_cate_approach='driv', discrete_treatment=True, discrete_instrument=True,
                       enable_missing=True),
            SparseLinearDRIV(model_y_xw=model_y, model_t_xw=model_t, model_z_xw=model_t, model_tz_xw=model_t,
                             prel_cate_approach='driv', discrete_treatment=True, discrete_instrument=True,
                             enable_missing=True),
            ForestDRIV(model_y_xw=model_y, model_t_xw=model_t, model_z_xw=model_t, model_tz_xw=model_t,
                       prel_cate_approach='driv', discrete_treatment=True, discrete_instrument=True,
                       enable_missing=True),
            LinearIntentToTreatDRIV(model_y_xw=model_y, model_t_xwz=model_t,
                                    prel_cate_approach='driv', enable_missing=True)
        ]

        for est in x_w_missing_models:
            print(est)
            if 'Z' in inspect.getfullargspec(est.fit).kwonlyargs:
                include_Z = True
            else:
                include_Z = False

            data_dict = create_data_dict(y, T, X, X_missing, W, W_missing, Z,
                                         X_has_missing=True, W_has_missing=True, include_Z=include_Z)
            est.fit(**data_dict)
            self.assertRaises(ValueError, est.dowhy.fit, **data_dict)  # missing in X should fail with dowhywrapper

            # assert that fitting with missing values fails when enable_missing is False
            # and that setting enable_missing after init still works
            est.enable_missing = False
            self.assertRaises(ValueError, est.fit, **data_dict)
            self.assertRaises(ValueError, est.dowhy.fit, **data_dict)

        for est in w_missing_models:
            print(est)
            if 'Z' in inspect.getfullargspec(est.fit).kwonlyargs:
                include_Z = True
            else:
                include_Z = False

            data_dict = create_data_dict(y, T, X, X_missing, W, W_missing, Z,
                                         X_has_missing=False, W_has_missing=True, include_Z=include_Z)
            est.fit(**data_dict)
            est.dowhy.fit(**data_dict)

            # assert that we fail with a value error when we pass missing X to a model that doesn't support it
            data_dict_to_fail = create_data_dict(y, T, X, X_missing, W, W_missing, Z,
                                                 X_has_missing=True, W_has_missing=True, include_Z=include_Z)
            self.assertRaises(ValueError, est.fit, **data_dict_to_fail)
            self.assertRaises(ValueError, est.dowhy.fit, **data_dict_to_fail)

            # assert that fitting with missing values fails when enable_missing is False
            # and that setting enable_missing after init still works
            est.enable_missing = False
            self.assertRaises(ValueError, est.fit, **data_dict)
            self.assertRaises(ValueError, est.dowhy.fit, **data_dict)
