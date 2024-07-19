# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer


from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML, SparseLinearDML, KernelDML, CausalForestDML, NonParamDML, DML
from econml.iv.dml import OrthoIV, DMLIV, NonParamDMLIV
from econml.iv.dr import DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV, IntentToTreatDRIV, LinearIntentToTreatDRIV
from econml.orf import DMLOrthoForest, DROrthoForest
from econml.dr import DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from econml.panel.dml import DynamicDML
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner

import inspect
import scipy


class ModelNuisance:
    def __init__(self, model_t, model_y):
        self._model_t = model_t
        self._model_y = model_y

    def train(self, is_selecting, folds, Y, T, W=None):
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
        # pass sample weight to final step of pipeline
        sample_weight_dict = {f'{self.pipeline.steps[-1][0]}__sample_weight': sample_weight}
        return self.pipeline.fit(*args, **kwargs, **sample_weight_dict)

    def predict(self, *args, **kwargs):
        return self.pipeline.predict(*args, **kwargs)

    def predict_proba(self, *args, **kwargs):
        return self.pipeline.predict_proba(*args, **kwargs)


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
        OrthoLearner(discrete_outcome=False, discrete_treatment=False, treatment_featurizer=None,
                     discrete_instrument=None, categories='auto', cv=3, random_state=1,
                     allow_missing=True).fit(y, T, W=W_missing)

        CausalForestDML(model_y=nuisance_model, model_t=nuisance_model,
                        allow_missing=True).fit(y, T, X=X, W=W_missing)

        DynamicDML(model_y=nuisance_model, model_t=nuisance_model,
                   allow_missing=True).fit(y, T, W=W_missing, groups=groups)

        LinearDML(model_y=nuisance_model, model_t=nuisance_model,
                  allow_missing=True).dowhy.fit(y, T, X=X, W=W_missing)

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

        clsf = make_pipeline(SimpleImputer(strategy='mean'), LogisticRegression())
        regr = make_pipeline(SimpleImputer(strategy='mean'), Lasso())
        model_final = make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
        param_model_final = ParametricModelFinalForMissing(make_pipeline(
            SimpleImputer(strategy='mean'), StatsModelsLinearRegression(fit_intercept=False)))
        non_param_model_final = NonParamModelFinal(make_pipeline(SimpleImputer(strategy='mean'), Lasso()))
        non_param_model_final_t = NonParamModelFinal(make_pipeline(
            SimpleImputer(strategy='mean'), LogisticRegression()))
        discrete_treatment = True
        discrete_instrument = True

        # test X, W only
        x_w_missing_models = [
            NonParamDML(model_y=regr, model_t=clsf, model_final=non_param_model_final,
                        discrete_treatment=discrete_treatment, allow_missing=True),
            DML(model_y=regr, model_t=clsf, discrete_treatment=discrete_treatment,
                model_final=param_model_final, allow_missing=True),
            DMLIV(model_y_xw=regr, model_t_xw=clsf, model_t_xwz=clsf,
                  model_final=param_model_final, discrete_treatment=discrete_treatment,
                  discrete_instrument=discrete_instrument, allow_missing=True),
            NonParamDMLIV(model_y_xw=regr, model_t_xw=clsf, model_t_xwz=clsf,
                          model_final=non_param_model_final, discrete_treatment=discrete_treatment,
                          discrete_instrument=discrete_instrument, allow_missing=True),
            DRLearner(model_propensity=clsf, model_regression=regr,
                      model_final=model_final, allow_missing=True)
        ]

        # test W only
        w_missing_models = [
            DRIV(model_y_xw=regr, model_t_xw=clsf, model_z_xw=clsf, model_tz_xw=regr,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='driv', projection=False, allow_missing=True),
            DRIV(model_y_xw=regr, model_t_xw=clsf, model_t_xwz=clsf, model_tz_xw=regr,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='driv', projection=True, allow_missing=True),
            DRIV(model_y_xw=regr, model_t_xw=clsf, model_z_xw=clsf, model_t_xwz=clsf, model_tz_xw=regr,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='dmliv', projection=False, allow_missing=True),
            DRIV(model_y_xw=regr, model_t_xw=clsf, model_t_xwz=clsf, model_tz_xw=regr,
                 discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument,
                 prel_cate_approach='dmliv', projection=True, allow_missing=True),
            IntentToTreatDRIV(model_y_xw=regr, model_t_xwz=clsf, prel_cate_approach='driv',
                              model_final=model_final, allow_missing=True),
            IntentToTreatDRIV(model_y_xw=regr, model_t_xwz=clsf, prel_cate_approach='dmliv',
                              model_final=model_final, allow_missing=True),
            LinearDML(model_y=regr, model_t=clsf, discrete_treatment=True, allow_missing=True),
            SparseLinearDML(model_y=regr, model_t=clsf, discrete_treatment=True, allow_missing=True),
            KernelDML(model_y=regr, model_t=clsf, discrete_treatment=True, allow_missing=True),
            CausalForestDML(model_y=regr, model_t=clsf, discrete_treatment=True, allow_missing=True),
            LinearDRLearner(model_propensity=clsf, model_regression=regr, allow_missing=True),
            SparseLinearDRLearner(model_propensity=clsf, model_regression=regr, allow_missing=True),
            ForestDRLearner(model_propensity=clsf, model_regression=regr, allow_missing=True),
            OrthoIV(model_y_xw=regr, model_t_xw=clsf, model_z_xw=clsf,
                    discrete_treatment=True, discrete_instrument=True, allow_missing=True),
            LinearDRIV(model_y_xw=regr, model_t_xw=clsf, model_z_xw=clsf, model_tz_xw=regr,
                       prel_cate_approach='driv', discrete_treatment=True, discrete_instrument=True,
                       allow_missing=True),
            SparseLinearDRIV(model_y_xw=regr, model_t_xw=clsf, model_z_xw=clsf, model_tz_xw=regr,
                             prel_cate_approach='driv', discrete_treatment=True, discrete_instrument=True,
                             allow_missing=True),
            ForestDRIV(model_y_xw=regr, model_t_xw=clsf, model_z_xw=clsf, model_tz_xw=regr,
                       prel_cate_approach='driv', discrete_treatment=True, discrete_instrument=True,
                       allow_missing=True),
            LinearIntentToTreatDRIV(model_y_xw=regr, model_t_xwz=clsf,
                                    prel_cate_approach='driv', allow_missing=True),
            DMLOrthoForest(model_T=clsf, model_Y=regr, model_T_final=non_param_model_final_t,
                           model_Y_final=non_param_model_final, discrete_treatment=True, allow_missing=True),
            DROrthoForest(propensity_model=clsf, model_Y=regr, propensity_model_final=non_param_model_final_t,
                          model_Y_final=non_param_model_final, allow_missing=True),
        ]

        metalearners = [
            SLearner(overall_model=regr, allow_missing=True),
            TLearner(models=regr, allow_missing=True),
            XLearner(models=regr, propensity_model=clsf, cate_models=regr, allow_missing=True),
            DomainAdaptationLearner(models=regr, final_models=regr,
                                    propensity_model=clsf, allow_missing=True)
        ]

        for est in x_w_missing_models:
            with self.subTest(est=est, kind='missing X and W'):

                if 'Z' in inspect.getfullargspec(est.fit).kwonlyargs:
                    include_Z = True
                else:
                    include_Z = False

                data_dict = create_data_dict(y, T, X, X_missing, W, W_missing, Z,
                                             X_has_missing=True, W_has_missing=True, include_Z=include_Z)
                est.fit(**data_dict)
                # dowhy does not support missing values in X
                self.assertRaises(ValueError, est.dowhy.fit, **data_dict)

                # assert that fitting with missing values fails when allow_missing is False
                # and that setting allow_missing after init still works
                est.allow_missing = False
                self.assertRaises(ValueError, est.fit, **data_dict)
                self.assertRaises(ValueError, est.dowhy.fit, **data_dict)

        for est in w_missing_models:
            with self.subTest(est=est, kind='missing W'):

                if 'Z' in inspect.getfullargspec(est.fit).kwonlyargs:
                    include_Z = True
                else:
                    include_Z = False

                data_dict = create_data_dict(y, T, X, X_missing, W, W_missing, Z,
                                             X_has_missing=False, W_has_missing=True, include_Z=include_Z)
                est.fit(**data_dict)
                est.effect(X)
                est.dowhy.fit(**data_dict)

                # assert that we fail with a value error when we pass missing X to a model that doesn't support it
                data_dict_to_fail = create_data_dict(y, T, X, X_missing, W, W_missing, Z,
                                                     X_has_missing=True, W_has_missing=True, include_Z=include_Z)
                self.assertRaises(ValueError, est.fit, **data_dict_to_fail)
                self.assertRaises(ValueError, est.dowhy.fit, **data_dict_to_fail)

                # assert that fitting with missing values fails when allow_missing is False
                # and that setting allow_missing after init still works
                est.allow_missing = False
                self.assertRaises(ValueError, est.fit, **data_dict)
                self.assertRaises(ValueError, est.dowhy.fit, **data_dict)

        for est in metalearners:
            with self.subTest(est=est, kind='metalearner'):

                data_dict = create_data_dict(y, T, X, X_missing, W, W_missing, Z,
                                             X_has_missing=True, W_has_missing=False, include_Z=False)
                # metalearners don't support W
                data_dict.pop('W')

                # metalearners do support missing values in X
                est.fit(**data_dict)

                # dowhy never supports missing values in X
                self.assertRaises(ValueError, est.dowhy.fit, **data_dict)
