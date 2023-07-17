# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
import pytest
import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML, CausalForestDML
from econml.panel.dml import DynamicDML


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


class OrthoLearner(_OrthoLearner):
    def _gen_ortho_learner_model_nuisance(self):
        return ModelNuisance(
            make_pipeline(SimpleImputer(strategy='mean'), LinearRegression()),
            make_pipeline(SimpleImputer(strategy='mean'), LinearRegression())
        )

    def _gen_ortho_learner_model_final(self):
        return ModelFinal()


class TestTreatmentFeaturization(unittest.TestCase):

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
                     categories='auto', cv=3, random_state=1).fit(y, T, W=W_missing)

        CausalForestDML(model_y=nuisance_model, model_t=nuisance_model).fit(y, T, X=X, W=W_missing)

        DynamicDML(model_y=nuisance_model, model_t=nuisance_model).fit(y, T, W=W_missing, groups=groups)

        LinearDML(model_y=nuisance_model, model_t=nuisance_model).dowhy.fit(y, T, X=X, W=W_missing)
