# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from econml._ortho_learner import _OrthoLearner, _crossfit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.model_selection import KFold
import numpy as np
import unittest
import joblib


class TestOrthoLearner(unittest.TestCase):

    def test_crossfit(self):

        class Wrapper:

            def __init__(self, model):
                self._model = model

            def fit(self, X, y, W=None):
                self._model.fit(X, y)
                return self

            def predict(self, X, y, W=None):
                return self._model.predict(X), y - self._model.predict(X), X

        X = np.random.normal(size=(1000, 3))
        y = X[:, 0] + np.random.normal(size=(1000,))
        folds = list(KFold(2).split(X, y))
        model = Lasso(alpha=0.01)
        nuisance, model_list = _crossfit(Wrapper(model),
                                         folds,
                                         X, y, W=y, Z=None)
        np.testing.assert_allclose(nuisance[0][folds[0][1]],
                                   model.fit(X[folds[0][0]], y[folds[0][0]]).predict(X[folds[0][1]]))
        np.testing.assert_allclose(nuisance[0][folds[0][0]],
                                   model.fit(X[folds[0][1]], y[folds[0][1]]).predict(X[folds[0][0]]))

        coef_ = np.zeros(X.shape[1])
        coef_[0] = 1
        [np.testing.assert_allclose(coef_, mdl._model.coef_, rtol=0, atol=0.08) for mdl in model_list]
