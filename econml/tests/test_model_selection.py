# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np
from scipy.special import expit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (ElasticNetCV, Lasso, LassoCV, LinearRegression, LogisticRegression,
                                  LogisticRegressionCV, MultiTaskElasticNetCV, MultiTaskLassoCV,
                                  RidgeCV, RidgeClassifierCV)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from econml.dml import LinearDML
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from econml.utilities import SeparateModel
from econml.dr import LinearDRLearner


class TestModelSelection(unittest.TestCase):

    def _simple_dgp(self, n, d_x, d_w, discrete_treatment):
        n = 500  # keep the data set small since we're testing a lot of models and don't care about the results
        X = np.random.normal(size=(n, d_x))
        W = np.random.normal(size=(n, d_w))
        alpha = np.random.normal(size=(X.shape[1]))
        n_f = d_w + d_x
        beta = np.random.normal(size=(n_f,))
        gamma = np.random.normal(size=(n_f,))
        XW = np.hstack([X, W])
        if discrete_treatment:
            T = np.random.binomial(1, expit(XW @ beta))
        else:
            T = XW @ beta + np.random.normal(size=(n,))
        Y = (X @ alpha) * T + XW @ gamma + np.random.normal(size=(n,))
        return Y, T, X, W

    def test_poly(self):
        # tests that we can recover the right degree of polynomial features
        # implicitly also tests ability to handle pipelines
        # since 'poly' uses pipelines containing PolynomialFeatures
        n = 5000
        X = np.random.normal(size=(n, 2))
        W = np.random.normal(size=(n, 3))

        for true_d in range(1, 4):
            with self.subTest(true_d=true_d):
                pf = PolynomialFeatures(degree=true_d)

                fts = pf.fit_transform(np.hstack([X, W]))
                k = fts.shape[1]
                m = X.shape[1] + W.shape[1]

                alpha_x = np.random.normal(size=(X.shape[1],))
                alpha_1 = np.random.normal(size=())
                beta = np.random.normal(size=(k,))
                gamma = np.random.normal(size=(k,))

                # generate larger coefficients in a set of high degree features,
                # weighted towards higher degree features
                ft_inds_beta = np.random.choice(k, size=m, replace=False, p=np.arange(k) / np.sum(np.arange(k)))
                ft_inds_gamma = np.random.choice(k, size=m, replace=False, p=np.arange(k) / np.sum(np.arange(k)))

                beta[ft_inds_beta] = 10 * np.random.normal(1, size=(m,))
                gamma[ft_inds_gamma] = 10 * np.random.normal(1, size=(m,))

                t = np.random.normal(size=(n,)) + fts @ beta + np.random.normal(scale=0.5, size=(n,))
                y = np.random.normal(size=(n,)) + t * (alpha_1 + X @ alpha_x) + fts @ gamma

                # just test a polynomial T model, since for Y the correct degree also depends on
                # the interation of T and X
                mdl = LinearDML(model_t='poly',
                                model_y=LinearRegression()).fit(y, t, X=X, W=W)
                for t in mdl.models_t[0]:
                    self.assertEqual(t[0].degree, true_d)

    def test_all_strings(self):
        for discrete_treatment in [True, False]:
            Y, T, X, W = self._simple_dgp(500, 2, 3, discrete_treatment)
            for model_t in ['auto', 'linear', 'poly', 'forest', 'gbf', 'nnet', 'automl']:
                with self.subTest(model_t=model_t, discrete_treatment=discrete_treatment):
                    mdl = LinearDML(model_t=model_t,
                                    discrete_treatment=discrete_treatment,
                                    model_y=LinearRegression())
                    mdl.fit(Y, T, X=X, W=W)

            model_t = 'some_random_string'
            with self.subTest(model_t=model_t, discrete_treatment=True):
                mdl = LinearDML(model_t=model_t,
                                discrete_treatment=discrete_treatment,
                                model_y=LinearRegression())
                with self.assertRaises(ValueError):
                    mdl.fit(Y, T, X=X, W=W)

    def test_list_selection(self):
        Y, T, X, W = self._simple_dgp(500, 2, 3, False)

        # test corner case with just one model in a list
        mdl = LinearDML(model_t=[LinearRegression()],
                        model_y=LinearRegression())
        mdl.fit(Y, T, X=X, W=W)

        # test corner case with empty list
        with self.assertRaises(Exception):
            mdl = LinearDML(model_t=[],
                            model_y=LinearRegression())
            mdl.fit(Y, T, X=X, W=W)

        # test selecting between two fixed models
        mdl = LinearDML(model_t=[LinearRegression(), RandomForestRegressor()],
                        model_y=LinearRegression())
        mdl.fit(Y, T, X=X, W=W)
        # DGP is a linear model, so linear regression should fit better
        assert isinstance(mdl.models_t[0][0], LinearRegression)

        T2 = T + 10 * (X[:, 1] > 0)  # add a non-linear effect
        mdl.fit(Y, T2, X=X, W=W)
        # DGP is now non-linear, so random forest should fit better
        assert isinstance(mdl.models_t[0][0], RandomForestRegressor)

    def test_sklearn_model_selection(self):
        for is_discrete, mdls in [(True, [LogisticRegressionCV(), RidgeClassifierCV(),
                                          GridSearchCV(LogisticRegression(), {'C': [1, 10]}),
                                          RandomizedSearchCV(LogisticRegression(), {'C': [1, 10]})]),
                                  (False, [ElasticNetCV(), LassoCV(), RidgeCV(),
                                           MultiTaskElasticNetCV(), MultiTaskLassoCV(), WeightedLassoCVWrapper(),
                                           GridSearchCV(Lasso(), {'alpha': [0.1, 1]}),
                                           RandomizedSearchCV(Lasso(), {'alpha': [0.1, 1]})])]:
            Y, T, X, W = self._simple_dgp(500, 2, 3, is_discrete)
            T2 = np.tile(T.reshape(-1, 1), (1, 2))  # multi-column T
            for mdl in mdls:
                # these models only work on multi-output data
                use_array = isinstance(mdl, (MultiTaskElasticNetCV, MultiTaskLassoCV))
                with self.subTest(model=mdl):
                    est = LinearDML(model_t=mdl,
                                    discrete_treatment=is_discrete,
                                    model_y=LinearRegression())
                    est.fit(Y, T2 if use_array else T, X=X, W=W)

    def test_fixed_model_scoring(self):
        Y, T, X, W = self._simple_dgp(500, 2, 3, True)

        # SeparatedModel doesn't support scoring; that should be fine when not compared to other models
        mdl = LinearDRLearner(model_regression=SeparateModel(LassoCV(), LassoCV()),
                              model_propensity=LogisticRegressionCV())
        mdl.fit(Y, T, X=X, W=W)

        # on the other hand, when we need to compare the score to other models, it should raise an error
        with self.assertRaises(Exception):
            mdl = LinearDRLearner(model_regression=[SeparateModel(LassoCV(), LassoCV()), Lasso()],
                                  model_propensity=LogisticRegressionCV())
            mdl.fit(Y, T, X=X, W=W)
