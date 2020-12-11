# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
import shap
from econml.dml import *
from econml.ortho_forest import *
from econml.drlearner import *
from econml.metalearners import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures


class TestShap(unittest.TestCase):
    def test_continuous_t(self):
        n = 100
        d_x = 3
        d_w = 2
        X = np.random.normal(size=(n, d_x))
        W = np.random.normal(size=(n, d_w))
        for d_t in [2, 1]:
            for d_y in [3, 1]:
                Y = np.random.normal(size=(n, d_y))
                Y = Y.flatten() if d_y == 1 else Y
                T = np.random.normal(size=(n, d_t))
                T = T.flatten() if d_t == 1 else T
                for featurizer in [None, PolynomialFeatures(degree=2, include_bias=False)]:

                    est_list = [LinearDML(model_y=LinearRegression(),
                                          model_t=LinearRegression(), featurizer=featurizer)]
                    if d_t == 1:
                        est_list += [
                            NonParamDML(model_y=LinearRegression(
                            ), model_t=LinearRegression(), model_final=RandomForestRegressor(), featurizer=featurizer),
                            ForestDML(model_y=LinearRegression(), model_t=LinearRegression())]
                    if d_y == 1:
                        est_list += [DMLOrthoForest()]
                    for est in est_list:
                        with self.subTest(est=est, featurizer=featurizer, d_y=d_y, d_t=d_t):
                            fd_x = featurizer.fit_transform(X).shape[1] if featurizer is not None else d_x
                            est.fit(Y, T, X, W)
                            shap_values = est.shap_values(X[:10], feature_names=["a", "b", "c"])

                            # test base values equals to mean of constant marginal effect
                            if not isinstance(est, (ForestDML, DMLOrthoForest)):
                                mean_cate = est.const_marginal_effect(X[:10]).mean(axis=0)
                                mean_cate = mean_cate.flatten()[0] if not np.isscalar(mean_cate) else mean_cate
                                self.assertAlmostEqual(shap_values["Y0"]["T0"].base_values[0], mean_cate, delta=1e-2)

                            if isinstance(est, (ForestDML, DMLOrthoForest)):
                                fd_x = d_x

                            # test shape of shap values output is as expected
                            self.assertEqual(len(shap_values["Y0"]), d_t)
                            self.assertEqual(len(shap_values), d_y)
                            # test shape of attribute of explanation object is as expected
                            self.assertEqual(shap_values["Y0"]["T0"].values.shape, (10, fd_x))
                            self.assertEqual(shap_values["Y0"]["T0"].data.shape, (10, fd_x))
                            self.assertEqual(shap_values["Y0"]["T0"].base_values.shape, (10,))
                            ind = 6
                            self.assertEqual(len(shap_values["Y0"]["T0"].feature_names), fd_x)
                            self.assertEqual(len(shap_values["Y0"]["T0"][ind].feature_names), fd_x)
                            # test shap could generate the plot from the shap_values
                            shap.plots.force(shap_values["Y0"]["T0"][ind], show=False)
                            shap.plots.beeswarm(shap_values["Y0"]["T0"], show=False)

    def test_discrete_t(self):
        n = 100
        d_x = 3
        d_w = 2
        X = np.random.normal(size=(n, d_x))
        W = np.random.normal(size=(n, d_w))
        for d_t in [3, 2]:
            for d_y in [3, 1]:
                Y = np.random.normal(size=(n, d_y))
                Y = Y.flatten() if d_y == 1 else Y
                T = np.random.choice(range(d_t), size=(n,))
                for featurizer in [None, PolynomialFeatures(degree=2, include_bias=False)]:

                    est_list = [LinearDML(featurizer=featurizer, discrete_treatment=True),
                                TLearner(models=RandomForestRegressor()),
                                SLearner(overall_model=RandomForestRegressor()),
                                XLearner(models=RandomForestRegressor()),
                                DomainAdaptationLearner(models=RandomForestRegressor(),
                                                        final_models=RandomForestRegressor())
                                ]
                    if d_t == 2:
                        est_list += [
                            NonParamDML(model_y=LinearRegression(
                            ), model_t=LogisticRegression(), model_final=RandomForestRegressor(),
                                featurizer=featurizer, discrete_treatment=True),
                            ForestDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                                      discrete_treatment=True)]
                    if d_y == 1:
                        est_list += [DRLearner(multitask_model_final=True, featurizer=featurizer),
                                     DRLearner(multitask_model_final=False, featurizer=featurizer),
                                     ForestDRLearner(),
                                     DROrthoForest()]
                    for est in est_list:
                        with self.subTest(est=est, featurizer=featurizer, d_y=d_y, d_t=d_t):
                            fd_x = featurizer.fit_transform(X).shape[1] if featurizer is not None else d_x
                            if isinstance(est, (TLearner, SLearner, XLearner, DomainAdaptationLearner)):
                                est.fit(Y, T, X)
                            else:
                                est.fit(Y, T, X, W)
                            shap_values = est.shap_values(X[:10], feature_names=["a", "b", "c"])

                            # test base values equals to mean of constant marginal effect
                            if not isinstance(est, (ForestDML, ForestDRLearner, DROrthoForest)):
                                mean_cate = est.const_marginal_effect(X[:10]).mean(axis=0)
                                mean_cate = mean_cate.flatten()[0] if not np.isscalar(mean_cate) else mean_cate
                                self.assertAlmostEqual(shap_values["Y0"]["T0"].base_values[0], mean_cate, delta=1e-2)

                            if isinstance(est, (TLearner, SLearner, XLearner, DomainAdaptationLearner, ForestDML,
                                                ForestDRLearner, DROrthoForest)):
                                fd_x = d_x
                            # test shape of shap values output is as expected
                            self.assertEqual(len(shap_values["Y0"]), d_t - 1)
                            self.assertEqual(len(shap_values), d_y)
                            # test shape of attribute of explanation object is as expected
                            self.assertEqual(shap_values["Y0"]["T0"].values.shape, (10, fd_x))
                            self.assertEqual(shap_values["Y0"]["T0"].data.shape, (10, fd_x))
                            self.assertEqual(shap_values["Y0"]["T0"].base_values.shape, (10,))
                            ind = 6
                            self.assertEqual(len(shap_values["Y0"]["T0"].feature_names), fd_x)
                            self.assertEqual(len(shap_values["Y0"]["T0"][ind].feature_names), fd_x)
                            # test shap could generate the plot from the shap_values
                            shap.plots.force(shap_values["Y0"]["T0"][ind], show=False)
                            shap.plots.beeswarm(shap_values["Y0"]["T0"], show=False)
