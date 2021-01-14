# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
import shap
from shap.plots import scatter, heatmap, bar, beeswarm, waterfall, force
from econml.dml import *
from econml.orf import *
from econml.dr import *
from econml.metalearners import *
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
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
                                          model_t=LinearRegression(), featurizer=featurizer),
                                CausalForestDML(model_y=LinearRegression(), model_t=LinearRegression())]
                    if d_t == 1:
                        est_list += [
                            NonParamDML(model_y=LinearRegression(
                            ), model_t=LinearRegression(), model_final=RandomForestRegressor(), featurizer=featurizer),
                        ]
                    for est in est_list:
                        with self.subTest(est=est, featurizer=featurizer, d_y=d_y, d_t=d_t):
                            fd_x = featurizer.fit_transform(X).shape[1] if featurizer is not None else d_x
                            est.fit(Y, T, X, W)
                            shap_values = est.shap_values(X[:10], feature_names=["a", "b", "c"],
                                                          background_samples=None)

                            # test base values equals to mean of constant marginal effect
                            if not isinstance(est, (CausalForestDML, DMLOrthoForest)):
                                mean_cate = est.const_marginal_effect(X[:10]).mean(axis=0)
                                mean_cate = mean_cate.flatten()[0] if not np.isscalar(mean_cate) else mean_cate
                                self.assertAlmostEqual(shap_values["Y0"]["T0"].base_values[0], mean_cate, delta=1e-2)

                            if isinstance(est, (CausalForestDML, DMLOrthoForest)):
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
                                                        final_models=RandomForestRegressor()),
                                CausalForestDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                                                discrete_treatment=True)
                                ]
                    if d_t == 2:
                        est_list += [
                            NonParamDML(model_y=LinearRegression(
                            ), model_t=LogisticRegression(), model_final=RandomForestRegressor(),
                                featurizer=featurizer, discrete_treatment=True)]
                    if d_y == 1:
                        est_list += [DRLearner(multitask_model_final=True, featurizer=featurizer),
                                     DRLearner(multitask_model_final=False, featurizer=featurizer),
                                     ForestDRLearner()]
                    for est in est_list:
                        with self.subTest(est=est, featurizer=featurizer, d_y=d_y, d_t=d_t):
                            fd_x = featurizer.fit_transform(X).shape[1] if featurizer is not None else d_x
                            if isinstance(est, (TLearner, SLearner, XLearner, DomainAdaptationLearner)):
                                est.fit(Y, T, X)
                            else:
                                est.fit(Y, T, X, W)
                            shap_values = est.shap_values(X[:10], feature_names=["a", "b", "c"],
                                                          background_samples=None)

                            # test base values equals to mean of constant marginal effect
                            if not isinstance(est, (CausalForestDML, ForestDRLearner, DROrthoForest)):
                                mean_cate = est.const_marginal_effect(X[:10]).mean(axis=0)
                                mean_cate = mean_cate.flatten()[0] if not np.isscalar(mean_cate) else mean_cate
                                self.assertAlmostEqual(shap_values["Y0"]["T0"].base_values[0], mean_cate, delta=1e-2)

                            if isinstance(est, (TLearner, SLearner, XLearner, DomainAdaptationLearner, CausalForestDML,
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

    def test_identical_output(self):
        # Treatment effect function
        def exp_te(x):
            return np.exp(2 * x[0])
        n = 500
        n_w = 10
        support_size = 5
        n_x = 2
        # Outcome support
        support_Y = np.random.choice(range(n_w), size=support_size, replace=False)
        coefs_Y = np.random.uniform(0, 1, size=(support_size,))

        def epsilon_sample(n):
            return np.random.uniform(-1, 1, size=(n,))
        # Treatment support
        support_T = support_Y
        coefs_T = np.random.uniform(0, 1, size=support_size)

        def eta_sample(n):
            return np.random.uniform(-1, 1, size=n)
        # Generate controls, covariates, treatments and outcomes
        W = np.random.normal(0, 1, size=(n, n_w))
        X = np.random.uniform(0, 1, size=(n, n_x))
        # Heterogeneous treatment effects
        TE = np.array([np.exp(2 * x_i[0]) for x_i in X]).flatten()
        T = np.dot(W[:, support_T], coefs_T) + eta_sample(n)
        Y = (TE * T) + np.dot(W[:, support_Y], coefs_Y) + epsilon_sample(n)
        Y = np.tile(Y.reshape(-1, 1), (1, 2))
        est = LinearDML(model_y=Lasso(),
                        model_t=Lasso(),
                        random_state=123,
                        fit_cate_intercept=True,
                        featurizer=PolynomialFeatures(degree=2, include_bias=False))
        est.fit(Y, T, X=X, W=W)
        shap_values1 = est.shap_values(X[:10], feature_names=["A", "B"], treatment_names=["orange"],
                                       background_samples=None)
        est = LinearDML(model_y=Lasso(),
                        model_t=Lasso(),
                        random_state=123,
                        fit_cate_intercept=True,
                        featurizer=PolynomialFeatures(degree=2, include_bias=False))
        est.fit(Y[:, 0], T, X=X, W=W)
        shap_values2 = est.shap_values(X[:10], feature_names=["A", "B"], treatment_names=["orange"],
                                       background_samples=None)
        np.testing.assert_allclose(shap_values1["Y0"]["orange"].data,
                                   shap_values2["Y0"]["orange"].data)
        np.testing.assert_allclose(shap_values1["Y0"]["orange"].values,
                                   shap_values2["Y0"]["orange"].values)
        # TODO There is a matrix dimension mismatch between multiple outcome and single outcome, should solve that
        # through shap package.
        np.testing.assert_allclose(shap_values1["Y0"]["orange"].main_effects,
                                   shap_values2["Y0"]["orange"].main_effects)
        np.testing.assert_allclose(shap_values1["Y0"]["orange"].base_values,
                                   shap_values2["Y0"]["orange"].base_values)

        # test shap could generate the plot from the shap_values
        heatmap(shap_values1["Y0"]["orange"], show=False)
        waterfall(shap_values1["Y0"]["orange"][6], show=False)
        scatter(shap_values1["Y0"]["orange"][:, "A"], show=False)
        bar(shap_values1["Y0"]["orange"], show=False)
        beeswarm(shap_values1["Y0"]["orange"], show=False)
