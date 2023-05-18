# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np
import scipy.special
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils import check_random_state

class TestDRLearner(unittest.TestCase):
    def test_default_models(self):
        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.001
        y = (1 + 0.5 * X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner()
        est.fit(y, T, X=X, W=None)
        assert est.const_marginal_effect(X[:2]).shape == (2, 2)
        assert est.effect(X[:2], T0=0, T1=1).shape == (2,)
        assert isinstance(est.score_, float)
        assert isinstance(est.score(y, T, X=X), float)
        assert len(est.model_cate(T=1).coef_.shape) == 1
        assert len(est.model_cate(T=2).coef_.shape) == 1
        assert isinstance(est.cate_feature_names(), list)
        assert isinstance(est.models_regression[0][0].coef_, np.ndarray)
        assert isinstance(est.models_propensity[0][0].coef_, np.ndarray)

    def test_custom_models(self):
        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.01
        y = (1 + 0.5 * X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner(
            model_propensity=RandomForestClassifier(n_estimators=100, min_samples_leaf=10),
            model_regression=RandomForestRegressor(n_estimators=100, min_samples_leaf=10),
            model_final=LassoCV(cv=3),
            featurizer=None
        )
        est.fit(y, T, X=X, W=None)
        assert isinstance(est.score_, float)
        assert est.const_marginal_effect(X[:3]).shape == (3, 2)
        assert len(est.model_cate(T=2).coef_.shape) == 1
        assert isinstance(est.model_cate(T=2).intercept_, float)
        assert len(est.model_cate(T=1).coef_.shape) == 1
        assert isinstance(est.model_cate(T=1).intercept_, float)

    def test_cv_splitting_strategy(self):
        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.001
        y = (1 + 0.5 * X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner(cv=2)
        est.fit(y, T, X=X, W=None)
        assert est.const_marginal_effect(X[:2]).shape == (2, 2)

    def test_mc_iters(self):
        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.001
        y = (1 + 0.5 * X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner()
        est.fit(y, T, X=X, W=None, inference='bootstrap', n_bootstrap_samples=50)

        self.assertAlmostEqual(est.effect(X[:2], T0=0, T1=1, inference='bootstrap', n_bootstrap_samples=50).shape[0], 50)
        self.assertAlmostEqual(est.effect_interval(X[:2], T0=0, T1=1, alpha=0.05, inference='bootstrap',
                                                   n_bootstrap_samples=50).shape, (2, 50, 2))
        self.assertAlmostEqual(est.ortho_summary(X[:2], T0=0, T1=1, inference='bootstrap',
                                                 n_bootstrap_samples=50).shape, (2, 2, 5))
        self.assertAlmostEqual(est.ortho_intervals(X[:2], T0=0, T1=1, inference='bootstrap', n_bootstrap_samples=50,
                                                   method='normal').shape, (2, 2, 2, 2))

    def test_score(self):
        np.random.seed(123)
        y = np.random.normal(size=(1000,))
        T = np.random.binomial(2, 0.5, size=(1000,))
        X = np.random.normal(size=(1000, 3))
        est = DRScorer()
        est.fit(y, T, X=X, W=None)
        score = est.score()
        self.assertAlmostEqual(score, 0.05778546)
        # Test using baseline method (e.g., RScorer)
        # Replace the following lines with the appropriate baseline method and its parameters
        baseline_est = BaselineScorer()
        baseline_est.fit(y, T, X=X, W=None)
        score_baseline = baseline_est.score()
        # Perform the comparison test
        self.assertAlmostEqual(score_dr, score_baseline)
