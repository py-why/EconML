# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
import pytest
import warnings
from numpy.random import binomial, choice, normal, uniform
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from econml.ortho_forest import ContinuousTreatmentOrthoForest, DiscreteTreatmentOrthoForest, \
    WeightedModelWrapper


class TestOrthoForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        # DGP constants
        cls.n = 1000
        cls.d_w = 30
        cls.support_size = 5
        cls.d_x = 1
        cls.epsilon_sample = lambda n: uniform(-1, 1, size=n)
        cls.eta_sample = lambda n: uniform(-1, 1, size=n)
        cls.support = choice(range(cls.d_w), size=cls.support_size, replace=False)
        cls.coefs_T = uniform(0, 1, size=cls.support_size)
        cls.coefs_Y = uniform(0, 1, size=cls.support_size)
        # Generate data
        cls.X = uniform(0, 1, size=(cls.n, cls.d_x))
        cls.W = normal(0, 1, size=(cls.n, cls.d_w))
        # Test data
        cls.x_test = normal(0, 1, size=(10, cls.d_x))
        cls.x_test[:, 0] = np.arange(0, 1, 0.1)
        cls.expected_exp_te = np.array([cls._exp_te(x) for x in TestOrthoForest.x_test])
        cls.expected_const_te = np.array([cls._const_te(x) for x in TestOrthoForest.x_test])
        # Remove warnings that might be raised by the models passed into the ORF
        warnings.filterwarnings("ignore")

    @pytest.mark.slow
    def test_continuous_treatments(self):
        np.random.seed(123)
        # Generate data with continuous treatments
        T = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_T) + \
            TestOrthoForest.eta_sample(TestOrthoForest.n)
        TE = np.array([self._exp_te(x) for x in TestOrthoForest.X])
        Y = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_Y) + \
            T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        # Instantiate model with most of the default parameters
        est = ContinuousTreatmentOrthoForest(n_jobs=4, n_trees=10,
                                             model_T=WeightedModelWrapper(Lasso(), sample_type="weighted"),
                                             model_Y=WeightedModelWrapper(Lasso(), sample_type="weighted"),
                                             model_T_final=WeightedModelWrapper(LassoCV(), sample_type="weighted"),
                                             model_Y_final=WeightedModelWrapper(LassoCV(), sample_type="weighted"))
        # Test inputs for continuous treatments
        # --> Check that one can pass in regular lists
        est.fit(list(Y), list(T), list(TestOrthoForest.X), list(TestOrthoForest.W))
        # --> Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, est.fit, Y[:TestOrthoForest.n // 2], T[:TestOrthoForest.n // 2],
                          TestOrthoForest.X, TestOrthoForest.W)
        # Check that outputs have the correct shape
        out_te = est.const_marginal_effect(TestOrthoForest.x_test)
        self.assertSequenceEqual((TestOrthoForest.x_test.shape[0], 1), out_te.shape)
        # Test continuous treatments with controls
        est = ContinuousTreatmentOrthoForest(n_trees=50, min_leaf_size=10,
                                             max_splits=50, subsample_ratio=0.30, bootstrap=False, n_jobs=4,
                                             model_T=WeightedModelWrapper(Lasso(alpha=0.024), sample_type="weighted"),
                                             model_Y=WeightedModelWrapper(Lasso(alpha=0.024), sample_type="weighted"),
                                             model_T_final=WeightedModelWrapper(LassoCV(), sample_type="weighted"),
                                             model_Y_final=WeightedModelWrapper(LassoCV(), sample_type="weighted"))
        est.fit(Y, T, TestOrthoForest.X, TestOrthoForest.W)
        self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.5)
        # Test continuous treatments without controls
        T = TestOrthoForest.eta_sample(TestOrthoForest.n)
        Y = T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        est.fit(Y, T, TestOrthoForest.X)
        self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.5)

    @pytest.mark.slow
    def test_binary_treatments(self):
        np.random.seed(123)
        # Generate data with binary treatments
        log_odds = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_T) + \
            TestOrthoForest.eta_sample(TestOrthoForest.n)
        T_sigmoid = 1 / (1 + np.exp(-log_odds))
        T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
        TE = np.array([self._exp_te(x) for x in TestOrthoForest.X])
        Y = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_Y) + \
            T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        # Instantiate model with default params
        est = DiscreteTreatmentOrthoForest(n_trees=10, n_jobs=4,
                                           propensity_model=LogisticRegression(), model_Y=Lasso(),
                                           propensity_model_final=LogisticRegressionCV(penalty='l1', solver='saga'),
                                           model_Y_final=WeightedModelWrapper(LassoCV(), sample_type="weighted"))
        # Test inputs for binary treatments
        # --> Check that one can pass in regular lists
        est.fit(list(Y), list(T), list(TestOrthoForest.X), list(TestOrthoForest.W))
        # --> Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, est.fit, Y[:TestOrthoForest.n // 2], T[:TestOrthoForest.n // 2],
                          TestOrthoForest.X, TestOrthoForest.W)
        # --> Check that it works when T, Y have shape (n, 1)
        est.fit(Y.reshape(-1, 1), T.reshape(-1, 1), TestOrthoForest.X, TestOrthoForest.W)
        # --> Check that it fails correctly when T has shape (n, 2)
        self.assertRaises(ValueError, est.fit, Y, np.ones((TestOrthoForest.n, 2)),
                          TestOrthoForest.X, TestOrthoForest.W)
        # --> Check that it fails correctly when the treatments are not numeric
        self.assertRaises(ValueError, est.fit, Y, np.array(["a"] * TestOrthoForest.n),
                          TestOrthoForest.X, TestOrthoForest.W)
        # Check that outputs have the correct shape
        out_te = est.const_marginal_effect(TestOrthoForest.x_test)
        self.assertSequenceEqual((TestOrthoForest.x_test.shape[0], 1), out_te.shape)
        # Test binary treatments with controls
        est = DiscreteTreatmentOrthoForest(n_trees=100, min_leaf_size=10,
                                           max_splits=30, subsample_ratio=0.30, bootstrap=False, n_jobs=4,
                                           propensity_model=LogisticRegression(C=1 / 0.024, penalty='l1'),
                                           model_Y=Lasso(alpha=0.024),
                                           propensity_model_final=LogisticRegressionCV(penalty='l1', solver='saga'),
                                           model_Y_final=WeightedModelWrapper(LassoCV(), sample_type="weighted"))
        est.fit(Y, T, TestOrthoForest.X, TestOrthoForest.W)
        self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.7, treatment_type='discrete')
        # Test binary treatments without controls
        log_odds = TestOrthoForest.eta_sample(TestOrthoForest.n)
        T_sigmoid = 1 / (1 + np.exp(-log_odds))
        T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
        Y = T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        est.fit(Y, T, TestOrthoForest.X)
        self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.5, treatment_type='discrete')

    @pytest.mark.slow
    def test_multiple_treatments(self):
        np.random.seed(123)
        # Only applicable to continuous treatments
        # Generate data for 2 treatments
        TE = np.array([[TestOrthoForest._exp_te(x), TestOrthoForest._const_te(x)] for x in TestOrthoForest.X])
        coefs_T = uniform(0, 1, size=(TestOrthoForest.support_size, 2))
        T = np.matmul(TestOrthoForest.W[:, TestOrthoForest.support], coefs_T) + \
            uniform(-1, 1, size=(TestOrthoForest.n, 2))
        delta_Y = np.array([np.dot(TE[i], T[i]) for i in range(TestOrthoForest.n)])
        Y = delta_Y + np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_Y) + \
            TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        # Test multiple treatments with controls
        est = ContinuousTreatmentOrthoForest(n_trees=50, min_leaf_size=10,
                                             max_splits=50, subsample_ratio=0.30, bootstrap=False, n_jobs=4,
                                             model_T=WeightedModelWrapper(
                                                 MultiOutputRegressor(Lasso(alpha=0.024)), sample_type="weighted"),
                                             model_Y=WeightedModelWrapper(Lasso(alpha=0.024), sample_type="weighted"),
                                             model_T_final=WeightedModelWrapper(
                                                 MultiOutputRegressor(LassoCV()), sample_type="weighted"),
                                             model_Y_final=WeightedModelWrapper(LassoCV(), sample_type="weighted"))
        est.fit(Y, T, TestOrthoForest.X, TestOrthoForest.W)
        expected_te = np.array([TestOrthoForest.expected_exp_te, TestOrthoForest.expected_const_te]).T
        self._test_te(est, expected_te, tol=0.5, treatment_type='multi')

    def _test_te(self, learner_instance, expected_te, tol, treatment_type='continuous'):
        # Compute the treatment effect on test points
        te_hat = learner_instance.const_marginal_effect(
            TestOrthoForest.x_test
        )
        # Compute treatment effect residuals
        if treatment_type == 'continuous':
            te_res = np.abs(expected_te - te_hat[:, 0])
        elif treatment_type == 'discrete':
            te_res = np.abs(expected_te - te_hat[:, 0])
        else:
            # Multiple treatments
            te_res = np.abs(expected_te - te_hat)
        # Allow at most 10% test points to be outside of the tolerance interval
        self.assertLessEqual(np.mean(te_res > tol), 0.1)

    @classmethod
    def _const_te(cls, x):
        return 2

    @classmethod
    def _exp_te(cls, x):
        return np.exp(x[0] * 2)
