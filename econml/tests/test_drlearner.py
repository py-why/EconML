# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from numpy.random import normal, multivariate_normal, binomial
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from econml.drlearner import DRLearner


class TestDRLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set random seed
        cls.random_state = np.random.RandomState(12345)
        # Generate data
        # DGP constants
        cls.d = 5
        cls.n = 1000
        cls.n_test = 200
        cls.beta = np.array([0.25, -0.38, 1.41, 0.50, -1.22])
        cls.heterogeneity_index = 1
        # Test data
        cls.X_test = cls.random_state.multivariate_normal(
            np.zeros(cls.d),
            np.diag(np.ones(cls.d)),
            cls.n_test)
        # Constant treatment effect and propensity
        cls.const_te_data = TestDRLearner._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestDRLearner._const_te,
            propensity=lambda x: 0.3)
        # Heterogeneous treatment and propensity
        cls.heterogeneous_te_data = TestDRLearner._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestDRLearner._heterogeneous_te,
            propensity=lambda x: (0.8 if (x[2] > -0.5 and x[2] < 0.5) else 0.2))

    def test_DRLearner(self):
        """Tests whether the DRLearner can accurately estimate constant and
           heterogeneous treatment effects.
        """
        # Instantiate DomainAdaptationLearner
        DR_learner = DRLearner(model_regression=LinearRegression(),
                               model_final=LinearRegression())
        # Test inputs
        self._test_inputs(DR_learner)
        # Test constant treatment effect
        self._test_te(DR_learner, tol=0.5, te_type="const")
        # Test heterogeneous treatment effect
        outcome_model = Pipeline([('poly', PolynomialFeatures()), ('model', LinearRegression())])
        DR_learner = DRLearner(model_regression=outcome_model,
                               model_final=LinearRegression())
        self._test_te(DR_learner, tol=0.5, te_type="heterogeneous")
        # Test heterogenous treatment effect for W =/= None
        self._test_with_W(DR_learner, tol=0.5)

    def _test_te(self, learner_instance, tol, te_type="const"):
        if te_type not in ["const", "heterogeneous"]:
            raise ValueError("Type of treatment effect must be 'const' or 'heterogeneous'.")
        X, T, Y = getattr(TestDRLearner, "{te_type}_te_data".format(te_type=te_type))
        te_func = getattr(TestDRLearner, "_{te_type}_te".format(te_type=te_type))
        # Fit learner and get the effect
        learner_instance.fit(Y, T, X)
        te_hat = learner_instance.effect(TestDRLearner.X_test)
        # Get the true treatment effect
        te = np.apply_along_axis(te_func, 1, TestDRLearner.X_test)
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)

    def _test_with_W(self, learner_instance, tol):
        # Only for heterogeneous TE
        X, T, Y = TestDRLearner.heterogeneous_te_data
        # Fit learner on X and W and get the effect
        learner_instance.fit(Y, T, X=X[:, [TestDRLearner.heterogeneity_index]], W=X)
        te_hat = learner_instance.effect(TestDRLearner.X_test[:, [TestDRLearner.heterogeneity_index]])
        # Get the true treatment effect
        te = np.apply_along_axis(TestDRLearner._heterogeneous_te, 1, TestDRLearner.X_test)
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)

    def _test_inputs(self, learner_instance):
        X, T, Y = TestDRLearner.const_te_data
        # Check that one can pass in regular lists
        learner_instance.fit(list(Y), list(T), list(X))
        learner_instance.effect(list(TestDRLearner.X_test))
        # Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, learner_instance.fit, Y, T, X[:TestDRLearner.n // 2])
        self.assertRaises(ValueError, learner_instance.fit, Y[:TestDRLearner.n // 2], T, X)
        # Check that it fails when T contains values other than 0 and 1
        self.assertRaises(ValueError, learner_instance.fit, Y, T + 1, X)
        # Check that it works when T, Y have shape (n, 1)
        self.assertWarns(DataConversionWarning,
                         learner_instance.fit, Y.reshape(-1, 1), T.reshape(-1, 1), X
                         )

    @classmethod
    def _untreated_outcome(cls, x):
        return np.dot(x, cls.beta) + cls.random_state.normal(0, 1)

    @classmethod
    def _const_te(cls, x):
        return 2

    @classmethod
    def _heterogeneous_te(cls, x):
        return x[cls.heterogeneity_index]

    @classmethod
    def _generate_data(cls, n, d, untreated_outcome, treatment_effect, propensity):
        """Generates population data for given untreated_outcome, treatment_effect and propensity functions.

        Parameters
        ----------
            n (int): population size
            d (int): number of covariates
            untreated_outcome (func): untreated outcome conditional on covariates
            treatment_effect (func): treatment effect conditional on covariates
            propensity (func): probability of treatment conditional on covariates
        """
        # Generate covariates
        X = cls.random_state.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
        # Generate treatment
        T = np.apply_along_axis(lambda x: cls.random_state.binomial(1, propensity(x), 1)[0], 1, X)
        # Calculate outcome
        Y0 = np.apply_along_axis(lambda x: untreated_outcome(x), 1, X)
        treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
        Y = Y0 + treat_effect * T
        return (X, T, Y)
