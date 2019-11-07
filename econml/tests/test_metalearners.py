# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from numpy.random import normal, multivariate_normal, binomial
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from econml.metalearners import *


class TestMetalearners(unittest.TestCase):

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
        cls.const_te_data = TestMetalearners._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestMetalearners._const_te,
            propensity=lambda x: 0.3)
        # Heterogeneous treatment and propensity
        cls.heterogeneous_te_data = TestMetalearners._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestMetalearners._heterogeneous_te,
            propensity=lambda x: (0.8 if (x[2] > -0.5 and x[2] < 0.5) else 0.2))

    def test_TLearner(self):
        """Tests whether the TLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # TLearner test
        # Instantiate TLearner
        T_learner = TLearner(models=LinearRegression())
        # Test inputs
        self._test_inputs(T_learner, T0=3, T1=5)
        # Test constant treatment effect
        self._test_te(T_learner, T0=3, T1=5, tol=0.5, te_type="const")
        # Test heterogeneous treatment effect
        self._test_te(T_learner, T0=3, T1=5, tol=0.5, te_type="heterogeneous")

    def test_SLearner(self):
        """Tests whether the SLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # Instantiate SLearner
        S_learner = SLearner(overall_model=LinearRegression())
        # Test inputs
        self._test_inputs(S_learner, T0=3, T1=5)
        # Test constant treatment effect
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="const")
        # Test heterogeneous treatment effect
        # Need interactions between T and features
        overall_model = Pipeline([('poly', PolynomialFeatures()), ('model', LinearRegression())])
        S_learner = SLearner(overall_model=overall_model)
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="heterogeneous")

    def test_XLearner(self):
        """Tests whether the XLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # Instantiate XLearner
        X_learner = XLearner(models=LinearRegression())
        # Test inputs
        self._test_inputs(X_learner, T0=3, T1=5)
        # Test constant treatment effect
        self._test_te(X_learner, T0=3, T1=5, tol=0.5, te_type="const")
        # Test heterogeneous treatment effect
        self._test_te(X_learner, T0=3, T1=5, tol=0.5, te_type="heterogeneous")

    def test_DALearner(self):
        """Tests whether the DomainAdaptationLearner can accurately estimate constant and
           heterogeneous treatment effects.
        """
        # Instantiate DomainAdaptationLearner
        DA_learner = DomainAdaptationLearner(models=LinearRegression(),
                                             final_models=LinearRegression())
        # Test inputs
        self._test_inputs(DA_learner, T0=3, T1=5)
        # Test constant treatment effect
        self._test_te(DA_learner, T0=3, T1=5, tol=0.5, te_type="const")
        # Test heterogeneous treatment effect
        self._test_te(DA_learner, T0=3, T1=5, tol=0.5, te_type="heterogeneous")

    def _test_te(self, learner_instance, T0, T1, tol, te_type="const"):
        if te_type not in ["const", "heterogeneous"]:
            raise ValueError("Type of treatment effect must be 'const' or 'heterogeneous'.")
        X, T, Y = getattr(TestMetalearners, "{te_type}_te_data".format(te_type=te_type))
        te_func = getattr(TestMetalearners, "_{te_type}_te".format(te_type=te_type))
        # Fit learner and get the effect and marginal effect
        learner_instance.fit(Y, T, X)
        te_hat = learner_instance.effect(TestMetalearners.X_test, T0, T1)
        marginal_te_hat = learner_instance.marginal_effect(TestMetalearners.X_test, T1)
        # Get the true treatment effect
        te = np.apply_along_axis(te_func, 1, TestMetalearners.X_test) * (T1 - T0)
        marginal_te = np.apply_along_axis(te_func, 1, TestMetalearners.X_test) * (T1 - T.min())
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        marginal_te_res = np.abs(marginal_te - marginal_te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)
        self.assertGreaterEqual(np.mean(marginal_te_res < tol), 0.90)

    def _test_inputs(self, learner_instance, T0, T1):
        X, T, Y = TestMetalearners.const_te_data
        # Check that one can pass in regular lists
        learner_instance.fit(list(Y), list(T), list(X))
        learner_instance.effect(list(TestMetalearners.X_test), T0, T1)
        # Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, learner_instance.fit, Y, T, X[:TestMetalearners.n // 2])
        self.assertRaises(ValueError, learner_instance.fit, Y[:TestMetalearners.n // 2], T, X)
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
        """
        # Generate covariates
        X = cls.random_state.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
        # Generate treatment
        T = cls.random_state.choice([1, 3, 5], size=n, p=[0.2, 0.3, 0.5])
        # Calculate outcome
        Y0 = np.apply_along_axis(lambda x: untreated_outcome(x), 1, X)
        treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
        Y = Y0 + treat_effect * T
        return (X, T, Y)
