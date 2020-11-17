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
import econml.tests.utilities  # bugfix for assertWarns


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
        cls.heterogeneity_index = 1
        # Test data
        cls.X_test = cls.random_state.multivariate_normal(
            np.zeros(cls.d),
            np.diag(np.ones(cls.d)),
            cls.n_test)
        # Constant treatment effect
        cls.const_te_data = TestMetalearners._generate_data(
            cls.n, cls.d, beta=cls.random_state.uniform(0, 1, cls.d),
            treatment_effect=TestMetalearners._const_te, multi_y=False)
        # Constant treatment with multi output Y
        cls.const_te_multiy_data = TestMetalearners._generate_data(
            cls.n, cls.d, beta=cls.random_state.uniform(0, 1, size=(cls.d, 2)),
            treatment_effect=TestMetalearners._const_te, multi_y=True)
        # Heterogeneous treatment
        cls.heterogeneous_te_data = TestMetalearners._generate_data(
            cls.n, cls.d, beta=cls.random_state.uniform(0, 1, cls.d),
            treatment_effect=TestMetalearners._heterogeneous_te, multi_y=False)
        # Heterogeneous treatment with multi output Y
        cls.heterogeneous_te_multiy_data = TestMetalearners._generate_data(
            cls.n, cls.d, beta=cls.random_state.uniform(0, 1, size=(cls.d, 2)),
            treatment_effect=TestMetalearners._heterogeneous_te, multi_y=True)

    def test_TLearner(self):
        """Tests whether the TLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # TLearner test
        # Instantiate TLearner
        T_learner = TLearner(models=LinearRegression())
        # Test inputs
        self._test_inputs(T_learner, T0=3, T1=5)
        # Test constant and heterogeneous treatment effect, single and multi output y
        for te_type in ["const", "heterogeneous"]:
            for multi_y in [False, True]:
                self._test_te(T_learner, T0=3, T1=5, tol=0.5, te_type=te_type, multi_y=multi_y)

    def test_SLearner(self):
        """Tests whether the SLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # Instantiate SLearner
        S_learner = SLearner(overall_model=LinearRegression())
        # Test inputs
        self._test_inputs(S_learner, T0=3, T1=5)
        # Test constant treatment effect
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="const", multi_y=False)
        # Test constant treatment effect with multi output Y
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="const", multi_y=True)
        # Test heterogeneous treatment effect
        # Need interactions between T and features
        overall_model = Pipeline([('poly', PolynomialFeatures()), ('model', LinearRegression())])
        S_learner = SLearner(overall_model=overall_model)
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="heterogeneous", multi_y=False)
        # Test heterogeneous treatment effect with multi output Y
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="heterogeneous", multi_y=True)

    def test_XLearner(self):
        """Tests whether the XLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # Instantiate XLearner
        X_learner = XLearner(models=LinearRegression())
        # Test inputs
        self._test_inputs(X_learner, T0=3, T1=5)
        # Test constant and heterogeneous treatment effect, single and multi output y
        for te_type in ["const", "heterogeneous"]:
            for multi_y in [False, True]:
                self._test_te(X_learner, T0=3, T1=5, tol=0.5, te_type=te_type, multi_y=multi_y)

    def test_DALearner(self):
        """Tests whether the DomainAdaptationLearner can accurately estimate constant and
           heterogeneous treatment effects.
        """
        # Instantiate DomainAdaptationLearner
        DA_learner = DomainAdaptationLearner(models=LinearRegression(),
                                             final_models=LinearRegression())
        # Test inputs
        self._test_inputs(DA_learner, T0=3, T1=5)
        # Test constant and heterogeneous treatment effect, single and multi output y
        for te_type in ["const", "heterogeneous"]:
            for multi_y in [False, True]:
                self._test_te(DA_learner, T0=3, T1=5, tol=0.5, te_type=te_type, multi_y=multi_y)

    def _test_te(self, learner_instance, T0, T1, tol, te_type="const", multi_y=False):
        if te_type not in ["const", "heterogeneous"]:
            raise ValueError("Type of treatment effect must be 'const' or 'heterogeneous'.")
        te_func = getattr(TestMetalearners, "_{te_type}_te".format(te_type=te_type))
        if multi_y:
            X, T, Y = getattr(TestMetalearners, "{te_type}_te_multiy_data".format(te_type=te_type))
            # Get the true treatment effect
            te = np.repeat((np.apply_along_axis(te_func, 1, TestMetalearners.X_test) *
                            (T1 - T0)).reshape(-1, 1), 2, axis=1)
            marginal_te = np.repeat(np.apply_along_axis(
                te_func, 1, TestMetalearners.X_test).reshape(-1, 1) * np.array([2, 4]), 2, axis=0).reshape((-1, 2, 2))
        else:
            X, T, Y = getattr(TestMetalearners, "{te_type}_te_data".format(te_type=te_type))
            # Get the true treatment effect
            te = np.apply_along_axis(te_func, 1, TestMetalearners.X_test) * (T1 - T0)
            marginal_te = np.apply_along_axis(te_func, 1, TestMetalearners.X_test).reshape(-1, 1) * np.array([2, 4])
        # Fit learner and get the effect and marginal effect
        learner_instance.fit(Y, T, X=X)
        te_hat = learner_instance.effect(TestMetalearners.X_test, T0=T0, T1=T1)
        marginal_te_hat = learner_instance.marginal_effect(T1, TestMetalearners.X_test)
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        marginal_te_res = np.abs(marginal_te - marginal_te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)
        self.assertGreaterEqual(np.mean(marginal_te_res < tol), 0.90)
        # Check whether the output shape is right
        m = TestMetalearners.X_test.shape[0]
        d_t = 2
        d_y = Y.shape[1:]
        self.assertEqual(te_hat.shape, (m,) + d_y)
        self.assertEqual(marginal_te_hat.shape, (m, d_t,) + d_y)

    def _test_inputs(self, learner_instance, T0, T1):
        X, T, Y = TestMetalearners.const_te_data
        # Check that one can pass in regular lists
        learner_instance.fit(list(Y), list(T), X=list(X))
        learner_instance.effect(list(TestMetalearners.X_test), T0=T0, T1=T1)
        # Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, learner_instance.fit, Y, T, X[:TestMetalearners.n // 2])
        self.assertRaises(ValueError, learner_instance.fit, Y[:TestMetalearners.n // 2], T, X)
        # Check that it works when T, Y have shape (n, 1)
        self.assertWarns(DataConversionWarning,
                         learner_instance.fit, Y.reshape(-1, 1), T.reshape(-1, 1), X
                         )

    @classmethod
    def _const_te(cls, x):
        return 2

    @classmethod
    def _heterogeneous_te(cls, x):
        return x[cls.heterogeneity_index]

    @classmethod
    def _generate_data(cls, n, d, beta, treatment_effect, multi_y):
        """Generates population data for given treatment_effect functions.

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
        Y0 = (np.dot(X, beta) + cls.random_state.normal(0, 1)).reshape(n, -1)
        treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
        Y = Y0 + (treat_effect * T).reshape(-1, 1)
        if not multi_y:
            Y = Y.flatten()
        return (X, T, Y)
