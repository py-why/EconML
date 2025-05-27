# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest

from econml.dml import LinearDML, CausalForestDML
from econml.dr import LinearDRLearner
from econml.validate.sensitivity_analysis import sensitivity_interval
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np



class TestSensitivityAnalysis(unittest.TestCase):

    def test_params(self):
        # test alpha when calling through lineardml

        n = 1000
        a = np.random.normal(size=5)
        X = np.random.normal(size=(n,5), loc=[1,0.5,0,0,0])/3 # closest to center 1, then 2, then 3
        centers = np.array([[1,0,0,0,0], [0,1,0,0,0]]) # uncomment for binary treatment

        ds = X[:,None,:]-centers[None,:,:]
        ds = np.einsum("nci,nci->nc", ds, ds)

        ps_r = np.exp(-ds)
        ps = ps_r / np.sum(ps_r, axis=1, keepdims=True)

        T = np.random.default_rng().multinomial(1, ps) @ np.arange(len(centers))
        Y = np.random.normal(size=n) + 3*(T == 1)*X[:,1] - (T == 2) + 2 * X @ a

        ests = [
            LinearDML(
                model_y=LinearRegression(),
                model_t=LogisticRegression(),
                # implicitly test that discrete binary treatment works for dml
                # other permutations are tested in test_dml
                discrete_treatment=True,
                cv=3,
                random_state=0,
            ),
            CausalForestDML(
                model_y=LinearRegression(),
                model_t=LogisticRegression(),
                discrete_treatment=True,
                cv=3,
                random_state=0,
            ),
            LinearDRLearner(
                model_regression=LinearRegression(),
                model_propensity=LogisticRegression(),
                cv=3,
                random_state=0,
            )
        ]

        for est in ests:

            est.fit(Y, T, X=X, W=None)

            T_arg = {}
            if isinstance(est, LinearDRLearner):
                T_arg = {'T': 1}

            # baseline sensitivity results
            lb, ub = est.sensitivity_interval(**T_arg, alpha=0.05, c_y=0.05, c_t=0.05, rho=1)
            rv = est.robustness_value(**T_arg, alpha=0.05)

            # check that alpha is passed through
            lb2, ub2 = est.sensitivity_interval(**T_arg, alpha=0.5, c_y=0.05, c_t=0.05, rho=1)
            self.assertTrue(lb < ub)
            self.assertTrue(lb2 < ub2)
            self.assertTrue(lb2 > lb)
            self.assertTrue(ub2 < ub)

            rv2 = est.robustness_value(**T_arg, alpha=0.5)
            self.assertTrue(rv2 > rv)

            # check that c_y, c_d are passed through
            lb3, ub3 = est.sensitivity_interval(**T_arg, alpha=0.05, c_y=0.3, c_t=0.3, rho=1)
            self.assertTrue(lb3 < lb)
            self.assertTrue(ub3 > ub)

            # check that interval_type is passed through
            lb4, ub4 = est.sensitivity_interval(**T_arg, alpha=0.05,
                                                c_y=0.05, c_t=0.05, rho=1, interval_type='theta')
            self.assertTrue(lb4 > lb)
            self.assertTrue(ub4 < ub)
            self.assertTrue(lb4 < ub4)

            rv4 = est.robustness_value(**T_arg, alpha=0.05, interval_type='theta')
            self.assertTrue(rv4 > rv)

            # check that null_hypothesis is passed through
            rv5 = est.robustness_value(**T_arg, alpha=0.05, null_hypothesis=10)
            self.assertNotEqual(rv5, rv)


    def test_invalid_params(self):

        theta = 0.5
        sigma = 0.5
        nu = 0.5
        cov = np.random.normal(size=(3, 3))

        sensitivity_interval(theta, sigma, nu, cov, alpha=0.05, c_y=0.05, c_t=0.05, rho=1)

        # check that c_y, c_y, rho are constrained
        with pytest.raises(ValueError):
            sensitivity_interval(theta, sigma, nu, cov, alpha=0.05, c_y=-0.5, c_t=0.05, rho=1)

        with pytest.raises(ValueError):
            sensitivity_interval(theta, sigma, nu, cov, alpha=0.05, c_y=0.05, c_t=-0.5, rho=1)

        with pytest.raises(ValueError):
            sensitivity_interval(theta, sigma, nu, cov, alpha=0.05, c_y=1.5, c_t=0.05, rho=1)

        with pytest.raises(ValueError):
            sensitivity_interval(theta, sigma, nu, cov, alpha=0.05, c_y=0.05, c_t=0.05, rho=-1.5)

        # ensure we raise an error on invalid sigma, nu
        with pytest.raises(ValueError):
            sensitivity_interval(theta, -1, nu, cov, alpha=0.05, c_y=0.05, c_t=0.05, rho=1)

        with pytest.raises(ValueError):
            sensitivity_interval(theta, sigma,-1, cov, alpha=0.05, c_y=0.05, c_t=0.05, rho=1)

        # ensure failure on invalid interval_type
        with pytest.raises(ValueError):
            sensitivity_interval(theta, sigma, nu, cov, alpha=0.05, c_y=0.05, c_t=0.05, rho=1, interval_type='foo')

