# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
import pytest
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML, SparseLinearDML, KernelDML, CausalForestDML, NonParamDML
from econml.dr import LinearDRLearner
from econml.iv.dml import OrthoIV, DMLIV, NonParamDMLIV
from econml.iv.dr import DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV
from econml.orf import DMLOrthoForest

from econml.utilities import filter_none_kwargs
from copy import deepcopy


class TestBinaryOutcome(unittest.TestCase):
    # accuracy test
    def test_accuracy(self):
        n = 1000
        binary_outcome = True
        discrete_treatment = True
        true_ate = 0.3
        W = np.random.uniform(-1, 1, size=(n, 1))
        D = np.random.binomial(1, .5 + .1 * W[:, 0], size=(n,))
        Y = np.random.binomial(1, .5 + true_ate * D + .1 * W[:, 0], size=(n,))

        ests = [
            LinearDML(binary_outcome=binary_outcome, discrete_treatment=discrete_treatment),
            CausalForestDML(binary_outcome=binary_outcome, discrete_treatment=discrete_treatment),
            LinearDRLearner(binary_outcome=binary_outcome)
        ]

        for est in ests:

            if isinstance(est, CausalForestDML):
                est.fit(Y, D, X=W)
                ate = est.ate(X=W)
                ate_lb, ate_ub = est.ate_interval(X=W)

            else:
                est.fit(Y, D, W=W)
                ate = est.ate()
                ate_lb, ate_ub = est.ate_interval()

            if isinstance(est, LinearDRLearner):
                est.summary(T=1)
            else:
                est.summary()

            proportion_in_interval = ((ate_lb < true_ate) & (true_ate < ate_ub)).mean()
            np.testing.assert_array_less(0.50, proportion_in_interval)

    # accuracy test, DML
    def test_accuracy_iv(self):
        n = 10000
        binary_outcome = True
        discrete_treatment = True
        true_ate = 0.3
        W = np.random.uniform(-1, 1, size=(n, 1))
        Z = np.random.uniform(-1, 1, size=(n, 1))
        D = np.random.binomial(1, .5 + .1 * W[:, 0] + .1 * Z[:, 0], size=(n,))
        Y = np.random.binomial(1, .5 + true_ate * D + .1 * W[:, 0], size=(n,))

        ests = [
            OrthoIV(binary_outcome=binary_outcome, discrete_treatment=discrete_treatment),
            LinearDRIV(binary_outcome=binary_outcome, discrete_treatment=discrete_treatment),
        ]

        for est in ests:

            est.fit(Y, D, W=W, Z=Z)
            ate = est.ate()
            ate_lb, ate_ub = est.ate_interval()

            est.summary()

            proportion_in_interval = ((ate_lb < true_ate) & (true_ate < ate_ub)).mean()
            np.testing.assert_array_less(0.50, proportion_in_interval)

    def test_string_outcome(self):
        n = 100
        true_ate = 0.3
        W = np.random.uniform(-1, 1, size=(n, 1))
        D = np.random.binomial(1, .5 + .1 * W[:, 0], size=(n,))
        Y = np.random.binomial(1, .5 + true_ate * D + .1 * W[:, 0], size=(n,))
        Y_str = pd.Series(Y).replace(0, 'a').replace(1, 'b').values
        est = LinearDML(binary_outcome=True, discrete_treatment=True)
        est.fit(Y_str, D, X=W)

    def test_basic_functionality(self):
        n = 100
        binary_outcome = True
        d_x = 3

        def gen_array(n, is_binary, d):
            sz = (n, d) if d > 0 else (n,)

            if is_binary:
                return np.random.choice([0, 1], size=sz)
            else:
                return np.random.normal(size=sz)

        for discrete_treatment in [True, False]:
            for discrete_instrument in [True, False, None]:

                Y = gen_array(n, binary_outcome, d=0)
                T = gen_array(n, discrete_treatment, d=0)
                Z = None
                if discrete_instrument is not None:
                    Z = gen_array(n, discrete_instrument, d=0)
                X = gen_array(n, is_binary=False, d=3)

                if Z is not None:
                    est_list = [
                        DRIV(binary_outcome=binary_outcome),
                        DMLIV(binary_outcome=binary_outcome),
                        OrthoIV(binary_outcome=binary_outcome),
                    ]

                else:
                    est_list = [
                        LinearDML(binary_outcome=binary_outcome, discrete_treatment=discrete_treatment),
                        CausalForestDML(binary_outcome=binary_outcome, discrete_treatment=discrete_treatment)
                    ]

                    if discrete_treatment:
                        est_list += [
                            LinearDRLearner(binary_outcome=binary_outcome),
                        ]

                for est in est_list:
                    print(est)
                    est.fit(Y, T, **filter_none_kwargs(X=X, Z=Z))
                    est.score(Y, T, **filter_none_kwargs(X=X, Z=Z))
                    est.effect(X=X)
                    est.const_marginal_effect(X=X)
                    est.marginal_effect(T, X=X)
                    est.ate(X=X)

                # make sure the auto outcome model is a classifier
                if hasattr(est, 'model_y'):
                    outcome_model_attr = 'models_y'
                elif hasattr(est, 'model_regression'):
                    outcome_model_attr = 'models_regression'
                elif hasattr(est, 'model_y_xw'):
                    outcome_model_attr = 'models_y_xw'
                assert (
                    hasattr(
                        getattr(est, outcome_model_attr)[0][0],
                        'predict_proba'
                    )
                ), 'Auto outcome model is not a classifier!'
