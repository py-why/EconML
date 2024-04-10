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
from econml.dr import LinearDRLearner, ForestDRLearner
from econml.iv.dml import OrthoIV, DMLIV, NonParamDMLIV
from econml.iv.dr import DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV, IntentToTreatDRIV, LinearIntentToTreatDRIV
from econml.orf import DMLOrthoForest

from econml.utilities import filter_none_kwargs
from copy import deepcopy


class TestDiscreteOutcome(unittest.TestCase):
    # accuracy test
    def test_accuracy(self):
        n = 1000
        discrete_outcome = True
        discrete_treatment = True
        true_ate = 0.3
        num_iterations = 10

        ests = [
            LinearDML(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment),
            CausalForestDML(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment),
            LinearDRLearner(discrete_outcome=discrete_outcome)
        ]

        for est in ests:

            count_within_interval = 0

            for _ in range(num_iterations):

                W = np.random.uniform(-1, 1, size=(n, 1))
                D = np.random.binomial(1, .5 + .1 * W[:, 0], size=(n,))
                Y = np.random.binomial(1, .5 + true_ate * D + .1 * W[:, 0], size=(n,))

                if isinstance(est, CausalForestDML):
                    est.fit(Y, D, X=W)
                    ate_lb, ate_ub = est.ate_interval(X=W)

                else:
                    est.fit(Y, D, W=W)
                    ate_lb, ate_ub = est.ate_interval()

                if isinstance(est, LinearDRLearner):
                    est.summary(T=1)
                else:
                    est.summary()

                if ate_lb <= true_ate <= ate_ub:
                    count_within_interval += 1

            assert count_within_interval >= 7, (
                f"{est.__class__.__name__}: True ATE falls within the interval bounds "
                f"only {count_within_interval} times out of {num_iterations}"
            )

    # accuracy test, DML
    def test_accuracy_iv(self):
        n = 1000
        discrete_outcome = True
        discrete_treatment = True
        true_ate = 0.3
        num_iterations = 10

        ests = [
            OrthoIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment),
            LinearDRIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment),
        ]

        for est in ests:

            count_within_interval = 0

            for _ in range(num_iterations):

                W = np.random.uniform(-1, 1, size=(n, 1))
                Z = np.random.uniform(-1, 1, size=(n, 1))
                D = np.random.binomial(1, .5 + .1 * W[:, 0] + .1 * Z[:, 0], size=(n,))
                Y = np.random.binomial(1, .5 + true_ate * D + .1 * W[:, 0], size=(n,))

                est.fit(Y, D, W=W, Z=Z)
                ate_lb, ate_ub = est.ate_interval()
                est.summary()

                if ate_lb <= true_ate <= ate_ub:
                    count_within_interval += 1

            assert count_within_interval >= 7, (
                f"{est.__class__.__name__}: True ATE falls within the interval bounds "
                f"only {count_within_interval} times out of {num_iterations}"
            )

    def test_string_outcome(self):
        n = 100
        true_ate = 0.3
        W = np.random.uniform(-1, 1, size=(n, 1))
        D = np.random.binomial(1, .5 + .1 * W[:, 0], size=(n,))
        Y = np.random.binomial(1, .5 + true_ate * D + .1 * W[:, 0], size=(n,))
        Y_str = pd.Series(Y).replace(0, 'a').replace(1, 'b').values
        est = LinearDML(discrete_outcome=True, discrete_treatment=True)
        est.fit(Y_str, D, X=W)

    def test_basic_functionality(self):
        n = 100
        discrete_outcome = True
        d_x = 3

        def gen_array(n, is_binary, d):
            sz = (n, d) if d > 0 else (n,)

            if is_binary:
                return np.random.choice([0, 1], size=sz)
            else:
                return np.random.normal(size=sz)

        for discrete_treatment in [True, False]:
            for discrete_instrument in [True, False, None]:

                Y = gen_array(n, discrete_outcome, d=0)
                T = gen_array(n, discrete_treatment, d=0)
                Z = None
                if discrete_instrument is not None:
                    Z = gen_array(n, discrete_instrument, d=0)
                X = gen_array(n, is_binary=False, d=3)

                if Z is not None:
                    est_list = [
                        DRIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment,
                             discrete_instrument=discrete_instrument),
                        DMLIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment,
                              discrete_instrument=discrete_instrument),
                        OrthoIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment,
                                discrete_instrument=discrete_instrument),
                        LinearDRIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment,
                                   discrete_instrument=discrete_instrument),
                        SparseLinearDRIV(discrete_outcome=discrete_outcome,
                                         discrete_treatment=discrete_treatment,
                                         discrete_instrument=discrete_instrument),
                        ForestDRIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment,
                                   discrete_instrument=discrete_instrument),
                        OrthoIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment,
                                discrete_instrument=discrete_instrument),
                        # uncomment when issue #837 is resolved
                        # NonParamDMLIV(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment,
                        #               discrete_instrument=discrete_instrument, model_final=LinearRegression())
                    ]

                    if discrete_instrument and discrete_treatment:
                        est_list += [
                            LinearIntentToTreatDRIV(discrete_outcome=discrete_outcome),
                            IntentToTreatDRIV(discrete_outcome=discrete_outcome),
                        ]

                else:
                    est_list = [
                        LinearDML(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment),
                        SparseLinearDML(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment),
                        CausalForestDML(discrete_outcome=discrete_outcome, discrete_treatment=discrete_treatment)
                    ]

                    if discrete_treatment:
                        est_list += [
                            LinearDRLearner(discrete_outcome=discrete_outcome),
                            ForestDRLearner(discrete_outcome=discrete_outcome),
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

    def test_constraints(self):
        """
        Confirm errors/warnings when discreteness is not handled correctly for
        discrete outcomes and treatments
        """
        X = np.random.normal(size=(100, 3))
        Y = np.random.choice([0, 1], size=(100))
        T = np.random.choice([0, 1], size=(100, 1))

        ests = [
            LinearDML()
        ]

        for est in ests:
            with self.subTest(est=est, kind='discrete treatment'):
                est.discrete_treatment = False
                est.model_t = LogisticRegression()
                with pytest.raises(AttributeError):
                    est.fit(Y=Y, T=T, X=X)
                est.discrete_treatment = True
                est.model_t = LinearRegression()
                with pytest.warns(UserWarning):
                    est.fit(Y=Y, T=T, X=X)

        ests += [LinearDRLearner()]
        for est in ests:
            print(est)
            with self.subTest(est=est, kind='discrete outcome'):
                est.discrete_outcome = False
                if isinstance(est, LinearDRLearner):
                    est.model_regression = LogisticRegression()
                else:
                    est.model_y = LogisticRegression()
                with pytest.raises(AttributeError):
                    est.fit(Y=Y, T=T, X=X)
                est.discrete_outcome = True
                if isinstance(est, LinearDRLearner):
                    est.model_regression = LinearRegression()
                else:
                    est.model_y = LinearRegression()
                with pytest.warns(UserWarning):
                    est.fit(Y=Y, T=T, X=X)
