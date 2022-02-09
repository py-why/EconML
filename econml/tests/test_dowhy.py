# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import unittest
from econml.dml import LinearDML, CausalForestDML
from econml.orf import DROrthoForest
from econml.dr import DRLearner, ForestDRLearner, LinearDRLearner
from econml.metalearners import XLearner
from econml.iv.dml import OrthoIV, DMLIV
from econml.iv.dr import LinearDRIV
from econml.iv.dr._dr import _DummyCATE
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression


class TestDowhy(unittest.TestCase):

    def _get_data(self):
        X = np.random.normal(0, 1, size=(250, 5))
        T = np.random.binomial(1, .5, size=(250,))
        Y = np.random.normal(0, 1, size=(250,))
        Z = np.random.normal(0, 1, size=(250,))
        return Y, T, X[:, [0]], X[:, 1:], Z

    def test_dowhy(self):
        def reg():
            return LinearRegression()

        def clf():
            return LogisticRegression()

        Y, T, X, W, Z = self._get_data()
        # test at least one estimator from each category
        models = {"dml": LinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True,
                                   linear_first_stages=False),
                  "dr": DRLearner(model_propensity=clf(), model_regression=reg(),
                                  model_final=reg()),
                  "forestdr": ForestDRLearner(model_propensity=clf(), model_regression=reg()),
                  "xlearner": XLearner(models=reg(), cate_models=reg(), propensity_model=clf()),
                  "cfdml": CausalForestDML(model_y=reg(), model_t=clf(), discrete_treatment=True),
                  "orf": DROrthoForest(n_trees=10, propensity_model=clf(), model_Y=reg()),
                  "orthoiv": OrthoIV(model_y_xw=reg(),
                                     model_t_xw=clf(),
                                     model_z_xw=reg(),
                                     discrete_treatment=True,
                                     discrete_instrument=False),
                  "dmliv": DMLIV(fit_cate_intercept=True,
                                 discrete_treatment=True,
                                 discrete_instrument=False),
                  "driv": LinearDRIV(flexible_model_effect=StatsModelsLinearRegression(fit_intercept=False),
                                     fit_cate_intercept=True,
                                     discrete_instrument=False,
                                     discrete_treatment=True)}
        for name, model in models.items():
            with self.subTest(name=name):
                est = model
                if name == "xlearner":
                    est_dowhy = est.dowhy.fit(Y, T, X=np.hstack((X, W)), W=None)
                elif name in ["orthoiv", "dmliv", "driv"]:
                    est_dowhy = est.dowhy.fit(Y, T, Z=Z, X=X, W=W)
                else:
                    est_dowhy = est.dowhy.fit(Y, T, X=X, W=W)
                # test causal graph
                est_dowhy.view_model()
                # test refutation estimate
                est_dowhy.refute_estimate(method_name="random_common_cause", num_simulations=3)
                if name != "orf":
                    est_dowhy.refute_estimate(method_name="add_unobserved_common_cause",
                                              confounders_effect_on_treatment="binary_flip",
                                              confounders_effect_on_outcome="linear",
                                              effect_strength_on_treatment=0.1,
                                              effect_strength_on_outcome=0.1,)
                    est_dowhy.refute_estimate(method_name="placebo_treatment_refuter", placebo_type="permute",
                                              num_simulations=3)
                    est_dowhy.refute_estimate(method_name="data_subset_refuter", subset_fraction=0.8,
                                              num_simulations=3)

    def test_store_dataframe_name(self):
        Y, T, X, W, Z = self._get_data()
        Y_name = "outcome"
        Y = pd.Series(Y, name=Y_name)
        T_name = "treatment"
        T = pd.Series(T, name=T_name)
        X_name = ["feature"]
        X = pd.DataFrame(X, columns=X_name)
        W_name = ["control1", "control2", "control3", "control4"]
        W = pd.DataFrame(W, columns=W_name)
        est = LinearDRLearner().dowhy.fit(Y, T, X, W)
        np.testing.assert_array_equal(est._common_causes, X_name + W_name)
        np.testing.assert_array_equal(est._effect_modifiers, X_name)
        np.testing.assert_array_equal(est._treatment, [T_name])
        np.testing.assert_array_equal(est._outcome, [Y_name])
