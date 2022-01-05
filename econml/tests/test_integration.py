# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import unittest
import pytest

try:
    import keras
    keras_installed = True
except ImportError:
    keras_installed = False

from econml.drlearner import LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from econml.dml import LinearDML, SparseLinearDML, ForestDML
from econml.ortho_forest import DMLOrthoForest, DROrthoForest
from econml.sklearn_extensions.linear_model import WeightedLasso
from econml.metalearners import XLearner, SLearner, TLearner
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, MultiTaskLasso, LassoCV
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from econml.iv.dr import LinearIntentToTreatDRIV
from econml.deepiv import DeepIVEstimator


class TestPandasIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        # DGP constants
        cls.n_controls = 10
        cls.n_features = 2
        cls.n = 100
        # Define data features
        # Added `_df`to names to be different from the default cate_estimator names
        cls.controls = [f"W{i}_df" for i in range(cls.n_controls)]
        cls.features = [f"X{i}_df" for i in range(cls.n_features)]
        cls.instrument = ["Z0_df"]
        cls.outcome = ["Y0_df"]
        cls.cont_treat = ["T0_df"]
        cls.bin_treat = ["T2_df"]
        cls.cat_treat = ["T_cat"]
        cls.cat_treat_labels = ["None", "One", "Two"]
        cls.outcome_multi = ["Y0_df", "Y1_df"]
        cls.cont_treat_multi = ["T0_df", "T1_df"]
        # Generate data
        d = {}
        d.update({w: np.random.normal(size=cls.n) for w in cls.controls})
        d.update({x: np.random.normal(size=cls.n) for x in cls.features})
        d.update({t: np.random.uniform(size=cls.n) for t in cls.cont_treat_multi})
        d.update({t: np.random.binomial(1, 0.5, size=cls.n) for t in cls.bin_treat})
        d.update({t: np.random.choice(["None", "One", "Two"], size=cls.n, p=[0.4, 0.3, 0.3]) for t in cls.cat_treat})
        d.update({z: np.random.binomial(1, 0.5, size=cls.n) for z in cls.instrument})
        d.update({y: np.random.normal(size=cls.n) for y in cls.outcome_multi})
        cls.df = pd.DataFrame(d)

    def test_dml(self):
        #################################
        #  Single treatment and outcome #
        #################################
        X = TestPandasIntegration.df[TestPandasIntegration.features]
        W = TestPandasIntegration.df[TestPandasIntegration.controls]
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome]
        T = TestPandasIntegration.df[TestPandasIntegration.cont_treat]
        # Test LinearDML
        est = LinearDML(model_y=LassoCV(), model_t=LassoCV())
        est.fit(Y, T, X=X, W=W, inference='statsmodels')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_input_names(est.summary())  # Check that names propagate as expected
        # |--> Test featurizers
        est.featurizer = PolynomialFeatures(degree=2, include_bias=False)
        est.fit(Y, T, X=X, W=W, inference='statsmodels')
        self._check_input_names(
            est.summary(),
            feat_comp=est.original_featurizer.get_feature_names(X.columns))
        est.featurizer = FunctionTransformer()
        est.fit(Y, T, X=X, W=W, inference='statsmodels')
        self._check_input_names(
            est.summary(),
            feat_comp=[f"feat(X){i}" for i in range(TestPandasIntegration.n_features)])
        est.featurizer = ColumnTransformer([('passthrough', 'passthrough', [0])])
        est.fit(Y, T, X=X, W=W, inference='statsmodels')
        # ColumnTransformer behaves differently depending on version of sklearn, so we no longer check the names

        # |--> Test re-fit
        est.featurizer = None
        X1 = X.rename(columns={c: "{}_1".format(c) for c in X.columns})
        est.fit(Y, T, X=X1, W=W, inference='statsmodels')
        self._check_input_names(est.summary(), feat_comp=X1.columns)
        # Test SparseLinearDML
        est = SparseLinearDML(model_y=LassoCV(), model_t=LassoCV())
        est.fit(Y, T, X=X, W=W, inference='debiasedlasso')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_input_names(est.summary())  # Check that names propagate as expected
        # Test ForestDML
        est = ForestDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor())
        est.fit(Y, T, X=X, W=W, inference='blb')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        ####################################
        #  Mutiple treatments and outcomes #
        ####################################
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome_multi]
        T = TestPandasIntegration.df[TestPandasIntegration.cont_treat_multi]
        # Test LinearDML
        est = LinearDML(model_y=MultiTaskLasso(), model_t=MultiTaskLasso())
        est.fit(Y, T, X=X, W=W, inference='statsmodels')
        self._check_input_names(est.summary(), True, True)  # Check that names propagate as expected
        self._check_popsum_names(est.effect_inference(X).population_summary(), True)
        est.fit(Y, T, X=X, W=W, inference='bootstrap')  # Check bootstrap as well
        self._check_input_names(est.summary(), True, True)
        self._check_popsum_names(est.effect_inference(X).population_summary(), True)
        # Test SparseLinearDML
        est = SparseLinearDML(model_y=MultiTaskLasso(), model_t=MultiTaskLasso())
        est.fit(Y, T, X=X, W=W, inference='debiasedlasso')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_input_names(est.summary(), True, True)  # Check that names propagate as expected
        self._check_popsum_names(est.effect_inference(X).population_summary(), True)

    def test_orf(self):
        # Single outcome only, ORF does not support multiple outcomes
        X = TestPandasIntegration.df[TestPandasIntegration.features]
        W = TestPandasIntegration.df[TestPandasIntegration.controls]
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome]
        T = TestPandasIntegration.df[TestPandasIntegration.cont_treat]
        # Test DMLOrthoForest
        est = DMLOrthoForest(
            n_trees=100, max_depth=2, model_T=WeightedLasso(), model_Y=WeightedLasso())
        est.fit(Y, T, X=X, W=W, inference='blb')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_popsum_names(est.effect_inference(X).population_summary())
        # Test DROrthoForest
        est = DROrthoForest(n_trees=100, max_depth=2)
        T = TestPandasIntegration.df[TestPandasIntegration.bin_treat]
        est.fit(Y, T, X=X, W=W, inference='blb')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_popsum_names(est.effect_inference(X).population_summary())

    def test_metalearners(self):
        X = TestPandasIntegration.df[TestPandasIntegration.features]
        W = TestPandasIntegration.df[TestPandasIntegration.controls]
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome]
        T = TestPandasIntegration.df[TestPandasIntegration.bin_treat]
        # Test XLearner
        # Skipping population summary names test because bootstrap inference is too slow
        est = XLearner(models=GradientBoostingRegressor(),
                       propensity_model=GradientBoostingClassifier(),
                       cate_models=GradientBoostingRegressor())
        est.fit(Y, T, X=np.hstack([X, W]))
        treatment_effects = est.effect(np.hstack([X, W]))
        # Test SLearner
        est = SLearner(overall_model=GradientBoostingRegressor())
        est.fit(Y, T, X=np.hstack([X, W]))
        treatment_effects = est.effect(np.hstack([X, W]))
        # Test TLearner
        est = TLearner(models=GradientBoostingRegressor())
        est.fit(Y, T, X=np.hstack([X, W]))
        treatment_effects = est.effect(np.hstack([X, W]))

    def test_drlearners(self):
        X = TestPandasIntegration.df[TestPandasIntegration.features]
        W = TestPandasIntegration.df[TestPandasIntegration.controls]
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome]
        T = TestPandasIntegration.df[TestPandasIntegration.bin_treat]
        # Test LinearDRLearner
        est = LinearDRLearner(model_propensity=GradientBoostingClassifier(),
                              model_regression=GradientBoostingRegressor())
        est.fit(Y, T, X=X, W=W, inference='statsmodels')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_input_names(est.summary(T=1))
        self._check_popsum_names(est.effect_inference(X).population_summary())
        # Test SparseLinearDRLearner
        est = SparseLinearDRLearner(model_propensity=GradientBoostingClassifier(),
                                    model_regression=GradientBoostingRegressor())
        est.fit(Y, T, X=X, W=W, inference='debiasedlasso')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_input_names(est.summary(T=1))
        self._check_popsum_names(est.effect_inference(X).population_summary())
        # Test ForestDRLearner
        est = ForestDRLearner(model_propensity=GradientBoostingClassifier(),
                              model_regression=GradientBoostingRegressor())
        est.fit(Y, T, X=X, W=W, inference='blb')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_popsum_names(est.effect_inference(X).population_summary())

    def test_orthoiv(self):
        X = TestPandasIntegration.df[TestPandasIntegration.features]
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome]
        T = TestPandasIntegration.df[TestPandasIntegration.bin_treat]
        Z = TestPandasIntegration.df[TestPandasIntegration.instrument]
        # Test LinearIntentToTreatDRIV
        est = LinearIntentToTreatDRIV(model_y_xw=GradientBoostingRegressor(),
                                      model_t_xwz=GradientBoostingClassifier(),
                                      flexible_model_effect=GradientBoostingRegressor())
        est.fit(Y, T, Z=Z, X=X, inference='statsmodels')
        treatment_effects = est.effect(X)
        lb, ub = est.effect_interval(X, alpha=0.05)
        self._check_input_names(est.summary())  # Check input names propagate
        self._check_popsum_names(est.effect_inference(X).population_summary())

    @pytest.mark.skipif(not keras_installed, reason="Keras not installed")
    def test_deepiv(self):
        X = TestPandasIntegration.df[TestPandasIntegration.features]
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome]
        T = TestPandasIntegration.df[TestPandasIntegration.cont_treat]
        Z = TestPandasIntegration.df[TestPandasIntegration.instrument]
        # Test DeepIV
        treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(3,)),
                                            keras.layers.Dropout(0.17),
                                            keras.layers.Dense(64, activation='relu'),
                                            keras.layers.Dropout(0.17),
                                            keras.layers.Dense(32, activation='relu'),
                                            keras.layers.Dropout(0.17)])
        response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(3,)),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(64, activation='relu'),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(32, activation='relu'),
                                           keras.layers.Dropout(0.17),
                                           keras.layers.Dense(1)])
        est = DeepIVEstimator(n_components=10,  # Number of gaussians in the mixture density networks)
                              m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])),  # Treatment model
                              h=lambda t, x: response_model(keras.layers.concatenate([t, x])),  # Response model
                              n_samples=1  # Number of samples used to estimate the response
                              )
        est.fit(Y, T, X=X, Z=Z)
        treatment_effects = est.effect(X)

    def test_cat_treatments(self):
        X = TestPandasIntegration.df[TestPandasIntegration.features]
        Y = TestPandasIntegration.df[TestPandasIntegration.outcome]
        T = TestPandasIntegration.df[TestPandasIntegration.cat_treat]
        # Test categorical treatments
        est = LinearDML(discrete_treatment=True, linear_first_stages=False,
                        categories=TestPandasIntegration.cat_treat_labels)
        est.fit(Y, T, X=X)
        self._check_input_names(est.summary(), T_cat=True)
        treat_name = "Category"
        self._check_input_names(est.summary(treatment_names=[treat_name]), T_cat=True, treat_comp=[
                                f"{treat_name}_{t}" for t in TestPandasIntegration.cat_treat_labels[1:]])
        # Check refit
        est.fit(Y, T, X=X)
        self._check_input_names(est.summary(), T_cat=True)
        # Check refit after setting categories
        est.categories = [f"{t}_1" for t in TestPandasIntegration.cat_treat_labels]
        T = T.apply(lambda t: t + "_1")
        est.fit(Y, T, X=X)
        self._check_input_names(est.summary(), T_cat=True, treat_comp=[
                                f"{TestPandasIntegration.cat_treat[0]}_{t}_1" for t in
                                TestPandasIntegration.cat_treat_labels[1:]])

    def _check_input_names(self, summary_table,
                           Y_multi=False, T_multi=False, T_cat=False, feat_comp=None, treat_comp=None):
        index_name = np.array(summary_table.tables[0].data)[1:, 0]
        if feat_comp is None:
            feat_comp = TestPandasIntegration.features
        if treat_comp is None:
            if T_multi:
                treat_comp = TestPandasIntegration.cont_treat_multi
            if T_cat:
                treat_comp = ["{}_{}".format(TestPandasIntegration.cat_treat[0], label)
                              for label in TestPandasIntegration.cat_treat_labels[1:]]

        if Y_multi:
            out_comp = TestPandasIntegration.outcome_multi
            if T_cat or T_multi:
                index_name_comp = [
                    f"{feat}|{outcome}|{treat}" for feat in feat_comp for outcome in out_comp for treat in treat_comp]

            else:
                index_name_comp = [
                    f"{feat}|{outcome}" for feat in feat_comp for outcome in out_comp]
        else:
            if T_cat or T_multi:
                index_name_comp = [
                    f"{feat}|{treat}" for feat in feat_comp for treat in treat_comp]
            else:
                index_name_comp = feat_comp
        np.testing.assert_array_equal(index_name, index_name_comp)

    def _check_popsum_names(self, popsum, Y_multi=False):
        np.testing.assert_array_equal(popsum.output_names,
                                      TestPandasIntegration.outcome_multi if Y_multi
                                      else TestPandasIntegration.outcome)
