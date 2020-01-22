# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, PolynomialFeatures
from sklearn.model_selection import KFold
from econml.dml import *
from econml.metalearners import *
from econml.automated_ml import *
from econml.drlearner import DRLearner
import numpy as np
from econml.utilities import shape, hstack, vstack, reshape, cross_product
from econml.inference import BootstrapInference
from contextlib import ExitStack
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
import itertools
from econml.sklearn_extensions.linear_model import WeightedLasso
from econml.automated_ml import addAutomatedML
from econml.tests.test_statsmodels import _summarize
import econml.tests.utilities  # bugfix for assertWarns
import copy

"""
Test strategy:

Integration test public functions, integration tests adding
automl to every model in the library.

Test will a varying set of hyperparameters.
"""
setAutomatedMLWorkspace(workspace_name="automl-dml-cloud-1",
                        subscription_id="a843f881-0968-4221-afce-2f99d7aed631",
                        resource_group="automl-dml")
AutomatedTLearner = addAutomatedML(TLearner)
AutomatedSLearner = addAutomatedML(SLearner)
AutomatedXLearner = addAutomatedML(XLearner)
AutomatedDomainAdaptationLearner = addAutomatedML(DomainAdaptationLearner)
AutomatedDRLearner = addAutomatedML(DRLearner)
AutomatedDMLCateEstimator = addAutomatedML(DMLCateEstimator)
AutomatedLinearDMLCateEstimator = addAutomatedML(LinearDMLCateEstimator)
AutomatedSparseLinearDMLCateEstimator = addAutomatedML(SparseLinearDMLCateEstimator)
AutomatedKernelDMLCateEstimator = addAutomatedML(KernelDMLCateEstimator)
AutomatedNonParamDMLCateEstimator = addAutomatedML(NonParamDMLCateEstimator)
AutomatedForestDMLCateEstimator = addAutomatedML(ForestDMLCateEstimator)
# all solutions to underdetermined (or exactly determined) Ax=b are given by A⁺b+(I-A⁺A)w for some arbitrary w
# note that if Ax=b is overdetermined, this will raise an assertion error


def rand_sol(A, b):
    """Generate a random solution to the equation Ax=b."""
    assert np.linalg.matrix_rank(A) <= len(b)
    A_plus = np.linalg.pinv(A)
    x = A_plus @ b
    return x + (np.eye(x.shape[0]) - A_plus @ A) @ np.random.normal(size=x.shape)


AUTOML_SETTINGS_REG = {
    "experiment_timeout_minutes": 1,
    "enable_early_stopping": True,
    "iteration_timeout_minutes": 1,
    "max_cores_per_iteration": 1,
    "n_cross_validations": 2,
    'preprocess': False,
    "featurization": 'off',
    "enable_stack_ensemble": False,
    "enable_voting_ensemble": False,
    "primary_metric": 'normalized_mean_absolute_error'
}

AUTOML_SETTINGS_CLF = {
    "experiment_timeout_minutes": 1,
    "enable_early_stopping": True,
    "iteration_timeout_minutes": 1,
    "max_cores_per_iteration": 1,
    "n_cross_validations": 2,
    "enable_stack_ensemble": False,
    'preprocess': False,
    "enable_voting_ensemble": False,
    "featurization": 'off',
    "primary_metric": 'AUC_weighted'
}

AUTOML_CONFIG_REG = EconAutoMLConfig(task='regression',
                             debug_log='automl_errors.log',
                             enable_onnx_compatible_models=True,
                             model_explainability=True,
                             **AUTOML_SETTINGS_REG
                            )

AUTOML_CONFIG_CLF = EconAutoMLConfig(task='classification',
                             debug_log='automl_errors.log',
                             enable_onnx_compatible_models=True,
                             model_explainability=True,
                             **AUTOML_SETTINGS_CLF
                            )

AUTOML_CONFIG_LINEAR_REG = EconAutoMLConfig(task='regression',
                             debug_log='automl_errors.log',
                             enable_onnx_compatible_models=True,
                             model_explainability=True,
                             **AUTOML_SETTINGS_REG
                            )

AUTOML_CONFIG_SAMPLE_WEIGHT_REG = EconAutoMLConfig(task='regression',
                             debug_log='automl_errors.log',
                             enable_onnx_compatible_models=True,
                             model_explainability=True,
                             **AUTOML_SETTINGS_REG
                            )


def automl_model_reg():
    return copy.deepcopy(AUTOML_CONFIG_REG)


def automl_model_clf():
    return copy.deepcopy(AUTOML_CONFIG_CLF)

# Linear models are required for parametric dml


def automl_model_linear_reg():
    return copy.deepcopy(AUTOML_CONFIG_LINEAR_REG)

# sample weighting models are required for nonparametric dml


def automl_model_sample_weight_reg():
    return copy.deepcopy(AUTOML_CONFIG_SAMPLE_WEIGHT_REG)


class TestDML(unittest.TestCase):
    def test_param(self):
        n = 20

        def make_random(is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d >= 0 else (n,)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b'], size=sz)
                    # ensure that we've got at least two of every element
                    _, counts = np.unique(arr, return_counts=True)
                    if len(counts) == 2 and counts.min() > 2:
                        return arr
            else:
                return np.random.normal(size=sz)
        W, X, Y, T = [make_random(is_discrete, d)
                        for is_discrete, d in [(False, 2),
                                               (False, 2),
                                               (False, 0),
                                               (True, 0)]]
        for estimator in [AutomatedKernelDMLCateEstimator, AutomatedSparseLinearDMLCateEstimator, AutomatedKernelDMLCateEstimator]:
            est = estimator(model_y=automl_model_reg(),
                                                                               model_t=GradientBoostingClassifier(),
                                                                               discrete_treatment=True)
            est.fit(Y, T, X, W)
            est.score(Y, T, X, W)

    def test_nonparam(self):
        n = 20

        def make_random(is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d >= 0 else (n,)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b'], size=sz)
                    # ensure that we've got at least two of every element
                    _, counts = np.unique(arr, return_counts=True)
                    if len(counts) == 2 and counts.min() > 2:
                        return arr
            else:
                return np.random.normal(size=sz)
        W, X, Y, T = [make_random(is_discrete, d)
                        for is_discrete, d in [(False, 2),
                                               (False, 2),
                                               (False, 1),
                                               (True, 1)]]
        est = AutomatedNonParamDMLCateEstimator(model_y=automl_model_reg(),
                                                                           model_t=GradientBoostingClassifier(),
                                                                           model_final=LinearRegression(),
                                                                           featurizer=None,
                                                                           discrete_treatment=True)
        est.fit(Y, T, X, W)

    def test_multioutput_nonparam(self):
        n = 20

        def make_random(is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d >= 0 else (n,)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b'], size=sz)
                    # ensure that we've got at least two of every element
                    _, counts = np.unique(arr, return_counts=True)
                    if len(counts) == 2 and counts.min() > 2:
                        return arr
            else:
                return np.random.normal(size=sz)
        W, X, Y, T = [make_random(is_discrete, d)
                        for is_discrete, d in [(False, 2),
                                               (False, 2),
                                               (False, 2),
                                               (True, 1)]]
        est = AutomatedNonParamDMLCateEstimator(model_y=automl_model_reg(),
                                                                           model_t=GradientBoostingClassifier(),
                                                                           model_final=LinearRegression(),
                                                                           featurizer=None,
                                                                           discrete_treatment=True)
       est.fit(Y, T, X, W)
    def test_forest_dml(self):
        """Testing accuracy of forest DML is reasonable"""
        np.random.seed(1234)
        n = 20000  # number of raw samples
        d = 10
        X = np.random.binomial(1, .5, size=(n, d))
        T = np.random.binomial(1, .5, size=(n,))

        def true_fn(x):
            return -1 + 2 * x[:, 0] + x[:, 1] * x[:, 2]
        y = true_fn(X) * T + X[:, 0] + (1 * X[:, 0] + 1) * np.random.normal(0, 1, size=(n,))

        XT = np.hstack([T.reshape(-1, 1), X])
        X1, X2, y1, y2, X1_sum, X2_sum, y1_sum, y2_sum, n1_sum, n2_sum, var1_sum, var2_sum = _summarize(XT, y)
        # We concatenate the two copies data
        X_sum = np.vstack([np.array(X1_sum)[:, 1:], np.array(X2_sum)[:, 1:]])
        T_sum = np.concatenate((np.array(X1_sum)[:, 0], np.array(X2_sum)[:, 0]))
        y_sum = np.concatenate((y1_sum, y2_sum))  # outcome
        n_sum = np.concatenate((n1_sum, n2_sum))  # number of summarized points
        var_sum = np.concatenate((var1_sum, var2_sum))  # variance of the summarized points
        summarized, min_samples_leaf, sample_var = (True, 1, False)
        est = AutomatedForestDMLCateEstimator(model_y=automl_model_reg(),
                                     model_t=automl_model_clf(),
                                     discrete_treatment=True,
                                     n_crossfit_splits=2,
                                     n_estimators=1000,
                                     subsample_fr=.8,
                                     min_samples_leaf=min_samples_leaf,
                                     min_impurity_decrease=0.001,
                                     verbose=0, min_weight_fraction_leaf=.03)
        if summarized:
            if sample_var:
                est.fit(y_sum, T_sum, X_sum[:, :4], X_sum[:, 4:],
                        sample_weight=n_sum, sample_var=var_sum, inference='blb')
            else:
                est.fit(y_sum, T_sum, X_sum[:, :4], X_sum[:, 4:],
                        sample_weight=n_sum, inference='blb')
        else:
            est.fit(y, T, X[:, :4], X[:, 4:], inference='blb')
        X_test = np.array(list(itertools.product([0, 1], repeat=4)))
        point = est.effect(X_test)


    def test_can_use_vectors(self):
        """Test that we can pass vectors for T and Y (not only 2-dimensional arrays)."""
        dmls = [
            AutomatedLinearDMLCateEstimator(model_y=automl_model_sample_weight_reg(),model_t= automl_model_sample_weight_reg(), fit_cate_intercept=False),
            AutomatedSparseLinearDMLCateEstimator(model_y=automl_model_sample_weight_reg(), model_t=automl_model_sample_weight_reg(), fit_cate_intercept=False)
        ]
        for dml in dmls:
            dml.fit(np.array([1, 2, 3, 4, 1, 2, 3, 4]), np.array([1, 2, 3,4, 1, 2, 3,4]), np.ones((8, 1)))
            self.assertAlmostEqual(dml.coef_.reshape(())[()], 1)
            score = dml.score(np.array([1, 2, 3, 4,1, 2, 3,4]), np.array([1, 2, 3,4, 1, 2, 3,4]), np.ones((8, 1)))
            self.assertAlmostEqual(score, 0)

    def test_can_use_sample_weights(self):
        """Test that we can pass sample weights to an estimator."""
        dmls = [
            AutomatedLinearDMLCateEstimator(model_y=automl_model_reg(), model_t='auto', fit_cate_intercept=False),
            AutomatedSparseLinearDMLCateEstimator(model_y=automl_model_reg(), model_t='auto', fit_cate_intercept=False)
        ]
        for dml in dmls:
            dml.fit(np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]),
                    np.ones((12, 1)), sample_weight=np.ones((12, )))
            self.assertAlmostEqual(dml.coef_.reshape(())[()], 1)

    def test_discrete_treatments(self):
        """Test that we can use discrete treatments"""
        dmls = [
            AutomatedLinearDMLCateEstimator(model_y=automl_model_reg(), model_t=automl_model_clf(),
                                   fit_cate_intercept=False, discrete_treatment=True),
            AutomatedSparseLinearDMLCateEstimator(model_y=automl_model_reg(), model_t=automl_model_clf(),
                                         fit_cate_intercept=False, discrete_treatment=True)
        ]
        for dml in dmls:
            # create a simple artificial setup where effect of moving from treatment
            #     1 -> 2 is 2,
            #     1 -> 3 is 1, and
            #     2 -> 3 is -1 (necessarily, by composing the previous two effects)
            # Using an uneven number of examples from different classes,
            # and having the treatments in non-lexicographic order,
            # Should rule out some basic issues.
            dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array([3, 2, 1, 2, 3, 1, 1, 1]), np.ones((8, 1)))
            np.testing.assert_almost_equal(
                dml.effect(
                    np.ones((9, 1)),
                    T0=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                    T1=np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
                ),
                [0, 2, 1, -2, 0, -1, -1, 1, 0],
                decimal=2)
            dml.score(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array([3, 2, 1, 2, 3, 1, 1, 1]), np.ones((8, 1)))



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
        T_learner = AutomatedTLearner(models=automl_model_reg())
        # Test constant and heterogeneous treatment effect, single and multi output y
        for te_type in ["const", "heterogeneous"]:
            self._test_te(T_learner, T0=3, T1=5, tol=0.5, te_type=te_type, multi_y=False)

    def test_SLearner(self):
        """Tests whether the SLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # Instantiate SLearner
        S_learner = AutomatedSLearner(overall_model=automl_model_reg())
        # Test constant treatment effect
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="const", multi_y=False)
        # Test constant treatment effect with multi output Y
        # Test heterogeneous treatment effect
        # Need interactions between T and features
        overall_model = Pipeline([('poly', PolynomialFeatures()), ('model', automl_model_reg())])
        S_learner = SLearner(overall_model=overall_model)
        self._test_te(S_learner, T0=3, T1=5, tol=0.5, te_type="heterogeneous", multi_y=False)
        # Test heterogeneous treatment effect with multi output Y

    def test_DALearner(self):
        """Tests whether the DomainAdaptationLearner can accurately estimate constant and
           heterogeneous treatment effects.
        """
        # Instantiate DomainAdaptationLearner
        DA_learner = AutomatedDomainAdaptationLearner(models=automl_model_reg(),
                                             final_models=automl_model_reg())
        # Test constant and heterogeneous treatment effect, single and multi output y
        for te_type in ["const", "heterogeneous"]:
            self._test_te(DA_learner, T0=3, T1=5, tol=0.5, te_type=te_type, multi_y=False)

    # TODO: Add tests for DR Learner and X-Learner once bugs making those learners compatible with
    # AutomatedML are resolved.
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
        learner_instance.fit(Y, T, X)
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
