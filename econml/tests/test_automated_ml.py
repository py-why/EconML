# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
from sklearn.linear_model import LinearRegression, Lasso, \
    LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, \
    PolynomialFeatures
from sklearn.model_selection import KFold
from econml.dml import *
from econml.metalearners import *
from econml.dr import DRLearner
import numpy as np
from econml.utilities import shape, hstack, vstack, reshape, \
    cross_product
from econml.inference import BootstrapInference
from contextlib import ExitStack
from sklearn.ensemble import RandomForestRegressor, \
    GradientBoostingRegressor, GradientBoostingClassifier
import itertools
from econml.sklearn_extensions.linear_model import WeightedLasso
from econml.tests.test_statsmodels import _summarize
import econml.tests.utilities  # bugfix for assertWarns
import copy
import logging
from econml.data.dgps import ihdp_surface_B
import os

try:
    from azureml.train.automl.exceptions import ClientException
    from azureml.core.authentication import AzureCliAuthentication
    from econml.automated_ml import *
    AutomatedTLearner = addAutomatedML(TLearner)
    AutomatedSLearner = addAutomatedML(SLearner)
    AutomatedXLearner = addAutomatedML(XLearner)
    AutomatedDomainAdaptationLearner = \
        addAutomatedML(DomainAdaptationLearner)
    AutomatedDRLearner = addAutomatedML(DRLearner)
    AutomatedDML = addAutomatedML(DML)
    AutomatedLinearDML = addAutomatedML(LinearDML)
    AutomatedSparseLinearDML = \
        addAutomatedML(SparseLinearDML)
    AutomatedKernelDML = addAutomatedML(KernelDML)
    AutomatedNonParamDML = \
        addAutomatedML(NonParamDML)
    AutomatedCausalForestDML = addAutomatedML(CausalForestDML)

    AUTOML_SETTINGS_REG = {
        'experiment_timeout_minutes': 15,
        'enable_early_stopping': True,
        'iteration_timeout_minutes': 1,
        'max_cores_per_iteration': 1,
        'n_cross_validations': 2,
        'preprocess': False,
        'featurization': 'off',
        'enable_stack_ensemble': False,
        'enable_voting_ensemble': False,
        'primary_metric': 'normalized_mean_absolute_error',
    }

    AUTOML_SETTINGS_CLF = {
        'experiment_timeout_minutes': 15,
        'enable_early_stopping': True,
        'iteration_timeout_minutes': 1,
        'max_cores_per_iteration': 1,
        'n_cross_validations': 2,
        'enable_stack_ensemble': False,
        'preprocess': False,
        'enable_voting_ensemble': False,
        'featurization': 'off',
        'primary_metric': 'AUC_weighted',
    }

    AUTOML_CONFIG_REG = EconAutoMLConfig(task='regression',
                                         debug_log='automl_errors.log',
                                         enable_onnx_compatible_models=True, model_explainability=True,
                                         **AUTOML_SETTINGS_REG)

    AUTOML_CONFIG_CLF = EconAutoMLConfig(task='classification',
                                         debug_log='automl_errors.log',
                                         enable_onnx_compatible_models=True, model_explainability=True,
                                         **AUTOML_SETTINGS_CLF)

    AUTOML_CONFIG_LINEAR_REG = EconAutoMLConfig(task='regression',
                                                debug_log='automl_errors.log',
                                                linear_model_required=True,
                                                enable_onnx_compatible_models=True, model_explainability=True,
                                                **AUTOML_SETTINGS_REG)

    AUTOML_CONFIG_SAMPLE_WEIGHT_REG = EconAutoMLConfig(task='regression',
                                                       debug_log='automl_errors.log',
                                                       linear_model_required=True,
                                                       enable_onnx_compatible_models=True, model_explainability=True,
                                                       **AUTOML_SETTINGS_REG)

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

    # Test values
    Y, T, X, _ = ihdp_surface_B()
except ImportError:
    pass  # automl not installed


@pytest.mark.automl
class TestAutomatedML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace_name = os.getenv("WORKSPACE_NAME")

        auth = AzureCliAuthentication()

        setAutomatedMLWorkspace(auth=auth,
                                subscription_id=subscription_id,
                                resource_group=resource_group, workspace_name=workspace_name)

    def test_nonparam(self):
        """Testing the completion of the fit and effect estimation of an automated Nonparametic DML"""
        est = AutomatedNonParamDML(model_y=automl_model_reg(),
                                   model_t=automl_model_clf(),
                                   model_final=automl_model_sample_weight_reg(), featurizer=None,
                                   discrete_treatment=True)
        est.fit(Y, T, X=X)
        _ = est.effect(X)

    def test_param(self):
        """Testing the completion of the fit and effect estimation of an automated Parametric DML"""
        est = AutomatedLinearDML(model_y=automl_model_reg(),
                                 model_t=GradientBoostingClassifier(),
                                 featurizer=None,
                                 discrete_treatment=True)
        est.fit(Y, T, X=X)
        _ = est.effect(X)

    def test_forest_dml(self):
        """Testing the completion of the fit and effect estimation of an AutomatedForestDML"""
        est = AutomatedCausalForestDML(model_y=automl_model_reg(),
                                       model_t=GradientBoostingClassifier(),
                                       discrete_treatment=True,
                                       n_estimators=1000,
                                       max_samples=.4,
                                       min_samples_leaf=10,
                                       min_impurity_decrease=0.001,
                                       verbose=0, min_weight_fraction_leaf=.01)
        est.fit(Y, T, X=X)
        _ = est.effect(X)

    def test_TLearner(self):
        """Testing the completion of the fit and effect estimation of an AutomatedTLearner"""
        # TLearner test
        # Instantiate TLearner
        est = AutomatedTLearner(models=automl_model_reg())

        # Test constant and heterogeneous treatment effect, single and multi output y

        est.fit(Y, T, X=X)
        _ = est.effect(X)

    def test_SLearner(self):
        """Testing the completion of the fit and effect estimation of an AutomatedSLearner"""
        # Test constant treatment effect with multi output Y
        # Test heterogeneous treatment effect
        # Need interactions between T and features
        est = AutomatedSLearner(overall_model=automl_model_reg())

        est.fit(Y, T, X=X)
        _ = est.effect(X)

        # Test heterogeneous treatment effect with multi output Y

    def test_DALearner(self):
        """Testing the completion of the fit and effect estimation of an AutomatedDomainAdaptationLearner"""

        # Instantiate DomainAdaptationLearner

        est = AutomatedDomainAdaptationLearner(models=automl_model_reg(),
                                               final_models=automl_model_reg())

        est.fit(Y, T, X=X)
        _ = est.effect(X)

    def test_positional(self):
        """Test that positional arguments can be used with AutoML wrappers"""

        class TestEstimator:
            def __init__(self, model_x):
                self.model_x = model_x

            def fit(self, X, Y):
                self.model_x.fit(X, Y)
                return self

            def predict(self, X):
                return self.model_x.predict(X)

        AutoMLTestEstimator = addAutomatedML(TestEstimator)
        AutoMLTestEstimator(automl_model_reg()).fit(X, Y).predict(X)
