#!/usr/bin/python
# -*- coding: utf-8 -*-

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
from econml.automated_ml import *
from econml.drlearner import DRLearner
import numpy as np
from econml.utilities import shape, hstack, vstack, reshape, \
    cross_product
from econml.inference import BootstrapInference
from contextlib import ExitStack
from sklearn.ensemble import RandomForestRegressor, \
    GradientBoostingRegressor, GradientBoostingClassifier
import itertools
from econml.sklearn_extensions.linear_model import WeightedLasso
from econml.automated_ml import addAutomatedML
from econml.tests.test_statsmodels import _summarize
import econml.tests.utilities  # bugfix for assertWarns
import copy
import logging
from econml.data.dgps import ihdp_surface_B
from azureml.train.automl.exceptions import ClientException
from azureml.core.authentication import ServicePrincipalAuthentication
import os

AutomatedTLearner = addAutomatedML(TLearner)
AutomatedSLearner = addAutomatedML(SLearner)
AutomatedXLearner = addAutomatedML(XLearner)
AutomatedDomainAdaptationLearner = \
    addAutomatedML(DomainAdaptationLearner)
AutomatedDRLearner = addAutomatedML(DRLearner)
AutomatedDMLCateEstimator = addAutomatedML(DMLCateEstimator)
AutomatedLinearDMLCateEstimator = addAutomatedML(LinearDMLCateEstimator)
AutomatedSparseLinearDMLCateEstimator = \
    addAutomatedML(SparseLinearDMLCateEstimator)
AutomatedKernelDMLCateEstimator = addAutomatedML(KernelDMLCateEstimator)
AutomatedNonParamDMLCateEstimator = \
    addAutomatedML(NonParamDMLCateEstimator)
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
    'experiment_timeout_minutes': 1,
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
    'experiment_timeout_minutes': 1,
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
                                            enable_onnx_compatible_models=True, model_explainability=True,
                                            **AUTOML_SETTINGS_REG)

AUTOML_CONFIG_SAMPLE_WEIGHT_REG = EconAutoMLConfig(task='regression',
                                                   debug_log='automl_errors.log',
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


@pytest.mark.automl
class TestDML(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        subscription_id = os.getenv("SUBSCRIPTION_ID")
        resource_group = os.getenv("RESOURCE_GROUP")
        workspace_name = os.getenv("WORKSPACE_NAME")
        tenant_id = os.getenv("TENANT_ID")
        service_principal_id = os.getenv("SERVICE_PRINCIPAL_ID")
        svc_pr_password = os.getenv("SVR_PR_PASSWORD")

        svc_pr = ServicePrincipalAuthentication(
            tenant_id=tenant_id,
            service_principal_id=service_principal_id,
            service_principal_password=svc_pr_password)

        setAutomatedMLWorkspace(auth=svc_pr,
                                subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)

    def test_nonparam(self):
        Y, T, X, _ = ihdp_surface_B()
        est = \
            AutomatedNonParamDMLCateEstimator(model_y=automl_model_reg(),
                                              model_t=automl_model_clf(),
                                              model_final=automl_model_reg(), featurizer=None,
                                              discrete_treatment=True)
        est.fit(Y, T, X)
        _ = est.effect(X)

    def test_param(self):
        Y, T, X, _ = ihdp_surface_B()
        est = \
            AutomatedDMLCateEstimator(model_y=automl_model_reg(),
                                      model_t=GradientBoostingClassifier(),
                                      model_final=RandomForestRegressor(),
                                      featurizer=None,
                                      discrete_treatment=True)
        est.fit(Y, T, X)
        _ = est.effect(X)

    def test_forest_dml(self):
        """Testing accuracy of forest DML is reasonable"""

        Y, T, X, _ = ihdp_surface_B()
        est = \
            AutomatedForestDMLCateEstimator(model_y=automl_model_reg(),
                                            model_t=GradientBoostingClassifier(),
                                            discrete_treatment=True,
                                            n_estimators=1000,
                                            subsample_fr=.8,
                                            min_samples_leaf=10,
                                            min_impurity_decrease=0.001,
                                            verbose=0, min_weight_fraction_leaf=.01)
        est.fit(Y, T, X)
        _ = est.effect(X)


@pytest.mark.automl
class TestMetalearners(unittest.TestCase):

    def test_TLearner(self):
        """Tests whether the TLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """

        # TLearner test
        # Instantiate TLearner
        Y, T, X, _ = ihdp_surface_B()
        est = AutomatedTLearner(models=automl_model_reg())

        # Test constant and heterogeneous treatment effect, single and multi output y

        est.fit(Y, T, X)
        _ = est.effect(X)

    def test_SLearner(self):
        """Tests whether the SLearner can accurately estimate constant and heterogeneous
           treatment effects.
        """
        # Test constant treatment effect with multi output Y
        # Test heterogeneous treatment effect
        # Need interactions between T and features
        Y, T, X, _ = ihdp_surface_B()
        est = AutomatedSLearner(overall_model=automl_model_reg())

        est.fit(Y, T, X)
        _ = est.effect(X)

        # Test heterogeneous treatment effect with multi output Y

    def test_DALearner(self):
        """Tests whether the DomainAdaptationLearner can accurately estimate constant and
           heterogeneous treatment effects.
        """

        # Instantiate DomainAdaptationLearner

        est = \
            AutomatedDomainAdaptationLearner(models=automl_model_reg(),
                                             final_models=automl_model_reg())

        est.fit(Y, T, X)
        _ = est.effect(X)
