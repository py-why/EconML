# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from econml.dml import DMLCateEstimator, LinearDMLCateEstimator, SparseLinearDMLCateEstimator, ForestDMLCateEstimator
import numpy as np
from itertools import product
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV,LinearRegression,MultiTaskElasticNet,MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

def instance_params(opts, seed):
    """
    Generates the instance parameters for the following DGP from double machine
    learning juypter notebook at
    https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb:

    Parameters
    ----------
    opts : dictionary
        dgp-speific options from config file
    seed : int
        random seed for random data generation

    Returns
    -------
    instance_params : dictionary of instance parameters for DGP
    """
    n_controls = opts['n_controls']
    n_features = opts['n_features']
    n_samples = opts['n_samples']
    support_size = opts['support_size']

    instance_params = {}
    instance_params['support_Y'] = np.random.choice(np.arange(n_controls), size=support_size, replace=False)
    instance_params['coefs_Y'] = np.random.uniform(0, 1, size=support_size)
    instance_params['support_T'] = instance_params['support_Y']
    instance_params['coefs_T'] = np.random.uniform(0, 1, size=support_size)
    return instance_params

def gen_data(opts, instance_params, seed):
    """
    Implements the following DGP from
    double machine learning jupyter notebook at
    https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb:

    Y = T * Theta(X) + <W, Beta> + epsilon, epsilon ~ N(0, 1)
    T = <W, gamma> + eta, eta ~ N(0, 1)
    W ~ N(O, I_n_controls)
    X ~ N(0, 1)^n_controls

    Parameters
    ----------
    opts : dictionary
        dgp-specific options from config file
    instance_params : dictionary
        instance parameters beta, gamma, etc. for DGP. These are
        fixed for each time the DGP is run, so that each experiment uses the same instance
        parameters but different Y, X, T, W.
    seed : int
        random seed for random data generation

    Returns
    -------
    (X_test, Y_rain, T_train, X_train, W_train), expected_te : tuple consisting of
    training and test data, and true_parameters.
    X_test : test data to evaluate metohd on, n x n_features
    Y_train : outcome data
    T_train : treatment data
    W_train : controls data
    expected_te : true value of treatment effect
    """
    n_controls = opts['n_controls']
    n_features = opts['n_features']
    n_samples = opts['n_samples']
    support_Y = instance_params['support_Y']
    coefs_Y = instance_params['coefs_Y']
    support_T = instance_params['support_T']
    coefs_T = instance_params['coefs_T']
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Generate controls, covariates, treatments, and outcomes
    W = np.random.normal(0, 1, size=(n_samples, n_controls))
    X = np.random.uniform(0, 1, size=(n_samples, n_features))

    # Treatment effect function
    def exp_te(x):
        return np.exp(2*x[0])

    # Heterogenous treatment effects
    TE = np.array([exp_te(x) for x in X])
    T = np.dot(W[:, support_T], coefs_T) + eta_sample(n_samples)
    Y = TE * T + np.dot(W[:, support_T], coefs_Y) + epsilon_sample(n_samples)
    Y_train, Y_val, T_train, T_val, X_train, X_val, W_train, W_val = train_test_split(Y, T, X, W, test_size=.2)
    # why use train_test_split at all here?

    # Generate test data
    X_test = np.array(list(product(np.arange(0, 1, 0.01), repeat=n_features)))
    expected_te = np.array([exp_te(x_i) for x_i in X_test])

    # data, true_param
    return (X_test, Y_train, T_train, X_train, W_train), expected_te

def linear_dml_fit(data, opts, seed):
    """
    Trains the LinearDMLCateEstimator to predict the heterogenous treatment effects

    Parameters
    ----------
    data : tuple
        relevant data that comes from the data generating process
    opts : dictionary
        method-specific options
    seed : int
        random seed for random data generation

    Returns
    -------
    (X_test, const_marginal_effect), (lb, ub) : tuple of HTE and confidence interval
    X_test : Test data for use in plotting downstream
    const_marginal_effect : desired treatment effect
    (lb, ub) : tuple of confidence intervals for each data point in X_test
    """
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    inference = opts['inference']

    est = LinearDMLCateEstimator(model_y=model_y, model_t=model_t, random_state=seed)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as  est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def sparse_linear_dml_poly_fit(data, opts, seed):
    """
    Trains the SparseLinearDMLCateEstimator to predict the heterogenous treatment effects

    Parameters
    ----------
    data : tuple
        relevant data that comes from the data generating process
    opts : dictionary
        method-specific options
    seed : int
        random seed for random data generation

    Returns
    -------
    (X_test, const_marginal_effect), (lb, ub) : tuple of HTE and confidence interval
    X_test : Test data for use in plotting downstream
    const_marginal_effect : desired treatment effect
    (lb, ub) : tuple of confidence intervals for each data point in X_test
    """
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    featurizer = opts['featurizer']
    inference = opts['inference']

    est = SparseLinearDMLCateEstimator(model_y=model_y,
        model_t=model_t,
        featurizer=featurizer,
        random_state=seed)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def dml_poly_fit(data, opts ,seed):
    """
    Trains the DMLCateEstimator to predict the heterogenous treatment effects

    Parameters
    ----------
    data : tuple
        relevant data that comes from the data generating process
    opts : dictionary
        method-specific options
    seed : int
        random seed for random data generation

    Returns
    -------
    (X_test, const_marginal_effect), (lb, ub) : tuple of HTE and confidence interval
    X_test : Test data for use in plotting downstream
    const_marginal_effect : desired treatment effect
    (lb, ub) : tuple of confidence intervals for each data point in X_test
    """
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    featurizer = opts['featurizer']
    model_final = opts['model_final']
    inference = opts['inference']

    est = DMLCateEstimator(model_y=model_y,
        model_t=model_t,
        model_final=model_final,
        featurizer=featurizer,
        random_state=seed)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def forest_dml_fit(data, opts, seed):
    """
    Trains the ForestDMLCateEstimator to predict the heterogenous treatment effects

    Parameters
    ----------
    data : tuple
        relevant data that comes from the data generating process
    opts : dictionary
        method-specific options
    seed : int
        random seed for random data generation

    Returns
    -------
    (X_test, const_marginal_effect), (lb, ub) : tuple of HTE and confidence interval
    X_test : Test data for use in plotting downstream
    const_marginal_effect : desired treatment effect
    (lb, ub) : tuple of confidence intervals for each data point in X_test
    """
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    discrete_treatment = opts['discrete_treatment']
    n_estimators = opts['n_estimators']
    subsample_fr = opts['subsample_fr']
    min_samples_leaf = opts['min_samples_leaf']
    min_impurity_decrease = opts['min_impurity_decrease']
    verbose = opts['verbose']
    min_weight_fraction_leaf = opts['min_weight_fraction_leaf']
    inference = opts['inference']

    est = ForestDMLCateEstimator(model_y=model_y,
        model_t=model_t,
        discrete_treatment=discrete_treatment,
        n_estimators=n_estimators,
        subsample_fr=subsample_fr,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        verbose=verbose,
        min_weight_fraction_leaf=min_weight_fraction_leaf)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def main():
    pass

if __name__=="__main__":
    main()
