# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Data generating processes for correctness testing."""

import numpy as np
from itertools import product
from econml.utilities import cross_product


########################################################
# Perfect Data DGPs for Testing Correctness of Code
########################################################

def dgp_perfect_data_multiple_treatments(n_samples, n_cov, n_treatments, Alpha, beta, effect):
    """Generate data with carefully crafted controls and noise for perfect estimation."""
    # Generate random control co-variates
    X = np.random.randint(low=-2, high=2, size=(n_samples, n_cov))
    # Create epsilon residual treatments that deterministically sum up to
    # zero
    epsilon = np.random.normal(size=(n_samples, n_treatments))
    # Re-calibrate epsilon to make sure that empirical distribution of epsilon
    # conditional on each co-variate vector is equal to zero
    unique_X = np.unique(X, axis=0)
    for u_row in unique_X:
        # We simply subtract the conditional mean from the epsilons
        epsilon[np.all(X == u_row, axis=1),
                :] -= np.mean(epsilon[np.all(X == u_row, axis=1), :])
    # Construct treatments as T = X*A + epsilon
    T = np.dot(X, Alpha) + epsilon
    # Construct outcomes as y = X*beta + T*effect
    y = np.dot(X, beta) + np.dot(T, effect)

    return y, T, X, epsilon


def dgp_perfect_data_multiple_treatments_and_features(n_samples, n_cov, feat_sizes, n_treatments, Alpha, beta, effect):
    """Generate data with carefully crafted controls and noise for perfect estimation."""
    # Generate random control co-variates
    X = np.random.randint(low=-2, high=2, size=(n_samples, n_cov))
    X_f = [
        c
        for c in (np.arange(s - 1).reshape((1, s - 1)) ==
                  np.random.randint(s, size=(X.shape[0], 1))).astype(np.int)
        for s in feat_sizes]
    # Create epsilon residual treatments that deterministically sum up to
    # zero
    epsilon = np.random.normal(size=(n_samples, n_treatments))
    # Re-calibrate epsilon to make sure that empirical distribution of epsilon
    # conditional on each co-variate vector is equal to zero
    unique_X = np.unique(X, axis=0)
    for u_row in unique_X:
        # We simply subtract the conditional mean from the epsilons
        epsilon[np.all(X == u_row, axis=1),
                :] -= np.mean(epsilon[np.all(X == u_row, axis=1), :])
    # Construct treatments as T = X*A + epsilon
    T = np.dot(X, Alpha) + epsilon
    # Construct outcomes as y = X*beta + T*effect
    y = np.dot(X, beta) + np.dot(T, effect)

    return y, T, X, epsilon


def dgp_perfect_counterfactual_data_multiple_treatments(n_samples, n_cov, beta, effect, treatment_vector):
    """Generate data with carefully crafted controls and noise for perfect estimation."""
    # Generate random control co-variates
    X = np.random.randint(low=-2, high=2, size=(n_samples, n_cov))
    # Construct treatments as T = X*A + epsilon
    T = np.repeat(treatment_vector.reshape(1, -1), n_samples, axis=0)
    # Construct outcomes as y = X*beta + T*effect
    y = np.dot(X, beta) + np.dot(T, effect)

    return y, T, X


def dgp_data_multiple_treatments(n_samples, n_cov, n_treatments, Alpha, beta, effect):
    """Generate data from a linear model using covariates drawn from gaussians and with gaussian noise."""
    # Generate random control co-variates
    X = np.random.normal(size=(n_samples, n_cov))
    # Create epsilon residual treatments
    epsilon = np.random.normal(size=(n_samples, n_treatments))
    # Construct treatments as T = X*A + epsilon
    T = np.dot(X, Alpha) + epsilon
    # Construct outcomes as y = X*beta + T*effect + eta
    y = np.dot(X, beta) + np.dot(T, effect) + np.random.normal(size=n_samples)

    return y, T, X, epsilon


def dgp_counterfactual_data_multiple_treatments(n_samples, n_cov, beta, effect, treatment_vector):
    """Generate data with carefully crafted controls and noise for perfect estimation."""
    # Generate random control co-variates
    X = np.random.normal(size=(n_samples, n_cov))
    # Use the same treatment vector for each row
    T = np.repeat(treatment_vector.reshape(1, -1), n_samples, axis=0)
    # Construct outcomes as y = X*beta + T*effect
    y = np.dot(X, beta) + np.dot(T, effect) + np.random.normal(size=n_samples)

    return y, T, X
