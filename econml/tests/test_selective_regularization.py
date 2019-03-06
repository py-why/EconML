# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# TODO: make this test actually test something instead of generating images
import pytest
import numpy as np
import econml.dgp
from econml.selective_regularization import SelectiveLasso


@pytest.mark.slow
def test_selective_regularization():

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_samples = 200  # Number of samples
    n_cov = 300  # Number of covariates
    n_treatments = 1

    # Sparse coefficients of treatment as a function of co-variates
    alpha_sparsity = 4
    alpha_support = np.random.choice(n_cov, alpha_sparsity, replace=False)
    alpha = np.zeros(n_cov)
    alpha[alpha_support] = np.random.normal(size=len(alpha_support))
    alpha = alpha.reshape((-1, 1))
    # Coefficients of outcomes as a function of co-variates
    beta_sparsity = 4
    beta_support = np.random.choice(n_cov, beta_sparsity, replace=False)
    beta = np.zeros(n_cov)
    beta[beta_support] = np.random.normal(size=len(beta_support))

    effect = 2. + np.arange(n_treatments)

    reg = SelectiveLasso(1, n_cov + 1, np.arange(0, n_cov), steps=1000, learning_rate=1.)

    for t in range(20):
        # DGP. Create samples of data (y, T, X) from known truth
        y, T, X, _ = econml.dgp.dgp_data_multiple_treatments(
            n_samples, n_cov, n_treatments, alpha, beta, np.array([t]))
        reg.fit(np.concatenate((X, T), axis=1), y)
        print(reg.coef_[X.shape[1], 0])

    coefs = []
    for t in range(20):
        # DGP. Create samples of data (y, T, X) from known truth
        y, T, X, _ = econml.dgp.dgp_data_multiple_treatments(
            n_samples, n_cov, n_treatments, alpha, beta, np.array([2.]))
        reg.fit(np.concatenate((X, T), axis=1), y)
        coef = reg.coef_[X.shape[1], 0]
        coefs.append(coef)
        print(coef)
    plt.figure()
    plt.hist(coefs)
    plt.savefig('selective_lasso_estimates.png')
