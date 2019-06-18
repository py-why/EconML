# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from sklearn.linear_model import Lasso, LassoCV, RidgeCV, LinearRegression
import scipy.sparse
import econml.dml
from econml.utilities import *
import econml.dgp
import datetime
import pickle
import pytest
from econml.selective_regularization import SelectiveLasso, SelectiveRidge


def apply_group_test(n_products, n_stores, controls, treatments, features, constant_controls,
                     t_model, y_model, f_model, X, Y):
    block_size = n_products * n_stores

    class GroupRegression():
        def __init__(self, model, compute_gp_avgs, is_first_stage,
                     effect_features=None, constant_features=None, constant_controls=None):
            self._model = model
            self._compute_gp_avgs = compute_gp_avgs
            assert is_first_stage == (effect_features is None) == (
                constant_features is not None) == (constant_controls is not None)
            self._is_first_stage = is_first_stage
            if is_first_stage:
                if np.size(constant_features) == 0:
                    constant_features = np.empty((block_size, 0))
                if np.size(constant_controls) == 0:
                    constant_controls = np.empty((block_size, 0))
                assert np.shape(constant_controls)[0] == block_size
                assert np.shape(constant_features)[0] == block_size
                constant_features = tocoo(constant_features)
                if compute_gp_avgs:
                    # group by product; sum and subtract original; divide by (n_p-1)
                    group_controls = (reshape(constant_controls, (n_stores, n_products, 1, -1)).sum(axis=1) -
                                      reshape(constant_controls, (n_stores, n_products, -1))) / (n_products - 1)
                    constant_controls = hstack([constant_controls, reshape(group_controls, (block_size, -1))])
                self._features = hstack([np.ones((block_size, 1)), constant_features])
                self._controls = cross_product(self._features, constant_controls)
            else:
                self._features = effect_features
                self._controls = scipy.sparse.csr_matrix(np.empty((block_size, 0)))

        def _reshape(self, X):
            assert X.shape[1] % block_size == 0
            n_features = X.shape[1] // block_size
            n_weeks = X.shape[0]

            X = reshape(X, (n_weeks * block_size, n_features))

            if self._compute_gp_avgs:
                # group by product; sum and subtract original; divide by (n_p-1)
                group_X = (reshape(X, (n_weeks * n_stores, n_products, 1, n_features)).sum(axis=1) -
                           reshape(X, (n_weeks * n_stores, n_products, n_features))) / (n_products - 1)
                X = hstack([X, reshape(group_X, (n_weeks * block_size, n_features))])

            X = cross_product(X, vstack([self._features for r in range(n_weeks)]))
            return hstack([X, vstack([self._controls for w in range(n_weeks)])])

        def fit(self, X, Y):
            assert Y.shape[1] == block_size
            # for now, require one feature per store/product combination
            # TODO: would be nice to relax this somehow
            assert X.shape[0] == Y.shape[0]
            n_weeks = Y.shape[0]
            Y = np.reshape(Y, (-1, 1))  # flatten prices into 1-d list
            self._model.fit(self._reshape(X), Y)

        def predict(self, X):
            predicted = self._model.predict(self._reshape(X))
            return np.reshape(predicted, (-1, block_size))

        @property
        def coef_(self):
            return self._model.coef_

    return econml.dml.LinearDMLCateEstimator(
        model_t=GroupRegression(t_model, constant_features=[], constant_controls=constant_controls,
                                compute_gp_avgs=False, is_first_stage=True),
        model_y=GroupRegression(y_model, constant_features=features, constant_controls=constant_controls,
                                compute_gp_avgs=True, is_first_stage=True),
        model_final=GroupRegression(f_model, effect_features=features, compute_gp_avgs=True, is_first_stage=False)
    ).fit(X, Y).coef_


@pytest.mark.xfail
def test_simple_groups():
    X = np.array([[1, 2, 3, 0, -2, -3],
                  [4, 5, 6, -3, -6, -6],
                  [1, 2, 3, 0, -2, -3],
                  [4, 5, 6, -3, -6, -6]])
    Y = np.array([[-2.4, 6.8, 6.6],
                  [1.6, 16.2, 12.6],
                  [-2.4, 6.8, 6.6],
                  [1.6, 16.2, 12.6]])
    return apply_group_test(3, 1, [0, 1, 2], [3, 4, 5], scipy.sparse.csr_matrix(np.array([[1], [1], [1]])), [],
                            LinearRegression(fit_intercept=False),
                            LinearRegression(fit_intercept=False),
                            LinearRegression(fit_intercept=False), X, Y)


@pytest.mark.xfail
def test_complex_groups(n_products=5, n_stores=3, n_weeks=7):
    block_size = n_products * n_stores

    # alphas vary by product, not by store
    op_alphas = np.random.normal(-3, 0.5, n_products)
    # note: product varies faster than store in the flattened list - this must also be true of gammas, betas, etc.
    op_alphas_tiled = np.tile(op_alphas, n_stores)

    # one cross-price term per product, which is based on the average price
    # of all other goods sold at the same store in the same week
    xp_alphas = np.random.normal(0.5, 0.3, n_products)
    # note: product varies faster than store in the flattened list - this must also be true of gammas, betas, etc.
    xp_alphas_tiled = np.tile(xp_alphas, n_stores)

    # store-specific and product-specific gammas and betas (which are positively correlated)
    coeffs_s = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=n_stores)
    coeffs_p = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=n_products)
    gammas = np.array([coeffs_s[s, 0] + coeffs_p[p, 0] for s in range(n_stores) for p in range(n_products)])
    betas = np.array([coeffs_s[s, 1] + coeffs_p[p, 1] for s in range(n_stores) for p in range(n_products)])

    # features: product dummies, store dummies
    # with one missing to preserve rank: store_n = sum(product_j) - sum(store_j, j!=n)
    product_features = scipy.sparse.csr_matrix(
        (np.ones(block_size), [p for s in range(n_stores) for p in range(n_products)], range(0, block_size + 1, 1)))
    store_features = scipy.sparse.csr_matrix((np.ones(block_size), [s for s in range(
        n_stores) for p in range(n_products)], range(0, block_size + 1, 1)))
    block_features = hstack([product_features, store_features[:, range(n_stores - 1)]])
    on_sale_feature = np.random.binomial(1, 0.2, size=(n_weeks, n_products * n_stores))

    compound_effects = np.empty((n_weeks, block_size))
    compound_treatments = np.empty((n_weeks, block_size))
    group_treatments = np.empty((n_weeks, block_size))  # needed only for direct regression comparisons

    # observe n_products * n_stores prices, same number of quantities
    for w in range(n_weeks):
        prices = gammas + on_sale_feature[w] + np.random.normal(size=gammas.shape)
        other_prices = (np.reshape(prices, (n_stores, n_products, 1)).sum(axis=1) -
                        np.reshape(prices, (n_stores, n_products))) / (n_products - 1)
        other_prices = np.reshape(other_prices, block_size)
        quantities = op_alphas_tiled * prices + \
            on_sale_feature[w] + xp_alphas_tiled * other_prices + betas + np.random.normal(size=betas.shape)

        compound_effects[w] = quantities
        compound_treatments[w] = prices

        group_treatments[w] = other_prices

    # for direct regression comparisons, we need a pivoted version
    simple_effects = np.reshape(compound_effects, (-1, 1))

    # "treatments" for direct regression include treatments, plus treatments interacted with product dummies,
    # plus the same for "group treatments" (average treatment of other products in the same store/week)
    simple_treatments = hstack([np.reshape(compound_treatments, (-1, 1)),
                                scipy.sparse.csr_matrix((np.reshape(compound_treatments, -1),
                                                         [p for w in range(n_weeks) for s in range(n_stores)
                                                          for p in range(n_products)],
                                                         range(0, n_weeks * block_size + 1, 1))),
                                np.reshape(group_treatments, (-1, 1)),
                                scipy.sparse.csr_matrix((np.reshape(group_treatments, -1),
                                                         [p for w in range(n_weeks) for s in range(n_stores)
                                                          for p in range(n_products)],
                                                         range(0, n_weeks * block_size + 1, 1)))])
    # for direct regression, we also need to append the features
    # (both the "constant features" as well as the normal ones)
    simple_results = hstack([simple_treatments,
                             on_sale_feature.reshape((-1, 1)),
                             vstack([block_features for _ in range(n_weeks)])]).tocsr()

    ridge = apply_group_test(n_products,
                             n_stores,
                             range(block_size),
                             range(block_size, block_size + block_size),
                             hstack([np.ones((block_size, 1)), product_features]),
                             block_features,
                             LinearRegression(),
                             LinearRegression(),
                             # NOTE: need to set cv because default generic algorithm is super slow for sparse matrices
                             RidgeCV(cv=5),
                             np.hstack([on_sale_feature, compound_treatments]),
                             compound_effects)

    ols = apply_group_test(n_products,
                           n_stores,
                           range(block_size),
                           range(block_size, block_size + block_size),
                           product_features,
                           block_features,
                           LinearRegression(),
                           LinearRegression(),
                           LinearRegression(),
                           np.hstack([on_sale_feature, compound_treatments]),
                           compound_effects)

    ridge_direct = RidgeCV(cv=5).fit(simple_results, simple_effects)
    sridge_direct = SelectiveRidge(
        1, simple_results.shape[1],
        list(range(1, n_products + 1)) + list(range(n_products + 2, 2 * n_products + 2)),
        alpha=ridge_direct.alpha_, learning_rate=1.0
    ).fit(simple_results.toarray(), simple_effects)

    return (op_alphas,
            xp_alphas,
            ridge.reshape((2, -1)),
            ols.reshape((2, -1)),
            ridge_direct.coef_[:, :(n_products + 1) * 2].reshape((2, -1)),
            sridge_direct.coef_[:(n_products + 1) * 2, :].reshape((2, -1)))


@pytest.mark.xfail
def test_compound_vs_simple_model(n_products=100, n_stores=2, n_weeks=10):

    block_size = n_products * n_stores

    # alphas vary by product, not by store
    alphas = np.random.normal(-3, 0.5, n_products)
    # note: product varies faster than store in the flattened list - this must also be true of gammas, betas, etc.
    alphas_tiled = np.tile(alphas, n_stores)

    # store-specific and product-specific gammas
    gamma_s = np.random.normal(size=n_stores)
    gamma_p = np.random.normal(size=n_products)
    gammas = np.array([gamma_s[s] + gamma_p[p] for s in range(n_stores) for p in range(n_products)])

    # store-specific and product-specific betas
    beta_s = np.random.normal(size=n_stores)
    beta_p = np.random.normal(size=n_products)
    betas = np.array([beta_s[s] + beta_p[p] for s in range(n_stores) for p in range(n_products)])

    # features: product dummies, store dummies
    # with one missing to preserve rank: store_n = sum(product_j) - sum(store_j, j!=n)
    product_features = scipy.sparse.csr_matrix(
        (np.ones(block_size), [p for s in range(n_stores) for p in range(n_products)], range(0, block_size + 1, 1)))
    store_features = scipy.sparse.csr_matrix((np.ones(block_size), [s for s in range(
        n_stores) for p in range(n_products)], range(0, block_size + 1, 1)))
    block_features = hstack([product_features, store_features[:, range(n_stores - 1)]])

    compound_effects = np.empty((n_weeks, block_size))

    # we need only the prices for the compound model; all dummies are created internally
    compound_treatments = np.empty((n_weeks, block_size))

    # observe n_products * n_stores prices, same number of quantities
    for w in range(n_weeks):
        prices = gammas + np.random.normal(size=gammas.shape)
        quantities = alphas_tiled * prices + betas + np.random.normal(size=betas.size)

        compound_effects[w] = quantities
        compound_treatments[w] = prices

    simple_effects = np.reshape(compound_effects, (-1, 1))

    # simple results include treatments, plus treatments interacted with product dummies,
    # for use with the direct methods
    simple_treatments = hstack([np.reshape(compound_treatments, (-1, 1)),
                                scipy.sparse.csr_matrix((np.reshape(compound_treatments, -1),
                                                         [p
                                                          for w in range(n_weeks)
                                                          for s in range(n_stores)
                                                          for p in range(n_products)],
                                                         range(0, n_weeks * block_size + 1, 1)))])
    simple_results = hstack([simple_treatments, vstack([block_features for _ in range(n_weeks)])]).tocsr()

    class FirstStageRegression():
        def __init__(self, model, block_features):
            self._model = model
            assert np.shape(block_features)[0] == block_size
            self._features = tocoo(block_features)

        def fit(self, X, Y):
            # X should have 0 columns; we will instead pivot Y and fit against the features passed into the constructor
            assert X.shape[1] == 0
            assert Y.shape[1] == block_size
            n_rows = Y.shape[0]
            Y = np.reshape(Y, -1)  # flatten prices into 1-d list
            X = vstack([self._features for _ in range(n_rows)])
            self._model.fit(X, Y)

        def predict(self, X):
            assert X.shape[1] == 0
            predicted = self._model.predict(self._features)
            return np.tile(np.reshape(predicted, (1, -1)), (X.shape[0], 1))

    class SecondStageRegression():
        def __init__(self, model):
            self._model = model

        def _reshape(self, X):
            n_weeks = X.shape[0]
            X = np.reshape(X, (-1, 1))
            return hstack([X,
                           cross_product(X, vstack([product_features for w in range(n_weeks)]))])

        def fit(self, X, Y):
            assert X.shape[1] == Y.shape[1] == block_size
            Y = np.reshape(Y, -1)
            self._model.fit(self._reshape(X), Y)

        def predict(self, X):
            return np.reshape(self._model.predict(self._reshape(X)), (-1, block_size))

        @property
        def coef_(self):
            return self._model.coef_

    compound = dml.LinearDMLCateEstimator(model_t=FirstStageRegression(LinearRegression(), block_features),
                                          model_y=FirstStageRegression(
        LinearRegression(), scipy.sparse.identity(block_size)),
        model_final=SecondStageRegression(RidgeCV())).fit(compound_treatments,
                                                          compound_effects).coef_
    simple = dml.LinearDMLCateEstimator(model_final=RidgeCV()).fit(simple_results, simple_effects).coef_

    return alphas, simple, compound


@pytest.mark.xfail
def test_many_effects(n_exp=20, n_products=100, n_stores=2, n_weeks=10):
    # underspecified model
    # Y = alpha T + \sum_i alpha_i T_i + X beta + eta
    # T = X gamma + eps

    # how to score? distance from line of all solutions?

    # given that 0, a, b, c, d is equivalent to x, a-x, b-x, c-x, d-x, compute the error
    def rmse(alphas, coefs):
        diff = np.concatenate(([0], alphas)) - coefs
        d_avg = (np.sum(diff) - 2 * diff[0]) / np.size(diff)
        ortho_d = diff - d_avg * np.ones(diff.size)
        ortho_d[0] += 2 * d_avg
        return np.linalg.norm(ortho_d) / np.sqrt(diff.size)

    # baselines: ridge, ridge-like (penalize alpha_i but not alpha_baseline or beta)
    # comparison: 2ml (OLS for T on X, OLS for Y on XxX^e, ridge or ridge-like for alphas)

    block_size = n_products * n_stores

    alphass = []
    lassos = []
    ridges = []
    doubleMls = []

    # features: one product dummy, one store dummy (each missing one level), constant
    for _ in range(n_exp):

        # alphas vary by product, not by store
        alphas = np.random.normal(-3, 0.5, n_products)
        alphas_tiled = np.tile(alphas, n_stores)

        # store-specific and product-specific gammas
        gamma_s = np.random.normal(size=(n_stores, 1))
        gamma_p = np.random.normal(size=(1, n_products))
        gammas = np.reshape(gamma_s + gamma_p, -1)

        # store-specific and product-specific betas
        beta_s = np.random.normal(size=(n_stores, 1))
        beta_p = np.random.normal(size=(1, n_products))
        betas = np.reshape(beta_s + beta_p, -1)

        # features: product dummies, store dummies
        # with one missing to preserve rank: store_n = sum(product_j) - sum(store_j, j!=n)
        block_features = np.array([[float(i == p) for i in range(n_products)] +
                                   [float(i == s) for i in range(n_stores - 1)]
                                   for s in range(n_stores)
                                   for p in range(n_products)])

        quantities = np.empty(n_weeks * block_size)

        # columns: prices interacted with products (and constant), features
        results = np.empty((n_weeks * block_size, (1 + n_products) + block_features.shape[1]))

        # observe n_products * n_stores prices, same number of quantities
        for w in range(n_weeks):
            prices = gammas + np.random.normal(size=gammas.shape)
            quantities[w * block_size: (w + 1) * block_size] = alphas_tiled * prices + \
                betas + np.random.normal(size=betas.size)
            results[w * block_size: (w + 1) * block_size, 0:1 + n_products] = prices.reshape((-1, 1)) * \
                np.concatenate((np.ones((block_features.shape[0], 1)), block_features[:, :n_products]), axis=1)
            results[w * block_size: (w + 1) * block_size, 1 + n_products:] = block_features

        lassos.append(LassoCV().fit(results, quantities).coef_[0:n_products + 1])
        ridges.append(RidgeCV().fit(results, quantities).coef_[0:n_products + 1])
        # use features starting at index 1+n_products to skip all prices
        doubleMls.append(econml.dml.LinearDMLCateEstimator(model_final=RidgeCV()).fit(results, quantities).coef_)
        alphass.append(alphas)

    pickleFile = open('pickledSparse_{0}_{1}_{2}_{3}.pickle'.format(n_exp, n_products, n_stores, n_weeks), 'wb')
    pickle.dump((alphass, ridges, lassos, doubleMls), pickleFile)
    pickleFile.close()

    # pickleFile = open('pickledSparse_{0}_{1}_{2}_{3}.pickle'.format(n_exp, n_products, n_stores, n_weeks), 'rb')
    # alphass, ridges, lassos, doubleMls = pickle.load(pickleFile)
    # pickleFile.close()


@pytest.mark.xfail
def test_integration():
    """ Monte carlo integration unit test of the whole heterogeneous treatment effect
    estimation process using DML. Also comparison with other simpler regression approaches. """
    np.random.seed(3)

    ##############################################
    # Defining the parameters of Monte Carlo
    ##############################################
    n_exp = 200  # Number of experiments
    n_samples = 200  # Number of samples
    n_cov = 300  # Number of covariates
    n_treatments = 1

    dml_r2scores = []
    direct_r2scores1 = []
    direct_r2scores2 = []
    dml_tes = []
    direct_tes1 = []
    direct_tes2 = []

    sparsities = range(2, 220, 10)
    for sparsity in sparsities:
        print("Sparsity: {0}".format(sparsity))
        effect = 2. + np.arange(n_treatments)

        # Estimation parameters
        alpha_lasso = np.sqrt(1. * np.log(n_cov) / n_samples)
        internal_reg_y = Lasso(alpha=alpha_lasso)  # Internal regressor for outcome
        internal_reg_t = Lasso(alpha=alpha_lasso)  # Internal regressor for treatment
        internal_reg_f = LinearRegression(fit_intercept=False)  # Internal regressor for final stage
        internal_reg_c = LassoCV()  # Internal regressor for treatment

        direct_reg1 = SelectiveLasso(1, n_cov + 1, np.arange(0, n_cov), alpha=alpha_lasso,
                                     learning_rate=1.)  # LassoCV() #Lasso(alpha=alpha_lasso)
        direct_reg2 = LassoCV()

        ####################################################################
        # Estimating the parameters of the DGP with DML. Running multiple
        # Monte Carlo experiments.
        ####################################################################
        dml_r2score = []
        direct_r2score1 = []
        direct_r2score2 = []
        dml_te = []
        direct_te1 = []
        direct_te2 = []
        for exp in range(n_exp):
            if exp % 10 == 0:
                print("  Experiment: {1}".format(sparsity, exp))

            # Sparse coefficients of treatment as a function of co-variates
            alpha_sparsity = sparsity
            alpha_support = np.random.choice(n_cov, alpha_sparsity, replace=False)
            alpha = np.zeros(n_cov)
            alpha[alpha_support] = np.random.normal(size=len(alpha_support))
            alpha = alpha.reshape((-1, 1))
            alpha = alpha / np.linalg.norm(alpha)
            # Coefficients of outcomes as a function of co-variates
            beta_sparsity = sparsity
            beta_support = np.random.choice(n_cov, beta_sparsity, replace=False)
            beta = np.zeros(n_cov)
            beta[beta_support] = np.random.normal(size=len(beta_support))
            beta = beta / np.linalg.norm(beta)

            # DGP. Create samples of data (y, T, X) from known truth
            y, T, X, _ = econml.dgp.dgp_data_multiple_treatments(
                n_samples, n_cov, n_treatments, alpha, beta, effect)

            # DML Estimation.
            dml_reg = econml.dml.DML(np.arange(X.shape[1]), [], np.arange(X.shape[1], X.shape[1] + T.shape[1]),
                                     model_y=internal_reg_y,
                                     model_t=internal_reg_t,
                                     model_f=internal_reg_f,
                                     model_c=internal_reg_c)
            dml_reg.fit(np.concatenate((X, T), axis=1), y)

            y_test, T_test, X_test = econml.dgp.dgp_counterfactual_data_multiple_treatments(
                n_samples, n_cov, beta, effect, 5. * np.ones(n_treatments))
            dml_r2score.append(dml_reg.score(np.concatenate((X_test, T_test), axis=1), y_test))
            dml_te.append(dml_reg.effect(np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 0))))

            # Estimation with other methods for comparison
            direct_reg1.fit(np.concatenate((X, T), axis=1), y)
            direct_r2score1.append(direct_reg1.score(np.concatenate((X_test, T_test), axis=1), y_test))
            direct_te1.append(direct_reg1.coef_[X.shape[1]])

            direct_reg2.fit(np.concatenate((X, T), axis=1), y)
            direct_r2score2.append(direct_reg2.score(np.concatenate((X_test, T_test), axis=1), y_test))
            direct_te2.append(direct_reg2.coef_[X.shape[1]])

        dml_r2scores.append(dml_r2score)
        direct_r2scores1.append(direct_r2score1)
        direct_r2scores2.append(direct_r2score2)
        dml_tes.append(dml_te)
        direct_tes1.append(direct_te1)
        direct_tes2.append(direct_te2)

        ##########################################################
        # Plotting the results and saving
        ##########################################################
        # plt.figure(figsize=(20, 10))
        # plt.subplot(1, 4, 1)
        # plt.title("DML R^2: median {:.3f}, mean {:.3f}".format(np.median(dml_r2score), np.mean(dml_r2score)))
        # plt.hist(dml_r2score)
        # plt.subplot(1, 4, 2)
        # plt.title("Direct Lasso R^2: median {:.3f}, mean {:.3f}".format(np.median(direct_r2score),
        #                                                                 np.mean(direct_r2score)))
        # plt.hist(direct_r2score)
        # plt.subplot(1, 4, 3)
        # plt.title("DML Treatment Effect Distribution: mean {:.3f}, std {:.3f}".format(np.mean(dml_te),
        #                                                                               np.std(dml_te)))
        # plt.hist(np.array(dml_te).flatten())
        # plt.subplot(1, 4, 4)
        # plt.title("Direct Treatment Effect Distribution: mean {:.3f}, std {:.3f}".format(np.mean(direct_te),
        #                                                                                  np.std(direct_te)))
        # plt.hist(np.array(direct_te).flatten())
        # plt.tight_layout()
        # plt.savefig("r2_comparison.png")

    # Plotting the results and saving
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title("Variance of estimated effect error")
    plt.plot(sparsities, [np.var(te - effect) for te in dml_tes], label='dml')
    plt.plot(sparsities, [np.var(te - effect) for te in direct_tes1], label='sel. lasso')
    plt.plot(sparsities, [np.var(te - effect) for te in direct_tes2], label='lassoCv')
    plt.subplot(1, 2, 2)
    plt.title("Median R^2")
    plt.plot(sparsities, [np.median(r2s) for r2s in dml_r2scores], label='dml')
    plt.plot(sparsities, [np.median(r2s) for r2s in direct_r2scores1], label='sel. lasso')
    plt.plot(sparsities, [np.median(r2s) for r2s in direct_r2scores2], label='lassoCv')
    plt.tight_layout()
    plt.savefig("sparsitySweep.png")

    pickleFile = open('pickledResults.pickle', 'wb')
    pickle.dump((sparsities, dml_r2scores, direct_r2scores1, direct_r2scores2,
                 dml_tes, direct_tes1, direct_tes2), pickleFile)
    pickleFile.close()


if __name__ == '__main__':
    test_integration()
