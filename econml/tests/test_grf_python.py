# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import logging
import time
import random
import numpy as np
import pandas as pd
import pytest
from econml.grf import RegressionForest, CausalForest, CausalIVForest, MultiOutputGRF
from econml.utilities import cross_product
from copy import deepcopy
from sklearn.utils import check_random_state
import scipy.stats


class TestGRFPython(unittest.TestCase):

    def _get_base_config(self):
        return {'n_estimators': 1, 'subforest_size': 1, 'max_depth': 2,
                'min_samples_split': 2, 'min_samples_leaf': 1,
                'inference': False, 'max_samples': 1.0, 'honest': False,
                'n_jobs': None, 'random_state': 123}

    def _get_regression_data(self, n, n_features, random_state):
        X = np.zeros((n, n_features))
        X[:, 0] = np.arange(X.shape[0])
        X[:, 1] = np.random.RandomState(random_state).normal(0, 1, size=(X.shape[0]))
        y = 1.0 * (X[:, 0] >= n / 2).reshape(-1, 1)
        y += 1.0 * (X[:, 0] >= n / 4).reshape(-1, 1)
        y += 1.0 * (X[:, 0] >= 3 * n / 4).reshape(-1, 1)
        return X, y, y

    def test_regression_tree_internals(self):
        base_config = self._get_base_config()
        n, n_features = 10, 2
        random_state = 123
        X, y, truth = self._get_regression_data(n, n_features, random_state)
        forest = RegressionForest(**base_config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([0, 0, -2, -2, 0, -2, -2]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, 2.5, - 2, -2, 7.5, -2, -2]))
        np.testing.assert_array_almost_equal(tree.value.flatten()[:3],
                                             np.array([np.mean(y),
                                                       np.mean(y[X[:, tree.feature[0]] < tree.threshold[0]]),
                                                       np.mean(y[(X[:, tree.feature[0]] < tree.threshold[0]) &
                                                                 (X[:, tree.feature[1]] < tree.threshold[1])])]),
                                             decimal=5)
        np.testing.assert_array_almost_equal(tree.predict(X), y, decimal=5)
        tree.predict_precond(X)
        tree.predict_jac(X)
        tree.predict_precond_and_jac(X)

        less = X[:, tree.feature[0]] < tree.threshold[0]

        # testing importances
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = np.var(y)
        np.testing.assert_array_almost_equal(tree.compute_feature_importances(normalize=False),
                                             feature_importances, decimal=5)
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = np.var(y) - np.var(y[less])
        np.testing.assert_array_almost_equal(tree.compute_feature_importances(normalize=False, max_depth=0),
                                             feature_importances, decimal=5)
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = np.var(y) - np.var(y[less]) + .5 * (np.var(y[less]))
        np.testing.assert_array_almost_equal(tree.compute_feature_importances(normalize=False,
                                                                              max_depth=1, depth_decay=1.0),
                                             feature_importances, decimal=5)
        # testing heterogeneity importances
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = 5 * 5 * (np.mean(y[less]) - np.mean(y[~less]))**2 / 100
        np.testing.assert_array_almost_equal(tree.compute_feature_heterogeneity_importances(normalize=False,
                                                                                            max_depth=0),
                                             feature_importances, decimal=5)
        feature_importances[0] += .5 * (2 * 2 * 3 * (1)**2 / 5) / 10
        np.testing.assert_array_almost_equal(tree.compute_feature_heterogeneity_importances(normalize=False,
                                                                                            max_depth=1,
                                                                                            depth_decay=1.0),
                                             feature_importances, decimal=5)
        feature_importances[0] += .5 * (2 * 2 * 3 * (1)**2 / 5) / 10
        np.testing.assert_array_almost_equal(tree.compute_feature_heterogeneity_importances(normalize=False),
                                             feature_importances, decimal=5)

        # Testing that all parameters do what they are supposed to
        config = deepcopy(base_config)
        config['min_samples_leaf'] = 5
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = deepcopy(base_config)
        config['min_samples_split'] = 11
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([-2]))
        np.testing.assert_array_equal(tree.threshold, np.array([-2]))
        np.testing.assert_array_almost_equal(tree.predict(X), np.mean(y), decimal=5)
        np.testing.assert_array_almost_equal(tree.predict_full(X), np.mean(y), decimal=5)

        config = deepcopy(base_config)
        config['min_weight_fraction_leaf'] = .5
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))
        # testing predict, apply and decision path
        less = X[:, tree.feature[0]] < tree.threshold[0]
        y_pred = np.zeros((X.shape[0], 1))
        y_pred[less] = np.mean(y[less])
        y_pred[~less] = np.mean(y[~less])
        np.testing.assert_array_almost_equal(tree.predict(X), y_pred, decimal=5)
        np.testing.assert_array_almost_equal(tree.predict_full(X), y_pred, decimal=5)
        decision_path = np.zeros((X.shape[0], len(tree.feature)))
        decision_path[less, :] = np.array([1, 1, 0])
        decision_path[~less, :] = np.array([1, 0, 1])
        np.testing.assert_array_equal(tree.decision_path(X).todense(), decision_path)
        apply = np.zeros(X.shape[0])
        apply[less] = 1
        apply[~less] = 2
        np.testing.assert_array_equal(tree.apply(X), apply)
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = 1
        np.testing.assert_array_equal(tree.compute_feature_importances(),
                                      feature_importances)

        config = deepcopy(base_config)
        config['min_balancedness_tol'] = 0.
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = deepcopy(base_config)
        config['min_balancedness_tol'] = 0.1
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([0, 0, -2, -2, 0, -2, -2]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, 2.5, - 2, -2, 7.5, -2, -2]))

        config = deepcopy(base_config)
        config['max_depth'] = 1
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = deepcopy(base_config)
        config['min_impurity_decrease'] = 0.9999
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = deepcopy(base_config)
        config['min_impurity_decrease'] = 1.0001
        forest = RegressionForest(**config).fit(X, y)
        tree = forest[0].tree_
        np.testing.assert_array_equal(tree.feature, np.array([-2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([-2, ]))

    def _get_causal_data(self, n, n_features, n_treatments, random_state):
        random_state = np.random.RandomState(random_state)
        X = random_state.normal(size=(n, n_features))
        T = np.zeros((n, n_treatments))
        for t in range(T.shape[1]):
            T[:, t] = random_state.binomial(1, .5, size=(T.shape[0],))
        y = ((X[:, [0]] > 0.0) + .5) * np.sum(T, axis=1, keepdims=True) + .5
        return (X, T, y, np.hstack([(X[:, [0]] > 0.0) + .5, (X[:, [0]] > 0.0) + .5]),
                np.hstack([(X[:, [0]] > 0.0) + .5, (X[:, [0]] > 0.0) + .5, .5 * np.ones((X.shape[0], 1))]))

    def _get_true_quantities(self, X, T, y, mask, criterion, fit_intercept, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        X, T, y, sample_weight = X[mask], T[mask], y[mask], sample_weight[mask]
        n_relevant_outputs = T.shape[1]
        if fit_intercept:
            T = np.hstack([T, np.ones((T.shape[0], 1))])
        alpha = y * T
        pointJ = cross_product(T, T)
        node_weight = np.sum(sample_weight)
        jac = node_weight * np.average(pointJ, axis=0, weights=sample_weight)
        precond = node_weight * np.average(alpha, axis=0, weights=sample_weight)

        if jac.shape[0] == 1:
            invJ = np.array([[1 / jac[0]]])
        elif jac.shape[0] == 4:
            det = jac[0] * jac[3] - jac[1] * jac[2]
            if abs(det) < 1e-6:
                det = 1e-6
            invJ = np.array([[jac[3], -jac[1]], [-jac[2], jac[0]]]) / det
        else:
            invJ = np.linalg.inv(jac.reshape((alpha.shape[1], alpha.shape[1])) + 1e-6 * np.eye(T.shape[1]))

        param = invJ @ precond
        jac = jac / node_weight
        precond = precond / node_weight
        if criterion == 'het':
            moment = alpha - pointJ.reshape((-1, alpha.shape[1], alpha.shape[1])) @ param
            rho = ((invJ @ moment.T).T)[:, :n_relevant_outputs] * node_weight
            impurity = np.mean(np.average(rho**2, axis=0, weights=sample_weight))
            impurity -= np.mean(np.average(rho, axis=0, weights=sample_weight)**2)
        else:
            impurity = np.mean(np.average(y**2, axis=0, weights=sample_weight))
            impurity -= (param.reshape(1, -1) @ jac.reshape((alpha.shape[1], alpha.shape[1])) @ param)[0]
        return jac, precond, param, impurity

    def _get_node_quantities(self, tree, node_id):
        return (tree.jac[node_id, :], tree.precond[node_id, :],
                tree.full_value[node_id, :, 0], tree.impurity[node_id])

    def _train_causal_forest(self, X, T, y, config, sample_weight=None):
        return CausalForest(**config).fit(X, T, y, sample_weight=sample_weight)

    def _train_iv_forest(self, X, T, y, config, sample_weight=None):
        return CausalIVForest(**config).fit(X, T, y, Z=T, sample_weight=sample_weight)

    def _test_causal_tree_internals(self, trainer):
        config = self._get_base_config()
        for criterion in ['het', 'mse']:
            for fit_intercept in [False, True]:
                for min_var_fraction_leaf in [None, .4]:
                    config['criterion'] = criterion
                    config['fit_intercept'] = fit_intercept
                    config['max_depth'] = 2
                    config['min_samples_leaf'] = 5
                    config['min_var_fraction_leaf'] = min_var_fraction_leaf
                    n, n_features, n_treatments = 100, 2, 2
                    random_state = 123
                    X, T, y, truth, truth_full = self._get_causal_data(n, n_features, n_treatments, random_state)
                    forest = trainer(X, T, y, config)
                    tree = forest[0].tree_
                    paths = np.array(forest[0].decision_path(X).todense())
                    for node_id in range(len(tree.feature)):
                        mask = paths[:, node_id] > 0
                        [np.testing.assert_allclose(a, b, atol=1e-4)
                         for a, b in zip(self._get_true_quantities(X, T, y, mask, criterion, fit_intercept),
                                         self._get_node_quantities(tree, node_id))]
                    if fit_intercept and (min_var_fraction_leaf is not None):
                        mask = np.abs(X[:, 0]) > .3
                        np.testing.assert_allclose(tree.predict(X[mask]), truth[mask], atol=.05)
                        np.testing.assert_allclose(tree.predict_full(X[mask]), truth_full[mask], atol=.05)

    def _test_causal_honesty(self, trainer):
        for criterion in ['het', 'mse']:
            for fit_intercept in [False, True]:
                for min_var_fraction_leaf, min_var_leaf_on_val in [(None, False), (.4, False), (.4, True)]:
                    for min_impurity_decrease in [0.0, 0.07]:
                        for inference in [False, True]:
                            for sample_weight in [None, 'rand']:
                                config = self._get_base_config()
                                config['honest'] = True
                                config['criterion'] = criterion
                                config['fit_intercept'] = fit_intercept
                                config['max_depth'] = 2
                                config['min_samples_leaf'] = 5
                                config['min_var_fraction_leaf'] = min_var_fraction_leaf
                                config['min_var_leaf_on_val'] = min_var_leaf_on_val
                                config['min_impurity_decrease'] = min_impurity_decrease
                                config['inference'] = inference
                                n, n_features, n_treatments = 400, 2, 2
                                if inference:
                                    config['n_estimators'] = 4
                                    config['subforest_size'] = 2
                                    config['max_samples'] = .4
                                    config['n_jobs'] = 1
                                    n = 800
                                random_state = 123
                                if sample_weight is not None:
                                    sample_weight = check_random_state(random_state).randint(0, 4, size=n)
                                X, T, y, truth, truth_full = self._get_causal_data(n, n_features,
                                                                                   n_treatments, random_state)
                                forest = trainer(X, T, y, config, sample_weight=sample_weight)
                                subinds = forest.get_subsample_inds()
                                if (sample_weight is None) and fit_intercept and (min_var_fraction_leaf is not None):
                                    mask = np.abs(X[:, 0]) > .5
                                    np.testing.assert_allclose(forest.predict(X[mask]),
                                                               truth[mask], atol=.07)
                                    np.testing.assert_allclose(forest.predict_full(X[mask]),
                                                               truth_full[mask], atol=.07)
                                    np.testing.assert_allclose(forest.predict_tree_average(X[mask]),
                                                               truth[mask], atol=.07)
                                    np.testing.assert_allclose(forest.predict_tree_average_full(X[mask]),
                                                               truth_full[mask], atol=.07)
                                forest_paths, ptr = forest.decision_path(X)
                                forest_paths = np.array(forest_paths.todense())
                                forest_apply = forest.apply(X)
                                for it, tree in enumerate(forest):
                                    tree_paths = np.array(tree.decision_path(X).todense())
                                    np.testing.assert_array_equal(tree_paths, forest_paths[:, ptr[it]:ptr[it + 1]])
                                    tree_apply = tree.apply(X)
                                    np.testing.assert_array_equal(tree_apply, forest_apply[:, it])
                                    _, samples_val = tree.get_train_test_split_inds()
                                    inds_val = subinds[it][samples_val]
                                    Xval, Tval, yval, truthval = X[inds_val], T[inds_val], y[inds_val], truth[inds_val]
                                    sample_weightval = sample_weight[inds_val] if sample_weight is not None else None
                                    paths = np.array(tree.decision_path(Xval).todense())
                                    for node_id in range(len(tree.tree_.feature)):
                                        mask = paths[:, node_id] > 0
                                        [np.testing.assert_allclose(a, b, atol=1e-4)
                                         for a, b in zip(self._get_true_quantities(Xval, Tval, yval, mask,
                                                                                   criterion, fit_intercept,
                                                                                   sample_weight=sample_weightval),
                                                         self._get_node_quantities(tree.tree_, node_id))]
                                    if ((sample_weight is None) and
                                            fit_intercept and (min_var_fraction_leaf is not None)):
                                        mask = np.abs(Xval[:, 0]) > .5
                                        np.testing.assert_allclose(tree.tree_.predict(Xval[mask]),
                                                                   truthval[mask], atol=.07)
                                    if (sample_weight is None) and min_impurity_decrease > 0.0005:
                                        assert np.all((tree.tree_.feature == 0) | (tree.tree_.feature == -2))

    def test_causal_tree(self,):
        self._test_causal_tree_internals(self._train_causal_forest)
        self._test_causal_honesty(self._train_causal_forest)

    def test_iv_tree(self,):
        self._test_causal_tree_internals(self._train_iv_forest)
        self._test_causal_honesty(self._train_iv_forest)

    def test_min_var_leaf(self,):
        random_state = np.random.RandomState(123)
        n, n_features, n_treatments = 200, 2, 1
        X = random_state.normal(size=(n, n_features))
        T = np.zeros((n, n_treatments))
        for t in range(T.shape[1]):
            T[:, t] = random_state.binomial(1, .5 + .2 * np.clip(X[:, 0], -1, 1), size=(T.shape[0],))
        y = ((X[:, [0]] > 0.0) + .5) * np.sum(T, axis=1, keepdims=True) + .5
        total_std = np.std(T)
        min_var = .7 * total_std
        for honest, min_var_fraction_leaf, min_var_leaf_on_val in [(False, None, False), (False, .8, False),
                                                                   (True, None, True), (True, .8, True)]:
            config = self._get_base_config()
            config['criterion'] = 'mse'
            config['n_estimators'] = 4
            config['max_samples'] = 1.0
            config['max_depth'] = None
            config['min_var_fraction_leaf'] = min_var_fraction_leaf
            config['fit_intercept'] = True
            config['honest'] = honest
            config['min_var_leaf_on_val'] = min_var_leaf_on_val
            forest = self._train_causal_forest(X, T, y, config)
            subinds = forest.get_subsample_inds()
            for it, tree in enumerate(forest):
                _, samples_val = tree.get_train_test_split_inds()
                inds_val = subinds[it][samples_val]
                Xval, Tval, _ = X[inds_val], T[inds_val], y[inds_val]
                paths = np.array(tree.decision_path(Xval).todense())
                if min_var_fraction_leaf is None:
                    with np.testing.assert_raises(AssertionError):
                        for node_id in range(len(tree.tree_.feature)):
                            mask = paths[:, node_id] > 0
                            np.testing.assert_array_less(min_var - 1e-7, np.std(Tval[mask]))
                else:
                    for node_id in range(len(tree.tree_.feature)):
                        mask = paths[:, node_id] > 0
                        np.testing.assert_array_less(min_var - 1e-7, np.std(Tval[mask]))

    def test_subsampling(self,):
        # test that the subsampling scheme past to the trees is correct
        random_state = 123
        n, n_features, n_treatments = 10, 2, 2
        n_estimators = 600
        config = self._get_base_config()
        config['n_estimators'] = n_estimators
        config['max_samples'] = .7
        config['max_depth'] = 1
        X, T, y, _, _ = self._get_causal_data(n, n_features, n_treatments, random_state)
        forest = self._train_causal_forest(X, T, y, config)
        subinds = forest.get_subsample_inds()
        inds, counts = np.unique(subinds, return_counts=True)
        np.testing.assert_allclose(counts / n_estimators, .7, atol=.06)
        counts = np.zeros(n)
        for it, tree in enumerate(forest):
            samples_train, samples_val = tree.get_train_test_split_inds()
            np.testing.assert_equal(samples_train, samples_val)
        config = self._get_base_config()
        config['n_estimators'] = n_estimators
        config['max_samples'] = 7
        config['max_depth'] = 1
        X, T, y, _, _ = self._get_causal_data(n, n_features, n_treatments, random_state)
        forest = self._train_causal_forest(X, T, y, config)
        subinds = forest.get_subsample_inds()
        inds, counts = np.unique(subinds, return_counts=True)
        np.testing.assert_allclose(counts / n_estimators, .7, atol=.06)
        config = self._get_base_config()
        config['n_estimators'] = n_estimators
        config['inference'] = True
        config['subforest_size'] = 2
        config['max_samples'] = .4
        config['max_depth'] = 1
        config['honest'] = True
        X, T, y, _, _ = self._get_causal_data(n, n_features, n_treatments, random_state)
        forest = self._train_causal_forest(X, T, y, config)
        subinds = forest.get_subsample_inds()
        inds, counts = np.unique(subinds, return_counts=True)
        np.testing.assert_allclose(counts / n_estimators, .4, atol=.06)
        counts = np.zeros(n)
        for it, tree in enumerate(forest):
            _, samples_val = tree.get_train_test_split_inds()
            inds_val = subinds[it][samples_val]
            counts[inds_val] += 1
        np.testing.assert_allclose(counts / n_estimators, .2, atol=.05)
        return

    def _get_step_regression_data(self, n, n_features, random_state):
        rnd = np.random.RandomState(random_state)
        X = rnd.uniform(-1, 1, size=(n, n_features))
        y = 1.0 * (X[:, 0] >= 0.0).reshape(-1, 1) + rnd.normal(0, 1, size=(n, 1))
        return X, y, y

    def test_var(self,):
        # test that the estimator calcualtes var correctly
        config = self._get_base_config()
        config['honest'] = True
        config['max_depth'] = 0
        config['inference'] = True
        config['n_estimators'] = 1000
        config['subforest_size'] = 2
        config['max_samples'] = .5
        config['n_jobs'] = 1
        n_features = 2
        # test api
        n = 100
        random_state = 123
        X, y, truth = self._get_regression_data(n, n_features, random_state)
        forest = RegressionForest(**config).fit(X, y)
        alpha = .1
        mean, var = forest.predict_and_var(X)
        lb = scipy.stats.norm.ppf(alpha / 2, loc=mean[:, 0], scale=np.sqrt(var[:, 0, 0])).reshape(-1, 1)
        ub = scipy.stats.norm.ppf(1 - alpha / 2, loc=mean[:, 0], scale=np.sqrt(var[:, 0, 0])).reshape(-1, 1)

        np.testing.assert_allclose(var, forest.predict_var(X))
        lbtest, ubtest = forest.predict_interval(X, alpha=alpha)
        np.testing.assert_allclose(lb, lbtest)
        np.testing.assert_allclose(ub, ubtest)
        meantest, lbtest, ubtest = forest.predict(X, interval=True, alpha=alpha)
        np.testing.assert_allclose(mean, meantest)
        np.testing.assert_allclose(lb, lbtest)
        np.testing.assert_allclose(ub, ubtest)
        np.testing.assert_allclose(np.sqrt(var[:, 0, 0]), forest.prediction_stderr(X)[:, 0])

        # test accuracy
        for n in [10, 100, 1000, 10000]:
            random_state = 123
            X, y, truth = self._get_regression_data(n, n_features, random_state)
            forest = RegressionForest(**config).fit(X, y)
            our_mean, our_var = forest.predict_and_var(X[:1])
            true_mean, true_var = np.mean(y), np.var(y) / y.shape[0]
            np.testing.assert_allclose(our_mean, true_mean, atol=0.05)
            np.testing.assert_allclose(our_var, true_var, atol=0.05, rtol=.1)
        for n, our_thr, true_thr in [(1000, .5, .25), (10000, .05, .05)]:
            random_state = 123
            config['max_depth'] = 1
            X, y, truth = self._get_step_regression_data(n, n_features, random_state)
            forest = RegressionForest(**config).fit(X, y)
            posX = X[X[:, 0] > our_thr]
            negX = X[X[:, 0] < -our_thr]
            our_pos_mean, our_pos_var = forest.predict_and_var(posX)
            our_neg_mean, our_neg_var = forest.predict_and_var(negX)
            pos = X[:, 0] > true_thr
            true_pos_mean, true_pos_var = np.mean(y[pos]), np.var(y[pos]) / y[pos].shape[0]
            neg = X[:, 0] < -true_thr
            true_neg_mean, true_neg_var = np.mean(y[neg]), np.var(y[neg]) / y[neg].shape[0]
            np.testing.assert_allclose(our_pos_mean, true_pos_mean, atol=0.07)
            np.testing.assert_allclose(our_pos_var, true_pos_var, atol=0.0, rtol=.25)
            np.testing.assert_allclose(our_neg_mean, true_neg_mean, atol=0.07)
            np.testing.assert_allclose(our_neg_var, true_neg_var, atol=0.0, rtol=.25)
        return

    def test_projection(self,):
        # test the projection functionality of forests
        # test that the estimator calcualtes var correctly
        np.set_printoptions(precision=10, suppress=True)
        config = self._get_base_config()
        config['honest'] = True
        config['max_depth'] = 0
        config['inference'] = True
        config['n_estimators'] = 100
        config['subforest_size'] = 2
        config['max_samples'] = .5
        config['n_jobs'] = 1
        n_features = 2
        # test api
        n = 100
        random_state = 123
        X, y, truth = self._get_regression_data(n, n_features, random_state)
        forest = RegressionForest(**config).fit(X, y)
        mean, var = forest.predict_and_var(X)
        mean = mean.flatten()
        var = var.flatten()
        y = np.hstack([y, y])
        truth = np.hstack([truth, truth])
        forest = RegressionForest(**config).fit(X, y)
        projector = np.ones((X.shape[0], 2)) / 2.0
        mean_proj, var_proj = forest.predict_projection_and_var(X, projector)
        np.testing.assert_array_equal(mean_proj, mean)
        np.testing.assert_array_equal(var_proj, var)
        np.testing.assert_array_equal(var_proj, forest.predict_projection_var(X, projector))
        np.testing.assert_array_equal(mean_proj, forest.predict_projection(X, projector))
        return

    def test_feature_importances(self,):
        # test that the estimator calcualtes var correctly
        for trainer in [self._train_causal_forest, self._train_iv_forest]:
            for criterion in ['het', 'mse']:
                for sample_weight in [None, 'rand']:
                    config = self._get_base_config()
                    config['honest'] = True
                    config['criterion'] = criterion
                    config['fit_intercept'] = True
                    config['max_depth'] = 2
                    config['min_samples_leaf'] = 5
                    config['min_var_fraction_leaf'] = None
                    config['min_impurity_decrease'] = 0.0
                    config['inference'] = True
                    config['n_estimators'] = 4
                    config['subforest_size'] = 2
                    config['max_samples'] = .4
                    config['n_jobs'] = 1

                    n, n_features, n_treatments = 800, 2, 2
                    random_state = 123
                    if sample_weight is not None:
                        sample_weight = check_random_state(random_state).randint(0, 4, size=n)
                    X, T, y, truth, truth_full = self._get_causal_data(n, n_features,
                                                                       n_treatments, random_state)
                    forest = trainer(X, T, y, config, sample_weight=sample_weight)
                    forest_het_importances = np.zeros(n_features)
                    for it, tree in enumerate(forest):
                        tree_ = tree.tree_
                        tfeature = tree_.feature
                        timpurity = tree_.impurity
                        tdepth = tree_.depth
                        tleft = tree_.children_left
                        tright = tree_.children_right
                        tw = tree_.weighted_n_node_samples
                        tvalue = tree_.value

                        for max_depth in [0, 2]:
                            feature_importances = np.zeros(n_features)
                            for it, (feat, impurity, depth, left, right, w) in\
                                    enumerate(zip(tfeature, timpurity, tdepth, tleft, tright, tw)):
                                if (left != -1) and (depth <= max_depth):
                                    gain = w * impurity - tw[left] * timpurity[left] - tw[right] * timpurity[right]
                                    feature_importances[feat] += gain / (depth + 1)**2.0
                            feature_importances /= tw[0]
                            totest = tree.tree_.compute_feature_importances(normalize=False,
                                                                            max_depth=max_depth, depth_decay=2.0)
                            np.testing.assert_array_equal(feature_importances, totest)

                            het_importances = np.zeros(n_features)
                            for it, (feat, depth, left, right, w) in\
                                    enumerate(zip(tfeature, tdepth, tleft, tright, tw)):
                                if (left != -1) and (depth <= max_depth):
                                    gain = tw[left] * tw[right] * np.mean((tvalue[left] - tvalue[right])**2) / w
                                    het_importances[feat] += gain / (depth + 1)**2.0
                            het_importances /= tw[0]
                            totest = tree.tree_.compute_feature_heterogeneity_importances(normalize=False,
                                                                                          max_depth=max_depth,
                                                                                          depth_decay=2.0)
                            np.testing.assert_allclose(het_importances, totest)
                        het_importances /= np.sum(het_importances)
                        forest_het_importances += het_importances / len(forest)

                    np.testing.assert_allclose(forest_het_importances,
                                               forest.feature_importances(max_depth=2, depth_decay_exponent=2.0))
                    np.testing.assert_allclose(forest_het_importances, forest.feature_importances_)
        return

    def test_non_standard_input(self,):
        # test that the estimator accepts lists, tuples and pandas data frames
        n_features = 2
        n = 100
        random_state = 123
        X, y, truth = self._get_regression_data(n, n_features, random_state)
        forest = RegressionForest(n_estimators=20, n_jobs=1, random_state=123).fit(X, y)
        pred = forest.predict(X)
        forest = RegressionForest(n_estimators=20, n_jobs=1, random_state=123).fit(tuple(X), tuple(y))
        np.testing.assert_allclose(pred, forest.predict(tuple(X)))
        forest = RegressionForest(n_estimators=20, n_jobs=1, random_state=123).fit(list(X), list(y))
        np.testing.assert_allclose(pred, forest.predict(list(X)))
        forest = RegressionForest(n_estimators=20, n_jobs=1, random_state=123).fit(pd.DataFrame(X), pd.DataFrame(y))
        np.testing.assert_allclose(pred, forest.predict(pd.DataFrame(X)))
        forest = RegressionForest(n_estimators=20, n_jobs=1, random_state=123).fit(
            pd.DataFrame(X), pd.Series(y.ravel()))
        np.testing.assert_allclose(pred, forest.predict(pd.DataFrame(X)))
        return

    def test_raise_exceptions(self,):
        # test that we raise errors in mishandled situations.
        n_features = 2
        n = 10
        random_state = 123
        X, y, truth = self._get_regression_data(n, n_features, random_state)
        with np.testing.assert_raises(ValueError):
            forest = RegressionForest(n_estimators=20).fit(X, y[:4])
        with np.testing.assert_raises(ValueError):
            forest = RegressionForest(n_estimators=20, subforest_size=3).fit(X, y)
        with np.testing.assert_raises(ValueError):
            forest = RegressionForest(n_estimators=20, inference=True, max_samples=.6).fit(X, y)
        with np.testing.assert_raises(ValueError):
            forest = RegressionForest(n_estimators=20, max_samples=20).fit(X, y)
        with np.testing.assert_raises(ValueError):
            forest = RegressionForest(n_estimators=20, max_samples=1.2).fit(X, y)
        with np.testing.assert_raises(ValueError):
            forest = RegressionForest(n_estimators=4, warm_start=True, inference=True).fit(X, y)
            forest.inference = False
            forest.n_estimators = 8
            forest.fit(X, y)
        with np.testing.assert_raises(KeyError):
            forest = CausalForest(n_estimators=4, criterion='peculiar').fit(X, y, y)
        with np.testing.assert_raises(ValueError):
            forest = CausalForest(n_estimators=4, max_depth=-1).fit(X, y, y)
        with np.testing.assert_raises(ValueError):
            forest = CausalForest(n_estimators=4, min_samples_split=-1).fit(X, y, y)
        with np.testing.assert_raises(ValueError):
            forest = CausalForest(n_estimators=4, min_samples_leaf=-1).fit(X, y, y)
        with np.testing.assert_raises(ValueError):
            forest = CausalForest(n_estimators=4, min_weight_fraction_leaf=-1.0).fit(X, y, y)
        with np.testing.assert_raises(ValueError):
            forest = CausalForest(n_estimators=4, min_var_fraction_leaf=-1.0).fit(X, y, y)
        with np.testing.assert_raises(ValueError):
            forest = CausalForest(n_estimators=4, max_features=10).fit(X, y, y)
        with np.testing.assert_raises(ValueError):
            forest = CausalForest(n_estimators=4, min_balancedness_tol=.55).fit(X, y, y)

        return

    def test_warm_start(self,):
        n_features = 2
        n = 10
        random_state = 123
        X, y, _ = self._get_regression_data(n, n_features, random_state)

        forest = RegressionForest(n_estimators=4, warm_start=True, random_state=123).fit(X, y)
        forest.n_estimators = 8
        forest.fit(X, y)
        pred1 = forest.predict(X)
        inds1 = forest.get_subsample_inds()
        tree_states1 = [t.random_state for t in forest]

        forest = RegressionForest(n_estimators=8, warm_start=True, random_state=123).fit(X, y)
        pred2 = forest.predict(X)
        inds2 = forest.get_subsample_inds()
        tree_states2 = [t.random_state for t in forest]

        np.testing.assert_allclose(pred1, pred2)
        np.testing.assert_allclose(inds1, inds2)
        np.testing.assert_allclose(tree_states1, tree_states2)
        return

    def test_multioutput(self,):
        # test that the subsampling scheme past to the trees is correct
        random_state = 123
        n, n_features, n_treatments = 10, 2, 2
        X, T, y, _, _ = self._get_causal_data(n, n_features, n_treatments, random_state)
        y = np.hstack([y, y])
        for est in [CausalForest(n_estimators=4, random_state=123),
                    CausalIVForest(n_estimators=4, random_state=123)]:
            forest = MultiOutputGRF(est)
            if isinstance(est, CausalForest):
                forest.fit(X, T, y)
            else:
                forest.fit(X, T, y, Z=T)
            pred, lb, ub = forest.predict(X, interval=True, alpha=.05)
            np.testing.assert_array_equal(pred.shape, (X.shape[0], 2, n_treatments))
            np.testing.assert_allclose(pred[:, 0, :], pred[:, 1, :])
            np.testing.assert_allclose(lb[:, 0, :], lb[:, 1, :])
            np.testing.assert_allclose(ub[:, 0, :], ub[:, 1, :])
            pred, var = forest.predict_and_var(X)
            np.testing.assert_array_equal(pred.shape, (X.shape[0], 2, n_treatments))
            np.testing.assert_array_equal(var.shape, (X.shape[0], 2, n_treatments, n_treatments))
            np.testing.assert_allclose(pred[:, 0, :], pred[:, 1, :])
            np.testing.assert_allclose(var[:, 0, :, :], var[:, 1, :, :])
            pred, var = forest.predict_projection_and_var(X, np.ones((X.shape[0], n_treatments)))
            np.testing.assert_array_equal(pred.shape, (X.shape[0], 2))
            np.testing.assert_array_equal(var.shape, (X.shape[0], 2))
            np.testing.assert_allclose(pred[:, 0], pred[:, 1])
            np.testing.assert_allclose(var[:, 0], var[:, 1])
            imps = forest.feature_importances(max_depth=3, depth_decay_exponent=1.0)
            np.testing.assert_array_equal(imps.shape, (X.shape[1], 2))
            np.testing.assert_allclose(imps[:, 0], imps[:, 1])
            imps = forest.feature_importances_
            np.testing.assert_array_equal(imps.shape, (2, X.shape[1]))
            np.testing.assert_allclose(imps[0, :], imps[1, :])

        return
