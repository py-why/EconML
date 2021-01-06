# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import logging
import time
import random
import numpy as np
import sparse as sp
import pytest
from econml.tree import DepthFirstTreeBuilder, BestSplitter, Tree, MSE
from econml.grf import LinearMomentGRFCriterion, LinearMomentGRFCriterionMSE
from econml.grf._utils import matinv, lstsq, pinv, fast_max_eigv, fast_min_eigv
from econml.utilities import cross_product


class TestGRFCython(unittest.TestCase):

    def _get_base_config(self, n_features=2, n_t=2, n_samples_train=1000):
        n_y = 1
        return {'criterion': 'het',
                'n_features': n_features,
                'n_y': n_y,
                'n_outputs': n_t + 1,
                'n_relevant_outputs': n_t,
                'store_jac': True,
                'n_samples': n_samples_train,
                'n_samples_train': n_samples_train,
                'max_features': n_features,
                'min_samples_split': 2,
                'min_samples_leaf': 10,
                'min_weight_leaf': 1,
                'min_eig_leaf': -1,
                'min_eig_leaf_on_val': False,
                'min_balancedness_tol': .3,
                'max_depth': 2,
                'min_impurity_decrease': 0.0,
                'honest': False,
                'random_state': 1234,
                'max_node_samples': n_samples_train,
                'samples_train': np.arange(n_samples_train, dtype=np.intp),
                'samples_val': np.arange(n_samples_train, dtype=np.intp)
                }

    def _get_base_honest_config(self, n_features=2, n_t=2, n_samples_train=1000):
        n_y = 1
        return {'criterion': 'het',
                'n_features': n_features,
                'n_y': n_y,
                'n_outputs': n_t + 1,
                'n_relevant_outputs': n_t,
                'store_jac': True,
                'n_samples': 2 * n_samples_train,
                'n_samples_train': n_samples_train,
                'max_features': n_features,
                'min_samples_split': 2,
                'min_samples_leaf': 10,
                'min_weight_leaf': 1,
                'min_eig_leaf': -1,
                'min_eig_leaf_on_val': False,
                'min_balancedness_tol': .3,
                'max_depth': 2,
                'min_impurity_decrease': 0.0,
                'honest': True,
                'random_state': 1234,
                'max_node_samples': n_samples_train,
                'samples_train': np.arange(n_samples_train, dtype=np.intp),
                'samples_val': np.arange(n_samples_train, 2 * n_samples_train, dtype=np.intp)
                }

    def _get_cython_objects(self, *, criterion, n_features, n_y, n_outputs, n_relevant_outputs,
                            store_jac, n_samples, n_samples_train, max_features,
                            min_samples_split, min_samples_leaf, min_weight_leaf,
                            min_eig_leaf, min_eig_leaf_on_val, min_balancedness_tol, max_depth, min_impurity_decrease,
                            honest, random_state, max_node_samples, samples_train,
                            samples_val):
        tree = Tree(n_features, n_outputs, n_relevant_outputs, store_jac)
        if criterion == 'het':
            criterion = LinearMomentGRFCriterion(n_outputs, n_relevant_outputs, n_features, n_y,
                                                 n_samples, max_node_samples, random_state)
            criterion_val = LinearMomentGRFCriterion(n_outputs, n_relevant_outputs, n_features, n_y,
                                                     n_samples, max_node_samples, random_state)
        else:
            criterion = LinearMomentGRFCriterionMSE(n_outputs, n_relevant_outputs, n_features, n_y,
                                                    n_samples, max_node_samples, random_state)
            criterion_val = LinearMomentGRFCriterionMSE(n_outputs, n_relevant_outputs, n_features, n_y,
                                                        n_samples, max_node_samples, random_state)
        splitter = BestSplitter(criterion, criterion_val,
                                max_features, min_samples_leaf, min_weight_leaf,
                                min_balancedness_tol, honest, min_eig_leaf, min_eig_leaf_on_val, random_state)
        builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                        min_samples_leaf, min_weight_leaf,
                                        max_depth, min_impurity_decrease)
        return tree, criterion, criterion_val, splitter, builder

    def _get_continuous_data(self, config):
        random_state = np.random.RandomState(config['random_state'])
        X = random_state.normal(size=(config['n_samples_train'], config['n_features']))
        T = np.zeros((config['n_samples_train'], config['n_relevant_outputs']))
        for t in range(T.shape[1]):
            T[:, t] = random_state.binomial(1, .5, size=(T.shape[0],))
        Taug = np.hstack([T, np.ones((T.shape[0], 1))])
        y = ((X[:, [0]] > 0.0) + .5) * np.sum(T, axis=1, keepdims=True) + .5
        yaug = np.hstack([y, y * Taug, cross_product(Taug, Taug)])
        X = np.vstack([X, X])
        yaug = np.vstack([yaug, yaug])
        return X, yaug, np.hstack([(X[:, [0]] > 0.0) + .5, (X[:, [0]] > 0.0) + .5])

    def _train_tree(self, config, X, y):
        tree, criterion, criterion_val, splitter, builder = self._get_cython_objects(**config)
        builder.build(tree, X, y,
                      config['samples_train'],
                      config['samples_val'],
                      store_jac=config['store_jac'])
        return tree

    def _get_true_quantities(self, config, X, y, mask, criterion):
        alpha = y[mask, config['n_y']:config['n_y'] + config['n_outputs']]
        pointJ = y[mask, config['n_y'] + config['n_outputs']:
                   config['n_y'] + (config['n_outputs'] + 1) * config['n_outputs']]
        jac = np.mean(pointJ, axis=0)
        precond = np.mean(alpha, axis=0)
        invJ = np.linalg.inv(jac.reshape((alpha.shape[1], alpha.shape[1])))
        param = invJ @ precond
        moment = alpha - pointJ.reshape((-1, alpha.shape[1], alpha.shape[1])) @ param
        rho = ((invJ @ moment.T).T)[:, :config['n_relevant_outputs']]
        if criterion == 'het':
            impurity = np.mean(rho**2) - np.mean(np.mean(rho, axis=0)**2)
        else:
            impurity = np.mean(y[mask, :config['n_y']]**2)
            impurity -= (param.reshape(1, -1) @ jac.reshape((alpha.shape[1], alpha.shape[1])) @ param)[0]
        return jac, precond, param, impurity

    def _get_node_quantities(self, tree, node_id):
        return (tree.jac[node_id, :], tree.precond[node_id, :],
                tree.full_value[node_id, :, 0], tree.impurity[node_id])

    def _test_tree_quantities(self, base_config_gen, criterion):
        config = base_config_gen()
        config['criterion'] = criterion
        config['max_depth'] = 1
        X, y, truth = self._get_continuous_data(config)
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(X[:config['n_samples_train']], X[config['n_samples_train']:])
        np.testing.assert_array_equal(y[:config['n_samples_train']], y[config['n_samples_train']:])
        np.testing.assert_array_equal(config['samples_train'], np.arange(config['n_samples_train']))
        if config['honest']:
            np.testing.assert_array_equal(config['samples_val'],
                                          np.arange(config['n_samples_train'], 2 * config['n_samples_train']))
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2]))
        np.testing.assert_allclose(tree.threshold, np.array([0, -2, -2]), atol=.1, rtol=0)
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, np.ones(X.shape[0]) > 0, criterion),
                         self._get_node_quantities(tree, 0))]
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y,
                                                   X[:, tree.feature[0]] < tree.threshold[0], criterion),
                         self._get_node_quantities(tree, 1))]
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y,
                                                   X[:, tree.feature[0]] >= tree.threshold[0], criterion),
                         self._get_node_quantities(tree, 2))]

        mask = np.abs(X[:, 0]) > .05
        np.testing.assert_allclose(tree.predict(X[mask]), truth[mask], atol=.05)

        config = base_config_gen()
        config['criterion'] = criterion
        config['max_depth'] = 2
        X, y, truth = self._get_continuous_data(config)
        tree = self._train_tree(config, X, y)
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, np.ones(X.shape[0]) > 0, criterion),
                         self._get_node_quantities(tree, 0))]
        mask0 = X[:, tree.feature[0]] < tree.threshold[0]
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, mask0, criterion),
                         self._get_node_quantities(tree, 1))]
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, ~mask0, criterion),
                         self._get_node_quantities(tree, 4))]
        mask1a = mask0 & (X[:, tree.feature[1]] < tree.threshold[1])
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, mask1a, criterion),
                         self._get_node_quantities(tree, 2))]
        mask1b = mask0 & (X[:, tree.feature[1]] >= tree.threshold[1])
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, mask1b, criterion),
                         self._get_node_quantities(tree, 3))]
        mask1c = (~mask0) & (X[:, tree.feature[4]] < tree.threshold[4])
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, mask1c, criterion),
                         self._get_node_quantities(tree, 5))]
        mask1d = (~mask0) & (X[:, tree.feature[4]] >= tree.threshold[4])
        [np.testing.assert_allclose(a, b, atol=1e-4)
         for a, b in zip(self._get_true_quantities(config, X, y, mask1d, criterion),
                         self._get_node_quantities(tree, 6))]

        mask = np.abs(X[:, 0]) > .05
        np.testing.assert_allclose(tree.predict(X[mask]), truth[mask], atol=.05)

    def test_dishonest_tree(self):
        self._test_tree_quantities(self._get_base_config, criterion='het')
        self._test_tree_quantities(self._get_base_config, criterion='mse')

    def test_honest_tree(self):
        self._test_tree_quantities(self._get_base_honest_config, criterion='het')
        self._test_tree_quantities(self._get_base_honest_config, criterion='mse')

    def test_honest_dishonest_equivalency(self):
        for criterion in ['het', 'mse']:
            config = self._get_base_config()
            config['criterion'] = criterion
            config['max_depth'] = 4
            X, y, _ = self._get_continuous_data(config)
            tree = self._train_tree(config, X, y)
            config = self._get_base_honest_config()
            config['criterion'] = criterion
            config['max_depth'] = 4
            X, y, _ = self._get_continuous_data(config)
            honest_tree = self._train_tree(config, X, y)
            np.testing.assert_equal(tree.feature, honest_tree.feature)
            np.testing.assert_equal(tree.threshold, honest_tree.threshold)
            np.testing.assert_equal(tree.value, honest_tree.value)
            np.testing.assert_equal(tree.full_value, honest_tree.full_value)
            np.testing.assert_equal(tree.impurity, honest_tree.impurity)
            np.testing.assert_equal(tree.impurity, honest_tree.impurity_train)
            np.testing.assert_equal(tree.n_node_samples, honest_tree.n_node_samples)
            np.testing.assert_equal(tree.weighted_n_node_samples, honest_tree.weighted_n_node_samples_train)
            np.testing.assert_equal(tree.n_node_samples, honest_tree.n_node_samples_train)
            np.testing.assert_equal(tree.jac, honest_tree.jac)
            np.testing.assert_equal(tree.precond, honest_tree.precond)
            np.testing.assert_equal(tree.predict(X), honest_tree.predict(X))
            np.testing.assert_equal(tree.predict_full(X), honest_tree.predict_full(X))
            np.testing.assert_equal(tree.compute_feature_importances(), honest_tree.compute_feature_importances())
            np.testing.assert_equal(tree.compute_feature_heterogeneity_importances(),
                                    honest_tree.compute_feature_heterogeneity_importances())

    def test_min_var_leaf(self):
        n_samples_train = 10
        for criterion in ['het', 'mse']:
            config = self._get_base_config(n_samples_train=n_samples_train, n_t=1, n_features=1)
            config['max_depth'] = 1
            config['min_samples_leaf'] = 1
            config['min_eig_leaf'] = .2
            config['criterion'] = criterion
            X = np.arange(n_samples_train).reshape(-1, 1)
            T = np.random.binomial(1, .5, size=(n_samples_train, 1))
            T[X[:, 0] < n_samples_train // 2] = 0
            T[X[:, 0] >= n_samples_train // 2] = 1
            Taug = np.hstack([T, np.ones((T.shape[0], 1))])
            y = np.zeros((n_samples_train, 1))
            yaug = np.hstack([y, y * Taug, cross_product(Taug, Taug)])
            tree = self._train_tree(config, X, yaug)
            if criterion == 'het':
                np.testing.assert_array_less(config['min_eig_leaf'], np.mean(T[X[:, 0] > tree.threshold[0]]**2))
                np.testing.assert_array_less(config['min_eig_leaf'], np.mean(T[X[:, 0] <= tree.threshold[0]]**2))
            else:
                np.testing.assert_array_equal(tree.feature, np.array([-2]))

    def test_fast_eigv(self):
        n = 4
        np.random.seed(123)
        for _ in range(10):
            A = np.random.normal(0, 1, size=(n, n))
            A = np.asfortranarray(A @ A.T)
            apx = fast_min_eigv(A, 5, 123)
            opt = np.min(np.linalg.eig(A)[0])
            np.testing.assert_allclose(apx, opt, atol=.01, rtol=.3)
            apx = fast_max_eigv(A, 10, 123)
            opt = np.max(np.linalg.eig(A)[0])
            np.testing.assert_allclose(apx, opt, atol=.5, rtol=.2)

    def test_linalg(self):
        np.random.seed(1235)
        for n, m, nrhs in [(3, 3, 3), (3, 2, 1), (3, 1, 2), (1, 4, 2), (3, 4, 5)]:
            for _ in range(100):
                A = np.random.normal(0, 1, size=(n, m))
                y = np.random.normal(0, 1, size=(n, nrhs))
                yf = y
                if m > n:
                    yf = np.zeros((m, nrhs))
                    yf[:n] = y
                ours = np.asfortranarray(np.zeros((m, nrhs)))
                lstsq(np.asfortranarray(A), np.asfortranarray(yf.copy()), ours, copy_b=True)
                true = np.linalg.lstsq(A, y, rcond=np.finfo(np.float64).eps * max(n, m))[0]
                np.testing.assert_allclose(ours, true, atol=.00001, rtol=.0)

                ours = np.asfortranarray(np.zeros(A.T.shape, dtype=np.float64))
                pinv(np.asfortranarray(A), ours)
                true = np.linalg.pinv(A)
                np.testing.assert_allclose(ours, true, atol=.00001, rtol=.0)

                if n == m:
                    ours = np.asfortranarray(np.zeros(A.T.shape, dtype=np.float64))
                    matinv(np.asfortranarray(A), ours)
                    true = np.linalg.inv(A)
                    np.testing.assert_allclose(ours, true, atol=.00001, rtol=.0)
