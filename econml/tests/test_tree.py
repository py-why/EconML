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


class TestTree(unittest.TestCase):

    def _get_base_config(self):
        n_features = 2
        n_samples_train = 10
        n_y = 1
        return {'n_features': n_features,
                'n_y': n_y,
                'n_outputs': n_y,
                'n_relevant_outputs': n_y,
                'store_jac': False,
                'n_samples': n_samples_train,
                'n_samples_train': n_samples_train,
                'max_features': n_features,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'min_weight_leaf': 1,
                'min_eig_leaf': -1,
                'min_eig_leaf_on_val': False,
                'min_balancedness_tol': .3,
                'max_depth': 2,
                'min_impurity_decrease': 0.0,
                'honest': False,
                'random_state': 123,
                'max_node_samples': n_samples_train,
                'samples_train': np.arange(n_samples_train, dtype=np.intp),
                'samples_val': np.arange(n_samples_train, dtype=np.intp)
                }

    def _get_base_honest_config(self):
        n_features = 2
        n_samples_train = 10
        n_y = 1
        return {'n_features': n_features,
                'n_y': n_y,
                'n_outputs': n_y,
                'n_relevant_outputs': n_y,
                'store_jac': False,
                'n_samples': 2 * n_samples_train,
                'n_samples_train': n_samples_train,
                'max_features': n_features,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'min_weight_leaf': 1,
                'min_eig_leaf': -1,
                'min_eig_leaf_on_val': False,
                'min_balancedness_tol': .3,
                'max_depth': 2,
                'min_impurity_decrease': 0.0,
                'honest': True,
                'random_state': 123,
                'max_node_samples': n_samples_train,
                'samples_train': np.arange(n_samples_train, dtype=np.intp),
                'samples_val': np.arange(n_samples_train, 2 * n_samples_train, dtype=np.intp)
                }

    def _get_cython_objects(self, *, n_features, n_y, n_outputs, n_relevant_outputs,
                            store_jac, n_samples, n_samples_train, max_features,
                            min_samples_split, min_samples_leaf, min_weight_leaf,
                            min_eig_leaf, min_eig_leaf_on_val, min_balancedness_tol, max_depth, min_impurity_decrease,
                            honest, random_state, max_node_samples, samples_train,
                            samples_val):
        tree = Tree(n_features, n_outputs, n_relevant_outputs, store_jac)
        criterion = MSE(n_outputs, n_relevant_outputs, n_features, n_y,
                        n_samples, max_node_samples, random_state)
        criterion_val = MSE(n_outputs, n_relevant_outputs, n_features, n_y,
                            n_samples, max_node_samples, random_state)
        splitter = BestSplitter(criterion, criterion_val,
                                max_features, min_samples_leaf, min_weight_leaf,
                                min_balancedness_tol, honest, min_eig_leaf, min_eig_leaf_on_val, random_state)
        builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                        min_samples_leaf, min_weight_leaf,
                                        max_depth, min_impurity_decrease)
        return tree, criterion, criterion_val, splitter, builder

    def _get_continuous_data(self, config):
        X = np.zeros((config['n_samples_train'], config['n_features']))
        X[:, 0] = np.arange(X.shape[0])
        X[:, 1] = np.random.RandomState(config['random_state']).normal(0, 1, size=(X.shape[0]))
        y = 1.0 * (X[:, 0] >= config['n_samples_train'] / 2).reshape(-1, 1)
        y += 1.0 * (X[:, 0] >= config['n_samples_train'] / 4).reshape(-1, 1)
        y += 1.0 * (X[:, 0] >= 3 * config['n_samples_train'] / 4).reshape(-1, 1)
        X = np.vstack([X, X])
        y = np.vstack([y, y])
        return X, y

    def _get_binary_data(self, config):
        n_samples_train = config['n_samples_train']
        X = np.zeros((n_samples_train, config['n_features']))
        X[:n_samples_train // 2, 0] = 1
        X[:n_samples_train // 4, 1] = 1
        X[3 * n_samples_train // 4:, 1] = 1
        y = 1.0 * (X[:, 0] + X[:, 1]).reshape(-1, 1)
        X = np.vstack([X, X])
        y = np.vstack([y, y])
        return X, y

    def _train_tree(self, config, X, y):
        tree, criterion, criterion_val, splitter, builder = self._get_cython_objects(**config)
        builder.build(tree, X, y,
                      config['samples_train'],
                      config['samples_val'],
                      store_jac=config['store_jac'])
        return tree

    def _test_tree_continuous(self, base_config_gen):
        config = base_config_gen()
        X, y = self._get_continuous_data(config)
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, 0, -2, -2, 0, -2, -2]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, 2.5, - 2, -2, 7.5, -2, -2]))
        np.testing.assert_array_equal(tree.value.flatten()[:3],
                                      np.array([np.mean(y),
                                                np.mean(y[X[:, tree.feature[0]] < tree.threshold[0]]),
                                                np.mean(y[(X[:, tree.feature[0]] < tree.threshold[0]) &
                                                          (X[:, tree.feature[1]] < tree.threshold[1])])]))
        np.testing.assert_array_almost_equal(tree.predict(X), y, decimal=10)
        with np.testing.assert_raises(AttributeError):
            tree.predict_precond(X)
        with np.testing.assert_raises(AttributeError):
            tree.predict_jac(X)
        with np.testing.assert_raises(AttributeError):
            tree.predict_precond_and_jac(X)

        less = X[:, tree.feature[0]] < tree.threshold[0]

        # testing importances
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = np.var(y)
        np.testing.assert_array_almost_equal(tree.compute_feature_importances(normalize=False),
                                             feature_importances, decimal=10)
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = np.var(y) - np.var(y[less])
        np.testing.assert_array_almost_equal(tree.compute_feature_importances(normalize=False, max_depth=0),
                                             feature_importances, decimal=10)
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = np.var(y) - np.var(y[less]) + .5 * (np.var(y[less]))
        np.testing.assert_array_almost_equal(tree.compute_feature_importances(normalize=False,
                                                                              max_depth=1, depth_decay=1.0),
                                             feature_importances, decimal=10)
        # testing heterogeneity importances
        feature_importances = np.zeros(X.shape[1])
        feature_importances[0] = 5 * 5 * (np.mean(y[less]) - np.mean(y[~less]))**2 / 100
        np.testing.assert_array_almost_equal(tree.compute_feature_heterogeneity_importances(normalize=False,
                                                                                            max_depth=0),
                                             feature_importances, decimal=10)
        feature_importances[0] += .5 * (2 * 2 * 3 * (1)**2 / 5) / 10
        np.testing.assert_array_almost_equal(tree.compute_feature_heterogeneity_importances(normalize=False,
                                                                                            max_depth=1,
                                                                                            depth_decay=1.0),
                                             feature_importances, decimal=10)
        feature_importances[0] += .5 * (2 * 2 * 3 * (1)**2 / 5) / 10
        np.testing.assert_array_almost_equal(tree.compute_feature_heterogeneity_importances(normalize=False),
                                             feature_importances, decimal=10)

        # Testing that all parameters do what they are supposed to
        config = base_config_gen()
        config['min_samples_leaf'] = 5
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = base_config_gen()
        config['min_samples_split'] = 11
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([-2]))
        np.testing.assert_array_equal(tree.threshold, np.array([-2]))
        np.testing.assert_array_almost_equal(tree.predict(X), np.mean(y), decimal=10)
        np.testing.assert_array_almost_equal(tree.predict_full(X), np.mean(y), decimal=10)

        config = base_config_gen()
        config['min_weight_leaf'] = 5
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))
        # testing predict, apply and decision path
        less = X[:, tree.feature[0]] < tree.threshold[0]
        y_pred = np.zeros((X.shape[0], 1))
        y_pred[less] = np.mean(y[less])
        y_pred[~less] = np.mean(y[~less])
        np.testing.assert_array_almost_equal(tree.predict(X), y_pred, decimal=10)
        np.testing.assert_array_almost_equal(tree.predict_full(X), y_pred, decimal=10)
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

        config = base_config_gen()
        config['min_balancedness_tol'] = .0
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = base_config_gen()
        config['min_balancedness_tol'] = .1
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, 0, -2, -2, 0, -2, -2]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, 2.5, - 2, -2, 7.5, -2, -2]))

        config = base_config_gen()
        config['max_depth'] = 1
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = base_config_gen()
        config['min_impurity_decrease'] = .99999
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, -2, -2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([4.5, -2, -2]))

        config = base_config_gen()
        config['min_impurity_decrease'] = 1.00001
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([-2, ]))
        np.testing.assert_array_equal(tree.threshold, np.array([-2, ]))

    def test_dishonest_tree(self):
        self._test_tree_continuous(self._get_base_config)

    def test_honest_tree(self):
        self._test_tree_continuous(self._get_base_honest_config)

    def test_multivariable_split(self):
        config = self._get_base_config()
        X, y = self._get_binary_data(config)
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, 1, -2, -2, 1, -2, -2]))
        np.testing.assert_array_equal(tree.threshold, np.array([0.5, 0.5, - 2, -2, 0.5, -2, -2]))

    def test_honest_values(self):
        config = self._get_base_honest_config()
        X, y = self._get_binary_data(config)
        y[config['n_samples_train']:] = .4
        tree = self._train_tree(config, X, y)
        np.testing.assert_array_equal(tree.feature, np.array([0, 1, -2, -2, 1, -2, -2]))
        np.testing.assert_array_equal(tree.threshold, np.array([0.5, 0.5, - 2, -2, 0.5, -2, -2]))
        np.testing.assert_array_almost_equal(tree.value.flatten(), .4 * np.ones(len(tree.value)))

    def test_noisy_instance(self):
        n_samples = 5000
        X = np.random.normal(0, 1, size=(n_samples, 1))
        y_base = 1.0 * X[:, [0]] * (X[:, [0]] > 0)
        y = y_base + np.random.normal(0, .1, size=(n_samples, 1))
        config = self._get_base_config()
        config['n_features'] = 1
        config['max_features'] = 1
        config['max_depth'] = 10
        config['min_samples_leaf'] = 20
        config['n_samples'] = X.shape[0]
        config['min_balancedness_tol'] = .5
        config['n_samples_train'] = X.shape[0]
        config['max_node_samples'] = X.shape[0]
        config['samples_train'] = np.arange(X.shape[0], dtype=np.intp)
        config['samples_val'] = np.arange(X.shape[0], dtype=np.intp)
        tree = self._train_tree(config, X, y)
        X_test = np.zeros((100, 1))
        X_test[:, 0] = np.linspace(np.percentile(X, 10), np.percentile(X, 90), 100)
        y_test = 1.0 * X_test[:, [0]] * (X_test[:, [0]] > 0)
        np.testing.assert_array_almost_equal(tree.predict(X_test), y_test, decimal=1)
        config = self._get_base_honest_config()
        config['n_features'] = 1
        config['max_features'] = 1
        config['max_depth'] = 10
        config['min_samples_leaf'] = 20
        config['n_samples'] = X.shape[0]
        config['min_balancedness_tol'] = .5
        config['n_samples_train'] = X.shape[0] // 2
        config['max_node_samples'] = X.shape[0] // 2
        config['samples_train'] = np.arange(X.shape[0] // 2, dtype=np.intp)
        config['samples_val'] = np.arange(X.shape[0] // 2, X.shape[0], dtype=np.intp)
        tree = self._train_tree(config, X, y)
        X_test = np.zeros((100, 1))
        X_test[:, 0] = np.linspace(np.percentile(X, 10), np.percentile(X, 90), 100)
        y_test = 1.0 * X_test[:, [0]] * (X_test[:, [0]] > 0)
        np.testing.assert_array_almost_equal(tree.predict(X_test), y_test, decimal=1)
