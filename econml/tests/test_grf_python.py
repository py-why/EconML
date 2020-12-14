
import unittest
import logging
import time
import random
import numpy as np
import pytest
from econml.grf import RegressionForest, CausalForest, CausalIVForest
from econml.utilities import cross_product
from copy import deepcopy
from sklearn.utils import check_random_state


class TestGRFPython(unittest.TestCase):

    def _get_base_config(self):
        return {'n_estimators': 1, 'subforest_size': 1, 'max_depth': 2,
                'min_samples_split': 2, 'min_samples_leaf': 1,
                'inference': False, 'max_samples': 1.0, 'honest': False,
                'random_state': 123}

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
        np.testing.assert_array_almost_equal(tree.compute_feature_heterogeneity_importances(normalize=False, max_depth=0),
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
        jac = np.average(pointJ, axis=0, weights=sample_weight) + 1e-6 * np.eye(T.shape[1]).flatten()
        precond = np.average(alpha, axis=0, weights=sample_weight)

        if jac.shape[0] == 4:
            det = jac[0] * jac[3] - jac[1] * jac[2]
            if abs(det) < 1e-6:
                det = 1e-6
            invJ = np.array([[jac[3], -jac[1]], [-jac[2], jac[0]]]) / det
        else:
            invJ = np.linalg.inv(jac.reshape((alpha.shape[1], alpha.shape[1])))

        param = invJ @ precond
        if criterion == 'het':
            moment = alpha - pointJ.reshape((-1, alpha.shape[1], alpha.shape[1])) @ param
            rho = ((invJ @ moment.T).T)[:, :n_relevant_outputs]
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
        np.set_printoptions(precision=2, suppress=True)
        config = self._get_base_config()
        for criterion in ['het', 'mse']:
            for fit_intercept in [False, True]:
                for min_var_leaf in [None, .1]:
                    config['criterion'] = criterion
                    config['fit_intercept'] = fit_intercept
                    config['max_depth'] = 2
                    config['min_samples_leaf'] = 5
                    config['min_var_leaf'] = min_var_leaf
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
                    if fit_intercept and (min_var_leaf is not None):
                        mask = np.abs(X[:, 0]) > .3
                        np.testing.assert_allclose(tree.predict(X[mask]), truth[mask], atol=.05)
                        np.testing.assert_allclose(tree.predict_full(X[mask]), truth_full[mask], atol=.05)

    def _test_causal_honesty(self, trainer):
        np.set_printoptions(precision=2, suppress=True)
        for criterion in ['mse']:
            for fit_intercept in [False, True]:
                for min_var_leaf in [None, .1]:
                    for min_impurity_decrease in [0.0, 0.07]:
                        for inference in [False, True]:
                            for sample_weight in [None, 'rand']:
                                config = self._get_base_config()
                                config['honest'] = True
                                config['criterion'] = criterion
                                config['fit_intercept'] = fit_intercept
                                config['max_depth'] = 2
                                config['min_samples_leaf'] = 5
                                config['min_var_leaf'] = min_var_leaf
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
                                if (sample_weight is None) and fit_intercept and (min_var_leaf is not None):
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
                                    if (sample_weight is None) and fit_intercept and (min_var_leaf is not None):
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

    def test_var(self,):
        # test that the estimator calcualtes var correctly
        return

    def test_projection(self,):
        # test the projection functionality of forests
        return

    def test_feature_importances(self,):
        # test that the estimator calcualtes var correctly
        return

    def test_non_standard_input(self,):
        # test that the estimator accepts lists, tuples and pandas data frames
        return

    def test_raise_exceptions(self,):
        # test that we raise errors in mishandled situations.
        return


if __name__ == "__main__":
    TestGRFPython().test_causal_tree()
    TestGRFPython().test_iv_tree()
    TestGRFPython().test_regression_tree_internals()
    TestGRFPython().test_subsampling()
