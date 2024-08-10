# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np
import pandas as pd
import pytest
import joblib
from econml.policy import PolicyTree, PolicyForest
from econml.policy import DRPolicyTree, DRPolicyForest
from sklearn.utils import check_random_state
from sklearn.preprocessing import PolynomialFeatures
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import GroupKFold


graphviz_works = True
try:
    from graphviz import Graph
    g = Graph()
    g.render()
except Exception:
    graphviz_works = False


class TestPolicyForest(unittest.TestCase):

    def _get_base_config(self):
        return {'n_estimators': 1, 'max_depth': 2,
                'min_samples_split': 2, 'min_samples_leaf': 1,
                'max_samples': 1.0, 'honest': False,
                'n_jobs': None, 'random_state': 123}

    def _get_policy_data(self, n, n_features, random_state, n_outcomes=2):
        random_state = np.random.RandomState(random_state)
        X = random_state.normal(size=(n, n_features))
        if n_outcomes == 1:
            y = (X[:, 0] > 0.0) - .5
        else:
            y = np.hstack([np.zeros((X.shape[0], 1)), (X[:, [0]] > 0.0) - .5])
        return (X, y, y)

    def _get_true_quantities(self, X, y, mask, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        X, y, sample_weight = X[mask], y[mask], sample_weight[mask]
        node_value = np.average(y, axis=0, weights=sample_weight)
        impurity = - np.max(node_value)
        return node_value, impurity

    def _get_node_quantities(self, tree, node_id):
        return (tree.full_value[node_id, :, 0], tree.impurity[node_id])

    def _train_policy_forest(self, X, y, config, sample_weight=None):
        return PolicyForest(**config).fit(X, y, sample_weight=sample_weight)

    def _train_dr_policy_forest(self, X, y, config, sample_weight=None):
        config.pop('criterion')
        if sample_weight is not None:
            sample_weight = np.repeat(sample_weight, 2)
        groups = np.repeat(np.arange(X.shape[0]), 2)
        X = np.repeat(X, 2, axis=0)
        T = np.zeros(y.shape)
        T[:, 1] = 1
        T = T.flatten()
        y = y.flatten()
        return DRPolicyForest(model_regression=DummyRegressor(strategy='constant', constant=0),
                              model_propensity=DummyClassifier(strategy='uniform'),
                              featurizer=PolynomialFeatures(degree=1, include_bias=False),
                              cv=GroupKFold(n_splits=2),
                              **config).fit(y, T, X=X,
                                            sample_weight=sample_weight,
                                            groups=groups).policy_model_

    def _train_dr_policy_tree(self, X, y, config, sample_weight=None):
        config.pop('n_estimators')
        config.pop('max_samples')
        config.pop('n_jobs')
        config.pop('criterion')
        if sample_weight is not None:
            sample_weight = np.repeat(sample_weight, 2)
        groups = np.repeat(np.arange(X.shape[0]), 2)
        X = np.repeat(X, 2, axis=0)
        T = np.zeros(y.shape)
        T[:, 1] = 1
        T = T.flatten()
        y = y.flatten()
        return [DRPolicyTree(model_regression=DummyRegressor(strategy='constant', constant=0),
                             model_propensity=DummyClassifier(strategy='uniform'),
                             featurizer=PolynomialFeatures(degree=1, include_bias=False),
                             cv=GroupKFold(n_splits=2),
                             **config).fit(y, T, X=X, sample_weight=sample_weight,
                                           groups=groups).policy_model_]

    def _test_policy_tree_internals(self, trainer):
        config = self._get_base_config()
        for criterion in ['neg_welfare']:
            config['criterion'] = criterion
            config['max_depth'] = 2
            config['min_samples_leaf'] = 5
            n, n_features = 100, 2
            random_state = 123
            X, y, _ = self._get_policy_data(n, n_features, random_state)
            forest = trainer(X, y, config)
            tree = forest[0].tree_
            paths = np.array(forest[0].decision_path(X).todense())
            for node_id in range(len(tree.feature)):
                mask = paths[:, node_id] > 0
                [np.testing.assert_allclose(a, b, atol=1e-4)
                    for a, b in zip(self._get_true_quantities(X, y, mask),
                                    self._get_node_quantities(tree, node_id))]

    def _test_policy_honesty(self, trainer, dr=False):
        n_outcome_list = [1, 2] if not dr else [2]
        for criterion in ['neg_welfare']:
            for min_impurity_decrease in [0.0, 0.07]:
                for sample_weight in [None, 'rand']:
                    for n_outcomes in n_outcome_list:
                        config = self._get_base_config()
                        config['honest'] = not dr
                        config['criterion'] = criterion
                        config['max_depth'] = 2
                        config['min_samples_leaf'] = 5
                        config['min_impurity_decrease'] = min_impurity_decrease
                        n, n_features = 800, 2
                        config['n_estimators'] = 4
                        config['max_samples'] = .4 if not dr else 1.0
                        config['n_jobs'] = 1
                        random_state = 123
                        if sample_weight is not None:
                            sample_weight = check_random_state(random_state).randint(0, 4, size=n)
                        X, y, truth = self._get_policy_data(n, n_features, random_state, n_outcomes)
                        forest = trainer(X, y, config, sample_weight=sample_weight)
                        subinds = forest.get_subsample_inds()
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
                            if dr:
                                inds_val = inds_val[np.array(inds_val) % 2 == 0] // 2
                            Xval, yval, _ = X[inds_val], y[inds_val], truth[inds_val]
                            sample_weightval = sample_weight[inds_val] if sample_weight is not None else None
                            paths = np.array(tree.decision_path(Xval).todense())
                            for node_id in range(len(tree.tree_.feature)):
                                mask = paths[:, node_id] > 0
                                [np.testing.assert_allclose(a, b, atol=1e-4)
                                    for a, b in zip(self._get_true_quantities(Xval, yval, mask,
                                                                              sample_weight=sample_weightval),
                                                    self._get_node_quantities(tree.tree_, node_id))]
                            if (sample_weight is None) and min_impurity_decrease > 0.0005:
                                assert np.all((tree.tree_.feature == 0) | (tree.tree_.feature == -2))

    def test_policy_tree(self,):
        self._test_policy_tree_internals(self._train_policy_forest)
        self._test_policy_honesty(self._train_policy_forest)

    def test_drpolicy_tree(self,):
        self._test_policy_tree_internals(self._train_dr_policy_tree)

    def test_drpolicy_forest(self,):
        self._test_policy_tree_internals(self._train_dr_policy_forest)
        self._test_policy_honesty(self._train_dr_policy_forest, dr=True)

    def test_subsampling(self,):
        # test that the subsampling scheme past to the trees is correct
        random_state = 123
        n, n_features = 10, 2
        n_estimators = 600
        config = self._get_base_config()
        config['n_estimators'] = n_estimators
        config['max_samples'] = .7
        config['max_depth'] = 1
        X, y, _ = self._get_policy_data(n, n_features, random_state)
        forest = self._train_policy_forest(X, y, config)
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
        X, y, _ = self._get_policy_data(n, n_features, random_state)
        forest = self._train_policy_forest(X, y, config)
        subinds = forest.get_subsample_inds()
        inds, counts = np.unique(subinds, return_counts=True)
        np.testing.assert_allclose(counts / n_estimators, .7, atol=.06)
        config = self._get_base_config()
        config['n_estimators'] = n_estimators
        config['max_samples'] = .4
        config['max_depth'] = 1
        config['honest'] = True
        X, y, _ = self._get_policy_data(n, n_features, random_state)
        forest = self._train_policy_forest(X, y, config)
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

    def test_feature_importances(self,):
        # test that the estimator calcualtes var correctly
        for trainer in [self._train_policy_forest]:
            for criterion in ['neg_welfare']:
                for sample_weight in [None, 'rand']:
                    config = self._get_base_config()
                    config['honest'] = True
                    config['criterion'] = criterion
                    config['max_depth'] = 2
                    config['min_samples_leaf'] = 5
                    config['min_impurity_decrease'] = 0.0
                    config['n_estimators'] = 4
                    config['max_samples'] = .4
                    config['n_jobs'] = 1

                    n, n_features = 800, 2
                    random_state = 123
                    if sample_weight is not None:
                        sample_weight = check_random_state(random_state).randint(0, 4, size=n)
                    X, y, _ = self._get_policy_data(n, n_features, random_state)
                    forest = trainer(X, y, config, sample_weight=sample_weight)
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
        n = 2000
        random_state = 123
        X, y, _ = self._get_policy_data(n, n_features, random_state)
        forest = PolicyForest(n_estimators=20, n_jobs=1, random_state=123).fit(X, y)
        pred = forest.predict(X)
        pred_val = forest.predict_value(X)
        pred_prob = forest.predict_proba(X)
        assert pred_prob.shape == (X.shape[0], 2)
        feat_imp = forest.feature_importances()
        forest = PolicyForest(n_estimators=20, n_jobs=1, random_state=123).fit(X.astype(np.float32),
                                                                               np.asfortranarray(y))
        np.testing.assert_allclose(pred, forest.predict(tuple(X)))
        np.testing.assert_allclose(pred_val, forest.predict_value(tuple(X)))
        forest = PolicyForest(n_estimators=20, n_jobs=1, random_state=123).fit(tuple(X), tuple(y))
        np.testing.assert_allclose(pred, forest.predict(tuple(X)))
        np.testing.assert_allclose(pred_val, forest.predict_value(tuple(X)))
        np.testing.assert_allclose(pred_prob, forest.predict_proba(tuple(X)))
        forest = PolicyForest(n_estimators=20, n_jobs=1, random_state=123).fit(list(X), list(y))
        np.testing.assert_allclose(pred, forest.predict(list(X)))
        np.testing.assert_allclose(pred_val, forest.predict_value(list(X)))
        np.testing.assert_allclose(pred_prob, forest.predict_proba(list(X)))
        forest = PolicyForest(n_estimators=20, n_jobs=1, random_state=123).fit(pd.DataFrame(X), pd.DataFrame(y))
        np.testing.assert_allclose(pred, forest.predict(pd.DataFrame(X)))
        np.testing.assert_allclose(pred_val, forest.predict_value(pd.DataFrame(X)))
        np.testing.assert_allclose(pred_prob, forest.predict_proba(pd.DataFrame(X)))

        groups = np.repeat(np.arange(X.shape[0]), 2)
        Xraw = X.copy()
        X = np.repeat(X, 2, axis=0)
        T = np.zeros(y.shape)
        T[:, 1] = 1
        T = T.flatten()
        y = y.flatten()
        forest = DRPolicyForest(model_regression=DummyRegressor(strategy='constant', constant=0),
                                model_propensity=DummyClassifier(strategy='uniform'),
                                featurizer=PolynomialFeatures(degree=1, include_bias=False),
                                cv=GroupKFold(n_splits=2),
                                n_estimators=100, n_jobs=1, random_state=123).fit(y, T, X=X,
                                                                                  groups=groups)
        mask = np.abs(Xraw[:, 0]) > .1
        np.testing.assert_allclose(pred[mask], forest.predict(Xraw[mask]))
        np.testing.assert_allclose(pred_val[mask, 1] - pred_val[mask, 0],
                                   forest.predict_value(Xraw[mask]).flatten(), atol=.08)
        np.testing.assert_allclose(feat_imp, forest.feature_importances(), atol=1e-4)
        np.testing.assert_allclose(feat_imp, forest.feature_importances_, atol=1e-4)
        pred = forest.predict(X)
        pred_val = forest.predict_value(X)
        pred_prob = forest.predict_proba(X)
        np.testing.assert_allclose(pred, forest.predict(tuple(X)))
        np.testing.assert_allclose(pred_val, forest.predict_value(tuple(X)))
        np.testing.assert_allclose(pred, forest.predict(pd.DataFrame(X)))
        np.testing.assert_allclose(pred_val, forest.predict_value(pd.DataFrame(X)))
        np.testing.assert_allclose(pred_prob, forest.predict_proba(pd.DataFrame(X)))

        return

    def test_raise_exceptions(self,):
        # test that we raise errors in mishandled situations.
        n_features = 2
        n = 10
        random_state = 123
        X, y, _ = self._get_policy_data(n, n_features, random_state)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=20, max_samples=20).fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=20, max_samples=1.2).fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=4, criterion='peculiar').fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=4, max_depth=-1).fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=4, min_samples_split=-1).fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=4, min_samples_leaf=-1).fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=4, min_weight_fraction_leaf=-1.0).fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=4, max_features=10).fit(X, y)
        with np.testing.assert_raises(ValueError):
            PolicyForest(n_estimators=4, min_balancedness_tol=.55).fit(X, y)

        return

    def test_warm_start(self,):
        n_features = 2
        n = 10
        random_state = 123
        X, y, _ = self._get_policy_data(n, n_features, random_state)

        forest = PolicyForest(n_estimators=4, warm_start=True, random_state=123).fit(X, y)
        with pytest.warns(UserWarning):
            forest.fit(X, y)
        forest.n_estimators = 3
        with np.testing.assert_raises(ValueError):
            forest.fit(X, y)
        forest.n_estimators = 8
        forest.fit(X, y)
        pred1 = forest.predict(X)
        inds1 = forest.get_subsample_inds()
        tree_states1 = [t.random_state for t in forest]

        forest = PolicyForest(n_estimators=8, warm_start=True, random_state=123).fit(X, y)
        pred2 = forest.predict(X)
        inds2 = forest.get_subsample_inds()
        tree_states2 = [t.random_state for t in forest]

        np.testing.assert_allclose(pred1, pred2)
        np.testing.assert_allclose(inds1, inds2)
        np.testing.assert_allclose(tree_states1, tree_states2)
        return

    @pytest.mark.skipif(not graphviz_works, reason="graphviz must be installed to test plotting")
    def test_plotting(self):
        n_features = 2
        n = 1000
        random_state = 123
        X, y, _ = self._get_policy_data(n, n_features, random_state)

        tree = PolicyTree(max_depth=4, random_state=123).fit(X, y)
        tree.plot(max_depth=2)
        tree.render('test', max_depth=2)

        groups = np.repeat(np.arange(X.shape[0]), 2)
        X = np.repeat(X, 2, axis=0)
        T = np.zeros(y.shape)
        T[:, 1] = 1
        T = T.flatten()
        y = y.flatten()
        forest = DRPolicyForest(model_regression=DummyRegressor(strategy='constant', constant=0),
                                model_propensity=DummyClassifier(strategy='uniform'),
                                featurizer=PolynomialFeatures(degree=1, include_bias=False),
                                cv=GroupKFold(n_splits=2),
                                n_estimators=20, n_jobs=1, random_state=123).fit(y, T, X=X,
                                                                                 groups=groups)
        forest.plot(0, max_depth=2)
        forest.render(0, 'testdrf', max_depth=2)
        forest.export_graphviz(0, max_depth=2)

        tree = DRPolicyTree(model_regression=DummyRegressor(strategy='constant', constant=0),
                            model_propensity=DummyClassifier(strategy='uniform'),
                            featurizer=PolynomialFeatures(degree=1, include_bias=False),
                            cv=GroupKFold(n_splits=2), random_state=123).fit(y, T, X=X,
                                                                             groups=groups)
        tree.plot(max_depth=2)
        tree.render('testdrt', max_depth=2)
        tree.export_graphviz(max_depth=2)

    def test_pickling(self,):

        n_features = 2
        n = 10
        random_state = 123
        X, y, _ = self._get_policy_data(n, n_features, random_state)

        forest = PolicyForest(n_estimators=4, warm_start=True, random_state=123).fit(X, y)
        forest.n_estimators = 8
        forest.fit(X, y)
        pred1 = forest.predict(X)

        joblib.dump(forest, 'forest.jbl')
        loaded_forest = joblib.load('forest.jbl')
        np.testing.assert_equal(loaded_forest.n_estimators, forest.n_estimators)
        np.testing.assert_allclose(loaded_forest.predict(X), pred1)
