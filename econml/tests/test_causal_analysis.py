# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np
import pandas as pd
from contextlib import ExitStack
from econml.solutions.causal_analysis import CausalAnalysis
from econml.solutions.causal_analysis._causal_analysis import _CausalInsightsConstants


class TestCausalAnalysis(unittest.TestCase):

    def test_basic_array(self):
        for d_y in [(), (1,)]:
            for classification in [False, True]:
                y = np.random.choice([0, 1], size=(500,) + d_y)
                X = np.hstack((np.random.normal(size=(500, 2)),
                               np.random.choice([0, 1], size=(500, 1)),
                               np.random.choice([0, 1, 2], size=(500, 1))))
                inds = [0, 1, 2, 3]
                cats = [2, 3]
                hinds = [0, 3]
                ca = CausalAnalysis(inds, cats, hinds, classification=classification)
                ca.fit(X, y)
                glo = ca.global_causal_effect()
                coh = ca.cohort_causal_effect(X[:2])
                loc = ca.local_causal_effect(X[:2])

                # global and cohort data should have exactly the same structure, but different values
                assert glo.index.equals(coh.index)

                # local index should have as many times entries as global as there were rows passed in
                assert len(loc.index) == 2 * len(glo.index)

                assert glo.index.names == ['feature', 'feature_value']
                assert loc.index.names == ['sample'] + glo.index.names

                glo_dict = ca._global_causal_effect_dict()
                coh_dict = ca._cohort_causal_effect_dict(X[:2])
                loc_dict = ca._local_causal_effect_dict(X[:2])

                glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
                coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
                loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

                ca._policy_tree_string(X, 1)
                ca._heterogeneity_tree_string(X, 1)
                ca._heterogeneity_tree_string(X, 3)

                # Can't handle multi-dimensional treatments
                with self.assertRaises(AssertionError):
                    ca._policy_tree_string(X, 3)

                # global shape is (d_y, sum(d_t))
                assert glo_point_est.shape == coh_point_est.shape == (1, 5)
                assert loc_point_est.shape == (2,) + glo_point_est.shape
                if not classification:
                    # ExitStack can be used as a "do nothing" ContextManager
                    cm = ExitStack()
                else:
                    cm = self.assertRaises(Exception)
                with cm:
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 1, y[:2])
                    assert np.shape(inf.point_estimate) == np.shape(y[:2])
                    inf.summary_frame()
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 2, y[:2])
                    assert np.shape(inf.point_estimate) == np.shape(y[:2])
                    inf.summary_frame()

                    ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2])

                # features; for categoricals they should appear #cats-1 times each
                fts = ['x0', 'x1', 'x2', 'x3', 'x3']

                for i in range(len(fts)):
                    assert fts[i] == glo.index[i][0] == loc.index[i][1] == loc.index[len(fts) + i][1]

                badargs = [
                    (inds, cats, [4]),  # hinds out of range
                    (inds, cats, ["test"])  # hinds out of range
                ]

                for args in badargs:
                    with self.assertRaises(Exception):
                        ca = CausalAnalysis(*args)
                        ca.fit(X, y)

    def test_basic_pandas(self):
        for classification in [False, True]:
            y = pd.Series(np.random.choice([0, 1], size=(500,)))
            X = pd.DataFrame({'a': np.random.normal(size=500),
                              'b': np.random.normal(size=500),
                              'c': np.random.choice([0, 1], size=500),
                              'd': np.random.choice(['a', 'b', 'c'], size=500)})
            n_inds = [0, 1, 2, 3]
            t_inds = ['a', 'b', 'c', 'd']
            n_cats = [2, 3]
            t_cats = ['c', 'd']
            n_hinds = [0, 3]
            t_hinds = ['a', 'd']
            for (inds, cats, hinds) in [(n_inds, n_cats, n_hinds), (t_inds, t_cats, t_hinds)]:
                ca = CausalAnalysis(inds, cats, hinds, classification=classification)
                ca.fit(X, y)
                glo = ca.global_causal_effect()
                coh = ca.cohort_causal_effect(X[:2])
                loc = ca.local_causal_effect(X[:2])

                # global and cohort data should have exactly the same structure, but different values
                assert glo.index.equals(coh.index)

                # local index should have as many times entries as global as there were rows passed in
                assert len(loc.index) == 2 * len(glo.index)

                assert glo.index.names == ['feature', 'feature_value']
                assert loc.index.names == ['sample'] + glo.index.names

                # features; for categoricals they should appear #cats-1 times each
                fts = ['a', 'b', 'c', 'd', 'd']

                for i in range(len(fts)):
                    assert fts[i] == glo.index[i][0] == loc.index[i][1] == loc.index[len(fts) + i][1]

                glo_dict = ca._global_causal_effect_dict()
                coh_dict = ca._cohort_causal_effect_dict(X[:2])
                loc_dict = ca._local_causal_effect_dict(X[:2])

                glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
                coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
                loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

                # global shape is (d_y, sum(d_t))
                assert glo_point_est.shape == coh_point_est.shape == (1, 5)
                assert loc_point_est.shape == (2,) + glo_point_est.shape

                ca._policy_tree_string(X, inds[1])
                ca._heterogeneity_tree_string(X, inds[1])
                ca._heterogeneity_tree_string(X, inds[3])

                # Can't handle multi-dimensional treatments
                with self.assertRaises(AssertionError):
                    ca._policy_tree_string(X, inds[3])

                if not classification:
                    # ExitStack can be used as a "do nothing" ContextManager
                    cm = ExitStack()
                else:
                    cm = self.assertRaises(Exception)
                with cm:
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), inds[1], y[:2])
                    assert np.shape(inf.point_estimate) == np.shape(y[:2])
                    inf.summary_frame()
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), inds[2], y[:2])
                    assert np.shape(inf.point_estimate) == np.shape(y[:2])
                    inf.summary_frame()

                    ca._whatif_dict(X[:2], np.ones(shape=(2,)), inds[1], y[:2])

            badargs = [
                (n_inds, n_cats, [4]),  # hinds out of range
                (n_inds, n_cats, ["test"])  # hinds out of range
            ]

            for args in badargs:
                with self.assertRaises(Exception):
                    ca = CausalAnalysis(*args)
                    ca.fit(X, y)

    def test_automl_first_stage(self):
        d_y = (1,)
        for classification in [False, True]:
            y = np.random.choice([0, 1], size=(500,) + d_y)
            X = np.hstack((np.random.normal(size=(500, 2)),
                           np.random.choice([0, 1], size=(500, 1)),
                           np.random.choice([0, 1, 2], size=(500, 1))))
            inds = [0, 1, 2, 3]
            cats = [2, 3]
            hinds = [0, 3]
            ca = CausalAnalysis(inds, cats, hinds, classification=classification, nuisance_models='automl')
            ca.fit(X, y)
            glo = ca.global_causal_effect()
            coh = ca.cohort_causal_effect(X[:2])
            loc = ca.local_causal_effect(X[:2])

            # global and cohort data should have exactly the same structure, but different values
            assert glo.index.equals(coh.index)

            # local index should have as many times entries as global as there were rows passed in
            assert len(loc.index) == 2 * len(glo.index)

            assert glo.index.names == ['feature', 'feature_value']
            assert loc.index.names == ['sample'] + glo.index.names

            glo_dict = ca._global_causal_effect_dict()
            coh_dict = ca._cohort_causal_effect_dict(X[:2])
            loc_dict = ca._local_causal_effect_dict(X[:2])

            glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
            coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
            loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

            ca._policy_tree_string(X, 1)
            ca._heterogeneity_tree_string(X, 1)
            ca._heterogeneity_tree_string(X, 3)

            # Can't handle multi-dimensional treatments
            with self.assertRaises(AssertionError):
                ca._policy_tree_string(X, 3)

            # global shape is (d_y, sum(d_t))
            assert glo_point_est.shape == coh_point_est.shape == (1, 5)
            assert loc_point_est.shape == (2,) + glo_point_est.shape
            if not classification:
                # ExitStack can be used as a "do nothing" ContextManager
                cm = ExitStack()
            else:
                cm = self.assertRaises(Exception)
            with cm:
                inf = ca.whatif(X[:2], np.ones(shape=(2,)), 1, y[:2])
                assert np.shape(inf.point_estimate) == np.shape(y[:2])
                inf.summary_frame()
                inf = ca.whatif(X[:2], np.ones(shape=(2,)), 2, y[:2])
                assert np.shape(inf.point_estimate) == np.shape(y[:2])
                inf.summary_frame()

                ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2])

            # features; for categoricals they should appear #cats-1 times each
            fts = ['x0', 'x1', 'x2', 'x3', 'x3']

            for i in range(len(fts)):
                assert fts[i] == glo.index[i][0] == loc.index[i][1] == loc.index[len(fts) + i][1]

            badargs = [
                (inds, cats, [4]),  # hinds out of range
                (inds, cats, ["test"])  # hinds out of range
            ]

            for args in badargs:
                with self.assertRaises(Exception):
                    ca = CausalAnalysis(*args)
                    ca.fit(X, y)

    def test_one_feature(self):
        # make sure we don't run into problems dropping every index
        y = pd.Series(np.random.choice([0, 1], size=(500,)))
        X = pd.DataFrame({'a': np.random.normal(size=500),
                          'b': np.random.normal(size=500),
                          'c': np.random.choice([0, 1], size=500),
                          'd': np.random.choice(['a', 'b', 'c'], size=500)})
        inds = ['a']
        cats = ['c', 'd']
        hinds = ['a', 'd']

        ca = CausalAnalysis(inds, cats, hinds, classification=False)
        ca.fit(X, y)
        glo = ca.global_causal_effect()
        coh = ca.cohort_causal_effect(X[:2])
        loc = ca.local_causal_effect(X[:2])

        # global and cohort data should have exactly the same structure, but different values
        assert glo.index.equals(coh.index)

        # local index should have as many times entries as global as there were rows passed in
        assert len(loc.index) == 2 * len(glo.index)

        assert glo.index.names == ['feature']
        assert loc.index.names == ['sample']

        glo_dict = ca._global_causal_effect_dict()
        coh_dict = ca._cohort_causal_effect_dict(X[:2])
        loc_dict = ca._local_causal_effect_dict(X[:2])

        glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
        coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
        loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

        # global shape is (d_y, sum(d_t))
        assert glo_point_est.shape == coh_point_est.shape == (1, 1)
        assert loc_point_est.shape == (2,) + glo_point_est.shape

        ca._policy_tree_string(X, inds[0])
        ca._heterogeneity_tree_string(X, inds[0])

    def test_final_models(self):
        d_y = (1,)
        y = np.random.choice([0, 1], size=(500,) + d_y)
        X = np.hstack((np.random.normal(size=(500, 2)),
                       np.random.choice([0, 1], size=(500, 1)),
                       np.random.choice([0, 1, 2], size=(500, 1))))
        inds = [0, 1, 2, 3]
        cats = [2, 3]
        hinds = [0, 3]
        for h_model in ['forest', 'linear']:
            for classification in [False, True]:
                ca = CausalAnalysis(inds, cats, hinds, classification=classification, heterogeneity_model=h_model)
                ca.fit(X, y)
                glo = ca.global_causal_effect()
                coh = ca.cohort_causal_effect(X[:2])
                loc = ca.local_causal_effect(X[:2])
                glo_dict = ca._global_causal_effect_dict()
                coh_dict = ca._cohort_causal_effect_dict(X[:2])
                loc_dict = ca._local_causal_effect_dict(X[:2])

                ca._policy_tree_string(X, 1)
                ca._heterogeneity_tree_string(X, 1)
                ca._heterogeneity_tree_string(X, 3)

                # Can't handle multi-dimensional treatments
                with self.assertRaises(AssertionError):
                    ca._policy_tree_string(X, 3)

                if not classification:
                    # ExitStack can be used as a "do nothing" ContextManager
                    cm = ExitStack()
                else:
                    cm = self.assertRaises(Exception)
                with cm:
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 1, y[:2])
                    inf.summary_frame()
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 2, y[:2])
                    inf.summary_frame()

                    ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2])

        with self.assertRaises(AssertionError):
            ca = CausalAnalysis(inds, cats, hinds, classification=classification, heterogeneity_model='other')
            ca.fit(X, y)

    def test_forest_with_pandas(self):
        y = pd.Series(np.random.choice([0, 1], size=(500,)))
        X = pd.DataFrame({'a': np.random.normal(size=500),
                          'b': np.random.normal(size=500),
                          'c': np.random.choice([0, 1], size=500),
                          'd': np.random.choice(['a', 'b', 'c'], size=500)})
        inds = ['a', 'b', 'c', 'd']
        cats = ['c', 'd']
        hinds = ['a', 'd']

        ca = CausalAnalysis(inds, cats, hinds, heterogeneity_model='forest')
        ca.fit(X, y)
        glo = ca.global_causal_effect()
        coh = ca.cohort_causal_effect(X[:2])
        loc = ca.local_causal_effect(X[:2])

        # global and cohort data should have exactly the same structure, but different values
        assert glo.index.equals(coh.index)

        # local index should have as many times entries as global as there were rows passed in
        assert len(loc.index) == 2 * len(glo.index)

        assert glo.index.names == ['feature', 'feature_value']
        assert loc.index.names == ['sample'] + glo.index.names

        # features; for categoricals they should appear #cats-1 times each
        fts = ['a', 'b', 'c', 'd', 'd']

        for i in range(len(fts)):
            assert fts[i] == glo.index[i][0] == loc.index[i][1] == loc.index[len(fts) + i][1]

        glo_dict = ca._global_causal_effect_dict()
        coh_dict = ca._cohort_causal_effect_dict(X[:2])
        loc_dict = ca._local_causal_effect_dict(X[:2])

        glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
        coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
        loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

        # global shape is (d_y, sum(d_t))
        assert glo_point_est.shape == coh_point_est.shape == (1, 5)
        assert loc_point_est.shape == (2,) + glo_point_est.shape

        ca._policy_tree_string(X, inds[1])
        ca._heterogeneity_tree_string(X, inds[1])
        ca._heterogeneity_tree_string(X, inds[3])

        # Can't handle multi-dimensional treatments
        with self.assertRaises(AssertionError):
            ca._policy_tree_string(X, inds[3])

    def test_warm_start(self):
        for classification in [True, False]:
            # dgp
            X1 = np.random.normal(0, 1, size=(500, 5))
            X2 = np.random.choice([0, 1], size=(500, 1))
            X3 = np.random.choice([0, 1, 2], size=(500, 1))
            X = np.hstack((X1, X2, X3))
            X_df = pd.DataFrame(X, columns=[f"x{i} "for i in range(7)])
            y = np.random.choice([0, 1], size=(500,))
            y_df = pd.Series(y)
            # model
            hetero_inds = [0, 1, 2]
            feat_inds = [1, 3, 5]
            categorical = [5, 6]
            ca = CausalAnalysis(feat_inds, categorical, heterogeneity_inds=hetero_inds,
                                classification=classification,
                                nuisance_models='linear', heterogeneity_model="linear", n_jobs=-1)
            ca.fit(X_df, y)
            eff = ca.global_causal_effect(alpha=0.05)
            eff = ca.local_causal_effect(X_df, alpha=0.05)

            ca.feature_inds = [1, 2, 3, 5]
            ca.fit(X_df, y, warm_start=True)
            eff = ca.global_causal_effect(alpha=0.05)
            eff = ca.local_causal_effect(X_df, alpha=0.05)

    def test_empty_hinds(self):
        for h_model in ['linear', 'forest']:
            for classification in [True, False]:
                X1 = np.random.normal(0, 1, size=(500, 5))
                X2 = np.random.choice([0, 1], size=(500, 1))
                X3 = np.random.choice([0, 1, 2], size=(500, 1))
                X = np.hstack((X1, X2, X3))
                X_df = pd.DataFrame(X, columns=[f"x{i} "for i in range(7)])
                y = np.random.choice([0, 1], size=(500,))
                y_df = pd.Series(y)
                # model
                hetero_inds = [[], [], []]
                feat_inds = [1, 3, 5]
                categorical = [5, 6]
                ca = CausalAnalysis(feat_inds, categorical, heterogeneity_inds=hetero_inds,
                                    classification=classification,
                                    nuisance_models='linear', heterogeneity_model="linear", n_jobs=-1)
                ca.fit(X_df, y)
                eff = ca.global_causal_effect(alpha=0.05)
                eff = ca.local_causal_effect(X_df, alpha=0.05)

    def test_can_serialize(self):
        import pickle
        y = pd.Series(np.random.choice([0, 1], size=(500,)))
        X = pd.DataFrame({'a': np.random.normal(size=500),
                          'b': np.random.normal(size=500),
                          'c': np.random.choice([0, 1], size=500),
                          'd': np.random.choice(['a', 'b', 'c'], size=500)})
        inds = ['a', 'b', 'c', 'd']
        cats = ['c', 'd']
        hinds = ['a', 'd']

        ca = CausalAnalysis(inds, cats, hinds, heterogeneity_model='linear')
        ca = pickle.loads(pickle.dumps(ca))
        ca.fit(X, y)
        ca = pickle.loads(pickle.dumps(ca))
        eff = ca.global_causal_effect()
