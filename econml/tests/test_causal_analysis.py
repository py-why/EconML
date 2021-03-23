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

            badargs = [
                (n_inds, n_cats, [4]),  # hinds out of range
                (n_inds, n_cats, ["test"])  # hinds out of range
            ]

            for args in badargs:
                with self.assertRaises(Exception):
                    ca = CausalAnalysis(*args)
                    ca.fit(X, y)
