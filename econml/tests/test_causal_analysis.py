# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from contextlib import ExitStack
import itertools
import numpy as np
from numpy.core.fromnumeric import squeeze
import pandas as pd
import pytest

from econml.solutions.causal_analysis import CausalAnalysis
from econml.solutions.causal_analysis._causal_analysis import _CausalInsightsConstants


def assert_less_close(arr1, arr2):
    assert np.all(np.logical_or(arr1 <= arr2, np.isclose(arr1, arr2)))


@pytest.mark.serial
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
                glo_dict2 = ca._global_causal_effect_dict(row_wise=True)

                coh_dict = ca._cohort_causal_effect_dict(X[:2])
                coh_dict2 = ca._cohort_causal_effect_dict(X[:2], row_wise=True)

                loc_dict = ca._local_causal_effect_dict(X[:2])
                loc_dict2 = ca._local_causal_effect_dict(X[:2], row_wise=True)

                glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
                coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
                loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

                ca._heterogeneity_tree_output(X, 1)
                ca._heterogeneity_tree_output(X, 3)

                # continuous treatments have typical treatment values equal to
                # the mean of the absolute value of non-zero entries
                np.testing.assert_allclose(ca.typical_treatment_value(0), np.mean(np.abs(X[:, 0])))
                np.testing.assert_allclose(ca.typical_treatment_value(1), np.mean(np.abs(X[:, 1])))
                # discrete treatments have typical treatment value 1
                assert ca.typical_treatment_value(2) == ca.typical_treatment_value(3) == 1

                # Make sure we handle continuous, binary, and multi-class treatments
                # For multiple discrete treatments, one "always treat" value per non-default treatment
                for (idx, length) in [(0, 1), (1, 1), (2, 1), (3, 2)]:
                    pto = ca._policy_tree_output(X, idx)
                    policy_val = pto.policy_value
                    always_trt = pto.always_treat
                    assert isinstance(pto.control_name, str)
                    assert isinstance(always_trt, dict)
                    assert np.array(policy_val).shape == ()
                    assert len(always_trt) == length
                    for val in always_trt.values():
                        assert np.array(val).shape == ()

                    # policy value should exceed always treating with any treatment
                    assert_less_close(np.array(list(always_trt.values())), policy_val)

                    ind_pol = ca.individualized_policy(X, inds[idx])

                # global shape is (d_y, sum(d_t))
                assert glo_point_est.shape == coh_point_est.shape == (1, 5)
                assert loc_point_est.shape == (2,) + glo_point_est.shape

                # global and cohort row-wise dicts have d_y * d_t entries
                assert len(
                    glo_dict2[_CausalInsightsConstants.RowData]) == len(
                    coh_dict2[_CausalInsightsConstants.RowData]) == 5
                # local dictionary is flattened to n_rows * d_y * d_t
                assert len(loc_dict2[_CausalInsightsConstants.RowData]) == 10

                if not classification:
                    # ExitStack can be used as a "do nothing" ContextManager
                    cm = ExitStack()
                else:
                    cm = self.assertRaises(Exception)
                with cm:
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 1, y[:2])
                    assert np.shape(inf.point_estimate) == (2,)
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 2, y[:2])
                    assert np.shape(inf.point_estimate) == (2,)

                    ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2])
                    ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2], row_wise=True)

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
            for category in [False, True]:
                y = pd.Series(np.random.choice([0, 1], size=(500,)))
                X = pd.DataFrame({'a': np.random.normal(size=500),
                                  'b': np.random.normal(size=500),
                                  'c': np.random.choice([0, 1], size=500),
                                  'd': np.random.choice(['a', 'b', 'c'], size=500)})

                if category:
                    X['c'] = X['c'].astype('category')
                    X['d'] = X['d'].astype('category')

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
                    glo_dict2 = ca._global_causal_effect_dict(row_wise=True)

                    coh_dict = ca._cohort_causal_effect_dict(X[:2])
                    coh_dict2 = ca._cohort_causal_effect_dict(X[:2], row_wise=True)

                    loc_dict = ca._local_causal_effect_dict(X[:2])
                    loc_dict2 = ca._local_causal_effect_dict(X[:2], row_wise=True)

                    glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
                    coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
                    loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

                    # global shape is (d_y, sum(d_t))
                    assert glo_point_est.shape == coh_point_est.shape == (1, 5)
                    assert loc_point_est.shape == (2,) + glo_point_est.shape

                    # global and cohort row-wise dicts have d_y * d_t entries
                    assert len(
                        glo_dict2[_CausalInsightsConstants.RowData]) == len(
                        coh_dict2[_CausalInsightsConstants.RowData]) == 5
                    # local dictionary is flattened to n_rows * d_y * d_t
                    assert len(loc_dict2[_CausalInsightsConstants.RowData]) == 10

                    pto = ca._policy_tree_output(X, inds[1])
                    ca._heterogeneity_tree_output(X, inds[1])
                    ca._heterogeneity_tree_output(X, inds[3])

                    # continuous treatments have typical treatment values equal to
                    # the mean of the absolute value of non-zero entries
                    np.testing.assert_allclose(ca.typical_treatment_value(inds[0]), np.mean(np.abs(X['a'])))
                    np.testing.assert_allclose(ca.typical_treatment_value(inds[1]), np.mean(np.abs(X['b'])))
                    # discrete treatments have typical treatment value 1
                    assert ca.typical_treatment_value(inds[2]) == ca.typical_treatment_value(inds[3]) == 1

                    # Make sure we handle continuous, binary, and multi-class treatments
                    # For multiple discrete treatments, one "always treat" value per non-default treatment
                    for (idx, length) in [(0, 1), (1, 1), (2, 1), (3, 2)]:
                        pto = ca._policy_tree_output(X, inds[idx])
                        policy_val = pto.policy_value
                        always_trt = pto.always_treat
                        assert isinstance(pto.control_name, str)
                        assert isinstance(always_trt, dict)
                        assert np.array(policy_val).shape == ()
                        assert len(always_trt) == length
                        for val in always_trt.values():
                            assert np.array(val).shape == ()

                        # policy value should exceed always treating with any treatment
                        assert_less_close(np.array(list(always_trt.values())), policy_val)

                        ind_pol = ca.individualized_policy(X, inds[idx])

                    if not classification:
                        # ExitStack can be used as a "do nothing" ContextManager
                        cm = ExitStack()
                    else:
                        cm = self.assertRaises(Exception)
                    with cm:
                        inf = ca.whatif(X[:2], np.ones(shape=(2,)), inds[1], y[:2])
                        assert np.shape(inf.point_estimate) == np.shape(y[:2])
                        inf = ca.whatif(X[:2], np.ones(shape=(2,)), inds[2], y[:2])
                        assert np.shape(inf.point_estimate) == np.shape(y[:2])

                        ca._whatif_dict(X[:2], np.ones(shape=(2,)), inds[1], y[:2])
                        ca._whatif_dict(X[:2], np.ones(shape=(2,)), inds[1], y[:2], row_wise=True)

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
            glo_dict2 = ca._global_causal_effect_dict(row_wise=True)

            coh_dict = ca._cohort_causal_effect_dict(X[:2])
            coh_dict2 = ca._cohort_causal_effect_dict(X[:2], row_wise=True)

            loc_dict = ca._local_causal_effect_dict(X[:2])
            loc_dict2 = ca._local_causal_effect_dict(X[:2], row_wise=True)

            glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
            coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
            loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

            ca._policy_tree_output(X, 1)
            ca._heterogeneity_tree_output(X, 1)
            ca._heterogeneity_tree_output(X, 3)

            # Make sure we handle continuous, binary, and multi-class treatments
            # For multiple discrete treatments, one "always treat" value per non-default treatment
            for (idx, length) in [(0, 1), (1, 1), (2, 1), (3, 2)]:
                pto = ca._policy_tree_output(X, idx)
                policy_val = pto.policy_value
                always_trt = pto.always_treat
                assert isinstance(pto.control_name, str)
                assert isinstance(always_trt, dict)
                assert np.array(policy_val).shape == ()
                assert len(always_trt) == length
                for val in always_trt.values():
                    assert np.array(val).shape == ()

                # policy value should exceed always treating with any treatment
                assert_less_close(np.array(list(always_trt.values())), policy_val)

                ind_pol = ca.individualized_policy(X, inds[idx])

            # global shape is (d_y, sum(d_t))
            assert glo_point_est.shape == coh_point_est.shape == (1, 5)
            assert loc_point_est.shape == (2,) + glo_point_est.shape

            # global and cohort row-wise dicts have d_y * d_t entries
            assert len(
                glo_dict2[_CausalInsightsConstants.RowData]) == len(
                coh_dict2[_CausalInsightsConstants.RowData]) == 5
            # local dictionary is flattened to n_rows * d_y * d_t
            assert len(loc_dict2[_CausalInsightsConstants.RowData]) == 10

            if not classification:
                # ExitStack can be used as a "do nothing" ContextManager
                cm = ExitStack()
            else:
                cm = self.assertRaises(Exception)
            with cm:
                inf = ca.whatif(X[:2], np.ones(shape=(2,)), 1, y[:2])
                assert np.shape(inf.point_estimate) == (2,)
                inf = ca.whatif(X[:2], np.ones(shape=(2,)), 2, y[:2])
                assert np.shape(inf.point_estimate) == (2,)

                ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2])
                ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2], row_wise=True)

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
        glo_dict2 = ca._global_causal_effect_dict(row_wise=True)

        coh_dict = ca._cohort_causal_effect_dict(X[:2])
        coh_dict2 = ca._cohort_causal_effect_dict(X[:2], row_wise=True)

        loc_dict = ca._local_causal_effect_dict(X[:2])
        loc_dict2 = ca._local_causal_effect_dict(X[:2], row_wise=True)

        glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
        coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
        loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

        # global shape is (d_y, sum(d_t))
        assert glo_point_est.shape == coh_point_est.shape == (1, 1)
        assert loc_point_est.shape == (2,) + glo_point_est.shape

        glo2 = ca.global_causal_effect(keep_all_levels=True)
        coh2 = ca.cohort_causal_effect(X[:2], keep_all_levels=True)
        loc2 = ca.local_causal_effect(X[:2], keep_all_levels=True)
        assert ({ind.name for ind in glo2.index.levels} ==
                {ind.name for ind in coh2.index.levels} ==
                {"outcome", "feature", "feature_value"})
        assert {ind.name for ind in loc2.index.levels} == {"sample", "outcome", "feature", "feature_value"}

        # global and cohort row-wise dicts have d_y * d_t entries
        assert len(
            glo_dict2[_CausalInsightsConstants.RowData]) == len(
            coh_dict2[_CausalInsightsConstants.RowData]) == 1
        # local dictionary is flattened to n_rows * d_y * d_t
        assert len(loc_dict2[_CausalInsightsConstants.RowData]) == 2

        ca._policy_tree_output(X, inds[0])
        ca._heterogeneity_tree_output(X, inds[0])

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

                ca._policy_tree_output(X, 1)
                ca._heterogeneity_tree_output(X, 1)
                ca._heterogeneity_tree_output(X, 3)

                # Make sure we handle continuous, binary, and multi-class treatments
                # For multiple discrete treatments, one "always treat" value per non-default treatment
                for (idx, length) in [(0, 1), (1, 1), (2, 1), (3, 2)]:
                    pto = ca._policy_tree_output(X, idx)
                    policy_val = pto.policy_value
                    always_trt = pto.always_treat
                    assert isinstance(pto.control_name, str)
                    assert isinstance(always_trt, dict)
                    assert np.array(policy_val).shape == ()
                    assert len(always_trt) == length
                    for val in always_trt.values():
                        assert np.array(val).shape == ()

                    # policy value should exceed always treating with any treatment
                    assert_less_close(np.array(list(always_trt.values())), policy_val)

                    ind_pol = ca.individualized_policy(X, inds[idx])

                if not classification:
                    # ExitStack can be used as a "do nothing" ContextManager
                    cm = ExitStack()
                else:
                    cm = self.assertRaises(Exception)
                with cm:
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 1, y[:2])
                    inf = ca.whatif(X[:2], np.ones(shape=(2,)), 2, y[:2])
                    ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2])
                    ca._whatif_dict(X[:2], np.ones(shape=(2,)), 1, y[:2], row_wise=True)

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
        glo_dict2 = ca._global_causal_effect_dict(row_wise=True)

        coh_dict = ca._cohort_causal_effect_dict(X[:2])
        coh_dict2 = ca._cohort_causal_effect_dict(X[:2], row_wise=True)

        loc_dict = ca._local_causal_effect_dict(X[:2])
        loc_dict2 = ca._local_causal_effect_dict(X[:2], row_wise=True)

        glo_point_est = np.array(glo_dict[_CausalInsightsConstants.PointEstimateKey])
        coh_point_est = np.array(coh_dict[_CausalInsightsConstants.PointEstimateKey])
        loc_point_est = np.array(loc_dict[_CausalInsightsConstants.PointEstimateKey])

        # global shape is (d_y, sum(d_t))
        assert glo_point_est.shape == coh_point_est.shape == (1, 5)
        assert loc_point_est.shape == (2,) + glo_point_est.shape

        # global and cohort row-wise dicts have d_y * d_t entries
        assert len(
            glo_dict2[_CausalInsightsConstants.RowData]) == len(
            coh_dict2[_CausalInsightsConstants.RowData]) == 5
        # local dictionary is flattened to n_rows * d_y * d_t
        assert len(loc_dict2[_CausalInsightsConstants.RowData]) == 10

        ca._policy_tree_output(X, inds[1])
        ca._heterogeneity_tree_output(X, inds[1])
        ca._heterogeneity_tree_output(X, inds[3])

        # Make sure we handle continuous, binary, and multi-class treatments
        # For multiple discrete treatments, one "always treat" value per non-default treatment
        for (idx, length) in [(0, 1), (1, 1), (2, 1), (3, 2)]:
            pto = ca._policy_tree_output(X, inds[idx])
            policy_val = pto.policy_value
            always_trt = pto.always_treat
            assert isinstance(pto.control_name, str)
            assert isinstance(always_trt, dict)
            assert np.array(policy_val).shape == ()
            assert len(always_trt) == length
            for val in always_trt.values():
                assert np.array(val).shape == ()

            # policy value should exceed always treating with any treatment
            assert_less_close(np.array(list(always_trt.values())), policy_val)

            ind_pol = ca.individualized_policy(X, inds[idx])

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
                                    nuisance_models='linear', heterogeneity_model=h_model, n_jobs=-1)
                ca.fit(X_df, y)
                eff = ca.global_causal_effect(alpha=0.05)
                eff = ca.local_causal_effect(X_df, alpha=0.05)
                for ind in feat_inds:
                    pto = ca._policy_tree_output(X_df, ind)
                    ca._individualized_policy_dict(X_df, ind)

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

    def test_over_cat_limit(self):
        y = pd.Series(np.random.choice([0, 1], size=(500,)))
        X = pd.DataFrame({'a': np.random.normal(size=500),
                          'b': np.random.normal(size=500),
                          'c': np.random.choice([0, 1], size=500),
                          'd': np.random.choice(['a', 'b', 'c', 'd'], size=500),
                          'e': np.random.choice([7, 8, 9, 10, 11], size=500),
                          'f': np.random.choice(['x', 'y'], size=500),
                          'g': np.random.choice([0, 1], size=500),
                          'h': np.random.choice(['q', 'r', 's'], size=500)})
        inds = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        cats = ['c', 'd', 'e', 'f', 'g', 'h']
        hinds = ['a', 'd']
        ca = CausalAnalysis(inds, cats, hinds, upper_bound_on_cat_expansion=2)
        ca.fit(X, y)

        # columns 'd', 'e', 'h' have too many values
        self.assertEqual([res.feature_name for res in ca._results], ['a', 'b', 'c', 'f', 'g'])

        inds = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        cats = ['c', 'd', 'e', 'f', 'g', 'h']
        hinds = ['a', 'd']
        ca = CausalAnalysis(inds, cats, hinds, upper_bound_on_cat_expansion=3)
        ca.fit(X, y)

        # columns 'd', 'e' have too many values
        self.assertEqual([res.feature_name for res in ca._results], ['a', 'b', 'c', 'f', 'g', 'h'])

        ca.upper_bound_on_cat_expansion = 2
        ca.fit(X, y, warm_start=True)

        # lowering bound shouldn't affect already fit columns when warm starting
        self.assertEqual([res.feature_name for res in ca._results], ['a', 'b', 'c', 'f', 'g', 'h'])

        ca.upper_bound_on_cat_expansion = 4
        ca.fit(X, y, warm_start=True)

        # column d is now okay, too
        self.assertEqual([res.feature_name for res in ca._results], ['a', 'b', 'c', 'd', 'f', 'g', 'h'])

    def test_individualized_policy(self):
        y_arr = np.random.choice([0, 1], size=(500,))
        X = pd.DataFrame({'a': np.random.normal(size=500),
                          'b': np.random.normal(size=500),
                          'c': np.random.choice([0, 1], size=500),
                          'd': np.random.choice(['a', 'b', 'c'], size=500)})
        inds = ['a', 'b', 'c', 'd']
        cats = ['c', 'd']
        hinds = ['a', 'd']

        for y in [pd.Series(y_arr), y_arr.reshape(-1, 1)]:
            for classification in [True, False]:
                ca = CausalAnalysis(inds, cats, hinds, heterogeneity_model='linear', classification=classification)
                ca.fit(X, y)
                df = ca.individualized_policy(X, 'a')
                self.assertEqual(df.shape[0], 500)  # all rows included by default
                self.assertEqual(df.shape[1], 4 + X.shape[1])  # new cols for policy, effect, upper and lower bounds
                df = ca.individualized_policy(X, 'b', n_rows=5)
                self.assertEqual(df.shape[0], 5)
                self.assertEqual(df.shape[1], 4 + X.shape[1])  # new cols for policy, effect, upper and lower bounds
                # verify that we can use a scalar treatment cost
                df = ca.individualized_policy(X, 'c', treatment_costs=100)
                self.assertEqual(df.shape[0], 500)
                self.assertEqual(df.shape[1], 4 + X.shape[1])  # new cols for policy, effect, upper and lower bounds
                # verify that we can specify per-treatment costs for each sample
                df = ca.individualized_policy(X, 'd', alpha=0.05, treatment_costs=np.random.normal(size=(500, 2)))
                self.assertEqual(df.shape[0], 500)
                self.assertEqual(df.shape[1], 4 + X.shape[1])  # new cols for policy, effect, upper and lower bounds

                dictionary = ca._individualized_policy_dict(X, 'a')

    def test_random_state(self):
        # verify that using the same state returns the same results each time
        y = np.random.choice([0, 1], size=(500,))
        X = np.hstack((np.random.normal(size=(500, 2)),
                       np.random.choice([0, 1], size=(500, 1)),
                       np.random.choice([0, 1, 2], size=(500, 1))))
        inds = [0, 1, 2, 3]
        cats = [2, 3]
        hinds = [0, 3]

        for n_model, h_model, classification in\
            itertools.product(['linear', 'automl'],
                              ['linear', 'forest'],
                              [True, False]):

            ca = CausalAnalysis(inds, cats, hinds, classification=classification,
                                nuisance_models=n_model, heterogeneity_model=h_model, random_state=123)
            ca.fit(X, y)
            glo = ca.global_causal_effect()

            ca2 = CausalAnalysis(inds, cats, hinds, classification=classification,
                                 nuisance_models=n_model, heterogeneity_model=h_model, random_state=123)
            ca2.fit(X, y)
            glo2 = ca.global_causal_effect()

            np.testing.assert_equal(glo.point.values, glo2.point.values)
            np.testing.assert_equal(glo.stderr.values, glo2.stderr.values)

    def test_can_set_categories(self):
        y = pd.Series(np.random.choice([0, 1], size=(500,)))
        X = pd.DataFrame({'a': np.random.normal(size=500),
                          'b': np.random.normal(size=500),
                          'c': np.random.choice([0, 1], size=500),
                          'd': np.random.choice(['a', 'b', 'c'], size=500)})
        inds = ['a', 'b', 'c', 'd']
        cats = ['c', 'd']
        hinds = ['a', 'd']

        # set the categories for column 'd' explicitly so that b is default
        categories = ['auto', ['b', 'c', 'a']]

        ca = CausalAnalysis(inds, cats, hinds, heterogeneity_model='linear', categories=categories)
        ca.fit(X, y)
        eff = ca.global_causal_effect()
        values = eff.loc['d'].index.values
        np.testing.assert_equal(eff.loc['d'].index.values, ['cvb', 'avb'])

    def test_policy_with_index(self):
        inds = np.arange(1000)
        np.random.shuffle(inds)
        X = pd.DataFrame(np.random.normal(0, 1, size=(1000, 2)), columns=['A', 'B'], index=inds)
        y = np.random.normal(0, 1, size=1000)
        ca_test = CausalAnalysis(feature_inds=['A'], categorical=[])
        ca_test.fit(X, y)
        ind_policy = ca_test.individualized_policy(X[:50], feature_index='A')
        self.assertFalse(ind_policy.isnull().values.any())

    def test_invalid_inds(self):
        X = np.zeros((300, 6))
        y = np.random.normal(size=(300,))

        # first column: 10 ones, this is fine
        X[np.random.choice(300, 10, replace=False), 0] = 1  # ten ones, should be fine

        # second column: 6 categories, plenty of random instances of each
        # this is fine only if we increase the cateogry limit
        X[:, 1] = np.random.choice(6, 300)  # six categories

        # third column: nine ones, lots of twos, not enough unless we disable check
        X[np.random.choice(300, 100, replace=False), 2] = 2
        X[np.random.choice(300, 9, replace=False), 2] = 1

        # fourth column: 5 ones, also not enough but barely works even with forest heterogeneity
        X[np.random.choice(300, 5, replace=False), 3] = 1

        # fifth column: 2 ones, ensures that we will change number of folds for linear heterogeneity
        # forest heterogeneity won't work
        X[np.random.choice(300, 2, replace=False), 4] = 1

        # sixth column: just 1 one, not enough even without check
        X[np.random.choice(300, 1), 5] = 1  # one instance of

        col_names = ['a', 'b', 'c', 'd', 'e', 'f']
        X = pd.DataFrame(X, columns=col_names)

        for n in ['linear', 'automl']:
            for h in ['linear', 'forest']:
                for warm_start in [True, False]:
                    ca = CausalAnalysis(col_names, col_names, col_names, verbose=1,
                                        nuisance_models=n, heterogeneity_model=h)
                    ca.fit(X, y)

                    self.assertEqual(ca.trained_feature_indices_, [0])  # only first column okay
                    self.assertEqual(ca.untrained_feature_indices_, [(1, 'upper_bound_on_cat_expansion'),
                                                                     (2, 'cat_limit'),
                                                                     (3, 'cat_limit'),
                                                                     (4, 'cat_limit'),
                                                                     (5, 'cat_limit')])

                    # increase bound on cat expansion
                    ca.upper_bound_on_cat_expansion = 6
                    ca.fit(X, y, warm_start=warm_start)

                    self.assertEqual(ca.trained_feature_indices_, [0, 1])  # second column okay also
                    self.assertEqual(ca.untrained_feature_indices_, [(2, 'cat_limit'),
                                                                     (3, 'cat_limit'),
                                                                     (4, 'cat_limit'),
                                                                     (5, 'cat_limit')])

                    # skip checks (reducing folds accordingly)
                    ca.skip_cat_limit_checks = True
                    ca.fit(X, y, warm_start=warm_start)

                    if h == 'linear':
                        self.assertEqual(ca.trained_feature_indices_, [0, 1, 2, 3, 4])  # all but last col okay
                        self.assertEqual(ca.untrained_feature_indices_, [(5, 'cat_limit')])
                    else:
                        self.assertEqual(ca.trained_feature_indices_, [0, 1, 2, 3])  # can't handle last two
                        self.assertEqual(ca.untrained_feature_indices_, [(4, 'cat_limit'),
                                                                         (5, 'cat_limit')])

    # Add tests that guarantee that the reliance on DML feature order is not broken, such as
    # Creare a transformer that zeros out all variables after the first n_x variables, so it zeros out W
    # Pass an example where W is irrelevant and X is confounder
    # As long as DML doesnt change the order of the inputs, then things should be good. Otherwise X would be
    # zeroed out and the test will fail

    def test_scaling_transforms(self):
        # shouldn't matter if X is scaled much larger or much smaller than W, we should still get good estimates
        n = 2000
        X = np.random.normal(size=(n, 5))
        W = np.random.normal(size=(n, 5))
        W[:, 0] = 1  # make one of the columns a constant
        xt, wt, xy, wy, theta = [np.random.normal(size=sz) for sz in [(5, 1), (5, 1), (5, 1), (5, 1), (1, 1)]]
        T = X @ xt + W @ wt + np.random.normal(size=(n, 1))
        Y = X @ xy + W @ wy + T @ theta
        arr1 = np.hstack([X, W, T])
        # rescaling X shouldn't affect the first stage models because they normalize the inputs
        arr2 = np.hstack([1000 * X, W, T])
        for hmodel in ['linear', 'forest']:
            inds = [-1]  # we just care about T
            cats = []
            hinds = list(range(X.shape[1]))
            ca = CausalAnalysis(inds, cats, hinds, heterogeneity_model=hmodel, random_state=123)
            ca.fit(arr1, Y)
            eff1 = ca.global_causal_effect()

            ca.fit(arr2, Y)
            eff2 = ca.global_causal_effect()

            np.testing.assert_allclose(eff1.point.values, eff2.point.values, rtol=1e-5)
            np.testing.assert_allclose(eff1.ci_lower.values, eff2.ci_lower.values, rtol=1e-5)
            np.testing.assert_allclose(eff1.ci_upper.values, eff2.ci_upper.values, rtol=1e-5)

            np.testing.assert_allclose(eff1.point.values, theta.flatten(), rtol=1e-2)

        # to recover individual coefficients with linear models, we need to be more careful in how we set up X to avoid
        # cross terms
        X = np.zeros(shape=(n, 5))
        X[range(X.shape[0]), np.random.choice(5, size=n)] = 1
        xt, wt, xy, wy, theta = [np.random.normal(size=sz) for sz in [(5, 1), (5, 1), (5, 1), (5, 1), (5, 1)]]
        T = X @ xt + W @ wt + np.random.normal(size=(n, 1))
        Y = X @ xy + W @ wy + T * (X @ theta)
        arr1 = np.hstack([X, W, T])
        arr2 = np.hstack([1000 * X, W, T])
        for hmodel in ['linear', 'forest']:
            inds = [-1]  # we just care about T
            cats = []
            hinds = list(range(X.shape[1]))
            ca = CausalAnalysis(inds, cats, hinds, heterogeneity_model=hmodel, random_state=123)
            ca.fit(arr1, Y)
            eff1 = ca.global_causal_effect()
            loc1 = ca.local_causal_effect(
                np.hstack([np.eye(X.shape[1]), np.zeros((X.shape[1], arr1.shape[1] - X.shape[1]))]))
            ca.fit(arr2, Y)
            eff2 = ca.global_causal_effect()
            loc2 = ca.local_causal_effect(
                # scale by 1000 to match the input to this model:
                # the scale of X does matter for the final model, which keeps results in user-denominated units
                1000 * np.hstack([np.eye(X.shape[1]), np.zeros((X.shape[1], arr1.shape[1] - X.shape[1]))]))

            # rescaling X still shouldn't affect the first stage models
            np.testing.assert_allclose(eff1.point.values, eff2.point.values, rtol=1e-5)
            np.testing.assert_allclose(eff1.ci_lower.values, eff2.ci_lower.values, rtol=1e-5)
            np.testing.assert_allclose(eff1.ci_upper.values, eff2.ci_upper.values, rtol=1e-5)

            np.testing.assert_allclose(loc1.point.values, loc2.point.values, rtol=1e-2)

            # TODO: we don't recover the correct values with enough accuracy to enable this assertion
            #       is there a different way to verify that we are learning the correct coefficients?

            # np.testing.assert_allclose(loc1.point.values, theta.flatten(), rtol=1e-1)
