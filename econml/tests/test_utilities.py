# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import time
import random
import warnings
import numpy as np
import sparse as sp
import scipy.sparse
import pytest
from econml.utilities import (check_high_dimensional, einsum_sparse, todense, tocoo, transpose,
                              inverse_onehot, cross_product, transpose_dictionary, deprecated, _deprecate_positional,
                              strata_from_discrete_arrays, add_constant, MultiModelWrapper, SeparateModel)
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from sklearn.linear_model import LinearRegression, LogisticRegressionCV, LassoCV


class TestUtilities(unittest.TestCase):
    def test_check_high_dimensional(self):
        X = np.repeat(
            a=np.expand_dims(np.arange(3), axis=1),
            repeats=2,
            axis=1,
        )
        T = np.expand_dims(np.arange(3), axis=1)

        check_high_dimensional(X=X, T=T, threshold=0, featurizer=SplineTransformer())

    def test_cross_product(self):
        X = np.array([[1, 2],
                      [3, 4]])
        Y = np.array([[1, 2, 3],
                      [4, 5, 6]])
        Z = np.array([1,
                      1])

        # make sure cross product varies more slowly with first array
        # and that vectors are okay as inputs
        assert np.all(cross_product(Z, Y, X) == np.array([[1, 2, 3, 2, 4, 6],
                                                          [12, 15, 18, 16, 20, 24]]))

        assert np.all(cross_product(X, Z, Y) == np.array([[1, 2, 2, 4, 3, 6],
                                                          [12, 16, 15, 20, 18, 24]]))

        ()

    def test_einsum_errors(self):
        # number of inputs in specification must match number of inputs
        with self.assertRaises(Exception):
            einsum_sparse('abc,def->ad', tocoo(np.ones((1, 2, 3))))
        with self.assertRaises(Exception):
            einsum_sparse('abc->a', tocoo(np.ones((1, 2, 3))), tocoo(np.ones((1, 2, 3))))

        # must have an output
        with self.assertRaises(Exception):
            einsum_sparse('abc', tocoo(np.ones((1, 2, 3))))

        # output indices must be unique
        with self.assertRaises(Exception):
            einsum_sparse('abc->bb', tocoo(np.ones((1, 2, 3))))

        # output indices must be present in an input
        with self.assertRaises(Exception):
            einsum_sparse('abc->bd', tocoo(np.ones((1, 2, 3))))

        # number of indices must match number of dimensions for each input
        with self.assertRaises(Exception):
            einsum_sparse('ab->a', tocoo(np.ones((1, 2, 3))))
        with self.assertRaises(Exception):
            einsum_sparse('abcd->a', tocoo(np.ones((1, 2, 3))), tocoo(np.ones((1, 2, 3))))

        # repeated indices must always have consistent sizes
        with self.assertRaises(Exception):
            einsum_sparse('aaa->a', tocoo(np.ones((1, 2, 3))))
        with self.assertRaises(Exception):
            einsum_sparse('abc,bac->a', tocoo(np.ones((1, 2, 3))), tocoo(np.ones((1, 2, 3))))

    def test_einsum_basic(self):
        # transpose
        arr = sp.random((20, 30, 40), 0.1)
        self.assertEqual((einsum_sparse('abc->cba', arr) != arr.transpose()).nnz, 0)

        # tensordot
        arr1 = sp.random((20, 30, 40), 0.1)
        arr2 = sp.random((40, 30), 0.1)
        arr3 = sp.random((40, 20, 10), 0.1)
        self.assertTrue(np.allclose(todense(einsum_sparse('abc,cb->a', arr1, arr2)),
                                    todense(sp.tensordot(arr1, arr2, axes=([1, 2], [1, 0])))))
        self.assertTrue(np.allclose(todense(einsum_sparse('ab,acd->bcd', arr2, arr3)),
                                    todense(sp.tensordot(arr2, arr3, axes=(0, 0)))))

        # trace
        arr = sp.random((100, 100), 0.1)
        self.assertAlmostEqual(einsum_sparse('aa->', arr)[()], np.trace(todense(arr)))

    def test_transpose_compatible(self):
        """Test that the results of `transpose` are compatible for sparse and dense arrays."""
        arr = tocoo(np.arange(27).reshape(3, 3, 3))
        np.testing.assert_array_equal(todense(transpose(arr, (1, 2, 0))), transpose(todense(arr), (1, 2, 0)))
        for _ in range(5):
            ndims = np.random.randint(2, 6)
            dims = tuple(np.random.randint(5, 20, size=ndims))
            axes = np.random.permutation(range(ndims))
            arr = sp.random(dims, density=0.1)
            out1 = todense(transpose(arr, axes))
            out2 = transpose(todense(arr), axes)
            np.testing.assert_allclose(out1, out2, verbose=True)

    def test_inverse_onehot(self):
        T = np.random.randint(4, size=100)
        T_oh = OneHotEncoder(categories='auto', sparse_output=False).fit_transform(T.reshape(-1, 1))[:, 1:]
        T_inv = inverse_onehot(T_oh)
        np.testing.assert_array_equal(T, T_inv)

    # TODO: set up proper flag for this
    @pytest.mark.slow
    def test_einsum_random(self):
        for _ in range(10):  # do 10 random tests
            num_arrs = random.randint(3, 5)  # use between 3 and 5 arrays as input
            arrs = [sp.random((20,) * random.randint(1, 5), 0.05) for _ in range(num_arrs)]
            # pick indices at random with replacement from the first 7 letters of the alphabet
            dims = [''.join(np.random.choice(list("abcdefg"), arr.ndim)) for arr in arrs]
            all_inds = set.union(*(set(inds) for inds in dims))
            # of all of the distinct indices that appear in any input,
            # pick a random subset of them (of size at most 5) to appear in the output
            output = ''.join(random.sample(sorted(all_inds), random.randint(0, min(len(all_inds), 5))))
            specification = ','.join(dims) + '->' + output
            with self.subTest(spec=specification):
                print(specification)
                start = time.perf_counter()
                spr = einsum_sparse(specification, *arrs)
                mid = time.perf_counter()
                der = np.einsum(specification, *[todense(arr) for arr in arrs])
                end = time.perf_counter()
                print(" sparse: {0}".format(mid - start))
                print(" dense:  {0}".format(end - mid))
                self.assertTrue(np.allclose(todense(spr),
                                            der))

    def test_transpose_dictionary(self):
        d1 = {1: {'a': '1a', 'b': '1b'}, 2: {'b': '2b', 'a': '2a', 'c': '2c'}}
        d2 = {'a': {1: '1a', 2: '2a'}, 'b': {2: '2b', 1: '1b'}, 'c': {2: '2c'}}
        assert d1 == transpose_dictionary(d2)
        assert d2 == transpose_dictionary(d1)

    def test_deprecated(self):

        @deprecated("This class is deprecated")
        class Deprecated:
            def __init__(self, a, b=1):
                self.sum = a + b

            def get_sum(self):
                return self.sum

        @deprecated("This method is deprecated", DeprecationWarning)
        def depr(x, *args, y=2):
            pass

        # creating an instance should warn
        with self.assertWarnsRegex(FutureWarning, "This class is deprecated"):
            instance = Deprecated(1)

        # using the instance should not warn
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert instance.get_sum() == 2

        # using the deprecated method should warn
        with self.assertWarnsRegex(DeprecationWarning, "This method is deprecated"):
            depr(1, 2, 3, y=4)

    def test_deprecate_positional(self):

        @_deprecate_positional("Don't pass b or c by position", ['b', 'c'])
        def m(a, b, c=1, *args, **kwargs):
            return a

        with self.assertWarnsRegex(FutureWarning, "Don't pass b or c by position"):
            m(1, 2)

        with self.assertWarnsRegex(FutureWarning, "Don't pass b or c by position"):
            m(1, 2, c=2)

        # don't warn if b and c are passed by keyword
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m(1, b=2)
            m(a=1, b=2)
            m(1, b=2, c=3, X='other')

    def test_single_strata_from_discrete_array(self):
        T = np.repeat([[0, 1, 2]], 4, axis=0).ravel()
        Z = np.repeat([[0, 1]], 6, axis=0).ravel()
        Y = np.repeat([0, 1], 6, axis=0)

        assert set(strata_from_discrete_arrays([T, Z, Y])) == set(np.arange(12))
        assert set(strata_from_discrete_arrays([T, Z])) == set(np.arange(6))
        assert set(strata_from_discrete_arrays([T])) == set(np.arange(3))
        assert strata_from_discrete_arrays([]) is None

    def test_add_constant(self):
        import pandas as pd
        from statsmodels.tools.tools import add_constant as sm_add_constant

        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 3))

        # Matches statsmodels for ndarray inputs.
        np.testing.assert_allclose(add_constant(X), sm_add_constant(X))
        np.testing.assert_allclose(add_constant(X, prepend=False),
                                   sm_add_constant(X, prepend=False))

        # 1D input is promoted to 2D and a constant column is added.
        v = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(add_constant(v),
                                      np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]))

        # 3D+ inputs are rejected.
        with self.assertRaises(ValueError):
            add_constant(np.zeros((2, 2, 2)))

        # has_constant policies on a column that is already constant.
        Xc = np.column_stack([np.ones(5), rng.standard_normal(5)])
        np.testing.assert_array_equal(add_constant(Xc, has_constant='skip'), Xc)
        with self.assertRaises(ValueError):
            add_constant(Xc, has_constant='raise')
        # 'add' should always prepend another ones column.
        out_add = add_constant(Xc, has_constant='add')
        assert out_add.shape == (5, 3)
        np.testing.assert_array_equal(out_add[:, 0], np.ones(5))

        # List input behaves like ndarray.
        np.testing.assert_array_equal(add_constant([[1.0, 2.0], [3.0, 4.0]]),
                                      np.array([[1.0, 1.0, 2.0], [1.0, 3.0, 4.0]]))

        # pandas DataFrame and Series inputs are accepted and produce
        # ndarrays (this differs from statsmodels, which preserves the
        # pandas type — see the docstring Notes section).
        df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
        out_df = add_constant(df)
        assert isinstance(out_df, np.ndarray)
        np.testing.assert_array_equal(out_df, np.array([[1.0, 1.0, 4.0],
                                                        [1.0, 2.0, 5.0],
                                                        [1.0, 3.0, 6.0]]))

        # Non-default index should not reorder the underlying values
        # (statsmodels behaves the same way).
        df_idx = pd.DataFrame({'a': [10.0, 20.0, 30.0]}, index=[7, 2, 5])
        np.testing.assert_array_equal(add_constant(df_idx),
                                      np.array([[1.0, 10.0], [1.0, 20.0], [1.0, 30.0]]))

        s = pd.Series([1.0, 2.0, 3.0], name='x')
        out_s = add_constant(s)
        assert isinstance(out_s, np.ndarray)
        np.testing.assert_array_equal(out_s, np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]))


class TestMultiModelWrapper(unittest.TestCase):

    @staticmethod
    def _encode_drop_first(T, K):
        out = np.zeros((T.shape[0], K - 1))
        for i in range(1, K):
            out[T == i, i - 1] = 1
        return out

    @staticmethod
    def _encode_full(T, K):
        out = np.zeros((T.shape[0], K))
        for i in range(K):
            out[T == i, i] = 1
        return out

    def test_drop_first_routes_rows_to_models(self):
        rng = np.random.default_rng(0)
        n, d, K = 90, 3, 3
        X = rng.normal(size=(n, d))
        T = rng.integers(0, K, size=n)
        # ground truth: arm k has slope k on the first feature
        Y = T * X[:, 0]
        Xt = np.hstack([X, self._encode_drop_first(T, K)])

        w = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
        )
        w.fit(Xt, Y)
        for k in range(K):
            self.assertAlmostEqual(float(w.models[k].coef_[0]), float(k), places=6)
        np.testing.assert_allclose(w.predict(Xt), Y, atol=1e-8)

    def test_full_encoding_routes_rows_to_models(self):
        rng = np.random.default_rng(1)
        n, d, K = 60, 2, 3
        X = rng.normal(size=(n, d))
        T = rng.integers(0, K, size=n)
        Y = (T + 1) * X[:, 0]
        Xt = np.hstack([X, self._encode_full(T, K)])

        w = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            encoding='full',
        )
        w.fit(Xt, Y)
        for k in range(K):
            self.assertAlmostEqual(float(w.models[k].coef_[0]), float(k + 1), places=6)
        np.testing.assert_allclose(w.predict(Xt), Y, atol=1e-8)

    def test_label_encoding_matches_drop_first(self):
        rng = np.random.default_rng(2)
        n, d, K = 80, 2, 3
        X = rng.normal(size=(n, d))
        T = rng.integers(0, K, size=n)
        Y = T * X[:, 0]
        Xt_lbl = np.hstack([X, T.reshape(-1, 1)])
        Xt_df = np.hstack([X, self._encode_drop_first(T, K)])

        w_lbl = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            encoding='label',
        )
        w_df = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
        )
        w_lbl.fit(Xt_lbl, Y)
        w_df.fit(Xt_df, Y)
        for k in range(K):
            np.testing.assert_allclose(w_lbl.models[k].coef_, w_df.models[k].coef_)
        np.testing.assert_allclose(w_lbl.predict(Xt_lbl), w_df.predict(Xt_df))

    def test_single_model_with_n_categories_clones(self):
        w = MultiModelWrapper(LinearRegression(fit_intercept=False), n_categories=4)
        self.assertEqual(w.n_categories, 4)
        self.assertEqual(len(w.models), 4)
        ids = {id(m) for m in w.models}
        self.assertEqual(len(ids), 4)  # genuine clones, not the same instance

    def test_single_model_without_n_categories_raises(self):
        with self.assertRaises(ValueError):
            MultiModelWrapper(LinearRegression())

    def test_mismatched_n_categories_raises(self):
        with self.assertRaises(ValueError):
            MultiModelWrapper(LinearRegression(), LinearRegression(), n_categories=3)

    def test_zero_models_raises(self):
        with self.assertRaises(ValueError):
            MultiModelWrapper()

    def test_invalid_encoding_raises(self):
        with self.assertRaises(ValueError):
            MultiModelWrapper(LinearRegression(), LinearRegression(), encoding='bogus')

    def test_too_few_columns_raises(self):
        w = MultiModelWrapper(LinearRegression(), LinearRegression(), LinearRegression())
        with self.assertRaises(ValueError):
            # K=3 with default drop_first needs >= 2 trailing one-hot columns
            w.fit(np.array([[1.0]]), np.array([0.0]))

    def test_sample_weight_is_forwarded(self):
        # With one heavily down-weighted point in arm 1, the fitted slope
        # should be determined by the other arm-1 point alone.
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        T = np.array([0, 0, 1, 1])
        Y = np.array([0.0, 0.0, 6.0, 100.0])  # (3, 6) -> slope 2; (4, 100) is noise
        Xt = np.hstack([X, T.reshape(-1, 1).astype(float)])
        sw = np.array([1.0, 1.0, 1.0, 1e-12])

        w = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            encoding='label',
        )
        w.fit(Xt, Y, sample_weight=sw)
        self.assertAlmostEqual(float(w.models[1].coef_[0]), 2.0, places=3)

    def test_integration_with_linear_drlearner_multinary(self):
        # End-to-end smoke test: drop-first MultiModelWrapper fed to LinearDRLearner
        # with 3 treatment categories (the case the old MultiModelWrapper couldn't handle).
        from econml.dr import LinearDRLearner
        rng = np.random.default_rng(3)
        n = 300
        X = rng.normal(size=(n, 2))
        T = rng.integers(0, 3, size=n)
        Y = X[:, 0] + T * (1 + X[:, 1]) + rng.normal(size=n)

        mdl = LinearDRLearner(
            model_regression=MultiModelWrapper(LassoCV(), n_categories=3),
            model_propensity=LogisticRegressionCV(max_iter=200),
        )
        mdl.fit(Y, T, X=X)
        effects = mdl.effect(X[:5], T0=0, T1=1)
        self.assertEqual(effects.shape, (5,))

    def test_sparse_input_drop_first(self):
        # csr_matrix with K=3, drop-first encoding: full matrix stays sparse,
        # only the trailing 2-column treatment block is densified internally.
        rng = np.random.default_rng(10)
        n, d, K = 60, 4, 3
        X = rng.normal(size=(n, d))
        T = rng.integers(0, K, size=n)
        Y = T * X[:, 0]
        Xt = np.hstack([X, self._encode_drop_first(T, K)])
        Xt_sparse = scipy.sparse.csr_matrix(Xt)

        w = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
        )
        w.fit(Xt_sparse, Y)
        for k in range(K):
            self.assertAlmostEqual(float(w.models[k].coef_[0]), float(k), places=6)
        np.testing.assert_allclose(w.predict(Xt_sparse), Y, atol=1e-8)

    def test_sparse_input_full_encoding(self):
        rng = np.random.default_rng(11)
        n, d, K = 50, 3, 2
        X = rng.normal(size=(n, d))
        T = rng.integers(0, K, size=n)
        Y = (T + 1) * X[:, 0]
        Xt = np.hstack([X, self._encode_full(T, K)])
        Xt_sparse = scipy.sparse.csr_matrix(Xt)

        w = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            encoding='full',
        )
        w.fit(Xt_sparse, Y)
        for k in range(K):
            self.assertAlmostEqual(float(w.models[k].coef_[0]), float(k + 1), places=6)
        np.testing.assert_allclose(w.predict(Xt_sparse), Y, atol=1e-8)

    def test_sparse_input_label_encoding(self):
        rng = np.random.default_rng(12)
        n, d, K = 70, 2, 3
        X = rng.normal(size=(n, d))
        T = rng.integers(0, K, size=n)
        Y = T * X[:, 0]
        Xt = np.hstack([X, T.reshape(-1, 1).astype(float)])
        Xt_sparse = scipy.sparse.csr_matrix(Xt)

        w = MultiModelWrapper(
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            LinearRegression(fit_intercept=False),
            encoding='label',
        )
        w.fit(Xt_sparse, Y)
        for k in range(K):
            self.assertAlmostEqual(float(w.models[k].coef_[0]), float(k), places=6)
        np.testing.assert_allclose(w.predict(Xt_sparse), Y, atol=1e-8)

    def test_model_list_kwarg_is_deprecated(self):
        # Old API: MultiModelWrapper(model_list=[...]) should still work but
        # emit a FutureWarning and produce a wrapper equivalent to the new
        # positional-args form.
        with self.assertWarnsRegex(FutureWarning, "model_list"):
            w = MultiModelWrapper(
                model_list=[LinearRegression(fit_intercept=False),
                            LinearRegression(fit_intercept=False)],
            )
        self.assertEqual(w.n_categories, 2)
        # Smoke-fit to make sure the wrapper is fully functional. Default
        # encoding is 'drop_first', so we one-hot-drop-first the binary T.
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        T = np.array([0, 0, 1, 1])
        Y = np.array([0.0, 0.0, 6.0, 8.0])
        Xt = np.hstack([X, (T == 1).reshape(-1, 1).astype(float)])
        w.fit(Xt, Y)
        np.testing.assert_allclose(w.predict(Xt), Y, atol=1e-8)

    def test_positional_list_is_deprecated(self):
        # Old API: MultiModelWrapper([m1, m2, ...]) should warn and unpack.
        with self.assertWarnsRegex(FutureWarning, "single positional argument"):
            w = MultiModelWrapper(
                [LinearRegression(fit_intercept=False),
                 LinearRegression(fit_intercept=False),
                 LinearRegression(fit_intercept=False)],
            )
        self.assertEqual(w.n_categories, 3)

    def test_model_list_and_positional_models_together_raises(self):
        with self.assertRaises(ValueError):
            MultiModelWrapper(
                LinearRegression(),
                model_list=[LinearRegression(), LinearRegression()],
            )


class TestSeparateModelDeprecation(unittest.TestCase):

    def test_separate_model_emits_future_warning(self):
        with self.assertWarnsRegex(FutureWarning, "SeparateModel is deprecated"):
            SeparateModel(LinearRegression(), LinearRegression())
