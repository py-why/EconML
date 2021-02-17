# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import logging
import time
import random
import numpy as np
import sparse as sp
import pytest
from econml.utilities import (einsum_sparse, todense, tocoo, transpose,
                              inverse_onehot, cross_product, transpose_dictionary, deprecated, _deprecate_positional)
from sklearn.preprocessing import OneHotEncoder


class TestUtilities(unittest.TestCase):

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
        T_oh = OneHotEncoder(categories='auto', sparse=False).fit_transform(T.reshape(-1, 1))[:, 1:]
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
            output = ''.join(random.sample(all_inds, random.randint(0, min(len(all_inds), 5))))
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
        with pytest.warns(None) as counter:
            assert instance.get_sum() == 2
        assert not counter

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
        with pytest.warns(None) as counter:
            m(1, b=2)
            m(a=1, b=2)
            m(1, b=2, c=3, X='other')
        assert not counter
