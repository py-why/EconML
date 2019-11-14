# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for two stage least squares module."""

import unittest
import numpy as np
import warnings
import pytest

from econml.utilities import shape, reshape
from econml.two_stage_least_squares import NonparametricTwoStageLeastSquares, HermiteFeatures
from sklearn.linear_model import LinearRegression


class Test2SLS(unittest.TestCase):

    def test_hermite_shape(self):
        for d, s in [(3, 0), (4, 2)]:
            for j in [True, False]:
                for n, x in [(5, 1), (7, 3)]:
                    last_dim = (d + 1)**x if j else (d + 1) * x
                    correct_shape = (n,) + (x,) * s + (last_dim,)
                    output_shape = shape(HermiteFeatures(d, s, j).fit_transform(np.zeros((n, x))))
                    assert output_shape == correct_shape

    def test_hermite_results(self):
        inputs = np.random.normal(size=(5, 1))
        hf = HermiteFeatures(3).fit_transform(inputs)
        # first polynomials are 1, x, x*x-1, x*x*x-3*x
        ones = np.ones(shape(inputs))
        polys = np.hstack([ones, inputs, inputs * inputs - ones, inputs * inputs * inputs - 3 * inputs])
        assert(np.allclose(hf, polys * np.exp(-inputs * inputs / 2)))

        for j in [True, False]:
            hf = HermiteFeatures(1, shift=1, joint=j).fit_transform(inputs)
            # first derivatives are -x, -x^2+1 (since there's just one column, joint-ness doesn't matter)
            polys = np.hstack([-inputs, -inputs * inputs + ones])
            assert(np.allclose(hf, reshape(polys * np.exp(-inputs * inputs / 2), (5, 1, 2))))

    @pytest.mark.slow
    def test_hermite_approx(self):
        n = 50000
        x = np.random.uniform(low=-0.5, high=1.5, size=(n, 2))
        y = (x[:, 1] > x[:, 0]) * \
            (0 <= x[:, 0]) * \
            (x[:, 0] <= 1) * \
            (0 <= x[:, 1]) * \
            (x[:, 1] <= 1) + \
            np.random.normal(0, 0.01, (n,))

        def err(k, j):
            hf = HermiteFeatures(k, joint=j)
            m = LinearRegression()
            m.fit(hf.fit_transform(x[:n // 2, :]), y[:n // 2])
            return ((y[n // 2:] - m.predict(hf.fit_transform(x[n // 2:, :])))**2).mean()
        print([(k, j, err(k, j)) for k in range(2, 15) for j in [False, True]])

    def test_2sls_shape(self):
        pass

    # TODO: this tests that we can run the method; how do we test that the results are reasonable?
    def test_2sls(self):
        n = 50000
        d_w = 2
        d_z = 1
        d_x = 1
        d_t = 1
        d_y = 1
        e = np.random.uniform(low=-0.5, high=0.5, size=(n, d_x))
        z = np.random.uniform(size=(n, 1))
        w = np.random.uniform(size=(n, d_w))
        a = np.random.normal(size=(d_w, d_t))
        b = np.random.normal(size=(d_w, d_y))
        x = np.random.uniform(size=(n, d_x)) + e
        p = x + z * e + w@a + np.random.uniform(size=(n, d_t))
        y = p * x + e + w@b

        losses = []
        marg_effs = []

        z_fresh = np.random.uniform(size=(n, d_z))
        e_fresh = np.random.uniform(low=-0.5, high=0.5, size=(n, d_x))
        x_fresh = np.random.uniform(size=(n, d_x)) + e_fresh
        w_fresh = np.random.uniform(size=(n, d_w))
        p_fresh = x_fresh + z_fresh * e_fresh + np.random.uniform(size=(n, d_t))

        for (dt, dx, dz) in [(0, 0, 0), (1, 1, 1), (5, 5, 5), (10, 10, 10), (3, 3, 10), (10, 10, 3)]:
            np2sls = NonparametricTwoStageLeastSquares(HermiteFeatures(
                dt), HermiteFeatures(dx), HermiteFeatures(dz), HermiteFeatures(dt, shift=1))
            np2sls.fit(y, p, x, w, z)
            effect = np2sls.effect(x_fresh, np.zeros(shape(p_fresh)), p_fresh)
            losses.append(np.mean(np.square(p_fresh * x_fresh - effect)))
            marg_effs.append(np2sls.marginal_effect(np.array([[0.3], [0.5], [0.7]]), np.array([[0.4], [0.6], [0.2]])))
        print("losses: {}".format(losses))
        print("marg_effs: {}".format(marg_effs))
