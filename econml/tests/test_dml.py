# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import pytest
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from econml.dml import DMLCateEstimator, SparseLinearDMLCateEstimator
import numpy as np
from econml.utilities import shape, hstack, vstack, reshape, cross_product


# all solutions to underdetermined (or exactly determined) Ax=b are given by A⁺b+(I-A⁺A)w for some arbitrary w
# note that if Ax=b is overdetermined, this will raise an assertion error
def rand_sol(A, b):
    """Generate a random solution to the equation Ax=b."""
    assert np.linalg.matrix_rank(A) <= len(b)
    A_plus = np.linalg.pinv(A)
    x = A_plus @ b
    return x + (np.eye(x.shape[0]) - A_plus @ A) @ np.random.normal(size=x.shape)


class TestDML(unittest.TestCase):

    def test_cate_api(self):
        """Test that we correctly implement the CATE API."""
        d_w = 2
        d_x = 2
        d_y = 2
        d_t = 2
        n = 20
        with self.subTest(d_w=d_w, d_x=d_x, d_y=d_y, d_t=d_t):
            W, X, Y, T = [np.random.normal(size=(n, d)) for d in [d_w, d_x, d_y, d_t]]

            dml = DMLCateEstimator(model_y=LinearRegression(), model_t=LinearRegression())
            dml.fit(Y, T, X, W)
            # just make sure we can call the marginal_effect and effect methods
            dml.marginal_effect(None, X)
            dml.effect(0, T, X)

    def test_can_use_vectors(self):
        """Test that we can pass vectors for T and Y (not only 2-dimensional arrays)."""
        dml = DMLCateEstimator(LinearRegression(), LinearRegression(), featurizer=FunctionTransformer())
        dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))
        self.assertAlmostEqual(dml.coef_.reshape(())[()], 1)

    @staticmethod
    def _generate_recoverable_errors(a_X, X, a_W=None, W=None, featurizer=FunctionTransformer()):
        """Return error vectors e_t and e_y such that OLS can recover the true coefficients from both stages."""
        if W is None:
            W = np.empty((shape(X)[0], 0))
        if a_W is None:
            a_W = np.zeros((shape(W)[1],))
        # to correctly recover coefficients for T via OLS, we need e_t to be orthogonal to [W;X]
        WX = hstack([W, X])
        e_t = rand_sol(WX.T, np.zeros((shape(WX)[1],)))

        # to correctly recover coefficients for Y via OLS, we need ([X; W]⊗[1; ϕ(X); W])⁺ e_y =
        #                                                          -([X; W]⊗[1; ϕ(X); W])⁺ ((ϕ(X)⊗e_t)a_X+(W⊗e_t)a_W)
        # then, to correctly recover a in the third stage, we additionally need (ϕ(X)⊗e_t)ᵀ e_y = 0

        ϕ = featurizer.fit_transform(X)

        v_X = cross_product(ϕ, e_t)
        v_W = cross_product(W, e_t)

        M = np.linalg.pinv(cross_product(WX, hstack([np.ones((shape(WX)[0], 1)), ϕ, W])))
        e_y = rand_sol(vstack([M, v_X.T]), vstack([-M @ (v_X @ a_X + v_W @ a_W), np.zeros((shape(v_X)[1],))]))

        return e_t, e_y

    # TODO: it seems like roughly 20% of the calls to _test_sparse are failing - find out what's going wrong
    @pytest.mark.xfail
    def test_sparse(self):
        for _ in range(5):
            n_p = np.random.randint(2, 5)  # 2 to 4 products
            d_w = np.random.randint(0, 5)  # random number of covariates
            min_n = np.ceil(2 + d_w * (1 + (d_w + 1) / n_p))  # minimum number of rows per product
            n_r = np.random.randint(min_n, min_n + 3)
            with self.subTest(n_p=n_p, d_w=d_w, n_r=n_r):
                TestDML._test_sparse(n_p, d_w, n_r)

    # sparse test case: heterogeneous effect by product
    @staticmethod
    def _test_sparse(n_p, d_w, n_r):
        # need at least as many rows in e_y as there are distinct columns
        # in [X;X⊗W;W⊗W;X⊗e_t] to find a solution for e_t
        assert n_p * n_r >= 2 * n_p + n_p * d_w + d_w * (d_w + 1) / 2
        a = np.random.normal(size=(n_p,))  # one effect per product
        n = n_p * n_r
        p = np.tile(range(n_p), n_r)  # product id

        b = np.random.normal(size=(d_w + n_p,))
        g = np.random.normal(size=(d_w + n_p,))

        x = np.empty((2 * n, n_p))  # product dummies
        w = np.empty((2 * n, d_w))
        y = np.empty(2 * n)
        t = np.empty(2 * n)

        for fold in range(0, 2):
            x_f = OneHotEncoder().fit_transform(np.reshape(p, (-1, 1))).toarray()
            w_f = np.random.normal(size=(n, d_w))
            xw_f = hstack([x_f, w_f])
            e_t_f, e_y_f = TestDML._generate_recoverable_errors(a, x_f, W=w_f)

            t_f = xw_f @ b + e_t_f
            y_f = t_f * np.choose(p, a) + xw_f @ g + e_y_f

            x[fold * n:(fold + 1) * n, :] = x_f
            w[fold * n:(fold + 1) * n, :] = w_f
            y[fold * n:(fold + 1) * n] = y_f
            t[fold * n:(fold + 1) * n] = t_f

        dml = SparseLinearDMLCateEstimator(LinearRegression(fit_intercept=False), LinearRegression(
            fit_intercept=False), featurizer=FunctionTransformer())
        dml.fit(y, t, x, w)

        # note that this would fail for the non-sparse DMLCateEstimator

        np.testing.assert_allclose(a, dml.coef_.reshape(-1))
        eff = reshape(t * np.choose(np.tile(p, 2), a), (-1, 1))
        np.testing.assert_allclose(eff, dml.effect(0, t, x))
