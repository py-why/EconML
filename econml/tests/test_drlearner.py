# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
import pytest
from sklearn.base import TransformerMixin
from numpy.random import normal, multivariate_normal, binomial
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from econml.drlearner import DRLearner, LinearDRLearner
from econml.utilities import shape, hstack, vstack, reshape, cross_product
from econml.inference import BootstrapInference
from contextlib import ExitStack


# all solutions to underdetermined (or exactly determined) Ax=b are given by A⁺b+(I-A⁺A)w for some arbitrary w
# note that if Ax=b is overdetermined, this will raise an assertion error
def rand_sol(A, b):
    """Generate a random solution to the equation Ax=b."""
    assert np.linalg.matrix_rank(A) <= len(b)
    A_plus = np.linalg.pinv(A)
    x = A_plus @ b
    return x + (np.eye(x.shape[0]) - A_plus @ A) @ np.random.normal(size=x.shape)


class TestDRLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set random seed
        cls.random_state = np.random.RandomState(12345)
        # Generate data
        # DGP constants
        cls.d = 5
        cls.n = 1000
        cls.n_test = 200
        cls.beta = np.array([0.25, -0.38, 1.41, 0.50, -1.22])
        cls.heterogeneity_index = 1
        # Test data
        cls.X_test = cls.random_state.multivariate_normal(
            np.zeros(cls.d),
            np.diag(np.ones(cls.d)),
            cls.n_test)
        # Constant treatment effect and propensity
        cls.const_te_data = TestDRLearner._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestDRLearner._const_te,
            propensity=lambda x: 0.3)
        # Heterogeneous treatment and propensity
        cls.heterogeneous_te_data = TestDRLearner._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestDRLearner._heterogeneous_te,
            propensity=lambda x: (0.8 if (x[2] > -0.5 and x[2] < 0.5) else 0.2))

    def test_cate_api(self):
        """Test that we correctly implement the CATE API."""
        n = 20

        def make_random(is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d > 0 else (n,)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b', 'c'], size=sz)
                    # ensure that we've got at least two of every element
                    _, counts = np.unique(arr, return_counts=True)
                    if len(counts) == 3 and counts.min() > 1:
                        return arr
            else:
                return np.random.normal(size=sz)

        d_y = 0
        is_discrete = True
        d_t = 1
        for d_x in [2, None]:
            for d_w in [2, None]:
                W, X, Y, T = [make_random(is_discrete, d)
                              for is_discrete, d in [(False, d_w),
                                                     (False, d_x),
                                                     (False, d_y),
                                                     (is_discrete, d_t)]]

                if (X is None) and (W is None):
                    continue
                d_t_final = 2 if is_discrete else d_t

                effect_shape = (n,) + ((d_y,) if d_y > 0 else ())
                marginal_effect_shape = ((n,) +
                                         ((d_y,) if d_y > 0 else ()) +
                                         ((d_t_final,) if d_t_final > 0 else ()))

                # since T isn't passed to const_marginal_effect, defaults to one row if X is None
                const_marginal_effect_shape = ((n if d_x else 1,) +
                                               ((d_y,) if d_y > 0 else ()) +
                                               ((d_t_final,) if d_t_final > 0 else()))

                # TODO: add stratification to bootstrap so that we can use it even with discrete treatments
                infs = [None, 'statsmodels']

                est = LinearDRLearner(model_regression=Lasso(),
                                      model_propensity=LogisticRegression(C=1000, solver='lbfgs'))

                for inf in infs:
                    with self.subTest(d_w=d_w, d_x=d_x, d_y=d_y, d_t=d_t,
                                      is_discrete=is_discrete, est=est, inf=inf):
                        est.fit(Y, T, X, W, inference=inf)
                        # make sure we can call the marginal_effect and effect methods
                        const_marg_eff = est.const_marginal_effect(X)
                        marg_eff = est.marginal_effect(T, X)
                        self.assertEqual(shape(marg_eff), marginal_effect_shape)
                        self.assertEqual(shape(const_marg_eff), const_marginal_effect_shape)

                        np.testing.assert_array_equal(
                            marg_eff if d_x else marg_eff[0:1], const_marg_eff)

                        T0 = np.full_like(T, 'a')
                        eff = est.effect(X, T0=T0, T1=T)
                        self.assertEqual(shape(eff), effect_shape)
                        if inf is not None:
                            const_marg_eff_int = est.const_marginal_effect_interval(X)
                            marg_eff_int = est.marginal_effect_interval(T, X)
                            self.assertEqual(shape(marg_eff_int),
                                             (2,) + marginal_effect_shape)
                            self.assertEqual(shape(const_marg_eff_int),
                                             (2,) + const_marginal_effect_shape)
                            self.assertEqual(shape(est.effect_interval(X, T0=T0, T1=T)),
                                             (2,) + effect_shape)

                        est.score(Y, T, X, W)

                        # make sure we can call effect with implied scalar treatments, no matter the
                        # dimensions of T, and also that we warn when there are multiple treatments
                        if d_t > 1:
                            cm = self.assertWarns(Warning)
                        else:
                            cm = ExitStack()  # ExitStack can be used as a "do nothing" ContextManager
                        with cm:
                            effect_shape2 = (n if d_x else 1,) + ((d_y,) if d_y > 0 else())
                            eff = est.effect(X, T0='a', T1='b')
                            self.assertEqual(shape(eff), effect_shape2)

    def test_can_use_vectors(self):
        """Test that we can pass vectors for T and Y (not only 2-dimensional arrays)."""
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(C=1000, solver='lbfgs'),
                              fit_cate_intercept=False,
                              featurizer=FunctionTransformer())
        dml.fit(np.array([1, 2, 1, 2]), np.array([1, 2, 1, 2]), X=np.ones((4, 1)))
        self.assertAlmostEqual(dml.coef(T=2).reshape(())[()], 1)

    def test_can_use_sample_weights(self):
        """Test that we can pass sample weights to an estimator."""
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(C=1000, solver='lbfgs'),
                              featurizer=FunctionTransformer())
        dml.fit(np.array([1, 2, 1, 2]), np.array([1, 2, 1, 2]), W=np.ones((4, 1)),
                sample_weight=np.ones((4, )))
        self.assertAlmostEqual(dml.intercept(T=2), 1)

    def test_discrete_treatments(self):
        """Test that we can use discrete treatments"""
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(C=1000, solver='lbfgs'),
                              featurizer=FunctionTransformer())
        # create a simple artificial setup where effect of moving from treatment
        #     1 -> 2 is 2,
        #     1 -> 3 is 1, and
        #     2 -> 3 is -1 (necessarily, by composing the previous two effects)
        # Using an uneven number of examples from different classes,
        # and having the treatments in non-lexicographic order,
        # Should rule out some basic issues.
        dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array([3, 2, 1, 2, 3, 1, 1, 1]), np.ones((8, 1)))
        np.testing.assert_almost_equal(dml.effect(np.ones((9, 1)),
                                                  T0=np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
                                                  T1=np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])),
                                       [0, 2, 1, -2, 0, -1, -1, 1, 0])
        dml.score(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array([3, 2, 1, 2, 3, 1, 1, 1]), np.ones((8, 1)))

    def test_can_custom_splitter(self):
        # test that we can fit with a KFold instance
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(C=1000, solver='lbfgs'),
                              n_splits=KFold())
        dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))
        dml.score(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))

        # test that we can fit with a train/test iterable
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(C=1000, solver='lbfgs'),
                              n_splits=[([0, 1, 2], [3, 4, 5])])
        dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))
        dml.score(np.array([1, 2, 3, 1, 2, 3]), np.array([1, 2, 3, 1, 2, 3]), np.ones((6, 1)))

    def test_can_use_statsmodel_inference(self):
        """Test that we can use statsmodels to generate confidence intervals"""
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(C=1000, solver='lbfgs'))
        dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array(
            [3, 2, 1, 2, 3, 1, 1, 1]), np.ones((8, 1)), inference='statsmodels')
        interval = dml.effect_interval(np.ones((9, 1)),
                                       T0=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                       T1=np.array([2, 2, 3, 2, 2, 3, 2, 2, 3]),
                                       alpha=0.05)
        point = dml.effect(np.ones((9, 1)),
                           T0=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                           T1=np.array([2, 2, 3, 2, 2, 3, 2, 2, 3]))
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        assert (lo < hi).any()  # for at least some of the examples, the CI should have nonzero width

        interval = dml.const_marginal_effect_interval(np.ones((9, 1)), alpha=0.05)
        point = dml.const_marginal_effect(np.ones((9, 1)))
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        assert (lo < hi).any()  # for at least some of the examples, the CI should have nonzero width

        interval = dml.coef_interval(T=2, alpha=0.05)
        point = dml.coef(T=2)
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        assert (lo < hi).any()  # for at least some of the examples, the CI should have nonzero width

    def test_drlearner_all_attributes(self):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        import scipy.special
        np.random.seed(123)
        controls = np.random.normal(size=(5000, 3))
        T = np.random.binomial(2, scipy.special.expit(controls[:, 0]))
        sigma = 0.01
        y = (1 + .5 * controls[:, 0]) * T + controls[:, 0] + np.random.normal(0, sigma, size=(5000,))
        for X in [controls]:
            for W in [None, controls]:
                for featurizer in [None, PolynomialFeatures(degree=2, include_bias=False)]:
                    for first_stage in [(GradientBoostingClassifier(), GradientBoostingRegressor()),
                                        (LogisticRegression(), LinearRegression())]:
                        for final_stage in [LinearRegression(), GradientBoostingRegressor()]:
                            for multitask_model_final in [False, True]:
                                est = DRLearner(model_propensity=first_stage[0],
                                                model_regression=first_stage[1],
                                                model_final=final_stage,
                                                featurizer=featurizer,
                                                multitask_model_final=multitask_model_final)
                                if (X is None) and (W is None):
                                    with pytest.raises(AttributeError) as e_info:
                                        est.fit(y, T, X=X, W=W)
                                    continue
                                est.fit(y, T, X=X, W=W)
                                np.testing.assert_allclose(est.effect(X[:3], T0=0, T1=1), 1 + .5 * X[:3, 0],
                                                           rtol=0, atol=.2)
                                np.testing.assert_allclose(est.const_marginal_effect(X[:3]),
                                                           np.hstack([1 + .5 * X[:3, [0]], 2 * (1 + .5 * X[:3, [0]])]),
                                                           rtol=0, atol=.2)
                                for t in [1, 2]:
                                    np.testing.assert_allclose(est.marginal_effect(t, X[:3]),
                                                               np.hstack(
                                                                   [1 + .5 * X[:3, [0]], 2 * (1 + .5 * X[:3, [0]])]),
                                                               rtol=0, atol=.2)
                                assert isinstance(est.score_, float)
                                assert isinstance(est.score(y, T, X=X, W=W), float)

                                feat_names = ['A', 'B', 'C']
                                out_feat_names = feat_names
                                if featurizer is not None:
                                    out_feat_names = featurizer.fit_transform(X).get_feature_names(feat_names)
                                    np.testing.assert_array_equal(est.featurizer.n_input_features_, 3)
                                np.testing.assert_array_equal(est.cate_feature_names(feat_names), out_feat_names)

                                if isinstance(first_stage[0], GradientBoostingClassifier):
                                    np.testing.assert_array_equal(np.array([mdl.feature_importances_ for mdl in est.models_regression]).shape,
                                                                  [2, 2 + X.shape[1] + (W.shape[1] if W is not None else 0)])
                                    np.testing.assert_array_equal(np.array([mdl.feature_importances_ for mdl in est.models_propensity]).shape,
                                                                  [2, X.shape[1] + (W.shape[1] if W is not None else 0)])
                                else:
                                    np.testing.assert_array_equal(np.array([mdl.coef_ for mdl in est.models_regression]).shape,
                                                                  [2, 2 + X.shape[1] + (W.shape[1] if W is not None else 0)])
                                    np.testing.assert_array_equal(np.array([mdl.coef_ for mdl in est.models_propensity]).shape,
                                                                  [2, X.shape[1] + (W.shape[1] if W is not None else 0)])
                                if multitask_model_final:
                                    if isinstance(final_stage, GradientBoostingRegressor):
                                        np.testing.assert_equal(np.argsort(
                                            est.multitask_model_cate.feature_importances_)[-1], 0)
                                    else:
                                        np.testing.assert_array_almost_equal(
                                            est.multitask_model_cate.coef_, [[.5, 0, 0], [1, 0, 0]], decimal=1)
                                        np.testing.assert_almost_equal(
                                            est.multitask_model_cate.intercept_, [1, 2], decimal=1)
                                else:
                                    for t in [1, 2]:
                                        if isinstance(final_stage, GradientBoostingRegressor):
                                            np.testing.assert_equal(np.argsort(
                                                est.model_cate(T=t).feature_importances_)[-1], 0)
                                        else:
                                            np.testing.assert_array_almost_equal(
                                                est.model_cate(T=t).coef_, [.5 * t, 0, 0], decimal=1)
                                            np.testing.assert_almost_equal(
                                                est.model_cate(T=t).intercept_, t, decimal=1)

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

    def test_DRLearner(self):
        """Tests whether the DRLearner can accurately estimate constant and
           heterogeneous treatment effects.
        """
        DR_learner = DRLearner(model_regression=LinearRegression(),
                               model_final=LinearRegression())
        # Test inputs
        # self._test_inputs(DR_learner)
        # Test constant treatment effect
        self._test_te(DR_learner, tol=0.5, te_type="const")
        # Test heterogeneous treatment effect
        outcome_model = Pipeline([('poly', PolynomialFeatures()), ('model', LinearRegression())])
        DR_learner = DRLearner(model_regression=outcome_model,
                               model_final=LinearRegression())
        self._test_te(DR_learner, tol=0.5, te_type="heterogeneous")
        # Test heterogenous treatment effect for W =/= None
        self._test_with_W(DR_learner, tol=0.5)

    def _test_te(self, learner_instance, tol, te_type="const"):
        if te_type not in ["const", "heterogeneous"]:
            raise ValueError("Type of treatment effect must be 'const' or 'heterogeneous'.")
        X, T, Y = getattr(TestDRLearner, "{te_type}_te_data".format(te_type=te_type))
        te_func = getattr(TestDRLearner, "_{te_type}_te".format(te_type=te_type))
        # Fit learner and get the effect
        learner_instance.fit(Y, T, X)
        te_hat = learner_instance.effect(TestDRLearner.X_test)
        # Get the true treatment effect
        te = np.apply_along_axis(te_func, 1, TestDRLearner.X_test)
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)

    def _test_with_W(self, learner_instance, tol):
        # Only for heterogeneous TE
        X, T, Y = TestDRLearner.heterogeneous_te_data
        # Fit learner on X and W and get the effect
        learner_instance.fit(Y, T, X=X[:, [TestDRLearner.heterogeneity_index]], W=X)
        te_hat = learner_instance.effect(TestDRLearner.X_test[:, [TestDRLearner.heterogeneity_index]])
        # Get the true treatment effect
        te = np.apply_along_axis(TestDRLearner._heterogeneous_te, 1, TestDRLearner.X_test)
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)

    def _test_inputs(self, learner_instance):
        X, T, Y = TestDRLearner.const_te_data
        # Check that one can pass in regular lists
        learner_instance.fit(list(Y), list(T), list(X))
        learner_instance.effect(list(TestDRLearner.X_test))
        # Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, learner_instance.fit, Y, T, X[:TestDRLearner.n // 2])
        self.assertRaises(ValueError, learner_instance.fit, Y[:TestDRLearner.n // 2], T, X)
        # Check that it fails when T contains values other than 0 and 1
        self.assertRaises(ValueError, learner_instance.fit, Y, T + 1, X)
        # Check that it works when T, Y have shape (n, 1)
        self.assertWarns(DataConversionWarning,
                         learner_instance.fit, Y.reshape(-1, 1), T.reshape(-1, 1), X
                         )

    @classmethod
    def _untreated_outcome(cls, x):
        return np.dot(x, cls.beta) + cls.random_state.normal(0, 1)

    @classmethod
    def _const_te(cls, x):
        return 2

    @classmethod
    def _heterogeneous_te(cls, x):
        return x[cls.heterogeneity_index]

    @classmethod
    def _generate_data(cls, n, d, untreated_outcome, treatment_effect, propensity):
        """Generates population data for given untreated_outcome, treatment_effect and propensity functions.

        Parameters
        ----------
            n (int): population size
            d (int): number of covariates
            untreated_outcome (func): untreated outcome conditional on covariates
            treatment_effect (func): treatment effect conditional on covariates
            propensity (func): probability of treatment conditional on covariates
        """
        # Generate covariates
        X = cls.random_state.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), n)
        # Generate treatment
        T = np.apply_along_axis(lambda x: cls.random_state.binomial(1, propensity(x), 1)[0], 1, X)
        # Calculate outcome
        Y0 = np.apply_along_axis(lambda x: untreated_outcome(x), 1, X)
        treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
        Y = Y0 + treat_effect * T
        return (X, T, Y)
