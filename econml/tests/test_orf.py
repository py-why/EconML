# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest
import unittest
import warnings
from types import SimpleNamespace
from numpy.random import choice, normal, uniform
from scipy.stats import norm
from sklearn.linear_model import Lasso, LogisticRegression, LogisticRegressionCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import FunctionTransformer
from econml.orf import DMLOrthoForest, DROrthoForest
from econml.orf._ortho_forest import BLBInference
from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
from econml.utilities import jacify_featurizer


@pytest.mark.parametrize("treatment_shape", [(), (1,), (2,)])
@pytest.mark.parametrize("outcome_shape", [(), (1,)])
@pytest.mark.parametrize("transform", ["identity", "linear", "quadratic"])
def test_blb_transformed_marginal_standard_errors(treatment_shape, outcome_shape, transform):
    """Propagate coefficient standard deviations through treatment transformations."""
    n = 4
    d_t = treatment_shape[0] if treatment_shape else 1
    T = np.linspace(-2, 3, n * d_t).reshape((n,) + treatment_shape)
    X = np.zeros((n, 1))
    params = np.arange(1, d_t + 1, dtype=float)
    factor = np.array([[2., 0.], [1., 3.]])[:d_t, :d_t]
    covariance = factor @ factor.T
    if transform == "identity":
        featurizer = FunctionTransformer()
        derivatives = np.tile(np.eye(d_t), (n, 1, 1))
    elif transform == "linear":
        matrix = np.array([[1., 2.], [-1., 0.5]]) if d_t == 2 else np.array([[-2.]])
        featurizer = FunctionTransformer(lambda value: (value.reshape(-1, d_t) @ matrix).reshape(value.shape))
        derivatives = np.tile(matrix, (n, 1, 1))
    else:
        featurizer = FunctionTransformer(lambda value: value**2)
        derivatives = 2 * T.reshape(n, d_t, 1) * np.eye(d_t)

    def expand_treatments(X, *treatments, transform=True):
        return (X, *(featurizer.transform(value) if transform else value for value in treatments))

    estimator = SimpleNamespace(
        _d_t=treatment_shape, _d_y=outcome_shape, _original_treatment_featurizer=featurizer,
        transformer=jacify_featurizer(featurizer),
        _expand_treatments=expand_treatments,
        _predict=lambda X, stderr: [(params, covariance) for _ in X],
        cate_feature_names=lambda: None, cate_output_names=lambda: None, cate_treatment_names=lambda: None
    )
    inference = BLBInference().fit(estimator)
    result = inference.marginal_effect_inference(T, X)
    alpha = 0.1
    if transform == "identity":
        marginal_interval = inference.marginal_effect_interval(T, X, alpha=alpha)
        for coordinate in range(d_t):
            T1 = T.copy()
            if treatment_shape:
                T1[:, coordinate] += 1
            else:
                T1 += 1
            effect_result = inference.effect_inference(X, T0=T, T1=T1)
            marginal_index = (..., coordinate) if treatment_shape else (...,)
            marginal_point = result.point_estimate[marginal_index]
            marginal_stderr = result.stderr[marginal_index]
            assert marginal_point.shape == effect_result.point_estimate.shape == (n,) + outcome_shape
            assert marginal_stderr.shape == effect_result.stderr.shape == (n,) + outcome_shape
            np.testing.assert_allclose(marginal_point, effect_result.point_estimate, atol=1e-10)
            np.testing.assert_allclose(marginal_stderr, effect_result.stderr, atol=1e-10)
            np.testing.assert_allclose(np.asarray(result.conf_int(alpha=alpha))[marginal_index],
                                       effect_result.conf_int(alpha=alpha), atol=1e-10)
            np.testing.assert_allclose(np.asarray(marginal_interval)[marginal_index],
                                       inference.effect_interval(X, T0=T, T1=T1, alpha=alpha), atol=1e-10)
    expected_shape = (n,) + outcome_shape + treatment_shape
    expected_effect = (derivatives @ params).reshape(expected_shape)
    # The factor maps independent unit-variance errors into coefficient errors.
    expected_stderr = np.linalg.norm(derivatives @ factor, axis=-1).reshape(expected_shape)
    np.testing.assert_allclose(result.point_estimate, expected_effect, atol=1e-10)
    np.testing.assert_allclose(result.stderr, expected_stderr, atol=1e-10)
    np.testing.assert_allclose(result.var, expected_stderr**2, atol=1e-10)
    expected_interval = (expected_effect - norm.ppf(1 - alpha / 2) * expected_stderr,
                         expected_effect + norm.ppf(1 - alpha / 2) * expected_stderr)
    np.testing.assert_allclose(result.conf_int(alpha=alpha), expected_interval, atol=1e-10)
    np.testing.assert_allclose(inference.marginal_effect_interval(T, X, alpha=alpha),
                               expected_interval, atol=1e-10)
    if transform == "identity":
        constant_result = inference.const_marginal_effect_inference(X)
        np.testing.assert_allclose(result.stderr, constant_result.stderr, atol=1e-10)
        np.testing.assert_allclose(result.conf_int(alpha=alpha), constant_result.conf_int(alpha=alpha), atol=1e-10)


class TestOrthoForest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        # DGP constants
        cls.n = 2000
        cls.d_w = 5
        cls.support_size = 1
        cls.d_x = 1
        cls.epsilon_sample = lambda n: uniform(-1, 1, size=n)
        cls.eta_sample = lambda n: uniform(-1, 1, size=n)
        cls.support = choice(range(cls.d_w), size=cls.support_size, replace=False)
        cls.coefs_T = uniform(0, 1, size=cls.support_size)
        cls.coefs_Y = uniform(0, 1, size=cls.support_size)
        # Generate data
        cls.X = uniform(0, 1, size=(cls.n, cls.d_x))
        cls.W = normal(0, 1, size=(cls.n, cls.d_w))
        # Test data
        cls.x_test = normal(0, 1, size=(10, cls.d_x))
        cls.x_test[:, 0] = np.arange(0, 1, 0.1)
        cls.expected_exp_te = np.array([cls._exp_te(x) for x in TestOrthoForest.x_test])
        cls.expected_const_te = np.array([cls._const_te(x) for x in TestOrthoForest.x_test])
        # Remove warnings that might be raised by the models passed into the ORF
        warnings.filterwarnings("ignore")

    def test_continuous_treatments(self):
        np.random.seed(123)
        for global_residualization in [False, True]:
            # Generate data with continuous treatments
            T = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_T) + \
                TestOrthoForest.eta_sample(TestOrthoForest.n)
            TE = np.array([self._exp_te(x) for x in TestOrthoForest.X])
            Y = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_Y) + \
                T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
            # Instantiate model with most of the default parameters. Using n_jobs=1 since code coverage
            # does not work well with parallelism.
            est = DMLOrthoForest(n_jobs=1, n_trees=10,
                                 model_T=Lasso(),
                                 model_Y=Lasso(),
                                 model_T_final=WeightedLassoCVWrapper(),
                                 model_Y_final=WeightedLassoCVWrapper(),
                                 global_residualization=global_residualization)
            # Test inputs for continuous treatments
            # --> Check that one can pass in regular lists
            est.fit(list(Y), list(T), X=list(TestOrthoForest.X), W=list(TestOrthoForest.W))
            # --> Check that it fails correctly if lists of different shape are passed in
            self.assertRaises(ValueError, est.fit, Y[:TestOrthoForest.n // 2], T[:TestOrthoForest.n // 2],
                              X=TestOrthoForest.X, W=TestOrthoForest.W)
            # Check that outputs have the correct shape
            out_te = est.const_marginal_effect(TestOrthoForest.x_test)
            self.assertEqual(TestOrthoForest.x_test.shape[0], out_te.shape[0])
            # Test continuous treatments with controls
            est = DMLOrthoForest(n_trees=100, min_leaf_size=10,
                                 max_depth=50, subsample_ratio=0.50, bootstrap=False, n_jobs=1,
                                 model_T=Lasso(alpha=0.024),
                                 model_Y=Lasso(alpha=0.024),
                                 model_T_final=WeightedLassoCVWrapper(cv=5),
                                 model_Y_final=WeightedLassoCVWrapper(cv=5),
                                 global_residualization=global_residualization,
                                 global_res_cv=5)
            est.fit(Y, T, X=TestOrthoForest.X, W=TestOrthoForest.W, inference="blb")
            self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.5)
            self._test_ci(est, TestOrthoForest.expected_exp_te, tol=1.5)
            # Test continuous treatments without controls
            T = TestOrthoForest.eta_sample(TestOrthoForest.n)
            Y = T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
            est.fit(Y, T, X=TestOrthoForest.X, inference="blb")
            self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.5)
            self._test_ci(est, TestOrthoForest.expected_exp_te, tol=1.5)

    def test_binary_treatments(self):
        np.random.seed(123)
        # Generate data with binary treatments
        log_odds = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_T) + \
            TestOrthoForest.eta_sample(TestOrthoForest.n)
        T_sigmoid = 1 / (1 + np.exp(-log_odds))
        T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
        TE = np.array([self._exp_te(x) for x in TestOrthoForest.X])
        Y = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_Y) + \
            T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        # Instantiate model with default params. Using n_jobs=1 since code coverage
        # does not work well with parallelism.
        est = DROrthoForest(n_trees=10, n_jobs=1,
                            propensity_model=LogisticRegression(), model_Y=Lasso(),
                            propensity_model_final=LogisticRegressionCV(penalty='l1', solver='saga'),
                            model_Y_final=WeightedLassoCVWrapper())
        # Test inputs for binary treatments
        # --> Check that one can pass in regular lists
        est.fit(list(Y), list(T), X=list(TestOrthoForest.X), W=list(TestOrthoForest.W))
        # --> Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, est.fit, Y[:TestOrthoForest.n // 2], T[:TestOrthoForest.n // 2],
                          X=TestOrthoForest.X, W=TestOrthoForest.W)
        # --> Check that it works when T, Y have shape (n, 1)
        est.fit(Y.reshape(-1, 1), T.reshape(-1, 1), X=TestOrthoForest.X, W=TestOrthoForest.W)
        # --> Check that it fails correctly when T has shape (n, 2)
        self.assertRaises(ValueError, est.fit, Y, np.ones((TestOrthoForest.n, 2)),
                          X=TestOrthoForest.X, W=TestOrthoForest.W)
        # --> Check that it fails correctly when the treatments are not numeric
        self.assertRaises(ValueError, est.fit, Y, np.array(["a"] * TestOrthoForest.n),
                          X=TestOrthoForest.X, W=TestOrthoForest.W)
        # Check that outputs have the correct shape
        out_te = est.const_marginal_effect(TestOrthoForest.x_test)
        self.assertSequenceEqual((TestOrthoForest.x_test.shape[0], 1, 1), out_te.shape)
        # Test binary treatments with controls
        est = DROrthoForest(n_trees=100, min_leaf_size=10,
                            max_depth=30, subsample_ratio=0.30, bootstrap=False, n_jobs=1,
                            propensity_model=LogisticRegression(
                                C=1 / 0.024, penalty='l1', solver='saga'),
                            model_Y=Lasso(alpha=0.024),
                            propensity_model_final=LogisticRegressionCV(penalty='l1', solver='saga'),
                            model_Y_final=WeightedLassoCVWrapper())
        est.fit(Y, T, X=TestOrthoForest.X, W=TestOrthoForest.W, inference="blb")
        self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.7, treatment_type='discrete')
        self._test_ci(est, TestOrthoForest.expected_exp_te, tol=1.5, treatment_type='discrete')
        # Test binary treatments without controls
        log_odds = TestOrthoForest.eta_sample(TestOrthoForest.n)
        T_sigmoid = 1 / (1 + np.exp(-log_odds))
        T = np.array([np.random.binomial(1, p) for p in T_sigmoid])
        Y = T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        est.fit(Y, T, X=TestOrthoForest.X, inference="blb")
        self._test_te(est, TestOrthoForest.expected_exp_te, tol=0.5, treatment_type='discrete')
        self._test_ci(est, TestOrthoForest.expected_exp_te, tol=1.5, treatment_type='discrete')

    def test_multiple_treatments(self):
        np.random.seed(123)
        # Only applicable to continuous treatments
        # Generate data for 2 treatments
        TE = np.array([[TestOrthoForest._exp_te(x), TestOrthoForest._const_te(x)] for x in TestOrthoForest.X])
        coefs_T = uniform(0, 1, size=(TestOrthoForest.support_size, 2))
        T = np.matmul(TestOrthoForest.W[:, TestOrthoForest.support], coefs_T) + \
            uniform(-1, 1, size=(TestOrthoForest.n, 2))
        delta_Y = np.array([np.dot(TE[i], T[i]) for i in range(TestOrthoForest.n)])
        Y = delta_Y + np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_Y) + \
            TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        for global_residualization in [False, True]:
            # Test multiple treatments with controls
            est = DMLOrthoForest(n_trees=100, min_leaf_size=10,
                                 max_depth=50, subsample_ratio=0.50, bootstrap=False, n_jobs=1,
                                 model_T=MultiOutputRegressor(Lasso(alpha=0.024)),
                                 model_Y=Lasso(alpha=0.024),
                                 model_T_final=WeightedLassoCVWrapper(cv=5),
                                 model_Y_final=WeightedLassoCVWrapper(cv=5),
                                 global_residualization=global_residualization,
                                 global_res_cv=5)
            est.fit(Y, T, X=TestOrthoForest.X, W=TestOrthoForest.W, inference="blb")
            expected_te = np.array([TestOrthoForest.expected_exp_te, TestOrthoForest.expected_const_te]).T
            self._test_te(est, expected_te, tol=0.5, treatment_type='multi')
            self._test_ci(est, expected_te, tol=2.0, treatment_type='multi')

    def test_effect_shape(self):
        np.random.seed(123)
        n = 40  # number of raw samples
        d = 4  # number of binary features + 1

        # Generating random segments aka binary features. We will use features 0,...,3 for heterogeneity.
        # The rest for controls. Just as an example.
        X = np.random.binomial(1, .5, size=(n, d))
        # Generating A/B test data
        T = np.random.binomial(2, .5, size=(n,))
        # Generating an outcome with treatment effect heterogeneity. The first binary feature creates heterogeneity
        # We also have confounding on the first variable. We also have heteroskedastic errors.
        y = (-1 + 2 * X[:, 0]) * T + X[:, 0] + (1 * X[:, 0] + 1) * np.random.normal(0, 1, size=(n,))
        from sklearn.dummy import DummyClassifier, DummyRegressor
        est = DROrthoForest(n_trees=10,
                            model_Y=DummyRegressor(strategy='mean'),
                            propensity_model=DummyClassifier(strategy='prior'),
                            n_jobs=1)
        est.fit(y, T, X=X)
        assert est.const_marginal_effect(X[:3]).shape == (3, 2), "Const Marginal Effect dimension incorrect"
        assert est.marginal_effect(1, X[:3]).shape == (3, 2), "Marginal Effect dimension incorrect"
        assert est.effect(X[:3]).shape == (3,), "Effect dimension incorrect"
        assert est.effect(X[:3], T0=0, T1=2).shape == (3,), "Effect dimension incorrect"
        assert est.effect(X[:3], T0=1, T1=2).shape == (3,), "Effect dimension incorrect"
        lb, _ = est.effect_interval(X[:3], T0=1, T1=2)
        assert lb.shape == (3,), "Effect interval dimension incorrect"
        lb, _ = est.effect_inference(X[:3], T0=1, T1=2).conf_int()
        assert lb.shape == (3,), "Effect interval dimension incorrect"
        lb, _ = est.const_marginal_effect_interval(X[:3])
        assert lb.shape == (3, 2), "Const Marginal Effect interval dimension incorrect"
        lb, _ = est.const_marginal_effect_inference(X[:3]).conf_int()
        assert lb.shape == (3, 2), "Const Marginal Effect interval dimension incorrect"
        lb, _ = est.marginal_effect_interval(1, X[:3])
        assert lb.shape == (3, 2), "Marginal Effect interval dimension incorrect"
        lb, _ = est.marginal_effect_inference(1, X[:3]).conf_int()
        assert lb.shape == (3, 2), "Marginal Effect interval dimension incorrect"
        est.fit(y.reshape(-1, 1), T, X=X)
        assert est.const_marginal_effect(X[:3]).shape == (3, 1, 2), "Const Marginal Effect dimension incorrect"
        assert est.marginal_effect(1, X[:3]).shape == (3, 1, 2), "Marginal Effect dimension incorrect"
        assert est.effect(X[:3]).shape == (3, 1), "Effect dimension incorrect"
        assert est.effect(X[:3], T0=0, T1=2).shape == (3, 1), "Effect dimension incorrect"
        assert est.effect(X[:3], T0=1, T1=2).shape == (3, 1), "Effect dimension incorrect"
        lb, _ = est.effect_interval(X[:3], T0=1, T1=2)
        assert lb.shape == (3, 1), "Effect interval dimension incorrect"
        lb, _ = est.effect_inference(X[:3], T0=1, T1=2).conf_int()
        assert lb.shape == (3, 1), "Effect interval dimension incorrect"
        lb, _ = est.const_marginal_effect_interval(X[:3])
        assert lb.shape == (3, 1, 2), "Const Marginal Effect interval dimension incorrect"
        lb, _ = est.const_marginal_effect_inference(X[:3]).conf_int()
        assert lb.shape == (3, 1, 2), "Const Marginal Effect interval dimension incorrect"
        lb, _ = est.marginal_effect_interval(1, X[:3])
        assert lb.shape == (3, 1, 2), "Marginal Effect interval dimension incorrect"
        lb, _ = est.marginal_effect_inference(1, X[:3]).conf_int()
        assert lb.shape == (3, 1, 2), "Marginal Effect interval dimension incorrect"

        from sklearn.preprocessing import FunctionTransformer
        from sklearn.dummy import DummyClassifier, DummyRegressor
        for global_residualization in [False, True]:
            for treatment_featurization in [None, FunctionTransformer()]:
                est = DMLOrthoForest(n_trees=10, model_Y=DummyRegressor(strategy='mean'),
                                     model_T=DummyRegressor(strategy='mean'),
                                     global_residualization=global_residualization,
                                     treatment_featurizer=treatment_featurization,
                                     n_jobs=1)
                est.fit(y.reshape(-1, 1), T.reshape(-1, 1), X=X)
                assert est.const_marginal_effect(X[:3]).shape == (3, 1, 1), "Const Marginal Effect dimension incorrect"
                assert est.marginal_effect(1, X[:3]).shape == (3, 1, 1), "Marginal Effect dimension incorrect"
                assert est.effect(X[:3]).shape == (3, 1), "Effect dimension incorrect"
                assert est.effect(X[:3], T0=0, T1=2).shape == (3, 1), "Effect dimension incorrect"
                assert est.effect(X[:3], T0=1, T1=2).shape == (3, 1), "Effect dimension incorrect"
                lb, _ = est.effect_interval(X[:3], T0=1, T1=2)
                assert lb.shape == (3, 1), "Effect interval dimension incorrect"
                lb, _ = est.effect_inference(X[:3], T0=1, T1=2).conf_int()
                assert lb.shape == (3, 1), "Effect interval dimension incorrect"
                lb, _ = est.const_marginal_effect_interval(X[:3])
                assert lb.shape == (3, 1, 1), "Const Marginal Effect interval dimension incorrect"
                lb, _ = est.const_marginal_effect_inference(X[:3]).conf_int()
                assert lb.shape == (3, 1, 1), "Const Marginal Effect interval dimension incorrect"
                lb, _ = est.marginal_effect_interval(1, X[:3])
                assert lb.shape == (3, 1, 1), "Marginal Effect interval dimension incorrect"
                lb, _ = est.marginal_effect_inference(1, X[:3]).conf_int()
                assert lb.shape == (3, 1, 1), "Marginal Effect interval dimension incorrect"
                est.fit(y.reshape(-1, 1), T, X=X)
                assert est.const_marginal_effect(X[:3]).shape == (3, 1), "Const Marginal Effect dimension incorrect"
                assert est.marginal_effect(1, X[:3]).shape == (3, 1), "Marginal Effect dimension incorrect"
                assert est.effect(X[:3]).shape == (3, 1), "Effect dimension incorrect"
                assert est.effect(X[:3], T0=0, T1=2).shape == (3, 1), "Effect dimension incorrect"
                assert est.effect(X[:3], T0=1, T1=2).shape == (3, 1), "Effect dimension incorrect"
                lb, _ = est.effect_interval(X[:3], T0=1, T1=2)
                assert lb.shape == (3, 1), "Effect interval dimension incorrect"
                lb, _ = est.effect_inference(X[:3], T0=1, T1=2).conf_int()
                assert lb.shape == (3, 1), "Effect interval dimension incorrect"
                lb, _ = est.const_marginal_effect_interval(X[:3])
                print(lb.shape)
                assert lb.shape == (3, 1), "Const Marginal Effect interval dimension incorrect"
                lb, _ = est.const_marginal_effect_inference(X[:3]).conf_int()
                assert lb.shape == (3, 1), "Const Marginal Effect interval dimension incorrect"
                lb, _ = est.marginal_effect_interval(1, X[:3])
                assert lb.shape == (3, 1), "Marginal Effect interval dimension incorrect"
                lb, _ = est.marginal_effect_inference(1, X[:3]).conf_int()
                assert lb.shape == (3, 1), "Marginal Effect interval dimension incorrect"
                est.fit(y, T, X=X)
                assert est.const_marginal_effect(X[:3]).shape == (3,), "Const Marginal Effect dimension incorrect"
                assert est.marginal_effect(1, X[:3]).shape == (3,), "Marginal Effect dimension incorrect"
                assert est.effect(X[:3]).shape == (3,), "Effect dimension incorrect"
                assert est.effect(X[:3], T0=0, T1=2).shape == (3,), "Effect dimension incorrect"
                assert est.effect(X[:3], T0=1, T1=2).shape == (3,), "Effect dimension incorrect"
                lb, _ = est.effect_interval(X[:3], T0=1, T1=2)
                assert lb.shape == (3,), "Effect interval dimension incorrect"
                lb, _ = est.effect_inference(X[:3], T0=1, T1=2).conf_int()
                assert lb.shape == (3,), "Effect interval dimension incorrect"
                lb, _ = est.const_marginal_effect_interval(X[:3])
                assert lb.shape == (3,), "Const Marginal Effect interval dimension incorrect"
                lb, _ = est.const_marginal_effect_inference(X[:3]).conf_int()
                assert lb.shape == (3,), "Const Marginal Effect interval dimension incorrect"
                lb, _ = est.marginal_effect_interval(1, X[:3])
                assert lb.shape == (3,), "Marginal Effect interval dimension incorrect"
                lb, _ = est.marginal_effect_inference(1, X[:3]).conf_int()
                assert lb.shape == (3,), "Marginal Effect interval dimension incorrect"

    def test_nuisance_model_has_weights(self):
        """Test whether the correct exception is being raised if model_final doesn't have weights."""

        # Create a wrapper around Lasso that doesn't support weights
        # since Lasso does natively support them starting in sklearn 0.23
        class NoWeightModel:
            def __init__(self):
                self.model = Lasso()

            def fit(self, X, y):
                self.model.fit(X, y)
                return self

            def predict(self, X):
                return self.model.predict(X)

        # Generate data with continuous treatments
        T = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_T) + \
            TestOrthoForest.eta_sample(TestOrthoForest.n)
        TE = np.array([self._exp_te(x) for x in TestOrthoForest.X])
        Y = np.dot(TestOrthoForest.W[:, TestOrthoForest.support], TestOrthoForest.coefs_Y) + \
            T * TE + TestOrthoForest.epsilon_sample(TestOrthoForest.n)
        # Instantiate model with most of the default parameters
        est = DMLOrthoForest(n_jobs=1, n_trees=10,
                             model_T=NoWeightModel(),
                             model_Y=NoWeightModel())
        est.fit(Y=Y, T=T, X=TestOrthoForest.X, W=TestOrthoForest.W)
        weights_error_msg = (
            "Estimators of type {} do not accept weights. "
            "Consider using the class WeightedModelWrapper from econml.utilities to build a weighted model."
        )
        self.assertRaisesRegex(TypeError, weights_error_msg.format("NoWeightModel"),
                               est.effect, X=TestOrthoForest.X)

    def _test_te(self, learner_instance, expected_te, tol, treatment_type='continuous'):
        # Compute the treatment effect on test points
        te_hat = learner_instance.const_marginal_effect(
            TestOrthoForest.x_test
        )
        # Compute treatment effect residuals
        if treatment_type == 'continuous':
            te_res = np.abs(expected_te - te_hat)
        elif treatment_type == 'discrete':
            te_res = np.abs(expected_te - te_hat[:, 0])
        else:
            # Multiple treatments
            te_res = np.abs(expected_te - te_hat)
        # Allow at most 10% test points to be outside of the tolerance interval
        self.assertLessEqual(np.mean(te_res > tol), 0.2)

    def _test_ci(self, learner_instance, expected_te, tol, treatment_type='continuous'):

        for te_lower, te_upper in\
            [learner_instance.const_marginal_effect_interval(TestOrthoForest.x_test),
             learner_instance.const_marginal_effect_inference(TestOrthoForest.x_test).conf_int()]:

            # Compute treatment effect residuals
            if treatment_type == 'continuous':
                delta_ci_upper = te_upper - expected_te
                delta_ci_lower = expected_te - te_lower
            elif treatment_type == 'discrete':
                delta_ci_upper = te_upper[:, 0] - expected_te
                delta_ci_lower = expected_te - te_lower[:, 0]
            else:
                # Multiple treatments
                delta_ci_upper = te_upper - expected_te
                delta_ci_lower = expected_te - te_lower
            # Allow at most 20% test points to be outside of the confidence interval
            # Check that the intervals are not too wide
            self.assertLessEqual(np.mean(delta_ci_upper < 0), 0.2)
            self.assertLessEqual(np.mean(np.abs(delta_ci_upper) > tol), 0.2)
            self.assertLessEqual(np.mean(delta_ci_lower < 0), 0.2)
            self.assertLessEqual(np.mean(np.abs(delta_ci_lower) > tol), 0.2)

    @classmethod
    def _const_te(cls, x):
        return 2

    @classmethod
    def _exp_te(cls, x):
        return np.exp(x[0] * 2)
