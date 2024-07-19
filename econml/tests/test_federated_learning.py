# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest

from econml.dml import LinearDML
from econml.dr import LinearDRLearner
from econml.inference import StatsModelsInference, StatsModelsInferenceDiscrete
from econml.federated_learning import FederatedEstimator


class FunctionRegressor:
    """A simple model that ignores the data it is fitted on, always just using the specified function to predict."""

    def __init__(self, func):
        self.func = func

    def fit(self, X, y, sample_weight=None):
        pass

    def predict(self, X):
        return self.func(X)


class FunctionClassifier(FunctionRegressor):
    """A simple model that ignores the data it is fitted on, always just using the specified function to predict."""

    def __init__(self, func):
        self.func = func

    def predict_proba(self, X):
        return self.func(X)


class TestFederatedLearning(unittest.TestCase):
    """
    A set of unit tests for the FederatedLearner class.

    These tests check various scenarios of splitting, aggregation, and comparison
    between FederatedLearner and individual LinearDML/LinearDRLearner estimators.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def test_lineardrlearner(self):

        num_samples = 1000
        n_w = 5
        n_x = 5
        for n_t in [2, 3]:
            X = np.random.normal(size=(num_samples, n_x))
            W = np.random.normal(size=(num_samples, n_w))
            T = np.random.choice(range(n_t), size=(num_samples,))
            Y = np.random.normal(size=(num_samples,))
            Y1, T1, X1, W1 = [A[:num_samples // 2] for A in [Y, T, X, W]]
            Y2, T2, X2, W2 = [A[num_samples // 2:] for A in [Y, T, X, W]]

            # fixed functions as first stage models
            # they can be anything as long as fitting doesn't modify the predictions
            # that way, it doesn't matter if they are trained on different subsets of the data
            a = np.random.normal(size=(n_x + n_w, n_t))
            b = np.random.normal(size=(n_x + n_w + n_t - 1))

            t_model = FunctionClassifier(lambda XW: np.exp(XW @ a))
            y_model = FunctionRegressor(lambda XW: XW @ b)

            for cov_type in ['HC0', 'HC1', 'nonrobust']:
                with self.subTest(n_t=n_t, cov_type=cov_type):
                    est_all = LinearDRLearner(model_propensity=t_model, model_regression=y_model,
                                              enable_federation=True)
                    est_h1 = LinearDRLearner(model_propensity=t_model, model_regression=y_model,
                                             enable_federation=True)
                    est_h2 = LinearDRLearner(model_propensity=t_model, model_regression=y_model,
                                             enable_federation=True)
                    est_h2_no_fed = LinearDRLearner(model_propensity=t_model, model_regression=y_model,
                                                    enable_federation=False)
                    est_h2_wrong_cov = LinearDRLearner(model_propensity=t_model, model_regression=y_model,
                                                       enable_federation=True)

                    est_all.fit(Y, T, X=X, W=W,
                                inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                    est_h1.fit(Y1, T1, X=X1, W=W1,
                               inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                    est_h2.fit(Y2, T2, X=X2, W=W2,
                               inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                    est_h2_no_fed.fit(Y2, T2, X=X2, W=W2,
                                      inference=StatsModelsInferenceDiscrete(cov_type=cov_type))
                    est_h2_wrong_cov.fit(Y2, T2, X=X2, W=W2,
                                         inference=StatsModelsInferenceDiscrete(cov_type=('HC0' if cov_type != 'HC0'
                                                                                          else 'HC1')))

                    est_fed1 = FederatedEstimator([est_all])

                    est_fed2 = FederatedEstimator([est_h1, est_h2])

                    with self.assertRaises(AssertionError):
                        # all estimators must have opted in to federation
                        FederatedEstimator([est_h1, est_h2_no_fed])

                    with self.assertRaises(AssertionError):
                        # all estimators must have the same covariance type
                        FederatedEstimator([est_h1, est_h2_wrong_cov])

                    # test coefficients
                    for t in range(1, n_t):
                        np.testing.assert_allclose(est_fed1.coef_(t),
                                                   est_fed2.coef_(t))
                        np.testing.assert_allclose(est_fed1.coef_(t),
                                                   est_all.coef_(t))
                        np.testing.assert_allclose(est_fed1.coef__interval(t),
                                                   est_fed2.coef__interval(t))
                        np.testing.assert_allclose(est_fed1.coef__interval(t),
                                                   est_all.coef__interval(t))

                    # test effects
                    np.testing.assert_allclose(est_fed1.effect(X[:10]),
                                               est_fed2.effect(X[:10]))
                    np.testing.assert_allclose(est_fed1.effect(X[:10]),
                                               est_all.effect(X[:10]))
                    np.testing.assert_allclose(est_fed1.effect_interval(X[:10]),
                                               est_fed2.effect_interval(X[:10]))
                    np.testing.assert_allclose(est_fed1.effect_interval(X[:10]),
                                               est_all.effect_interval(X[:10]))

    def test_splitting_works(self):

        num_samples = 1000
        n_w = 5
        for d_x in [None, 5]:
            for d_t in [(), (2,)]:
                for d_y in [(), (2,)]:
                    for weights in [None, np.random.uniform(size=num_samples)]:
                        for sample_var in [None, np.random.uniform(size=(num_samples,) + d_y)]:

                            n_x = d_x if d_x is not None else 0
                            X = np.random.normal(size=(num_samples, n_x)) if d_x is not None else None
                            W = np.random.normal(size=(num_samples, n_w))
                            T = np.random.normal(size=(num_samples,) + d_t)
                            Y = np.random.normal(size=(num_samples,) + d_y)
                            freq_weights = np.random.choice(
                                [2, 3, 4], size=num_samples) if sample_var is not None else None

                            (Y1, T1, X1, W1,
                             weights1, freq_weights1, sample_var1) = [A[:num_samples // 2] if A is not None else None
                                                                      for A in [Y, T, X, W,
                                                                                weights, freq_weights, sample_var]]
                            (Y2, T2, X2, W2,
                             weights2, freq_weights2, sample_var2) = [A[num_samples // 2:] if A is not None else None
                                                                      for A in [Y, T, X, W,
                                                                                weights, freq_weights, sample_var]]

                            # fixed functions as first stage models
                            # they can be anything as long as fitting doesn't modify the predictions
                            # that way, it doesn't matter if they are trained on different subsets of the data
                            a = np.random.normal(size=(n_x + n_w,) + d_t)
                            b = np.random.normal(size=(n_x + n_w,) + d_y)

                            t_model = FunctionRegressor(lambda XW: XW @ a)
                            y_model = FunctionRegressor(lambda XW: XW @ b)

                            for cov_type in ['HC0', 'HC1', 'nonrobust']:
                                with self.subTest(d_x=d_x, d_t=d_t, d_y=d_y,
                                                  weights=(weights is not None), fw=(freq_weights is not None),
                                                  cov_type=cov_type):
                                    est_all = LinearDML(model_t=t_model, model_y=y_model,
                                                        enable_federation=True)
                                    est_h1 = LinearDML(model_t=t_model, model_y=y_model,
                                                       enable_federation=True)
                                    est_h2 = LinearDML(model_t=t_model, model_y=y_model,
                                                       enable_federation=True)

                                    est_all.fit(Y, T, X=X, W=W,
                                                sample_weight=weights,
                                                freq_weight=freq_weights,
                                                sample_var=sample_var,
                                                inference=StatsModelsInference(cov_type=cov_type))
                                    est_h1.fit(Y1, T1, X=X1, W=W1,
                                               sample_weight=weights1,
                                               freq_weight=freq_weights1,
                                               sample_var=sample_var1,
                                               inference=StatsModelsInference(cov_type=cov_type))
                                    est_h2.fit(Y2, T2, X=X2, W=W2,
                                               sample_weight=weights2,
                                               freq_weight=freq_weights2,
                                               sample_var=sample_var2,
                                               inference=StatsModelsInference(cov_type=cov_type))

                                    est_fed1 = FederatedEstimator([est_all])

                                    est_fed2 = FederatedEstimator([est_h1, est_h2])
                                    # test coefficients
                                    np.testing.assert_allclose(est_fed1.coef_,
                                                               est_fed2.coef_)
                                    np.testing.assert_allclose(est_fed1.coef_,
                                                               est_all.coef_)
                                    np.testing.assert_allclose(est_fed1.coef__interval(),
                                                               est_fed2.coef__interval())
                                    np.testing.assert_allclose(est_fed1.coef__interval(),
                                                               est_all.coef__interval())

                                    # test effects
                                    X_test = X[:10] if X is not None else None
                                    np.testing.assert_allclose(est_fed1.effect(X_test),
                                                               est_fed2.effect(X_test))
                                    np.testing.assert_allclose(est_fed1.effect(X_test),
                                                               est_all.effect(X_test))
                                    np.testing.assert_allclose(est_fed1.effect_interval(X_test),
                                                               est_fed2.effect_interval(X_test))
                                    np.testing.assert_allclose(est_fed1.effect_interval(X_test),
                                                               est_all.effect_interval(X_test))
