# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import numpy as np
from scipy.stats import norm
from .bootstrap import BootstrapEstimator
from .utilities import (cross_product, broadcast_unit_treatments, reshape_treatmentwise_effects,
                        ndim, inverse_onehot, parse_final_model_params)

"""Options for performing inference in estimators."""


class Inference(metaclass=abc.ABCMeta):
    def prefit(self, estimator, *args, **kwargs):
        """Performs any necessary logic before the estimator's fit has been called."""
        pass

    @abc.abstractmethod
    def fit(self, estimator, *args, **kwargs):
        """
        Fits the inference model.

        This is called after the estimator's fit.
        """
        pass


class BootstrapInference(Inference):
    """
    Inference instance to perform bootstrapping.

    This class can be used for inference with any CATE estimator.

    Parameters
    ----------
    n_bootstrap_samples : int, optional (default 100)
        How many draws to perform.

    n_jobs: int, optional (default -1)
        The maximum number of concurrently running jobs, as in joblib.Parallel.

    """

    def __init__(self, n_bootstrap_samples=100, n_jobs=-1):
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs

    def fit(self, estimator, *args, **kwargs):
        est = BootstrapEstimator(estimator, self._n_bootstrap_samples, self._n_jobs, compute_means=False)
        est.fit(*args, **kwargs)
        self._est = est

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError()

        m = getattr(self._est, name)

        def wrapped(*args, alpha=0.1, **kwargs):
            return m(*args, lower=100 * alpha / 2, upper=100 * (1 - alpha / 2), **kwargs)
        return wrapped


class LinearModelFinalInference(Inference):

    def __init__(self):
        pass

    def prefit(self, estimator, *args, **kwargs):
        self.model_final = estimator.model_final
        self.featurizer = estimator.featurizer if hasattr(estimator, 'featurizer') else None

    def fit(self, estimator, *args, **kwargs):
        # once the estimator has been fit, it's kosher to access its effect_op and store it here
        # (which needs to have seen the expanded d_t if there's a discrete treatment, etc.)
        self._est = estimator
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self._d_t_in = estimator._d_t_in
        self.bias_part_of_coef = estimator.bias_part_of_coef
        self.fit_cate_intercept = estimator.fit_cate_intercept

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if X is None:
            X = np.ones((T0.shape[0], 1))
        elif self.featurizer is not None:
            X = self.featurizer.transform(X)
        return self._predict_interval(cross_product(X, T1 - T0), alpha=alpha)

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        if X is None:
            X = np.ones((1, 1))
        elif self.featurizer is not None:
            X = self.featurizer.fit_transform(X)
        X, T = broadcast_unit_treatments(X, self._d_t[0] if self._d_t else 1)
        preds = self._predict_interval(cross_product(X, T), alpha=alpha)
        return tuple(reshape_treatmentwise_effects(pred, self._d_t, self._d_y)
                     for pred in preds)

    def coef__interval(self, *, alpha=0.1):
        lo, hi = self.model_final.coef__interval(alpha)
        lo_int, hi_int = self.model_final.intercept__interval(alpha)
        lo = parse_final_model_params(lo, lo_int,
                                      self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                      self.fit_cate_intercept)[0]
        hi = parse_final_model_params(hi, hi_int,
                                      self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                      self.fit_cate_intercept)[0]
        return lo, hi

    def intercept__interval(self, *, alpha=0.1):
        if not self.fit_cate_intercept:
            raise AttributeError("No intercept was fitted!")
        lo, hi = self.model_final.coef__interval(alpha)
        lo_int, hi_int = self.model_final.intercept__interval(alpha)
        lo = parse_final_model_params(lo, lo_int,
                                      self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                      self.fit_cate_intercept)[1]
        hi = parse_final_model_params(hi, hi_int,
                                      self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                      self.fit_cate_intercept)[1]
        return lo, hi

    def _predict_interval(self, X, alpha):
        return self.model_final.predict_interval(X, alpha=alpha)


class StatsModelsInference(LinearModelFinalInference):
    """Stores statsmodels covariance options.

    This class can be used for inference by the LinearDMLCateEstimator.

    Parameters
    ----------
    cov_type : string, optional (default 'HC1')
        The type of covariance estimation method to use.  Supported values are 'nonrobust',
        'HC0', 'HC1'.
    """

    def __init__(self, cov_type='HC1'):
        if cov_type not in ['nonrobust', 'HC0', 'HC1']:
            raise ValueError("Unsupported cov_type; "
                             "must be one of 'nonrobust', "
                             "'HC0', 'HC1'")

        self.cov_type = cov_type

    def prefit(self, estimator, *args, **kwargs):
        super().prefit(estimator, *args, **kwargs)
        self.model_final.cov_type = self.cov_type


class LinearModelFinalInferenceDiscrete(Inference):
    """
    Inference method for estimators with categorical treatments, where a linear in X model is used
    for the CATE associated with each treatment.

    TODO Create parent LinearModelFinalInference class so that some functionalities can be shared
    """

    def __init__(self):
        pass

    def prefit(self, estimator, *args, **kwargs):
        self.model_final = estimator.model_final
        self.featurizer = estimator.featurizer if hasattr(estimator, 'featurizer') else None

    def fit(self, estimator, *args, **kwargs):
        # once the estimator has been fit, it's kosher to access its effect_op and store it here
        # (which needs to have seen the expanded d_t if there's a discrete treatment, etc.)
        self._est = estimator
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self.fitted_models_final = estimator.fitted_models_final

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        if (X is not None) and (self.featurizer is not None):
            X = self.featurizer.fit_transform(X)
        preds = np.array([mdl.predict_interval(X, alpha=alpha) for mdl in self.fitted_models_final])
        return tuple([preds[:, 0, :].T, preds[:, 1, :].T])

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if np.any(np.any(T0 > 0, axis=1)):
            raise AttributeError("Can only calculate intervals of effects with respect to baseline treatment!")
        ind = (T1 @ np.arange(1, T1.shape[1] + 1)).astype(int)
        lower, upper = self.const_marginal_effect_interval(X, alpha=alpha)
        lower = np.hstack([np.zeros((lower.shape[0], 1)), lower])
        upper = np.hstack([np.zeros((upper.shape[0], 1)), upper])
        if X is None:  # Then statsmodels will return a single row
            lower, upper = np.tile(lower, (T0.shape[0], 1)), np.tile(upper, (T0.shape[0], 1))
        return lower[np.arange(T0.shape[0]), ind], upper[np.arange(T0.shape[0]), ind]

    def coef__interval(self, T, *, alpha=0.1):
        _, T = self._est._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].coef__interval(alpha)

    def intercept__interval(self, T, *, alpha=0.1):
        _, T = self._est._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].intercept__interval(alpha)


class StatsModelsInferenceDiscrete(LinearModelFinalInferenceDiscrete):
    """
    Special case where final model is a StatsModelsLinearRegression

    Parameters
    ----------
    cov_type : string, optional (default 'HC1')
        The type of covariance estimation method to use.  Supported values are 'nonrobust',
        'HC0', 'HC1'.
    """

    def __init__(self, cov_type='HC1'):
        if cov_type not in ['nonrobust', 'HC0', 'HC1']:
            raise ValueError("Unsupported cov_type; "
                             "must be one of 'nonrobust', "
                             "'HC0', 'HC1'")

        self.cov_type = cov_type

    def prefit(self, estimator, *args, **kwargs):
        super().prefit(estimator, *args, **kwargs)
        # need to set the fit args before the estimator is fit
        self.model_final.cov_type = self.cov_type
