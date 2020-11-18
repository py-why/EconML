# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
from collections import OrderedDict
from warnings import warn

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from statsmodels.iolib.table import SimpleTable

from .bootstrap import BootstrapEstimator
from .sklearn_extensions.linear_model import StatsModelsLinearRegression
from .utilities import (Summary, _safe_norm_ppf, broadcast_unit_treatments,
                        cross_product, inverse_onehot, ndim,
                        parse_final_model_params,
                        reshape_treatmentwise_effects, shape)

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

    bootstrap_type: 'percentile', 'pivot', or 'normal', default 'pivot'
        Bootstrap method used to compute results.
        'percentile' will result in using the empiracal CDF of the replicated computations of the statistics.
        'pivot' will also use the replicates but create a pivot interval that also relies on the estimate
        over the entire dataset.
        'normal' will instead compute a pivot interval assuming the replicates are normally distributed.
    """

    def __init__(self, n_bootstrap_samples=100, n_jobs=-1, bootstrap_type='pivot'):
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs
        self._bootstrap_type = bootstrap_type

    def fit(self, estimator, *args, **kwargs):
        est = BootstrapEstimator(estimator, self._n_bootstrap_samples, self._n_jobs, compute_means=False,
                                 bootstrap_type=self._bootstrap_type)
        est.fit(*args, **kwargs)
        self._est = est
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self.d_t = self._d_t[0] if self._d_t else 1
        self.d_y = self._d_y[0] if self._d_y else 1

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError()

        m = getattr(self._est, name)
        if name.endswith('_interval'):  # convert alpha to lower/upper
            def wrapped(*args, alpha=0.1, **kwargs):
                return m(*args, lower=100 * alpha / 2, upper=100 * (1 - alpha / 2), **kwargs)
            return wrapped
        else:
            return m


class GenericModelFinalInference(Inference):
    """
    Inference based on predict_interval of the model_final model. Assumes that estimator
    class has a model_final method, whose predict(cross_product(X, [0, ..., 1, ..., 0])) gives
    the const_marginal_effect of the treamtnent at the column with value 1 and which also supports
    prediction_stderr(X).
    """

    def prefit(self, estimator, *args, **kwargs):
        self.model_final = estimator.model_final
        self.featurizer = estimator.featurizer if hasattr(estimator, 'featurizer') else None

    def fit(self, estimator, *args, **kwargs):
        # once the estimator has been fit, it's kosher to store d_t here
        # (which needs to have been expanded if there's a discrete treatment)
        self._est = estimator
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self.d_t = self._d_t[0] if self._d_t else 1
        self.d_y = self._d_y[0] if self._d_y else 1

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        return self.const_marginal_effect_inference(X).conf_int(alpha=alpha)

    def const_marginal_effect_inference(self, X):
        if X is None:
            X = np.ones((1, 1))
        elif self.featurizer is not None:
            X = self.featurizer.transform(X)
        X, T = broadcast_unit_treatments(X, self.d_t)
        pred = reshape_treatmentwise_effects(self._predict(cross_product(X, T)), self._d_t, self._d_y)
        if not hasattr(self.model_final, 'prediction_stderr'):
            raise AttributeError("Final model doesn't support prediction standard eror, "
                                 "please call const_marginal_effect_interval to get confidence interval.")
        pred_stderr = reshape_treatmentwise_effects(self._prediction_stderr(cross_product(X, T)), self._d_t, self._d_y)
        return NormalInferenceResults(d_t=self.d_t, d_y=self.d_y, pred=pred,
                                      pred_stderr=pred_stderr, inf_type='effect')

    def _predict(self, X):
        return self.model_final.predict(X)

    def _prediction_stderr(self, X):
        return self.model_final.prediction_stderr(X)


class GenericSingleTreatmentModelFinalInference(GenericModelFinalInference):
    """
    Inference based on predict_interval of the model_final model. Assumes that treatment is single dimensional.
    Thus, the predict(X) of model_final gives the const_marginal_effect(X). The single dimensionality allows us
    to implement effect_interval(X, T0, T1) based on the const_marginal_effect_interval.
    """

    def fit(self, estimator, *args, **kwargs):
        super().fit(estimator, *args, **kwargs)
        if len(self._d_t) > 1 and (self._d_t[0] > 1):
            raise AttributeError("This method only works for single-dimensional continuous treatment "
                                 "or binary categorical treatment")

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
        # We can write effect interval as a function of const_marginal_effect_interval for a single treatment
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        lb_pre, ub_pre = self.const_marginal_effect_interval(X, alpha=alpha)
        dT = T1 - T0
        einsum_str = 'myt,mt->my'
        if ndim(dT) == 1:
            einsum_str = einsum_str.replace('t', '')
        if ndim(lb_pre) == ndim(dT):  # y is a vector, rather than a 2D array
            einsum_str = einsum_str.replace('y', '')
        intrv_pre = np.array([np.einsum(einsum_str, lb_pre, dT), np.einsum(einsum_str, ub_pre, dT)])
        lb = np.min(intrv_pre, axis=0)
        ub = np.max(intrv_pre, axis=0)
        return lb, ub

    def effect_inference(self, X, *, T0, T1):
        # We can write effect inference as a function of const_marginal_effect_inference for a single treatment
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        cme_pred = self.const_marginal_effect_inference(X).point_estimate
        cme_stderr = self.const_marginal_effect_inference(X).stderr
        dT = T1 - T0
        einsum_str = 'myt,mt->my'
        if ndim(dT) == 1:
            einsum_str = einsum_str.replace('t', '')
        if ndim(cme_pred) == ndim(dT):  # y is a vector, rather than a 2D array
            einsum_str = einsum_str.replace('y', '')
        e_pred = np.einsum(einsum_str, cme_pred, dT)
        e_stderr = np.einsum(einsum_str, cme_stderr, np.abs(dT))
        d_y = self._d_y[0] if self._d_y else 1
        # d_t=1 here since we measure the effect across all Ts
        return NormalInferenceResults(d_t=1, d_y=d_y, pred=e_pred,
                                      pred_stderr=e_stderr, inf_type='effect')


class LinearModelFinalInference(GenericModelFinalInference):
    """
    Inference based on predict_interval of the model_final model. Assumes that estimator
    class has a model_final method and that model is linear. Thus, the predict(cross_product(X, T1 - T0)) gives
    the effect(X, T0, T1). This allows us to implement effect_interval(X, T0, T1) based on the
    predict_interval of model_final.
    """

    def fit(self, estimator, *args, **kwargs):
        # once the estimator has been fit
        super().fit(estimator, *args, **kwargs)
        self._d_t_in = estimator._d_t_in
        self.bias_part_of_coef = estimator.bias_part_of_coef
        self.fit_cate_intercept = estimator.fit_cate_intercept

    # replacing _predict of super to fend against misuse, when the user has used a final linear model with
    # an intercept even when bias is part of coef.
    def _predict(self, X):
        intercept = 0
        if self.bias_part_of_coef:
            intercept = self.model_final.predict(np.zeros((1, X.shape[1])))
            if np.any(np.abs(intercept) > 0):
                warn("The final model has a nonzero intercept for at least one outcome; "
                     "it will be subtracted, but consider fitting a model without an intercept if possible. "
                     "Standard errors will also be slightly incorrect if the final model used fits an intercept "
                     "as they will be including the variance of the intercept parameter estimate.",
                     UserWarning)
        return self.model_final.predict(X) - intercept

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
        return self.effect_inference(X, T0=T0, T1=T1).conf_int(alpha=alpha)

    def effect_inference(self, X, *, T0, T1):
        # We can write effect inference as a function of prediction and prediction standard error of
        # the final method for linear models
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if X is None:
            X = np.ones((T0.shape[0], 1))
        elif self.featurizer is not None:
            X = self.featurizer.transform(X)
        e_pred = self._predict(cross_product(X, T1 - T0))
        e_stderr = self._prediction_stderr(cross_product(X, T1 - T0))
        d_y = self._d_y[0] if self._d_y else 1
        # d_t=1 here since we measure the effect across all Ts
        return NormalInferenceResults(d_t=1, d_y=d_y, pred=e_pred,
                                      pred_stderr=e_stderr, inf_type='effect')

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

    def coef__inference(self):
        coef = self.model_final.coef_
        coef_stderr = self.model_final.coef_stderr_
        intercept = self.model_final.intercept_
        intercept_stderr = self.model_final.intercept_stderr_
        coef = parse_final_model_params(coef, intercept,
                                        self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                        self.fit_cate_intercept)[0]
        coef_stderr = parse_final_model_params(coef_stderr, intercept_stderr,
                                               self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                               self.fit_cate_intercept)[0]
        if coef.size == 0:  # X is None
            raise AttributeError("X is None, please call intercept_inference to learn the constant!")

        if hasattr(self._est, 'cate_feature_names') and callable(self._est.cate_feature_names):
            def fname_transformer(x):
                return self._est.cate_feature_names(x)
        else:
            def fname_transformer(x):
                return x
        return NormalInferenceResults(d_t=self.d_t, d_y=self.d_y, pred=coef, pred_stderr=coef_stderr,
                                      inf_type='coefficient', fname_transformer=fname_transformer)

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

    def intercept__inference(self):
        if not self.fit_cate_intercept:
            raise AttributeError("No intercept was fitted!")
        coef = self.model_final.coef_
        coef_stderr = self.model_final.coef_stderr_
        intercept = self.model_final.intercept_
        intercept_stderr = self.model_final.intercept_stderr_
        intercept = parse_final_model_params(coef, intercept,
                                             self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                             self.fit_cate_intercept)[1]
        intercept_stderr = parse_final_model_params(coef_stderr, intercept_stderr,
                                                    self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                                    self.fit_cate_intercept)[1]
        return NormalInferenceResults(d_t=self.d_t, d_y=self.d_y, pred=intercept, pred_stderr=intercept_stderr,
                                      inf_type='intercept')


class StatsModelsInference(LinearModelFinalInference):
    """Stores statsmodels covariance options.

    This class can be used for inference by the LinearDML.

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
        assert not (self.model_final.fit_intercept), ("Inference can only be performed on models linear in "
                                                      "their features, but here fit_intercept is True")
        self.model_final.cov_type = self.cov_type


class GenericModelFinalInferenceDiscrete(Inference):
    """
    Assumes estimator is fitted on categorical treatment and a separate generic model_final is used to
    fit the CATE associated with each treatment. This model_final supports predict_interval. Inference is
    based on predict_interval of the model_final model.
    """

    def prefit(self, estimator, *args, **kwargs):
        self.model_final = estimator.model_final
        self.featurizer = estimator.featurizer if hasattr(estimator, 'featurizer') else None

    def fit(self, estimator, *args, **kwargs):
        # once the estimator has been fit, it's kosher to store d_t here
        # (which needs to have been expanded if there's a discrete treatment)
        self._est = estimator
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self.fitted_models_final = estimator.fitted_models_final
        self.d_t = self._d_t[0] if self._d_t else 1
        self.d_y = self._d_y[0] if self._d_y else 1
        if hasattr(estimator, 'fit_cate_intercept'):
            self.fit_cate_intercept = estimator.fit_cate_intercept

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        if (X is not None) and (self.featurizer is not None):
            X = self.featurizer.transform(X)
        preds = np.array([mdl.predict_interval(X, alpha=alpha) for mdl in self.fitted_models_final])
        return tuple(np.moveaxis(preds, [0, 1], [-1, 0]))  # send treatment to the end, pull bounds to the front

    def const_marginal_effect_inference(self, X):
        if (X is not None) and (self.featurizer is not None):
            X = self.featurizer.transform(X)
        pred = np.array([mdl.predict(X) for mdl in self.fitted_models_final])
        if not hasattr(self.fitted_models_final[0], 'prediction_stderr'):
            raise AttributeError("Final model doesn't support prediction standard eror, "
                                 "please call const_marginal_effect_interval to get confidence interval.")
        pred_stderr = np.array([mdl.prediction_stderr(X) for mdl in self.fitted_models_final])
        return NormalInferenceResults(d_t=self.d_t, d_y=self.d_y, pred=np.moveaxis(pred, 0, -1),
                                      # send treatment to the end, pull bounds to the front
                                      pred_stderr=np.moveaxis(pred_stderr, 0, -1), inf_type='effect')

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if np.any(np.any(T0 > 0, axis=1)):
            raise AttributeError("Can only calculate intervals of effects with respect to baseline treatment!")
        ind = inverse_onehot(T1)
        lower, upper = self.const_marginal_effect_interval(X, alpha=alpha)
        lower = np.concatenate([np.zeros(lower.shape[0:-1] + (1,)), lower], -1)
        upper = np.concatenate([np.zeros(upper.shape[0:-1] + (1,)), upper], -1)
        if X is None:  # Then const_marginal_effect_interval will return a single row
            lower, upper = np.repeat(lower, T0.shape[0], axis=0), np.repeat(upper, T0.shape[0], axis=0)
        return lower[np.arange(T0.shape[0]), ..., ind], upper[np.arange(T0.shape[0]), ..., ind]

    def effect_inference(self, X, *, T0, T1):
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if np.any(np.any(T0 > 0, axis=1)) or np.any(np.all(T1 == 0, axis=1)):
            raise AttributeError("Can only calculate inference of effects between a non-baseline treatment "
                                 "and the baseline treatment!")
        ind = inverse_onehot(T1)
        pred = self.const_marginal_effect_inference(X).point_estimate
        pred = np.concatenate([np.zeros(pred.shape[0:-1] + (1,)), pred], -1)
        pred_stderr = self.const_marginal_effect_inference(X).stderr
        pred_stderr = np.concatenate([np.zeros(pred_stderr.shape[0:-1] + (1,)), pred_stderr], -1)
        if X is None:  # Then const_marginal_effect_interval will return a single row
            pred = np.repeat(pred, T0.shape[0], axis=0)
            pred_stderr = np.repeat(pred_stderr, T0.shape[0], axis=0)
        # d_t=1 here since we measure the effect across all Ts
        return NormalInferenceResults(d_t=1, d_y=self.d_y, pred=pred[np.arange(T0.shape[0]), ..., ind],
                                      pred_stderr=pred_stderr[np.arange(T0.shape[0]), ..., ind],
                                      inf_type='effect')


class LinearModelFinalInferenceDiscrete(GenericModelFinalInferenceDiscrete):
    """
    Inference method for estimators with categorical treatments, where a linear in X model is used
    for the CATE associated with each treatment. Implements the coef__interval and intercept__interval
    based on the corresponding methods of the underlying model_final estimator.
    """

    def coef__interval(self, T, *, alpha=0.1):
        _, T = self._est._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].coef__interval(alpha)

    def coef__inference(self, T):
        _, T = self._est._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        coef = self.fitted_models_final[ind].coef_
        coef_stderr = self.fitted_models_final[ind].coef_stderr_
        if coef.size == 0:  # X is None
            raise AttributeError("X is None, please call intercept_inference to learn the constant!")
        if hasattr(self._est, 'cate_feature_names') and callable(self._est.cate_feature_names):
            def fname_transformer(x):
                return self._est.cate_feature_names(x)
        else:
            def fname_transformer(x):
                return x
        return NormalInferenceResults(d_t=1, d_y=self.d_y, pred=coef, pred_stderr=coef_stderr,
                                      inf_type='coefficient', fname_transformer=fname_transformer)

    def intercept__interval(self, T, *, alpha=0.1):
        if not self.fit_cate_intercept:
            raise AttributeError("No intercept was fitted!")
        _, T = self._est._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].intercept__interval(alpha)

    def intercept__inference(self, T):
        if not self.fit_cate_intercept:
            raise AttributeError("No intercept was fitted!")
        _, T = self._est._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return NormalInferenceResults(d_t=1, d_y=self.d_y, pred=self.fitted_models_final[ind].intercept_,
                                      pred_stderr=self.fitted_models_final[ind].intercept_stderr_,
                                      inf_type='intercept')


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


class InferenceResults(metaclass=abc.ABCMeta):
    """
    Results class for inferences.

    Parameters
    ----------
    d_t: int
        Number of treatments
    d_y: int
        Number of outputs
    pred : array-like, shape (m, d_y, d_t) or (m, d_y)
        The prediction of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    inf_type: string
        The type of inference result.
        It could be either 'effect', 'coefficient' or 'intercept'.
    fname_transformer: None or predefined function
        The transform function to get the corresponding feature names from featurizer
    """

    def __init__(self, d_t, d_y, pred, inf_type, fname_transformer=lambda nm: nm):
        self.d_t = d_t
        self.d_y = d_y
        self.pred = pred
        self.inf_type = inf_type
        self.fname_transformer = fname_transformer

    @property
    def point_estimate(self):
        """
        Get the point estimate of each treatment on each outcome for each sample X[i].

        Returns
        -------
        prediction : array-like, shape (m, d_y, d_t) or (m, d_y)
            The point estimate of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return self.pred

    @property
    @abc.abstractmethod
    def stderr(self):
        """
        Get the standard error of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        stderr : array-like, shape (m, d_y, d_t) or (m, d_y)
            The standard error of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pass

    @property
    def var(self):
        """
        Get the variance of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        var : array-like, shape (m, d_y, d_t) or (m, d_y)
            The variance of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return self.stderr**2

    @abc.abstractmethod
    def conf_int(self, alpha=0.1):
        """
        Get the confidence interval of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper: tuple of arrays, shape (m, d_y, d_t) or (m, d_y)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pass

    @abc.abstractmethod
    def pvalue(self, value=0):
        """
        Get the p value of the z test of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        pvalue : array-like, shape (m, d_y, d_t) or (m, d_y)
            The p value of the z test of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pass

    def zstat(self, value=0):
        """
        Get the z statistic of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        zstat : array-like, shape (m, d_y, d_t) or (m, d_y)
            The z statistic of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return (self.point_estimate - value) / self.stderr

    def summary_frame(self, alpha=0.1, value=0, decimals=3, feat_name=None, output_name=None, treatment_name=None):
        """
        Output the dataframe for all the inferences above.

        Parameters
        ----------
        alpha: optional float in [0, 1] (default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.
        feat_name: optional list of strings or None (default is None)
            The names of the features X
        output_name: optional list of strings or None (default is None)
            The names of the outputs
        treatment_name: optional list of strings or None (default is None)
            The names of the treatments

        Returns
        -------
        output: pandas dataframe
            The output dataframe includes point estimate, standard error, z score, p value and confidence intervals
            of the estimated metric of each treatment on each outcome for each sample X[i]
        """

        ci_mean = self.conf_int(alpha=alpha)
        to_include = OrderedDict()
        to_include['point_estimate'] = self._array_to_frame(self.d_t, self.d_y, self.point_estimate,
                                                            output_name=output_name, treatment_name=treatment_name)
        to_include['stderr'] = self._array_to_frame(self.d_t, self.d_y, self.stderr,
                                                    output_name=output_name, treatment_name=treatment_name)
        to_include['zstat'] = self._array_to_frame(self.d_t, self.d_y, self.zstat(value),
                                                   output_name=output_name, treatment_name=treatment_name)
        to_include['pvalue'] = self._array_to_frame(self.d_t, self.d_y, self.pvalue(value),
                                                    output_name=output_name, treatment_name=treatment_name)
        to_include['ci_lower'] = self._array_to_frame(self.d_t, self.d_y, ci_mean[0],
                                                      output_name=output_name, treatment_name=treatment_name)
        to_include['ci_upper'] = self._array_to_frame(self.d_t, self.d_y, ci_mean[1],
                                                      output_name=output_name, treatment_name=treatment_name)
        res = pd.concat(to_include, axis=1, keys=to_include.keys()).round(decimals)
        if self.d_t == 1:
            res.columns = res.columns.droplevel(1)
        if self.d_y == 1:
            res.index = res.index.droplevel(1)
        if self.inf_type == 'coefficient':
            if self.fname_transformer is not None:
                feat_name = self.fname_transformer(feat_name)
            if feat_name is not None:
                ind = feat_name
            else:
                ct = res.shape[0] // self.d_y
                ind = ['X' + str(i) for i in range(ct)]

            if self.d_y > 1:
                res.index = res.index.set_levels(ind, level=0)
            else:
                res.index = ind
        elif self.inf_type == 'intercept':
            if self.d_y > 1:
                res.index = res.index.set_levels(['cate_intercept'], level=0)
            else:
                res.index = ['cate_intercept']
        return res

    def population_summary(self, alpha=0.1, value=0, decimals=3, tol=0.001, output_name=None, treatment_name=None):
        """
        Output the object of population summary results.

        Parameters
        ----------
        alpha: optional float in [0, 1] (default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.
        tol:  optinal float (default=0.001)
            The stopping criterion. The iterations will stop when the outcome is less than ``tol``
        output_name: optional list of strings or None (default is None)
            The names of the outputs
        treatment_name: optional list of strings or None (default is None)
            The names of the treatments

        Returns
        -------
        PopulationSummaryResults: object
            The population summary results instance contains the different summary analysis of point estimate
            for sample X on each treatment and outcome.
        """
        if self.inf_type == 'effect':
            return PopulationSummaryResults(pred=self.point_estimate, pred_stderr=self.stderr,
                                            d_t=self.d_t, d_y=self.d_y,
                                            alpha=alpha, value=value, decimals=decimals, tol=tol,
                                            output_name=output_name, treatment_name=treatment_name)
        else:
            raise AttributeError(self.inf_type + " inference doesn't support population_summary function!")

    def _array_to_frame(self, d_t, d_y, arr, output_name=None, treatment_name=None):
        if np.isscalar(arr):
            arr = np.array([arr])
        if self.inf_type == 'coefficient':
            arr = np.moveaxis(arr, -1, 0)
        arr = arr.reshape((-1, d_y, d_t))
        df = pd.concat([pd.DataFrame(x) for x in arr], keys=np.arange(arr.shape[0]))
        if output_name is None:
            output_name = ['Y' + str(i) for i in range(d_y)]
        assert len(output_name) == d_y, "Incompatible length of output names"
        if treatment_name is None:
            treatment_name = ['T' + str(i) for i in range(d_t)]
        assert len(treatment_name) == d_t, "Incompatible length of treatment names"
        df.index = df.index.set_levels(output_name, level=1)
        df.columns = treatment_name
        return df

    @abc.abstractmethod
    def _expand_outputs(self, n_rows):
        """
        Expand the inference results from 1 row to n_rows identical rows.  This is used internally when
        we move from constant effects when X is None to a marginal effect of a different dimension.

        Parameters
        ----------
        n_rows: positive int
            The number of rows to expand to

        Returns
        -------
        results: InferenceResults
            The expanded results
        """
        pass


class NormalInferenceResults(InferenceResults):
    """
    Results class for inference assuming a normal distribution.

    Parameters
    ----------
    d_t: int
        Number of treatments
    d_y: int
        Number of outputs
    pred : array-like, shape (m, d_y, d_t) or (m, d_y)
        The prediction of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    pred_stderr : array-like, shape (m, d_y, d_t) or (m, d_y)
        The prediction standard error of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    inf_type: string
        The type of inference result.
        It could be either 'effect', 'coefficient' or 'intercept'.
    fname_transformer: None or predefined function
        The transform function to get the corresponding feature names from featurizer
    """

    def __init__(self, d_t, d_y, pred, pred_stderr, inf_type, fname_transformer=lambda nm: nm):
        self.pred_stderr = pred_stderr
        super().__init__(d_t, d_y, pred, inf_type, fname_transformer)

    @property
    def stderr(self):
        """
        Get the standard error of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        stderr : array-like, shape (m, d_y, d_t) or (m, d_y)
            The standard error of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return self.pred_stderr

    def conf_int(self, alpha=0.1):
        """
        Get the confidence interval of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper: tuple of arrays, shape (m, d_y, d_t) or (m, d_y)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        if np.isscalar(self.pred):
            return _safe_norm_ppf(alpha / 2, loc=self.pred, scale=self.pred_stderr),\
                _safe_norm_ppf(1 - alpha / 2, loc=self.pred, scale=self.pred_stderr)
        else:
            return np.array([_safe_norm_ppf(alpha / 2, loc=p, scale=err)
                             for p, err in zip(self.pred, self.pred_stderr)]),\
                np.array([_safe_norm_ppf(1 - alpha / 2, loc=p, scale=err)
                          for p, err in zip(self.pred, self.pred_stderr)])

    def pvalue(self, value=0):
        """
        Get the p value of the z test of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        pvalue : array-like, shape (m, d_y, d_t) or (m, d_y)
            The p value of the z test of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """

        return norm.sf(np.abs(self.zstat(value)), loc=0, scale=1) * 2

    def _expand_outputs(self, n_rows):
        assert shape(self.pred)[0] == shape(self.pred_stderr)[0] == 1
        pred = np.repeat(self.pred, n_rows, axis=0)
        pred_stderr = np.repeat(self.pred_stderr, n_rows, axis=0)
        return NormalInferenceResults(self.d_t, self.d_y, pred, pred_stderr, self.inf_type, self.fname_transformer)


class EmpiricalInferenceResults(InferenceResults):
    """
    Results class for inference with an empirical set of samples.

    Parameters
    ----------
    pred : array-like, shape (m, d_y, d_t) or (m, d_y)
        the point estimates of the metric using the full sample
    pred_dist : array-like, shape (b, m, d_y, d_t) or (b, m, d_y)
        the raw predictions of the metric sampled b times.
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
    d_t: int
        Number of treatments
    d_y: int
        Number of outputs
    inf_type: string
        The type of inference result.
        It could be either 'effect', 'coefficient' or 'intercept'.
    fname_transformer: None or predefined function
        The transform function to get the corresponding feature names from featurizer
    """

    def __init__(self, d_t, d_y, pred, pred_dist, inf_type, fname_transformer=lambda nm: nm):
        self.pred_dist = pred_dist
        super().__init__(d_t, d_y, pred, inf_type, fname_transformer)

    @property
    def stderr(self):
        """
        Get the standard error of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        stderr : array-like, shape (m, d_y, d_t) or (m, d_y)
            The standard error of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return np.std(self.pred_dist, axis=0)

    def conf_int(self, alpha=0.1):
        """
        Get the confidence interval of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper: tuple of arrays, shape (m, d_y, d_t) or (m, d_y)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        lower = alpha / 2
        upper = 1 - alpha / 2
        return np.percentile(self.pred_dist, lower * 100, axis=0), np.percentile(self.pred_dist, upper * 100, axis=0)

    def pvalue(self, value=0):
        """
        Get the p value of the each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        pvalue : array-like, shape (m, d_y, d_t) or (m, d_y)
            The p value of of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pvalues = np.minimum((self.pred_dist <= value).sum(axis=0),
                             (self.pred_dist >= value).sum(axis=0)) / self.pred_dist.shape[0]
        # in the degenerate case where every point in the distribution is equal to the value tested, return nan
        return np.where(np.all(self.pred_dist == value, axis=0), np.nan, pvalues)

    def _expand_outputs(self, n_rows):
        assert shape(self.pred)[0] == shape(self.pred_dist)[1] == 1
        pred = np.repeat(self.pred, n_rows, axis=0)
        pred_dist = np.repeat(self.pred_dist, n_rows, axis=1)
        return EmpiricalInferenceResults(self.d_t, self.d_y, pred, pred_dist, self.inf_type, self.fname_transformer)


class PopulationSummaryResults:
    """
    Population summary results class for inferences.

    Parameters
    ----------
    d_t: int
        Number of treatments
    d_y: int
        Number of outputs
    pred : array-like, shape (m, d_y, d_t) or (m, d_y)
        The prediction of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    pred_stderr : array-like, shape (m, d_y, d_t) or (m, d_y)
        The prediction standard error of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    alpha: optional float in [0, 1] (default=0.1)
        The overall level of confidence of the reported interval.
        The alpha/2, 1-alpha/2 confidence interval is reported.
    value: optinal float (default=0)
        The mean value of the metric you'd like to test under null hypothesis.
    decimals: optinal int (default=3)
        Number of decimal places to round each column to.
    tol:  optinal float (default=0.001)
        The stopping criterion. The iterations will stop when the outcome is less than ``tol``
    output_name: optional list of strings or None (default is None)
            The names of the outputs
    treatment_name: optional list of strings or None (default is None)
        The names of the treatments

    """

    def __init__(self, pred, pred_stderr, d_t, d_y, alpha, value, decimals, tol,
                 output_name=None, treatment_name=None):
        self.pred = pred
        self.pred_stderr = pred_stderr
        self.d_t = d_t
        self.d_y = d_y
        self.alpha = alpha
        self.value = value
        self.decimals = decimals
        self.tol = tol
        self.output_name = output_name
        self.treatment_name = treatment_name

    def __str__(self):
        return self.print().as_text()

    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        return self.print().as_html()

    @property
    def mean_point(self):
        """
        Get the mean of the point estimate of each treatment on each outcome for sample X.

        Returns
        -------
        mean_point : array-like, shape (d_y, d_t)
            The point estimate of each treatment on each outcome for sample X.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return np.mean(self.pred, axis=0)

    @property
    def stderr_mean(self):
        """
        Get the standard error of the mean point estimate of each treatment on each outcome for sample X.
        The output is a conservative upper bound.

        Returns
        -------
        stderr_mean : array-like, shape (d_y, d_t)
            The standard error of the mean point estimate of each treatment on each outcome for sample X.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return np.sqrt(np.mean(self.pred_stderr**2, axis=0))

    @property
    def zstat(self):
        """
        Get the z statistic of the mean point estimate of each treatment on each outcome for sample X.

        Returns
        -------
        zstat : array-like, shape (d_y, d_t)
            The z statistic of the mean point estimate of each treatment on each outcome for sample X.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        zstat = (self.mean_point - self.value) / self.stderr_mean
        return zstat

    @property
    def pvalue(self):
        """
        Get the p value of the z test of each treatment on each outcome for sample X.

        Returns
        -------
        pvalue : array-like, shape (d_y, d_t)
            The p value of the z test of each treatment on each outcome for sample X.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        pvalue = norm.sf(np.abs(self.zstat), loc=0, scale=1) * 2
        return pvalue

    @property
    def conf_int_mean(self):
        """
        Get the confidence interval of the mean point estimate of each treatment on each outcome for sample X.

        Returns
        -------
        lower, upper: tuple of arrays, shape (d_y, d_t)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """

        return np.array([_safe_norm_ppf(self.alpha / 2, loc=p, scale=err)
                         for p, err in zip([self.mean_point] if np.isscalar(self.mean_point) else self.mean_point,
                                           [self.stderr_mean] if np.isscalar(self.stderr_mean)
                                           else self.stderr_mean)]),\
            np.array([_safe_norm_ppf(1 - self.alpha / 2, loc=p, scale=err)
                      for p, err in zip([self.mean_point] if np.isscalar(self.mean_point) else self.mean_point,
                                        [self.stderr_mean] if np.isscalar(self.stderr_mean) else self.stderr_mean)])

    @property
    def std_point(self):
        """
        Get the standard deviation of the point estimate of each treatment on each outcome for sample X.

        Returns
        -------
        std_point : array-like, shape (d_y, d_t)
            The standard deviation of the point estimate of each treatment on each outcome for sample X.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return np.std(self.pred, axis=0)

    @property
    def percentile_point(self):
        """
        Get the confidence interval of the point estimate of each treatment on each outcome for sample X.

        Returns
        -------
        lower, upper: tuple of arrays, shape (d_y, d_t)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        lower_percentile_point = np.percentile(self.pred, (self.alpha / 2) * 100, axis=0)
        upper_percentile_point = np.percentile(self.pred, (1 - self.alpha / 2) * 100, axis=0)
        return np.array([lower_percentile_point]) if np.isscalar(lower_percentile_point) else lower_percentile_point, \
            np.array([upper_percentile_point]) if np.isscalar(upper_percentile_point) else upper_percentile_point

    @property
    def stderr_point(self):
        """
        Get the standard error of the point estimate of each treatment on each outcome for sample X.

        Returns
        -------
        stderr_point : array-like, shape (d_y, d_t)
            The standard error of the point estimate of each treatment on each outcome for sample X.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return np.sqrt(self.stderr_mean**2 + self.std_point**2)

    @property
    def conf_int_point(self):
        """
        Get the confidence interval of the point estimate of each treatment on each outcome for sample X.

        Returns
        -------
        lower, upper: tuple of arrays, shape (d_y, d_t)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        lower_ci_point = np.array([self._mixture_ppf(self.alpha / 2, self.pred, self.pred_stderr, self.tol)])
        upper_ci_point = np.array([self._mixture_ppf(1 - self.alpha / 2, self.pred, self.pred_stderr, self.tol)])
        return np.array([lower_ci_point]) if np.isscalar(lower_ci_point) else lower_ci_point,\
            np.array([upper_ci_point]) if np.isscalar(upper_ci_point) else upper_ci_point

    def print(self):
        """
        Output the summary inferences above.

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.
        """

        # 1. Uncertainty of Mean Point Estimate
        res1 = self._res_to_2darray(self.d_t, self.d_y, self.mean_point, self.decimals)
        res1 = np.hstack((res1, self._res_to_2darray(self.d_t, self.d_y, self.stderr_mean, self.decimals)))
        res1 = np.hstack((res1, self._res_to_2darray(self.d_t, self.d_y, self.zstat, self.decimals)))
        res1 = np.hstack((res1, self._res_to_2darray(self.d_t, self.d_y, self.pvalue, self.decimals)))
        res1 = np.hstack((res1, self._res_to_2darray(self.d_t, self.d_y, self.conf_int_mean[0], self.decimals)))
        res1 = np.hstack((res1, self._res_to_2darray(self.d_t, self.d_y, self.conf_int_mean[1], self.decimals)))

        treatment_name = self.treatment_name
        if treatment_name is None:
            treatment_name = ['T' + str(i) for i in range(self.d_t)]
        output_name = self.output_name
        if output_name is None:
            output_name = ['Y' + str(i) for i in range(self.d_y)]

        metric_name1 = ['mean_point', 'stderr_mean', 'zstat', 'pvalue', 'ci_mean_lower', 'ci_mean_upper']
        myheaders1 = [name + '\n' + tname for name in metric_name1 for tname in treatment_name
                      ] if self.d_t > 1 else [name for name in metric_name1]
        mystubs1 = output_name if self.d_y > 1 else []
        title1 = "Uncertainty of Mean Point Estimate"
        text1 = "Note: The stderr_mean is a conservative upper bound."

        # 2. Distribution of Point Estimate
        res2 = self._res_to_2darray(self.d_t, self.d_y, self.std_point, self.decimals)
        res2 = np.hstack((res2, self._res_to_2darray(self.d_t, self.d_y, self.percentile_point[0], self.decimals)))
        res2 = np.hstack((res2, self._res_to_2darray(self.d_t, self.d_y, self.percentile_point[1], self.decimals)))
        metric_name2 = ['std_point', 'pct_point_lower', 'pct_point_upper']
        myheaders2 = [name + '\n' + tname for name in metric_name2 for tname in treatment_name
                      ] if self.d_t > 1 else [name for name in metric_name2]
        mystubs2 = output_name if self.d_y > 1 else []
        title2 = "Distribution of Point Estimate"

        # 3. Total Variance of Point Estimate
        res3 = self._res_to_2darray(self.d_t, self.d_y, self.stderr_point, self.decimals)
        res3 = np.hstack((res3, self._res_to_2darray(self.d_t, self.d_y,
                                                     self.conf_int_point[0], self.decimals)))
        res3 = np.hstack((res3, self._res_to_2darray(self.d_t, self.d_y,
                                                     self.conf_int_point[1], self.decimals)))
        metric_name3 = ['stderr_point', 'ci_point_lower', 'ci_point_upper']
        myheaders3 = [name + '\n' + tname for name in metric_name3 for tname in treatment_name
                      ] if self.d_t > 1 else [name for name in metric_name3]
        mystubs3 = output_name if self.d_y > 1 else []
        title3 = "Total Variance of Point Estimate"

        smry = Summary()
        smry.add_table(res1, myheaders1, mystubs1, title1)
        smry.add_extra_txt([text1])
        smry.add_table(res2, myheaders2, mystubs2, title2)
        smry.add_table(res3, myheaders3, mystubs3, title3)
        return smry

    def _mixture_ppf(self, alpha, mean, stderr, tol):
        """
        Helper function to get the confidence interval of mixture gaussian distribution
        """
        # if stderr is zero, ppf will return nans and the loop below would never terminate
        # so bail out early; note that it might be possible to correct the algorithm for
        # this scenario, but since scipy's cdf returns nan whenever scale is zero it won't
        # be clean
        if (np.any(stderr == 0)):
            return np.full(shape(mean)[1:], np.nan)
        mix_ppf = scipy.stats.norm.ppf(alpha, loc=mean, scale=stderr)
        lower = np.min(mix_ppf, axis=0)
        upper = np.max(mix_ppf, axis=0)
        while True:
            cur = (lower + upper) / 2
            cur_mean = np.mean(scipy.stats.norm.cdf(cur, loc=mean, scale=stderr), axis=0)
            if np.isscalar(cur):
                if np.abs(cur_mean - alpha) < tol or (cur == lower):
                    return cur
                elif cur_mean < alpha:
                    lower = cur
                else:
                    upper = cur
            else:
                if np.all((np.abs(cur_mean - alpha) < tol) | (cur == lower)):
                    return cur
                lower[cur_mean < alpha] = cur[cur_mean < alpha]
                upper[cur_mean > alpha] = cur[cur_mean > alpha]

    def _res_to_2darray(self, d_t, d_y, res, decimals):
        arr = np.array([[res]]) if np.isscalar(res) else res.reshape((d_y, d_t))
        arr = np.round(arr, decimals)
        return arr
