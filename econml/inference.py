# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import numpy as np
from scipy.stats import norm
import pandas as pd
from collections import OrderedDict
from .bootstrap import BootstrapEstimator
from .utilities import (cross_product, broadcast_unit_treatments, reshape_treatmentwise_effects,
                        ndim, inverse_onehot, parse_final_model_params, _safe_norm_ppf)

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


class GenericModelFinalInference(Inference):
    """
    Inference based on predict_interval of the model_final model. Assumes that estimator
    class has a model_final method, whose predict(cross_product(X, [0, ..., 1, ..., 0])) gives
    the const_marginal_effect of the treamtnent at the column with value 1 and which also supports
    predict_interval(X).
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

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        if X is None:
            X = np.ones((1, 1))
        elif self.featurizer is not None:
            X = self.featurizer.fit_transform(X)
        X, T = broadcast_unit_treatments(X, self._d_t[0] if self._d_t else 1)
        preds = self._predict_interval(cross_product(X, T), alpha=alpha)
        return tuple(reshape_treatmentwise_effects(pred, self._d_t, self._d_y)
                     for pred in preds)

    def const_marginal_effect_inference(self, X):
        if X is None:
            X = np.ones((1, 1))
        elif self.featurizer is not None:
            X = self.featurizer.fit_transform(X)
        d_t = self._d_t[0] if self._d_t else 1
        d_y = self._d_y[0] if self._d_y else 1
        X, T = broadcast_unit_treatments(X, d_t)
        pred = reshape_treatmentwise_effects(self._predict(cross_product(X, T)), self._d_t, self._d_y)
        if not hasattr(self.model_final, 'prediction_stderr'):
            raise AttributeError("Final model doesn't support prediction standard eror, "
                                 "please call const_marginal_effect_interval to get confidence interval.")
        pred_stderr = reshape_treatmentwise_effects(self._prediction_stderr(cross_product(X, T)), self._d_t, self._d_y)
        return InferenceResults(d_t=d_t, d_y=d_y, pred=pred,
                                pred_stderr=pred_stderr, pred_dist=None)

    def _predict_interval(self, X, alpha):
        return self.model_final.predict_interval(X, alpha=alpha)

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

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
        # We can write effect interval as a function of predict_interval of the final method for linear models
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if X is None:
            X = np.ones((T0.shape[0], 1))
        elif self.featurizer is not None:
            X = self.featurizer.transform(X)
        return self._predict_interval(cross_product(X, T1 - T0), alpha=alpha)

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
        if X is None:  # Then const_marginal_effect_interval will return a single row
            lower, upper = np.tile(lower, (T0.shape[0], 1)), np.tile(upper, (T0.shape[0], 1))
        return lower[np.arange(T0.shape[0]), ind], upper[np.arange(T0.shape[0]), ind]


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


class InferenceResults(object):
    """
    Results class for inferences.

    Parameters
    ----------
    d_t: int
        Number of treatments
    d_y: int
        Number of outputs
    pred : array-like, shape (m, d_y, d_t)
        The prediction of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    pred_stderr : array-like, shape (m, d_y, d_t)
        The prediction standard error of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    pred_dist : array-like, shape (b, m, d_y, d_t)
        the raw predictions of the metric using b times bootstrap.
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
    """

    def __init__(self, d_t, d_y, pred, pred_stderr, pred_dist=None):
        self.d_t = d_t
        self.d_y = d_y
        self.pred = pred
        self.pred_stderr = pred_stderr
        self.pred_dist = pred_dist

    @property
    def stderr(self):
        """
        Get the standard error of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        stderr : array-like, shape (m, d_y, d_t)
            The standard error of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return self.pred_stderr

    @property
    def var(self):
        """
        Get the variance of the metric of each treatment on each outcome for each sample X[i].

        Returns
        -------
        var : array-like, shape (m, d_y, d_t)
            The variance of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return self.pred_stderr**2

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
        lower, upper: tuple of arrays, shape (m, d_y, d_t)
            The lower and the upper bounds of the confidence interval for each quantity.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """

        return np.array([_safe_norm_ppf(alpha / 2, loc=p, scale=err)
                         for p, err in zip(self.pred, self.pred_stderr)]),\
            np.array([_safe_norm_ppf(1 - alpha / 2, loc=p, scale=err)
                      for p, err in zip(self.pred, self.pred_stderr)])

    def pvalue(self, value=0):
        """
        Get the p value of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        pvalue : array-like, shape (m, d_y, d_t)
            The p value of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """

        return norm.sf(np.abs(self.zstat(value)), loc=0, scale=1) * 2

    def zstat(self, value=0):
        """
        Get the z statistic of the metric of each treatment on each outcome for each sample X[i].

        Parameters
        ----------
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.

        Returns
        -------
        zstat : array-like, shape (m, d_y, d_t)
            The z statistic of the metric of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return (self.pred - value) / self.pred_stderr

    def summary_frame(self, alpha=0.1, value=0, decimals=3):
        """
        Output the dataframe for all the inferences above.

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.

        Returns
        -------
        output: pandas dataframe
            the output dataframe includes point estimate, standard error, z score, p value and confidence intervals
            of the estimated metric of each treatment on each outcome for each sample X[i]
        """

        ci_mean = self.conf_int(alpha=alpha)
        to_include = OrderedDict()
        to_include['point_estimate'] = self._array_to_frame(self.d_t, self.d_y, self.pred)
        to_include['stderr'] = self._array_to_frame(self.d_t, self.d_y, self.pred_stderr)
        to_include['zstat'] = self._array_to_frame(self.d_t, self.d_y, self.zstat(value))
        to_include['pvalue'] = self._array_to_frame(self.d_t, self.d_y, self.pvalue(value))
        to_include['ci_lower'] = self._array_to_frame(self.d_t, self.d_y, ci_mean[0])
        to_include['ci_upper'] = self._array_to_frame(self.d_t, self.d_y, ci_mean[1])
        res = pd.concat(to_include, axis=1, keys=to_include.keys()).round(decimals)
        if self.d_t == 1:
            res.columns = res.columns.droplevel(1)
        if self.d_y == 1:
            res.index = res.index.droplevel(1)
        return res

    def population_summary(self, alpha=0.1, value=0, decimals=3):
        """
        Output the dataframe of population summary result.

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.

        Returns
        -------
        output: pandas dataframe
            the output dataframe includes average of point estimate, standard error of the mean,
        z statitic, p value and the confidence interval of the mean.
        """
        res_dic = OrderedDict()
        pop_mean = np.mean(self.pred, axis=0)
        var_pe = np.var(self.pred, axis=0)
        avg_stderr = np.mean(self.pred_stderr, axis=0)
        pop_stderr = np.sqrt(var_pe + avg_stderr)
        pop_zstat = (pop_mean - value) / pop_stderr
        pop_pvalue = norm.sf(np.abs(pop_zstat), loc=0, scale=1) * 2
        pop_ci_lower = np.array([_safe_norm_ppf(alpha / 2, loc=p, scale=err)
                                 for p, err in zip([pop_mean] if np.isscalar(pop_mean) else pop_mean,
                                                   [pop_stderr] if np.isscalar(pop_stderr) else pop_stderr)])
        pop_ci_upper = np.array([_safe_norm_ppf(1 - alpha / 2, loc=p, scale=err)
                                 for p, err in zip([pop_mean] if np.isscalar(pop_mean) else pop_mean,
                                                   [pop_stderr] if np.isscalar(pop_stderr) else pop_stderr)])

        res_dic['mean'] = self._array_to_frame(self.d_t, self.d_y, pop_mean)
        res_dic['stderr'] = self._array_to_frame(self.d_t, self.d_y, pop_stderr)
        res_dic['zstat'] = self._array_to_frame(self.d_t, self.d_y, pop_zstat)
        res_dic['pvalue'] = self._array_to_frame(self.d_t, self.d_y, pop_pvalue)
        res_dic['ci_lower'] = self._array_to_frame(self.d_t, self.d_y, pop_ci_lower)
        res_dic['ci_upper'] = self._array_to_frame(self.d_t, self.d_y, pop_ci_upper)
        res = pd.concat(res_dic, axis=1, keys=res_dic.keys()).round(decimals)
        res.index = res.index.droplevel(0)
        if self.d_t == 1:
            res.columns = res.columns.droplevel(1)
        return res

    def _array_to_frame(self, d_t, d_y, arr):
        arr = arr.reshape((-1, d_y, d_t))
        df = pd.Panel(arr).transpose(2, 0, 1).to_frame()
        df.index.names = [None, None]
        df.index = df.index.set_levels(['Y' + str(i) for i in range(d_y)], level=1)
        df.columns = ['T' + str(i) for i in range(d_t)]
        return df
