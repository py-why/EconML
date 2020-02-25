# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import numpy as np
import scipy
from scipy.stats import norm
import pandas as pd
from collections import OrderedDict
from statsmodels.iolib.table import SimpleTable
from .bootstrap import BootstrapEstimator
from .utilities import (cross_product, broadcast_unit_treatments, reshape_treatmentwise_effects,
                        ndim, inverse_onehot, parse_final_model_params, _safe_norm_ppf, Summary)


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
        self.d_t = self._d_t[0] if self._d_t else 1
        self.d_y = self._d_y[0] if self._d_y else 1

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        if X is None:
            X = np.ones((1, 1))
        elif self.featurizer is not None:
            X = self.featurizer.transform(X)
        X, T = broadcast_unit_treatments(X, self.d_t)
        preds = self._predict_interval(cross_product(X, T), alpha=alpha)
        return tuple(reshape_treatmentwise_effects(pred, self._d_t, self._d_y)
                     for pred in preds)

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
        return InferenceResults(d_t=self.d_t, d_y=self.d_y, pred=pred,
                                pred_stderr=pred_stderr, inf_type='effect', pred_dist=None, fname_transformer=None)

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

    def effect_inference(self, X, *, T0, T1):
        # We can write effect inference as a function of const_marginal_effect_inference for a single treatment
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if (T0 == T1).all():
            raise AttributeError("T0 is the same as T1, please input different treatment!")
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
        return InferenceResults(d_t=1, d_y=d_y, pred=e_pred,
                                pred_stderr=e_stderr, inf_type='effect', pred_dist=None, fname_transformer=None)


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

    def effect_inference(self, X, *, T0, T1):
        # We can write effect inference as a function of prediction and prediction standard error of
        # the final method for linear models
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if (T0 == T1).all():
            raise AttributeError("T0 is the same as T1, please input different treatment!")
        if X is None:
            X = np.ones((T0.shape[0], 1))
        elif self.featurizer is not None:
            X = self.featurizer.transform(X)
        e_pred = self._predict(cross_product(X, T1 - T0))
        e_stderr = self._prediction_stderr(cross_product(X, T1 - T0))
        d_y = self._d_y[0] if self._d_y else 1
        # d_t=1 here since we measure the effect across all Ts
        return InferenceResults(d_t=1, d_y=d_y, pred=e_pred,
                                pred_stderr=e_stderr, inf_type='effect', pred_dist=None, fname_transformer=None)

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
            fname_transformer = None
        return InferenceResults(d_t=self.d_t, d_y=self.d_y, pred=coef,
                                pred_stderr=coef_stderr,
                                inf_type='coefficient', pred_dist=None, fname_transformer=fname_transformer)

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
        return InferenceResults(d_t=self.d_t, d_y=self.d_y, pred=intercept, pred_stderr=intercept_stderr,
                                inf_type='intercept', pred_dist=None, fname_transformer=None)

    def summary(self, alpha=0.1, value=0, decimals=3, feat_name=None):
        smry = Summary()
        try:
            coef_table = self.coef__inference().summary_frame(alpha=alpha,
                                                              value=value, decimals=decimals, feat_name=feat_name)
            coef_array = coef_table.values
            coef_headers = [i + '\n' +
                            j for (i, j) in coef_table.columns] if self.d_t > 1 else coef_table.columns.tolist()
            coef_stubs = [i + ' | ' + j for (i, j) in coef_table.index] if self.d_y > 1 else coef_table.index.tolist()
            coef_title = 'Coefficient Results'
            smry.add_table(coef_array, coef_headers, coef_stubs, coef_title)
        except Exception as e:
            print("Coefficient Results: ", str(e))
        try:
            intercept_table = self.intercept__inference().summary_frame(alpha=alpha,
                                                                        value=value, decimals=decimals, feat_name=None)
            intercept_array = intercept_table.values
            intercept_headers = [i + '\n' + j for (i, j)
                                 in intercept_table.columns] if self.d_t > 1 else intercept_table.columns.tolist()
            intercept_stubs = [i + ' | ' + j for (i, j)
                               in intercept_table.index] if self.d_y > 1 else intercept_table.index.tolist()
            intercept_title = 'Intercept Results'
            smry.add_table(intercept_array, intercept_headers, intercept_stubs, intercept_title)
        except Exception as e:
            print("Intercept Results: ", str(e))
        if len(smry.tables) > 0:
            return smry


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
        return InferenceResults(d_t=self.d_t, d_y=self.d_y, pred=np.moveaxis(pred, 0, -1),
                                # send treatment to the end, pull bounds to the front
                                pred_stderr=np.moveaxis(pred_stderr, 0, -1), inf_type='effect',
                                pred_dist=None, fname_transformer=None)

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
        if (T0 == T1).all():
            raise AttributeError("T0 is the same with T1, please input different treatment!")
        if np.any(np.any(T0 > 0, axis=1)):
            raise AttributeError("Can only calculate inference of effects with respect to baseline treatment!")
        ind = inverse_onehot(T1)
        pred = self.const_marginal_effect_inference(X).point_estimate
        pred = np.concatenate([np.zeros(pred.shape[0:-1] + (1,)), pred], -1)
        pred_stderr = self.const_marginal_effect_inference(X).stderr
        pred_stderr = np.concatenate([np.zeros(pred_stderr.shape[0:-1] + (1,)), pred_stderr], -1)
        if X is None:  # Then const_marginal_effect_interval will return a single row
            pred = np.repeat(pred, T0.shape[0], axis=0)
            pred_stderr = np.repeat(pred_stderr, T0.shape[0], axis=0)
        # d_t=1 here since we measure the effect across all Ts
        return InferenceResults(d_t=1, d_y=self.d_y, pred=pred[np.arange(T0.shape[0]), ..., ind],
                                pred_stderr=pred_stderr[np.arange(T0.shape[0]), ..., ind],
                                inf_type='effect', pred_dist=None, fname_transformer=None)


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
            fname_transformer = None
        return InferenceResults(d_t=1, d_y=self.d_y, pred=coef, pred_stderr=coef_stderr,
                                inf_type='coefficient', pred_dist=None, fname_transformer=fname_transformer)

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
        return InferenceResults(d_t=1, d_y=self.d_y, pred=self.fitted_models_final[ind].intercept_,
                                pred_stderr=self.fitted_models_final[ind].intercept_stderr_,
                                inf_type='intercept', pred_dist=None, fname_transformer=None)

    def summary(self, T, *, alpha=0.1, value=0, decimals=3, feat_name=None):
        smry = Summary()
        try:
            coef_table = self.coef__inference(T).summary_frame(
                alpha=alpha, value=value, decimals=decimals, feat_name=feat_name)
            coef_array = coef_table.values
            coef_headers = coef_table.columns.tolist()
            coef_stubs = coef_table.index.tolist()
            coef_title = 'Coefficient Results'
            smry.add_table(coef_array, coef_headers, coef_stubs, coef_title)
        except Exception as e:
            print("Coefficient Results: ", e)
        try:
            intercept_table = self.intercept__inference(T).summary_frame(
                alpha=alpha, value=value, decimals=decimals, feat_name=None)
            intercept_array = intercept_table.values
            intercept_headers = intercept_table.columns.tolist()
            intercept_stubs = intercept_table.index.tolist()
            intercept_title = 'Intercept Results'
            smry.add_table(intercept_array, intercept_headers, intercept_stubs, intercept_title)
        except Exception as e:
            print("Intercept Results: ", e)

        if len(smry.tables) > 0:
            return smry


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


class InferenceResults:
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
    pred_stderr : array-like, shape (m, d_y, d_t) or (m, d_y)
        The prediction standard error of the metric for each sample X[i].
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
        (e.g. if both are vectors, then the input of this argument will also be a vector)
    inf_type: string
        The type of inference result.
        It could be either 'effect', 'coefficient' or 'intercept'.
    pred_dist : array-like, shape (b, m, d_y, d_t) or (b, m, d_y)
        the raw predictions of the metric using b times bootstrap.
        Note that when Y or T is a vector rather than a 2-dimensional array,
        the corresponding singleton dimensions should be collapsed
    fname_transformer: None or predefined function
        The transform function to get the corresponding feature names from featurizer
    """

    def __init__(self, d_t, d_y, pred, pred_stderr, inf_type, pred_dist=None, fname_transformer=None):
        self.d_t = d_t
        self.d_y = d_y
        self.pred = pred
        self.pred_stderr = pred_stderr
        self.inf_type = inf_type
        self.pred_dist = pred_dist
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
        return (self.pred - value) / self.pred_stderr

    def summary_frame(self, alpha=0.1, value=0, decimals=3, feat_name=None):
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
            The input of the feature names

        Returns
        -------
        output: pandas dataframe
            The output dataframe includes point estimate, standard error, z score, p value and confidence intervals
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
        if self.inf_type == 'coefficient':
            if feat_name is not None and self.fname_transformer:
                ind = self.fname_transformer(feat_name)
            else:
                ct = res.shape[0] // self.d_y
                ind = ['X' + str(i) for i in range(ct)]

            if self.d_y > 1:
                res.index = res.index.set_levels(ind, level=0)
            else:
                res.index = ind
        elif self.inf_type == 'intercept':
            if self.d_y > 1:
                res.index = res.index.set_levels(['intercept'], level=0)
            else:
                res.index = ['intercept']
        return res

    def population_summary(self, alpha=0.1, value=0, decimals=3, tol=0.001):
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

        Returns
        -------
        PopulationSummaryResults: object
            The population summary results instance contains the different summary analysis of point estimate
            for sample X on each treatment and outcome.
        """
        if self.inf_type == 'effect':
            return PopulationSummaryResults(pred=self.pred, pred_stderr=self.pred_stderr, d_t=self.d_t, d_y=self.d_y,
                                            alpha=alpha, value=value, decimals=decimals, tol=tol)
        else:
            raise AttributeError(self.inf_type + " inference doesn't support population_summary function!")

    def _array_to_frame(self, d_t, d_y, arr):
        if np.isscalar(arr):
            arr = np.array([arr])
        if self.inf_type == 'coefficient':
            arr = np.moveaxis(arr, -1, 0)
        arr = arr.reshape((-1, d_y, d_t))
        df = pd.concat([pd.DataFrame(x) for x in arr], keys=np.arange(arr.shape[0]))
        df.index = df.index.set_levels(['Y' + str(i) for i in range(d_y)], level=1)
        df.columns = ['T' + str(i) for i in range(d_t)]
        return df


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

    """

    def __init__(self, pred, pred_stderr, d_t, d_y, alpha, value, decimals, tol):
        self.pred = pred
        self.pred_stderr = pred_stderr
        self.d_t = d_t
        self.d_y = d_y
        self.alpha = alpha
        self.value = value
        self.decimals = decimals
        self.tol = tol

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
        metric_name1 = ['mean_point', 'stderr_mean', 'zstat', 'pvalue', 'ci_mean_lower', 'ci_mean_upper']
        myheaders1 = [name + '\nT' + str(i) for name in metric_name1 for i in range(self.d_t)
                      ] if self.d_t > 1 else [name for name in metric_name1]
        mystubs1 = ["Y" + str(i) for i in range(self.d_y)] if self.d_y > 1 else []
        title1 = "Uncertainty of Mean Point Estimate"
        text1 = "Note: The stderr_mean is a conservative upper bound."

        # 2. Distribution of Point Estimate
        res2 = self._res_to_2darray(self.d_t, self.d_y, self.std_point, self.decimals)
        res2 = np.hstack((res2, self._res_to_2darray(self.d_t, self.d_y, self.percentile_point[0], self.decimals)))
        res2 = np.hstack((res2, self._res_to_2darray(self.d_t, self.d_y, self.percentile_point[1], self.decimals)))
        metric_name2 = ['std_point', 'pct_point_lower', 'pct_point_upper']
        myheaders2 = [name + '\nT' + str(i) for name in metric_name2 for i in range(self.d_t)
                      ] if self.d_t > 1 else [name for name in metric_name2]
        mystubs2 = ["Y" + str(i) for i in range(self.d_y)] if self.d_y > 1 else []
        title2 = "Distribution of Point Estimate"

        # 3. Total Variance of Point Estimate
        res3 = self._res_to_2darray(self.d_t, self.d_y, self.stderr_point, self.decimals)
        res3 = np.hstack((res3, self._res_to_2darray(self.d_t, self.d_y,
                                                     self.conf_int_point[0], self.decimals)))
        res3 = np.hstack((res3, self._res_to_2darray(self.d_t, self.d_y,
                                                     self.conf_int_point[1], self.decimals)))
        metric_name3 = ['stderr_point', 'ci_point_lower', 'ci_point_upper']
        myheaders3 = [name + '\nT' + str(i) for name in metric_name3 for i in range(self.d_t)
                      ] if self.d_t > 1 else [name for name in metric_name3]
        mystubs3 = ["Y" + str(i) for i in range(self.d_y)] if self.d_y > 1 else []
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
        done = False
        mix_ppf = scipy.stats.norm.ppf(alpha, loc=mean, scale=stderr)
        lower = np.min(mix_ppf, axis=0)
        upper = np.max(mix_ppf, axis=0)
        while not done:
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
