# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all CATE estimators."""

import abc
import numpy as np
from functools import wraps
from copy import deepcopy
from warnings import warn
from .inference import BootstrapInference
from .utilities import (tensordot, ndim, reshape, shape, parse_final_model_params,
                        inverse_onehot, Summary, get_input_columns)
from .inference import StatsModelsInference, StatsModelsInferenceDiscrete, LinearModelFinalInference,\
    LinearModelFinalInferenceDiscrete, NormalInferenceResults, GenericSingleTreatmentModelFinalInference,\
    GenericModelFinalInferenceDiscrete
from ._shap import _shap_explain_cme, _shap_explain_joint_linear_model_cate


class BaseCateEstimator(metaclass=abc.ABCMeta):
    """Base class for all CATE estimators in this package."""

    def _get_inference_options(self):
        """
        Produce a dictionary mapping string names to :class:`.Inference` types.

        This is used by the :meth:`fit` method when a string is passed rather than an :class:`.Inference` type.
        """
        return {'bootstrap': BootstrapInference}

    def _get_inference(self, inference):
        options = self._get_inference_options()
        if isinstance(inference, str):
            if inference in options:
                inference = options[inference]()
            else:
                raise ValueError("Inference option '%s' not recognized; valid values are %s" %
                                 (inference, [*options]))
        # since inference objects can be stateful, we must copy it before fitting;
        # otherwise this sequence wouldn't work:
        #   est1.fit(..., inference=inf)
        #   est2.fit(..., inference=inf)
        #   est1.effect_interval(...)
        # because inf now stores state from fitting est2
        return deepcopy(inference)

    def _set_input_names(self, Y, T, X, set_flag=True):
        """Set input column names if inputs have column metadata."""
        self._input_names = {
            "feature_names": get_input_columns(X),
            "output_names": get_input_columns(Y),
            "treatment_names": get_input_columns(T)
        }
        if set_flag:
            # This flag is true when names are set in a child class instead
            # If names are set in a child class, add an attribute reflecting that
            self._input_names_set = True

    def _strata(self, Y, T, *args, **kwargs):
        """
        Get an array of values representing strata that should be preserved by bootstrapping.  For example,
        if treatment is discrete, then each bootstrapped estimator needs to be given at least one instance
        with each treatment type.  For estimators like DRIV, then the same is true of the combination of
        treatment and instrument.  The arguments to this method will match those to fit.

        Returns
        -------
        strata : array or None
            A vector with the same number of rows as the inputs, where the unique values represent
            the strata that need to be preserved by bootstrapping, or None if no preservation is necessary.
        """
        return None

    def _prefit(self, Y, T, *args, **kwargs):
        self._d_y = np.shape(Y)[1:]
        self._d_t = np.shape(T)[1:]
        # This works only if X is passed as a kwarg
        # We plan to enforce X as kwarg only in new releases
        if not hasattr(self, "_input_names_set"):
            # This checks if names have been set in a child class
            # If names were set in a child class, don't do it again
            X = kwargs.get('X')
            self._set_input_names(Y, T, X)

    @abc.abstractmethod
    def fit(self, *args, inference=None, **kwargs):
        """
        Estimate the counterfactual model from data, i.e. estimates functions
        :math:`\\tau(X, T0, T1)`, :math:`\\partial \\tau(T, X)`.

        Note that the signature of this method may vary in subclasses (e.g. classes that don't
        support instruments will not allow a `Z` argument)

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        X: optional (n, d_x) matrix
            Features for each sample
        W: optional (n, d_w) matrix
            Controls for each sample
        Z: optional (n, d_z) matrix
            Instruments for each sample
        inference: optional string, :class:`.Inference` instance, or None
            Method for performing inference.  All estimators support ``'bootstrap'``
            (or an instance of :class:`.BootstrapInference`), some support other methods as well.

        Returns
        -------
        self

        """
        pass

    def _wrap_fit(m):
        @wraps(m)
        def call(self, Y, T, *args, inference=None, **kwargs):
            inference = self._get_inference(inference)
            self._prefit(Y, T, *args, **kwargs)
            if inference is not None:
                inference.prefit(self, Y, T, *args, **kwargs)
            # call the wrapped fit method
            m(self, Y, T, *args, **kwargs)
            if inference is not None:
                # NOTE: we call inference fit *after* calling the main fit method
                inference.fit(self, Y, T, *args, **kwargs)
            self._inference = inference
            return self
        return call

    @abc.abstractmethod
    def effect(self, X=None, *, T0, T1):
        """
        Calculate the heterogeneous treatment effect :math:`\\tau(X, T0, T1)`.

        The effect is calculated between the two treatment points
        conditional on a vector of features on a set of m test samples :math:`\\{T0_i, T1_i, X_i\\}`.

        Parameters
        ----------
        T0: (m, d_t) matrix or vector of length m
            Base treatments for each sample
        T1: (m, d_t) matrix or vector of length m
            Target treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        τ: (m, d_y) matrix
            Heterogeneous treatment effects on each outcome for each sample
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        pass

    @abc.abstractmethod
    def marginal_effect(self, T, X=None):
        """
        Calculate the heterogeneous marginal effect :math:`\\partial\\tau(T, X)`.

        The marginal effect is calculated around a base treatment
        point conditional on a vector of features on a set of m test samples :math:`\\{T_i, X_i\\}`.

        Parameters
        ----------
        T: (m, d_t) matrix
            Base treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        grad_tau: (m, d_y, d_t) array
            Heterogeneous marginal effects on each outcome for each sample
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pass

    def ate(self, X=None, *, T0, T1):
        """
        Calculate the average treatment effect :math:`E_X[\\tau(X, T0, T1)]`.

        The effect is calculated between the two treatment points and is averaged over
        the population of X variables.

        Parameters
        ----------
        T0: (m, d_t) matrix or vector of length m
            Base treatments for each sample
        T1: (m, d_t) matrix or vector of length m
            Target treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        τ: float or (d_y,) array
            Average treatment effects on each outcome
            Note that when Y is a vector rather than a 2-dimensional array, the result will be a scalar
        """
        return np.mean(self.effect(X=X, T0=T0, T1=T1), axis=0)

    def marginal_ate(self, T, X=None):
        """
        Calculate the average marginal effect :math:`E_{T, X}[\\partial\\tau(T, X)]`.

        The marginal effect is calculated around a base treatment
        point and averaged over the population of X.

        Parameters
        ----------
        T: (m, d_t) matrix
            Base treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        grad_tau: (d_y, d_t) array
            Average marginal effects on each outcome
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return np.mean(self.marginal_effect(T, X=X), axis=0)

    def _expand_treatments(self, X=None, *Ts):
        """
        Given a set of features and treatments, return possibly modified features and treatments.

        Parameters
        ----------
        X: optional (m, d_x) matrix
            Features for each sample, or None
        Ts: sequence of (m, d_t) matrices
            Base treatments for each sample

        Returns
        -------
        output : tuple (X',T0',T1',...)
        """
        return (X,) + Ts

    def _defer_to_inference(m):
        @wraps(m)
        def call(self, *args, **kwargs):
            name = m.__name__
            if self._inference is not None:
                return getattr(self._inference, name)(*args, **kwargs)
            else:
                raise AttributeError("Can't call '%s' because 'inference' is None" % name)
        return call

    @_defer_to_inference
    def effect_interval(self, X=None, *, T0=0, T1=1, alpha=0.1):
        """ Confidence intervals for the quantities :math:`\\tau(X, T0, T1)` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix
            Features for each sample
        T0: optional (m, d_t) matrix or vector of length m (Default=0)
            Base treatments for each sample
        T1: optional (m, d_t) matrix or vector of length m (Default=1)
            Target treatments for each sample
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`effect(X, T0, T1)<effect>`, type of :meth:`effect(X, T0, T1))<effect>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        pass

    @_defer_to_inference
    def effect_inference(self, X=None, *, T0=0, T1=1):
        """ Inference results for the quantities :math:`\\tau(X, T0, T1)` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix
            Features for each sample
        T0: optional (m, d_t) matrix or vector of length m (Default=0)
            Base treatments for each sample
        T1: optional (m, d_t) matrix or vector of length m (Default=1)
            Target treatments for each sample

        Returns
        -------
        InferenceResults: object
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        pass

    @_defer_to_inference
    def marginal_effect_interval(self, T, X=None, *, alpha=0.1):
        """ Confidence intervals for the quantities :math:`\\partial \\tau(T, X)` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        T: (m, d_t) matrix
            Base treatments for each sample
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`marginal_effect(T, X)<marginal_effect>`, \
                             type of :meth:`marginal_effect(T, X)<marginal_effect>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        pass

    @_defer_to_inference
    def marginal_effect_inference(self, T, X=None):
        """ Inference results for the quantities :math:`\\partial \\tau(T, X)` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        T: (m, d_t) matrix
            Base treatments for each sample
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample

        Returns
        -------
        InferenceResults: object
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        pass

    @_defer_to_inference
    def ate_interval(self, X=None, *, T0, T1, alpha=0.1):
        """ Confidence intervals for the quantity :math:`E_X[\\tau(X, T0, T1)]` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix
            Features for each sample
        T0: optional (m, d_t) matrix or vector of length m (Default=0)
            Base treatments for each sample
        T1: optional (m, d_t) matrix or vector of length m (Default=1)
            Target treatments for each sample
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`ate(X, T0, T1)<ate>`, type of :meth:`ate(X, T0, T1))<ate>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        pass

    @_defer_to_inference
    def ate_inference(self, X=None, *, T0, T1):
        """ Inference results for the quantity :math:`E_X[\\tau(X, T0, T1)]` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix
            Features for each sample
        T0: optional (m, d_t) matrix or vector of length m (Default=0)
            Base treatments for each sample
        T1: optional (m, d_t) matrix or vector of length m (Default=1)
            Target treatments for each sample

        Returns
        -------
        PopulationSummaryResults: object
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        pass

    @_defer_to_inference
    def marginal_ate_interval(self, T, X=None, *, alpha=0.1):
        """ Confidence intervals for the quantities :math:`E_{T,X}[\\partial \\tau(T, X)]` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        T: (m, d_t) matrix
            Base treatments for each sample
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`marginal_ate(T, X)<marginal_ate>`, \
                             type of :meth:`marginal_ate(T, X)<marginal_ate>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        pass

    @_defer_to_inference
    def marginal_ate_inference(self, T, X=None):
        """ Inference results for the quantities :math:`E_{T,X}[\\partial \\tau(T, X)]` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        T: (m, d_t) matrix
            Base treatments for each sample
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample

        Returns
        -------
        PopulationSummaryResults: object
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        pass


class LinearCateEstimator(BaseCateEstimator):
    """Base class for all CATE estimators with linear treatment effects in this package."""

    @abc.abstractmethod
    def const_marginal_effect(self, X=None):
        """
        Calculate the constant marginal CATE :math:`\\theta(·)`.

        The marginal effect is conditional on a vector of
        features on a set of m test samples X[i].

        Parameters
        ----------
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample.

        Returns
        -------
        theta: (m, d_y, d_t) matrix or (d_y, d_t) matrix if X is None
            Constant marginal CATE of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pass

    def effect(self, X=None, *, T0, T1):
        """
        Calculate the heterogeneous treatment effect :math:`\\tau(X, T0, T1)`.

        The effect is calculatred between the two treatment points
        conditional on a vector of features on a set of m test samples :math:`\\{T0_i, T1_i, X_i\\}`.
        Since this class assumes a linear effect, only the difference between T0ᵢ and T1ᵢ
        matters for this computation.

        Parameters
        ----------
        T0: (m, d_t) matrix
            Base treatments for each sample
        T1: (m, d_t) matrix
            Target treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        effect: (m, d_y) matrix (or length m vector if Y was a vector)
            Heterogeneous treatment effects on each outcome for each sample.
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        X, T0, T1 = self._expand_treatments(X, T0, T1)
        # TODO: what if input is sparse? - there's no equivalent to einsum,
        #       but tensordot can't be applied to this problem because we don't sum over m
        eff = self.const_marginal_effect(X)
        # if X is None then the shape of const_marginal_effect will be wrong because the number
        # of rows of T was not taken into account
        if X is None:
            eff = np.repeat(eff, shape(T0)[0], axis=0)
        m = shape(eff)[0]
        dT = T1 - T0
        einsum_str = 'myt,mt->my'
        if ndim(dT) == 1:
            einsum_str = einsum_str.replace('t', '')
        if ndim(eff) == ndim(dT):  # y is a vector, rather than a 2D array
            einsum_str = einsum_str.replace('y', '')
        return np.einsum(einsum_str, eff, dT)

    def marginal_effect(self, T, X=None):
        """
        Calculate the heterogeneous marginal effect :math:`\\partial\\tau(T, X)`.

        The marginal effect is calculated around a base treatment
        point conditional on a vector of features on a set of m test samples :math:`\\{T_i, X_i\\}`.
        Since this class assumes a linear model, the base treatment is ignored in this calculation.

        Parameters
        ----------
        T: (m, d_t) matrix
            Base treatments for each sample
        X: optional (m, d_x) matrix
            Features for each sample

        Returns
        -------
        grad_tau: (m, d_y, d_t) array
            Heterogeneous marginal effects on each outcome for each sample
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        X, T = self._expand_treatments(X, T)
        eff = self.const_marginal_effect(X)
        return np.repeat(eff, shape(T)[0], axis=0) if X is None else eff

    def marginal_effect_interval(self, T, X=None, *, alpha=0.1):
        X, T = self._expand_treatments(X, T)
        effs = self.const_marginal_effect_interval(X=X, alpha=alpha)
        if X is None:  # need to repeat by the number of rows of T to ensure the right shape
            effs = tuple(np.repeat(eff, shape(T)[0], axis=0) for eff in effs)
        return effs
    marginal_effect_interval.__doc__ = BaseCateEstimator.marginal_effect_interval.__doc__

    def marginal_effect_inference(self, T, X=None):
        X, T = self._expand_treatments(X, T)
        cme_inf = self.const_marginal_effect_inference(X=X)
        if X is None:
            cme_inf = cme_inf._expand_outputs(shape(T)[0])
        return cme_inf
    marginal_effect_inference.__doc__ = BaseCateEstimator.marginal_effect_inference.__doc__

    @BaseCateEstimator._defer_to_inference
    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        """ Confidence intervals for the quantities :math:`\\theta(X)` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`const_marginal_effect(X)<const_marginal_effect>` ,\
                             type of :meth:`const_marginal_effect(X)<const_marginal_effect>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def const_marginal_effect_inference(self, X=None):
        """ Inference results for the quantities :math:`\\theta(X)` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample

        Returns
        -------
        InferenceResults: object
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        pass

    def const_marginal_ate(self, X=None):
        """
        Calculate the average constant marginal CATE :math:`E_X[\\theta(X)]`.

        Parameters
        ----------
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample.

        Returns
        -------
        theta: (d_y, d_t) matrix
            Average constant marginal CATE of each treatment on each outcome.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return np.mean(self.const_marginal_effect(X=X), axis=0)

    @BaseCateEstimator._defer_to_inference
    def const_marginal_ate_interval(self, X=None, *, alpha=0.1):
        """ Confidence intervals for the quantities :math:`E_X[\\theta(X)]` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`const_marginal_ate(X)<const_marginal_ate>` ,\
                             type of :meth:`const_marginal_ate(X)<const_marginal_ate>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def const_marginal_ate_inference(self, X=None):
        """ Inference results for the quantities :math:`E_X[\\theta(X)]` produced
        by the model. Available only when ``inference`` is not ``None``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample

        Returns
        -------
        PopulationSummaryResults: object
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        pass

    def marginal_ate(self, T, X=None):
        return self.const_marginal_ate(X=X)
    marginal_ate.__doc__ = BaseCateEstimator.marginal_ate.__doc__

    def marginal_ate_interval(self, T, X=None, *, alpha=0.1):
        return self.const_marginal_ate_interval(X=X, alpha=alpha)
    marginal_ate_interval.__doc__ = BaseCateEstimator.marginal_ate_interval.__doc__

    def marginal_ate_inference(self, T, X=None):
        return self.const_marginal_ate_inference(X=X)
    marginal_ate_inference.__doc__ = BaseCateEstimator.marginal_ate_inference.__doc__

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        """ Shap value for the final stage models (const_marginal_effect)

        Parameters
        ----------
        X: (m, d_x) matrix
            Features for each sample. Should be in the same shape of fitted X in final stage.
        feature_names: optional None or list of strings of length X.shape[1] (Default=None)
            The names of input features.
        treatment_names: optional None or list (Default=None)
            The name of treatment. In discrete treatment scenario, the name should not include the name of
            the baseline treatment (i.e. the control treatment, which by default is the alphabetically smaller)
        output_names:  optional None or list (Default=None)
            The name of the outcome.
        background_samples: int or None, (Default=100)
            How many samples to use to compute the baseline effect. If None then all samples are used.

        Returns
        -------
        shap_outs: nested dictionary of Explanation object
            A nested dictionary by using each output name (e.g. 'Y0', 'Y1', ... when `output_names=None`) and
            each treatment name (e.g. 'T0', 'T1', ... when `treatment_names=None`) as key
            and the shap_values explanation object as value. If the input data at fit time also contain metadata,
            (e.g. are pandas DataFrames), then the column metatdata for the treatments, outcomes and features
            are used instead of the above defaults (unless the user overrides with explicitly passing the
            corresponding names).
        """
        return _shap_explain_cme(self.const_marginal_effect, X, self._d_t, self._d_y,
                                 feature_names=feature_names, treatment_names=treatment_names,
                                 output_names=output_names, input_names=self._input_names,
                                 background_samples=background_samples)


class TreatmentExpansionMixin(BaseCateEstimator):
    """Mixin which automatically handles promotions of scalar treatments to the appropriate shape."""

    transformer = None

    def _prefit(self, Y, T, *args, **kwargs):
        super()._prefit(Y, T, *args, **kwargs)
        # need to store the *original* dimensions of T so that we can expand scalar inputs to match;
        # subclasses should overwrite self._d_t with post-transformed dimensions of T for generating treatments
        self._d_t_in = self._d_t

    def _expand_treatments(self, X=None, *Ts):
        n_rows = 1 if X is None else shape(X)[0]
        outTs = []
        for T in Ts:
            if (ndim(T) == 0) and self._d_t_in and self._d_t_in[0] > 1:
                warn("A scalar was specified but there are multiple treatments; "
                     "the same value will be used for each treatment.  Consider specifying"
                     "all treatments, or using the const_marginal_effect method.")
            if ndim(T) == 0:
                T = np.full((n_rows,) + self._d_t_in, T)

            if self.transformer:
                T = self.transformer.transform(T)
            outTs.append(T)

        return (X,) + tuple(outTs)

    # override effect to set defaults, which works with the new definition of _expand_treatments
    def effect(self, X=None, *, T0=0, T1=1):
        # NOTE: don't explicitly expand treatments here, because it's done in the super call
        return super().effect(X, T0=T0, T1=T1)
    effect.__doc__ = BaseCateEstimator.effect.__doc__

    def ate(self, X=None, *, T0=0, T1=1):
        return super().ate(X=X, T0=T0, T1=T1)
    ate.__doc__ = BaseCateEstimator.ate.__doc__

    def ate_interval(self, X=None, *, T0=0, T1=1, alpha=0.1):
        return super().ate_interval(X=X, T0=T0, T1=T1, alpha=alpha)
    ate_interval.__doc__ = BaseCateEstimator.ate_interval.__doc__

    def ate_inference(self, X=None, *, T0=0, T1=1):
        return super().ate_inference(X=X, T0=T0, T1=T1)
    ate_inference.__doc__ = BaseCateEstimator.ate_inference.__doc__


class LinearModelFinalCateEstimatorMixin(BaseCateEstimator):
    """
    Base class for models where the final stage is a linear model.

    Such an estimator must implement a :attr:`model_final_` attribute that points
    to the fitted final :class:`.StatsModelsLinearRegression` object that
    represents the fitted CATE model. Also must implement :attr:`featurizer_` that points
    to the fitted featurizer and :attr:`bias_part_of_coef` that designates
    if the intercept is the first element of the :attr:`model_final_` coefficient.

    Attributes
    ----------
    bias_part_of_coef: bool
        Whether the CATE model's intercept is contained in the final model's ``coef_`` rather
        than as a separate ``intercept_``
    """

    def _get_inference_options(self):
        options = super()._get_inference_options()
        options.update(auto=LinearModelFinalInference)
        return options

    @property
    def bias_part_of_coef(self):
        return False

    @property
    def coef_(self):
        """ The coefficients in the linear model of the constant marginal treatment
        effect.

        Returns
        -------
        coef: (n_x,) or (n_t, n_x) or (n_y, n_t, n_x) array like
            Where n_x is the number of features that enter the final model (either the
            dimension of X or the dimension of featurizer.fit_transform(X) if the CATE
            estimator has a featurizer.), n_t is the number of treatments, n_y is
            the number of outcomes. Dimensions are omitted if the original input was
            a vector and not a 2D array. For binary treatment the n_t dimension is
            also omitted.
        """
        return parse_final_model_params(self.model_final_.coef_, self.model_final_.intercept_,
                                        self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                        self.fit_cate_intercept_)[0]

    @property
    def intercept_(self):
        """ The intercept in the linear model of the constant marginal treatment
        effect.

        Returns
        -------
        intercept: float or (n_y,) or (n_y, n_t) array like
            Where n_t is the number of treatments, n_y is
            the number of outcomes. Dimensions are omitted if the original input was
            a vector and not a 2D array. For binary treatment the n_t dimension is
            also omitted.
        """
        if not self.fit_cate_intercept_:
            raise AttributeError("No intercept was fitted!")
        return parse_final_model_params(self.model_final_.coef_, self.model_final_.intercept_,
                                        self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                        self.fit_cate_intercept_)[1]

    @BaseCateEstimator._defer_to_inference
    def coef__interval(self, *, alpha=0.1):
        """ The coefficients in the linear model of the constant marginal treatment
        effect.

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lb, ub: tuple(type of :meth:`coef_()<coef_>`, type of :meth:`coef_()<coef_>`)
            The lower and upper bounds of the confidence interval for each quantity.
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def coef__inference(self):
        """ The inference of coefficients in the linear model of the constant marginal treatment
        effect.

        Returns
        -------
        InferenceResults: object
            The inference of the coefficients in the final linear model
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def intercept__interval(self, *, alpha=0.1):
        """ The intercept in the linear model of the constant marginal treatment
        effect.

        Parameters
        ----------
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper: tuple(type of :meth:`intercept_()<intercept_>`, type of :meth:`intercept_()<intercept_>`)
            The lower and upper bounds of the confidence interval.
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def intercept__inference(self):
        """ The inference of intercept in the linear model of the constant marginal treatment
        effect.

        Returns
        -------
        InferenceResults: object
            The inference of the intercept in the final linear model
        """
        pass

    def summary(self, alpha=0.1, value=0, decimals=3, feature_names=None, treatment_names=None, output_names=None):
        """ The summary of coefficient and intercept in the linear model of the constant marginal treatment
        effect.

        Parameters
        ----------
        alpha: optional float in [0, 1] (default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.
        feature_names: optional list of strings or None (default is None)
            The input of the feature names
        treatment_names: optional list of strings or None (default is None)
            The names of the treatments
        output_names: optional list of strings or None (default is None)
            The names of the outputs

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.
        """
        # Get input names
        treatment_names = self._input_names["treatment_names"] if treatment_names is None else treatment_names
        output_names = self._input_names["output_names"] if output_names is None else output_names
        # Summary
        smry = Summary()
        smry.add_extra_txt(["<sub>A linear parametric conditional average treatment effect (CATE) model was fitted:",
                            "$Y = \\Theta(X)\\cdot T + g(X, W) + \\epsilon$",
                            "where for every outcome $i$ and treatment $j$ the CATE $\\Theta_{ij}(X)$ has the form:",
                            "$\\Theta_{ij}(X) = \\phi(X)' coef_{ij} + cate\\_intercept_{ij}$",
                            "where $\\phi(X)$ is the output of the `featurizer` or $X$ if `featurizer`=None. "
                            "Coefficient Results table portrays the $coef_{ij}$ parameter vector for "
                            "each outcome $i$ and treatment $j$. "
                            "Intercept Results table portrays the $cate\\_intercept_{ij}$ parameter.</sub>"])
        d_t = self._d_t[0] if self._d_t else 1
        d_y = self._d_y[0] if self._d_y else 1
        try:
            coef_table = self.coef__inference().summary_frame(alpha=alpha,
                                                              value=value, decimals=decimals,
                                                              feature_names=feature_names,
                                                              treatment_names=treatment_names,
                                                              output_names=output_names)
            coef_array = coef_table.values
            coef_headers = [i + '\n' +
                            j for (i, j) in coef_table.columns] if d_t > 1 else coef_table.columns.tolist()
            coef_stubs = [i + ' | ' + j for (i, j) in coef_table.index] if d_y > 1 else coef_table.index.tolist()
            coef_title = 'Coefficient Results'
            smry.add_table(coef_array, coef_headers, coef_stubs, coef_title)
        except Exception as e:
            print("Coefficient Results: ", str(e))
        try:
            intercept_table = self.intercept__inference().summary_frame(alpha=alpha,
                                                                        value=value, decimals=decimals,
                                                                        feature_names=None,
                                                                        treatment_names=treatment_names,
                                                                        output_names=output_names)
            intercept_array = intercept_table.values
            intercept_headers = [i + '\n' + j for (i, j)
                                 in intercept_table.columns] if d_t > 1 else intercept_table.columns.tolist()
            intercept_stubs = [i + ' | ' + j for (i, j)
                               in intercept_table.index] if d_y > 1 else intercept_table.index.tolist()
            intercept_title = 'CATE Intercept Results'
            smry.add_table(intercept_array, intercept_headers, intercept_stubs, intercept_title)
        except Exception as e:
            print("CATE Intercept Results: ", str(e))
        if len(smry.tables) > 0:
            return smry

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        if hasattr(self, "featurizer_") and self.featurizer_ is not None:
            X = self.featurizer_.transform(X)
        feature_names = self.cate_feature_names(feature_names)
        return _shap_explain_joint_linear_model_cate(self.model_final_, X, self._d_t, self._d_y,
                                                     self.bias_part_of_coef,
                                                     feature_names=feature_names, treatment_names=treatment_names,
                                                     output_names=output_names,
                                                     input_names=self._input_names,
                                                     background_samples=background_samples)

    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__


class StatsModelsCateEstimatorMixin(LinearModelFinalCateEstimatorMixin):
    """
    Mixin class that offers `inference='statsmodels'` options to the CATE estimator
    that inherits it.

    Such an estimator must implement a :attr:`model_final_` attribute that points
    to the fitted final :class:`.StatsModelsLinearRegression` object that
    represents the fitted CATE model. Also must implement :attr:`featurizer_` that points
    to the fitted featurizer and :attr:`bias_part_of_coef` that designates
    if the intercept is the first element of the :attr:`model_final_` coefficient.
    """

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(statsmodels=StatsModelsInference)
        options.update(auto=StatsModelsInference)
        return options


class DebiasedLassoCateEstimatorMixin(LinearModelFinalCateEstimatorMixin):
    """Mixin for cate models where the final stage is a debiased lasso model."""

    def _get_inference_options(self):
        # add debiasedlasso to parent's options
        options = super()._get_inference_options()
        options.update(debiasedlasso=LinearModelFinalInference)
        options.update(auto=LinearModelFinalInference)
        return options


class ForestModelFinalCateEstimatorMixin(BaseCateEstimator):

    def _get_inference_options(self):
        # add blb to parent's options
        options = super()._get_inference_options()
        options.update(blb=GenericSingleTreatmentModelFinalInference)
        options.update(auto=GenericSingleTreatmentModelFinalInference)
        return options

    @property
    def feature_importances_(self):
        return self.model_final_.feature_importances_


class LinearModelFinalCateEstimatorDiscreteMixin(BaseCateEstimator):
    # TODO Share some logic with non-discrete version
    """
    Base class for models where the final stage is a linear model.

    Subclasses must expose a ``fitted_models_final`` attribute
    returning an array of the fitted models for each non-control treatment
    """

    def _get_inference_options(self):
        options = super()._get_inference_options()
        options.update(auto=LinearModelFinalInferenceDiscrete)
        return options

    def coef_(self, T):
        """ The coefficients in the linear model of the constant marginal treatment
        effect associated with treatment T.

        Parameters
        ----------
        T: alphanumeric
            The input treatment for which we want the coefficients.

        Returns
        -------
        coef: (n_x,) or (n_y, n_x) array like
            Where n_x is the number of features that enter the final model (either the
            dimension of X or the dimension of featurizer.fit_transform(X) if the CATE
            estimator has a featurizer.)
        """
        _, T = self._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].coef_

    def intercept_(self, T):
        """ The intercept in the linear model of the constant marginal treatment
        effect associated with treatment T.

        Parameters
        ----------
        T: alphanumeric
            The input treatment for which we want the coefficients.

        Returns
        -------
        intercept: float or (n_y,) array like
        """
        if not self.fit_cate_intercept_:
            raise AttributeError("No intercept was fitted!")
        _, T = self._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].intercept_.reshape(self._d_y)

    @BaseCateEstimator._defer_to_inference
    def coef__interval(self, T, *, alpha=0.1):
        """ The confidence interval for the coefficients in the linear model of the
        constant marginal treatment effect associated with treatment T.

        Parameters
        ----------
        T: alphanumeric
            The input treatment for which we want the coefficients.
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper: tuple(type of :meth:`coef_(T)<coef_>`, type of :meth:`coef_(T)<coef_>`)
            The lower and upper bounds of the confidence interval for each quantity.
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def coef__inference(self, T):
        """ The inference for the coefficients in the linear model of the
        constant marginal treatment effect associated with treatment T.

        Parameters
        ----------
        T: alphanumeric
            The input treatment for which we want the coefficients.

        Returns
        -------
        InferenceResults: object
            The inference of the coefficients in the final linear model
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def intercept__interval(self, T, *, alpha=0.1):
        """ The intercept in the linear model of the constant marginal treatment
        effect associated with treatment T.

        Parameters
        ----------
        T: alphanumeric
            The input treatment for which we want the coefficients.
        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper: tuple(type of :meth:`intercept_(T)<intercept_>`, type of :meth:`intercept_(T)<intercept_>`)
            The lower and upper bounds of the confidence interval.
        """
        pass

    @BaseCateEstimator._defer_to_inference
    def intercept__inference(self, T):
        """ The inference of the intercept in the linear model of the constant marginal treatment
        effect associated with treatment T.

        Parameters
        ----------
        T: alphanumeric
            The input treatment for which we want the coefficients.

        Returns
        -------
        InferenceResults: object
            The inference of the intercept in the final linear model

        """
        pass

    def summary(self, T, *, alpha=0.1, value=0, decimals=3,
                feature_names=None, treatment_names=None, output_names=None):
        """ The summary of coefficient and intercept in the linear model of the constant marginal treatment
        effect associated with treatment T.

        Parameters
        ----------
        alpha: optional float in [0, 1] (default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: optinal float (default=0)
            The mean value of the metric you'd like to test under null hypothesis.
        decimals: optinal int (default=3)
            Number of decimal places to round each column to.
        feature_names: optional list of strings or None (default is None)
            The input of the feature names
        treatment_names: optional list of strings or None (default is None)
            The names of the treatments
        output_names: optional list of strings or None (default is None)
            The names of the outputs

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.
        """
        # Get input names
        feature_names = self.cate_feature_names() if feature_names is None else feature_names
        treatment_names = self._input_names["treatment_names"] if treatment_names is None else treatment_names
        output_names = self._input_names["output_names"] if output_names is None else output_names
        # Summary
        smry = Summary()
        smry.add_extra_txt(["<sub>A linear parametric conditional average treatment effect (CATE) model was fitted:",
                            "$Y = \\Theta(X)\\cdot T + g(X, W) + \\epsilon$",
                            "where $T$ is the one-hot-encoding of the discrete treatment and "
                            "for every outcome $i$ and treatment $j$ the CATE $\\Theta_{ij}(X)$ has the form:",
                            "$\\Theta_{ij}(X) = \\phi(X)' coef_{ij} + cate\\_intercept_{ij}$",
                            "where $\\phi(X)$ is the output of the `featurizer` or $X$ if `featurizer`=None. "
                            "Coefficient Results table portrays the $coef_{ij}$ parameter vector for "
                            "each outcome $i$ and the designated treatment $j$ passed to summary. "
                            "Intercept Results table portrays the $cate\\_intercept_{ij}$ parameter.</sub>"])
        try:
            coef_table = self.coef__inference(T).summary_frame(
                alpha=alpha, value=value, decimals=decimals, feature_names=feature_names,
                treatment_names=treatment_names,
                output_names=output_names)
            coef_array = coef_table.values
            coef_headers = coef_table.columns.tolist()
            coef_stubs = coef_table.index.tolist()
            coef_title = 'Coefficient Results'
            smry.add_table(coef_array, coef_headers, coef_stubs, coef_title)
        except Exception as e:
            print("Coefficient Results: ", e)
        try:
            intercept_table = self.intercept__inference(T).summary_frame(
                alpha=alpha, value=value, decimals=decimals, feature_names=None,
                treatment_names=treatment_names,
                output_names=output_names)
            intercept_array = intercept_table.values
            intercept_headers = intercept_table.columns.tolist()
            intercept_stubs = intercept_table.index.tolist()
            intercept_title = 'CATE Intercept Results'
            smry.add_table(intercept_array, intercept_headers, intercept_stubs, intercept_title)
        except Exception as e:
            print("CATE Intercept Results: ", e)

        if len(smry.tables) > 0:
            return smry


class StatsModelsCateEstimatorDiscreteMixin(LinearModelFinalCateEstimatorDiscreteMixin):
    """
    Mixin class that offers `inference='statsmodels'` options to the CATE estimator
    that inherits it.

    Such an estimator must implement a :attr:`model_final_` attribute that points
    to a :class:`.StatsModelsLinearRegression` object that is cloned to fit
    each discrete treatment target CATE model and a :attr:`fitted_models_final` attribute
    that returns the list of fitted final models that represent the CATE for each categorical treatment.
    """

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(statsmodels=StatsModelsInferenceDiscrete)
        options.update(auto=StatsModelsInferenceDiscrete)
        return options


class DebiasedLassoCateEstimatorDiscreteMixin(LinearModelFinalCateEstimatorDiscreteMixin):
    """Mixin for cate models where the final stage is a debiased lasso model."""

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(debiasedlasso=LinearModelFinalInferenceDiscrete)
        options.update(auto=LinearModelFinalInferenceDiscrete)
        return options


class ForestModelFinalCateEstimatorDiscreteMixin(BaseCateEstimator):

    def _get_inference_options(self):
        # add blb to parent's options
        options = super()._get_inference_options()
        options.update(blb=GenericModelFinalInferenceDiscrete)
        options.update(auto=GenericModelFinalInferenceDiscrete)
        return options

    def feature_importances_(self, T):
        _, T = self._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].feature_importances_
