# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all CATE estimators."""

import abc
import numpy as np
from functools import wraps
from copy import deepcopy
from warnings import warn
from .bootstrap import BootstrapEstimator
from .inference import BootstrapInference
from .utilities import tensordot, ndim, reshape, shape, parse_final_model_params, inverse_onehot
from .inference import StatsModelsInference, StatsModelsInferenceDiscrete, LinearModelFinalInference,\
    LinearModelFinalInferenceDiscrete, InferenceResults


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

    def _prefit(self, Y, T, *args, **kwargs):
        self._d_y = np.shape(Y)[1:]
        self._d_t = np.shape(T)[1:]

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
        return tuple(np.repeat(eff, shape(T)[0], axis=0) if X is None else eff
                     for eff in effs)
    marginal_effect_interval.__doc__ = BaseCateEstimator.marginal_effect_interval.__doc__

    def marginal_effect_inference(self, T, X=None):
        X, T = self._expand_treatments(X, T)
        cme_inf = self.const_marginal_effect_inference(X=X)
        pred = cme_inf.point_estimate
        pred_stderr = cme_inf.stderr
        if X is None:
            pred = np.repeat(pred, shape(T)[0], axis=0)
            pred_stderr = np.repeat(pred_stderr, shape(T)[0], axis=0)
        return InferenceResults(d_t=cme_inf.d_t, d_y=cme_inf.d_y, pred=pred,
                                pred_stderr=pred_stderr, inf_type='effect', pred_dist=None, fname_transformer=None)
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


class LinearModelFinalCateEstimatorMixin(BaseCateEstimator):
    """
    Base class for models where the final stage is a linear model.

    Subclasses must expose a ``model_final`` attribute containing the model's
    final stage model.

    Attributes
    ----------
    bias_part_of_coef: bool
        Whether the CATE model's intercept is contained in the final model's ``coef_`` rather
        than as a separate ``intercept_``
    """

    bias_part_of_coef = False

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
        return parse_final_model_params(self.model_final.coef_, self.model_final.intercept_,
                                        self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                        self.fit_cate_intercept)[0]

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
        if not self.fit_cate_intercept:
            raise AttributeError("No intercept was fitted!")
        return parse_final_model_params(self.model_final.coef_, self.model_final.intercept_,
                                        self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                        self.fit_cate_intercept)[1]

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

    @BaseCateEstimator._defer_to_inference
    def summary(self, alpha=0.1, value=0, decimals=3, feat_name=None):
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
        feat_name: optional list of strings or None (default is None)
            The input of the feature names

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.
        """
        pass


class StatsModelsCateEstimatorMixin(LinearModelFinalCateEstimatorMixin):
    """
    Mixin class that offers `inference='statsmodels'` options to the CATE estimator
    that inherits it.

    Such an estimator must implement a :attr:`model_final` attribute that points
    to the fitted final :class:`.StatsModelsLinearRegression` object that
    represents the fitted CATE model.
    """

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(statsmodels=StatsModelsInference)
        return options


class DebiasedLassoCateEstimatorMixin(LinearModelFinalCateEstimatorMixin):
    """Mixin for cate models where the final stage is a debiased lasso model."""

    def _get_inference_options(self):
        # add debiasedlasso to parent's options
        options = super()._get_inference_options()
        options.update(debiasedlasso=LinearModelFinalInference)
        return options


class LinearModelFinalCateEstimatorDiscreteMixin(BaseCateEstimator):
    # TODO Share some logic with non-discrete version
    """
    Base class for models where the final stage is a linear model.

    Subclasses must expose a ``fitted_models_final`` attribute
    returning an array of the fitted models for each non-control treatment
    """

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
        _, T = self._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.fitted_models_final[ind].intercept_

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

    @BaseCateEstimator._defer_to_inference
    def summary(self, T, *, alpha=0.1, value=0, decimals=3, feat_name=None):
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
        feat_name: optional list of strings or None (default is None)
            The input of the feature names

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.
        """
        pass


class StatsModelsCateEstimatorDiscreteMixin(LinearModelFinalCateEstimatorDiscreteMixin):
    """
    Mixin class that offers `inference='statsmodels'` options to the CATE estimator
    that inherits it.

    Such an estimator must implement a :attr:`model_final` attribute that points
    to a :class:`.StatsModelsLinearRegression` object that is cloned to fit
    each discrete treatment target CATE model and a :attr:`fitted_models_final` attribute
    that returns the list of fitted final models that represent the CATE for each categorical treatment.
    """

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(statsmodels=StatsModelsInferenceDiscrete)
        return options


class DebiasedLassoCateEstimatorDiscreteMixin(LinearModelFinalCateEstimatorDiscreteMixin):
    """Mixin for cate models where the final stage is a debiased lasso model."""

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(debiasedlasso=LinearModelFinalInferenceDiscrete)
        return options
