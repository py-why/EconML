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
from .utilities import tensordot, ndim, reshape, shape, check_treatments
from .inference import StatsModelsInference


class BaseCateEstimator(metaclass=abc.ABCMeta):
    """Base class for all CATE estimators in this package."""

    def _get_inference_options(self):
        """
        Produce a dictionary mapping string names to `Inference` types.

        This is used by the `fit` method when a string is passed rather than an `Inference` type.
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
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

        Note that the signature of this method may vary in subclasses (e.g. classes that don't
        support instruments will not allow a `Z` argument)

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: optional (n × dₓ) matrix
            Features for each sample
        W: optional (n × d_w) matrix
            Controls for each sample
        Z: optional (n × d_z) matrix
            Instruments for each sample
        inference: optional string, `Inference` instance, or None
            Method for performing inference.  All estimators support 'bootstrap'
            (or an instance of `BootstrapInference`), some support other methods as well.

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
    def effect(self, X=None, T0=0, T1=1):
        """
        Calculate the heterogeneous treatment effect τ(·,·,·).

        The effect is calculated between the two treatment points
        conditional on a vector of features on a set of m test samples {T0ᵢ, T1ᵢ, Xᵢ}.

        Parameters
        ----------
        T0: (m × dₜ) matrix or vector of length m
            Base treatments for each sample
        T1: (m × dₜ) matrix or vector of length m
            Target treatments for each sample
        X: optional (m × dₓ) matrix
            Features for each sample

        Returns
        -------
        τ: (m × d_y) matrix
            Heterogeneous treatment effects on each outcome for each sample
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        pass

    @abc.abstractmethod
    def marginal_effect(self, T, X=None):
        """
        Calculate the heterogeneous marginal effect ∂τ(·,·).

        The marginal effect is calculated around a base treatment
        point conditional on a vector of features on a set of m test samples {Tᵢ, Xᵢ}.

        Parameters
        ----------
        T: (m × dₜ) matrix
            Base treatments for each sample
        X: optional (m × dₓ) matrix
            Features for each sample

        Returns
        -------
        grad_tau: (m × d_y × dₜ) array
            Heterogeneous marginal effects on each outcome for each sample
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pass

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
        pass

    @_defer_to_inference
    def marginal_effect_interval(self, T, X=None, *, alpha=0.1):
        pass


class LinearCateEstimator(BaseCateEstimator):
    """Base class for all CATE estimators with linear treatment effects in this package."""

    @abc.abstractmethod
    def const_marginal_effect(self, X=None):
        """
        Calculate the constant marginal CATE θ(·).

        The marginal effect is conditional on a vector of
        features on a set of m test samples {Xᵢ}.

        Parameters
        ----------
        X: optional (m × dₓ) matrix
            Features for each sample

        Returns
        -------
        theta: (m × d_y × dₜ) matrix
            Constant marginal CATE of each treatment on each outcome for each sample.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        pass

    def effect(self, X=None, T0=0, T1=1):
        """
        Calculate the heterogeneous treatment effect τ(·,·,·).

        The effect is calculatred between the two treatment points
        conditional on a vector of features on a set of m test samples {T0ᵢ, T1ᵢ, Xᵢ}.
        Since this class assumes a linear effect, only the difference between T0ᵢ and T1ᵢ
        matters for this computation.

        Parameters
        ----------
        T0: (m × dₜ) matrix
            Base treatments for each sample
        T1: (m × dₜ) matrix
            Target treatments for each sample
        X: optional (m × dₓ) matrix
            Features for each sample

        Returns
        -------
        τ: (m × d_y) matrix (or length m vector if Y was a vector)
            Heterogeneous treatment effects on each outcome for each sample.
            Note that when Y is a vector rather than a 2-dimensional array, the corresponding
            singleton dimension will be collapsed (so this method will return a vector)
        """
        # TODO: what if input is sparse? - there's no equivalent to einsum,
        #       but tensordot can't be applied to this problem because we don't sum over m
        eff = self.const_marginal_effect(X)
        m = shape(eff)[0]
        T0, T1 = check_treatments(T0, T1, m, self._d_t)
        dT = T1 - T0
        einsum_str = 'myt,mt->my'
        if ndim(dT) == 1:
            einsum_str = einsum_str.replace('t', '')
        if ndim(eff) == ndim(dT):  # y is a vector, rather than a 2D array
            einsum_str = einsum_str.replace('y', '')
        return np.einsum(einsum_str, eff, dT)

    def marginal_effect(self, T, X=None):
        """
        Calculate the heterogeneous marginal effect ∂τ(·,·).

        The marginal effect is calculated around a base treatment
        point conditional on a vector of features on a set of m test samples {Tᵢ, Xᵢ}.
        Since this class assumes a linear model, the base treatment is ignored in this calculation.

        Parameters
        ----------
        T: (m × dₜ) matrix
            Base treatments for each sample
        X: optional (m × dₓ) matrix
            Features for each sample

        Returns
        -------
        grad_tau: (m × d_y × dₜ) array
            Heterogeneous marginal effects on each outcome for each sample
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return self.const_marginal_effect(X)

    def marginal_effect_interval(self, T, X=None, *, alpha=0.1):
        return self.const_marginal_effect_interval(X=X, alpha=alpha)

    @BaseCateEstimator._defer_to_inference
    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        pass


class StatsModelsCateEstimatorMixin(BaseCateEstimator):

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(statsmodels=StatsModelsInference)
        return options

    @property
    @abc.abstractmethod
    def statsmodels(self):
        pass

    @property
    def coef_(self):
        return self.statsmodels.coef_

    @property
    def intercept_(self):
        return self.statsmodels.intercept_

    @BaseCateEstimator._defer_to_inference
    def coef__interval(self, *, alpha=0.1):
        pass

    @BaseCateEstimator._defer_to_inference
    def intercept__interval(self, *, alpha=0.1):
        pass
