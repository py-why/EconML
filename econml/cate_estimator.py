# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all CATE estimators."""

import abc
import numpy as np
from .utilities import tensordot, ndim, reshape


class BaseCateEstimator:
    """Base class for all CATE estimators in this package."""

    @abc.abstractmethod
    def fit(self, Y, T, X=None, W=None, Z=None):
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

        Returns
        -------
        self

        """
        pass

    @abc.abstractmethod
    def effect(self, T0, T1, X=None):
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

    def effect(self, T0, T1, X=None):
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
        dT = T1 - T0
        eff = self.const_marginal_effect(X)
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
