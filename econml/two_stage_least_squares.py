# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Provides a non - parametric two - stage least squares instrumental variable estimator."""

import numpy as np
from copy import deepcopy
from sklearn import clone
from sklearn.linear_model import LinearRegression
from .utilities import shape, transpose, reshape, cross_product, ndim, size
from .cate_estimator import BaseCateEstimator, LinearCateEstimator
from numpy.polynomial.hermite_e import hermeval
from sklearn.base import TransformerMixin
from itertools import product


class HermiteFeatures(TransformerMixin):
    """
    Featurizer that returns(unscaled) Hermite function evaluations.

    The evaluated functions are of degrees 0..`degree`, differentiated `shift` times.

    If the input has shape(n, x) and `joint` is False, the output will have shape(n, (`degree`+ 1)×x) if `shift` is 0.
    If the input has shape(n, x) and `joint` is True, the output will have shape(n, (`degree`+ 1) ^ x) if `shift` is 0.
    In either case, if `shift` is nonzero there will be `shift` additional dimensions of size x
    between the first and last.
    """

    def __init__(self, degree, shift=0, joint=False):
        self._degree = degree
        self._shift = shift
        self._joint = joint

    def _column_feats(self, X, shift):
        """
        Apply Hermite function evaluations of degrees 0..`degree` differentiated `shift` times.

        When applied to the column `X` of shape(n,), the resulting array has shape(n, (degree + 1)).
        """
        assert ndim(X) == 1
        # this will have dimension (d,) + shape(X)
        coeffs = np.identity(self._degree + shift + 1)[:, shift:]
        feats = ((-1) ** shift) * hermeval(X, coeffs) * np.exp(-X * X / 2)
        # send the first dimension to the end
        return transpose(feats)

    def fit(self, X):
        """Fits the data(a NOP for this class) and returns self."""
        return self

    def transform(self, X):
        """
        Transform the data by applying the appropriate Hermite functions.

        Parameters
        ----------
        X: array_like
            2-dimensional array of input features

        Returns
        -------
        The transformed data
        """
        assert ndim(X) == 2
        n = shape(X)[0]
        ncols = shape(X)[1]
        columns = []
        for indices in product(*[range(ncols) for i in range(self._shift)]):
            if self._joint:
                columns.append(cross_product(*[self._column_feats(X[:, i], indices.count(i))
                                               for i in range(shape(X)[1])]))
            else:
                indices = set(indices)
                if self._shift == 0:  # return features for all columns:
                    columns.append(np.hstack([self._column_feats(X[:, i], self._shift) for i in range(shape(X)[1])]))
                # columns are featurized independently; partial derivatives are only non-zero
                # when taken with respect to the same column each time
                elif len(indices) == 1:
                    index = list(indices)[0]
                    feats = self._column_feats(X[:, index], self._shift)
                    columns.append(np.hstack([feats if i == index else np.zeros(shape(feats))
                                              for i in range(shape(X)[1])]))
                else:
                    columns.append(np.zeros((n, (self._degree + 1) * ncols)))
        return reshape(np.hstack(columns), (n,) + (ncols,) * self._shift + (-1,))


def _add_ones(arr):
    """Add a column of ones to the front of an array."""
    return np.hstack([np.ones((shape(arr)[0], 1)), arr])


def _add_zeros(arr):
    """Add a column of zeros to the front of an array."""
    return np.hstack([np.zeros((shape(arr)[0], 1)), arr])


class NonparametricTwoStageLeastSquares(BaseCateEstimator):
    """
    Non-parametric instrumental variables estimator.

    Supports the use of arbitrary featurizers for the features, treatments, and instruments.

    Parameters
    ----------
    t_featurizer: transformer
        Featurizer used to transform the treatments

    x_featurizer: transformer
        Featurizer used to transform the raw features

    z_featurizer: transformer
        Featurizer used to transform the instruments

    dt_featurizer: transformer
        Featurizer used to transform the treatments for the computation of the marginal effect.
        This should produce a 3 - dimensional array, containing the per - treatment derivative of
        each transformed treatment. That is, given a treatment array of shape(n, dₜ),
        the output should have shape(n, dₜ, fₜ), where fₜ is the number of columns produced by `t_featurizer`.
    """

    def __init__(self, t_featurizer, x_featurizer, z_featurizer, dt_featurizer):
        self._t_featurizer = clone(t_featurizer, safe=False)
        self._x_featurizer = clone(x_featurizer, safe=False)
        self._z_featurizer = clone(z_featurizer, safe=False)
        self._dt_featurizer = clone(dt_featurizer, safe=False)
        # don't fit intercept; manually add column of ones to the data instead;
        # this allows us to ignore the intercept when computing marginal effects
        self._model_T = LinearRegression(fit_intercept=False)
        self._model_Y = LinearRegression(fit_intercept=False)

    def fit(self, Y, T, X, Z):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·, ·, ·), ∂τ(·, ·).

        Parameters
        ----------
        Y: (n × d_y) matrix
            Outcomes for each sample
        T: (n × dₜ) matrix
            Treatments for each sample
        X: optional(n × dₓ) matrix
            Features for each sample
        Z: optional(n × d_z) matrix
            Instruments for each sample

        Returns
        -------
        self

        """
        if X is None:
            X = np.empty((shape(Y)[0], 0))
        assert shape(Y)[0] == shape(T)[0] == shape(X)[0] == shape(Z)[0]
        # two stage approximation
        # first, get basis expansions of T, X, and Z
        # regress T expansion on X,Z expansions
        ft_X = self._x_featurizer.fit_transform(X)
        ft_Z = self._z_featurizer.fit_transform(Z)
        ft_T = self._t_featurizer.fit_transform(T)
        # predict ft_T from interacted ft_X, ft_Z
        features = _add_ones(cross_product(ft_X, ft_Z))
        self._model_T.fit(features, ft_T)
        ft_T_hat = self._model_T.predict(features)
        self._model_Y.fit(_add_ones(cross_product(ft_T_hat, ft_X)), Y)
        return self

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
        if ndim(T0) == 0:
            T0 = np.repeat(T0, 1 if X is None else shape(X)[0])
        if ndim(T1) == 0:
            T1 = np.repeat(T1, 1 if X is None else shape(X)[0])
        if X is None:
            X = np.empty((shape(T0)[0], 0))
        assert shape(T0) == shape(T1)
        assert shape(T0)[0] == shape(X)[0]
        ft_X = self._x_featurizer.fit_transform(X)
        ft_T0 = self._t_featurizer.fit_transform(T0)
        ft_T1 = self._t_featurizer.fit_transform(T1)
        Y0 = self._model_Y.predict(_add_ones(cross_product(ft_T0, ft_X)))
        Y1 = self._model_Y.predict(_add_ones(cross_product(ft_T1, ft_X)))
        return Y1 - Y0

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
        if X is None:
            X = np.empty((shape(T)[0], 0))
        assert shape(T)[0] == shape(X)[0]
        ft_X = self._x_featurizer.fit_transform(X)
        n = shape(T)[0]
        dT = self._dt_featurizer.fit_transform(T)
        # dT should be an n×dₜ×fₜ array (but if T was a vector, or if there is only one feature,
        # dT may be only 2-dimensional)
        # promote dT to 3D if necessary (e.g. if T was a vector)
        if ndim(dT) < 3:
            dT = reshape(dT, (n, 1, shape(dT)[1]))

        # reshape ft_X and dT to allow cross product (result has shape n×dₜ×fₜ×f_x)
        features = reshape(ft_X, (n, 1, 1, -1)) * reshape(dT, shape(dT) + (1,))

        features = reshape(features, (size(T), -1))
        output = self._model_Y.predict(_add_zeros(features))
        return reshape(output, shape(T) + (shape(output)[-1],))
