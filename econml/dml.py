# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Double ML.

"Double Machine Learning" is an algorithm that applies arbitrary machine learning methods
to fit the treatment and response, then uses a linear model to predict the response residuals
from the treatment residuals.

"""

import numpy as np
import copy
from .utilities import shape, reshape, ndim, hstack, cross_product, transpose
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from .cate_estimator import LinearCateEstimator


class DMLCateEstimator(LinearCateEstimator):
    """
    The Double ML Estimator.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.

    model_final: estimator, optional (default is `LinearRegression(fit_intercept=False)`)
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: transformer, optional (default is `PolynomialFeatures(degree=1, include_bias=True)`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.

    """

    def __init__(self,
                 model_y, model_t, model_final=LinearRegression(fit_intercept=False),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 n_splits=2):
        self._models_y = [clone(model_y) for _ in range(n_splits)]
        self._models_t = [clone(model_t) for _ in range(n_splits)]
        self._model_final = clone(model_final)
        self._featurizer = clone(featurizer, safe=False)
        self._n_splits = n_splits

    def fit(self, Y, T, X=None, W=None):
        """
        Fit (Y, T, X, W).

        Parameters
        ----------
        Y : array_like, shape (n, d_y)
            Outcome for the treatment policy.

        T : array_like, shape (n, d_t)
            Treatment policy.

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array-like, shape (n, d_w) or None (default=None)
            High-dimensional controls.

        Returns
        -------
        self: an instance of self.

        """
        if X is None:
            X = np.empty((shape(Y)[0], 0))
        if W is None:
            W = np.empty((shape(Y)[0], 0))
        assert shape(Y)[0] == shape(T)[0] == shape(X)[0] == shape(W)[0]

        # Handle case where Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T)[1] if ndim(T) == 2 else 1
        self._d_y = shape(Y)[1] if ndim(Y) == 2 else 1

        y_res = np.zeros(shape(Y))
        t_res = np.zeros(shape(T))
        for idx, (train_idxs, test_idxs) in enumerate(KFold(self._n_splits).split(X)):
            Y_train, Y_test = Y[train_idxs], Y[test_idxs]
            T_train, T_test = T[train_idxs], T[test_idxs]
            X_train, X_test = X[train_idxs], X[test_idxs]
            W_train, W_test = W[train_idxs], W[test_idxs]
            # TODO: If T is a vector rather than a 2-D array, then the model's fit must accept a vector...
            #       Do we want to reshape to an nx1, or just trust the user's choice of input?
            #       (Likewise for Y below)
            self._models_t[idx].fit(hstack([X_train, W_train]), T_train)
            t_res[test_idxs] = T_test - self._models_t[idx].predict(hstack([X_test, W_test]))
            # NOTE: the fact that we stack X first then W is relied upon
            #       by the SparseLinearDMLCateEstimator implementation;
            #       if it's changed here then it needs to be changed there, too
            self._models_y[idx].fit(hstack([X_train, W_train]), Y_train)
            y_res[test_idxs] = Y_test - self._models_y[idx].predict(hstack([X_test, W_test]))

        phi_X = self._featurizer.fit_transform(X)
        self._d_phi = shape(phi_X)[1]

        self._model_final.fit(cross_product(phi_X, t_res), y_res)

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
        if X is None:
            X = np.empty((1, 0))
        # TODO: Doing this kronecker/reshaping/transposing stuff so that predict can be called
        #       rather than just using coef_ seems silly, but one benefit is that we can use linear models
        #       that don't expose a coef_ (e.g. a GridSearchCV over underlying linear models)
        flat_eye = reshape(np.eye(self._d_t), (1, -1))
        XT = reshape(np.kron(flat_eye, self._featurizer.fit_transform(X)),
                     (self._d_t * shape(X)[0], -1))
        effects = reshape(self._model_final.predict(XT), (-1, self._d_t, self._d_y))
        return transpose(effects, (0, 2, 1))  # need to return as m by d_y by d_t matrix

    @property
    def coef_(self):
        """
        Get the final model's coefficients.

        Note that this relies on the final model having a `coef_` property of its own.
        Most sklearn linear models support this, but there are cases that don't
        (e.g. a `Pipeline` or `GridSearchCV` which wraps a linear model)
        """
        # TODO: handle case where final model doesn't directly expose coef_?
        return reshape(self._model_final.coef_, (self._d_y, self._d_t, self._d_phi))


class SparseLinearDMLCateEstimator(DMLCateEstimator):
    """
    A specialized version of the Double ML estimator for the sparse linear case.

    Specifically, this estimator can be used when the controls are high-dimensional,
    the treatment and response are linear functions of the features and controls,
    and the coefficients of the nuisance functions are sparse.

    Parameters
    ----------
    linear_model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    linear_model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    model_final: estimator, optional (default is `LinearRegression(fit_intercept=False)`)
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: transformer, optional (default is `PolynomialFeatures(degree=1, include_bias=True)`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.
    """

    def __init__(self,
                 linear_model_y=LassoCV(), linear_model_t=LassoCV(), model_final=LinearRegression(fit_intercept=False),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 n_splits=2):
        def transform(XW):
            # TODO: yuck, is there any way to avoid having to know the structure of XW
            #       and the size of X to apply the features here?
            X = XW[:, :self._d_x]
            W = XW[:, self._d_x:]
            F = featurizer.fit_transform(X)
            return cross_product(XW, hstack([np.ones((shape(XW)[0], 1)), F, W]))

        model_y = Pipeline([("transform", FunctionTransformer(transform)), ("model", linear_model_y)])

        super().__init__(model_y, linear_model_t, model_final, featurizer, n_splits)

    def fit(self, Y, T, X=None, W=None):
        """
        Fit  (Y, T, X, W).

        Parameters
        ----------
        Y : array_like, shape (n, d_y)
            Outcome for the treatment policy.

        T : array_like, shape (n, d_t)
            Treatment policy.

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array-like, shape (n, d_w) or None (default=None)
            High-dimensional controls.

        Returns
        -------
        self: an instance of self.

        """
        self._d_x = shape(X)[1]
        super().fit(Y, T, X, W)
