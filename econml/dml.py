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
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from .cate_estimator import LinearCateEstimator


class _RLearner(LinearCateEstimator):
    """
    Base class for orthogonal learners.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features and controls. Must implement
        `fit` and `predict` methods.  Unlike sklearn estimators both methods must
        take an extra second argument (the controls).

    model_t: estimator
        The estimator for fitting the treatment to the features and controls. Must implement
        `fit` and `predict` methods.  Unlike sklearn estimators both methods must
        take an extra second argument (the controls).

    model_final: estimator for fitting the response residuals to the features and treatment residuals
        Must implement `fit` and `predict` methods. Unlike sklearn estimators the fit methods must
        take an extra second argument (the treatment residuals).  Predict, on the other hand,
        should just take the features and return the constant marginal effect.

    n_splits: int
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    inference: string, inference method, or None
        Method for performing inference.  This estimator supports 'bootstrap'
        (or an instance of `BootstrapOptions`)
    """

    def __init__(self, model_y, model_t, model_final,
                 discrete_treatment, n_splits, random_state, inference):
        self._models_y = [clone(model_y, safe=False) for _ in range(n_splits)]
        self._models_t = [clone(model_t, safe=False) for _ in range(n_splits)]
        self._model_final = clone(model_final, safe=False)
        self._n_splits = n_splits
        self._discrete_treatment = discrete_treatment
        self._random_state = check_random_state(random_state)
        super().__init__(inference=inference)

    def _fit_impl(self, Y, T, X=None, W=None, sample_weight=None):
        if X is None:
            X = np.ones((shape(Y)[0], 1))
        if W is None:
            W = np.empty((shape(Y)[0], 0))
        assert shape(Y)[0] == shape(T)[0] == shape(X)[0] == shape(W)[0]

        Y_res, T_res = self.fit_nuisances(Y, T, X, W, sample_weight=sample_weight)

        self.fit_final(X, Y_res, T_res, sample_weight=sample_weight)

        return self

    def fit_nuisances(self, Y, T, X, W, sample_weight=None):
        if self._discrete_treatment:
            folds = StratifiedKFold(self._n_splits, shuffle=True,
                                    random_state=self._random_state).split(np.empty_like(X), T)
            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
            T = self._label_encoder.fit_transform(T)
            T_out = self._one_hot_encoder.fit_transform(reshape(T, (-1, 1)))
            T_out = T_out[:, 1:]  # drop first column since all columns sum to one
        else:
            folds = KFold(self._n_splits, shuffle=True, random_state=self._random_state).split(X)
            T_out = T

        Y_res = np.zeros(shape(Y))
        T_res = np.zeros(shape(T_out))
        for idx, (train_idxs, test_idxs) in enumerate(folds):
            Y_train, Y_test = Y[train_idxs], Y[test_idxs]
            T_train, T_test = T[train_idxs], T_out[test_idxs]
            X_train, X_test = X[train_idxs], X[test_idxs]
            W_train, W_test = W[train_idxs], W[test_idxs]
            # TODO: If T is a vector rather than a 2-D array, then the model's fit must accept a vector...
            #       Do we want to reshape to an nx1, or just trust the user's choice of input?
            #       (Likewise for Y below)
            if sample_weight is not None:
                self._models_t[idx].fit(X_train, W_train, T_train, sample_weight=sample_weight[train_idxs])
            else:
                self._models_t[idx].fit(X_train, W_train, T_train)
            if self._discrete_treatment:
                T_res[test_idxs] = T_test - self._models_t[idx].predict(X_test, W_test)[:, 1:]
            else:
                T_res[test_idxs] = T_test - self._models_t[idx].predict(X_test, W_test)
            if sample_weight is not None:
                self._models_y[idx].fit(X_train, W_train, Y_train, sample_weight=sample_weight[train_idxs])
            else:
                self._models_y[idx].fit(X_train, W_train, Y_train)
            Y_res[test_idxs] = Y_test - self._models_y[idx].predict(X_test, W_test)

        return Y_res, T_res

    def fit_final(self, X, Y_res, T_res, sample_weight=None):
        if sample_weight is not None:
            self._model_final.fit(X, T_res, Y_res, sample_weight=sample_weight)
        else:
            self._model_final.fit(X, T_res, Y_res)

    def const_marginal_effect(self, X=None):
        """
        Calculate the constant marginal CATE θ(·).

        The marginal effect is conditional on a vector of
        features on a set of m test samples {Xᵢ}.

        Parameters
        ----------
        X: optional (m × dₓ) matrix
            Features for each sample.
            If X is None, it will be treated as a column of ones with a single row

        Returns
        -------
        theta: (m × d_y × dₜ) matrix
            Constant marginal CATE of each treatment on each outcome for each sample.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        if X is None:
            X = np.ones((1, 1))
        return self._model_final.predict(X)

    # need to override effect in case of discrete treatments
    # TODO: should this logic be moved up to the LinearCateEstimator class and
    #       removed from here and from the OrthoForest implementation?
    def effect(self, X, T0=0, T1=1):
        if ndim(T0) == 0:
            T0 = np.repeat(T0, 1 if X is None else shape(X)[0])
        if ndim(T1) == 0:
            T1 = np.repeat(T1, 1 if X is None else shape(X)[0])
        if self._discrete_treatment:
            T0 = self._one_hot_encoder.transform(reshape(self._label_encoder.transform(T0), (-1, 1)))[:, 1:]
            T1 = self._one_hot_encoder.transform(reshape(self._label_encoder.transform(T1), (-1, 1)))[:, 1:]
        return super().effect(X, T0, T1)

    def score(self, Y, T, X=None, W=None):
        if self._discrete_treatment:
            T = self._one_hot_encoder.transform(reshape(self._label_encoder.transform(T), (-1, 1)))[:, 1:]
        if T.ndim == 1:
            T = reshape(T, (-1, 1))
        if Y.ndim == 1:
            Y = reshape(Y, (-1, 1))
        if X is None:
            X = np.ones((shape(Y)[0], 1))
        if W is None:
            W = np.empty((shape(Y)[0], 0))
        Y_test_pred = np.zeros(shape(Y) + (self._n_splits,))
        T_test_pred = np.zeros(shape(T) + (self._n_splits,))
        for ind in range(self._n_splits):
            if self._discrete_treatment:
                T_test_pred[:, :, ind] = reshape(self._models_t[ind].predict(X, W)[:, 1:], shape(T))
            else:
                T_test_pred[:, :, ind] = reshape(self._models_t[ind].predict(X, W), shape(T))
            Y_test_pred[:, :, ind] = reshape(self._models_y[ind].predict(X, W), shape(Y))
        Y_test_pred = Y_test_pred.mean(axis=2)
        T_test_pred = T_test_pred.mean(axis=2)
        Y_test_res = Y - Y_test_pred
        T_test_res = T - T_test_pred
        effects = reshape(self._model_final.predict(X), (-1, shape(Y)[1], shape(T)[1]))
        Y_test_res_pred = reshape(np.einsum('ijk,ik->ij', effects, T_test_res), shape(Y))
        mse = ((Y_test_res - Y_test_res_pred)**2).mean()
        return mse


class DMLCateEstimator(_RLearner):
    """
    The base class for parametric Double ML estimators.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when sparseLinear is `True`.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when sparseLinear is `True`.

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    sparseLinear: bool
        Whether to use sparse linear model assumptions

    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    inference: string, inference method, or None
        Method for performing inference.  This estimator supports 'bootstrap'
        (or an instance of `BootstrapOptions`).
    """

    def __init__(self,
                 model_y, model_t, model_final,
                 featurizer,
                 sparseLinear=False,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None,
                 inference=None):

        class FirstStageWrapper:
            def __init__(self, model, is_Y):
                self._model = clone(model, safe=False)
                self._featurizer = clone(featurizer, safe=False)
                self._is_Y = is_Y

            def _combine(self, X, W):
                if self._is_Y and sparseLinear:
                    F = self._featurizer.fit_transform(X)
                    XW = hstack([X, W])
                    return cross_product(XW, hstack([np.ones((shape(XW)[0], 1)), F, W]))
                else:
                    return hstack([X, W])

            def fit(self, X, W, Target, sample_weight=None):
                if sample_weight is not None:
                    self._model.fit(self._combine(X, W), Target, sample_weight=sample_weight)
                else:
                    self._model.fit(self._combine(X, W), Target)

            def predict(self, X, W):
                if (not self._is_Y) and discrete_treatment:
                    return self._model.predict_proba(self._combine(X, W))
                else:
                    return self._model.predict(self._combine(X, W))

        class FinalWrapper:
            def __init__(self):
                self._model = clone(model_final, safe=False)
                self._featurizer = clone(featurizer, safe=False)

            def fit(self, X, T_res, Y_res, sample_weight=None):
                # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
                self._d_t = shape(T_res)[1:]
                self._d_y = shape(Y_res)[1:]
                if sample_weight is not None:
                    self._model.fit(cross_product(self._featurizer.fit_transform(X), T_res),
                                    Y_res, sample_weight=sample_weight)
                else:
                    self._model.fit(cross_product(self._featurizer.fit_transform(X), T_res), Y_res)

            def predict(self, X, T_res=None):
                # create an identity matrix of size d_t (or just a 1-element array if T was a vector)
                # the nth row will allow us to compute the marginal effect of the nth component of treatment
                eye = np.eye(self._d_t[0]) if self._d_t else np.array([1])
                # TODO: Doing this kronecker/reshaping/transposing stuff so that predict can be called
                #       rather than just using coef_ seems silly, but one benefit is that we can use linear models
                #       that don't expose a coef_ (e.g. a GridSearchCV over underlying linear models)
                flat_eye = reshape(eye, (1, -1))
                XT = reshape(np.kron(flat_eye, self._featurizer.fit_transform(X)),
                             ((self._d_t[0] if self._d_t else 1) * shape(X)[0], -1))
                effects = reshape(self._model.predict(XT), (-1,) + self._d_t + self._d_y)
                if self._d_t and self._d_y:
                    return transpose(effects, (0, 2, 1))  # need to return as m by d_y by d_t matrix
                else:
                    return effects

            @property
            def coef_(self):
                # TODO: handle case where final model doesn't directly expose coef_?
                return reshape(self._model.coef_, self._d_y + self._d_t + (-1,))

        super().__init__(model_y=FirstStageWrapper(model_y, is_Y=True),
                         model_t=FirstStageWrapper(model_t, is_Y=False),
                         model_final=FinalWrapper(),
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state,
                         inference=inference)

    @property
    def coef_(self):
        """
        Get the final model's coefficients.

        Note that this relies on the final model having a `coef_` property of its own.
        Most sklearn linear models support this, but there are cases that don't
        (e.g. a `Pipeline` or `GridSearchCV` which wraps a linear model)
        """
        return self._model_final.coef_


class LinearDMLCateEstimator(DMLCateEstimator):
    """
    The Double ML Estimator with a low-dimensional linear final stage.

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

    discrete_treatment: bool, optional (default is False)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    inference: string, inference method, or None
        Method for performing inference.  This estimator supports 'bootstrap'
        (or an instance of `BootstrapOptions`)
    """

    def __init__(self,
                 model_y=LassoCV(), model_t=LassoCV(), model_final=LinearRegression(fit_intercept=False),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None,
                 inference=None):
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=model_final,
                         featurizer=featurizer,
                         sparseLinear=True,
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state,
                         inference=inference)


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

    discrete_treatment: bool, optional (default is False)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    inference: string, inference method, or None
        Method for performing inference.  This estimator supports 'bootstrap'
        (or an instance of `BootstrapOptions`)
    """

    def __init__(self,
                 linear_model_y=LassoCV(), linear_model_t=LassoCV(), model_final=LinearRegression(fit_intercept=False),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None,
                 inference=None):
        super().__init__(model_y=linear_model_y,
                         model_t=linear_model_t,
                         model_final=model_final,
                         featurizer=featurizer,
                         sparseLinear=True,
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state,
                         inference=inference)


class KernelDMLCateEstimator(LinearDMLCateEstimator):
    """
    A specialized version of the linear Double ML Estimator that uses random fourier features.

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

    dim: int, optional (default is 20)
        The number of random Fourier features to generate

    bw: float, optional (default is 1.0)
        The bandwidth of the Gaussian used to generate features

    n_splits: int, optional (default is 2)
        The number of splits to use when fitting the first-stage models.

    random_state: int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    inference: string, inference method, or None
        Method for performing inference.  This estimator supports 'bootstrap'
        (or an instance of `BootstrapOptions`)
    """

    def __init__(self, model_y, model_t, model_final=LinearRegression(fit_intercept=False),
                 dim=20, bw=1.0, n_splits=2, random_state=None, inference=None):
        class RandomFeatures(TransformerMixin):
            def fit(innerself, X):
                innerself.omegas = self._random_state.normal(0, 1 / bw, size=(shape(X)[1], dim))
                innerself.biases = self._random_state.uniform(0, 2 * np.pi, size=(1, dim))
                return innerself

            def transform(innerself, X):
                return np.sqrt(2 / dim) * np.cos(np.matmul(X, innerself.omegas) + innerself.biases)

        super().__init__(model_y=model_y, model_t=model_t, model_final=model_final,
                         featurizer=RandomFeatures(), n_splits=n_splits, random_state=random_state,
                         inference=inference)
