# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Double ML.

"Double Machine Learning" is an algorithm that applies arbitrary machine learning methods
to fit the treatment and response, then uses a linear model to predict the response residuals
from the treatment residuals.

"""

import numpy as np
import copy
from warnings import warn
from .utilities import (shape, reshape, ndim, hstack, cross_product, transpose,
                        broadcast_unit_treatments, reshape_treatmentwise_effects,
                        StatsModelsLinearRegression, LassoCVWrapper)
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import (PolynomialFeatures, LabelEncoder, OneHotEncoder,
                                   FunctionTransformer)
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from .cate_estimator import (BaseCateEstimator, LinearCateEstimator,
                             TreatmentExpansionMixin, StatsModelsCateEstimatorMixin)
from .inference import StatsModelsInference


class _OrthoLearner(TreatmentExpansionMixin, LinearCateEstimator):
    """
    Base class for all orthogonal learners.

    Parameters
    ----------
    model_nuisance: estimator
        The estimator for fitting the nuisance function. Must implement
        `fit` and `predict` methods that both take as input Y, T, X, W, Z.

    model_final: estimator for fitting the response residuals to the features and treatment residuals
        Must implement `fit` and `predict` methods. The fit method takes as input, Y, T, X, W, Z, nuisances.
        Predict, on the other hand, should just take the features X and return the constant marginal effect.

    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by `np.random`.
    """

    def __init__(self, model_nuisance, model_final,
                 discrete_treatment, n_splits, random_state):
        self._model_nuisance = clone(model_nuisance, safe=False)
        self._models_nuisance = []
        self._model_final = clone(model_final, safe=False)
        self._n_splits = n_splits
        self._discrete_treatment = discrete_treatment
        self._random_state = check_random_state(random_state)
        if discrete_treatment:
            self._label_encoder = LabelEncoder()
            self._one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, sample_var=None, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

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
        sample_weight: optional (n,) vector
            Weights for each row
        sample_var: optional (n,) vector
            Sample variance 
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`).

        Returns
        -------
        self
        """
        self._X_is_None = (X is None)
        nuisances = self.fit_nuisances(Y, T, X, W, Z, sample_weight=sample_weight)
        self.fit_final(Y, T, X, W, Z, nuisances, sample_weight=sample_weight, sample_var=sample_var)

    def fit_nuisances(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        # use a binary array to get stratified split in case of discrete treatment
        splitter = check_cv(self._n_splits, [0], classifier=self._discrete_treatment)
        # if check_cv produced a new KFold or StratifiedKFold object, we need to set shuffle and random_state
        if splitter != self._n_splits and isinstance(splitter, (KFold, StratifiedKFold)):
            splitter.shuffle = True
            splitter.random_state = self._random_state

        all_vars = [var if ndim(x)==2 else var.reshape(-1, 1) for var in [Z, W, X] if var is not None]
        if all_vars:
            all_vars = np.hstack(all_vars)
            folds = splitter.split(all_vars, T)
        else:
            folds = splitter.split(np.ones((T.shape[0], 1)), T)

        if self._discrete_treatment:
            T = self._label_encoder.fit_transform(T)
            T = self._one_hot_encoder.fit_transform(reshape(T, (-1, 1)))[:, 1:] # drop first column since all columns sum to one
            self._d_t = shape(T)[1:]
            self.transformer = FunctionTransformer(
                func=(lambda T:
                      self._one_hot_encoder.transform(
                          reshape(self._label_encoder.transform(T), (-1, 1)))[:, 1:]),
                validate=False)

        for idx, (train_idxs, test_idxs) in enumerate(folds):
            self._models_nuisance.append(clone(self._model_nuisance, safe=False))
            Y_train, Y_test = Y[train_idxs], Y[test_idxs]
            T_train, T_test = T[train_idxs], T[test_idxs]
            X_train, X_test = X[train_idxs], X[test_idxs] if X is not None else None, None
            W_train, W_test = W[train_idxs], W[test_idxs] if W is not None else None, None
            Z_train, Z_test = Z[train_idxs], Z[test_idxs] if Z is not None else None, None

            if sample_weight is not None:
                self._models_nuisance[idx].fit(Y_train, T_train, X=X_train, W=W_train, Z=Z_train, sample_weight=sample_weight[train_idxs])
            else:
                self._models_nuisance[idx].fit(Y_train, T_train, X=X_train, W=W_train, Z=Z_train)
            
            nuisance_temp = self._models_nuisance[idx].predict(Y_test, T_test, X=X_test, W=W_test, Z=Z_test)
            if not isinstance(nuisance_temp, tuple):
                nuisance_temp = (nuisance_temp,)
            
            if idx == 0:
                    nuisances = tuple([np.array((Y.shape[0], nuis.shape[1]) for nuis in nuisance_temp])
            
            for it, nuis in enumerate(nuisance_temp):    
                nuisances[it][test_idxs] = nuis

        return nuisances

    def fit_final(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        self._model_final.fit(Y, T, X=X, W=W, Z=Z, nuisances=nuisances, sample_weight=sample_weight, sample_var=sample_var))

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
            assert self._X_is_None, "X was not None when fitting, so can't be none for effect"
        return self._model_final.predict(X)

    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        if X is None:
            assert self._X_is_None, "X was not None when fitting, so can't be none for effect"
        return super().const_marginal_effect_interval(X, alpha=alpha)

    def effect_interval(self, X=None, T0=0, T1=1, *, alpha=0.1):
        if X is None:
            assert self._X_is_None, "X was not None when fitting, so can't be none for effect"
        return super().effect_interval(X, T0=T0, T1=T1, alpha=alpha)

    def score(self, Y, T, X=None, W=None, Z=None):
        n_splits = len(self._models_nuisance)
        for idx, mdl in enumerate(self._models_nuisance):
            nuisance_temp = mdl.predict(Y, T, X, W, Z)
            if not isinstance(nuisance_temp, tuple):
                nuisance_temp = (nuisance_temp,)

            if idx == 0:
                    nuisances = tuple([np.array((Y.shape[0], nuis.shape[1], n_splits) for nuis in nuisance_temp])

            for it, nuis in enumerate(nuisance_temp):    
                nuisances[it][:, idx] = nuis

        for it in range(len(nuisances)):
            nuisances[it] = np.mean(nuisances[it], axis=2)
        
        return self._model_final.score(Y, T, X=X, W=W, Z=Z, nuisances=nuisances)
    
    @property
    def model_final(self):
        return self._model_final


class _RLearner(TreatmentExpansionMixin, _OrthoLearner):
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

    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by `np.random`.
    """

    def __init__(self, model_y, model_t, model_final,
                 discrete_treatment, n_splits, random_state):
        self._model_y = clone(model_y, safe=False)
        self._model_t = clone(model_t, safe=False)
        self._models_y = []
        self._models_t = []
        self._model_final = clone(model_final, safe=False)
        self._n_splits = n_splits
        self._discrete_treatment = discrete_treatment
        

        class ModelNuisance:
            def __init__(self, model_y, model_t, discrete_treatment):
                self._model_y = clone(model_y, safe=False)
                self._model_t = clone(model_t, safe=False)
                self._discrete_treatment = discrete_treatment
                self._random_state = check_random_state(random_state)

            @staticmethod
            def _check_X_W(X, W, Y):
                if X is None:
                    X = np.ones((shape(Y)[0], 1))
                if W is None:
                    W = np.empty((shape(Y)[0], 0))
                return X, W

            def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
                X, W = self._check_X_W(X, W, Y)
                assert Z is None, "Cannot accept instrument!"
                assert shape(Y)[0] == shape(T)[0] == shape(X)[0] == shape(W)[0], "Dimension mis-match!"
                self._d_x = shape(X)[1:]
                self._model_t.fit(X, W, T, sample_weight=sample_weight)
                self._model_y.fit(X, W, Y, sample_weight=sample_weight)
                return self
            
            def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
                Y_res = Y - self._model_y.predict(X, W).reshape(Y.shape)
                T_res = T - self._model_t.predict(X, W)
                return Y_res, T_res

        class ModelFinal:
            def __init__(self, model_final):
                self._model_final = clone(model_final, safe=False)

            def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
                Y_res, T_res = nuisances
                self._model_final.fit(X, T_res, Y_res, sample_weight=sample_weight, sample_var=sample_var)
                return self

            def predict(self, X):
                if X is None:
                    X = np.ones((1, 1))
                return self._model_final.predict(X)

            def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
                Y_res, T_res = nuisances
                effects = reshape(self._model_final.predict(X), (-1, shape(Y)[1], shape(T_res)[1]))
                Y_res_pred = reshape(np.einsum('ijk,ik->ij', effects, T_res), shape(Y))
                return ((Y_res - Y_res_pred)**2).mean()

        super().__init__(ModelNuisance(model_y, model_t, discrete_treatment),
                         ModelFinal(model_final), discrete_treatment, n_splits, random_state)
    
    @property
    def model_final(self):
        super().model_final
