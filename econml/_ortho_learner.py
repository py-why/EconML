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


def _crossfit(model, folds, *args, **kwargs):
    model_list = []
    for idx, (train_idxs, test_idxs) in enumerate(folds):
        model_list.append(clone(model, safe=False))

        args_train = ()
        args_test = ()
        for var in args:
            args_train += (var[train_idxs],)
            args_test += (var[test_idxs],)

        kwargs_train = {}
        kwargs_test = {}
        for key, var in kwargs.items():
            if var is not None:
                kwargs_train[key] = var[train_idxs]
                kwargs_test[key] = var[test_idxs]

        model_list[idx].fit(*args_train, **kwargs_train)

        nuisance_temp = model_list[idx].predict(*args_test, **kwargs_test)

        if not isinstance(nuisance_temp, tuple):
            nuisance_temp = (nuisance_temp,)

        if idx == 0:
            nuisances = tuple([np.zeros((args[0].shape[0],) + nuis.shape[1:]) for nuis in nuisance_temp])

        for it, nuis in enumerate(nuisance_temp):
            nuisances[it][test_idxs] = nuis

    return nuisances, model_list

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
    
    def _check_input_dims(self, Y, T, X=None, W=None, Z=None, sample_weight=None, sample_var=None):
        assert shape(Y)[0] == shape(T)[0], "Dimension mis-match!"
        assert (X is None) or (X.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (W is None) or (W.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (Z is None) or (Z.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (sample_weight is None) or (sample_weight.shape[0] == Y.shape[0]), "Dimension mis-match!"
        assert (sample_var is None) or (sample_var.shape[0] == Y.shape[0]), "Dimension mis-match!"
        self._d_x = X.shape[1:] if X is not None else None

    def _check_fitted_dims(self, X):
        if X is None:
            assert self._d_x is None, "X was not None when fitting, so can't be none for effect"
        else:
            assert self._d_x == X.shape[1:], "Dimension mis-match of X with fitted X"

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
        self._check_input_dims(Y, T, X, W, Z, sample_weight, sample_var)
        nuisances = self.fit_nuisances(Y, T, X, W, Z, sample_weight=sample_weight)
        self.fit_final(Y, T, X, W, Z, nuisances, sample_weight=sample_weight, sample_var=sample_var)
        return self

    def fit_nuisances(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        # use a binary array to get stratified split in case of discrete treatment
        splitter = check_cv(self._n_splits, [0], classifier=self._discrete_treatment)
        # if check_cv produced a new KFold or StratifiedKFold object, we need to set shuffle and random_state
        if splitter != self._n_splits and isinstance(splitter, (KFold, StratifiedKFold)):
            splitter.shuffle = True
            splitter.random_state = self._random_state

        all_vars = [var if np.ndim(var)==2 else var.reshape(-1, 1) for var in [Z, W, X] if var is not None]
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

        nuisances, fitted_models = _crossfit(self._model_nuisance, folds,
                                             Y, T, X=X, W=W, Z=Z,sample_weight=sample_weight)
        self._models_nuisance = fitted_models
        return nuisances

    def fit_final(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        self._model_final.fit(Y, T, X=X, W=W, Z=Z, nuisances=nuisances, sample_weight=sample_weight, sample_var=sample_var)

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
        self._check_fitted_dims(X)
        return self._model_final.predict(X)

    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        self._check_fitted_dims(X)
        return super().const_marginal_effect_interval(X, alpha=alpha)

    def effect_interval(self, X=None, T0=0, T1=1, *, alpha=0.1):
        self._check_fitted_dims(X)
        return super().effect_interval(X, T0=T0, T1=T1, alpha=alpha)

    def score(self, Y, T, X=None, W=None, Z=None):
        n_splits = len(self._models_nuisance)
        for idx, mdl in enumerate(self._models_nuisance):
            nuisance_temp = mdl.predict(Y, T, X, W, Z)
            if not isinstance(nuisance_temp, tuple):
                nuisance_temp = (nuisance_temp,)

            if idx == 0:
                nuisances = [np.zeros((n_splits,) + nuis.shape) for nuis in nuisance_temp]

            for it, nuis in enumerate(nuisance_temp):
                nuisances[it][idx] = nuis

        for it in range(len(nuisances)):
            nuisances[it] = np.mean(nuisances[it], axis=0)
        
        return self._model_final.score(Y, T, X=X, W=W, Z=Z, nuisances=tuple(nuisances))
    
    @property
    def model_final(self):
        return self._model_final
