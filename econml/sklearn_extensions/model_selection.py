# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
"""Collection of scikit-learn extensions for model selection techniques."""

from inspect import signature
import inspect
import numbers
from typing import List, Optional
import warnings
import abc

import numpy as np
from collections.abc import Iterable
import scipy.sparse as sp
import sklearn
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import (ElasticNet, ElasticNetCV, Lasso, LassoCV, MultiTaskElasticNet, MultiTaskElasticNetCV,
                                  MultiTaskLasso, MultiTaskLassoCV, Ridge, RidgeCV, RidgeClassifier, RidgeClassifierCV,
                                  LogisticRegression, LogisticRegressionCV)
from sklearn.model_selection import (BaseCrossValidator, GridSearchCV, GroupKFold, KFold,
                                     RandomizedSearchCV, StratifiedKFold,
                                     check_cv)
# TODO: conisder working around relying on sklearn implementation details
from sklearn.model_selection._validation import (_check_is_permutation,
                                                 _fit_and_predict)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures, StandardScaler
from sklearn.utils import check_random_state, indexable
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import _num_samples

from .linear_model import WeightedLassoCVWrapper, WeightedLassoWrapper


def _split_weighted_sample(self, X, y, sample_weight, is_stratified=False):
    random_state = self.random_state if self.shuffle else None
    if is_stratified:
        kfold_model = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                      random_state=random_state)
    else:
        kfold_model = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                            random_state=random_state)

    if sample_weight is None:
        return kfold_model.split(X, y)
    else:
        random_state = self.random_state
        kfold_model.shuffle = True
        kfold_model.random_state = random_state

    weights_sum = np.sum(sample_weight)
    max_deviations = []
    all_splits = []
    for _ in range(self.n_trials + 1):
        splits = [test for (train, test) in list(kfold_model.split(X, y))]
        weight_fracs = np.array([np.sum(sample_weight[split]) / weights_sum for split in splits])
        if np.all(weight_fracs > .95 / self.n_splits):
            # Found a good split, return.
            return self._get_folds_from_splits(splits, X.shape[0])
        # Record all splits in case the stratification by weight yeilds a worse partition
        all_splits.append(splits)
        max_deviation = np.max(np.abs(weight_fracs - 1 / self.n_splits))
        max_deviations.append(max_deviation)
        # Reseed random generator and try again
        if isinstance(kfold_model.random_state, numbers.Integral):
            kfold_model.random_state = kfold_model.random_state + 1
        elif kfold_model.random_state is not None:
            kfold_model.random_state = np.random.RandomState(kfold_model.random_state.randint(np.iinfo(np.int32).max))

    # If KFold fails after n_trials, we try the next best thing: stratifying by weight groups
    warnings.warn("The KFold algorithm failed to find a weight-balanced partition after " +
                  "{n_trials} trials. Falling back on a weight stratification algorithm.".format(
                      n_trials=self.n_trials), UserWarning)
    if is_stratified:
        stratified_weight_splits = [[]] * self.n_splits
        for y_unique in np.unique(y.flatten()):
            class_inds = np.argwhere(y == y_unique).flatten()
            class_splits = self._get_splits_from_weight_stratification(sample_weight[class_inds])
            stratified_weight_splits = [split + list(class_inds[class_split]) for split, class_split in zip(
                stratified_weight_splits, class_splits)]
    else:
        stratified_weight_splits = self._get_splits_from_weight_stratification(sample_weight)
    weight_fracs = np.array([np.sum(sample_weight[split]) / weights_sum for split in stratified_weight_splits])

    if np.all(weight_fracs > .95 / self.n_splits):
        # Found a good split, return.
        return self._get_folds_from_splits(stratified_weight_splits, X.shape[0])
    else:
        # Did not find a good split
        # Record the devaiation for the weight-stratified split to compare with KFold splits
        all_splits.append(stratified_weight_splits)
        max_deviation = np.max(np.abs(weight_fracs - 1 / self.n_splits))
        max_deviations.append(max_deviation)
    # Return most weight-balanced partition
    min_deviation_index = np.argmin(max_deviations)
    return self._get_folds_from_splits(all_splits[min_deviation_index], X.shape[0])


class WeightedKFold:
    """K-Folds cross-validator for weighted data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k folds of roughly equal size and equal total weight.

    The default is to try sklearn.model_selection.KFold a number of trials to find
    a weight-balanced k-way split. If it cannot find such a split, it will fall back
    onto a more rigorous weight stratification algorithm.

    Parameters
    ----------
    n_splits : int, default 3
        Number of folds. Must be at least 2.

    n_trials : int, default 10
        Number of times to try sklearn.model_selection.KFold before falling back to another
        weight stratification algorithm.

    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance, or None, default None
            If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`. Used when ``shuffle`` == True.
    """

    def __init__(self, n_splits=3, n_trials=10, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.n_trials = n_trials
        self.random_state = random_state

    def split(self, X, y, sample_weight=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array_like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array_like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=False)

    def get_n_splits(self, X, y, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def _get_folds_from_splits(self, splits, sample_size):
        folds = []
        sample_indices = np.arange(sample_size)
        for it in range(self.n_splits):
            folds.append([np.setdiff1d(sample_indices, splits[it], assume_unique=True), splits[it]])
        return folds

    def _get_splits_from_weight_stratification(self, sample_weight):
        # Weight stratification algorithm
        # Sort weights for weight strata search
        random_state = check_random_state(self.random_state)
        sorted_inds = np.argsort(sample_weight)
        sorted_weights = sample_weight[sorted_inds]
        max_split_size = sorted_weights.shape[0] // self.n_splits
        max_divisible_length = max_split_size * self.n_splits
        sorted_inds_subset = np.reshape(sorted_inds[:max_divisible_length], (max_split_size, self.n_splits))
        shuffled_sorted_inds_subset = np.apply_along_axis(random_state.permutation, axis=1, arr=sorted_inds_subset)
        splits = [list(shuffled_sorted_inds_subset[:, i]) for i in range(self.n_splits)]
        if max_divisible_length != sorted_weights.shape[0]:
            # There are some leftover indices that have yet to be assigned
            subsample = sorted_inds[max_divisible_length:]
            if self.shuffle:
                random_state.shuffle(subsample)
            new_splits = np.array_split(subsample, self.n_splits)
            random_state.shuffle(new_splits)
            # Append stratum splits to overall splits
            splits = [split + list(new_split) for split, new_split in zip(splits, new_splits)]
        return splits


class WeightedStratifiedKFold(WeightedKFold):
    """Stratified K-Folds cross-validator for weighted data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k folds of roughly equal size and equal total weight.

    The default is to try sklearn.model_selection.StratifiedKFold a number of trials to find
    a weight-balanced k-way split. If it cannot find such a split, it will fall back
    onto a more rigorous weight stratification algorithm.

    Parameters
    ----------
    n_splits : int, default 3
        Number of folds. Must be at least 2.

    n_trials : int, default 10
        Number of times to try sklearn.model_selection.StratifiedKFold before falling back to another
        weight stratification algorithm.

    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance, or None, default None
            If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`. Used when ``shuffle`` == True.
    """

    def split(self, X, y, sample_weight=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array_like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array_like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=True)

    def get_n_splits(self, X, y, groups=None):
        """Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


class ModelSelector(metaclass=abc.ABCMeta):
    """
    This class enables a two-stage fitting process, where first a model is selected
    by calling `train` with `is_selecting=True`, and then the selected model is fit (presumably
    on a different data set) by calling train with `is_selecting=False`.


    """

    @abc.abstractmethod
    def train(self, is_selecting: bool, folds: Optional[List], *args, **kwargs):
        """
        Either selects a model or fits a model, depending on the value of `is_selecting`.
        If `is_selecting` is `False`, then `folds` should not be provided because they are only during selection.
        """
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def predict(self, *args, **kwargs):
        """
        Predicts using the selected model; should not be called until after `train` has been used
        both to select a model and to fit it.
        """
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def score(self, *args, **kwargs):
        """
        Gets the score of the selected model on the given data; should not be called until after `train` has been used
        both to select a model and to fit it.
        """
        raise NotImplementedError("Abstract method")


class SingleModelSelector(ModelSelector):
    """
    A model selection class that selects a single best model;
    this encompasses random search, grid search, ensembling, etc.
    """

    @property
    @abc.abstractmethod
    def best_model(self):
        raise NotImplementedError("Abstract method")

    @property
    @abc.abstractmethod
    def best_score(self):
        raise NotImplementedError("Abstract method")

    def predict(self, *args, **kwargs):
        return self.best_model.predict(*args, **kwargs)

    # only expose predict_proba if best_model has predict_proba
    # used because logic elsewhere uses hasattr predict proba to check if model is a classifier
    def __getattr__(self, name):
        if name == 'predict_proba':
            return getattr(self.best_model, name)
        else:
            self.__getattribute__(name)

    def score(self, *args, **kwargs):
        if hasattr(self.best_model, 'score'):
            return self.best_model.score(*args, **kwargs)
        else:
            return None


def _fit_with_groups(model, X, y, *, sub_model=None, groups, **kwargs):
    """
    Fits a model while correctly handling grouping if necessary.

    This enables us to perform an inner-loop cross-validation of a model
    which handles grouping correctly, which is not easy using typical sklearn models.

    For example, GridSearchCV and RandomSearchCV both support passing `groups` to fit,
    but other CV-related estimators (e.g. LassoCV) do not, which means that GroupKFold
    cannot be used as the cv instance, because the `groups` argument will never be passed through
    to GroupKFold's `split` method.

    The hacky workaround here is to explicitly set the `cv` attribute to the set of
    rows that GroupKFold would have generated rather than using GroupKFold as the cv instance.
    """
    if groups is not None:
        if sub_model is None:
            sub_model = model
        if hasattr(sub_model, 'cv'):
            old_cv = sub_model.cv
            # logic copied from check_cv
            cv = 5 if old_cv is None else old_cv
            if isinstance(cv, numbers.Integral):
                cv = GroupKFold(cv)
            # otherwise we will assume the user already set the cv attribute to something
            # compatible with splitting with a `groups` argument

            splits = list(cv.split(X, y, groups=groups))
            try:
                sub_model.cv = splits
                return model.fit(X, y, **kwargs)  # drop groups from arg list
            finally:
                sub_model.cv = old_cv

    # drop groups from arg list, which were already used at the outer level and may not be supported by the model
    return model.fit(X, y, **kwargs)


class FixedModelSelector(SingleModelSelector):
    """
    Model selection class that always selects the given sklearn-compatible model
    """

    def __init__(self, model, score_during_selection):
        self.model = clone(model, safe=False)
        self.score_during_selection = score_during_selection

    def train(self, is_selecting, folds: Optional[List], X, y, groups=None, **kwargs):
        if is_selecting:
            if self.score_during_selection:
                # the score needs to be compared to another model's
                # so we don't need to fit the model itself on all of the data, just get the out-of-sample score
                assert hasattr(self.model, 'score'), (f"Can't select between a fixed {type(self.model)} model "
                                                      "and others because it doesn't have a score method")
                scores = []
                for train, test in folds:
                    # use _fit_with_groups instead of just fit to handle nested grouping
                    _fit_with_groups(self.model, X[train], y[train],
                                     groups=None if groups is None else groups[train],
                                     **{key: val[train] for key, val in kwargs.items()})
                    scores.append(self.model.score(X[test], y[test]))
                self._score = np.mean(scores)
        else:
            # we need to train the model on the data
            _fit_with_groups(self.model, X, y, groups=groups, **kwargs)

        return self

    @property
    def best_model(self):
        return self.model

    @property
    def best_score(self):
        if hasattr(self, '_score'):
            return self._score
        else:
            raise ValueError("No score was computed during selection")


def _copy_to(m1, m2, attrs, insert_underscore=False):
    for attr in attrs:
        setattr(m2, attr, getattr(m1, attr + "_" if insert_underscore else attr))


def _convert_linear_model(model, new_cls):
    new_model = new_cls()
    # copy common parameters
    _copy_to(model, new_model, ["fit_intercept"])
    # copy common fitted variables
    _copy_to(model, new_model, ["coef_", "intercept_", "n_features_in_"])
    return new_model


def _to_logisticRegression(model: LogisticRegressionCV):
    lr = _convert_linear_model(model, LogisticRegression)
    _copy_to(model, lr, ["penalty", "dual", "intercept_scaling",
                         "class_weight",
                         "solver", "multi_class",
                         "verbose", "n_jobs",
                         "tol", "max_iter", "random_state", "n_iter_"])
    _copy_to(model, lr, ["classes_"])

    _copy_to(model, lr, ["C", "l1_ratio"], True)  # these are arrays in LogisticRegressionCV, need to convert them next

    # make sure all classes agree on best c/l1 combo
    assert np.isclose(lr.C, lr.C.flatten()[0]).all()
    assert np.equal(lr.l1_ratio, None).all() or np.isclose(lr.l1_ratio, lr.l1_ratio.flatten()[0]).all()
    lr.C = lr.C[0]
    lr.l1_ratio = lr.l1_ratio[0]
    avg_scores = np.average([v for k, v in model.scores_.items()], axis=1)  # average over folds
    best_scores = np.max(avg_scores, axis=tuple(range(1, avg_scores.ndim)))  # average score of best c/l1 combo
    assert np.isclose(best_scores, best_scores.flatten()[0]).all()  # make sure all folds agree on best c/l1 combo
    return lr, best_scores[0]


def _convert_linear_regression(model, new_cls, extra_attrs=["positive"]):
    new_model = _convert_linear_model(model, new_cls)
    _copy_to(model, new_model, ["alpha"], True)
    return new_model


def _to_elasticNet(model: ElasticNetCV, args, kwargs, is_lasso=False, cls=None, extra_attrs=[]):
    # We need an R^2 score to compare to other models; ElasticNetCV doesn't provide it,
    # but we can calculate it ourselves from the MSE plus the variance of the target y
    y = signature(model.fit).bind(*args, **kwargs).arguments["y"]
    cls = cls or (Lasso if is_lasso else ElasticNet)
    new_model = _convert_linear_regression(model, cls, extra_attrs + ['selection', 'warm_start', 'dual_gap_',
                                                                      'tol', 'max_iter', 'random_state', 'n_iter_',
                                                                      'copy_X'])
    if not is_lasso:
        # l1 ratio doesn't apply to Lasso, only ElasticNet
        _copy_to(model, new_model, ["l1_ratio"], True)
    # max R^2 corresponds to min MSE
    min_mse = np.min(np.mean(model.mse_path_, axis=-1))  # last dimension in mse_path is folds, so average over that
    r2 = 1 - min_mse / np.var(y)  # R^2 = 1 - MSE / Var(y)
    return new_model, r2


def _to_ridge(model, cls=Ridge, extra_attrs=["positive"]):
    ridge = _convert_linear_regression(model, cls, extra_attrs + ["_normalize", "solver"])
    best_score = model.best_score_
    return ridge, best_score


class SklearnCVSelector(SingleModelSelector):
    """
    Wraps one of sklearn's CV classes in the ModelSelector interface
    """

    def __init__(self, searcher):
        self.searcher = clone(searcher)

    @staticmethod
    def convertible_types():
        return {GridSearchCV, RandomizedSearchCV} | SklearnCVSelector._model_mapping().keys()

    @staticmethod
    def can_wrap(model):
        if isinstance(model, Pipeline):
            return SklearnCVSelector.can_wrap(model.steps[-1][1])
        return any(isinstance(model, model_type) for model_type in SklearnCVSelector.convertible_types())

    @staticmethod
    def _model_mapping():
        return {LogisticRegressionCV: lambda model, _args, _kwargs: _to_logisticRegression(model),
                ElasticNetCV: lambda model, args, kwargs: _to_elasticNet(model, args, kwargs),
                LassoCV: lambda model, args, kwargs: _to_elasticNet(model, args, kwargs, True, None, ["positive"]),
                RidgeCV: lambda model, _args, _kwargs: _to_ridge(model),
                RidgeClassifierCV: lambda model, _args, _kwargs: _to_ridge(model, RidgeClassifier,
                                                                           ["positive", "class_weight",
                                                                            "_label_binarizer"]),
                MultiTaskElasticNetCV: lambda model, args, kwargs: _to_elasticNet(model, args, kwargs,
                                                                                  False, MultiTaskElasticNet,
                                                                                  extra_attrs=[]),
                MultiTaskLassoCV: lambda model, args, kwargs: _to_elasticNet(model, args, kwargs,
                                                                             True, MultiTaskLasso, extra_attrs=[]),
                WeightedLassoCVWrapper: lambda model, args, kwargs: _to_elasticNet(model, args, kwargs,
                                                                                   True, WeightedLassoWrapper,
                                                                                   extra_attrs=[]),
                }

    @staticmethod
    def _convert_model(model, args, kwargs):
        if isinstance(model, Pipeline):
            name, inner_model = model.steps[-1]
            best_model, score = SklearnCVSelector._convert_model(inner_model, args, kwargs)
            return Pipeline(steps=[*model.steps[:-1], (name, best_model)]), score

        if isinstance(model, GridSearchCV) or isinstance(model, RandomizedSearchCV):
            return model.best_estimator_, model.best_score_

        for known_type in SklearnCVSelector._model_mapping().keys():
            if isinstance(model, known_type):
                converter = SklearnCVSelector._model_mapping()[known_type]
                return converter(model, args, kwargs)

    def train(self, is_selecting: bool, folds: Optional[List], *args, groups=None, **kwargs):
        if is_selecting:
            sub_model = self.searcher
            if isinstance(self.searcher, Pipeline):
                sub_model = self.searcher.steps[-1][1]

            init_params = inspect.signature(sub_model.__init__).parameters
            if 'cv' in init_params:
                default_cv = init_params['cv'].default
            else:
                # constructor takes cv as a positional or kwarg, just pull it out of a new instance
                default_cv = type(sub_model)().cv

            if sub_model.cv != default_cv:
                warnings.warn(f"Model {sub_model} has a non-default cv attribute, which will be ignored")
            sub_model.cv = folds

            self.searcher.fit(*args, **kwargs)

            self._best_model, self._best_score = self._convert_model(self.searcher, args, kwargs)

        else:
            self.best_model.fit(*args, **kwargs)
        return self

    @property
    def best_model(self):
        return self._best_model

    @property
    def best_score(self):
        return self._best_score


class ListSelector(SingleModelSelector):
    """
    Model selection class that selects the best model from a list of model selectors

    Parameters
    ----------
    models : list of ModelSelector
        The list of model selectors to choose from
    unwrap : bool, default True
        Whether to return the best model's best model, rather than just the outer best model selector
    """

    def __init__(self, models, unwrap=True):
        self.models = [clone(model, safe=False) for model in models]
        self.unwrap = unwrap

    def train(self, is_selecting, folds: Optional[List], *args, **kwargs):
        assert len(self.models) > 0, "ListSelector must have at least one model"
        if is_selecting:
            scores = []
            for model in self.models:
                model.train(is_selecting, folds, *args, **kwargs)
                scores.append(model.best_score)
            self._all_scores = scores
            self._best_score = np.max(scores)
            self._best_model = self.models[np.argmax(scores)]

        else:
            self._best_model.train(is_selecting, folds, *args, **kwargs)

    @property
    def best_model(self):
        """
        Gets the best model; note that if we were selecting over SingleModelSelectors and `unwrap` is `False`,
        we will return the SingleModelSelector instance, not its best model.
        """
        return self._best_model.best_model if self.unwrap else self._best_model

    @property
    def best_score(self):
        return self._best_score


def get_selector(input, is_discrete, *, random_state=None, cv=None, wrapper=GridSearchCV, needs_scoring=False):
    named_models = {
        'linear': (LogisticRegressionCV(random_state=random_state, cv=cv) if is_discrete
                   else WeightedLassoCVWrapper(random_state=random_state, cv=cv)),
        'poly': ([make_pipeline(PolynomialFeatures(d),
                                (LogisticRegressionCV(random_state=random_state, cv=cv) if is_discrete
                                 else WeightedLassoCVWrapper(random_state=random_state, cv=cv)))
                  for d in range(1, 4)]),
        'forest': (GridSearchCV(RandomForestClassifier(random_state=random_state) if is_discrete
                                else RandomForestRegressor(random_state=random_state),
                                param_grid={}, cv=cv)),
        'gbf': (GridSearchCV(GradientBoostingClassifier(random_state=random_state) if is_discrete
                             else GradientBoostingRegressor(random_state=random_state),
                             param_grid={}, cv=cv)),
        'nnet': (GridSearchCV(MLPClassifier(random_state=random_state) if is_discrete
                              else MLPRegressor(random_state=random_state),
                              param_grid={}, cv=cv)),
        'automl': ["poly", "forest", "gbf", "nnet"],
    }
    if isinstance(input, ModelSelector):  # we've already got a model selector, don't need to do anything
        return input
    elif isinstance(input, list):  # we've got a list; call get_selector on each element, then wrap in a ListSelector
        models = [get_selector(model, is_discrete,
                               random_state=random_state, cv=cv, wrapper=wrapper,
                               needs_scoring=True)  # we need to score to compare outputs to each other
                  for model in input]
        return ListSelector(models)
    elif isinstance(input, str):  # we've got a string; look it up
        if input in named_models:
            return get_selector(named_models[input], is_discrete,
                                random_state=random_state, cv=cv, wrapper=wrapper,
                                needs_scoring=needs_scoring)
        else:
            raise ValueError(f"Unknown model type: {input}, must be one of {named_models.keys()}")
    elif SklearnCVSelector.can_wrap(input):
        return SklearnCVSelector(input)
    else:  # assume this is an sklearn-compatible model
        return FixedModelSelector(input, needs_scoring)


class GridSearchCVList(BaseEstimator):
    """ An extension of GridSearchCV that allows for passing a list of estimators each with their own
    parameter grid and returns the best among all estimators in the list and hyperparameter in their
    corresponding grid. We are only changing the estimator parameter to estimator_list and the param_grid
    parameter to be a list of parameter grids. The rest of the parameters are the same as in
    :meth:`~sklearn.model_selection.GridSearchCV`. See the documentation of that class
    for explanation of the remaining parameters.

    Parameters
    ----------
    estimator_list : list of estimator object.
        Each estimator in th list is assumed to implement the scikit-learn estimator interface.
        Either estimator needs to provide a ``score`` function,
        or ``scoring`` must be passed.

    param_grid : list of dict or list of list of dictionaries
        For each estimator, the dictionary with parameters names (`str`) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.
    """

    def __init__(self, estimator_list, param_grid_list, scoring=None,
                 n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        self.estimator_list = estimator_list
        self.param_grid_list = param_grid_list
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        return

    def fit(self, X, y=None, **fit_params):
        self._gcv_list = [GridSearchCV(estimator, param_grid, scoring=self.scoring,
                                       n_jobs=self.n_jobs, refit=self.refit, cv=self.cv, verbose=self.verbose,
                                       pre_dispatch=self.pre_dispatch, error_score=self.error_score,
                                       return_train_score=self.return_train_score)
                          for estimator, param_grid in zip(self.estimator_list, self.param_grid_list)]
        self.best_ind_ = np.argmax([gcv.fit(X, y, **fit_params).best_score_ for gcv in self._gcv_list])
        self.best_estimator_ = self._gcv_list[self.best_ind_].best_estimator_
        self.best_score_ = self._gcv_list[self.best_ind_].best_score_
        self.best_params_ = self._gcv_list[self.best_ind_].best_params_
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


def _cross_val_predict(estimator, X, y=None, *, groups=None, cv=None,
                       n_jobs=None, verbose=0, fit_params=None,
                       pre_dispatch='2*n_jobs', method='predict', safe=True):
    """This is a fork from :meth:`~sklearn.model_selection.cross_val_predict` to allow for
    non-safe cloning of the models for each fold.

    Parameters
    ----------
    estimator : estimator object implementing 'fit' and 'predict'
        The object to use to fit the data.

    X : array_like of shape (n_samples, n_features)
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array_like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array_like of shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    cv : int, cross-validation generator or an iterable, default None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - CV splitter,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            ``cv`` default value if None changed from 3-fold to 5-fold.

    n_jobs : int, default None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default 0
        The verbosity level.

    fit_params : dict, defualt=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default '2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    method : str, default 'predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    safe : bool, default True
        Whether to clone with safe option.

    Returns
    -------
    predictions : ndarray
        This is the result of calling ``method``
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    splits = list(cv.split(X, y, groups))

    test_indices = np.concatenate([test for _, test in splits])
    if not _check_is_permutation(test_indices, _num_samples(X)):
        raise ValueError('cross_val_predict only works for partitions')

    # If classification methods produce multiple columns of output,
    # we need to manually encode classes to ensure consistent column ordering.
    encode = method in ['decision_function', 'predict_proba',
                        'predict_log_proba'] and y is not None
    if encode:
        y = np.asarray(y)
        if y.ndim == 1:
            le = LabelEncoder()
            y = le.fit_transform(y)
        elif y.ndim == 2:
            y_enc = np.zeros_like(y, dtype=int)
            for i_label in range(y.shape[1]):
                y_enc[:, i_label] = LabelEncoder().fit_transform(y[:, i_label])
            y = y_enc

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    parallel = Parallel(n_jobs=n_jobs, verbose=verbose,
                        pre_dispatch=pre_dispatch)

    from packaging.version import parse
    # verbose was removed from sklearn's non-public _fit_and_predict method in 1.4
    if parse(sklearn.__version__) < parse("1.4"):
        predictions = parallel(delayed(_fit_and_predict)(
            clone(estimator, safe=safe), X, y, train, test, verbose, fit_params, method)
            for train, test in splits)
    else:
        predictions = parallel(delayed(_fit_and_predict)(
            clone(estimator, safe=safe), X, y, train, test, fit_params, method)
            for train, test in splits)

    inv_test_indices = np.empty(len(test_indices), dtype=int)
    inv_test_indices[test_indices] = np.arange(len(test_indices))

    if sp.issparse(predictions[0]):
        predictions = sp.vstack(predictions, format=predictions[0].format)
    elif encode and isinstance(predictions[0], list):
        # `predictions` is a list of method outputs from each fold.
        # If each of those is also a list, then treat this as a
        # multioutput-multiclass task. We need to separately concatenate
        # the method outputs for each label into an `n_labels` long list.
        n_labels = y.shape[1]
        concat_pred = []
        for i_label in range(n_labels):
            label_preds = np.concatenate([p[i_label] for p in predictions])
            concat_pred.append(label_preds)
        predictions = concat_pred
    else:
        predictions = np.concatenate(predictions)

    if isinstance(predictions, list):
        return [p[inv_test_indices] for p in predictions]
    else:
        return predictions[inv_test_indices]
