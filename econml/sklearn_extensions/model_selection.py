# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Collection of scikit-learn extensions for model selection techniques."""

import numbers
import warnings
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.model_selection import KFold, StratifiedKFold, check_cv, GridSearchCV
# TODO: conisder working around relying on sklearn implementation details
from sklearn.model_selection._validation import (_check_is_permutation,
                                                 _fit_and_predict)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


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
    weights_sum = np.sum(sample_weight)
    max_deviations = []
    all_splits = []
    for i in range(self.n_trials + 1):
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
        kfold_model.shuffle = True
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
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    n_trials : int, default=10
        Number of times to try sklearn.model_selection.KFold before falling back to another
        weight stratification algorithm.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional, default=None
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
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array-like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=False)

    def _get_folds_from_splits(self, splits, sample_size):
        folds = []
        sample_indices = np.arange(sample_size)
        for it in range(self.n_splits):
            folds.append([np.setdiff1d(sample_indices, splits[it], assume_unique=True), splits[it]])
        return folds

    def _get_splits_from_weight_stratification(self, sample_weight):
        # Weight stratification algorithm
        # Sort weights for weight strata search
        sorted_inds = np.argsort(sample_weight)
        sorted_weights = sample_weight[sorted_inds]
        max_split_size = sorted_weights.shape[0] // self.n_splits
        max_divisible_length = max_split_size * self.n_splits
        sorted_inds_subset = np.reshape(sorted_inds[:max_divisible_length], (max_split_size, self.n_splits))
        shuffled_sorted_inds_subset = np.apply_along_axis(np.random.permutation, axis=1, arr=sorted_inds_subset)
        splits = [list(shuffled_sorted_inds_subset[:, i]) for i in range(self.n_splits)]
        if max_divisible_length != sorted_weights.shape[0]:
            # There are some leftover indices that have yet to be assigned
            subsample = sorted_inds[max_divisible_length:]
            if self.shuffle:
                np.random.shuffle(subsample)
            new_splits = np.array_split(subsample, self.n_splits)
            np.random.shuffle(new_splits)
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
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    n_trials : int, default=10
        Number of times to try sklearn.model_selection.StratifiedKFold before falling back to another
        weight stratification algorithm.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`. Used when ``shuffle`` == True.
    """

    def split(self, X, y, sample_weight=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array-like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=True)


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

    X : array-like of shape (n_samples, n_features)
        The data to fit. Can be, for example a list, or an array at least 2d.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

    cv : int, cross-validation generator or an iterable, default=None
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

    n_jobs : int, default=None
        The number of CPUs to use to do the computation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    fit_params : dict, defualt=None
        Parameters to pass to the fit method of the estimator.

    pre_dispatch : int or str, default='2*n_jobs'
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

    method : str, default='predict'
        Invokes the passed method name of the passed estimator. For
        method='predict_proba', the columns correspond to the classes
        in sorted order.

    safe : bool, default=True
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
    # TODO. The API of the private scikit-learn `_fit_and_predict` has changed
    # between 0.23.2 and 0.24. For this to work with <0.24, we need to add a
    # case analysis based on sklearn version.
    predictions = parallel(delayed(_fit_and_predict)(
        clone(estimator, safe=safe), X, y, train, test, verbose, fit_params, method)
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
