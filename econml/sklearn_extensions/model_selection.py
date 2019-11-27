# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Collection of scikit-learn extensions for model selection techniques."""

import numbers
import numpy as np
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.multiclass import type_of_target


def _split_weighted_sample(self, X, y, sample_weight, is_stratified=False):
    if is_stratified:
        kfold_model = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                      random_state=self.random_state)
    else:
        kfold_model = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                            random_state=self.random_state)
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
        kfold_model.random_state = None

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
