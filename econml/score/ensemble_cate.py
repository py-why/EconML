# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.utils.validation import check_array
from .._cate_estimator import BaseCateEstimator, LinearCateEstimator


class EnsembleCateEstimator:
    """ A CATE estimator that represents a weighted ensemble of many
    CATE estimators. Returns their weighted effect prediction.

    Parameters
    ----------
    cate_models : list of BaseCateEstimator objects
        A list of fitted cate estimator objects that will be used in the ensemble.
        The models are passed by reference, and not copied internally, because we
        need the fitted objects, so any change to the passed models will affect
        the internal predictions (e.g. if the input models are refitted).
    weights : np.ndarray of shape (len(cate_models),)
        The weight placed on each model. Weights must be non-positive. The
        ensemble will predict effects based on the weighted average predictions
        of the cate_models estiamtors, weighted by the corresponding weight in `weights`.
    """

    def __init__(self, *, cate_models, weights):
        self.cate_models = cate_models
        self.weights = weights

    def effect(self, X=None, *, T0=0, T1=1):
        return np.average([mdl.effect(X=X, T0=T0, T1=T1) for mdl in self.cate_models],
                          weights=self.weights, axis=0)
    effect.__doc__ = BaseCateEstimator.effect.__doc__

    def marginal_effect(self, T, X=None):
        return np.average([mdl.marginal_effect(T, X=X) for mdl in self.cate_models],
                          weights=self.weights, axis=0)
    marginal_effect.__doc__ = BaseCateEstimator.marginal_effect.__doc__

    def const_marginal_effect(self, X=None):
        if np.any([not hasattr(mdl, 'const_marginal_effect') for mdl in self.cate_models]):
            raise ValueError("One of the base CATE models in parameter `cate_models` does not support "
                             "the `const_marginal_effect` method.")
        return np.average([mdl.const_marginal_effect(X=X) for mdl in self.cate_models],
                          weights=self.weights, axis=0)
    const_marginal_effect.__doc__ = LinearCateEstimator.const_marginal_effect.__doc__

    @property
    def cate_models(self):
        return self._cate_models

    @cate_models.setter
    def cate_models(self, value):
        if (not isinstance(value, list)) or (not np.all([isinstance(model, BaseCateEstimator) for model in value])):
            raise ValueError('Parameter `cate_models` should be a list of `BaseCateEstimator` objects.')
        self._cate_models = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        weights = check_array(value, accept_sparse=False, ensure_2d=False, allow_nd=False, dtype='numeric',
                              force_all_finite=True)
        if np.any(weights < 0):
            raise ValueError("All weights in parameter `weights` must be non-negative.")
        self._weights = weights
