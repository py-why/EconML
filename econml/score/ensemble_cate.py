# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License

import numpy as np
from sklearn.utils.validation import check_array
from .._cate_estimator import BaseCateEstimator, LinearCateEstimator


class EnsembleCateEstimator(BaseCateEstimator):
    """
    A CATE estimator that represents a weighted ensemble of many CATE estimators.

    Predicts treatment effects as the weighted average of predictions from base estimators.

    Parameters
    ----------
    cate_models : list of BaseCateEstimator
        List of *fitted* CATE estimators. Models are held by reference — changes to them affect ensemble predictions.
        All models must implement the methods being called (e.g., `effect`, `const_marginal_effect`).

    weights : array-like of shape (n_models,)
        Non-negative weights for each model. Must sum to > 0. If not normalized, will be normalized internally.
        Weights determine contribution of each model to the ensemble prediction.

    normalize_weights : bool, default=True
        If True, weights are normalized to sum to 1. If False, raw weights are used.

    Attributes
    ----------
    n_models_ : int
        Number of base models in the ensemble.

    d_t_ : int or None
        Dimensionality of treatment (inferred from first model supporting `marginal_effect` or `const_marginal_effect`).

    d_y_ : int or None
        Dimensionality of outcome (inferred similarly).

    Notes
    -----
    - This class inherits from `BaseCateEstimator` to ensure compatibility with EconML APIs.
    - Lazy inference of `d_t_`, `d_y_` avoids forcing all models to expose these unless needed.
    - Supports heterogeneous models: some may support `effect`, others only `const_marginal_effect`.
    """

    def __init__(self, *, cate_models, weights, normalize_weights=True):
        self.cate_models = cate_models
        self.weights = weights
        self.normalize_weights = normalize_weights

    @property
    def cate_models(self):
        """List of base CATE estimators."""
        return self._cate_models

    @cate_models.setter
    def cate_models(self, value):
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError("`cate_models` must be a non-empty list.")
        if not all(isinstance(model, BaseCateEstimator) for model in value):
            raise ValueError("All elements in `cate_models` must be instances of `BaseCateEstimator`.")
        self._cate_models = value
        # Invalidate cached metadata
        self._d_t = None
        self._d_y = None

    @property
    def weights(self):
        """Weights assigned to each base model."""
        return self._weights

    @weights.setter
    def weights(self, value):
        weights = check_array(value, accept_sparse=False, ensure_2d=False, dtype='numeric',
                              force_all_finite=True, copy=True).ravel()
        if weights.shape[0] != len(self.cate_models):
            raise ValueError(f"Length of `weights` ({weights.shape[0]}) must match "
                             f"number of models ({len(self.cate_models)}).")
        if np.any(weights < 0):
            raise ValueError("All weights must be non-negative.")
        if np.sum(weights) <= 0:
            raise ValueError("Sum of weights must be positive.")

        if getattr(self, 'normalize_weights', True):
            weights = weights / np.sum(weights)

        self._weights = weights

    @property
    def d_t(self):
        """Treatment dimensionality (lazy inference)."""
        if self._d_t is None:
            self._infer_shapes()
        return self._d_t

    @property
    def d_y(self):
        """Outcome dimensionality (lazy inference)."""
        if self._d_y is None:
            self._infer_shapes()
        return self._d_y

    def _infer_shapes(self):
        """Infer d_t and d_y from first model that supports const_marginal_effect or marginal_effect."""
        for mdl in self.cate_models:
            if hasattr(mdl, 'const_marginal_effect'):
                try:
                    # Try dummy call to infer shapes
                    dummy_X = np.zeros((1, 1))  # minimal shape
                    eff = mdl.const_marginal_effect(X=dummy_X)
                    if eff.ndim == 3:
                        _, d_y, d_t = eff.shape
                        self._d_t = d_t
                        self._d_y = d_y
                        return
                    elif eff.ndim == 2:
                        # Assume (n, d_t) and d_y=1
                        self._d_t = eff.shape[1]
                        self._d_y = 1
                        return
                except Exception:
                    continue
            elif hasattr(mdl, 'marginal_effect'):
                try:
                    dummy_T = np.zeros((1, 1))
                    dummy_X = np.zeros((1, 1))
                    meff = mdl.marginal_effect(T=dummy_T, X=dummy_X)
                    if meff.ndim == 3:
                        _, d_y, d_t = meff.shape
                        self._d_t = d_t
                        self._d_y = d_y
                        return
                except Exception:
                    continue
        # Fallback: unknown
        self._d_t = None
        self._d_y = None

    def effect(self, X=None, *, T0=0, T1=1):
        """
        Calculate the average treatment effect.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Features for each sample.
        T0 : array-like or scalar, default=0
            Baseline treatment.
        T1 : array-like or scalar, default=1
            Target treatment.

        Returns
        -------
        τ : array-like of shape (n_samples,) or (n_samples, d_y)
            Estimated treatment effects.
        """
        if not self.cate_models:
            raise ValueError("No models in ensemble.")

        predictions = []
        for mdl in self.cate_models:
            if not hasattr(mdl, 'effect'):
                raise AttributeError(f"Model {type(mdl).__name__} does not implement 'effect' method.")
            pred = mdl.effect(X=X, T0=T0, T1=T1)
            predictions.append(np.asarray(pred))

        # Stack and validate shapes
        stacked = np.stack(predictions, axis=0)  # (n_models, n_samples, ...)
        return np.average(stacked, weights=self.weights, axis=0)

    effect.__doc__ = BaseCateEstimator.effect.__doc__

    def marginal_effect(self, T, X=None):
        """
        Calculate the heterogeneous marginal effect.

        Parameters
        ----------
        T : array-like of shape (n_samples, d_t)
            Treatment values at which to calculate the effect.
        X : array-like of shape (n_samples, n_features), optional
            Features for each sample.

        Returns
        -------
        τ : array-like of shape (n_samples, d_y, d_t)
            Estimated marginal effects.
        """
        if not self.cate_models:
            raise ValueError("No models in ensemble.")

        predictions = []
        for mdl in self.cate_models:
            if not hasattr(mdl, 'marginal_effect'):
                raise AttributeError(f"Model {type(mdl).__name__} does not implement 'marginal_effect' method.")
            pred = mdl.marginal_effect(T=T, X=X)
            pred = np.asarray(pred)
            # Ensure 3D: (n, d_y, d_t)
            if pred.ndim == 2:
                pred = pred[:, None, :]  # assume d_y=1
            elif pred.ndim != 3:
                raise ValueError(f"Unexpected shape {pred.shape} from {type(mdl).__name__}.marginal_effect")
            predictions.append(pred)

        stacked = np.stack(predictions, axis=0)  # (n_models, n, d_y, d_t)
        return np.average(stacked, weights=self.weights, axis=0)

    marginal_effect.__doc__ = BaseCateEstimator.marginal_effect.__doc__

    def const_marginal_effect(self, X=None):
        """
        Calculate the constant marginal CATE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features), optional
            Features for each sample.

        Returns
        -------
        τ : array-like of shape (n_samples, d_y, d_t)
            Estimated constant marginal effects.
        """
        if not self.cate_models:
            raise ValueError("No models in ensemble.")

        predictions = []
        for mdl in self.cate_models:
            if not hasattr(mdl, 'const_marginal_effect'):
                raise AttributeError(
                    f"Model {type(mdl).__name__} does not implement 'const_marginal_effect' method."
                )
            pred = mdl.const_marginal_effect(X=X)
            pred = np.asarray(pred)
            if pred.ndim == 2:
                pred = pred[:, None, :]  # assume d_y=1
            elif pred.ndim != 3:
                raise ValueError(f"Unexpected shape {pred.shape} from {type(mdl).__name__}.const_marginal_effect")
            predictions.append(pred)

        stacked = np.stack(predictions, axis=0)  # (n_models, n, d_y, d_t)
        return np.average(stacked, weights=self.weights, axis=0)

    const_marginal_effect.__doc__ = LinearCateEstimator.const_marginal_effect.__doc__

    def __repr__(self):
        return (f"{self.__class__.__name__}(n_models={len(self.cate_models)}, "
                f"normalize_weights={getattr(self, 'normalize_weights', True)})")

    def __str__(self):
        model_types = [type(mdl).__name__ for mdl in self.cate_models]
        return (f"Ensemble of {len(self.cate_models)} models: {model_types}\n"
                f"Weights: {self.weights}")

