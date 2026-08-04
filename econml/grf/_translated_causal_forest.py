"""GRF-style causal forest wrapper.

This module provides a high-level wrapper around EconML's low-level
``econml.grf.classes.CausalForest`` so it can be used in a way that mirrors
``grf::causal_forest(X, Y, W, ...)`` from the R package.

The existing ``econml.grf.CausalForest`` class is intentionally left unchanged
because it is the residualized final-stage forest used by ``CausalForestDML``.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold
from sklearn.utils import check_array

from .classes import CausalForest as _LowLevelCausalForest, RegressionForest


@dataclass
class GRFPredictionResult:
    """Container mirroring the shape of ``grf::predict(... )`` output."""

    predictions: np.ndarray
    variance_estimates: np.ndarray | None = None
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None
    debiased_error: np.ndarray | None = None
    excess_error: np.ndarray | None = None


def _default_mtry(n_features):
    return min(int(np.ceil(np.sqrt(n_features) + 20)), n_features)


def _validate_nuisance(name, value, n):
    if value is None:
        return None
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size == 1:
        return np.repeat(arr.item(), n)
    if arr.size != n:
        raise ValueError(f"{name} has incorrect length.")
    return arr


def _same_matrix(X, X_ref):
    return X_ref is not None and X.shape == X_ref.shape and np.array_equal(X, X_ref)


def _diag_or_ravel(var):
    var = np.asarray(var)
    if var.ndim == 3:
        return np.diagonal(var, axis1=1, axis2=2).reshape(var.shape[0], -1)[:, 0]
    return var.ravel()


class GRFCausalForest(BaseEstimator):
    """Python translation of ``grf::causal_forest`` built on EconML GRF primitives.

    Parameters mirror the core R API where possible. Unsupported GRF features
    are accepted only at their default values and otherwise raise.
    """

    def __init__(self, *,
                 model_y=None,
                 model_t=None,
                 num_trees=2000,
                 sample_fraction=0.5,
                 mtry=None,
                 min_node_size=5,
                 honesty=True,
                 honesty_fraction=0.5,
                 honesty_prune_leaves=True,
                 alpha=0.05,
                 imbalance_penalty=0.0,
                 stabilize_splits=True,
                 ci_group_size=2,
                 compute_oob_predictions=True,
                 num_threads=None,
                 seed=None,
                 clusters=None,
                 equalize_cluster_weights=False,
                 tune_parameters="none",
                 tune_num_trees=200,
                 tune_num_reps=50,
                 tune_num_draws=1000,
                 max_depth=None,
                 min_impurity_decrease=0.0,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 inference=True,
                 fit_intercept=True,
                 verbose=0):
        self.model_y = model_y
        self.model_t = model_t
        self.num_trees = num_trees
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.stabilize_splits = stabilize_splits
        self.ci_group_size = ci_group_size
        self.compute_oob_predictions = compute_oob_predictions
        self.num_threads = num_threads
        self.seed = seed
        self.clusters = clusters
        self.equalize_cluster_weights = equalize_cluster_weights
        self.tune_parameters = tune_parameters
        self.tune_num_trees = tune_num_trees
        self.tune_num_reps = tune_num_reps
        self.tune_num_draws = tune_num_draws
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_var_fraction_leaf = min_var_fraction_leaf
        self.min_var_leaf_on_val = min_var_leaf_on_val
        self.inference = inference
        self.fit_intercept = fit_intercept
        self.verbose = verbose

    def _validate_supported_args(self):
        if self.clusters is not None:
            raise NotImplementedError("clusters are not yet supported in GRFCausalForest.")
        if self.equalize_cluster_weights:
            raise NotImplementedError("equalize_cluster_weights is not yet supported in GRFCausalForest.")
        if self.tune_parameters != "none":
            raise NotImplementedError("tune_parameters is not yet supported in GRFCausalForest.")
        if self.honesty_fraction != 0.5:
            raise NotImplementedError("honesty_fraction values other than 0.5 are not yet supported.")
        if not self.honesty_prune_leaves:
            raise NotImplementedError("honesty_prune_leaves=False is not yet supported.")
        if self.imbalance_penalty != 0.0:
            raise NotImplementedError("imbalance_penalty is not yet supported in GRFCausalForest.")
        if not self.stabilize_splits:
            raise NotImplementedError("stabilize_splits=False is not yet supported in GRFCausalForest.")

    def _make_default_nuisance_forest(self, n_features):
        return RegressionForest(
            n_estimators=max(50, self.num_trees // 4),
            max_depth=self.max_depth,
            min_samples_split=max(2 * self.min_node_size, 2),
            min_samples_leaf=self.min_node_size,
            max_features=self.mtry_ if self.mtry_ is not None else _default_mtry(n_features),
            min_impurity_decrease=self.min_impurity_decrease,
            max_samples=self.sample_fraction,
            honest=True,
            inference=False,
            n_jobs=-1 if self.num_threads is None else self.num_threads,
            random_state=self.seed,
            verbose=self.verbose,
        )

    def _crossfit_regression(self, model, X, y, sample_weight):
        fitted = clone(model)
        fitted.fit(X, y, sample_weight=sample_weight)

        preds = None
        if hasattr(fitted, "oob_predict"):
            try:
                preds = np.asarray(fitted.oob_predict(X)).ravel()
            except Exception:
                preds = None

        if preds is None or np.any(~np.isfinite(preds)):
            splitter = KFold(n_splits=2, shuffle=True, random_state=self.seed)
            preds = np.empty(X.shape[0], dtype=float)
            for train_idx, test_idx in splitter.split(X):
                fold_model = clone(model)
                fold_sw = None if sample_weight is None else sample_weight[train_idx]
                fold_model.fit(X[train_idx], y[train_idx], sample_weight=fold_sw)
                preds[test_idx] = np.asarray(fold_model.predict(X[test_idx])).ravel()
        return fitted, preds

    def _crossfit_final_predictions(self, X):
        splitter = KFold(n_splits=2, shuffle=True, random_state=self.seed)
        preds = np.empty(X.shape[0], dtype=float)
        for train_idx, test_idx in splitter.split(X):
            fold_forest = self._make_final_forest()
            fold_sw = None if self.sample_weight_ is None else self.sample_weight_[train_idx]
            fold_forest.fit(
                self.X_train_[train_idx],
                self.W_centered_[train_idx],
                self.Y_centered_[train_idx],
                sample_weight=fold_sw,
            )
            preds[test_idx] = np.asarray(fold_forest.predict(self.X_train_[test_idx])).ravel()
        return preds

    def _make_final_forest(self):
        return _LowLevelCausalForest(
            n_estimators=self.num_trees,
            criterion="mse",
            max_depth=self.max_depth,
            min_samples_split=max(2 * self.min_node_size, 2),
            min_samples_leaf=self.min_node_size,
            min_var_fraction_leaf=self.min_var_fraction_leaf,
            min_var_leaf_on_val=self.min_var_leaf_on_val,
            max_features=self.mtry_,
            min_impurity_decrease=self.min_impurity_decrease,
            max_samples=self.sample_fraction,
            min_balancedness_tol=0.5 - self.alpha,
            honest=self.honesty,
            inference=self.inference,
            fit_intercept=self.fit_intercept,
            subforest_size=self.ci_group_size,
            n_jobs=-1 if self.num_threads is None else self.num_threads,
            random_state=self.seed,
            verbose=self.verbose,
        )

    def fit(self, X, Y, W, *, sample_weight=None, Y_hat=None, W_hat=None):
        self._validate_supported_args()
        X = check_array(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float).ravel()
        W = np.asarray(W, dtype=float).ravel()
        if X.shape[0] != Y.shape[0] or X.shape[0] != W.shape[0]:
            raise ValueError("X, Y and W must have the same number of rows.")
        if not (0 <= self.alpha < 0.5):
            raise ValueError("alpha must be in [0, 0.5).")

        self.mtry_ = self.mtry if self.mtry is not None else _default_mtry(X.shape[1])
        self.X_train_ = np.array(X, copy=True)
        self.Y_train_ = np.array(Y, copy=True)
        self.W_train_ = np.array(W, copy=True)
        self.sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()

        Y_hat = _validate_nuisance("Y_hat", Y_hat, X.shape[0])
        W_hat = _validate_nuisance("W_hat", W_hat, X.shape[0])

        if Y_hat is None:
            base_model_y = self.model_y if self.model_y is not None else self._make_default_nuisance_forest(X.shape[1])
            self.model_y_nuisance_, Y_hat = self._crossfit_regression(base_model_y, X, Y, self.sample_weight_)
        else:
            self.model_y_nuisance_ = None

        if W_hat is None:
            base_model_t = self.model_t if self.model_t is not None else self._make_default_nuisance_forest(X.shape[1])
            self.model_t_nuisance_, W_hat = self._crossfit_regression(base_model_t, X, W, self.sample_weight_)
        else:
            self.model_t_nuisance_ = None

        self.Y_hat_ = Y_hat
        self.W_hat_ = W_hat
        self.Y_centered_ = Y - Y_hat
        self.W_centered_ = (W - W_hat).reshape(-1, 1)

        self.forest_ = self._make_final_forest()
        self.forest_.fit(X, self.W_centered_, self.Y_centered_, sample_weight=self.sample_weight_)

        if self.compute_oob_predictions:
            try:
                self.oob_predictions_ = np.asarray(self.forest_.oob_predict(self.X_train_)).ravel()
                if np.any(~np.isfinite(self.oob_predictions_)):
                    raise ValueError("non-finite oob predictions")
            except Exception:
                self.oob_predictions_ = self._crossfit_final_predictions(self.X_train_)
        else:
            self.oob_predictions_ = None

        return self

    def effect(self, X=None):
        return self.predict(X).predictions

    def predict(self, X=None, interval=False, alpha=0.05, estimate_variance=False):
        if X is None:
            if not self.compute_oob_predictions:
                raise ValueError("predict(X=None) requires compute_oob_predictions=True.")
            if interval or estimate_variance:
                raise NotImplementedError("OOB intervals and OOB variance estimates are not implemented.")
            return GRFPredictionResult(predictions=np.array(self.oob_predictions_, copy=True))

        X = check_array(np.asarray(X, dtype=float))
        if _same_matrix(X, self.X_train_) and self.compute_oob_predictions and not interval and not estimate_variance:
            return GRFPredictionResult(predictions=np.array(self.oob_predictions_, copy=True))

        if interval:
            pred, lb, ub = self.forest_.predict(X, interval=True, alpha=alpha)
            return GRFPredictionResult(
                predictions=np.asarray(pred).ravel(),
                lower_bound=np.asarray(lb).ravel(),
                upper_bound=np.asarray(ub).ravel(),
            )

        if estimate_variance:
            pred, var = self.forest_.predict_and_var(X)
            return GRFPredictionResult(
                predictions=np.asarray(pred).ravel(),
                variance_estimates=_diag_or_ravel(var),
            )

        pred = self.forest_.predict(X)
        return GRFPredictionResult(predictions=np.asarray(pred).ravel())


def causal_forest(X, Y, W, *,
                  Y_hat=None,
                  W_hat=None,
                  sample_weight=None,
                  **kwargs):
    """Fit and return a GRF-style causal forest.

    This is the closest Python analogue to ``grf::causal_forest(X, Y, W, ...)``
    in this repository.
    """
    forest = GRFCausalForest(**kwargs)
    return forest.fit(X, Y, W, sample_weight=sample_weight, Y_hat=Y_hat, W_hat=W_hat)
