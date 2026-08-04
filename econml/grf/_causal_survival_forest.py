# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""
Causal Survival Forest for heterogeneous treatment effect estimation with
right-censored outcomes.

Mirrors ``grf::causal_survival_forest`` from:
  grf-master/r-package/grf/R/causal_survival_forest.R

Strategy: estimate nuisance quantities (propensity, event survival, censoring
survival) in ``fit()``, then pass the pseudo-outcome and sample weights to a
dedicated internal causal-survival trainer. The public estimator coordinates
the survival-specific workflow while the internal trainer handles only the
final GRF stage.

References
----------
Cui, Y., Kosorok, M. R., Sverdrup, E., Wager, S., & Zhu, R. (2023).
  Estimating heterogeneous treatment effects with right-censored data via
  causal survival forests.
  Journal of the Royal Statistical Society Series B, 85(2), 179-211.
  https://doi.org/10.1093/jrsssb/qkac085
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import clone as sk_clone
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .classes import RegressionForest
from ._survival_forest import SurvivalForest, _cluster_weight_vector


_PROPENSITY_CLIP = 1e-3


class _CausalSurvivalTreeNode:
    def __init__(self, *, is_leaf, theta=0.0, intercept=0.0, node_id=0,
                 feature=-1, threshold=0.0, left=None, right=None):
        self.is_leaf = is_leaf
        self.theta = float(theta)
        self.intercept = float(intercept)
        self.node_id = int(node_id)
        self.feature = int(feature)
        self.threshold = float(threshold)
        self.left = left
        self.right = right


def _make_sksurv_y(time, event):
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event, time)],
        dtype=[("event", bool), ("time", float)],
    )


def _predict_treatment_model(model, X):
    """Predict propensities from a fitted treatment nuisance model."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
    return np.asarray(model.predict(X)).ravel()


def _extract_oob_treatment_predictions(model, X):
    """Extract OOB treatment predictions when the fitted model exposes them."""
    if hasattr(model, "oob_decision_function_"):
        proba = np.asarray(model.oob_decision_function_)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
    if hasattr(model, "oob_prediction_"):
        return np.asarray(model.oob_prediction_).ravel()
    if hasattr(model, "oob_predict"):
        try:
            return np.asarray(model.oob_predict(X)).ravel()
        except Exception:
            return None
    return None


def _build_csf_folds(X, T, event, *, random_state, n_splits=5):
    """Build held-out folds for nuisance estimation.

    We mirror grf's OOB nuisance philosophy with explicit out-of-fold predictions
    when the nuisance model does not natively support OOB prediction.
    """
    n = X.shape[0]
    n_splits = max(2, min(int(n_splits), n))
    dummy = np.zeros(n)

    joint = np.array([f"{int(t)}_{int(e)}" for t, e in zip(T, event)], dtype=object)
    _, joint_counts = np.unique(joint, return_counts=True)
    if joint_counts.size > 0 and np.min(joint_counts) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(dummy, joint))

    treat = np.asarray(T).astype(int)
    _, treat_counts = np.unique(treat, return_counts=True)
    if treat_counts.size > 0 and np.min(treat_counts) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(dummy, treat))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(dummy))


def _build_csf_cluster_folds(X, T, event, clusters, *, n_splits=5):
    n = X.shape[0]
    groups = np.asarray(clusters)
    unique_groups = np.unique(groups)
    n_splits = max(2, min(int(n_splits), unique_groups.shape[0]))
    dummy = np.zeros(n)
    splitter = GroupKFold(n_splits=n_splits)
    return list(splitter.split(dummy, groups=groups))


def _cluster_subsample_indices(cluster_info, frac, rng, *, equalize_cluster_weights):
    unique_clusters, inverse, counts = cluster_info
    n_clusters = len(unique_clusters)
    n_sub_clusters = max(1, int(np.ceil(frac * n_clusters)))
    n_sub_clusters = min(n_sub_clusters, n_clusters)
    sampled_cluster_pos = rng.choice(n_clusters, size=n_sub_clusters, replace=False)
    sampled_pos_set = set(sampled_cluster_pos.tolist())
    if equalize_cluster_weights:
        k = int(np.min(counts))
        pieces = []
        for pos in sampled_cluster_pos:
            members = np.flatnonzero(inverse == pos)
            pieces.append(rng.choice(members, size=k, replace=False))
        sample_idx = np.concatenate(pieces)
    else:
        sample_idx = np.concatenate([np.flatnonzero(inverse == pos) for pos in sampled_cluster_pos])
    unsampled = np.flatnonzero(np.array([pos not in sampled_pos_set for pos in inverse]))
    return sample_idx, unsampled


def _crossfit_treatment_predictions(model, X, T, *, sample_weight=None, random_state, n_splits=5, clusters=None):
    """Estimate W_hat with OOB predictions when available, else explicit OOF."""
    fitted_model = sk_clone(model)
    if sample_weight is None:
        fitted_model.fit(X, T)
    else:
        fitted_model.fit(X, T, sample_weight=sample_weight)

    W_hat = _extract_oob_treatment_predictions(fitted_model, X)

    if W_hat is None or np.any(~np.isfinite(W_hat)):
        if clusters is None:
            folds = _build_csf_folds(
                X, T, np.zeros_like(T), random_state=random_state, n_splits=n_splits
            )
        else:
            folds = _build_csf_cluster_folds(
                X, T, np.zeros_like(T), clusters, n_splits=n_splits
            )
        W_hat = np.empty(X.shape[0], dtype=float)
        for train_idx, test_idx in folds:
            fold_model = sk_clone(model)
            if sample_weight is None:
                fold_model.fit(X[train_idx], T[train_idx])
            else:
                fold_model.fit(X[train_idx], T[train_idx], sample_weight=sample_weight[train_idx])
            W_hat[test_idx] = _predict_treatment_model(fold_model, X[test_idx])

    return fitted_model, np.clip(W_hat, _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP)


def _fit_survival_nuisance_model(model, X, time, event, *, sample_weight=None):
    y_struct = _make_sksurv_y(time, event)
    try:
        if sample_weight is not None:
            return model.fit(X, time, event, sample_weight=sample_weight)
        return model.fit(X, time, event)
    except TypeError:
        if sample_weight is not None:
            try:
                return model.fit(X, y_struct, sample_weight=sample_weight)
            except TypeError:
                pass
        return model.fit(X, y_struct)


def _expected_survival_rmst(S_hat, Y_grid):
    """Compute E[min(T, max(Y_grid)) | X] = integral_0^max S(t|X) dt.

    Mirrors R's ``expected_survival(S.hat, Y.grid)``.

    Parameters
    ----------
    S_hat  : ndarray (n, ns)
    Y_grid : ndarray (ns,)

    Returns
    -------
    ndarray (n,)
    """
    grid_diff = np.diff(np.concatenate([[0.0], Y_grid, [Y_grid[-1]]]))
    # prepend column of 1s (S(0)=1)
    S_aug = np.hstack([np.ones((S_hat.shape[0], 1)), S_hat])
    return S_aug @ grid_diff


def _compute_psi(S_hat, C_hat, C_Y_hat, Y_hat, W_centered,
                 D, fY, Y_index, Y_grid, target, horizon):
    """Compute the GRF causal survival pseudo-outcome (numerator & denominator).

    Direct port of ``compute_psi`` from:
      grf-master/r-package/grf/R/causal_survival_forest.R  lines 522-577

    Parameters
    ----------
    S_hat     : ndarray (n, ns)   event survival S(t|X,W) on Y_grid
    C_hat     : ndarray (n, ns)   censoring survival C(t|X,W) on Y_grid
    C_Y_hat   : ndarray (n,)      C(Y_i|X_i,W_i) — censoring prob at obs time
    Y_hat     : ndarray (n,)      E[f(T)|X]
    W_centered: ndarray (n,)      W - W_hat
    D         : ndarray (n,)      event indicator (possibly modified for RMST)
    fY        : ndarray (n,)      f(T) = min(T, horizon) for RMST, 1{T>horizon} for SP
    Y_index   : ndarray (n,) int  index into Y_grid of each obs time
    Y_grid    : ndarray (ns,)
    target    : str               "RMST" or "survival.probability"
    horizon   : float

    Returns
    -------
    numerator   : ndarray (n,)
    denominator : ndarray (n,)
    """
    n, ns = S_hat.shape

    # --- Compute Q_hat(t, X) = E[f(T) | X, W, T > t] ---
    if target == "RMST":
        # Backward cumsum approach (mirrors R)
        Y_diff = np.diff(np.concatenate([[0.0], Y_grid]))       # (ns,)
        # dot_products[i, j] = S_hat[i, j] * Y_diff[j+1]  for j in 0..ns-2
        dot_products = S_hat[:, :-1] * Y_diff[1:]               # (n, ns-1)
        Q_hat = np.empty((n, ns))
        Q_hat[:, 0] = dot_products.sum(axis=1)
        for j in range(1, ns - 1):
            Q_hat[:, j] = Q_hat[:, j - 1] - dot_products[:, j - 1]
        # Divide by S_hat and add back t
        Q_hat = Q_hat / S_hat + Y_grid[np.newaxis, :]
        Q_hat[:, -1] = Y_grid[-1]
    else:
        # survival.probability: Q(t, X) = P(T > horizon | T > t) = S(horizon) / S(t)
        horizon_idx = np.searchsorted(Y_grid, horizon, side='right') - 1
        horizon_idx = np.clip(horizon_idx, 0, ns - 1)
        Q_hat = S_hat[:, horizon_idx:horizon_idx + 1] / S_hat   # (n, ns)
        Q_hat[:, horizon_idx:] = 1.0

    # Pick out Q(Y_i, X_i)
    Q_Y_hat = Q_hat[np.arange(n), Y_index]                      # (n,)

    # --- numerator_one ---
    numerator_one = (
        (D * (fY - Y_hat) + (1 - D) * (Q_Y_hat - Y_hat))
        * W_centered / C_Y_hat
    )

    # --- numerator_two: integral correction for censoring ---
    # dlambda_C_hat = -d log C_hat  (forward difference)
    # log_surv_C prepended with col of 0 (log 1 = 0)
    log_surv_C = np.hstack([np.zeros((n, 1)), -np.log(C_hat)])  # (n, ns+1)
    dlambda_C_hat = log_surv_C[:, 1:] - log_surv_C[:, :-1]     # (n, ns)

    integrand = dlambda_C_hat / C_hat * (Q_hat - Y_hat[:, np.newaxis])  # (n, ns)

    # For each sample, sum integrand up to Y_index[i] (inclusive)
    # Use cumsum then index — vectorised equivalent of R's loop
    cum_integrand = np.cumsum(integrand, axis=1)                # (n, ns)
    integral_at_Yi = cum_integrand[np.arange(n), Y_index]       # (n,)
    numerator_two = integral_at_Yi * W_centered

    numerator = numerator_one - numerator_two
    denominator = W_centered ** 2                                # simplifies in GRF paper

    return numerator, denominator


def _csf_candidate_thresholds(values):
    uniq = np.unique(values)
    if uniq.size <= 1:
        return np.array([], dtype=float)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    if mids.size <= 32:
        return mids
    return np.unique(np.quantile(mids, np.linspace(0.05, 0.95, 16)))


def _fit_default_regression_forest(X, T, *, sample_weight, n_estimators, max_features,
                                   max_samples, min_balancedness_tol, random_state,
                                   min_samples_leaf=5, clusters=None,
                                   equalize_cluster_weights=False):
    if clusters is None:
        model = RegressionForest(
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            min_balancedness_tol=min_balancedness_tol,
            min_samples_leaf=min_samples_leaf,
            inference=False,
            random_state=random_state,
        )
        if sample_weight is None:
            model.fit(X, T)
        else:
            model.fit(X, T, sample_weight=sample_weight)
        oob = np.asarray(model.oob_predict(X)).ravel()
    else:
        model = _ClusteredTreatmentForest(
            n_estimators=n_estimators,
            max_features=max_features,
            sample_fraction=max_samples,
            min_balancedness_tol=min_balancedness_tol,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            clusters=clusters,
            equalize_cluster_weights=equalize_cluster_weights,
        )
        if sample_weight is None:
            model.fit(X, T)
        else:
            model.fit(X, T, sample_weight=sample_weight)
        oob = np.asarray(model.oob_predict(X)).ravel()
    mse = np.mean((T - oob) ** 2)
    return model, mse


def _draw_treatment_tuning_candidates(X, *, defaults, tune_parameters, num_draws, random_state):
    p = X.shape[1]
    rng = np.random.RandomState(random_state)
    params = []
    for _ in range(max(1, int(num_draws))):
        draw = dict(defaults)
        if "sample.fraction" in tune_parameters:
            draw["sample.fraction"] = float(rng.uniform(0.05, 0.5))
        if "mtry" in tune_parameters:
            hi = max(1, p)
            lo = 1
            draw["mtry"] = int(rng.randint(lo, hi + 1))
        if "min.node.size" in tune_parameters:
            draw["min.node.size"] = int(rng.choice([3, 5, 7, 10, 15]))
        if "alpha" in tune_parameters:
            draw["alpha"] = float(rng.choice([0.01, 0.025, 0.05, 0.1]))
        params.append(draw)

    deduped = []
    seen = set()
    for draw in [defaults] + params:
        key = (
            round(float(draw["sample.fraction"]), 6),
            int(draw["mtry"]),
            int(draw["min.node.size"]),
            round(float(draw["alpha"]), 6),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(draw)
    return deduped


def _tune_default_treatment_forest(X, T, *, sample_weight, n_estimators, max_features,
                                   max_samples, min_balancedness_tol, random_state,
                                   tune_parameters, clusters=None,
                                   tune_num_draws=32, tune_num_reps=1, tune_num_trees=32,
                                   equalize_cluster_weights=False):
    default_mtry = max_features if isinstance(max_features, int) else max(1, int(np.ceil(np.sqrt(X.shape[1]))))
    defaults = {
        "sample.fraction": max_samples,
        "mtry": default_mtry,
        "min.node.size": 5,
        "alpha": 0.5 - min_balancedness_tol,
        "imbalance.penalty": 0.0,
    }
    if tune_parameters == "all":
        params_to_tune = ["sample.fraction", "mtry", "min.node.size", "alpha"]
    else:
        allowed = {"sample.fraction", "mtry", "min.node.size", "honesty.fraction",
                   "honesty.prune.leaves", "alpha", "imbalance.penalty"}
        params_to_tune = [p for p in np.atleast_1d(tune_parameters).tolist() if p in allowed]

    tunable_keys = [k for k in ["sample.fraction", "mtry", "min.node.size", "alpha"] if k in params_to_tune]
    if not tunable_keys:
        model, _ = _fit_default_regression_forest(
            X, T,
            sample_weight=sample_weight,
            n_estimators=n_estimators,
            max_features=max_features,
            max_samples=max_samples,
            min_balancedness_tol=min_balancedness_tol,
            random_state=random_state,
            min_samples_leaf=5,
            clusters=clusters,
            equalize_cluster_weights=equalize_cluster_weights,
        )
        return model, defaults, {
            "metric": "oob_mse",
            "num.draws": 1,
            "num.reps": 1,
            "num.trees": int(n_estimators),
            "results": [dict(params=dict(defaults), mean_oob_mse=np.nan, rep_oob_mse=[])],
        }

    best_model = None
    best_params = None
    best_score = np.inf
    results = []
    candidate_params = _draw_treatment_tuning_candidates(
        X,
        defaults=defaults,
        tune_parameters=tunable_keys,
        num_draws=tune_num_draws,
        random_state=random_state,
    )
    tune_trees = max(12, int(tune_num_trees))
    for draw_id, params in enumerate(candidate_params):
        rep_scores = []
        for rep in range(max(1, int(tune_num_reps))):
            _, mse = _fit_default_regression_forest(
                X, T,
                sample_weight=sample_weight,
                n_estimators=tune_trees,
                max_features=params["mtry"],
                max_samples=params["sample.fraction"],
                min_balancedness_tol=0.5 - params["alpha"],
                random_state=None if random_state is None else int(random_state + 1009 * draw_id + 37 * rep),
                min_samples_leaf=params["min.node.size"],
                clusters=clusters,
                equalize_cluster_weights=equalize_cluster_weights,
            )
            rep_scores.append(float(mse))
        mean_score = float(np.mean(rep_scores))
        results.append({
            "params": dict(params),
            "mean_oob_mse": mean_score,
            "rep_oob_mse": rep_scores,
        })
        if mean_score < best_score:
            best_score = mean_score
            best_params = dict(params)

    best_model, _ = _fit_default_regression_forest(
        X, T,
        sample_weight=sample_weight,
        n_estimators=n_estimators,
        max_features=best_params["mtry"],
        max_samples=best_params["sample.fraction"],
        min_balancedness_tol=0.5 - best_params["alpha"],
        random_state=random_state,
        min_samples_leaf=best_params["min.node.size"],
        clusters=clusters,
        equalize_cluster_weights=equalize_cluster_weights,
    )
    tuning_output = {
        "metric": "oob_mse",
        "num.draws": len(candidate_params),
        "num.reps": max(1, int(tune_num_reps)),
        "num.trees": tune_trees,
        "results": results,
        "best.params": dict(best_params),
        "best.mean_oob_mse": float(best_score),
    }
    return best_model, best_params, tuning_output


class _ClusteredTreatmentForest(BaseEstimator):
    """Cluster-aware treatment nuisance using cluster-level subsampling and OOB aggregation."""

    def __init__(self, *, n_estimators, max_features, sample_fraction,
                 min_balancedness_tol, min_samples_leaf, random_state,
                 clusters, equalize_cluster_weights):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.sample_fraction = sample_fraction
        self.min_balancedness_tol = min_balancedness_tol
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.clusters = clusters
        self.equalize_cluster_weights = equalize_cluster_weights

    def fit(self, X, T, sample_weight=None):
        X = check_array(np.asarray(X, dtype=float))
        T = np.asarray(T, dtype=float).ravel()
        n = X.shape[0]
        self.X_train_ = np.array(X, copy=True)
        self.T_train_ = np.array(T, copy=True)
        _, self.cluster_info_ = _cluster_weight_vector(self.clusters, self.equalize_cluster_weights, n)
        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.subsample_indices_ = []
        oob_sum = np.zeros(n, dtype=float)
        oob_count = np.zeros(n, dtype=int)

        for _ in range(self.n_estimators):
            sample_idx, unsampled = _cluster_subsample_indices(
                self.cluster_info_, self.sample_fraction, rng,
                equalize_cluster_weights=self.equalize_cluster_weights
            )
            est = RegressionForest(
                n_estimators=1,
                max_features=self.max_features,
                max_samples=1.0,
                min_balancedness_tol=self.min_balancedness_tol,
                min_samples_leaf=self.min_samples_leaf,
                honest=True,
                inference=False,
                random_state=rng.randint(np.iinfo(np.int32).max),
            )
            if sample_weight is None:
                est.fit(X[sample_idx], T[sample_idx])
            else:
                est.fit(X[sample_idx], T[sample_idx], sample_weight=np.asarray(sample_weight)[sample_idx])
            self.estimators_.append(est)
            self.subsample_indices_.append(sample_idx)
            if unsampled.size > 0:
                pred = np.asarray(est.predict(X[unsampled])).ravel()
                oob_sum[unsampled] += pred
                oob_count[unsampled] += 1

        self.oob_prediction_ = np.empty(n, dtype=float)
        mask = oob_count > 0
        if np.any(mask):
            self.oob_prediction_[mask] = oob_sum[mask] / oob_count[mask]
        if np.any(~mask):
            self.oob_prediction_[~mask] = self.predict(X[~mask])
        return self

    def predict(self, X):
        X = check_array(np.asarray(X, dtype=float))
        preds = np.vstack([np.asarray(est.predict(X)).ravel() for est in self.estimators_])
        return np.mean(preds, axis=0)

    def oob_predict(self, Xtrain):
        Xtrain = check_array(np.asarray(Xtrain, dtype=float))
        if Xtrain.shape != self.X_train_.shape or not np.array_equal(Xtrain, self.X_train_):
            raise ValueError("oob_predict is only defined on the training sample.")
        return np.array(self.oob_prediction_, copy=True)


def _weighted_leaf_params(w, y, sample_weight, fit_intercept):
    if y.size == 0:
        return np.nan, np.nan
    sw = np.ones_like(w, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    total = np.sum(sw)
    if total <= 1e-10:
        return np.nan, np.nan
    theta = np.sum(sw * y) / total
    return theta, 0.0


def _weighted_leaf_loss(y, sample_weight):
    if y.size == 0:
        return 0.0
    sw = np.ones_like(y, dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    total = np.sum(sw)
    if total <= 1e-10:
        return 0.0
    mean = np.sum(sw * y) / total
    resid = y - mean
    return np.sum(sw * resid * resid)


class _CausalSurvivalTree:
    def __init__(self, *, mtry, min_samples_split, min_samples_leaf, max_depth,
                 min_balancedness_tol, fit_intercept, honest, honesty_fraction,
                 honesty_prune_leaves, alpha, imbalance_penalty, stabilize_splits,
                 random_state):
        self.mtry = mtry
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.min_balancedness_tol = min_balancedness_tol
        self.fit_intercept = fit_intercept
        self.honest = honest
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.stabilize_splits = stabilize_splits
        self.random_state = random_state
        self._next_node_id = 0
        self.feature_importances_ = None

    def fit(self, X, W, y, sample_weight, *, treatment_raw=None, censor=None):
        n = X.shape[0]
        self.treatment_raw_ = None if treatment_raw is None else np.asarray(treatment_raw, dtype=int)
        self.censor_ = None if censor is None else np.asarray(censor, dtype=int)
        if self.honest:
            perm = self.random_state.permutation(n)
            split_n = int(np.floor(n * self.honesty_fraction))
            est_n = n - split_n
            if split_n <= 0 or est_n <= 0:
                raise ValueError("honesty_fraction leaves no split or estimation sample.")
            split_idx = perm[:split_n]
            est_idx = perm[split_n:]
        else:
            split_idx = np.arange(n)
            est_idx = np.arange(n)
        self.feature_importances_ = np.zeros(X.shape[1], dtype=float)
        self.root_ = self._grow(X, W, y, sample_weight, split_idx, est_idx, depth=0)
        return self

    def _new_node_id(self):
        node_id = self._next_node_id
        self._next_node_id += 1
        return node_id

    def _best_split(self, X, W, y, sample_weight, split_idx):
        n = split_idx.size
        min_frac = max(0.0, 0.5 - self.min_balancedness_tol)
        features = self.random_state.choice(X.shape[1], size=min(self.mtry, X.shape[1]), replace=False)
        best = (-np.inf, None, None)
        parent_loss = _weighted_leaf_loss(y[split_idx], sample_weight[split_idx])
        for feature in features:
            thresholds = _csf_candidate_thresholds(X[split_idx, feature])
            for threshold in thresholds:
                left = split_idx[X[split_idx, feature] <= threshold]
                right = split_idx[X[split_idx, feature] > threshold]
                if left.size < self.min_samples_leaf or right.size < self.min_samples_leaf:
                    continue
                if min(left.size, right.size) / n < min_frac:
                    continue
                imbalance = abs(left.size - right.size) / max(n, 1)
                if self.stabilize_splits:
                    for labels in (self.treatment_raw_, self.censor_):
                        if labels is None:
                            continue
                        parent_labels = labels[split_idx]
                        for cls in np.unique(parent_labels):
                            parent_count = np.sum(parent_labels == cls)
                            if parent_count == 0:
                                continue
                            left_count = np.sum(labels[left] == cls)
                            right_count = np.sum(labels[right] == cls)
                            min_required = max(1, int(np.floor(self.alpha * n)))
                            if min(left_count, right_count) < min_required:
                                imbalance = np.inf
                                break
                            imbalance += abs(left_count - right_count) / parent_count
                        if not np.isfinite(imbalance):
                            break
                    if not np.isfinite(imbalance):
                        continue
                left_loss = _weighted_leaf_loss(y[left], sample_weight[left])
                right_loss = _weighted_leaf_loss(y[right], sample_weight[right])
                score = parent_loss - left_loss - right_loss - self.imbalance_penalty * imbalance
                if score > best[0]:
                    best = (score, feature, threshold)
        return best

    def _grow(self, X, W, y, sample_weight, split_idx, est_idx, depth):
        theta, intercept = _weighted_leaf_params(
            W[est_idx], y[est_idx], sample_weight[est_idx], self.fit_intercept
        )
        node_id = self._new_node_id()
        if split_idx.size < self.min_samples_split or split_idx.size < 2 * self.min_samples_leaf:
            return _CausalSurvivalTreeNode(is_leaf=True, theta=theta, intercept=intercept, node_id=node_id)
        if self.max_depth is not None and depth >= self.max_depth:
            return _CausalSurvivalTreeNode(is_leaf=True, theta=theta, intercept=intercept, node_id=node_id)

        score, feature, threshold = self._best_split(X, W, y, sample_weight, split_idx)
        if feature is None or not np.isfinite(score) or score <= 0:
            return _CausalSurvivalTreeNode(is_leaf=True, theta=theta, intercept=intercept, node_id=node_id)

        split_left = split_idx[X[split_idx, feature] <= threshold]
        split_right = split_idx[X[split_idx, feature] > threshold]
        est_left = est_idx[X[est_idx, feature] <= threshold]
        est_right = est_idx[X[est_idx, feature] > threshold]
        if self.honest and self.honesty_prune_leaves and (est_left.size == 0 or est_right.size == 0):
            return _CausalSurvivalTreeNode(is_leaf=True, theta=theta, intercept=intercept, node_id=node_id)

        self.feature_importances_[feature] += score / ((1 + depth) ** 2)
        left = self._grow(X, W, y, sample_weight, split_left, est_left, depth + 1)
        right = self._grow(X, W, y, sample_weight, split_right, est_right, depth + 1)
        return _CausalSurvivalTreeNode(
            is_leaf=False, theta=theta, intercept=intercept, node_id=node_id,
            feature=feature, threshold=float(threshold), left=left, right=right
        )

    def _leaf_for_row(self, x):
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X):
        out = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            node = self._leaf_for_row(x)
            out[i] = node.theta
        return out

    def apply(self, X):
        out = np.empty(X.shape[0], dtype=int)
        for i, x in enumerate(X):
            out[i] = self._leaf_for_row(x).node_id
        return out


class _CausalSurvivalForestTrainer(BaseEstimator):
    """Dedicated final-stage trainer for the causal survival forest."""

    def __init__(self, *, n_estimators=100, max_depth=None, min_samples_split=10,
                 min_samples_leaf=5, max_features="auto", max_samples=0.45,
                 min_balancedness_tol=0.45, honest=True, honesty_fraction=0.5,
                 honesty_prune_leaves=True, inference=True,
                 fit_intercept=True, subforest_size=4, n_jobs=-1, random_state=None, verbose=0,
                 criterion="mse", min_weight_fraction_leaf=0.0,
                 min_var_fraction_leaf=None, min_var_leaf_on_val=False,
                 min_impurity_decrease=0.0, alpha=0.05, imbalance_penalty=0.0,
                 stabilize_splits=True, clusters=None, equalize_cluster_weights=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.inference = inference
        self.fit_intercept = fit_intercept
        self.subforest_size = subforest_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.criterion = criterion
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_var_fraction_leaf = min_var_fraction_leaf
        self.min_var_leaf_on_val = min_var_leaf_on_val
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.stabilize_splits = stabilize_splits
        self.clusters = clusters
        self.equalize_cluster_weights = equalize_cluster_weights

    def _subsample_indices(self, n, rng):
        if self.cluster_info_ is None:
            n_sub = max(1, int(np.ceil(self.max_samples * n)))
            n_sub = min(n_sub, n)
            sample_idx = rng.choice(n, size=n_sub, replace=False)
            unsampled = np.setdiff1d(np.arange(n), sample_idx, assume_unique=True)
            return sample_idx, unsampled

        unique_clusters, inverse, counts = self.cluster_info_
        n_clusters = len(unique_clusters)
        n_sub_clusters = max(1, int(np.ceil(self.max_samples * n_clusters)))
        n_sub_clusters = min(n_sub_clusters, n_clusters)
        sampled_cluster_pos = rng.choice(n_clusters, size=n_sub_clusters, replace=False)
        sampled_pos_set = set(sampled_cluster_pos.tolist())
        if self.equalize_cluster_weights:
            k = int(np.min(counts))
            pieces = []
            for pos in sampled_cluster_pos:
                members = np.flatnonzero(inverse == pos)
                pieces.append(rng.choice(members, size=k, replace=False))
            sample_idx = np.concatenate(pieces)
        else:
            sample_idx = np.concatenate([np.flatnonzero(inverse == pos) for pos in sampled_cluster_pos])
        unsampled = np.flatnonzero(np.array([pos not in sampled_pos_set for pos in inverse]))
        return sample_idx, unsampled

    def fit(self, X, W, y, *, sample_weight=None, treatment_raw=None, censor=None):
        X = check_array(np.asarray(X, dtype=float))
        W = np.asarray(W, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        sw = np.ones(X.shape[0], dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
        n = X.shape[0]
        self.X_train_ = np.array(X, copy=True)
        self.W_train_ = np.array(W, copy=True)
        self.y_train_ = np.array(y, copy=True)
        self.sample_weight_ = np.array(sw, copy=True)
        self.treatment_raw_ = None if treatment_raw is None else np.asarray(treatment_raw, dtype=int).reshape(-1)
        self.censor_ = None if censor is None else np.asarray(censor, dtype=int).reshape(-1)
        self.n_features_in_ = X.shape[1]
        self.mtry_ = min(self.n_features_in_, int(np.ceil(np.sqrt(self.n_features_in_) + 20))) if self.max_features == "auto" else int(self.max_features)
        self.n_outputs_ = 1
        self.n_relevant_outputs_ = 1
        _, self.cluster_info_ = _cluster_weight_vector(self.clusters, self.equalize_cluster_weights, n)
        if self.inference and self.n_estimators % self.subforest_size != 0:
            raise ValueError("n_estimators must be divisible by subforest_size when inference=True.")

        rng = np.random.RandomState(self.random_state)
        self.estimators_ = []
        self.subsample_indices_ = []
        oob_sum = np.zeros(n, dtype=float)
        oob_count = np.zeros(n, dtype=int)

        for _ in range(self.n_estimators):
            sample_idx, unsampled = self._subsample_indices(n, rng)
            tree = _CausalSurvivalTree(
                mtry=self.mtry_,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                min_balancedness_tol=self.min_balancedness_tol,
                fit_intercept=self.fit_intercept,
                honest=self.honest,
                honesty_fraction=self.honesty_fraction,
                honesty_prune_leaves=self.honesty_prune_leaves,
                alpha=self.alpha,
                imbalance_penalty=self.imbalance_penalty,
                stabilize_splits=self.stabilize_splits,
                random_state=np.random.RandomState(rng.randint(np.iinfo(np.int32).max)),
            )
            tr_raw = None if self.treatment_raw_ is None else self.treatment_raw_[sample_idx]
            censor = None if self.censor_ is None else self.censor_[sample_idx]
            tree.fit(X[sample_idx], W[sample_idx], y[sample_idx], sw[sample_idx],
                     treatment_raw=tr_raw, censor=censor)
            self.estimators_.append(tree)
            self.subsample_indices_.append(sample_idx)
            if unsampled.size > 0:
                preds = tree.predict(X[unsampled])
                valid = np.isfinite(preds)
                if np.any(valid):
                    oob_sum[unsampled[valid]] += preds[valid]
                    oob_count[unsampled[valid]] += 1

        self.oob_predictions_ = np.empty(n, dtype=float)
        mask = oob_count > 0
        self.oob_predictions_[mask] = oob_sum[mask] / oob_count[mask]
        if np.any(~mask):
            self.oob_predictions_[~mask] = self.predict(X[~mask]).reshape(-1)

        self.feature_importances_ = np.mean(
            [tree.feature_importances_ for tree in self.estimators_], axis=0
        )
        total = np.sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = self.feature_importances_ / total

        if self.inference:
            self.subforest_groups_ = [
                self.estimators_[i:i + self.subforest_size]
                for i in range(0, len(self.estimators_), self.subforest_size)
            ]
        else:
            self.subforest_groups_ = None
        return self

    def predict(self, X, interval=False, alpha=0.05):
        X = check_array(np.asarray(X, dtype=float))
        tree_preds = np.vstack([tree.predict(X) for tree in self.estimators_])
        finite = np.isfinite(tree_preds)
        counts = np.maximum(np.sum(finite, axis=0), 1)
        preds = (np.sum(np.where(finite, tree_preds, 0.0), axis=0) / counts).reshape(-1, 1)
        if not interval:
            return preds
        _, var = self.predict_and_var(X)
        std = np.sqrt(np.clip(var[:, 0, 0], 0.0, None))
        z = 1.959963984540054
        return preds, (preds[:, 0] - z * std).reshape(-1, 1), (preds[:, 0] + z * std).reshape(-1, 1)

    def predict_and_var(self, X):
        X = check_array(np.asarray(X, dtype=float))
        point = self.predict(X).reshape(-1, 1)
        if not self.inference or self.subforest_groups_ is None or len(self.subforest_groups_) <= 1:
            var = np.zeros((X.shape[0], 1, 1), dtype=float)
            return point, var
        group_preds = []
        for group in self.subforest_groups_:
            gp = np.vstack([tree.predict(X) for tree in group])
            finite = np.isfinite(gp)
            counts = np.maximum(np.sum(finite, axis=0), 1)
            group_preds.append(np.sum(np.where(finite, gp, 0.0), axis=0) / counts)
        group_preds = np.stack(group_preds, axis=0)
        var_vec = np.var(group_preds, axis=0, ddof=1) / group_preds.shape[0]
        return point, var_vec.reshape(-1, 1, 1)

    def predict_var(self, X):
        return self.predict_and_var(X)[1]

    def predict_projection_and_var(self, X, projector):
        proj = np.asarray(projector, dtype=float).reshape(-1, 1)
        point, var = self.predict_and_var(X)
        return point * proj, var * (proj[:, :, None] ** 2)

    def predict_projection(self, X, projector):
        return self.predict_projection_and_var(X, projector)[0]

    def predict_projection_var(self, X, projector):
        return self.predict_projection_and_var(X, projector)[1]

    def oob_predict(self, Xtrain):
        Xtrain = check_array(np.asarray(Xtrain, dtype=float))
        if Xtrain.shape != self.X_train_.shape or not np.array_equal(Xtrain, self.X_train_):
            raise ValueError("oob_predict is only defined on the training sample.")
        return self.oob_predictions_.reshape(-1, 1)

    def apply(self, X):
        X = check_array(np.asarray(X, dtype=float))
        return np.column_stack([tree.apply(X) for tree in self.estimators_])

    def decision_path(self, X):
        raise NotImplementedError("decision_path is not yet implemented for the dedicated causal survival trainer.")

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        return np.array(self.feature_importances_, copy=True)


# ---------------------------------------------------------------------------
# CausalSurvivalForest
# ---------------------------------------------------------------------------

class CausalSurvivalForest(BaseEstimator):
    """Causal Survival Forest for RMST-based heterogeneous treatment effects.

    Takes raw survival inputs ``(X, T, time, event)`` and internally:

    1. Estimates propensity ``W_hat = E[T|X]`` with out-of-bag predictions when
       available, otherwise explicit held-out predictions.
    2. Fits event survival ``S_hat(t|X,W)`` and censoring survival
       ``C_hat(t|X,W)`` using held-out predictions that mirror grf's nuisance
       OOB workflow.
    3. Computes the doubly-robust GRF pseudo-outcome ``psi`` via ``_compute_psi``
       (direct port of ``grf::compute_psi``).
    4. Calls a dedicated internal survival trainer on
       ``(X, W_centered, psi_numerator / psi_denominator)`` with
       ``sample_weight=psi_denominator``.

    This is statistically equivalent to Cui et al. (2023): the GRF causal survival
    forest embeds the same doubly-robust moment equation in its splitting criterion.

    Parameters
    ----------
    horizon : float or None
        RMST time horizon τ. Either ``horizon`` or ``tau`` must be provided.
    tau : float or None, default None
        Alias for ``horizon`` to match the survival learner APIs and notebooks.
    target : {"RMST", "survival.probability"}, default "RMST"
        Estimand.  "RMST" targets E[min(T,τ)|a=1,X] − E[min(T,τ)|a=0,X].
        "survival.probability" targets P(T>τ|a=1,X) − P(T>τ|a=0,X).
    model_t : sklearn regressor or None, default None
        Propensity model E[T|X]. Defaults to
        ``RegressionForest(n_estimators=max(52, n_estimators // 4))``,
        mirroring ``grf::causal_survival_forest`` using a regression forest
        for ``W.hat``.
    model_cens : grf-style survival estimator or None, default None
        Censoring survival model. Defaults to
        ``econml.grf.SurvivalForest(num_trees=max(50, min(n_estimators // 4, 500)),``
        ``min_node_size=15)``.
    model_event : grf-style survival estimator or None, default None
        Event survival model. Defaults to
        ``econml.grf.SurvivalForest(num_trees=max(50, min(n_estimators // 4, 500)),``
        ``min_node_size=15)``.
    failure_times : array-like or None, default None
        Optional event-time grid used for nuisance survival estimation.
    sample_fraction : float or None, default None
        GRF-style sample fraction. If supplied, this also controls the nuisance
        survival forests.
    mtry : int or None, default None
        Number of candidate split features. If supplied, this is used for both
        nuisance survival forests and the final forest.
    alpha : float, default 0.05
        GRF-style imbalance control parameter.
    imbalance_penalty : float, default 0.0
        Penalty applied to imbalanced split candidates in the nuisance survival
        forests and in the final causal-survival forest.
    stabilize_splits : bool, default True
        Whether treatment and censoring status should contribute to the split
        imbalance check in the final causal-survival forest.
    fast_logrank : bool, default False
        Whether nuisance survival forests should use the approximate fast
        log-rank candidate-threshold search.
    tune_parameters : {"none", "all"} or iterable, default "none"
        Parameters to tune for the default treatment nuisance forest. Mirrors
        GRF's current behavior where tuning applies to ``W.hat`` estimation.
    tune_num_draws : int, default 32
        Number of random hyperparameter draws scored during default treatment
        nuisance tuning.
    tune_num_reps : int, default 1
        Number of repeated fits used to average the OOB tuning score for each
        random draw.
    tune_num_trees : int, default 32
        Number of trees used inside each tuning fit for the default treatment
        nuisance.
    compute_oob_predictions : bool, default True
        Whether nuisance forests and the final forest should cache OOB
        predictions on the training sample.
    clusters : array-like or None, default None
        Optional cluster labels. The nuisance survival forests support
        cluster-aware resampling directly; the treatment nuisance and final
        forest consume the induced effective sample weights.
    equalize_cluster_weights : bool, default False
        Whether to reweight clusters to contribute equally. Requires
        ``clusters``.
    n_estimators : int, default 100
        Number of trees in the causal forest.
    criterion : {"mse", "het"}, default "mse"
        Splitting criterion — passed to the dedicated final GRF trainer.
    max_depth : int or None, default None
    min_samples_split : int, default 10
    min_samples_leaf : int, default 5
    min_weight_fraction_leaf : float, default 0.0
    min_var_fraction_leaf : float or None, default None
    max_features : "auto" or int, default "auto"
    min_impurity_decrease : float, default 0.0
    max_samples : float, default 0.45
    min_balancedness_tol : float, default 0.45
    honest : bool, default True
    honesty_fraction : float, default 0.5
        Fraction of each honest subsample used for split selection in the final
        forest and the default nuisance survival forests.
    honesty_prune_leaves : bool, default True
        Whether honest trees should prune leaves without estimation-sample
        support. If False, such leaves are skipped at prediction time.
    inference : bool, default True
        Whether to enable variance / confidence interval estimation.
    fit_intercept : bool, default True
    subforest_size : int, default 4
    n_jobs : int, default -1
    nuisance_cv : int, default 5
        Backward-compatible fallback used only when the treatment nuisance
        model does not expose native OOB predictions.
    random_state : int or None, default None
    verbose : int, default 0

    Examples
    --------
    >>> import numpy as np
    >>> from econml.grf import CausalSurvivalForest
    >>> n = 300
    >>> X = np.random.normal(size=(n, 5))
    >>> T = np.random.binomial(1, 0.5, size=n)
    >>> time = np.abs(np.random.normal(3, 1, size=n))
    >>> event = np.random.binomial(1, 0.7, size=n)
    >>> est = CausalSurvivalForest(horizon=4.0)
    >>> est.fit(X, T, time, event)         # doctest: +SKIP
    >>> est.predict(X[:5])                 # doctest: +SKIP
    """

    def __init__(self, horizon=None, *,
                 tau=None,
                 target="RMST",
                 model_t=None,
                 model_cens=None,
                 model_event=None,
                 failure_times=None,
                 sample_fraction=None,
                 mtry=None,
                 alpha=0.05,
                 imbalance_penalty=0.0,
                 stabilize_splits=True,
                 fast_logrank=False,
                 tune_parameters="none",
                 tune_num_draws=32,
                 tune_num_reps=1,
                 tune_num_trees=32,
                 compute_oob_predictions=True,
                 clusters=None,
                 equalize_cluster_weights=False,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 honesty_fraction=0.5,
                 honesty_prune_leaves=True,
                 inference=True,
                 fit_intercept=True,
                 subforest_size=4,
                 n_jobs=-1,
                 nuisance_cv=5,
                 random_state=None,
                 verbose=0):
        if horizon is None and tau is None:
            raise TypeError("CausalSurvivalForest requires either `horizon` or `tau`.")
        if horizon is not None and tau is not None and not np.isclose(horizon, tau):
            raise ValueError("`horizon` and `tau` must match when both are provided.")

        resolved_horizon = float(tau if horizon is None else horizon)
        self.horizon = resolved_horizon
        self.tau = tau
        self.target = target
        self.model_t = model_t
        self.model_cens = model_cens
        self.model_event = model_event
        self.failure_times = failure_times
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.stabilize_splits = stabilize_splits
        self.fast_logrank = fast_logrank
        self.tune_parameters = tune_parameters
        self.tune_num_draws = tune_num_draws
        self.tune_num_reps = tune_num_reps
        self.tune_num_trees = tune_num_trees
        self.compute_oob_predictions = compute_oob_predictions
        self.clusters = clusters
        self.equalize_cluster_weights = equalize_cluster_weights
        self.nuisance_cv = nuisance_cv
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_var_fraction_leaf = min_var_fraction_leaf
        self.min_var_leaf_on_val = min_var_leaf_on_val
        self.max_features = mtry if mtry is not None else max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = sample_fraction if sample_fraction is not None else max_samples
        self.min_balancedness_tol = 0.5 - alpha
        self.honest = honest
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.inference = inference
        self.fit_intercept = fit_intercept
        self.subforest_size = subforest_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _validate_grf_args(self):
        if not (0 <= self.alpha < 0.5):
            raise ValueError("alpha must be in [0, 0.5).")
        if self.honest and not (0 < self.honesty_fraction < 1):
            raise ValueError("honesty_fraction must be in (0, 1) when honest=True.")
        if self.equalize_cluster_weights and self.clusters is None:
            raise ValueError("equalize_cluster_weights=True requires clusters.")
        if self.tune_parameters == "all":
            return
        if self.tune_parameters != "none":
            allowed = {"sample.fraction", "mtry", "min.node.size", "honesty.fraction",
                       "honesty.prune.leaves", "alpha", "imbalance.penalty"}
            requested = set(np.atleast_1d(self.tune_parameters).tolist())
            if not requested.issubset(allowed):
                raise ValueError("Unsupported tune_parameters entries: {}".format(sorted(requested - allowed)))

    def _make_final_trainer(self):
        return _CausalSurvivalForestTrainer(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            min_var_fraction_leaf=self.min_var_fraction_leaf,
            min_var_leaf_on_val=self.min_var_leaf_on_val,
            max_features=self.max_features,
            min_impurity_decrease=self.min_impurity_decrease,
            max_samples=self.max_samples,
            min_balancedness_tol=self.min_balancedness_tol,
            honest=self.honest,
            honesty_fraction=self.honesty_fraction,
            honesty_prune_leaves=self.honesty_prune_leaves,
            inference=self.inference,
            fit_intercept=self.fit_intercept,
            subforest_size=self.subforest_size,
            n_jobs=self.n_jobs,
            alpha=self.alpha,
            imbalance_penalty=self.imbalance_penalty,
            stabilize_splits=self.stabilize_splits,
            clusters=self.clusters,
            equalize_cluster_weights=self.equalize_cluster_weights,
            random_state=self.random_state,
            verbose=self.verbose,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, T, time, event, *, sample_weight=None):
        """Fit the causal survival forest.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Pre-treatment covariates.
        T : array_like, shape (n,)
            Binary treatment indicator (0 = control, 1 = treated).
        time : array_like, shape (n,)
            Observed (possibly censored) survival time.  Must be non-negative.
        event : array_like, shape (n,)
            Event indicator: 1 = event occurred, 0 = censored.
        sample_weight : array_like, shape (n,) or None

        Returns
        -------
        self
        """
        X = check_array(np.asarray(X, dtype=float))
        T = np.asarray(T, dtype=float).ravel()
        time = np.asarray(time, dtype=float).ravel()
        event = np.asarray(event, dtype=float).ravel()
        n = X.shape[0]

        if np.any(time < 0):
            raise ValueError("Event times must be non-negative.")
        if not np.all(np.isin(event, [0, 1])):
            raise ValueError("Event indicator must be 0 or 1.")
        if event.sum() == 0:
            raise ValueError("All observations are censored.")
        if not np.all(np.isin(T, [0, 1])):
            raise NotImplementedError("CausalSurvivalForest currently supports binary treatment only.")

        self._validate_grf_args()
        if self.equalize_cluster_weights and sample_weight is not None:
            raise ValueError("sample_weight must be None when equalize_cluster_weights=True.")
        _, self.cluster_info_ = _cluster_weight_vector(
            self.clusters, self.equalize_cluster_weights, n
        )
        base_sample_weight = None if sample_weight is None else np.asarray(sample_weight, dtype=float).ravel()
        if base_sample_weight is not None and base_sample_weight.shape[0] != n:
            raise ValueError("sample_weight must have the same length as X.")
        effective_sample_weight = base_sample_weight

        horizon = self.horizon
        target = self.target
        if target not in ("RMST", "survival.probability"):
            raise ValueError("target must be 'RMST' or 'survival.probability'.")

        # Default nuisance models
        model_t = self.model_t
        if model_t is None:
            if self.tune_parameters == "none":
                if self.clusters is None:
                    model_t = RegressionForest(
                        n_estimators=max(52, self.n_estimators // 4),
                        max_features=self.max_features,
                        max_samples=self.max_samples,
                        min_balancedness_tol=self.min_balancedness_tol,
                        random_state=self.random_state,
                    )
                else:
                    model_t = _ClusteredTreatmentForest(
                        n_estimators=max(52, self.n_estimators // 4),
                        max_features=self.max_features,
                        sample_fraction=self.max_samples,
                        min_balancedness_tol=self.min_balancedness_tol,
                        min_samples_leaf=5,
                        random_state=self.random_state,
                        clusters=self.clusters,
                        equalize_cluster_weights=self.equalize_cluster_weights,
                    )
                self.tuned_model_t_params_ = None
                self.tuning_output_ = None
            else:
                model_t, tuned, tuning_output = _tune_default_treatment_forest(
                    X, T,
                    sample_weight=effective_sample_weight,
                    n_estimators=max(52, self.n_estimators // 4),
                    max_features=self.max_features,
                    max_samples=self.max_samples,
                    min_balancedness_tol=self.min_balancedness_tol,
                    random_state=self.random_state,
                    tune_parameters=self.tune_parameters,
                    clusters=self.clusters,
                    tune_num_draws=self.tune_num_draws,
                    tune_num_reps=self.tune_num_reps,
                    tune_num_trees=self.tune_num_trees,
                    equalize_cluster_weights=self.equalize_cluster_weights,
                )
                self.tuned_model_t_params_ = tuned
                self.tuning_output_ = tuning_output

        model_cens = self.model_cens
        if model_cens is None:
            model_cens = SurvivalForest(
                failure_times=self.failure_times,
                num_trees=max(50, min(self.n_estimators // 4, 500)),
                sample_fraction=self.max_samples,
                mtry=self.max_features if isinstance(self.max_features, int) else None,
                min_node_size=15,
                honesty=self.honest,
                honesty_fraction=self.honesty_fraction,
                honesty_prune_leaves=self.honesty_prune_leaves,
                alpha=self.alpha,
                clusters=self.clusters,
                equalize_cluster_weights=self.equalize_cluster_weights,
                imbalance_penalty=self.imbalance_penalty,
                compute_oob_predictions=self.compute_oob_predictions,
                fast_logrank=self.fast_logrank,
                num_threads=self.n_jobs,
                seed=self.random_state,
                max_depth=self.max_depth,
                verbose=self.verbose,
            )

        model_event = self.model_event
        if model_event is None:
            model_event = SurvivalForest(
                failure_times=self.failure_times,
                num_trees=max(50, min(self.n_estimators // 4, 500)),
                sample_fraction=self.max_samples,
                mtry=self.max_features if isinstance(self.max_features, int) else None,
                min_node_size=15,
                honesty=self.honest,
                honesty_fraction=self.honesty_fraction,
                honesty_prune_leaves=self.honesty_prune_leaves,
                alpha=self.alpha,
                clusters=self.clusters,
                equalize_cluster_weights=self.equalize_cluster_weights,
                imbalance_penalty=self.imbalance_penalty,
                compute_oob_predictions=self.compute_oob_predictions,
                fast_logrank=self.fast_logrank,
                num_threads=self.n_jobs,
                seed=self.random_state,
                max_depth=self.max_depth,
                verbose=self.verbose,
            )

        # ---- Modify time/event for RMST (mirror R lines 201-208) ----
        time_mod = time.copy()
        event_mod = event.copy()
        fY = time_mod.copy()
        if target == "RMST":
            event_mod[time_mod >= horizon] = 1
            time_mod[time_mod >= horizon] = horizon
            fY = time_mod
        else:
            fY = (time_mod > horizon).astype(float)

        # ---- Time grid (unique observed times) ----
        Y_grid = np.sort(np.unique(time_mod)) if self.failure_times is None else np.sort(np.asarray(self.failure_times, dtype=float).ravel())
        if len(Y_grid) <= 2:
            raise ValueError("Number of distinct event times must be > 2.")
        if horizon < Y_grid[0]:
            raise ValueError("`horizon` cannot be before the first event time.")

        # ---- Propensity W_hat = E[T|X] ----
        model_t_, W_hat = _crossfit_treatment_predictions(
            model_t, X, T, sample_weight=effective_sample_weight,
            random_state=self.random_state, n_splits=self.nuisance_cv,
            clusters=self.clusters
        )
        W_centered = T - W_hat

        # ---- Survival nuisance models ----
        XT = np.column_stack([X, T])

        try:
            model_event_ = sk_clone(model_event)
            model_event_.set_params(failure_times=Y_grid)
        except Exception:
            model_event_ = model_event
        try:
            model_cens_ = sk_clone(model_cens)
            model_cens_.set_params(failure_times=Y_grid)
        except Exception:
            model_cens_ = model_cens

        try:
            _fit_survival_nuisance_model(
                model_event_, XT, time_mod, event_mod, sample_weight=effective_sample_weight
            )
            S_hat = model_event_.oob_predict(XT) if self.compute_oob_predictions else model_event_.predict(XT).predictions
            S1_hat = model_event_.predict(np.column_stack([X, np.ones(n)]), failure_times=Y_grid).predictions
            S0_hat = model_event_.predict(np.column_stack([X, np.zeros(n)]), failure_times=Y_grid).predictions
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, NotImplementedError) as exc:
            warnings.warn(
                "CausalSurvivalForest event nuisance fit failed; falling back to a constant survival curve. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            S_hat = np.ones((n, len(Y_grid)), dtype=float)
            S1_hat = np.ones((n, len(Y_grid)), dtype=float)
            S0_hat = np.ones((n, len(Y_grid)), dtype=float)
            model_event_ = None

        try:
            _fit_survival_nuisance_model(
                model_cens_, XT, time_mod, 1 - event_mod, sample_weight=effective_sample_weight
            )
            C_hat = model_cens_.oob_predict(XT) if self.compute_oob_predictions else model_cens_.predict(XT).predictions
        except (FloatingPointError, ValueError, np.linalg.LinAlgError, NotImplementedError) as exc:
            warnings.warn(
                "CausalSurvivalForest censoring nuisance fit failed; falling back to a constant survival curve. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            C_hat = np.ones((n, len(Y_grid)), dtype=float)
            model_cens_ = None

        if target == "RMST":
            Y_hat = (W_hat * _expected_survival_rmst(S1_hat, Y_grid) +
                     (1 - W_hat) * _expected_survival_rmst(S0_hat, Y_grid))
        else:
            horizon_idx = np.searchsorted(Y_grid, horizon, side='right') - 1
            horizon_idx = np.clip(horizon_idx, 0, len(Y_grid) - 1)
            Y_hat = W_hat * S1_hat[:, horizon_idx] + (1 - W_hat) * S0_hat[:, horizon_idx]

        # ---- Censoring probability at each unit's observed time ----
        if target == "survival.probability":
            time_mod2 = time_mod.copy()
            event_mod2 = event_mod.copy()
            time_mod2[time_mod2 > horizon] = horizon
            event_mod2[time_mod > horizon] = 1
        else:
            time_mod2, event_mod2 = time_mod, event_mod

        Y_index = np.searchsorted(Y_grid, time_mod2, side='right') - 1
        Y_index = np.clip(Y_index, 0, len(Y_grid) - 1)
        C_Y_hat = C_hat[np.arange(n), Y_index]

        if target == "RMST" and np.any(C_Y_hat <= 0.05):
            warnings.warn(
                "Estimated censoring probabilities go as low as "
                f"{C_Y_hat.min():.5f}. Causal survival forest may be unreliable.",
                stacklevel=2,
            )

        # ---- Compute GRF pseudo-outcome psi ----
        D_use = event_mod2 if target == "survival.probability" else event_mod
        psi_num, psi_denom = _compute_psi(
            S_hat, C_hat, C_Y_hat, Y_hat, W_centered,
            D_use, fY, Y_index, Y_grid, target, horizon
        )

        # Normalise: pass psi_num/psi_denom as y; psi_denom as sample_weight
        # (mirrors GRF's numerator/denominator weighting)
        denom_safe = np.where(psi_denom > 0, psi_denom, 1e-10)
        pseudo_y = psi_num / denom_safe

        sw = psi_denom if effective_sample_weight is None else psi_denom * effective_sample_weight

        # Store nuisance estimates for diagnostics
        self.W_hat_ = W_hat
        self.Y_hat_ = Y_hat
        self.S_hat_ = S_hat
        self.C_hat_ = C_hat
        self.S1_hat_ = S1_hat
        self.S0_hat_ = S0_hat
        self.model_t_nuisance_ = model_t_
        self.model_event_nuisance_ = model_event_
        self.model_cens_nuisance_ = model_cens_
        self.psi_numerator_ = psi_num
        self.psi_denominator_ = psi_denom
        self.effective_sample_weight_ = effective_sample_weight
        self.X_train_ = np.array(X, copy=True)

        # ---- Fit dedicated causal-survival trainer on pseudo-outcome ----
        trainer = self._make_final_trainer()
        trainer.fit(X, W_centered.reshape(-1, 1), pseudo_y.reshape(-1, 1),
                    sample_weight=sw, treatment_raw=T, censor=D_use)
        self.final_trainer_ = trainer
        self.estimators_ = trainer.estimators_
        return self

    def predict(self, X, interval=False, alpha=0.05):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.predict(X, interval=interval, alpha=alpha)

    def oob_predict(self, Xtrain):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.oob_predict(Xtrain)

    def predict_and_var(self, X):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.predict_and_var(X)

    def predict_var(self, X):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.predict_var(X)

    def predict_projection_and_var(self, X, projector):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.predict_projection_and_var(X, projector)

    def predict_projection(self, X, projector):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.predict_projection(X, projector)

    def predict_projection_var(self, X, projector):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.predict_projection_var(X, projector)

    def apply(self, X):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.apply(X)

    def decision_path(self, X):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.decision_path(X)

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.feature_importances(
            max_depth=max_depth, depth_decay_exponent=depth_decay_exponent
        )

    @property
    def feature_importances_(self):
        check_is_fitted(self, "final_trainer_")
        return self.final_trainer_.feature_importances_

    def effect(self, X=None):
        """Return treatment-effect predictions.

        When called on the training covariates, return out-of-bag predictions
        to mirror grf's training-sample prediction behavior.
        """
        if X is None:
            X = self.X_train_
        X_arr = check_array(np.asarray(X, dtype=float))
        if hasattr(self, 'X_train_') and X_arr.shape == self.X_train_.shape and np.array_equal(X_arr, self.X_train_):
            try:
                return np.asarray(self.oob_predict(self.X_train_)).ravel()
            except Exception:
                pass
        return np.asarray(self.predict(X_arr)).ravel()
