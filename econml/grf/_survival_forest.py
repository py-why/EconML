"""GRF-style survival forest wrapper.

This module provides a light-weight Python analogue of
``grf::survival_forest`` for the censored-outcome workstream. It is not yet a
line-by-line port of GRF's C++ survival trainer, but it mirrors the API shape,
implements honest log-rank splitting in Python, and provides native out-of-bag
survival predictions on the training sample so downstream estimators can follow
the same nuisance-estimation pattern as ``grf::causal_survival_forest``.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import _check_sample_weight


@dataclass
class SurvivalForestPredictionResult:
    predictions: np.ndarray
    failure_times: np.ndarray


@dataclass
class _SurvivalTreeNode:
    is_leaf: bool
    feature: int = -1
    threshold: float = 0.0
    left: object = None
    right: object = None
    survival: np.ndarray = None


def _default_mtry(n_features):
    return min(int(np.ceil(np.sqrt(n_features) + 20)), n_features)


def _cluster_weight_vector(clusters, equalize_cluster_weights, n):
    if clusters is None:
        return np.ones(n, dtype=float), None

    clusters = np.asarray(clusters)
    if clusters.shape[0] != n:
        raise ValueError("clusters must have the same length as X.")

    unique, inverse, counts = np.unique(clusters, return_inverse=True, return_counts=True)
    if equalize_cluster_weights:
        weights = 1.0 / counts[inverse]
    else:
        weights = np.ones(n, dtype=float)
    return weights, (unique, inverse, counts)


def _event_grid_counts(time, event, grid):
    idx = np.searchsorted(grid, time, side="right") - 1
    idx = np.clip(idx, 0, len(grid) - 1)
    failures = np.bincount(idx[event == 1], minlength=len(grid)).astype(float)
    at_risk = np.array([(time >= t).sum() for t in grid], dtype=float)
    return failures, at_risk


def _leaf_survival_curve(time, event, grid, prediction_type):
    if time.size == 0:
        return None
    failures, at_risk = _event_grid_counts(time, event, grid)
    safe_risk = np.maximum(at_risk, 1.0)
    hazard = failures / safe_risk
    if prediction_type == "Kaplan-Meier":
        surv = np.cumprod(1.0 - hazard)
    else:
        surv = np.exp(-np.cumsum(hazard))
    return np.clip(surv, 1e-3, 1.0)


def _logrank_statistic(time, event, left_mask):
    if left_mask.sum() == 0 or left_mask.sum() == len(left_mask):
        return -np.inf
    event_times = np.unique(time[event == 1])
    if event_times.size == 0:
        return -np.inf
    U = 0.0
    V = 0.0
    left_mask = left_mask.astype(bool)
    for t in event_times:
        at_risk = time >= t
        y = np.sum(at_risk)
        if y <= 1:
            continue
        y1 = np.sum(at_risk & left_mask)
        d = np.sum((time == t) & (event == 1))
        d1 = np.sum((time == t) & (event == 1) & left_mask)
        expected = y1 * d / y
        U += d1 - expected
        V += (y1 * (y - y1) * d * (y - d)) / (y * y * (y - 1))
    if V <= 0:
        return -np.inf
    return (U * U) / V


def _candidate_thresholds(values, *, fast_logrank):
    uniq = np.unique(values)
    if uniq.size <= 1:
        return np.array([], dtype=float)
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    if not fast_logrank or mids.size <= 32:
        return mids
    qs = np.linspace(0.05, 0.95, 16)
    return np.unique(np.quantile(mids, qs))


class _GRFSurvivalTree:
    def __init__(self, *, mtry, min_node_size, max_depth, alpha,
                 honesty, honesty_fraction, honesty_prune_leaves,
                 imbalance_penalty, fast_logrank, prediction_type, random_state):
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.alpha = alpha
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.imbalance_penalty = imbalance_penalty
        self.fast_logrank = fast_logrank
        self.prediction_type = prediction_type
        self.random_state = random_state

    def fit(self, X, Y, D, failure_times):
        n = X.shape[0]
        if self.honesty:
            split_n = int(np.floor(n * self.honesty_fraction))
            est_n = n - split_n
            if split_n <= 0 or est_n <= 0:
                raise ValueError("honesty_fraction leaves no split or estimation sample.")
            perm = self.random_state.permutation(n)
            split_idx = perm[:split_n]
            est_idx = perm[split_n:]
        else:
            split_idx = np.arange(n)
            est_idx = np.arange(n)

        self.failure_times_ = failure_times
        self.root_ = self._grow(
            X, Y, D, split_idx, est_idx, depth=0
        )
        return self

    def _grow(self, X, Y, D, split_idx, est_idx, depth):
        if split_idx.size == 0:
            return _SurvivalTreeNode(
                is_leaf=True,
                survival=_leaf_survival_curve(Y[est_idx], D[est_idx], self.failure_times_, self.prediction_type)
            )
        if (self.max_depth is not None and depth >= self.max_depth) or split_idx.size < 2 * self.min_node_size:
            return _SurvivalTreeNode(
                is_leaf=True,
                survival=_leaf_survival_curve(Y[est_idx], D[est_idx], self.failure_times_, self.prediction_type)
            )

        feature, threshold = self._best_split(X[split_idx], Y[split_idx], D[split_idx])
        if feature is None:
            return _SurvivalTreeNode(
                is_leaf=True,
                survival=_leaf_survival_curve(Y[est_idx], D[est_idx], self.failure_times_, self.prediction_type)
            )

        split_left = split_idx[X[split_idx, feature] <= threshold]
        split_right = split_idx[X[split_idx, feature] > threshold]
        est_left = est_idx[X[est_idx, feature] <= threshold]
        est_right = est_idx[X[est_idx, feature] > threshold]

        if self.honesty and self.honesty_prune_leaves and (est_left.size == 0 or est_right.size == 0):
            return _SurvivalTreeNode(
                is_leaf=True,
                survival=_leaf_survival_curve(Y[est_idx], D[est_idx], self.failure_times_, self.prediction_type)
            )

        left = self._grow(X, Y, D, split_left, est_left, depth + 1)
        right = self._grow(X, Y, D, split_right, est_right, depth + 1)
        return _SurvivalTreeNode(
            is_leaf=False, feature=feature, threshold=float(threshold), left=left, right=right
        )

    def _best_split(self, X, Y, D):
        n, p = X.shape
        features = self.random_state.choice(p, size=min(self.mtry, p), replace=False)
        min_failures = max(1.0, self.alpha * n)
        best_score = -np.inf
        best_feature = None
        best_threshold = None
        for feature in features:
            thresholds = _candidate_thresholds(X[:, feature], fast_logrank=self.fast_logrank)
            for threshold in thresholds:
                left = X[:, feature] <= threshold
                n_left = np.sum(left)
                n_right = n - n_left
                if n_left < self.min_node_size or n_right < self.min_node_size:
                    continue
                fail_left = np.sum(D[left] == 1)
                fail_right = np.sum(D[~left] == 1)
                if fail_left < min_failures or fail_right < min_failures:
                    continue
                score = _logrank_statistic(Y, D, left)
                if not np.isfinite(score):
                    continue
                imbalance = abs(n_left - n_right) / n
                score = score - self.imbalance_penalty * imbalance
                if score > best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _predict_one(self, x, node):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.survival

    def predict_survival_matrix(self, X):
        preds = np.empty((X.shape[0], len(self.failure_times_)), dtype=float)
        for i, x in enumerate(X):
            surv = self._predict_one(x, self.root_)
            if surv is None:
                preds[i] = np.nan
            else:
                preds[i] = surv
        return preds


class SurvivalForest(BaseEstimator):
    """Approximate Python translation of ``grf::survival_forest``.

    The implementation uses an explicit ensemble of honest survival trees with
    log-rank-based splitting so it can expose native out-of-bag predictions on
    the training sample.
    """

    def __init__(self, *,
                 failure_times=None,
                 num_trees=1000,
                 sample_weights=None,
                 clusters=None,
                 equalize_cluster_weights=False,
                 sample_fraction=0.5,
                 mtry=None,
                 min_node_size=15,
                 honesty=True,
                 honesty_fraction=0.5,
                 honesty_prune_leaves=True,
                 alpha=0.05,
                 imbalance_penalty=0.0,
                 prediction_type="Nelson-Aalen",
                 compute_oob_predictions=True,
                 fast_logrank=False,
                 num_threads=None,
                 seed=None,
                 tune_parameters="none",
                 max_depth=None,
                 verbose=0):
        self.failure_times = failure_times
        self.num_trees = num_trees
        self.sample_weights = sample_weights
        self.clusters = clusters
        self.equalize_cluster_weights = equalize_cluster_weights
        self.sample_fraction = sample_fraction
        self.mtry = mtry
        self.min_node_size = min_node_size
        self.honesty = honesty
        self.honesty_fraction = honesty_fraction
        self.honesty_prune_leaves = honesty_prune_leaves
        self.alpha = alpha
        self.imbalance_penalty = imbalance_penalty
        self.prediction_type = prediction_type
        self.compute_oob_predictions = compute_oob_predictions
        self.fast_logrank = fast_logrank
        self.num_threads = num_threads
        self.seed = seed
        self.tune_parameters = tune_parameters
        self.max_depth = max_depth
        self.verbose = verbose

    def _validate_supported_args(self):
        if self.prediction_type not in {"Nelson-Aalen", "Kaplan-Meier"}:
            raise ValueError("prediction_type must be 'Kaplan-Meier' or 'Nelson-Aalen'.")
        if not (0 < self.sample_fraction <= 1):
            raise ValueError("sample_fraction must be in (0, 1].")
        if not (0 <= self.alpha < 0.5):
            raise ValueError("alpha must be in [0, 0.5).")
        if self.honesty and not (0 < self.honesty_fraction < 1):
            raise ValueError("honesty_fraction must be in (0, 1) when honesty=True.")
        if self.equalize_cluster_weights and self.sample_weights is not None:
            raise ValueError("sample_weights must be None when equalize_cluster_weights=True.")
        if self.equalize_cluster_weights and self.clusters is None:
            raise ValueError("equalize_cluster_weights=True requires clusters.")
    def _make_tree(self, random_state):
        return _GRFSurvivalTree(
            mtry=self.mtry_,
            min_node_size=self.min_node_size,
            max_depth=self.max_depth,
            alpha=self.alpha,
            honesty=self.honesty,
            honesty_fraction=self.honesty_fraction,
            honesty_prune_leaves=self.honesty_prune_leaves,
            imbalance_penalty=self.imbalance_penalty,
            fast_logrank=self.fast_logrank,
            prediction_type=self.prediction_type,
            random_state=np.random.RandomState(random_state),
        )

    def _bootstrap_indices(self, n, rng):
        if self.cluster_info_ is None:
            n_boot = max(1, int(np.ceil(self.sample_fraction * n)))
            n_boot = min(n_boot, n)
            sample_idx = rng.choice(n, size=n_boot, replace=False)
            unsampled = np.setdiff1d(np.arange(n), sample_idx, assume_unique=True)
            return sample_idx, unsampled

        unique_clusters, inverse, counts = self.cluster_info_
        n_clusters = len(unique_clusters)
        n_boot_clusters = max(1, int(np.ceil(self.sample_fraction * n_clusters)))
        n_boot_clusters = min(n_boot_clusters, n_clusters)
        sampled_cluster_pos = rng.choice(n_clusters, size=n_boot_clusters, replace=False)
        sampled_pos_set = set(sampled_cluster_pos.tolist())

        if self.equalize_cluster_weights:
            k = int(np.min(counts))
            pieces = []
            for pos in sampled_cluster_pos:
                members = np.flatnonzero(inverse == pos)
                pieces.append(rng.choice(members, size=k, replace=False))
            sample_idx = np.concatenate(pieces)
        else:
            pieces = [np.flatnonzero(inverse == pos) for pos in sampled_cluster_pos]
            sample_idx = np.concatenate(pieces)

        unsampled_cluster_mask = np.array([idx not in sampled_pos_set for idx in inverse])
        unsampled = np.flatnonzero(unsampled_cluster_mask)
        return sample_idx, unsampled

    def _predict_survival_matrix_from_trees(self, estimators, X, grid):
        preds = np.zeros((X.shape[0], len(grid)), dtype=float)
        counts = np.zeros(X.shape[0], dtype=float)
        for tree in estimators:
            tree_pred = tree.predict_survival_matrix(X)
            mask = np.all(np.isfinite(tree_pred), axis=1)
            if np.any(mask):
                preds[mask] += tree_pred[mask]
                counts[mask] += 1
        counts = np.maximum(counts, 1.0)
        preds /= counts[:, None]
        return np.clip(preds, 1e-3, 1.0)

    def fit(self, X, Y, D, *, sample_weight=None):
        self._validate_supported_args()
        X = check_array(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float).ravel()
        D = np.asarray(D, dtype=float).ravel()
        if X.shape[0] != Y.shape[0] or X.shape[0] != D.shape[0]:
            raise ValueError("X, Y and D must have the same number of rows.")
        if np.any(Y < 0):
            raise ValueError("Event times must be non-negative.")
        if not np.all(np.isin(D, [0, 1])):
            raise ValueError("D must contain only 0/1 values.")

        n = X.shape[0]
        self.mtry_ = self.mtry if self.mtry is not None else _default_mtry(X.shape[1])
        self.X_train_ = np.array(X, copy=True)
        self.Y_train_ = np.array(Y, copy=True)
        self.D_train_ = np.array(D, copy=True)
        base_sample_weight = self.sample_weights if sample_weight is None else sample_weight
        self.sample_weight_ = None if base_sample_weight is None else _check_sample_weight(base_sample_weight, X, dtype=float)
        cluster_weights, self.cluster_info_ = _cluster_weight_vector(
            self.clusters, self.equalize_cluster_weights, n
        )
        if self.sample_weight_ is None:
            self.effective_sample_weight_ = cluster_weights
        else:
            self.effective_sample_weight_ = self.sample_weight_ * cluster_weights

        if self.failure_times is None:
            self.failure_times_ = np.sort(np.unique(Y))
        else:
            grid = np.asarray(self.failure_times, dtype=float).ravel()
            if grid.ndim != 1 or grid.size == 0:
                raise ValueError("failure_times must be a non-empty 1D array.")
            if np.min(Y) < np.min(grid):
                raise ValueError("failure_times should start on or before min(Y).")
            self.failure_times_ = np.sort(grid)

        rng = np.random.RandomState(self.seed)
        self.estimators_ = []
        self.bootstrap_indices_ = []

        if self.compute_oob_predictions:
            oob_sum = np.zeros((n, len(self.failure_times_)), dtype=float)
            oob_count = np.zeros(n, dtype=int)

        for _ in range(self.num_trees):
            sample_idx, unsampled = self._bootstrap_indices(n, rng)
            tree = self._make_tree(rng.randint(np.iinfo(np.int32).max))
            tree.fit(
                X[sample_idx],
                Y[sample_idx],
                D[sample_idx],
                self.failure_times_,
            )
            self.estimators_.append(tree)
            self.bootstrap_indices_.append(sample_idx)

            if self.compute_oob_predictions and unsampled.size > 0:
                preds = tree.predict_survival_matrix(X[unsampled])
                valid = np.all(np.isfinite(preds), axis=1)
                if np.any(valid):
                    oob_sum[unsampled[valid]] += preds[valid]
                    oob_count[unsampled[valid]] += 1

        if self.compute_oob_predictions:
            self.oob_predictions_ = np.empty_like(oob_sum)
            mask = oob_count > 0
            if np.any(mask):
                self.oob_predictions_[mask] = oob_sum[mask] / oob_count[mask][:, None]
            if np.any(~mask):
                warn(
                    "Some inputs do not have OOB survival estimates. Falling back to full-forest predictions.",
                    UserWarning,
                    stacklevel=2,
                )
                self.oob_predictions_[~mask] = self._predict_survival_matrix_from_trees(
                    self.estimators_, X[~mask], self.failure_times_
                )
            self.oob_predictions_ = np.clip(self.oob_predictions_, 1e-3, 1.0)
            self.predictions_ = self.oob_predictions_
        else:
            self.oob_predictions_ = None
            self.predictions_ = self._predict_survival_matrix_from_trees(
                self.estimators_, X, self.failure_times_
            )
        return self

    def predict_survival_matrix(self, X, *, failure_times=None):
        X = check_array(np.asarray(X, dtype=float))
        grid = self.failure_times_ if failure_times is None else np.asarray(failure_times, dtype=float).ravel()
        if (
            self.compute_oob_predictions
            and failure_times is None
            and X.shape == self.X_train_.shape
            and np.array_equal(X, self.X_train_)
        ):
            return np.array(self.oob_predictions_, copy=True)
        return self._predict_survival_matrix_from_trees(self.estimators_, X, grid)

    def predict_cumulative_hazard_matrix(self, X, *, failure_times=None):
        surv = self.predict_survival_matrix(X, failure_times=failure_times)
        return -np.log(np.clip(surv, 1e-8, 1.0))

    def oob_predict(self, Xtrain=None):
        if not self.compute_oob_predictions:
            raise ValueError("compute_oob_predictions=False, so no OOB predictions are stored.")
        if Xtrain is not None:
            Xtrain = check_array(np.asarray(Xtrain, dtype=float))
            if Xtrain.shape != self.X_train_.shape or not np.array_equal(Xtrain, self.X_train_):
                raise ValueError("oob_predict is only defined on the training sample.")
        return np.array(self.oob_predictions_, copy=True)

    def predict(self, X=None, *, failure_times=None):
        if X is None:
            preds = self.oob_predict()
            grid = self.failure_times_
        else:
            preds = self.predict_survival_matrix(X, failure_times=failure_times)
            grid = self.failure_times_ if failure_times is None else np.asarray(failure_times, dtype=float).ravel()
        return SurvivalForestPredictionResult(predictions=preds, failure_times=np.array(grid, copy=True))


def survival_forest(X, Y, D, *,
                    failure_times=None,
                    num_trees=1000,
                    sample_weights=None,
                    clusters=None,
                    equalize_cluster_weights=False,
                    sample_fraction=0.5,
                    mtry=None,
                    min_node_size=15,
                    honesty=True,
                    honesty_fraction=0.5,
                    honesty_prune_leaves=True,
                    alpha=0.05,
                    imbalance_penalty=0.0,
                    prediction_type="Nelson-Aalen",
                    compute_oob_predictions=True,
                    fast_logrank=False,
                    num_threads=None,
                    seed=None,
                    tune_parameters="none",
                    max_depth=None,
                    verbose=0):
    """GRF-style functional entry point mirroring ``grf::survival_forest``."""
    est = SurvivalForest(
        failure_times=failure_times,
        num_trees=num_trees,
        sample_weights=sample_weights,
        clusters=clusters,
        equalize_cluster_weights=equalize_cluster_weights,
        sample_fraction=sample_fraction,
        mtry=mtry,
        min_node_size=min_node_size,
        honesty=honesty,
        honesty_fraction=honesty_fraction,
        honesty_prune_leaves=honesty_prune_leaves,
        alpha=alpha,
        imbalance_penalty=imbalance_penalty,
        prediction_type=prediction_type,
        compute_oob_predictions=compute_oob_predictions,
        fast_logrank=fast_logrank,
        num_threads=num_threads,
        seed=seed,
        tune_parameters=tune_parameters,
        max_depth=max_depth,
        verbose=verbose,
    )
    return est.fit(X, Y, D, sample_weight=sample_weights)
