# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Cross-fitted metalearners for heterogeneous treatment effects.

This module implements OOS-only learners without relying on ``_OrthoLearner``.
The implementation is organized around explicit stages:

``_crossfit_nuisances``
    Cross-fit nuisance models only.
``_crossfit_final`` / ``_fit_final``
    Cross-fit or fit the learner-specific final model.
``effects``
    Return HTE estimates, using cached OOF values on the training rows.
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.utils import check_array
from sksurv.ensemble import RandomSurvivalForest
from sksurv.functions import StepFunction

from ..censor._nuisance import _make_sksurv_y


_PROPENSITY_CLIP = 1e-3


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_default_binary_nuisance_model():
    return RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=123,
    )


def _make_default_continuous_nuisance_model():
    return RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=123,
    )


def _make_default_survival_nuisance_model():
    return RandomSurvivalForest(
        n_estimators=100,
        min_samples_leaf=5,
        random_state=123,
    )

def _resolve_binary_nuisance_model(model):
    return _make_default_binary_nuisance_model() if model in ('auto', None) else model


def _resolve_continuous_nuisance_model(model):
    return _make_default_continuous_nuisance_model() if model in ('auto', None) else model


def _resolve_survival_nuisance_model(model):
    return _make_default_survival_nuisance_model() if model in ('auto', None) else model


def _build_interaction_features(treatment, X):
    t = np.asarray(treatment, dtype=float).reshape(-1, 1)
    X_arr = np.asarray(X, dtype=float)
    return np.hstack([t, X_arr, t * X_arr])


def _eval_survival_on_grid(model, X, time_grid):
    surv_fns = model.predict_survival_function(np.asarray(X))
    grid = np.asarray(time_grid, dtype=float)
    vals = []
    for fn in surv_fns:
        dom = getattr(fn, 'domain', None)
        if dom is not None:
            grid_eval = np.clip(grid, dom[0], dom[1])
        else:
            x = np.asarray(getattr(fn, 'x', grid), dtype=float)
            grid_eval = np.clip(grid, x[0], x[-1])
        vals.append(np.clip(fn(grid_eval), 1e-3, 1.0))
    return np.asarray(vals, dtype=float)


def _compute_rmst(surv_fns, tau, time_grid=None):
    tau = float(tau)
    if isinstance(surv_fns, np.ndarray) and surv_fns.dtype != object:
        surv_arr = np.asarray(surv_fns, dtype=float)
        if time_grid is None:
            raise TypeError("_compute_rmst needs time_grid when given a raw survival array.")
        grid = np.asarray(time_grid, dtype=float)
        if grid.ndim != 1:
            raise ValueError("time_grid must be one-dimensional.")
        if surv_arr.ndim != 2 or surv_arr.shape[1] != grid.shape[0]:
            raise ValueError("Raw survival array must have shape (n, len(time_grid)).")
        grid_tau = grid[grid <= tau]
        if grid_tau.size == 0 or grid_tau[-1] < tau:
            grid_tau = np.append(grid_tau, tau)
        vals = np.empty((surv_arr.shape[0], grid_tau.size), dtype=float)
        for i in range(surv_arr.shape[0]):
            vals[i] = np.interp(grid_tau, grid, surv_arr[i], left=1.0, right=surv_arr[i, -1])
        return np.trapz(vals, grid_tau, axis=1)
    out = []
    for fn in surv_fns:
        x = np.asarray(fn.x, dtype=float)
        y = np.asarray(fn.y, dtype=float)
        if x[0] > 0:
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, 1.0)
        x = np.append(x, tau)
        x = np.clip(x, 0.0, tau)
        x = np.maximum.accumulate(x)
        y_ext = np.append(y, y[-1])
        area = 0.0
        for left, right, val in zip(x[:-1], x[1:], y_ext[:-1]):
            if right > left:
                area += (right - left) * val
        out.append(area)
    return np.asarray(out, dtype=float)


def _compute_cif_on_grid(overall_surv, cause_surv):
    overall = np.clip(np.asarray(overall_surv, dtype=float), 1e-3, 1.0)
    cause = np.clip(np.asarray(cause_surv, dtype=float), 1e-3, 1.0)
    n, m = overall.shape
    cif = np.zeros((n, m), dtype=float)
    prev_overall = np.ones(n, dtype=float)
    prev_cause = np.ones(n, dtype=float)
    running = np.zeros(n, dtype=float)
    for j in range(m):
        hazard = 1.0 - np.clip(cause[:, j] / prev_cause, 0.0, 1.0)
        running = running + prev_overall * hazard
        cif[:, j] = np.clip(running, 0.0, 1.0)
        prev_overall = overall[:, j]
        prev_cause = cause[:, j]
    return cif


def _compute_rmtl_from_cif(cif, time_grid, tau):
    cif_arr = np.asarray(cif, dtype=float)
    grid = np.asarray(time_grid, dtype=float)
    tau = float(tau)
    if grid.ndim != 1:
        raise ValueError("time_grid must be one-dimensional.")
    if grid.size == 0:
        return np.zeros(cif_arr.shape[0], dtype=float)
    grid_tau = grid[grid <= tau]
    if grid_tau.size == 0 or grid_tau[-1] < tau:
        grid_tau = np.append(grid_tau, tau)
    vals = np.empty((cif_arr.shape[0], grid_tau.size), dtype=float)
    for i in range(cif_arr.shape[0]):
        vals[i] = np.interp(grid_tau, grid, cif_arr[i], left=0.0, right=cif_arr[i, -1])
    return np.trapz(vals, grid_tau, axis=1)

def _fit_propensity_fold(model, T, X):
    """Fit propensity model and return clipped probabilities."""
    m = clone(model, safe=False)
    m.fit(X, T)
    if hasattr(m, 'predict_proba'):
        e = m.predict_proba(X)[:, 1]
    else:
        e = m.predict(X)
    return np.clip(e, _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP), m


def _fit_mu_fold(model, Y, T, X):
    """Fit per-arm outcome models; return (mu_0, mu_1, model_0, model_1)."""
    m0 = clone(model, safe=False)
    m1 = clone(model, safe=False)
    mask0, mask1 = T == 0, T == 1
    m0.fit(X[mask0], Y[mask0])
    m1.fit(X[mask1], Y[mask1])
    mu0 = m0.predict(X)
    mu1 = m1.predict(X)
    return mu0, mu1, m0, m1


def _same_train_features(X, X_train):
    """Whether X matches the training covariates exactly."""
    if X is None or X_train is None:
        return False
    X_arr = np.asarray(X)
    return X_arr.shape == X_train.shape and np.array_equal(X_arr, X_train)


def _build_outer_oof_folds(cv, X, T, random_state):
    """Build OOF folds for nuisance or final cross-fitting."""
    if cv == 1:
        raise ValueError("CrossFit learners require cv >= 2 for OOF prediction caching")

    splitter = check_cv(cv, [0], classifier=True)
    if splitter != cv and isinstance(splitter, (KFold, StratifiedKFold)):
        splitter.shuffle = True
        splitter.random_state = random_state
    return list(splitter.split(np.asarray(X), np.asarray(T).ravel().astype(int)))


def _as_tuple(values):
    return values if isinstance(values, tuple) else (values,)


def _subset_tuple(values, idx):
    return tuple(np.asarray(v)[idx] for v in values)


def _crossfit_nuisances(nuisance_factory, Y, T, *, X, cv, random_state):
    """Cross-fit nuisance models and stitch held-out nuisance predictions."""
    X_arr = np.asarray(X)
    T_arr = np.asarray(T).ravel().astype(int)
    folds = _build_outer_oof_folds(cv, X_arr, T_arr, random_state)
    fitted = []
    oof = None

    for train_idx, test_idx in folds:
        nuisance = nuisance_factory()
        nuisance.fit(Y[train_idx], T_arr[train_idx], X=X_arr[train_idx])
        preds = _as_tuple(nuisance.predict(Y[test_idx], T_arr[test_idx], X=X_arr[test_idx]))
        preds = tuple(np.asarray(p) for p in preds)
        if oof is None:
            oof = [np.full((X_arr.shape[0],) + p.shape[1:], np.nan, dtype=float) for p in preds]
        for slot, pred in zip(oof, preds):
            slot[test_idx] = pred
        fitted.append(nuisance)

    if oof is None:
        return (), []
    return tuple(oof), fitted


def _fit_final(final_model, Y, T, *, X, nuisances=(), sample_weight=None):
    """Fit the final model once on the full sample only."""
    model = clone(final_model, safe=False)
    model.fit(
        Y, T, X=np.asarray(X), nuisances=nuisances,
        sample_weight=sample_weight,
    )
    return model


def _crossfit_final(final_model, Y, T, *, X, nuisances=(), cv, random_state):
    """Cross-fit the final model only and return the full fit plus OOF effects."""
    X_arr = np.asarray(X)
    T_arr = np.asarray(T).ravel().astype(int)
    folds = _build_outer_oof_folds(cv, X_arr, T_arr, random_state)
    full_model = _fit_final(final_model, Y, T_arr, X=X_arr, nuisances=nuisances)
    oof = np.full(X_arr.shape[0], np.nan, dtype=float)

    for train_idx, test_idx in folds:
        fold_model = _fit_final(
            final_model,
            Y[train_idx],
            T_arr[train_idx],
            X=X_arr[train_idx],
            nuisances=_subset_tuple(nuisances, train_idx),
        )
        oof[test_idx] = np.asarray(fold_model.predict(X_arr[test_idx])).ravel()

    return full_model, oof


def _cache_training_oof_effects(estimator, Y, T, *, X):
    """Cache an extra outer OOF effect layer on the full training dataset."""
    X_arr = np.asarray(X)
    T_arr = np.asarray(T).ravel().astype(int)
    oof = np.full(X_arr.shape[0], np.nan, dtype=float)

    for train_idx, test_idx in _build_outer_oof_folds(estimator.cv, X_arr, T_arr, estimator.random_state):
        est = clone(estimator, safe=False)
        est._skip_training_oof_ = True
        est._skip_training_separable_oof_ = True
        est.fit(Y[train_idx], T_arr[train_idx], X=X_arr[train_idx])
        oof[test_idx] = np.asarray(est.effect(X_arr[test_idx])).ravel()

    return oof


def effects(estimator, X=None):
    """Return HTE estimates, using cached OOF values on the training rows."""
    if _same_train_features(X, getattr(estimator, '_training_X_oof_', None)):
        return np.asarray(estimator._training_oof_effect_).ravel()
    return np.asarray(estimator.const_marginal_effect(X)).ravel()


class _BaseCrossfitEstimator(BaseEstimator):
    """Minimal estimator API for OOS-only metalearners."""

    def __init__(self, *, cv=3, categories='auto', random_state=None):
        self.cv = cv
        self.categories = categories
        self.random_state = random_state
        self._training_X_oof_ = None
        self._training_oof_effect_ = None
        self._crossfit_split_stages_ = 0
        self._entire_dataset_caching_split_stages_ = 0

    def _check_fitted_dims(self, X):
        X_arr = np.asarray(X)
        n_features = getattr(self, '_n_features_in_', None)
        if n_features is not None and X_arr.shape[1] != n_features:
            raise ValueError(
                f"X has {X_arr.shape[1]} features but this estimator was fit with {n_features}."
            )

    def effect(self, X):
        return effects(self, X)

    def effects(self, X):
        return self.effect(X)

    def ate(self, X):
        return float(np.mean(self.effect(X)))


def _cache_training_oof_separable(estimator, Y, T, X):
    """Cache outer-fold OOF separable direct/indirect predictions on training rows."""
    X_arr = np.asarray(X)
    T_arr = np.asarray(T).ravel().astype(int)
    direct = np.full(X_arr.shape[0], np.nan)
    indirect = np.full(X_arr.shape[0], np.nan)

    for train_idx, test_idx in _build_outer_oof_folds(estimator.cv, X_arr, T_arr, estimator.random_state):
        est = clone(estimator, safe=False)
        est._skip_training_oof_ = True
        est._skip_training_separable_oof_ = True
        est.fit(Y[train_idx], T_arr[train_idx], X=X_arr[train_idx])
        direct_fold, indirect_fold = est._compute_separable_effects(X_arr[test_idx])
        direct[test_idx] = np.asarray(direct_fold).ravel()
        indirect[test_idx] = np.asarray(indirect_fold).ravel()

    return direct, indirect


class _ConstantSurvivalModel:
    """Fallback survival model for fold slices with no observed events."""

    def __init__(self, max_time):
        max_time = float(max(max_time, 1e-8))
        self._fn = StepFunction(
            x=np.array([0.0, max_time], dtype=float),
            y=np.array([1.0, 1.0], dtype=float),
        )

    def predict_survival_function(self, X):
        return [self._fn] * np.asarray(X).shape[0]


def _fit_survival_or_constant(model, X, time, event_bool):
    """Fit a survival model, falling back to constant survival if needed."""
    if not np.any(event_bool):
        return _ConstantSurvivalModel(np.max(time) if len(time) else 1.0)
    fitted = clone(model, safe=False)
    try:
        fitted.fit(X, _make_sksurv_y(time, event_bool))
        return fitted
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        warnings.warn(
            "Survival nuisance model fit failed on a fold; "
            "falling back to a constant survival curve. "
            f"Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return _ConstantSurvivalModel(np.max(time) if len(time) else 1.0)


def _mean_nuisance_predictions(models_nuisance, X):
    """Average nuisance predictions across fitted cross-fit models."""
    X_arr = check_array(np.asarray(X, dtype=float))
    accum = None
    n_models = 0
    for fitted_per_iter in models_nuisance:
        for mdl in fitted_per_iter:
            nuisances = mdl.predict(None, None, X=X_arr)
            if not isinstance(nuisances, tuple):
                nuisances = (nuisances,)
            nuisances = tuple(np.asarray(n, dtype=float) for n in nuisances)
            if accum is None:
                accum = [np.array(n, copy=True) for n in nuisances]
            else:
                for idx, nuis in enumerate(nuisances):
                    accum[idx] += nuis
            n_models += 1
    if accum is None or n_models == 0:
        raise RuntimeError("No fitted nuisance models are available for prediction.")
    return tuple(n / n_models for n in accum)


class _DirectNuisanceCateMixin:
    """Predict CATE directly from averaged nuisance models instead of a final regressor."""

    def _effect_from_nuisances(self, nuisances):
        raise NotImplementedError

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        X_arr = getattr(self, '_training_X_oof_', None) if X is None else X
        if X_arr is None:
            raise ValueError("X must be provided for direct learner effect prediction.")
        nuisances = _mean_nuisance_predictions(self.models_nuisance_, X_arr)
        return np.asarray(self._effect_from_nuisances(nuisances), dtype=float).reshape(-1, 1)


def _fit_oof_direct(estimator, nuisance_factory, Y, T, *, X, cache_training_oof=False):
    """Fit a direct learner whose effect is computed only from nuisances."""
    X_arr = np.asarray(X)
    T_arr = np.asarray(T).ravel().astype(int)
    estimator._n_features_in_ = X_arr.shape[1]
    estimator._training_X_oof_ = np.array(X_arr, copy=True)
    estimator._crossfit_split_stages_ = 1
    estimator._entire_dataset_caching_split_stages_ = 1 if cache_training_oof else 0
    nuisances, fitted = _crossfit_nuisances(
        nuisance_factory, Y, T_arr, X=X_arr, cv=estimator.cv, random_state=estimator.random_state
    )
    estimator.models_nuisance_ = [tuple(fitted)]
    base_oof = np.asarray(estimator._effect_from_nuisances(nuisances), dtype=float).ravel()
    estimator._training_oof_effect_ = base_oof
    if cache_training_oof and not getattr(estimator, '_skip_training_oof_', False):
        estimator._training_oof_effect_ = _cache_training_oof_effects(
            estimator, Y, T_arr, X=X_arr
        )
    return estimator


def _fit_oof_two_stage(estimator, nuisance_factory, final_factory, Y, T, *, X,
                       cache_training_oof=False, caching_stage_count=None):
    """Fit a learner with cross-fitted nuisances and a cross-fitted final model."""
    X_arr = np.asarray(X)
    T_arr = np.asarray(T).ravel().astype(int)
    estimator._n_features_in_ = X_arr.shape[1]
    estimator._training_X_oof_ = np.array(X_arr, copy=True)
    estimator._crossfit_split_stages_ = 2
    if caching_stage_count is None:
        estimator._entire_dataset_caching_split_stages_ = 1 if cache_training_oof else 0
    else:
        estimator._entire_dataset_caching_split_stages_ = int(caching_stage_count)
    nuisances, fitted = _crossfit_nuisances(
        nuisance_factory, Y, T_arr, X=X_arr, cv=estimator.cv, random_state=estimator.random_state
    )
    estimator.models_nuisance_ = [tuple(fitted)]
    estimator.model_final_, oof = _crossfit_final(
        final_factory(), Y, T_arr, X=X_arr, nuisances=nuisances,
        cv=estimator.cv, random_state=estimator.random_state,
    )
    estimator._training_oof_effect_ = oof
    if cache_training_oof and not getattr(estimator, '_skip_training_oof_', False):
        estimator._training_oof_effect_ = _cache_training_oof_effects(
            estimator, Y, T_arr, X=X_arr
        )
    return estimator


# ---------------------------------------------------------------------------
# CrossFitTLearner
# ---------------------------------------------------------------------------

class _TLNuisance:
    """Nuisance for CrossFitTLearner: per-arm outcome models mu_0, mu_1."""

    def __init__(self, model_mu):
        self._model_mu = model_mu

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int)
        Y_arr = np.asarray(Y).ravel()
        X_arr = np.asarray(X)
        self._mu0, self._mu1, self._m0, self._m1 = _fit_mu_fold(
            self._model_mu, Y_arr, T_arr, X_arr)
        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        mu0 = self._m0.predict(X_arr)
        mu1 = self._m1.predict(X_arr)
        return mu0, mu1

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class _TLFinal:
    """Final stage for CrossFitTLearner: regress (mu_1 - mu_0) on X."""

    def __init__(self, model_final):
        self._model_final = clone(model_final, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        mu0, mu1 = nuisances
        target = np.asarray(mu1 - mu0, dtype=float).ravel()
        mask = np.isfinite(target)
        if not np.any(mask):
            raise ValueError("Final TLearner target contains no finite entries.")
        self._model_final.fit(np.asarray(X)[mask], target[mask])
        return self

    def predict(self, X=None):
        return self._model_final.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        mu0, mu1 = nuisances
        target = np.asarray(mu1 - mu0, dtype=float).ravel()
        mask = np.isfinite(target)
        pred = self._model_final.predict(np.asarray(X)[mask])
        return np.mean((target[mask] - pred) ** 2)


class TLearner(_DirectNuisanceCateMixin, _BaseCrossfitEstimator):
    """T-learner with cross-fitting.

    Fits per-arm outcome models mu_0(X) and mu_1(X) on held-out folds and
    predicts CATE directly as mu_1(X) - mu_0(X). Cross-fitting via ``cv``
    folds yields fold-held-out training predictions.

    Parameters
    ----------
    model_mu : sklearn regressor
        Outcome model fitted separately for each treatment arm.
    cv : int, default 3
        Number of cross-fitting folds.
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, model_mu='auto', cv=3,
                 categories='auto', random_state=None):
        self.model_mu = _resolve_continuous_nuisance_model(model_mu)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def _effect_from_nuisances(self, nuisances):
        mu0, mu1 = nuisances
        return mu1 - mu0

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitTLearner.

        Parameters
        ----------
        Y : array_like, shape (n,)
            Outcome or pre-computed pseudo-outcome.
        T : array_like, shape (n,)
            Binary treatment (0/1).
        X : array_like, shape (n, d_x)
            Covariates.
        """
        return _fit_oof_direct(
            self, lambda: _TLNuisance(self.model_mu), np.asarray(Y), T, X=X,
            cache_training_oof=True,
        )


# ---------------------------------------------------------------------------
# CrossFitSLearner
# ---------------------------------------------------------------------------

class _SLNuisance:
    """Nuisance for CrossFitSLearner: single pooled model mu(X, T)."""

    def __init__(self, overall_model):
        self._overall_model = overall_model

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int).reshape(-1, 1)
        Y_arr = np.asarray(Y).ravel()
        X_arr = np.asarray(X)
        feat = np.hstack([T_arr, X_arr, T_arr * X_arr])
        self._m = clone(self._overall_model, safe=False)
        self._m.fit(feat, Y_arr)
        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        n = X_arr.shape[0]
        feat0 = np.hstack([np.zeros((n, 1)), X_arr, np.zeros_like(X_arr)])
        feat1 = np.hstack([np.ones((n, 1)), X_arr, X_arr])
        mu0 = self._m.predict(feat0)
        mu1 = self._m.predict(feat1)
        return mu0, mu1

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class SLearner(_DirectNuisanceCateMixin, _BaseCrossfitEstimator):
    """S-learner with cross-fitting.

    Fits a single pooled model mu(X, T) with treatment-covariate interactions
    and predicts CATE directly as mu(X,1) - mu(X,0).

    Parameters
    ----------
    overall_model : sklearn regressor
        Pooled outcome model (features = [T, X, T*X]).
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, overall_model='auto', cv=3,
                 categories='auto', random_state=None):
        self.overall_model = _resolve_continuous_nuisance_model(overall_model)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def _effect_from_nuisances(self, nuisances):
        mu0, mu1 = nuisances
        return mu1 - mu0

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitSLearner."""
        return _fit_oof_direct(
            self, lambda: _SLNuisance(self.overall_model), np.asarray(Y), T, X=X,
            cache_training_oof=True,
        )


# ---------------------------------------------------------------------------
# CrossFitXLearner
# ---------------------------------------------------------------------------

class _XLNuisance:
    """Nuisance for CrossFitXLearner.

    Computes per-arm imputed effects (D_0, D_1) and propensity scores OOS.
    """

    def __init__(self, model_mu, propensity_model):
        self._model_mu = model_mu
        self._propensity_model = propensity_model

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int)
        Y_arr = np.asarray(Y).ravel()
        X_arr = np.asarray(X)
        self._mu0, self._mu1, self._m0, self._m1 = _fit_mu_fold(
            self._model_mu, Y_arr, T_arr, X_arr)
        self._e, self._pm = _fit_propensity_fold(self._propensity_model, T_arr, X_arr)
        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int)
        Y_arr = np.asarray(Y).ravel()
        X_arr = np.asarray(X)
        n = X_arr.shape[0]

        mu0 = self._m0.predict(X_arr)
        mu1 = self._m1.predict(X_arr)
        e = np.clip(
            self._pm.predict_proba(X_arr)[:, 1]
            if hasattr(self._pm, 'predict_proba')
            else self._pm.predict(X_arr),
            _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP)

        # imputed effects per arm
        D0 = np.where(T_arr == 0, mu1 - Y_arr, np.nan)  # for control units
        D1 = np.where(T_arr == 1, Y_arr - mu0, np.nan)  # for treated units
        return D0, D1, e

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class _XLFinal:
    """Final stage for CrossFitXLearner.

    Fits per-arm CATE models on imputed effects, then combines via propensity.
    """

    def __init__(self, model_final, propensity_model):
        self._model_final = model_final
        self._propensity_model = propensity_model

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        D0, D1, e = nuisances
        T_arr = np.asarray(T).ravel().astype(int)
        X_arr = np.asarray(X)

        mask0 = ~np.isnan(D0)
        mask1 = ~np.isnan(D1)

        self._m0 = clone(self._model_final, safe=False)
        self._m1 = clone(self._model_final, safe=False)

        if mask0.sum() > 0:
            self._m0.fit(X_arr[mask0], D0[mask0])
        if mask1.sum() > 0:
            self._m1.fit(X_arr[mask1], D1[mask1])

        self._pm = clone(self._propensity_model, safe=False)
        self._pm.fit(X_arr, T_arr)
        return self

    def predict(self, X=None):
        X_arr = np.asarray(X)
        tau0 = self._m0.predict(X_arr)
        tau1 = self._m1.predict(X_arr)
        if hasattr(self._pm, 'predict_proba'):
            e = self._pm.predict_proba(X_arr)[:, 1]
        else:
            e = self._pm.predict(X_arr)
        e = np.clip(np.asarray(e, dtype=float).ravel(), _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP)
        return ((1 - e) * tau1 + e * tau0).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        return None


class XLearner(_BaseCrossfitEstimator):
    """X-learner with cross-fitting.

    Computes per-arm imputed effects out-of-sample, fits per-arm CATE models,
    and combines them via propensity weighting.

    Parameters
    ----------
    model_mu : sklearn regressor
        Outcome model fitted separately per arm.
    propensity_model : sklearn classifier
        Propensity model P(T=1|X).
    model_final : sklearn regressor
        Per-arm CATE model fitted on imputed effects.
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, model_mu='auto', propensity_model='auto',
                 model_final='auto', cv=3, categories='auto', random_state=None):
        self.model_mu = _resolve_continuous_nuisance_model(model_mu)
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_final = _resolve_continuous_nuisance_model(model_final)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitXLearner."""
        return _fit_oof_two_stage(
            self,
            lambda: _XLNuisance(self.model_mu, self.propensity_model),
            lambda: _XLFinal(self.model_final, self.propensity_model),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitIPTWLearner
# ---------------------------------------------------------------------------

class _IPTWNuisance:
    """Nuisance for CrossFitIPTWLearner: propensity e(X)."""

    def __init__(self, propensity_model):
        self._propensity_model = propensity_model

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int)
        X_arr = np.asarray(X)
        self._e, self._pm = _fit_propensity_fold(self._propensity_model, T_arr, X_arr)
        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        if hasattr(self._pm, 'predict_proba'):
            e = self._pm.predict_proba(X_arr)[:, 1]
        else:
            e = self._pm.predict(X_arr)
        return (np.clip(e, _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP),)

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class _IPTWFinal:
    """Final stage for CrossFitIPTWLearner: regress phi = Y*(T/e - (1-T)/(1-e)) on X."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        (e,) = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        phi = Y_arr * (T_arr / e - (1 - T_arr) / (1 - e))
        self._model_cate.fit(X, phi)
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        (e,) = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        phi = Y_arr * (T_arr / e - (1 - T_arr) / (1 - e))
        pred = self._model_cate.predict(X)
        return np.mean((phi - pred) ** 2)


class IPTWLearner(_BaseCrossfitEstimator):
    """IPTW learner with cross-fitting of propensity.

    Cross-fits the propensity model P(T=1|X), then regresses the IPTW
    pseudo-outcome on X using the final CATE model.

    Parameters
    ----------
    propensity_model : sklearn classifier, default LogisticRegression()
        Model for P(T=1|X).
    model_cate : sklearn regressor
        Final CATE regression model.
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, propensity_model='auto', model_cate='auto',
                 cv=3, categories='auto', random_state=None):
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitIPTWLearner.

        Parameters
        ----------
        Y : array_like, shape (n,)
            Pre-computed CUT pseudo-outcome (e.g. from ``aipcw_cut_rmst``).
        T : array_like, shape (n,)
            Binary treatment (0/1).
        X : array_like, shape (n, d_x)
            Covariates.
        """
        return _fit_oof_two_stage(
            self,
            lambda: _IPTWNuisance(self.propensity_model),
            lambda: _IPTWFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitAIPTWLearner
# ---------------------------------------------------------------------------

class _AIPTWFinal:
    """Final stage for CrossFitAIPTWLearner: regress AIPTW pseudo-outcome on X."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        e, mu0, mu1 = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        phi = (
            Y_arr * (T_arr / e - (1 - T_arr) / (1 - e))
            + (1 - T_arr / e) * mu1
            - (1 - (1 - T_arr) / (1 - e)) * mu0
        )
        self._model_cate.fit(X, phi)
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        return None


class AIPTWLearner(_BaseCrossfitEstimator):
    """Augmented IPTW learner with cross-fitting.

    Cross-fits propensity and per-arm outcome models, builds the AIPTW
    pseudo-outcome, and regresses that pseudo-outcome on X.

    Parameters
    ----------
    propensity_model : sklearn classifier, default LogisticRegression()
    model_mu : sklearn regressor
        Per-arm outcome model.
    model_cate : sklearn regressor
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, propensity_model='auto', model_mu='auto',
                 model_cate='auto', cv=3, categories='auto', random_state=None):
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_mu = _resolve_continuous_nuisance_model(model_mu)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitAIPTWLearner."""
        return _fit_oof_two_stage(
            self,
            lambda: _MCEANuisance(self.propensity_model, self.model_mu),
            lambda: _AIPTWFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitMCLearner
# ---------------------------------------------------------------------------

class _MCFinal:
    """Final stage for CrossFitMCLearner: weighted regression of MC pseudo-outcome on X."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        (e,) = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        sign = 2 * T_arr - 1
        phi = 2 * sign * Y_arr
        weights = sign * (T_arr - e) / (4 * e * (1 - e))
        self._model_cate.fit(X, phi, sample_weight=weights)
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        return None


class MCLearner(_BaseCrossfitEstimator):
    """Modified-covariate (MC) learner with cross-fitting.

    Cross-fits the propensity model, then performs weighted regression using
    the MC pseudo-outcome and IPW weights.

    Parameters
    ----------
    propensity_model : sklearn classifier, default LogisticRegression()
    model_cate : sklearn regressor supporting ``sample_weight``
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, propensity_model='auto', model_cate='auto',
                 cv=3, categories='auto', random_state=None):
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitMCLearner."""
        return _fit_oof_two_stage(
            self,
            lambda: _IPTWNuisance(self.propensity_model),
            lambda: _MCFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitMCEALearner
# ---------------------------------------------------------------------------

class _MCEANuisance:
    """Nuisance for CrossFitMCEALearner: propensity e(X) + per-arm mu_0, mu_1."""

    def __init__(self, propensity_model, model_mu):
        self._propensity_model = propensity_model
        self._model_mu = model_mu

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int)
        Y_arr = np.asarray(Y).ravel()
        X_arr = np.asarray(X)
        self._e, self._pm = _fit_propensity_fold(self._propensity_model, T_arr, X_arr)
        self._mu0, self._mu1, self._m0, self._m1 = _fit_mu_fold(
            self._model_mu, Y_arr, T_arr, X_arr)
        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        if hasattr(self._pm, 'predict_proba'):
            e = self._pm.predict_proba(X_arr)[:, 1]
        else:
            e = self._pm.predict(X_arr)
        e = np.clip(e, _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP)
        mu0 = self._m0.predict(X_arr)
        mu1 = self._m1.predict(X_arr)
        return e, mu0, mu1

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class _MCEAFinal:
    """Final stage for CrossFitMCEALearner: augmented MC weighted regression."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        e, mu0, mu1 = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        residual = Y_arr - e * mu1 - (1 - e) * mu0
        sign = 2 * T_arr - 1
        phi = 2 * sign * residual
        weights = sign * (T_arr - e) / (4 * e * (1 - e))
        self._model_cate.fit(X, phi, sample_weight=weights)
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        return None


class MCEALearner(_BaseCrossfitEstimator):
    """MC learner with efficiency augmentation and cross-fitting.

    Cross-fits propensity and per-arm outcome models, then performs augmented
    MC weighted regression.

    Parameters
    ----------
    propensity_model : sklearn classifier, default LogisticRegression()
    model_mu : sklearn regressor
        Per-arm outcome model.
    model_cate : sklearn regressor supporting ``sample_weight``
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, propensity_model='auto', model_mu='auto',
                 model_cate='auto', cv=3, categories='auto', random_state=None):
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_mu = _resolve_continuous_nuisance_model(model_mu)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitMCEALearner."""
        return _fit_oof_two_stage(
            self,
            lambda: _MCEANuisance(self.propensity_model, self.model_mu),
            lambda: _MCEAFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitRALearner
# ---------------------------------------------------------------------------

class _RAFinal:
    """Final stage for CrossFitRALearner: regress RA pseudo-outcome on X."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        mu0, mu1 = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        phi = T_arr * (Y_arr - mu0) + (1 - T_arr) * (mu1 - Y_arr)
        self._model_cate.fit(X, phi)
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        return None


class RALearner(_BaseCrossfitEstimator):
    """RA (regression-adjustment) learner with cross-fitting.

    Cross-fits per-arm outcome models, then regresses the RA pseudo-outcome
    T*(Y - mu_0) + (1-T)*(mu_1 - Y) on X.

    Parameters
    ----------
    model_mu : sklearn regressor
        Per-arm outcome model.
    model_cate : sklearn regressor
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, model_mu='auto', model_cate='auto', cv=3,
                 categories='auto', random_state=None):
        self.model_mu = _resolve_continuous_nuisance_model(model_mu)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitRALearner."""
        return _fit_oof_two_stage(
            self,
            lambda: _TLNuisance(self.model_mu),
            lambda: _RAFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitULearner
# ---------------------------------------------------------------------------

class _UFinal:
    """Final stage for CrossFitULearner: regress U pseudo-outcome on X."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        e, mu0, mu1 = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        denom = T_arr - e
        # Avoid division by near-zero
        safe = np.abs(denom) > 1e-6
        phi = np.where(safe,
                       (Y_arr - e * mu1 - (1 - e) * mu0) / np.where(safe, denom, 1.0),
                       0.0)
        self._model_cate.fit(X[safe], phi[safe])
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        return None


class ULearner(_BaseCrossfitEstimator):
    """U-learner with cross-fitting.

    Cross-fits propensity and per-arm outcome models, then regresses the
    U pseudo-outcome (Y - e*mu_1 - (1-e)*mu_0) / (T - e) on X.

    Parameters
    ----------
    propensity_model : sklearn classifier, default LogisticRegression()
    model_mu : sklearn regressor
    model_cate : sklearn regressor
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, propensity_model='auto', model_mu='auto',
                 model_cate='auto', cv=3, categories='auto', random_state=None):
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_mu = _resolve_continuous_nuisance_model(model_mu)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitULearner."""
        return _fit_oof_two_stage(
            self,
            lambda: _MCEANuisance(self.propensity_model, self.model_mu),
            lambda: _UFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitRLearner
# ---------------------------------------------------------------------------

class _RFinal:
    """Final stage for CrossFitRLearner: weighted regression of U pseudo-outcome on X."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        e, mu0, mu1 = nuisances
        Y_arr = np.asarray(Y).ravel()
        T_arr = np.asarray(T).ravel().astype(int)
        denom = T_arr - e
        safe = np.abs(denom) > 1e-6
        phi = np.zeros_like(Y_arr, dtype=float)
        phi[safe] = (Y_arr[safe] - e[safe] * mu1[safe] - (1 - e[safe]) * mu0[safe]) / denom[safe]
        weights = denom[safe] ** 2
        self._model_cate.fit(X[safe], phi[safe], sample_weight=weights)
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        return None


class RLearner(_BaseCrossfitEstimator):
    """R-learner with cross-fitting.

    Cross-fits propensity and per-arm outcome models, constructs the U-style
    pseudo-outcome, and fits a weighted regression on X with weights
    ``(T - e(X))^2``.

    Parameters
    ----------
    propensity_model : sklearn classifier, default LogisticRegression()
    model_mu : sklearn regressor
    model_cate : sklearn regressor supporting ``sample_weight``
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, propensity_model='auto', model_mu='auto',
                 model_cate='auto', cv=3, categories='auto', random_state=None):
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_mu = _resolve_continuous_nuisance_model(model_mu)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitRLearner."""
        return _fit_oof_two_stage(
            self,
            lambda: _MCEANuisance(self.propensity_model, self.model_mu),
            lambda: _RFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True, caching_stage_count=1,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# CrossFitIFLearner
# ---------------------------------------------------------------------------

class _IFFinal:
    """Final stage for CrossFitIFLearner: regress UIF scores on X."""

    def __init__(self, model_cate):
        self._model_cate = clone(model_cate, safe=False)

    def fit(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
            freq_weight=None, sample_var=None, groups=None):
        # Y here IS the UIF score (pre-computed); nuisances = (e,) but unused in final stage
        Y_arr = np.asarray(Y).ravel()
        self._model_cate.fit(X, Y_arr)
        return self

    def predict(self, X=None):
        return self._model_cate.predict(X).reshape(-1, 1)

    def score(self, Y, T, *, X, W=None, Z=None, nuisances, sample_weight=None,
              groups=None):
        Y_arr = np.asarray(Y).ravel()
        pred = self._model_cate.predict(X)
        return np.mean((Y_arr - pred) ** 2)


class _NoopNuisance:
    """No-op nuisance for learners whose pseudo-outcome is already complete."""

    def fit(self, Y, T, *, X=None, W=None, Z=None,
            sample_weight=None, groups=None):
        return self

    def predict(self, Y, T, *, X=None, W=None, Z=None, sample_weight=None, groups=None):
        return ()

    def score(self, Y, T, *, X=None, W=None, Z=None, sample_weight=None, groups=None):
        return ()


class IFLearner(_BaseCrossfitEstimator):
    """IF (Uncentered Influence Function) learner with cross-fitting.

    The input Y is pre-computed UIF scores (output of ``uif_diff_rmst`` or
    ``uif_diff_rmtlj``). Cross-fitting the CATE regression avoids overfitting
    to the already doubly-robust pseudo-outcomes.

    Parameters
    ----------
    propensity_model : sklearn classifier, default LogisticRegression()
        Accepted for API compatibility but not used. The UIF score already
        encodes the treatment contrast.
    model_cate : sklearn regressor
        Final CATE model regressing UIF scores on X.
    cv : int, default 3
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, propensity_model='auto', model_cate='auto',
                 cv=3, categories='auto', random_state=None):
        self.propensity_model = _resolve_binary_nuisance_model(propensity_model)
        self.model_cate = _resolve_continuous_nuisance_model(model_cate)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def fit(self, Y, T=None, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitIFLearner.

        Parameters
        ----------
        Y : array_like, shape (n,)
            Pre-computed UIF scores (``uif_diff_rmst`` or ``uif_diff_rmtlj``).
        T : array_like, shape (n,), optional
            Accepted for API compatibility but ignored. If omitted, a dummy
            treatment vector is supplied internally.
        X : array_like, shape (n, d_x)
            Covariates.
        """
        if T is None:
            T = (np.arange(np.asarray(Y).shape[0]) % 2).astype(int)
        return _fit_oof_two_stage(
            self,
            lambda: _NoopNuisance(),
            lambda: _IFFinal(self.model_cate),
            np.asarray(Y), T, X=X, cache_training_oof=True,
        )

    def const_marginal_effect(self, X=None):
        if _same_train_features(X, getattr(self, '_training_X_oof_', None)):
            return np.asarray(self._training_oof_effect_).reshape(-1, 1)
        return np.asarray(self.model_final_.predict(X)).reshape(-1, 1)

# ---------------------------------------------------------------------------
# CrossFitSurvivalTLearner
# ---------------------------------------------------------------------------

class _SurvivalTLNuisance:
    """Nuisance for CrossFitSurvivalTLearner.

    Fits per-arm survival models on fold train data; predicts RMST values
    out-of-sample on fold test data.
    """

    def __init__(self, models, tau):
        self._models = models
        self._tau = tau

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int)
        X_arr = np.asarray(X)

        if hasattr(self._models, '__iter__') and not hasattr(self._models, 'fit'):
            models_list = list(self._models)
            if len(models_list) != 2:
                raise ValueError("models must be a single estimator or list of exactly 2")
            self._m0 = clone(models_list[0])
            self._m1 = clone(models_list[1])
        else:
            self._m0 = clone(self._models)
            self._m1 = clone(self._models)

        mask0 = T_arr == 0
        mask1 = T_arr == 1
        self._m0.fit(X_arr[mask0], Y[mask0])
        self._m1.fit(X_arr[mask1], Y[mask1])
        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        surv_a0 = self._m0.predict_survival_function(X_arr)
        surv_a1 = self._m1.predict_survival_function(X_arr)
        rmst_a0 = _compute_rmst(surv_a0, self._tau, getattr(self._m0, 'unique_times_', None))
        rmst_a1 = _compute_rmst(surv_a1, self._tau, getattr(self._m1, 'unique_times_', None))
        return rmst_a0, rmst_a1

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class SurvivalTLearner(_DirectNuisanceCateMixin, _BaseCrossfitEstimator):
    """T-learner for survival outcomes with cross-fitting.

    Fits per-arm survival models on held-out folds and predicts CATE directly
    as RMST_a1(X) - RMST_a0(X).

    Parameters
    ----------
    models : scikit-survival estimator or list of two estimators
        Survival model(s) with scikit-survival API (fit / predict_survival_function).
    tau : float
        RMST time horizon.
    cv : int, default 3
        Number of cross-fitting folds.
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, models='auto', tau, cv=3,
                 categories='auto', random_state=None):
        self.models = _resolve_survival_nuisance_model(models)
        self.tau = tau
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def _effect_from_nuisances(self, nuisances):
        rmst_a0, rmst_a1 = nuisances
        return rmst_a1 - rmst_a0

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitSurvivalTLearner.

        Parameters
        ----------
        Y : structured ndarray, dtype [('event', bool), ('time', float)]
            Survival outcomes.
        T : array_like, shape (n,)
            Binary treatment (0/1).
        X : array_like, shape (n, d_x)
            Covariates.
        """
        return _fit_oof_direct(self, lambda: _SurvivalTLNuisance(self.models, self.tau), Y, T, X=X)


# ---------------------------------------------------------------------------
# CrossFitSurvivalSLearner
# ---------------------------------------------------------------------------

class _SurvivalSLNuisance:
    """Nuisance for CrossFitSurvivalSLearner.

    Fits a single pooled survival model with [a, X, a*X] features on fold train;
    predicts RMST for a=0 and a=1 on fold test.
    """

    def __init__(self, overall_model, tau):
        self._overall_model = overall_model
        self._tau = tau

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(float)
        X_arr = np.asarray(X)
        feat = _build_interaction_features(T_arr, X_arr)
        self._m = clone(self._overall_model)
        self._m.fit(feat, Y)
        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        m = X_arr.shape[0]
        feat_a0 = _build_interaction_features(np.zeros(m), X_arr)
        feat_a1 = _build_interaction_features(np.ones(m), X_arr)
        surv_a0 = self._m.predict_survival_function(feat_a0)
        surv_a1 = self._m.predict_survival_function(feat_a1)
        rmst_a0 = _compute_rmst(surv_a0, self._tau, getattr(self._m, 'unique_times_', None))
        rmst_a1 = _compute_rmst(surv_a1, self._tau, getattr(self._m, 'unique_times_', None))
        return rmst_a0, rmst_a1

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class SurvivalSLearner(_DirectNuisanceCateMixin, _BaseCrossfitEstimator):
    """S-learner for survival outcomes with cross-fitting.

    Fits a single pooled survival model with [a, X, a*X] features on held-out
    folds and predicts CATE directly as RMST_a1(X) - RMST_a0(X).

    Parameters
    ----------
    overall_model : scikit-survival estimator
        Pooled survival model with scikit-survival API.
    tau : float
        RMST time horizon.
    cv : int, default 3
        Number of cross-fitting folds.
    categories : 'auto' or list, default 'auto'
    random_state : int or None, default None
    """

    def __init__(self, *, overall_model='auto', tau, cv=3,
                 categories='auto', random_state=None):
        self.overall_model = _resolve_survival_nuisance_model(overall_model)
        self.tau = tau
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def _effect_from_nuisances(self, nuisances):
        rmst_a0, rmst_a1 = nuisances
        return rmst_a1 - rmst_a0

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        """Fit CrossFitSurvivalSLearner.

        Parameters
        ----------
        Y : structured ndarray, dtype [('event', bool), ('time', float)]
            Survival outcomes.
        T : array_like, shape (n,)
            Binary treatment (0/1).
        X : array_like, shape (n, d_x)
            Covariates.
        """
        return _fit_oof_direct(self, lambda: _SurvivalSLNuisance(self.overall_model, self.tau), Y, T, X=X)


# ---------------------------------------------------------------------------
# CrossFitCompetingRisksTLearner
# ---------------------------------------------------------------------------

class _CompetingTLNuisance:
    """Nuisance for CrossFitCompetingRisksTLearner.

    Fits per-arm overall and cause-specific survival models on fold-train data
    and predicts restricted mean time lost (RMTL) for a=0 and a=1 on fold-test data.
    """

    def __init__(self, models, models_cause, tau, cause,
                 compute_separable=False, models_competing=None):
        self._models = models
        self._models_cause = models_cause
        self._tau = tau
        self._cause = cause
        self._compute_separable = compute_separable
        self._models_competing = models_competing

    def _clone_pair(self, estimator):
        if hasattr(estimator, '__iter__') and not hasattr(estimator, 'fit'):
            est_list = list(estimator)
            if len(est_list) != 2:
                raise ValueError("estimator must be a single estimator or list of exactly 2")
            return clone(est_list[0]), clone(est_list[1])
        return clone(estimator), clone(estimator)

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(int)
        X_arr = np.asarray(X)
        event = np.asarray(Y['event'])
        time = np.asarray(Y['time'])

        if np.unique(T_arr).size < 2:
            raise ValueError("Both treatment arms must be present in each fold for CrossFitCompetingRisksTLearner.")

        self._time_grid = np.sort(np.unique(time))

        self._overall0, self._overall1 = self._clone_pair(self._models)
        self._cause0, self._cause1 = self._clone_pair(self._models_cause)

        mask0 = T_arr == 0
        mask1 = T_arr == 1

        self._overall0 = _fit_survival_or_constant(
            self._overall0, X_arr[mask0], time[mask0], event[mask0] != 0)
        self._overall1 = _fit_survival_or_constant(
            self._overall1, X_arr[mask1], time[mask1], event[mask1] != 0)

        self._cause0 = _fit_survival_or_constant(
            self._cause0, X_arr[mask0], time[mask0], event[mask0] == self._cause)
        self._cause1 = _fit_survival_or_constant(
            self._cause1, X_arr[mask1], time[mask1], event[mask1] == self._cause)

        self._competing0 = None
        self._competing1 = None
        if self._compute_separable:
            if self._models_competing is None:
                raise ValueError("models_competing must be provided when compute_separable=True.")
            self._competing0, self._competing1 = self._clone_pair(self._models_competing)
            self._competing0 = _fit_survival_or_constant(
                self._competing0, X_arr[mask0], time[mask0], event[mask0] > self._cause)
            self._competing1 = _fit_survival_or_constant(
                self._competing1, X_arr[mask1], time[mask1], event[mask1] > self._cause)

        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        S_overall_a0 = _eval_survival_on_grid(self._overall0, X_arr, self._time_grid)
        S_overall_a1 = _eval_survival_on_grid(self._overall1, X_arr, self._time_grid)
        S_cause_a0 = _eval_survival_on_grid(self._cause0, X_arr, self._time_grid)
        S_cause_a1 = _eval_survival_on_grid(self._cause1, X_arr, self._time_grid)

        Fj_a0 = _compute_cif_on_grid(S_overall_a0, S_cause_a0)
        Fj_a1 = _compute_cif_on_grid(S_overall_a1, S_cause_a1)

        rmtl_a0 = _compute_rmtl_from_cif(Fj_a0, self._time_grid, self._tau)
        rmtl_a1 = _compute_rmtl_from_cif(Fj_a1, self._time_grid, self._tau)
        return rmtl_a0, rmtl_a1

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class CompetingRisksTLearner(_DirectNuisanceCateMixin, _BaseCrossfitEstimator):
    """T-learner for competing risks outcomes with cross-fitting."""

    def __init__(self, *, models='auto', models_cause='auto', tau, cause=1,
                 cv=3, categories='auto', random_state=None,
                 compute_separable=False, models_competing=None):
        self.models = _resolve_survival_nuisance_model(models)
        self.models_cause = _resolve_survival_nuisance_model(models_cause)
        self.tau = tau
        self.cause = cause
        self.compute_separable = compute_separable
        self.models_competing = _resolve_survival_nuisance_model(models_competing)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def _effect_from_nuisances(self, nuisances):
        rmtl_a0, rmtl_a1 = nuisances
        return rmtl_a1 - rmtl_a0

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        if self.compute_separable and self.models_competing is None:
            raise ValueError("models_competing must be provided when compute_separable=True.")
        result = _fit_oof_direct(
            self,
            lambda: _CompetingTLNuisance(
                self.models,
                self.models_cause,
                self.tau,
                self.cause,
                self.compute_separable,
                self.models_competing,
            ),
            Y, T, X=X,
        )
        self._training_X_separable_oof_ = None if X is None else np.array(X, copy=True)
        self._training_oof_separable_direct_ = None
        self._training_oof_separable_indirect_ = None
        if self.compute_separable:
            T_arr = np.asarray(T).ravel().astype(int)
            X_arr = np.asarray(X)
            event = np.asarray(Y['event'])
            time = np.asarray(Y['time'])
            mask0 = T_arr == 0
            mask1 = T_arr == 1
            self._sep_time_grid = np.sort(np.unique(time))
            self._sep_overall0 = _fit_survival_or_constant(self.models, X_arr[mask0], time[mask0], event[mask0] != 0)
            self._sep_overall1 = _fit_survival_or_constant(self.models, X_arr[mask1], time[mask1], event[mask1] != 0)
            self._sep_cause0 = _fit_survival_or_constant(self.models_cause, X_arr[mask0], time[mask0], event[mask0] == self.cause)
            self._sep_cause1 = _fit_survival_or_constant(self.models_cause, X_arr[mask1], time[mask1], event[mask1] == self.cause)
            self._sep_competing0 = _fit_survival_or_constant(
                self.models_competing, X_arr[mask0], time[mask0], event[mask0] > self.cause)
            self._sep_competing1 = _fit_survival_or_constant(
                self.models_competing, X_arr[mask1], time[mask1], event[mask1] > self.cause)
            if (X is not None) and (not getattr(self, '_skip_training_separable_oof_', False)):
                direct, indirect = _cache_training_oof_separable(self, Y, T, X)
                self._training_oof_separable_direct_ = direct
                self._training_oof_separable_indirect_ = indirect
        return result

    def _compute_separable_effects(self, X):
        if not self.compute_separable:
            raise RuntimeError("Separable effects available only when compute_separable=True.")
        if not hasattr(self, '_sep_time_grid'):
            raise RuntimeError("Call fit() before requesting separable effects.")
        if _same_train_features(X, getattr(self, '_training_X_separable_oof_', None)):
            return (np.asarray(self._training_oof_separable_direct_).ravel(),
                    np.asarray(self._training_oof_separable_indirect_).ravel())
        X_arr = np.asarray(X)
        S_overall_a0 = _eval_survival_on_grid(self._sep_overall0, X_arr, self._sep_time_grid)
        S_overall_a1 = _eval_survival_on_grid(self._sep_overall1, X_arr, self._sep_time_grid)
        S_cause_a0 = _eval_survival_on_grid(self._sep_cause0, X_arr, self._sep_time_grid)
        S_cause_a1 = _eval_survival_on_grid(self._sep_cause1, X_arr, self._sep_time_grid)
        S_comp_a1 = _eval_survival_on_grid(self._sep_competing1, X_arr, self._sep_time_grid)
        Fj_a0 = _compute_cif_on_grid(S_overall_a0, S_cause_a0)
        Fj_a1 = _compute_cif_on_grid(S_overall_a1, S_cause_a1)
        S_cross = np.clip(S_cause_a0 * S_comp_a1, 1e-3, 1.0)
        Fj_cross = _compute_cif_on_grid(S_cross, S_cause_a0)
        rmtl_a0 = _compute_rmtl_from_cif(Fj_a0, self._sep_time_grid, self.tau)
        rmtl_a1 = _compute_rmtl_from_cif(Fj_a1, self._sep_time_grid, self.tau)
        rmtl_cross = _compute_rmtl_from_cif(Fj_cross, self._sep_time_grid, self.tau)
        return rmtl_a1 - rmtl_cross, rmtl_cross - rmtl_a0


# ---------------------------------------------------------------------------
# CrossFitCompetingRisksSLearner
# ---------------------------------------------------------------------------

class _CompetingSLNuisance:
    """Nuisance for CrossFitCompetingRisksSLearner.

    Fits pooled overall and cause-specific survival models with [a, X, a*X]
    features on fold-train data and predicts RMTL for a=0 and a=1 on fold-test data.
    """

    def __init__(self, overall_model, cause_model, tau, cause,
                 compute_separable=False, competing_model=None):
        self._overall_model = overall_model
        self._cause_model = cause_model
        self._tau = tau
        self._cause_id = cause
        self._compute_separable = compute_separable
        self._competing_model = competing_model

    def fit(self, Y, T, *, X, W=None, Z=None,
            sample_weight=None, groups=None):
        T_arr = np.asarray(T).ravel().astype(float)
        X_arr = np.asarray(X)
        event = np.asarray(Y['event'])
        time = np.asarray(Y['time'])

        self._time_grid = np.sort(np.unique(time))

        feat = _build_interaction_features(T_arr, X_arr)

        self._overall = _fit_survival_or_constant(
            self._overall_model, feat, time, event != 0)

        self._cause = _fit_survival_or_constant(
            self._cause_model, feat, time, event == self._cause_id)

        self._competing = None
        if self._compute_separable:
            if self._competing_model is None:
                raise ValueError("competing_model must be provided when compute_separable=True.")
            self._competing = _fit_survival_or_constant(
                self._competing_model, feat, time, event > self._cause_id)

        return self

    def predict(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        X_arr = np.asarray(X)
        m = X_arr.shape[0]

        feat_a0 = _build_interaction_features(np.zeros(m), X_arr)
        feat_a1 = _build_interaction_features(np.ones(m), X_arr)

        S_overall_a0 = _eval_survival_on_grid(self._overall, feat_a0, self._time_grid)
        S_overall_a1 = _eval_survival_on_grid(self._overall, feat_a1, self._time_grid)

        S_cause_a0 = _eval_survival_on_grid(self._cause, feat_a0, self._time_grid)
        S_cause_a1 = _eval_survival_on_grid(self._cause, feat_a1, self._time_grid)

        Fj_a0 = _compute_cif_on_grid(S_overall_a0, S_cause_a0)
        Fj_a1 = _compute_cif_on_grid(S_overall_a1, S_cause_a1)

        rmtl_a0 = _compute_rmtl_from_cif(Fj_a0, self._time_grid, self._tau)
        rmtl_a1 = _compute_rmtl_from_cif(Fj_a1, self._time_grid, self._tau)
        return rmtl_a0, rmtl_a1

    def score(self, Y, T, *, X, W=None, Z=None, sample_weight=None, groups=None):
        return None


class CompetingRisksSLearner(_DirectNuisanceCateMixin, _BaseCrossfitEstimator):
    """S-learner for competing risks outcomes with cross-fitting."""

    def __init__(self, *, overall_model='auto', cause_model='auto', tau, cause=1,
                 cv=3, categories='auto', random_state=None,
                 compute_separable=False, competing_model=None):
        self.overall_model = _resolve_survival_nuisance_model(overall_model)
        self.cause_model = _resolve_survival_nuisance_model(cause_model)
        self.tau = tau
        self.cause = cause
        self.compute_separable = compute_separable
        self.competing_model = _resolve_survival_nuisance_model(competing_model)
        super().__init__(cv=cv, categories=categories, random_state=random_state)

    def _effect_from_nuisances(self, nuisances):
        rmtl_a0, rmtl_a1 = nuisances
        return rmtl_a1 - rmtl_a0

    def fit(self, Y, T, *, X=None, W=None, groups=None,
            cache_values=False, inference=None):
        if self.compute_separable and self.competing_model is None:
            raise ValueError("competing_model must be provided when compute_separable=True.")
        result = _fit_oof_direct(
            self,
            lambda: _CompetingSLNuisance(
                self.overall_model,
                self.cause_model,
                self.tau,
                self.cause,
                self.compute_separable,
                self.competing_model,
            ),
            Y, T, X=X,
        )
        self._training_X_separable_oof_ = None if X is None else np.array(X, copy=True)
        self._training_oof_separable_direct_ = None
        self._training_oof_separable_indirect_ = None
        if self.compute_separable:
            T_arr = np.asarray(T).ravel().astype(float)
            X_arr = np.asarray(X)
            event = np.asarray(Y['event'])
            time = np.asarray(Y['time'])
            feat = _build_interaction_features(T_arr, X_arr)
            self._sep_time_grid = np.sort(np.unique(time))
            self._sep_overall = _fit_survival_or_constant(self.overall_model, feat, time, event != 0)
            self._sep_cause = _fit_survival_or_constant(self.cause_model, feat, time, event == self.cause)
            self._sep_competing = _fit_survival_or_constant(self.competing_model, feat, time, event > self.cause)
            if (X is not None) and (not getattr(self, '_skip_training_separable_oof_', False)):
                direct, indirect = _cache_training_oof_separable(self, Y, T, X)
                self._training_oof_separable_direct_ = direct
                self._training_oof_separable_indirect_ = indirect
        return result

    def _compute_separable_effects(self, X):
        if not self.compute_separable:
            raise RuntimeError("Separable effects available only when compute_separable=True.")
        if not hasattr(self, '_sep_time_grid'):
            raise RuntimeError("Call fit() before requesting separable effects.")
        if _same_train_features(X, getattr(self, '_training_X_separable_oof_', None)):
            return (np.asarray(self._training_oof_separable_direct_).ravel(),
                    np.asarray(self._training_oof_separable_indirect_).ravel())
        X_arr = np.asarray(X)
        m = X_arr.shape[0]
        feat_a0 = _build_interaction_features(np.zeros(m), X_arr)
        feat_a1 = _build_interaction_features(np.ones(m), X_arr)
        S_overall_a0 = _eval_survival_on_grid(self._sep_overall, feat_a0, self._sep_time_grid)
        S_overall_a1 = _eval_survival_on_grid(self._sep_overall, feat_a1, self._sep_time_grid)
        S_cause_a0 = _eval_survival_on_grid(self._sep_cause, feat_a0, self._sep_time_grid)
        S_cause_a1 = _eval_survival_on_grid(self._sep_cause, feat_a1, self._sep_time_grid)
        S_comp_a1 = _eval_survival_on_grid(self._sep_competing, feat_a1, self._sep_time_grid)
        Fj_a0 = _compute_cif_on_grid(S_overall_a0, S_cause_a0)
        Fj_a1 = _compute_cif_on_grid(S_overall_a1, S_cause_a1)
        S_cross = np.clip(S_cause_a0 * S_comp_a1, 1e-3, 1.0)
        Fj_cross = _compute_cif_on_grid(S_cross, S_cause_a0)
        rmtl_a0 = _compute_rmtl_from_cif(Fj_a0, self._sep_time_grid, self.tau)
        rmtl_a1 = _compute_rmtl_from_cif(Fj_a1, self._sep_time_grid, self.tau)
        rmtl_cross = _compute_rmtl_from_cif(Fj_cross, self._sep_time_grid, self.tau)
        return rmtl_a1 - rmtl_cross, rmtl_cross - rmtl_a0


class _SeparableAstar1CateMixin:
    """Expose one separable astar1 estimand as the learner's main effect."""

    _separable_component = None

    def _select_separable_component(self, direct, indirect):
        if self._separable_component == 'direct':
            return direct
        if self._separable_component == 'indirect':
            return indirect
        raise RuntimeError("Unknown separable component.")

    def effect(self, X):
        return np.asarray(self.const_marginal_effect(X)).ravel()

    def const_marginal_effect(self, X=None):
        if X is None:
            raise ValueError("X must be provided for separable-effect learners.")
        X_arr = check_array(X, ensure_2d=True, dtype=float)
        self._check_fitted_dims(X_arr)
        if _same_train_features(X_arr, getattr(self, '_training_X_separable_oof_', None)):
            direct = np.asarray(getattr(self, '_training_oof_separable_direct_', None))
            indirect = np.asarray(getattr(self, '_training_oof_separable_indirect_', None))
        else:
            direct, indirect = self._compute_separable_effects(X_arr)
        selected = self._select_separable_component(direct, indirect)
        return np.asarray(selected, dtype=float).reshape(-1, 1)


class SeparableDirectAstar1TLearner(_SeparableAstar1CateMixin, CompetingRisksTLearner):
    """Direct astar1 separable-effect T-learner for competing risks."""

    _separable_component = 'direct'

    def __init__(self, *, models='auto', models_cause='auto', tau, cause=1,
                 cv=3, categories='auto', random_state=None, models_competing='auto'):
        super().__init__(
            models=models,
            models_cause=models_cause,
            tau=tau,
            cause=cause,
            cv=cv,
            categories=categories,
            random_state=random_state,
            compute_separable=True,
            models_competing=models_competing,
        )


class SeparableIndirectAstar1TLearner(_SeparableAstar1CateMixin, CompetingRisksTLearner):
    """Indirect astar1 separable-effect T-learner for competing risks."""

    _separable_component = 'indirect'

    def __init__(self, *, models='auto', models_cause='auto', tau, cause=1,
                 cv=3, categories='auto', random_state=None, models_competing='auto'):
        super().__init__(
            models=models,
            models_cause=models_cause,
            tau=tau,
            cause=cause,
            cv=cv,
            categories=categories,
            random_state=random_state,
            compute_separable=True,
            models_competing=models_competing,
        )


class SeparableDirectAstar1SLearner(_SeparableAstar1CateMixin, CompetingRisksSLearner):
    """Direct astar1 separable-effect S-learner for competing risks."""

    _separable_component = 'direct'

    def __init__(self, *, overall_model='auto', cause_model='auto', tau, cause=1,
                 cv=3, categories='auto', random_state=None, competing_model='auto'):
        super().__init__(
            overall_model=overall_model,
            cause_model=cause_model,
            tau=tau,
            cause=cause,
            cv=cv,
            categories=categories,
            random_state=random_state,
            compute_separable=True,
            competing_model=competing_model,
        )


class SeparableIndirectAstar1SLearner(_SeparableAstar1CateMixin, CompetingRisksSLearner):
    """Indirect astar1 separable-effect S-learner for competing risks."""

    _separable_component = 'indirect'

    def __init__(self, *, overall_model='auto', cause_model='auto', tau, cause=1,
                 cv=3, categories='auto', random_state=None, competing_model='auto'):
        super().__init__(
            overall_model=overall_model,
            cause_model=cause_model,
            tau=tau,
            cause=cause,
            cv=cv,
            categories=categories,
            random_state=random_state,
            compute_separable=True,
            competing_model=competing_model,
        )
