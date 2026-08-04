# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""
Nuisance survival model fitting utilities.

Fits per-arm survival models for censoring G(t|a,X), overall event S(t|a,X),
cause-j S_j(t|a,X), and competing-cause S_jbar(t|a,X), then evaluates them
on a common time grid to produce (n, ns) matrices ready for CUT transforms.

This is a standalone utility — it does not depend on any specific learner.
Users can feed the resulting matrices into:

- CUT transforms in ``econml.censor`` (ipcw, bj, aipcw, uif)
- Metalearners in ``econml.metalearners``
- ``CausalSurvivalForest`` in ``econml.grf``
- Any custom analysis
"""

import warnings

import numpy as np
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sksurv.ensemble import RandomSurvivalForest


_PROPENSITY_CLIP = 1e-3
_DEFAULT_NUISANCE_RANDOM_STATE = 123


def _make_default_propensity_model():
    return RandomForestClassifier(
        n_estimators=50,
        min_samples_leaf=5,
        random_state=_DEFAULT_NUISANCE_RANDOM_STATE,
    )


def _make_default_survival_model():
    return RandomSurvivalForest(
        n_estimators=50,
        min_samples_leaf=5,
        random_state=_DEFAULT_NUISANCE_RANDOM_STATE,
    )


def _resolve_propensity_model(model):
    return _make_default_propensity_model() if model == 'auto' else model


def _resolve_survival_model(model):
    return _make_default_survival_model() if model == 'auto' else model


class NuisanceResult:
    """Container for fitted nuisance survival matrices.

    Attributes
    ----------
    time_grid : ndarray (ns,)
        Sorted time grid on which all matrices are evaluated.
    G_a0, G_a1 : ndarray (n, ns) or None
        Censoring survival P(C > t | a, X).
    S_a0, S_a1 : ndarray (n, ns) or None
        Overall event survival P(T > t | a, X).
    Sj_a0, Sj_a1 : ndarray (n, ns) or None
        Cause-j subdistribution survival.
    Sjbar_a0, Sjbar_a1 : ndarray (n, ns) or None
        Competing-cause subdistribution survival.
    """

    def __init__(self, time_grid, **matrices):
        self.time_grid = time_grid
        self.G_a0 = matrices.get('G_a0')
        self.G_a1 = matrices.get('G_a1')
        self.S_a0 = matrices.get('S_a0')
        self.S_a1 = matrices.get('S_a1')
        self.Sj_a0 = matrices.get('Sj_a0')
        self.Sj_a1 = matrices.get('Sj_a1')
        self.Sjbar_a0 = matrices.get('Sjbar_a0')
        self.Sjbar_a1 = matrices.get('Sjbar_a1')


def _make_sksurv_y(time, event_bool):
    """Build scikit-survival structured array with dtype [('event', bool), ('time', float)]."""
    return np.array(
        [(bool(e), float(t)) for e, t in zip(event_bool, time)],
        dtype=[('event', bool), ('time', float)]
    )


def _eval_on_grid(model, X, time_grid):
    """Evaluate a fitted scikit-survival model's survival function on a time grid.

    Parameters
    ----------
    model : fitted sksurv estimator
        Must implement ``predict_survival_function(X)``.
    X : ndarray (m, d)
    time_grid : ndarray (ns,)

    Returns
    -------
    S_mat : ndarray (m, ns)
        ``S_mat[i, k] = S(time_grid[k] | X[i])``, clamped to [1e-3, 1].
    """
    surv_fns = model.predict_survival_function(X)
    m = len(surv_fns)
    ns = len(time_grid)
    S_mat = np.empty((m, ns))
    for i, fn in enumerate(surv_fns):
        grid = np.asarray(time_grid, dtype=float)
        x = np.asarray(fn.x, dtype=float)
        y = np.asarray(fn(x), dtype=float)
        S_mat[i] = np.interp(grid, x, y, left=1.0, right=y[-1])
    return np.clip(S_mat, 1e-3, 1.0)


def _fit_and_eval_per_arm(model_template, X, time, event_bool, T_arr, time_grid, X_pred=None):
    """Clone, fit per arm, and evaluate on a held-out prediction sample.

    Parameters
    ----------
    model_template : sksurv estimator (unfitted)
    X : ndarray (n, d) — training covariates
    time : ndarray (n,) — observed times
    event_bool : ndarray (n,) bool — event indicator for this model type
    T_arr : ndarray (n,) int — treatment assignment (0/1)
    time_grid : ndarray (ns,) — evaluation grid
    X_pred : ndarray (m, d)
        Held-out prediction covariates. In-sample nuisance predictions are
        intentionally unsupported in this project.

    Returns
    -------
    mat_a0 : ndarray (m, ns)
    mat_a1 : ndarray (m, ns)
    """
    if X_pred is None:
        raise ValueError("_fit_and_eval_per_arm is OOS-only and requires X_pred")

    results = []
    for arm in (0, 1):
        mask = T_arr == arm
        if mask.sum() < 2:
            # Not enough data for this arm — return ones (no information)
            results.append(np.ones((X_pred.shape[0], len(time_grid))))
            continue
        if not np.any(event_bool[mask]):
            # No observed events for this nuisance within the fold/arm slice
            results.append(np.ones((X_pred.shape[0], len(time_grid))))
            continue
        m = clone(model_template)
        y_arm = _make_sksurv_y(time[mask], event_bool[mask])
        try:
            m.fit(X[mask], y_arm)
            results.append(_eval_on_grid(m, X_pred, time_grid))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            warnings.warn(
                "Nuisance survival model fit failed on a fold/arm slice; "
                "falling back to a constant survival curve. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            results.append(np.ones((X_pred.shape[0], len(time_grid))))

    return results[0], results[1]


def fit_nuisance_survival(time, event, T, X, *,
                          model_censoring='auto',
                          model_event='auto',
                          model_cause='auto',
                          model_competing='auto',
                          propensity_model='auto',
                          cause=1,
                          time_grid=None,
                          X_pred=None,
                          cv=2,
                          random_state=None):
    """Cross-fit per-arm survival nuisance models and evaluate on a common time grid.

    This project uses out-of-sample nuisance predictions only. Accordingly,
    ``fit_nuisance_survival`` is now a thin compatibility wrapper around
    :func:`fit_nuisance_survival_crossfit` and returns OOF nuisance matrices
    on the training sample.

    Only models that are provided (not None) are fitted. For example, IPCW
    only needs ``model_censoring``; AIPCW needs both ``model_censoring`` and
    ``model_event``; competing-risk transforms also need ``model_cause``.

    Parameters
    ----------
    time : ndarray (n,)
        Observed (possibly censored) event times.
    event : ndarray (n,)
        Event indicator. 0 = censored, 1 = event (survival) or cause codes
        (competing risks: 1 = cause j, 2 = competing cause, etc.).
    T : ndarray (n,)
        Binary treatment indicator (0/1).
    X : ndarray (n, d)
        Covariates.
    model_censoring : scikit-survival estimator or None
        Template for censoring survival G(t|a,X) = P(C > t | a, X).
        Must implement ``fit(X, y)`` and ``predict_survival_function(X)``.
    model_event : scikit-survival estimator or None
        Template for overall event survival S(t|a,X) = P(T > t | a, X).
    model_cause : scikit-survival estimator or None
        Template for cause-j subdistribution survival S_j(t|a,X).
    model_competing : scikit-survival estimator or None
        Template for competing-cause survival S_jbar(t|a,X).
    cause : int, default 1
        Target cause code (for ``model_cause`` and ``model_competing``).
    time_grid : ndarray (ns,) or None
        Sorted time points for evaluation. ``None`` → ``sort(unique(time))``.
    X_pred : ndarray (m, d) or None
        Unsupported in the OOS-only project configuration. Predictions are
        always returned for the training rows using held-out folds.
    cv : int or splitter, default 2
        Cross-fitting specification used to generate OOF nuisance predictions.
    random_state : int or None, default None
        Random seed used when ``cv`` is an integer.

    Returns
    -------
    result : NuisanceResult
        Container with OOF nuisance matrices on the training sample (those not
        requested are ``None``).

    Examples
    --------
    >>> from sksurv.linear_model import CoxPHSurvivalAnalysis
    >>> from econml.censor import fit_nuisance_survival, aipcw_cut_rmst
    >>> result = fit_nuisance_survival(
    ...     time, event, T, X,
    ...     model_censoring=CoxPHSurvivalAnalysis(),
    ...     model_event=CoxPHSurvivalAnalysis(),
    ...     cv=2)
    >>> Y_star = aipcw_cut_rmst(
    ...     T, time, event, tau=4.0,
    ...     result.G_a0, result.G_a1,
    ...     result.S_a0, result.S_a1,
    ...     time_grid=result.time_grid)
    """
    if X_pred is not None:
        raise ValueError(
            "fit_nuisance_survival is OOS-only in this project and does not "
            "support X_pred. Use the returned training-row OOF nuisances.")

    result = fit_nuisance_survival_crossfit(
        time, event, T, X,
        model_censoring=model_censoring,
        model_event=model_event,
        model_cause=model_cause,
        model_competing=model_competing,
        propensity_model=propensity_model,
        cause=cause,
        time_grid=time_grid,
        cv=cv,
        random_state=random_state,
    )
    return CrossFitNuisanceResult(
        result.time_grid,
        ps=result.ps,
        iptw=result.iptw,
        naive=result.naive,
        ow=result.ow,
        ow_tilt_hat=result.ow_tilt_hat,
        G_a0=result.G_a0,
        G_a1=result.G_a1,
        S_a0=result.S_a0,
        S_a1=result.S_a1,
        Sj_a0=result.Sj_a0,
        Sj_a1=result.Sj_a1,
        Sjbar_a0=result.Sjbar_a0,
        Sjbar_a1=result.Sjbar_a1,
    )


class CrossFitNuisanceResult(NuisanceResult):
    """Container for cross-fitted nuisance matrices and propensity-derived weights.

    Attributes
    ----------
    ps : ndarray (n,) or None
        Out-of-fold propensity scores P(T=1 | X).
    iptw : ndarray (n,) or None
        ATE balancing weights ``1 / (ps * a + (1 - ps) * (1 - a))``.
    naive : ndarray (n,) or None
        Constant tilt weights for the ATE target.
    ow : ndarray (n,) or None
        Overlap balancing weights ``ps`` for controls and ``1 - ps`` for treated.
    ow_tilt_hat : ndarray (n,) or None
        Overlap tilt weights ``ps * (1 - ps)``.
    """

    def __init__(self, time_grid, *, ps=None, iptw=None, naive=None,
                 ow=None, ow_tilt_hat=None, **matrices):
        super().__init__(time_grid, **matrices)
        self.ps = ps
        self.iptw = iptw
        self.naive = naive
        self.ow = ow
        self.ow_tilt_hat = ow_tilt_hat


def _build_crossfit_folds(X, T_arr, event, cv, random_state):
    """Create folds for nuisance cross-fitting, favoring treatment/event stratification."""
    from sklearn.model_selection import KFold, StratifiedKFold

    if hasattr(cv, "split"):
        return list(cv.split(X, T_arr))

    n_splits = int(cv)
    if n_splits < 2:
        raise ValueError("cv must be at least 2 for cross-fitted nuisance estimation")

    event_arr = np.asarray(event).astype(int).ravel()
    strata_joint = np.asarray([f"{t}_{e}" for t, e in zip(T_arr, event_arr)])
    _, joint_counts = np.unique(strata_joint, return_counts=True)
    if joint_counts.size > 0 and np.min(joint_counts) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(X, strata_joint))

    _, treatment_counts = np.unique(T_arr, return_counts=True)
    if treatment_counts.size > 0 and np.min(treatment_counts) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(X, T_arr))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(X))


def _fit_and_eval_propensity(propensity_model, X_train, T_train, X_test):
    """Fit a propensity model on a training fold and predict on its held-out fold."""
    m = clone(propensity_model)
    m.fit(X_train, T_train)
    if hasattr(m, "predict_proba"):
        ps = m.predict_proba(X_test)[:, 1]
    else:
        ps = m.predict(X_test)
    return np.clip(np.asarray(ps, dtype=float).ravel(), _PROPENSITY_CLIP, 1 - _PROPENSITY_CLIP)


def _sanitize_survival_matrix(mat):
    """Clamp fold-wise survival predictions to a finite probability range."""
    arr = np.asarray(mat, dtype=float)
    arr = np.nan_to_num(arr, nan=1.0, posinf=1.0, neginf=1e-3)
    return np.clip(arr, 1e-3, 1.0)


def fit_nuisance_survival_crossfit(time, event, T, X, *,
                                   model_censoring='auto',
                                   model_event='auto',
                                   model_cause='auto',
                                   model_competing='auto',
                                   propensity_model='auto',
                                   cause=1,
                                   time_grid=None,
                                   cv=3,
                                   random_state=None):
    """Cross-fit survival nuisance models and return out-of-fold predictions.

    This utility mirrors :func:`fit_nuisance_survival`, but each nuisance model
    is trained on fold complements and evaluated only on held-out observations.
    The resulting matrices are ready to plug into CUT and UIF transformations.

    Parameters
    ----------
    time, event, T, X : see :func:`fit_nuisance_survival`
    model_censoring, model_event, model_cause, model_competing : see
        :func:`fit_nuisance_survival`
    propensity_model : sklearn classifier or None, default None
        If provided, produces out-of-fold propensity scores and the associated
        ATE / overlap balancing weights used by UIF transforms.
    cause : int, default 1
        Target cause code for competing-risk nuisances.
    time_grid : ndarray (ns,) or None
        Common evaluation grid. ``None`` uses ``sort(unique(time))``.
    cv : int or splitter, default 3
        Cross-fitting specification. Integers use shuffled folds.
    random_state : int or None, default None
        Random seed used when ``cv`` is an integer.

    Returns
    -------
    result : CrossFitNuisanceResult
        Out-of-fold nuisance matrices, optional propensity scores, and weight
        vectors (`iptw`, `naive`, `ow`, `ow_tilt_hat`).
    """
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()
    T_arr = np.asarray(T, dtype=int).ravel()
    X = np.asarray(X, dtype=float)

    if time_grid is None:
        time_grid = np.sort(np.unique(time))

    model_censoring = _resolve_survival_model(model_censoring)
    model_event = _resolve_survival_model(model_event)
    model_cause = _resolve_survival_model(model_cause)
    model_competing = _resolve_survival_model(model_competing)
    propensity_model = _resolve_propensity_model(propensity_model)

    n = len(time)
    ns = len(time_grid)

    matrices = {}
    for name, model in (
        ("G_a0", model_censoring),
        ("G_a1", model_censoring),
        ("S_a0", model_event),
        ("S_a1", model_event),
        ("Sj_a0", model_cause),
        ("Sj_a1", model_cause),
        ("Sjbar_a0", model_competing),
        ("Sjbar_a1", model_competing),
    ):
        matrices[name] = np.full((n, ns), np.nan) if model is not None else None

    ps = np.full(n, np.nan) if propensity_model is not None else None
    folds = _build_crossfit_folds(X, T_arr, event, cv, random_state)

    for train_idx, test_idx in folds:
        X_train = X[train_idx]
        X_test = X[test_idx]
        time_train = time[train_idx]
        event_train = event[train_idx]
        T_train = T_arr[train_idx]

        if model_censoring is not None:
            g0, g1 = _fit_and_eval_per_arm(
                model_censoring, X_train, time_train, event_train == 0,
                T_train, time_grid, X_pred=X_test)
            matrices["G_a0"][test_idx] = _sanitize_survival_matrix(g0)
            matrices["G_a1"][test_idx] = _sanitize_survival_matrix(g1)

        if model_event is not None:
            s0, s1 = _fit_and_eval_per_arm(
                model_event, X_train, time_train, event_train != 0,
                T_train, time_grid, X_pred=X_test)
            matrices["S_a0"][test_idx] = _sanitize_survival_matrix(s0)
            matrices["S_a1"][test_idx] = _sanitize_survival_matrix(s1)

        if model_cause is not None:
            sj0, sj1 = _fit_and_eval_per_arm(
                model_cause, X_train, time_train, event_train == cause,
                T_train, time_grid, X_pred=X_test)
            matrices["Sj_a0"][test_idx] = _sanitize_survival_matrix(sj0)
            matrices["Sj_a1"][test_idx] = _sanitize_survival_matrix(sj1)

        if model_competing is not None:
            sjbar0, sjbar1 = _fit_and_eval_per_arm(
                model_competing, X_train, time_train, event_train > cause,
                T_train, time_grid, X_pred=X_test)
            matrices["Sjbar_a0"][test_idx] = _sanitize_survival_matrix(sjbar0)
            matrices["Sjbar_a1"][test_idx] = _sanitize_survival_matrix(sjbar1)

        if propensity_model is not None:
            ps[test_idx] = _fit_and_eval_propensity(propensity_model, X_train, T_train, X_test)

    for name, value in matrices.items():
        if value is not None and np.isnan(value).any():
            raise RuntimeError(f"Cross-fitted nuisance matrix {name} contains unfilled entries")

    iptw = None
    naive = None
    ow = None
    ow_tilt_hat = None
    if ps is not None:
        if np.isnan(ps).any():
            raise RuntimeError("Cross-fitted propensity scores contain unfilled entries")
        iptw = 1.0 / (ps * T_arr + (1.0 - ps) * (1.0 - T_arr))
        naive = np.ones_like(ps)
        ow = np.where(T_arr == 1, 1.0 - ps, ps)
        ow_tilt_hat = ps * (1.0 - ps)

    return CrossFitNuisanceResult(
        time_grid,
        ps=ps,
        iptw=iptw,
        naive=naive,
        ow=ow,
        ow_tilt_hat=ow_tilt_hat,
        **matrices,
    )


def fit_nuisance_competing_crossfit(time, event, T, X, *,
                                    model_censoring='auto',
                                    model_event='auto',
                                    model_cause='auto',
                                    model_competing='auto',
                                    propensity_model='auto',
                                    cause=1,
                                    time_grid=None,
                                    cv=3,
                                    random_state=None):
    """Cross-fit nuisance functions for competing-risk CUT and UIF transforms."""
    return fit_nuisance_survival_crossfit(
        time, event, T, X,
        model_censoring=model_censoring,
        model_event=model_event,
        model_cause=model_cause,
        model_competing=model_competing,
        propensity_model=propensity_model,
        cause=cause,
        time_grid=time_grid,
        cv=cv,
        random_state=random_state,
    )
