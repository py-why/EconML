# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""
Censoring Unbiased Transformations (CUT) and Influence Function (UIF) transforms.

These functions transform censored survival/competing-risk data into per-subject
pseudo-outcomes that can be fed into any standard HTE learner.

**RMST transforms** (from ``survival_cut.R``):
    ipcw_cut_rmst, bj_cut_rmst, aipcw_cut_rmst, uif_diff_rmst

**RMTL transforms** (from ``competing_cut.R``):
    ipcw_cut_rmtlj, bj_cut_rmtlj, aipcw_cut_rmtlj,
    aipcw_cut_rmtlj_sep_direct_astar1, aipcw_cut_rmtlj_sep_indirect_astar1,
    uif_diff_rmtlj, uif_diff_rmtlj_sep_direct_astar1, uif_diff_rmtlj_sep_indirect_astar1

All survival/censoring matrices are shape ``(n, ns)`` where ``n`` is the number
of subjects and ``ns`` is the number of time-grid points.  Functions return
per-subject pseudo-outcomes of shape ``(n,)``.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _setup_time_grid(time, tau, time_grid=None, admin_cens=None):
    """Build time grid, interval widths, and tau interpolation index.

    Parameters
    ----------
    time : ndarray (n,)
        Observed event/censoring times.
    tau : float
        Time horizon.
    time_grid : ndarray, float, or None
        - ``ndarray``: pre-built sorted time grid.
        - ``float``: frequency step — builds ``arange(freq, admin_cens, freq)``.
        - ``None``: built from ``sort(unique(time))``.
    admin_cens : float or None
        Administrative censoring time (required when *time_grid* is a float).

    Returns
    -------
    s : ndarray (ns,)
        Sorted time-grid points.
    ds : ndarray (ns,)
        Interval widths ``diff([0, s])``.
    ind : int
        Last index where ``s[ind] < tau``.
    """
    if time_grid is not None:
        if isinstance(time_grid, np.ndarray):
            s = time_grid
        elif np.isscalar(time_grid):
            freq_time = float(time_grid)
            if admin_cens is None:
                raise ValueError("admin_cens is required when time_grid is a frequency")
            s = np.arange(freq_time, admin_cens + freq_time * 0.5, freq_time)
        else:
            s = np.asarray(time_grid, dtype=float)
    else:
        s = np.sort(np.unique(time))
    ds = np.diff(np.concatenate([[0.0], s]))
    ind = int(np.max(np.where(s < tau)[0]))
    return s, ds, ind


def _clamp_survival(S, min_val=1e-3):
    """Clamp survival matrix to ``[min_val, 1]`` and forward-fill NaN along time axis.

    Mirrors R's ``t(na.locf(t(ifelse(S < 1e-3, 1e-3, S))))``.
    """
    S = np.clip(S, min_val, 1.0)
    # Forward-fill NaN along rows (time axis = axis 1)
    mask = np.isnan(S)
    if mask.any():
        for j in range(1, S.shape[1]):
            bad = np.isnan(S[:, j])
            S[bad, j] = S[bad, j - 1]
    return S


def _clamp_product_denom(*terms, min_val=1e-3):
    """Form a denominator as a product of nuisance arrays and floor it at ``min_val``.

    This is stricter than clipping each factor separately and avoids tiny formed
    products such as ``S * G`` or ``Sjbar * S * G`` destabilizing the transforms.
    """
    den = np.ones_like(np.asarray(terms[0], dtype=float), dtype=float)
    for term in terms:
        den = den * np.asarray(term, dtype=float)
    return np.maximum(den, min_val)


def _incremental_hazard(S):
    """Incremental cumulative hazard: ``diff([0, -log(S)])`` per row.

    Mirrors R's ``t(apply(cbind(0, -log(S)), 1, diff))``.

    Returns ndarray (n, ns).
    """
    n, ns = S.shape
    neg_log_S = -np.log(S)
    return np.diff(np.hstack([np.zeros((n, 1)), neg_log_S]), axis=1)


def _compute_cif_matrix(S_overall, dH_cause):
    """Product-limit CIF from overall survival and cause-j hazard increments.

    ``Fj[k] = cumsum_{l=0}^{k} S_lag[l] * dH_j[l]``

    where ``S_lag = hstack([1, S_overall[:, :-1]])``.

    Mirrors R's ``t(apply(cbind(1, S[,1:(ns-1)]) * Fj.dHazard, 1, cumsum))``.

    Returns ndarray (n, ns).
    """
    n, ns = S_overall.shape
    S_lag = np.hstack([np.ones((n, 1)), S_overall[:, :ns - 1]])
    return np.cumsum(S_lag * dH_cause, axis=1)


def _cumulative_integral(values, ds):
    """Row-wise cumulative sum of ``values * ds``.

    Mirrors R's ``t(apply(S * ds_matrix, 1, cumsum))``.

    Returns ndarray (n, ns).
    """
    return np.cumsum(values * ds[np.newaxis, :], axis=1)


def _interpolate_to_tau(matrix, s, tau, ind):
    """Linearly interpolate an ``(n, ns)`` matrix between columns *ind* and *ind+1*
    to the exact time *tau*.

    Mirrors R's ``M[, ind] + (tau - s[ind]) * (M[, ind+1] - M[, ind]) / (s[ind+1] - s[ind])``.

    Returns ndarray (n,).
    """
    frac = (tau - s[ind]) / (s[ind + 1] - s[ind])
    return matrix[:, ind] + frac * (matrix[:, ind + 1] - matrix[:, ind])


def _build_indicators(time, event, s, cause=None):
    """Build counting-process indicator matrices.

    Parameters
    ----------
    time : ndarray (n,)
    event : ndarray (n,)  integer event codes (0 = censored)
    s : ndarray (ns,)  time grid
    cause : int or None
        If given, also build cause-j and competing-cause indicators.

    Returns
    -------
    dict with keys:
        ``Yt``      (n, ns) at-risk indicator  ``time >= s[k]``
        ``dNct``    (n, ns) censoring jump  ``time ≈ s[k] and event == 0``
        ``dNt``     (n, ns) any-event jump  ``time ≈ s[k] and event != 0``
        ``dNjt``    (n, ns) cause-j jump  (only if *cause* given)
        ``dNjbart`` (n, ns) competing-cause jump  (only if *cause* given)
    """
    t_col = time[:, np.newaxis]          # (n, 1)
    s_row = s[np.newaxis, :]             # (1, ns)

    Yt = (t_col >= s_row).astype(float)
    at_time = np.isclose(t_col, s_row, atol=1e-10, rtol=0)  # (n, ns) bool

    dNct = at_time * (event[:, np.newaxis] == 0).astype(float)
    dNt = at_time * (event[:, np.newaxis] != 0).astype(float)

    out = dict(Yt=Yt, dNct=dNct, dNt=dNt)
    if cause is not None:
        out['dNjt'] = at_time * (event[:, np.newaxis] == cause).astype(float)
        out['dNjbart'] = at_time * (event[:, np.newaxis] > cause).astype(float)
    return out


def _ipcw_weight_matrix(time, event, G, s):
    r"""IPCW debiasing weight matrix :math:`\Delta / G`.

    For subject *i* at grid point *k*:

    - If ``time_i > s[k]``:  weight = ``1 / G(s[k])``   (still at risk)
    - If ``time_i <= s[k]`` and ``event_i != 0``:  weight = ``event_i / G(time_i)``
    - If ``time_i <= s[k]`` and ``event_i == 0``:  weight = 0

    Mirrors R lines 17-18 of ``survival_cut.R``.

    Returns ndarray (n, ns).
    """
    n = len(time)
    ns = len(s)
    t_col = time[:, np.newaxis]
    s_row = s[np.newaxis, :]

    # Part 1: (time > s[k]) / G
    still_at_risk = (t_col > s_row).astype(float) / G

    # Part 2: event / G(time_i) * (time <= s[k])
    # G(time_i) = G evaluated at observed time → rowSums(G * indicator_at_time)
    at_time = np.isclose(t_col, s_row, atol=1e-10, rtol=0)
    G_at_obs = np.sum(G * at_time, axis=1)    # (n,)
    G_at_obs = np.maximum(G_at_obs, 1e-3)     # safety clamp
    event_weight = (event / G_at_obs)[:, np.newaxis]  # (n, 1)
    after_event = (t_col <= s_row).astype(float)

    return still_at_risk + event_weight * after_event


def _extract_at_time(matrix, time, s):
    """Extract ``matrix[i, k]`` where ``s[k] ≈ time[i]`` for each row.

    Returns ndarray (n,).
    """
    at_time = np.isclose(time[:, np.newaxis], s[np.newaxis, :], atol=1e-10, rtol=0)
    return np.sum(matrix * at_time, axis=1)


# ===========================================================================
# RMST transformations  (from survival_cut.R)
# ===========================================================================

def ipcw_cut_rmst(a, time, event, tau, G_a0, G_a1,
                  time_grid=None, admin_cens=None):
    """IPCW censoring unbiased transformation for RMST.

    Singly robust — requires correct censoring model G.

    Parameters
    ----------
    a : ndarray (n,)
        Binary treatment indicator (0/1).
    time : ndarray (n,)
        Observed (possibly censored) event times.
    event : ndarray (n,)
        Event indicator (1 = event, 0 = censored).
    tau : float
        RMST truncation horizon.
    G_a0, G_a1 : ndarray (n, ns)
        Censoring survival matrices for arm 0 and arm 1.
    time_grid : ndarray (ns,) or None
        Sorted time grid.  ``None`` → built from ``sort(unique(time))``.
    admin_cens : float or None
        Administrative censoring time (for fixed-frequency grid).

    Returns
    -------
    pseudo_y : ndarray (n,)
        Per-subject pseudo-outcome.
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))

    # IPCW weight × min(time, s[k])
    s_row = s[np.newaxis, :]                              # (1, ns)
    t_col = time[:, np.newaxis]                           # (n, 1)
    min_time_s = np.minimum(t_col, s_row)                 # (n, ns)

    DG_a0 = _ipcw_weight_matrix(time, event, G_a0, s)
    DG_a1 = _ipcw_weight_matrix(time, event, G_a1, s)

    cut_a0 = DG_a0 * min_time_s
    cut_a1 = DG_a1 * min_time_s

    cut_a0 = _interpolate_to_tau(cut_a0, s, tau, ind)
    cut_a1 = _interpolate_to_tau(cut_a1, s, tau, ind)

    return cut_a0 * (1 - a) + cut_a1 * a


def bj_cut_rmst(a, time, event, tau, S_a0, S_a1,
                time_grid=None, admin_cens=None):
    """Buckley-James censoring unbiased transformation for RMST.

    Singly robust — requires correct event survival model S.

    Parameters
    ----------
    a, time, event, tau : see :func:`ipcw_cut_rmst`.
    S_a0, S_a1 : ndarray (n, ns)
        Event survival matrices for arm 0 and arm 1.
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))

    min_time_tau = np.minimum(time, tau)

    results = []
    for S in (S_a0, S_a1):
        # S at tau
        S_tau = _interpolate_to_tau(S, s, tau, ind)

        # S at observed time
        S_Ttilde = _extract_at_time(S, time, s)

        # S at min(tau, time)
        S_min = np.where(time > tau, S_tau, S_Ttilde)

        # RMST cumulative integral
        RMST = _cumulative_integral(S, ds)
        RMST_tau = _interpolate_to_tau(RMST, s, tau, ind)
        RMST_Ttilde = _extract_at_time(RMST, time, s)
        RMST_min = np.where(time > tau, RMST_tau, RMST_Ttilde)

        # BJ formula: min(time, tau) + (1 - event*(time<=tau) - (time>tau)) * (RMST_tau - RMST_min) / S_min
        indicator = 1.0 - event * (time <= tau).astype(float) - (time > tau).astype(float)
        cut = min_time_tau + indicator * (RMST_tau - RMST_min) / np.maximum(S_min, 1e-3)
        results.append(cut)

    return results[0] * (1 - a) + results[1] * a


def aipcw_cut_rmst(a, time, event, tau, G_a0, G_a1, S_a0, S_a1,
                   time_grid=None, admin_cens=None):
    """AIPCW (doubly robust) censoring unbiased transformation for RMST.

    Doubly robust — requires correct censoring model G **or** correct event model S.

    Parameters
    ----------
    a, time, event, tau : see :func:`ipcw_cut_rmst`.
    G_a0, G_a1 : ndarray (n, ns)
        Censoring survival matrices.
    S_a0, S_a1 : ndarray (n, ns)
        Event survival matrices.
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))

    ind_dict = _build_indicators(time, event, s)
    Yt = ind_dict['Yt']
    dNct = ind_dict['dNct']

    s_row = s[np.newaxis, :]
    t_col = time[:, np.newaxis]
    min_time_s = np.minimum(t_col, s_row)

    results = []
    for S, G in ((S_a0, G_a0), (S_a1, G_a1)):
        RMST = _cumulative_integral(S, ds)
        G_dH = _incremental_hazard(G)
        SG = _clamp_product_denom(S, G)

        DG = _ipcw_weight_matrix(time, event, G, s)

        # term1: IPCW
        term1 = DG * min_time_s

        # term3: cumsum of s * (dNc - Y*dH_G) / G
        martingale = (dNct - Yt * G_dH) / G
        term3 = np.cumsum(s_row * martingale, axis=1)

        # term4: RMST * cumsum((dNc - Y*dH_G) / (S*G))
        term4 = RMST * np.cumsum((dNct - Yt * G_dH) / SG, axis=1)

        # term5: cumsum(RMST * (dNc - Y*dH_G) / (S*G))
        term5 = np.cumsum(RMST * (dNct - Yt * G_dH) / SG, axis=1)

        cut = term1 + term3 + term4 - term5
        results.append(_interpolate_to_tau(cut, s, tau, ind))

    return results[0] * (1 - a) + results[1] * a


def uif_diff_rmst(a, time, event, tau, bw, tilt, G_a0, G_a1, S_a0, S_a1,
                  time_grid=None, admin_cens=None):
    """Uncentered influence function transformation for RMST difference.

    Returns treatment-effect pseudo-outcome directly (``uif_a1 - uif_a0``).

    Parameters
    ----------
    a, time, event, tau : see :func:`ipcw_cut_rmst`.
    bw : ndarray (n,)
        Balancing weights.
    tilt : ndarray (n,)
        Tilting weights.
    G_a0, G_a1, S_a0, S_a1 : ndarray (n, ns)
        Censoring and event survival matrices.
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
        UIF difference pseudo-outcome per subject.
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=float).ravel()
    bw = np.asarray(bw, dtype=float).ravel()
    tilt = np.asarray(tilt, dtype=float).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))

    ind_dict = _build_indicators(time, event, s)
    Yt = ind_dict['Yt']
    dNct = ind_dict['dNct']

    s_row = s[np.newaxis, :]
    t_col = time[:, np.newaxis]
    min_time_s = np.minimum(t_col, s_row)

    uif_arms = []
    for arm, S, G in ((0, S_a0, G_a0), (1, S_a1, G_a1)):
        RMST = _cumulative_integral(S, ds)
        G_dH = _incremental_hazard(G)
        SG = _clamp_product_denom(S, G)
        DG = _ipcw_weight_matrix(time, event, G, s)

        bw_arm = bw * (a == arm).astype(float)
        bw_mat = bw_arm[:, np.newaxis]           # (n, 1)
        tilt_mat = tilt[:, np.newaxis]            # (n, 1)

        martingale = (dNct - Yt * G_dH) / G

        term1 = bw_mat * DG * min_time_s
        term2 = tilt_mat * RMST
        term3 = bw_mat * np.cumsum(s_row * martingale, axis=1)
        term4 = bw_mat * RMST * (np.cumsum((dNct - Yt * G_dH) / SG, axis=1) - 1.0)
        term5 = bw_mat * np.cumsum(RMST * (dNct - Yt * G_dH) / SG, axis=1)

        mean_tilt = np.mean(tilt)
        mean_bw_arm = np.mean(bw_arm)
        # Avoid division by zero
        mean_tilt = max(mean_tilt, 1e-10)
        mean_bw_arm = max(mean_bw_arm, 1e-10)

        uif = (1.0 / mean_tilt) * term2 + (1.0 / mean_bw_arm) * (term1 + term3 + term4 - term5)
        uif_arms.append(_interpolate_to_tau(uif, s, tau, ind))

    return uif_arms[1] - uif_arms[0]


# ===========================================================================
# RMTL transformations  (from competing_cut.R)
# ===========================================================================

def ipcw_cut_rmtlj(a, time, event, tau, G_a0, G_a1, cause=1,
                   time_grid=None, admin_cens=None):
    """IPCW censoring unbiased transformation for RMTL (cause j).

    Singly robust — requires correct censoring model G.

    Parameters
    ----------
    a, time, event, tau : see :func:`ipcw_cut_rmst`.
        *event* uses codes 0 = censored, 1 = cause 1, 2 = cause 2, etc.
    G_a0, G_a1 : ndarray (n, ns)
        Censoring survival matrices.
    cause : int, default 1
        Target cause.
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))

    # Observed RMTL contribution: (s[k] - min(time, s[k])) * (event == cause)
    s_row = s[np.newaxis, :]
    t_col = time[:, np.newaxis]
    is_cause = (event == cause).astype(float)[:, np.newaxis]
    RMTLj_obs = (s_row - np.minimum(t_col, s_row)) * is_cause   # (n, ns)

    results = []
    for G in (G_a0, G_a1):
        # G at observed time
        G_at_obs = _extract_at_time(G, time, s)
        G_at_obs = np.maximum(G_at_obs, 1e-3)

        term1 = RMTLj_obs / G_at_obs[:, np.newaxis]
        results.append(_interpolate_to_tau(term1, s, tau, ind))

    return results[0] * (1 - a) + results[1] * a


def bj_cut_rmtlj(a, time, event, tau, S_a0, S_a1, Sj_a0, Sj_a1, cause=1,
                 time_grid=None, admin_cens=None):
    """Buckley-James censoring unbiased transformation for RMTL (cause j).

    Singly robust — requires correct event survival model S and cause-j model Sj.

    Parameters
    ----------
    a, time, event, tau, cause : see :func:`ipcw_cut_rmtlj`.
    S_a0, S_a1 : ndarray (n, ns)
        Overall event survival matrices.
    Sj_a0, Sj_a1 : ndarray (n, ns)
        Cause-j specific survival matrices.
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    Sj_a0 = _clamp_survival(np.array(Sj_a0, dtype=float, copy=True))
    Sj_a1 = _clamp_survival(np.array(Sj_a1, dtype=float, copy=True))

    min_time_tau = np.minimum(time, tau)
    is_cause = (event == cause).astype(float)
    is_any_event = (event != 0).astype(float)

    results = []
    for S, Sj in ((S_a0, Sj_a0), (S_a1, Sj_a1)):
        # Overall survival at tau and observed time
        S_tau = _interpolate_to_tau(S, s, tau, ind)
        S_Ttilde = _extract_at_time(S, time, s)
        S_min = np.where(time > tau, S_tau, S_Ttilde)

        # CIF Fj via product-limit
        Fj_dH = _incremental_hazard(Sj)
        Fj = _compute_cif_matrix(S, Fj_dH)

        Fj_tau = _interpolate_to_tau(Fj, s, tau, ind)
        Fj_Ttilde = _extract_at_time(Fj, time, s)
        Fj_min = np.where(time > tau, Fj_tau, Fj_Ttilde)

        # RMTL = cumulative integral of Fj
        RMTLj = _cumulative_integral(Fj, ds)
        RMTLj_tau = _interpolate_to_tau(RMTLj, s, tau, ind)
        RMTLj_Ttilde = _extract_at_time(RMTLj, time, s)
        RMTLj_min = np.where(time > tau, RMTLj_tau, RMTLj_Ttilde)

        # BJ formula for RMTL
        indicator = 1.0 - is_any_event * (time <= tau).astype(float) - (time > tau).astype(float)
        cut = ((tau - min_time_tau) * is_cause
               + indicator * (RMTLj_tau - RMTLj_min - Fj_min * (tau - min_time_tau))
               / np.maximum(S_min, 1e-3))
        results.append(cut)

    return results[0] * (1 - a) + results[1] * a


def aipcw_cut_rmtlj(a, time, event, tau, G_a0, G_a1, S_a0, S_a1,
                    Sj_a0, Sj_a1, cause=1, time_grid=None, admin_cens=None):
    """AIPCW (doubly robust) censoring unbiased transformation for RMTL.

    Parameters
    ----------
    a, time, event, tau, cause : see :func:`ipcw_cut_rmtlj`.
    G_a0, G_a1, S_a0, S_a1, Sj_a0, Sj_a1 : ndarray (n, ns)
        Censoring, overall event, and cause-j survival matrices.
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))
    Sj_a0 = _clamp_survival(np.array(Sj_a0, dtype=float, copy=True))
    Sj_a1 = _clamp_survival(np.array(Sj_a1, dtype=float, copy=True))

    ind_dict = _build_indicators(time, event, s, cause=cause)
    Yt = ind_dict['Yt']
    dNct = ind_dict['dNct']

    s_row = s[np.newaxis, :]
    t_col = time[:, np.newaxis]
    is_cause = (event == cause).astype(float)[:, np.newaxis]
    RMTLj_obs = (s_row - np.minimum(t_col, s_row)) * is_cause

    results = []
    for S, G, Sj in ((S_a0, G_a0, Sj_a0), (S_a1, G_a1, Sj_a1)):
        G_dH = _incremental_hazard(G)
        Fj_dH = _incremental_hazard(Sj)
        Fj = _compute_cif_matrix(S, Fj_dH)
        RMTLj = _cumulative_integral(Fj, ds)
        SG = _clamp_product_denom(S, G)

        martingale_c = (dNct - Yt * G_dH) / G  # censoring martingale / G

        # G at observed time for IPCW
        G_at_obs = _extract_at_time(G, time, s)
        G_at_obs = np.maximum(G_at_obs, 1e-3)

        # term1: IPCW
        term1 = RMTLj_obs / G_at_obs[:, np.newaxis]

        # term3: cumsum(s * Fj * martingale_c / S) - s * cumsum(Fj * martingale_c / S)
        Fj_mc_over_S = Fj * martingale_c / S
        term3 = (np.cumsum(s_row * Fj_mc_over_S, axis=1)
                 - s_row * np.cumsum(Fj_mc_over_S, axis=1))

        # term4: RMTLj * cumsum(martingale_c / (S * G) ... wait, we already divided by G)
        # R: RMTLj * cumsum((dNc - Y*dH_G) / (S*G))
        mc_over_SG = (dNct - Yt * G_dH) / SG
        term4 = RMTLj * np.cumsum(mc_over_SG, axis=1)

        # term5: cumsum(RMTLj * (dNc - Y*dH_G) / (S*G))
        term5 = np.cumsum(RMTLj * mc_over_SG, axis=1)

        cut = term1 + term3 + term4 - term5
        results.append(_interpolate_to_tau(cut, s, tau, ind))

    return results[0] * (1 - a) + results[1] * a


def aipcw_cut_rmtlj_sep_direct_astar1(a, time, event, tau, G_a0, G_a1,
                                S_a0, S_a1, Sj_a0, Sj_a1,
                                Sjbar_a0, Sjbar_a1, cause=1,
                                time_grid=None, admin_cens=None):
    """AIPCW CUT for separable direct effect of RMTL (a* = 1).

    Parameters
    ----------
    a, time, event, tau, cause : see :func:`ipcw_cut_rmtlj`.
    G_a0, G_a1, S_a0, S_a1, Sj_a0, Sj_a1 : ndarray (n, ns)
    Sjbar_a0, Sjbar_a1 : ndarray (n, ns)
        Competing-cause survival matrices.
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))
    Sj_a0 = _clamp_survival(np.array(Sj_a0, dtype=float, copy=True))
    Sj_a1 = _clamp_survival(np.array(Sj_a1, dtype=float, copy=True))
    Sjbar_a0 = _clamp_survival(np.array(Sjbar_a0, dtype=float, copy=True))
    Sjbar_a1 = _clamp_survival(np.array(Sjbar_a1, dtype=float, copy=True))

    a_col = a[:, np.newaxis]
    S_a = S_a0 * (1 - a_col) + S_a1 * a_col
    G_a = G_a0 * (1 - a_col) + G_a1 * a_col
    Sj_a = Sj_a0 * (1 - a_col) + Sj_a1 * a_col
    Sjbar_a = Sjbar_a0 * (1 - a_col) + Sjbar_a1 * a_col

    ind_dict = _build_indicators(time, event, s, cause=cause)
    Yt = ind_dict['Yt']
    dNt = ind_dict['dNt']
    dNjt = ind_dict['dNjt']
    dNjbart = ind_dict['dNjbart']

    s_row = s[np.newaxis, :]

    # Hazard increments
    Fj_dH_a0 = _incremental_hazard(Sj_a0)
    Fj_dH_a1 = _incremental_hazard(Sj_a1)
    S_dH_a0 = _incremental_hazard(S_a0)
    S_dH_a1 = _incremental_hazard(S_a1)
    Fjbar_dH_a0 = _incremental_hazard(Sjbar_a0)
    Fjbar_dH_a1 = _incremental_hazard(Sjbar_a1)

    # Martingales
    dMjt_a0 = dNjt - Yt * Fj_dH_a0
    dMjt_a1 = dNjt - Yt * Fj_dH_a1
    dMjt_a = dMjt_a0 * (1 - a_col) + dMjt_a1 * a_col

    dMt_a0 = dNt - Yt * S_dH_a0
    dMt_a1 = dNt - Yt * S_dH_a1
    dMt_a = dMt_a0 * (1 - a_col) + dMt_a1 * a_col

    dMjbart_a0 = dNjbart - Yt * Fjbar_dH_a0
    dMjbart_a1 = dNjbart - Yt * Fjbar_dH_a1
    dMjbart_a = dMjbart_a0 * (1 - a_col) + dMjbart_a1 * a_col

    # CIF and RMTL
    Fj_a0 = _compute_cif_matrix(S_a0, Fj_dH_a0)
    Fj_a1 = _compute_cif_matrix(S_a1, Fj_dH_a1)
    Fj_a = Fj_a0 * (1 - a_col) + Fj_a1 * a_col

    # Cross-world CIF: Fj(a_j=0, a_jbar=1)
    S_cross = np.clip(Sj_a0 * Sjbar_a1, 1e-3, 1.0)
    Fj_cross = _compute_cif_matrix(S_cross, Fj_dH_a0)

    RMTLj_a0 = _cumulative_integral(Fj_a0, ds)
    RMTLj_a1 = _cumulative_integral(Fj_a1, ds)
    RMTLj_a = RMTLj_a0 * (1 - a_col) + RMTLj_a1 * a_col
    RMTLj_cross = _cumulative_integral(Fj_cross, ds)

    # 6 terms
    # term1: plug-in
    term1 = RMTLj_cross * (1 - a_col) + RMTLj_a1 * a_col

    # term2
    w2 = Sjbar_a1 * dMjt_a / _clamp_product_denom(Sjbar_a, G_a)
    term2 = np.cumsum(s_row * w2, axis=1) - s_row * np.cumsum(w2, axis=1)

    # term3
    w3 = Sjbar_a1 * dMt_a / _clamp_product_denom(Sjbar_a, S_a, G_a)
    term3 = -(RMTLj_a * np.cumsum(w3, axis=1) - np.cumsum(RMTLj_a * w3, axis=1))

    # term4
    w4 = Sjbar_a1 * Fj_a * dMt_a / _clamp_product_denom(Sjbar_a, S_a, G_a)
    term4 = np.cumsum(s_row * w4, axis=1) - s_row * np.cumsum(w4, axis=1)

    # term5: only for a==0
    is_a0 = (a == 0).astype(float)[:, np.newaxis]
    w5 = dMjbart_a / _clamp_product_denom(S_a, G_a)
    term5 = is_a0 * (RMTLj_cross * np.cumsum(w5, axis=1) - np.cumsum(RMTLj_cross * w5, axis=1))

    # term6: only for a==0
    w6 = Fj_cross * dMjbart_a / _clamp_product_denom(S_a, G_a)
    term6 = -is_a0 * (np.cumsum(s_row * w6, axis=1) - s_row * np.cumsum(w6, axis=1))

    cut = term1 + term2 + term3 + term4 + term5 + term6
    return _interpolate_to_tau(cut, s, tau, ind)


def aipcw_cut_rmtlj_sep_indirect_astar1(a, time, event, tau, G_a0, G_a1,
                                  S_a0, S_a1, Sj_a0, Sj_a1,
                                  Sjbar_a0, Sjbar_a1, cause=1,
                                  time_grid=None, admin_cens=None):
    """AIPCW CUT for separable indirect effect of RMTL (a* = 1).

    Parameters
    ----------
    Same as :func:`aipcw_cut_rmtlj_sep_direct_astar1`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))
    Sj_a0 = _clamp_survival(np.array(Sj_a0, dtype=float, copy=True))
    Sj_a1 = _clamp_survival(np.array(Sj_a1, dtype=float, copy=True))
    Sjbar_a0 = _clamp_survival(np.array(Sjbar_a0, dtype=float, copy=True))
    Sjbar_a1 = _clamp_survival(np.array(Sjbar_a1, dtype=float, copy=True))

    a_col = a[:, np.newaxis]
    S_a = S_a0 * (1 - a_col) + S_a1 * a_col
    G_a = G_a0 * (1 - a_col) + G_a1 * a_col
    Sj_a = Sj_a0 * (1 - a_col) + Sj_a1 * a_col
    Sjbar_a = Sjbar_a0 * (1 - a_col) + Sjbar_a1 * a_col

    ind_dict = _build_indicators(time, event, s, cause=cause)
    Yt = ind_dict['Yt']
    dNt = ind_dict['dNt']
    dNjt = ind_dict['dNjt']
    dNjbart = ind_dict['dNjbart']

    s_row = s[np.newaxis, :]

    Fj_dH_a0 = _incremental_hazard(Sj_a0)
    Fj_dH_a1 = _incremental_hazard(Sj_a1)
    S_dH_a0 = _incremental_hazard(S_a0)
    S_dH_a1 = _incremental_hazard(S_a1)
    Fjbar_dH_a0 = _incremental_hazard(Sjbar_a0)
    Fjbar_dH_a1 = _incremental_hazard(Sjbar_a1)

    dMjt_a0 = dNjt - Yt * Fj_dH_a0
    dMjt_a1 = dNjt - Yt * Fj_dH_a1
    dMjt_a = dMjt_a0 * (1 - a_col) + dMjt_a1 * a_col

    dMt_a0 = dNt - Yt * S_dH_a0
    dMt_a1 = dNt - Yt * S_dH_a1
    dMt_a = dMt_a0 * (1 - a_col) + dMt_a1 * a_col

    dMjbart_a0 = dNjbart - Yt * Fjbar_dH_a0
    dMjbart_a1 = dNjbart - Yt * Fjbar_dH_a1
    dMjbart_a = dMjbart_a0 * (1 - a_col) + dMjbart_a1 * a_col

    Fj_a0 = _compute_cif_matrix(S_a0, Fj_dH_a0)
    Fj_a1 = _compute_cif_matrix(S_a1, Fj_dH_a1)
    Fj_a = Fj_a0 * (1 - a_col) + Fj_a1 * a_col

    S_cross = np.clip(Sj_a0 * Sjbar_a1, 1e-3, 1.0)
    Fj_cross = _compute_cif_matrix(S_cross, Fj_dH_a0)

    RMTLj_a0 = _cumulative_integral(Fj_a0, ds)
    RMTLj_a1 = _cumulative_integral(Fj_a1, ds)
    RMTLj_a = RMTLj_a0 * (1 - a_col) + RMTLj_a1 * a_col
    RMTLj_cross = _cumulative_integral(Fj_cross, ds)

    # 6 terms — indirect effect
    # term1: different from direct
    term1 = RMTLj_a0 * (1 - a_col) + RMTLj_cross * a_col

    # term2: weighted by (1 - Sjbar_a1 / Sjbar_a)
    ratio = 1.0 - Sjbar_a1 / Sjbar_a
    w2 = dMjt_a / G_a * ratio
    term2 = np.cumsum(s_row * w2, axis=1) - s_row * np.cumsum(w2, axis=1)

    # term3
    w3 = dMt_a / _clamp_product_denom(S_a, G_a) * ratio
    term3 = -(RMTLj_a * np.cumsum(w3, axis=1) - np.cumsum(RMTLj_a * w3, axis=1))

    # term4
    w4 = Fj_a * dMt_a / _clamp_product_denom(S_a, G_a) * ratio
    term4 = np.cumsum(s_row * w4, axis=1) - s_row * np.cumsum(w4, axis=1)

    # term5: only for a==0 (sign is negative vs direct)
    is_a0 = (a == 0).astype(float)[:, np.newaxis]
    w5 = dMjbart_a / _clamp_product_denom(S_a, G_a)
    term5 = -is_a0 * (RMTLj_cross * np.cumsum(w5, axis=1) - np.cumsum(RMTLj_cross * w5, axis=1))

    # term6: only for a==0 (sign is positive vs direct)
    w6 = Fj_cross * dMjbart_a / _clamp_product_denom(S_a, G_a)
    term6 = is_a0 * (np.cumsum(s_row * w6, axis=1) - s_row * np.cumsum(w6, axis=1))

    cut = term1 + term2 + term3 + term4 + term5 + term6
    return _interpolate_to_tau(cut, s, tau, ind)


def uif_diff_rmtlj(a, time, event, tau, bw, tilt, G_a0, G_a1,
                   S_a0, S_a1, Sj_a0, Sj_a1, cause=1,
                   time_grid=None, admin_cens=None):
    """UIF transformation for RMTL total effect difference.

    Returns ``uif_a1 - uif_a0`` directly.

    Parameters
    ----------
    a, time, event, tau, cause : see :func:`ipcw_cut_rmtlj`.
    bw : ndarray (n,)
        Balancing weights.
    tilt : ndarray (n,)
        Tilting weights.
    G_a0, G_a1, S_a0, S_a1, Sj_a0, Sj_a1 : ndarray (n, ns)
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()
    bw = np.asarray(bw, dtype=float).ravel()
    tilt = np.asarray(tilt, dtype=float).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))
    Sj_a0 = _clamp_survival(np.array(Sj_a0, dtype=float, copy=True))
    Sj_a1 = _clamp_survival(np.array(Sj_a1, dtype=float, copy=True))

    ind_dict = _build_indicators(time, event, s, cause=cause)
    Yt = ind_dict['Yt']
    dNjt = ind_dict['dNjt']
    dNt = ind_dict['dNt']

    s_row = s[np.newaxis, :]

    uif_arms = []
    for arm, S, G, Sj in ((0, S_a0, G_a0, Sj_a0), (1, S_a1, G_a1, Sj_a1)):
        Fj_dH = _incremental_hazard(Sj)
        S_dH = _incremental_hazard(S)
        Fj = _compute_cif_matrix(S, Fj_dH)
        RMTLj = _cumulative_integral(Fj, ds)
        SG = _clamp_product_denom(S, G)

        bw_arm = bw * (a == arm).astype(float)
        bw_mat = bw_arm[:, np.newaxis]
        tilt_mat = tilt[:, np.newaxis]

        dMjt = dNjt - Yt * Fj_dH
        dMt = dNt - Yt * S_dH

        # term1: cause-j martingale correction
        w1 = dMjt / G
        term1 = bw_mat * (np.cumsum(s_row * w1, axis=1) - s_row * np.cumsum(w1, axis=1))

        # term2: plug-in
        term2 = tilt_mat * RMTLj

        # term3: overall martingale × RMTL
        w3 = dMt / SG
        term3 = bw_mat * RMTLj * np.cumsum(w3, axis=1)

        # term4: cumsum of RMTL × overall martingale
        term4 = bw_mat * np.cumsum(RMTLj * w3, axis=1)

        # term5: CIF × overall martingale correction
        w5 = Fj * dMt / SG
        term5 = bw_mat * (np.cumsum(s_row * w5, axis=1) - s_row * np.cumsum(w5, axis=1))

        uif_val = term1 + term2 - term3 + term4 + term5
        uif_arms.append(_interpolate_to_tau(uif_val, s, tau, ind))

    return uif_arms[1] - uif_arms[0]


def uif_diff_rmtlj_sep_direct_astar1(a, ps, time, event, tau, bw, tilt,
                               G_a0, G_a1, S_a0, S_a1,
                               Sj_a0, Sj_a1, Sjbar_a0, Sjbar_a1,
                               cause=1, time_grid=None, admin_cens=None):
    """UIF transformation for separable direct RMTL effect (a* = 1).

    Parameters
    ----------
    a : ndarray (n,)
    ps : ndarray (n,)
        Propensity scores P(A=1|X).
    time, event, tau, cause : see :func:`ipcw_cut_rmtlj`.
    bw, tilt : ndarray (n,)
    G_a0, G_a1, S_a0, S_a1, Sj_a0, Sj_a1, Sjbar_a0, Sjbar_a1 : ndarray (n, ns)
    time_grid, admin_cens : see :func:`ipcw_cut_rmst`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    ps = np.asarray(ps, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()
    bw = np.asarray(bw, dtype=float).ravel()
    tilt = np.asarray(tilt, dtype=float).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))
    Sj_a0 = _clamp_survival(np.array(Sj_a0, dtype=float, copy=True))
    Sj_a1 = _clamp_survival(np.array(Sj_a1, dtype=float, copy=True))
    Sjbar_a0 = _clamp_survival(np.array(Sjbar_a0, dtype=float, copy=True))
    Sjbar_a1 = _clamp_survival(np.array(Sjbar_a1, dtype=float, copy=True))

    ind_dict = _build_indicators(time, event, s, cause=cause)
    Yt = ind_dict['Yt']
    dNt = ind_dict['dNt']
    dNjt = ind_dict['dNjt']
    dNjbart = ind_dict['dNjbart']

    s_row = s[np.newaxis, :]

    Fj_dH_a0 = _incremental_hazard(Sj_a0)
    Fj_dH_a1 = _incremental_hazard(Sj_a1)
    S_dH_a0 = _incremental_hazard(S_a0)
    S_dH_a1 = _incremental_hazard(S_a1)
    Fjbar_dH_a0 = _incremental_hazard(Sjbar_a0)
    Fjbar_dH_a1 = _incremental_hazard(Sjbar_a1)

    Fj_a0 = _compute_cif_matrix(S_a0, Fj_dH_a0)
    Fj_a1 = _compute_cif_matrix(S_a1, Fj_dH_a1)
    S_cross = np.clip(Sj_a0 * Sjbar_a1, 1e-3, 1.0)
    Fj_cross = _compute_cif_matrix(S_cross, Fj_dH_a0)

    RMTLj_a0 = _cumulative_integral(Fj_a0, ds)
    RMTLj_a1 = _cumulative_integral(Fj_a1, ds)
    RMTLj_cross = _cumulative_integral(Fj_cross, ds)

    bw_a0 = bw * (a == 0).astype(float)
    bw_a1 = bw * (a == 1).astype(float)

    # --- a=0 arm ---
    bw_a0_mat = bw_a0[:, np.newaxis]

    dMjt_a0 = dNjt - Yt * Fj_dH_a0
    dMt_a0 = dNt - Yt * S_dH_a0
    dMjbart_a0 = dNjbart - Yt * Fjbar_dH_a0

    term1_0 = RMTLj_cross

    w2_0 = Sjbar_a1 * dMjt_a0 / _clamp_product_denom(Sjbar_a0, G_a0)
    term2_0 = np.cumsum(s_row * w2_0, axis=1) - s_row * np.cumsum(w2_0, axis=1)

    w3_0 = Sjbar_a1 * dMt_a0 / _clamp_product_denom(Sjbar_a0, S_a0, G_a0)
    term3_0 = -(RMTLj_a0 * np.cumsum(w3_0, axis=1) - np.cumsum(RMTLj_a0 * w3_0, axis=1))

    w4_0 = Sjbar_a1 * Fj_a0 * dMt_a0 / _clamp_product_denom(Sjbar_a0, S_a0, G_a0)
    term4_0 = np.cumsum(s_row * w4_0, axis=1) - s_row * np.cumsum(w4_0, axis=1)

    w5_0 = dMjbart_a0 / _clamp_product_denom(S_a0, G_a0)
    term5_0 = RMTLj_cross * np.cumsum(w5_0, axis=1) - np.cumsum(RMTLj_cross * w5_0, axis=1)

    w6_0 = Fj_cross * dMjbart_a0 / _clamp_product_denom(S_a0, G_a0)
    term6_0 = -(np.cumsum(s_row * w6_0, axis=1) - s_row * np.cumsum(w6_0, axis=1))

    uif_a0 = (term1_0
              + bw_a0_mat * (term2_0 + term3_0 + term4_0)
              + (bw_a0_mat - bw_a1[:, np.newaxis]) * (term5_0 + term6_0))
    uif_a0 = _interpolate_to_tau(uif_a0, s, tau, ind)

    # --- a=1 arm ---
    bw_a1_mat = bw_a1[:, np.newaxis]

    dMjt_a1 = dNjt - Yt * Fj_dH_a1
    dMt_a1 = dNt - Yt * S_dH_a1

    term1_1 = RMTLj_a1

    w2_1 = dMjt_a1 / G_a1
    term2_1 = np.cumsum(s_row * w2_1, axis=1) - s_row * np.cumsum(w2_1, axis=1)

    w3_1 = dMt_a1 / _clamp_product_denom(S_a1, G_a1)
    term3_1 = -(RMTLj_a1 * np.cumsum(w3_1, axis=1) - np.cumsum(RMTLj_a1 * w3_1, axis=1))

    w4_1 = Fj_a1 * dMt_a1 / _clamp_product_denom(S_a1, G_a1)
    term4_1 = np.cumsum(s_row * w4_1, axis=1) - s_row * np.cumsum(w4_1, axis=1)

    uif_a1 = term1_1 + bw_a1_mat * (term2_1 + term3_1 + term4_1)
    uif_a1 = _interpolate_to_tau(uif_a1, s, tau, ind)

    return uif_a1 - uif_a0


def uif_diff_rmtlj_sep_indirect_astar1(a, ps, time, event, tau, bw, tilt,
                                 G_a0, G_a1, S_a0, S_a1,
                                 Sj_a0, Sj_a1, Sjbar_a0, Sjbar_a1,
                                 cause=1, time_grid=None, admin_cens=None):
    """UIF transformation for separable indirect RMTL effect (a* = 1).

    Parameters
    ----------
    Same as :func:`uif_diff_rmtlj_sep_direct_astar1`.

    Returns
    -------
    pseudo_y : ndarray (n,)
    """
    a = np.asarray(a, dtype=float).ravel()
    ps = np.asarray(ps, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=int).ravel()
    bw = np.asarray(bw, dtype=float).ravel()
    tilt = np.asarray(tilt, dtype=float).ravel()

    s, ds, ind = _setup_time_grid(time, tau, time_grid, admin_cens)
    n = len(time)
    ns = len(s)

    S_a0 = _clamp_survival(np.array(S_a0, dtype=float, copy=True))
    S_a1 = _clamp_survival(np.array(S_a1, dtype=float, copy=True))
    G_a0 = _clamp_survival(np.array(G_a0, dtype=float, copy=True))
    G_a1 = _clamp_survival(np.array(G_a1, dtype=float, copy=True))
    Sj_a0 = _clamp_survival(np.array(Sj_a0, dtype=float, copy=True))
    Sj_a1 = _clamp_survival(np.array(Sj_a1, dtype=float, copy=True))
    Sjbar_a0 = _clamp_survival(np.array(Sjbar_a0, dtype=float, copy=True))
    Sjbar_a1 = _clamp_survival(np.array(Sjbar_a1, dtype=float, copy=True))

    ind_dict = _build_indicators(time, event, s, cause=cause)
    Yt = ind_dict['Yt']
    dNt = ind_dict['dNt']
    dNjt = ind_dict['dNjt']
    dNjbart = ind_dict['dNjbart']

    s_row = s[np.newaxis, :]

    Fj_dH_a0 = _incremental_hazard(Sj_a0)
    Fj_dH_a1 = _incremental_hazard(Sj_a1)
    S_dH_a0 = _incremental_hazard(S_a0)
    S_dH_a1 = _incremental_hazard(S_a1)
    Fjbar_dH_a0 = _incremental_hazard(Sjbar_a0)
    Fjbar_dH_a1 = _incremental_hazard(Sjbar_a1)

    Fj_a0 = _compute_cif_matrix(S_a0, Fj_dH_a0)
    Fj_a1 = _compute_cif_matrix(S_a1, Fj_dH_a1)
    S_cross = np.clip(Sj_a0 * Sjbar_a1, 1e-3, 1.0)
    Fj_cross = _compute_cif_matrix(S_cross, Fj_dH_a0)

    RMTLj_a0 = _cumulative_integral(Fj_a0, ds)
    RMTLj_a1 = _cumulative_integral(Fj_a1, ds)
    RMTLj_cross = _cumulative_integral(Fj_cross, ds)

    bw_a0 = bw * (a == 0).astype(float)
    bw_a1 = bw * (a == 1).astype(float)

    # --- a=0 arm ---
    bw_a0_mat = bw_a0[:, np.newaxis]

    dMjt_a0 = dNjt - Yt * Fj_dH_a0
    dMt_a0 = dNt - Yt * S_dH_a0
    dMjbart_a0 = dNjbart - Yt * Fjbar_dH_a0

    term1_0 = RMTLj_a0

    # ratio (1 - Sjbar_a1/Sjbar_a0) for a=0 arm
    ratio_0 = 1.0 - Sjbar_a1 / Sjbar_a0
    w2_0 = dMjt_a0 / G_a0 * ratio_0
    term2_0 = np.cumsum(s_row * w2_0, axis=1) - s_row * np.cumsum(w2_0, axis=1)

    w3_0 = dMt_a0 / _clamp_product_denom(S_a0, G_a0) * ratio_0
    term3_0 = -(RMTLj_a0 * np.cumsum(w3_0, axis=1) - np.cumsum(RMTLj_a0 * w3_0, axis=1))

    w4_0 = Fj_a0 * dMt_a0 / _clamp_product_denom(S_a0, G_a0) * ratio_0
    term4_0 = np.cumsum(s_row * w4_0, axis=1) - s_row * np.cumsum(w4_0, axis=1)

    # term5, term6: sign is opposite from direct
    w5_0 = dMjbart_a0 / _clamp_product_denom(S_a0, G_a0)
    term5_0 = -(RMTLj_cross * np.cumsum(w5_0, axis=1) - np.cumsum(RMTLj_cross * w5_0, axis=1))

    w6_0 = Fj_cross * dMjbart_a0 / _clamp_product_denom(S_a0, G_a0)
    term6_0 = np.cumsum(s_row * w6_0, axis=1) - s_row * np.cumsum(w6_0, axis=1)

    uif_a0 = (term1_0
              + bw_a0_mat * (term2_0 + term3_0 + term4_0)
              + (bw_a0_mat - bw_a1[:, np.newaxis]) * (term5_0 + term6_0))
    uif_a0 = _interpolate_to_tau(uif_a0, s, tau, ind)

    # --- a=1 arm ---
    bw_a1_mat = bw_a1[:, np.newaxis]

    dMjt_a1 = dNjt - Yt * Fj_dH_a1
    dMt_a1 = dNt - Yt * S_dH_a1

    term1_1 = RMTLj_cross

    # ratio (1 - Sjbar_a1/Sjbar_a1) = 0 for a=1 arm
    # So terms 2-4 are zero. Only term1 contributes.
    # But we still need to compute formally for consistency:
    ratio_1 = 1.0 - Sjbar_a1 / Sjbar_a1  # = 0
    w2_1 = dMjt_a1 / G_a1 * ratio_1
    term2_1 = np.cumsum(s_row * w2_1, axis=1) - s_row * np.cumsum(w2_1, axis=1)

    w3_1 = dMt_a1 / _clamp_product_denom(S_a1, G_a1) * ratio_1
    term3_1 = -(RMTLj_a1 * np.cumsum(w3_1, axis=1) - np.cumsum(RMTLj_a1 * w3_1, axis=1))

    w4_1 = Fj_a1 * dMt_a1 / _clamp_product_denom(S_a1, G_a1) * ratio_1
    term4_1 = np.cumsum(s_row * w4_1, axis=1) - s_row * np.cumsum(w4_1, axis=1)

    uif_a1 = term1_1 + bw_a1_mat * (term2_1 + term3_1 + term4_1)
    uif_a1 = _interpolate_to_tau(uif_a1, s, tau, ind)

    return uif_a1 - uif_a0
