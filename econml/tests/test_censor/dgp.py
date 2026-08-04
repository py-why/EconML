"""Survival DGP ported from original/simulation/survival/survival_hte_simulation_case1.R.

Generates a single-failure survival dataset with:
- 6 covariates: x1,x2,x3 ~ MVN(0, Sigma), x4,x5,x6 ~ Bernoulli(0.5)
- Binary treatment with logistic propensity score
- Counterfactual T(a=0) ~ log-logistic; T(a=1) ~ log-normal
- Dependent censoring (lognormal for a=0, log-logistic for a=1)
- Administrative censoring at admin_cens
- True CATE = E[min(T(1),tau)] - E[min(T(0),tau)]  (Monte-Carlo over 1000 reps)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Parametric survival time generators
# (mirror simsurv's loglogistic / lognormal AFT parameterisation)
# ---------------------------------------------------------------------------

def _loglogistic_times(x0, covariates, betas, gamma, rng):
    """Log-logistic AFT survival times.

    T = exp(eta) * W,  W ~ Logistic(0,1) re-parameterised so that
    the scale is exp(eta) and the shape is 1/gamma.

    Parameters
    ----------
    x0 : float
        Intercept value (constant column added to covariates).
    covariates : ndarray, shape (n, p)
    betas : dict  key order matches [x0, x1, ..., xp]
    gamma : float  shape (> 0)
    rng : np.random.Generator

    Returns
    -------
    times : ndarray, shape (n,)
    """
    n = covariates.shape[0]
    beta_vals = np.array(list(betas.values()))          # [x0, x1, ...]
    X_aug = np.column_stack([np.full(n, x0), covariates])
    eta = X_aug @ beta_vals                              # linear predictor

    # Log-logistic: log T = eta + gamma * log(U/(1-U)),  U ~ Uniform(0,1)
    u = rng.uniform(0, 1, size=n)
    log_t = eta + gamma * np.log(u / (1.0 - u))
    return np.exp(log_t)


def _lognormal_times(x0, covariates, betas, sigma, rng):
    """Log-normal AFT survival times.

    T = exp(eta + sigma * Z),  Z ~ N(0,1).
    """
    n = covariates.shape[0]
    beta_vals = np.array(list(betas.values()))
    X_aug = np.column_stack([np.full(n, x0), covariates])
    eta = X_aug @ beta_vals
    z = rng.standard_normal(size=n)
    return np.exp(eta + sigma * z)


# ---------------------------------------------------------------------------
# True RMST by Monte-Carlo integration (mirror of R's 1000-rep approach)
# ---------------------------------------------------------------------------

def _true_rmst_mc(covariates, betas_a0, gamma_a0, betas_a1, sigma_a1, tau,
                  n_mc=1000, seed=0):
    """Compute E[min(T(a),tau)] for a=0 and a=1 by Monte-Carlo.

    Returns
    -------
    rmst_a0 : ndarray (n,)
    rmst_a1 : ndarray (n,)
    """
    n = covariates.shape[0]
    rng = np.random.default_rng(seed)

    sum_a0 = np.zeros(n)
    sum_a1 = np.zeros(n)
    for _ in range(n_mc):
        ta0 = _loglogistic_times(1, covariates, betas_a0, gamma_a0, rng)
        ta1 = _lognormal_times(1, covariates, betas_a1, sigma_a1, rng)
        sum_a0 += np.minimum(ta0, tau)
        sum_a1 += np.minimum(ta1, tau)

    return sum_a0 / n_mc, sum_a1 / n_mc


# ---------------------------------------------------------------------------
# Main DGP
# ---------------------------------------------------------------------------

def make_survival_data(n=400, admin_cens=10.0, time_grid=0.01, tau=2.0,
                       seed=0, compute_true_cate=True):
    """Generate a survival dataset matching case1 from the R simulation.

    Parameters
    ----------
    n : int
        Sample size.
    admin_cens : float
        Administrative censoring time.
    time_grid : float
        Rounding grid for observed times (mirrors R's ceiling(…/grid)*grid).
    tau : float
        RMST truncation horizon.
    seed : int
        Random seed.
    compute_true_cate : bool
        If True, attach ``true_cate`` (Monte-Carlo RMST differences, 1000 reps).
        Set False for speed when not needed.

    Returns
    -------
    data : dict with keys
        X         : ndarray (n, 6)  covariates
        T         : ndarray (n,)    binary treatment  0/1
        time      : ndarray (n,)    observed (possibly censored) time
        event     : ndarray (n,)    event indicator  1=event, 0=censored
        Y         : structured ndarray  dtype [('event',bool),('time',float)]
        ps        : ndarray (n,)    true propensity score
        true_cate : ndarray (n,) or None
    """
    rng = np.random.default_rng(seed)

    # --- covariates ---
    corr = np.array([[1.0, 0.5, 0.5],
                     [0.5, 1.0, 0.5],
                     [0.5, 0.5, 1.0]])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, 3))
    x_cont = z @ L.T                           # x1, x2, x3
    x_bin = rng.binomial(1, 0.5, size=(n, 3))  # x4, x5, x6
    X = np.column_stack([x_cont, x_bin])
    x1, x2, x3, x4, x5, x6 = X.T

    # --- propensity & treatment ---
    logit_ps = 0.3 + 0.2*x1 + 0.3*x2 + 0.3*x3 - 0.2*x4 - 0.3*x5 - 0.2*x6
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    A = rng.binomial(1, ps).astype(float)

    # --- counterfactual event times ---
    cov6 = X  # (n, 6)

    betas_a0 = dict(x0=0.8, x1=-0.8, x2=1.0, x3=0.8, x4=0.4, x5=-0.4, x6=0.8)
    betas_a1 = dict(x0=0.4, x1=0.6,  x2=-0.8, x3=1.2, x4=0.6, x5=-0.3, x6=0.5)

    Ta0 = _loglogistic_times(1, cov6, betas_a0, gamma=0.2, rng=rng)
    Ta1 = _lognormal_times(1, cov6, betas_a1, sigma=1.0, rng=rng)

    # round up to time_grid (mirror ceiling(…/grid)*grid in R)
    Ta0 = np.ceil(Ta0 / time_grid) * time_grid
    Ta1 = np.ceil(Ta1 / time_grid) * time_grid
    Ta = (1 - A) * Ta0 + A * Ta1

    # --- censoring (arm-dependent) ---
    betas_c0 = dict(x0=1.8, x1=0.6, x2=-0.8, x3=0.5, x4=0.7, x5=-0.4, x6=-0.2)
    betas_c1 = dict(x0=2.2, x1=0.6, x2=-0.8, x3=0.5, x4=0.7, x5=0.8,  x6=1.2)

    C = np.empty(n)
    mask0 = A == 0
    mask1 = A == 1

    C[mask0] = _lognormal_times(1, cov6[mask0], betas_c0, sigma=0.8, rng=rng)
    C[mask1] = _loglogistic_times(1, cov6[mask1], betas_c1, gamma=0.8, rng=rng)

    C = np.minimum(C, admin_cens)

    # --- observed time & event ---
    time = np.ceil(np.minimum(C, Ta) / time_grid) * time_grid
    event = (Ta <= C).astype(int)

    Y = np.array([(bool(e), float(t)) for e, t in zip(event, time)],
                 dtype=[('event', bool), ('time', float)])

    # --- true CATE via Monte-Carlo ---
    true_cate = None
    if compute_true_cate:
        rmst_a0, rmst_a1 = _true_rmst_mc(cov6, betas_a0, 0.2, betas_a1, 1.0,
                                          tau, n_mc=1000, seed=seed + 1)
        true_cate = rmst_a1 - rmst_a0

    return dict(X=X, T=A, time=time, event=event, Y=Y, ps=ps,
                true_cate=true_cate)


# ---------------------------------------------------------------------------
# Competing risks helpers
# ---------------------------------------------------------------------------

def _exponential_times(x0, covariates, lambdas, betas, rng):
    """Exponential survival times with covariate-dependent rate.

    T = -log(U) / (lambda * exp(X @ beta)),  U ~ Uniform(0,1).

    Mirrors R's ``exponenetial()`` function (line 13 of competing_hte_simulation_helper.R).
    """
    n = covariates.shape[0]
    beta_vals = np.array(list(betas.values()))
    X_aug = np.column_stack([np.full(n, x0), covariates])
    log_rate = X_aug @ beta_vals          # log(lambda * exp(X@beta)) = log(lambda) + X@beta
    u = rng.uniform(0, 1, size=n)
    return -np.log(u) / (lambdas * np.exp(log_rate))


def _true_rmtl_mc(covariates, tj1a0_params, tj2a0_params, tj1a1_params, tj2a1_params,
                  tau, n_mc=1000, seed=0):
    """Monte-Carlo RMTL for competing risks (mirrors R lines 89-128 of case1.R).

    Computes E[(tau - min(T,tau)) * 1(J=1)] under four counterfactual scenarios:
      (aj=0, ajbar=0), (aj=0, ajbar=1), (aj=1, ajbar=0), (aj=1, ajbar=1)

    Returns
    -------
    rmtl_aj0_ajbar0, rmtl_aj0_ajbar1, rmtl_aj1_ajbar0, rmtl_aj1_ajbar1 : ndarray (n,)
    """
    n = covariates.shape[0]
    rng = np.random.default_rng(seed)

    sums = {k: np.zeros(n) for k in ('00', '01', '10', '11')}
    for _ in range(n_mc):
        tj1_a0 = _exponential_times(1, covariates, **tj1a0_params, rng=rng)
        tj2_a0 = _exponential_times(1, covariates, **tj2a0_params, rng=rng)
        tj1_a1 = _exponential_times(1, covariates, **tj1a1_params, rng=rng)
        tj2_a1 = _exponential_times(1, covariates, **tj2a1_params, rng=rng)

        # aj=0 ajbar=0: both cause times from a=0
        T00 = np.minimum(tj1_a0, tj2_a0)
        J00 = (tj1_a0 < tj2_a0).astype(int) + 1 * (tj1_a0 >= tj2_a0) * 2
        J00 = np.where(tj1_a0 < tj2_a0, 1, 2)
        sums['00'] += (tau - np.minimum(T00, tau)) * (J00 == 1)

        # aj=0 ajbar=1: cause-j from a=0, competing from a=1
        T01 = np.minimum(tj1_a0, tj2_a1)
        J01 = np.where(tj1_a0 < tj2_a1, 1, 2)
        sums['01'] += (tau - np.minimum(T01, tau)) * (J01 == 1)

        # aj=1 ajbar=0: cause-j from a=1, competing from a=0
        T10 = np.minimum(tj1_a1, tj2_a0)
        J10 = np.where(tj1_a1 < tj2_a0, 1, 2)
        sums['10'] += (tau - np.minimum(T10, tau)) * (J10 == 1)

        # aj=1 ajbar=1: both from a=1
        T11 = np.minimum(tj1_a1, tj2_a1)
        J11 = np.where(tj1_a1 < tj2_a1, 1, 2)
        sums['11'] += (tau - np.minimum(T11, tau)) * (J11 == 1)

    return (sums['00'] / n_mc, sums['01'] / n_mc,
            sums['10'] / n_mc, sums['11'] / n_mc)


# ---------------------------------------------------------------------------
# Competing risks DGP
# ---------------------------------------------------------------------------

def make_competing_data(n=400, admin_cens=10.0, time_grid=0.01, tau=4.0,
                        seed=0, compute_true_cate=True):
    """Generate a competing risks dataset matching case1 from the R simulation.

    Ports ``dgp_competing()`` from
    ``original/simulation/competing/competing_hte_simulation_case1.R`` (lines 43–134).

    Parameters
    ----------
    n : int
        Sample size.
    admin_cens : float
        Administrative censoring time (default 10, as in R).
    time_grid : float
        Rounding grid for observed times (ceiling(…/grid)*grid).
    tau : float
        RMTL truncation horizon (R uses taus[4]=4).
    seed : int
        Random seed.
    compute_true_cate : bool
        If True, attach true CATE columns via 1000-rep Monte Carlo.

    Returns
    -------
    data : dict with keys
        X                 : ndarray (n, 6)  covariates
        T                 : ndarray (n,)    binary treatment 0/1
        time              : ndarray (n,)    observed time
        event             : ndarray (n,)    event indicator 0=censored, 1=cause1, 2=cause2
        Y                 : structured ndarray  dtype [('event', int), ('time', float)]
        ps                : ndarray (n,)    true propensity score
        true_cate_total   : ndarray (n,) or None   RMTL(a=1) - RMTL(a=0) total
        true_cate_direct  : ndarray (n,) or None   separable direct effect
        true_cate_indirect: ndarray (n,) or None   separable indirect effect
    """
    rng = np.random.default_rng(seed)

    # --- covariates (identical to survival DGP) ---
    corr = np.array([[1.0, 0.5, 0.5],
                     [0.5, 1.0, 0.5],
                     [0.5, 0.5, 1.0]])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n, 3))
    x_cont = z @ L.T
    x_bin = rng.binomial(1, 0.5, size=(n, 3))
    X = np.column_stack([x_cont, x_bin])
    x1, x2, x3, x4, x5, x6 = X.T

    # --- propensity & treatment (identical formula) ---
    logit_ps = 0.3 + 0.2*x1 + 0.3*x2 + 0.3*x3 - 0.2*x4 - 0.3*x5 - 0.2*x6
    ps = 1.0 / (1.0 + np.exp(-logit_ps))
    A = rng.binomial(1, ps).astype(float)

    cov6 = X

    # --- cause 1 event times (exponential AFT) ---
    tj1a0_params = dict(lambdas=0.12, betas=dict(x0=-0.1, x1=0.1, x2=-0.2, x3=0.2, x4=0.1, x5=0.8, x6=-0.2))
    tj1a1_params = dict(lambdas=0.15, betas=dict(x0=-0.3, x1=0.2, x2=-0.1, x3=0.4, x4=0.2, x5=0.3, x6=0.4))

    # --- cause 2 event times (exponential AFT) ---
    tj2a0_params = dict(lambdas=0.10, betas=dict(x0=0.0, x1=0.1, x2=-0.3, x3=0.1, x4=0.2, x5=0.4, x6=0.3))
    tj2a1_params = dict(lambdas=0.08, betas=dict(x0=-0.2, x1=0.2, x2=0.1, x3=-0.2, x4=0.4, x5=0.3, x6=0.6))

    Tj1_a0 = _exponential_times(1, cov6, **tj1a0_params, rng=rng)
    Tj1_a1 = _exponential_times(1, cov6, **tj1a1_params, rng=rng)
    Tj2_a0 = _exponential_times(1, cov6, **tj2a0_params, rng=rng)
    Tj2_a1 = _exponential_times(1, cov6, **tj2a1_params, rng=rng)

    # observed cause times (use actual arm)
    Tj1 = (1 - A) * Tj1_a0 + A * Tj1_a1
    Tj2 = (1 - A) * Tj2_a0 + A * Tj2_a1

    # round up to time_grid
    Tj1 = np.ceil(Tj1 / time_grid) * time_grid
    Tj2 = np.ceil(Tj2 / time_grid) * time_grid

    # --- censoring (arm-dependent) ---
    betas_c0 = dict(x0=2.5, x1=0.6, x2=-0.4, x3=0.7, x4=1.5, x5=1.2, x6=1.6)
    betas_c1 = dict(x0=2.0, x1=0.6, x2=0.8,  x3=0.5, x4=1.2, x5=1.6, x6=1.2)
    C = np.empty(n)
    mask0 = A == 0
    mask1 = A == 1
    C[mask0] = _lognormal_times(1, cov6[mask0], betas_c0, sigma=0.8, rng=rng)
    C[mask1] = _loglogistic_times(1, cov6[mask1], betas_c1, gamma=0.8, rng=rng)
    C = np.minimum(C, admin_cens)

    # --- observed outcome ---
    T_first = np.minimum(Tj1, Tj2)
    time = np.ceil(np.minimum(C, T_first) / time_grid) * time_grid
    # event: 0=censored, 1=cause1, 2=cause2
    event = np.where(T_first > C, 0, np.where(Tj1 < Tj2, 1, 2)).astype(int)

    Y = np.array([(int(e), float(t)) for e, t in zip(event, time)],
                 dtype=[('event', int), ('time', float)])

    # --- true CATE via Monte-Carlo ---
    true_cate_total = true_cate_direct = true_cate_indirect = None
    if compute_true_cate:
        rmtl_00, rmtl_01, rmtl_10, rmtl_11 = _true_rmtl_mc(
            cov6, tj1a0_params, tj2a0_params, tj1a1_params, tj2a1_params,
            tau, n_mc=1000, seed=seed + 1)
        # total: (aj=1,ajbar=1) vs (aj=0,ajbar=0)
        true_cate_total = rmtl_11 - rmtl_00
        # separable direct: (aj=1,ajbar=1) vs (aj=0,ajbar=1)
        true_cate_direct = rmtl_11 - rmtl_01
        # separable indirect: (aj=0,ajbar=1) vs (aj=0,ajbar=0)
        true_cate_indirect = rmtl_01 - rmtl_00

    return dict(X=X, T=A, time=time, event=event, Y=Y, ps=ps,
                true_cate_total=true_cate_total,
                true_cate_direct=true_cate_direct,
                true_cate_indirect=true_cate_indirect)
