# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from econml.utilities import _safe_norm_ppf, Summary
from collections import namedtuple

SensitivityParams = namedtuple('SensitivityParams', ['theta', 'sigma', 'nu', 'cov'])

def sensitivity_summary(theta, sigma, nu, cov, null_hypothesis=0, alpha=0.05, c_y=0.05, c_t=0.05, rho=1., decimals=3):
    theta_lb, theta_ub = sensitivity_interval(
        theta, sigma, nu, cov, alpha, c_y, c_t, rho, interval_type='theta')

    ci_lb, ci_ub = sensitivity_interval(
        theta, sigma, nu, cov, alpha, c_y, c_t, rho, interval_type='ci')

    rv_theta = RV(theta, sigma, nu, cov, alpha, null_hypothesis=null_hypothesis, interval_type='theta')
    rv_ci = RV(theta, sigma, nu, cov, alpha, null_hypothesis=null_hypothesis, interval_type='ci')


    smry = Summary()
    title = f'Sensitivity Analysis Summary for c_y={c_y}, c_t={c_t}, rho={rho}'
    res = np.array([[ci_lb, theta_lb, theta, theta_ub, ci_ub]])
    res = np.round(res, decimals)
    headers = ['CI Lower', 'Theta Lower', 'Theta', 'Theta Upper', 'CI Upper']

    smry.add_table(res, headers, [], title=title)

    res_rv = [[rv_theta, rv_ci]]
    res_rv = np.round(res_rv, decimals)
    headers_rv = ['Robustness Value (Theta)', 'Robustness Value (CI)']
    title_rv = f'Robustness Values for null_hypothesis={null_hypothesis}'
    smry.add_table(res_rv, headers_rv, [], title=title_rv)

    return smry


def sensitivity_interval(theta, sigma, nu, cov, alpha, c_y, c_t, rho, interval_type='ci'):
    """Calculate the sensitivity interval."""
    if interval_type not in ['theta', 'ci']:
        raise ValueError(
            f"interval_type for sensitivity_interval must be 'theta' or 'ci'. Received {interval_type}")

    if not (c_y >= 0 and c_y < 1 and c_t >= 0 and c_t < 1):
        raise ValueError(
            "Invalid input: c_y and c_t must be between 0 and 1.")

    if rho < -1 or rho > 1:
        raise ValueError(
            "Invalid input: rho must be between -1 and 1.")

    if sigma < 0 or nu < 0:
        raise ValueError(
            "Invalid input: sigma and nu must be non-negative. "
            "Negative values may indicate issues with the underlying nuisance model estimations.")

    C = np.abs(rho) * np.sqrt(c_y) * np.sqrt(c_t/(1-c_t))/2
    ests = np.array([theta, sigma, nu])

    coefs_p = np.array([1, C*np.sqrt(nu/sigma), C*np.sqrt(sigma/nu)])
    coefs_n = np.array([1, -C*np.sqrt(nu/sigma), -C*np.sqrt(sigma/nu)])

    lb = ests @ coefs_n
    ub = ests @ coefs_p

    if interval_type == 'ci':
        # One dimensional normal distribution:
        sigma_p = coefs_p @ cov @ coefs_p
        sigma_n = coefs_n @ cov @ coefs_n

        lb = _safe_norm_ppf(alpha / 2, loc=lb, scale=np.sqrt(sigma_n))
        ub = _safe_norm_ppf(1 - alpha / 2, loc=ub, scale=np.sqrt(sigma_p))

    return (lb, ub)


def RV(theta, sigma, nu, cov, alpha, null_hypothesis=0, interval_type='ci'):
    """
    Calculate the robustness value.

    The robustness value is the degree of confounding of *both* the
    treatment and the outcome that still produces an interval
    that excludes zero.

    When null_hypothesis is default of zero, we're looking for a value of r
    such that the sensitivity bounds just touch zero.

    Returns
    -------
    float
        The robustness value - the level of confounding (between 0 and 1) that would make
        the ATE reach the null_hypothesis. A higher value indicates
        a more robust estimate.
        Returns 0 if the original interval already includes the null_hypothesis.

    Notes
    -----
    This function uses a binary search approach to find the value of r where the
    sensitivity interval just touches zero.
    """
    if interval_type not in ['theta', 'ci']:
        raise ValueError(
            f"interval_type for sensitivity_interval must be 'theta' or 'ci'. Received {interval_type}")

    r = 0
    r_up = 1
    r_down = 0
    lb, ub = sensitivity_interval(theta, sigma, nu, cov,
                                  alpha, 0, 0, 1, interval_type=interval_type)
    if lb < null_hypothesis and ub > null_hypothesis:
        return 0

    else:
        if lb > null_hypothesis:
            target_ind = 0
            mult = 1
            d = lb - null_hypothesis
        else:
            target_ind = 1
            mult = -1
            d = ub - null_hypothesis

    while abs(d) > 1e-6 and r_up - r_down > 1e-10:
        interval = sensitivity_interval(
            theta, sigma, nu, cov, alpha, r, r, 1, interval_type=interval_type)
        bound = interval[target_ind]
        d = mult * (bound - null_hypothesis)
        if d > 0:
            r_down = r
        else:
            r_up = r

        r = (r_down + r_up) / 2

    return r


def dml_sensitivity_values(t_res, y_res):

    t_res = t_res.reshape(-1, 1)
    y_res = y_res.reshape(-1, 1)

    theta = np.mean(y_res*t_res) / np.mean(t_res**2)  # Estimate the treatment effect
    # Estimate the variance of the outcome residuals (after subtracting the treatment effect)
    sigma2 = np.mean((y_res - theta*t_res)**2)
    nu2 = 1/np.mean(t_res**2)  # Estimate the variance of the treatment

    ls = np.concatenate([t_res**2, np.ones_like(t_res), t_res**2], axis=1)

    G = np.diag(np.mean(ls, axis=0))  # G matrix, diagonal with means of ls
    G_inv = np.linalg.inv(G)  # Inverse of G matrix, could just take reciprocals since it's diagonal

    residuals = np.concatenate([y_res*t_res-theta*t_res*t_res,
                                (y_res-theta*t_res) ** 2-sigma2, t_res**2*nu2-1], axis=1)  # Combine residuals
    # Estimate the covariance matrix of the residuals
    Ω = residuals.T @ residuals / len(residuals)
    # Estimate the variance of the parameters
    cov = G_inv @ Ω @ G_inv / len(residuals)

    return SensitivityParams(
        theta=theta,
        sigma=sigma2,
        nu=nu2,
        cov=cov
    )


def dr_sensitivity_values(Y, T, y_pred, t_pred):
    n_treatments = T.shape[1]

    # squeeze extra dimension if exists
    y_pred = y_pred.squeeze(1) if np.ndim(y_pred) == 3 else y_pred

    T_complete = np.hstack(((np.all(T == 0, axis=1) * 1).reshape(-1, 1), T))
    Y = Y.squeeze()
    alpha = np.zeros(shape=(T.shape[0], n_treatments))
    T = np.argmax(T_complete, axis=1)  # Convert dummy encoding to a factor vector

    # undo doubly robust correction for y pred
    # i.e. reconstruct original y_pred
    y_pred = y_pred.copy() # perform operations on copy to avoid modifying original
    for t in np.arange(T_complete.shape[1]):
        npp = (T_complete[:, t] == 1) / t_pred[:, t]
        y_pred[:, t] = (y_pred[:, t] - npp * Y) / (1 - npp)

    # one value of theta, sigma, nu for each non-control treatment
    theta = np.zeros(n_treatments)
    sigma = np.zeros(n_treatments)
    nu = np.zeros(n_treatments)

    # one theta, sigma, nu covariance matrix for each non-control treatment
    cov = np.zeros((n_treatments, 3, 3))

    # ATE, sigma^2, and nu^2
    for i in range(n_treatments):
        theta_score = y_pred[:, i+1] - y_pred[:, 0] + (Y-y_pred[:, i+1]) * (
            T_complete[:, i+1] == 1)/t_pred[:, i+1] - (Y-y_pred[:, 0]) * ((T_complete[:,0] == 1)/t_pred[:, 0])
        sigma_score = (Y-np.choose(T, y_pred.T))**2
        # exclude rows with other treatments
        sigma_keep = np.isin(T, [0, i+1])
        sigma_score = np.where(sigma_keep, sigma_score, 0)
        alpha[:,i] = (T_complete[:,i+1] == 1)/t_pred[:,i+1] - (T_complete[:,0] == 1)/t_pred[:,0]
        nu_score = 2*(1/t_pred[:,i+1]+1/t_pred[:,0])-alpha[:,i]**2
        theta[i] = np.mean(theta_score)
        sigma[i] = np.mean(sigma_score)
        nu[i] = np.mean(nu_score)
        scores = np.stack([theta_score-theta[i], sigma_score-sigma[i], nu_score-nu[i]], axis=1)
        cov[i,:,:] = (scores.T @ scores / len(scores)) / len(scores)

    return SensitivityParams(
        theta=theta,
        sigma=sigma,
        nu=nu,
        cov=cov
    )
