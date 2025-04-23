# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from econml.utilities import _safe_norm_ppf

def sensitivity_interval(theta, sigma, nu, cov, alpha, c_y, c_t, rho, interval_type='ci'):
    """Calculate the sensitivity interval."""
    if interval_type not in ['theta', 'ci']:
        raise ValueError(
            f"interval_type for sensitivity_interval must be 'theta' or 'ci'. Received {interval_type}")

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


def RV(theta, sigma, nu, cov, alpha, interval_type='ci'):
    """
    Calculate the robustness value.

    The robustness value is the degree of confounding of *both* the
    treatment and the outcome that still produces an interval
    that excludes zero.

    We're looking for a value of r such that the sensitivity bounds just touch zero

    Returns
    -------
    float
        The robustness value - the level of confounding (between 0 and 1) that would make
        the ATE not statistically significant. A higher value indicates
        a more robust estimate.
        Returns 0 if the original interval already includes zero.

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
    if lb < 0 and ub > 0:
        return 0

    else:
        if lb > 0:
            target = 0
            mult = 1
            d = lb
        else:
            target = 1
            mult = -1
            d = ub

    while abs(d) > 1e-6:
        d = mult * sensitivity_interval(theta, sigma, nu, cov,
                                        alpha, r, r, 1, interval_type=interval_type)[target]
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

    return {
        "theta": theta,
        "sigma": sigma2,
        "nu": nu2,
        "cov": cov
    }


def dr_sensitivity_values(Y, T, y_pred, t_pred):
    y_pred = y_pred.squeeze(1) if np.ndim(y_pred) == 3 else y_pred

    Y = Y.squeeze()
    T_ohe = T[:, 1:].astype(int) # drop first column of T since it's the control
    alpha = np.zeros_like(T_ohe)
    T = np.argmax(T, axis=1)  # Convert dummy encoding to a factor vector

    # one value of theta, sigma, nu for each non-control treatment
    theta = np.zeros(T_ohe.shape[1])
    sigma = np.zeros(T_ohe.shape[1])
    nu = np.zeros(T_ohe.shape[1])

    # one theta, sigma, nu covariance matrix for each non-control treatment
    cov = np.zeros((T_ohe.shape[1], 3, 3))

    # ATE, sigma^2, and nu^2
    for i in range(T_ohe.shape[1]):
        theta_score = y_pred[:, i+1] - y_pred[:, 0] + (Y-y_pred[:, i+1]) * (
            T_ohe[:, i] == 1)/t_pred[:, i+1] - (Y-y_pred[:, 0]) * (np.all(T_ohe == 0, axis=1)/t_pred[:, 0])
        # exclude rows with other treatments
        sigma_score = (Y-np.choose(T, y_pred.T))**2
        alpha[:,i] = (T_ohe[:,i] == 1)/t_pred[:,i+1] - (np.all(T_ohe==0, axis=1))/t_pred[:,0]
        nu_score = 2*(1/t_pred[:,i+1]+1/t_pred[:,0])-alpha[:,i]**2
        theta[i] = np.mean(theta_score)
        sigma[i] = np.mean(sigma_score)
        nu[i] = np.mean(nu_score)
        scores = np.stack([theta_score-theta[i], sigma_score-sigma[i], nu_score-nu[i]], axis=1)
        cov[i,:,:] = (scores.T @ scores / len(scores)) / len(scores)

    return {
        "theta": theta,
        "sigma": sigma,
        "nu": nu,
        "cov": cov
    }
