from typing import Tuple

import numpy as np
import pandas as pd


def calculate_dr_outcomes(
    D: np.array,
    y: np.array,
    reg_preds: np.array,
    prop_preds: np.array
) -> np.array:
    """
    Calculates doubly-robust (DR) outcomes using predictions from nuisance models

    Parameters
    ----------
    D: vector of length n
        Treatment assignments. Should have integer values with the lowest-value corresponding to the
        control treatment. It is recommended to have the control take value 0 and all other treatments be integers
        starting at 1
    y: vector of length n
        Outcomes
    reg_preds: (n x n_treat) matrix
        Outcome predictions for each potential treatment
    prop_preds: (n x n_treat) matrix
        Propensity score predictions for each treatment

    Returns
    -------
    Doubly robust outcome values
    """

    # treat each treatment as a separate regression
    # here, prop_preds should be a matrix
    # with rows corresponding to units and columns corresponding to treatment statuses
    dr_vec = []
    d0_mask = np.where(D == 0, 1, 0)
    y_dr_0 = reg_preds[:, 0] + (d0_mask / np.clip(prop_preds[:, 0], .01, np.inf)) * (y - reg_preds[:, 0])
    for k in np.sort(np.unique(D)):  # pick a treatment status
        if k > 0:  # make sure it is not control
            dk_mask = np.where(D == k, 1, 0)
            y_dr_k = reg_preds[:, k] + (dk_mask / np.clip(prop_preds[:, k], .01, np.inf)) * (y - reg_preds[:, k])
            dr_k = y_dr_k - y_dr_0  # this is an n x 1 vector
            dr_vec.append(dr_k)
    dr = np.column_stack(dr_vec)  # this is an n x n_treatment matrix

    return dr


def calc_uplift(
    cate_preds_train: np.array,
    cate_preds_val: np.array,
    dr_val: np.array,
    percentiles: np.array,
    metric: str,
    n_bootstrap: int = 1000
) -> Tuple[float, float, pd.DataFrame]:
    """
    Helper function for uplift curve generation and coefficient calculation.
    Calculates uplift curve points, integral, and errors on both points and integral.
    Also calculates appropriate critical value multipliers for confidence intervals (via multiplier bootstrap).
    See documentation for "drtester.evaluate_uplift" method for more details.

    Parameters
    ----------
    cate_preds_train: (n_train x n_treatment) matrix
        Predicted CATE values for the training sample.
    cate_preds_val: (n_val x n_treatment) matrix
        Predicted CATE values for the validation sample.
    dr_val: (n_val x n_treatment) matrix
        Doubly robust outcome values for each treatment status in validation sample. Each value is relative to
        control, e.g. for treatment k the value is Y(k) - Y(0), where 0 signifies no treatment.
    percentiles: one-dimensional array
        Array of percentiles over which the QINI curve should be constructed. Defaults to 5%-95% in intervals of 5%.
    metric: string
        String indicating whether to calculate TOC or QINI; should be one of ['toc', 'qini']
    n_bootstrap: integer, default 1000
        Number of bootstrap samples to run when calculating uniform confidence bands.

    Returns
    -------
    Uplift coefficient and associated standard error, as well as associated curve.
    """
    qs = np.percentile(cate_preds_train, percentiles)
    toc, toc_std, group_prob = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
    toc_psi = np.zeros((len(qs), dr_val.shape[0]))
    n = len(dr_val)
    ate = np.mean(dr_val)
    for it in range(len(qs)):
        inds = (qs[it] <= cate_preds_val)  # group with larger CATE prediction than the q-th quantile
        group_prob = np.sum(inds) / n  # fraction of population in this group
        if metric == 'qini':
            toc[it] = group_prob * (
                np.mean(dr_val[inds]) - ate)  # tau(q) = q * E[Y(1) - Y(0) | tau(X) >= q[it]] - E[Y(1) - Y(0)]
            toc_psi[it, :] = np.squeeze(
                (dr_val - ate) * (inds - group_prob) - toc[it])  # influence function for the tau(q)
        elif metric == 'toc':
            toc[it] = np.mean(dr_val[inds]) - ate  # tau(q) := E[Y(1) - Y(0) | tau(X) >= q[it]] - E[Y(1) - Y(0)]
            toc_psi[it, :] = np.squeeze((dr_val - ate) * (inds / group_prob - 1) - toc[it])
        else:
            raise ValueError(f"Unsupported metric {metric!r} - must be one of ['toc', 'qini']")

        toc_std[it] = np.sqrt(np.mean(toc_psi[it] ** 2) / n)  # standard error of tau(q)

    w = np.random.normal(0, 1, size=(n, n_bootstrap))
    mboot = (toc_psi / toc_std.reshape(-1, 1)) @ w / n

    max_mboot = np.max(np.abs(mboot), axis=0)
    uniform_critical_value = np.percentile(max_mboot, 95)

    min_mboot = np.min(mboot, axis=0)
    uniform_one_side_critical_value = np.abs(np.percentile(min_mboot, 5))

    coeff_psi = np.sum(toc_psi[:-1] * np.diff(percentiles).reshape(-1, 1) / 100, 0)
    coeff = np.sum(toc[:-1] * np.diff(percentiles) / 100)
    coeff_stderr = np.sqrt(np.mean(coeff_psi ** 2) / n)

    curve_df = pd.DataFrame({
        'Percentage treated': 100 - percentiles,
        'value': toc,
        'err': toc_std,
        'uniform_critical_value': uniform_critical_value,
        'uniform_one_side_critical_value': uniform_one_side_critical_value
    })

    return coeff, coeff_stderr, curve_df
