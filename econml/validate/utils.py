import warnings
from numbers import Real
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _validate_dr_inputs(D, y, reg_preds, prop_preds, min_propensity):
    """Validate inputs shared by the public and diagnostic DR outcome paths."""
    if not isinstance(min_propensity, Real) or isinstance(min_propensity, (bool, np.bool_)):
        raise ValueError("min_propensity must be a finite number strictly between 0 and 0.5")
    min_propensity = float(min_propensity)
    if not np.isfinite(min_propensity) or not 0 < min_propensity < 0.5:
        raise ValueError("min_propensity must be a finite number strictly between 0 and 0.5")

    D = np.asarray(D)
    y = np.asarray(y)
    reg_preds = np.asarray(reg_preds)
    prop_preds = np.asarray(prop_preds)
    if D.ndim != 1 or y.ndim != 1:
        raise ValueError("D and y must be one-dimensional arrays")
    if reg_preds.ndim != 2 or prop_preds.ndim != 2:
        raise ValueError("reg_preds and prop_preds must be two-dimensional arrays")
    if not (D.shape[0] == y.shape[0] == reg_preds.shape[0] == prop_preds.shape[0]):
        raise ValueError("D, y, reg_preds, and prop_preds must have the same number of rows")
    if reg_preds.shape[1] != prop_preds.shape[1]:
        raise ValueError("reg_preds and prop_preds must have the same number of treatment columns")
    if not all(np.all(np.isfinite(values)) for values in (D, y, reg_preds, prop_preds)):
        raise ValueError("D, y, reg_preds, and prop_preds must contain only finite values")

    treatments = np.sort(np.unique(D))
    if treatments.size == 0 or treatments[0] != 0:
        raise ValueError("D must include control treatment 0")
    if not np.all(np.equal(treatments, treatments.astype(int))) or np.any(treatments < 0):
        raise ValueError("D must contain nonnegative integer treatment labels")
    if int(treatments[-1]) >= reg_preds.shape[1]:
        raise ValueError("reg_preds and prop_preds must include a column for every treatment label in D")
    return D, y, reg_preds, prop_preds, treatments.astype(int), min_propensity


def _format_clip_counts(diagnostics: Dict) -> str:
    """Format nonzero per-arm clipping counts for warning messages."""
    return ", ".join(
        f"{label}: {count}"
        for label, count in diagnostics["clipped_by_treatment"].items()
        if count
    )


def _calculate_dr_outcomes_with_diagnostics(
    D: np.array,
    y: np.array,
    reg_preds: np.array,
    prop_preds: np.array,
    *,
    min_propensity: float = 0.01,
) -> Tuple[np.array, Dict]:
    """Compute doubly robust outcomes and propensity clipping diagnostics."""
    D, y, reg_preds, prop_preds, treatments, min_propensity = _validate_dr_inputs(
        D, y, reg_preds, prop_preds, min_propensity
    )
    below_threshold_by_treatment = {
        int(treatment): int(np.count_nonzero(prop_preds[:, treatment] < min_propensity))
        for treatment in treatments
    }
    clipped_by_treatment = {
        int(treatment): int(
            np.count_nonzero((D == treatment) & (prop_preds[:, treatment] < min_propensity))
        )
        for treatment in treatments
    }

    dr_vec = []
    d0_mask = np.where(D == 0, 1, 0)
    y_dr_0 = reg_preds[:, 0] + (d0_mask / np.clip(prop_preds[:, 0], min_propensity, np.inf)) * (
        y - reg_preds[:, 0]
    )
    for k in treatments:
        if k > 0:
            dk_mask = np.where(D == k, 1, 0)
            y_dr_k = reg_preds[:, k] + (dk_mask / np.clip(prop_preds[:, k], min_propensity, np.inf)) * (
                y - reg_preds[:, k]
            )
            dr_vec.append(y_dr_k - y_dr_0)

    diagnostics = {
        "min_propensity": min_propensity,
        "n_samples": D.shape[0],
        "treatment_labels": tuple(int(treatment) for treatment in treatments),
        "below_threshold_by_treatment": below_threshold_by_treatment,
        "n_below_threshold": sum(below_threshold_by_treatment.values()),
        "clipped_by_treatment": clipped_by_treatment,
        "n_clipped": sum(clipped_by_treatment.values()),
    }
    return np.column_stack(dr_vec), diagnostics


def calculate_dr_outcomes(
    D: np.array,
    y: np.array,
    reg_preds: np.array,
    prop_preds: np.array,
    *,
    min_propensity: float = 0.01,
    warn_on_clip: bool = True,
) -> np.array:
    """
    Calculate doubly-robust (DR) outcomes using predictions from nuisance models.

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
    min_propensity: float, default 0.01
        Lower bound applied independently to each treatment propensity used as a denominator.
    warn_on_clip: bool, default True
        Whether to emit a warning when any propensity is below ``min_propensity``. Clipping is a numerical
        stabilization signal and does not by itself prove that identification failed.

    Returns
    -------
    Doubly robust outcome values

    Raises
    ------
    ValueError
        If the clipping threshold is invalid, inputs are nonfinite or have incompatible shapes, control treatment 0
        is absent, or predictions do not contain a column for every observed treatment label.
    """
    dr, diagnostics = _calculate_dr_outcomes_with_diagnostics(
        D, y, reg_preds, prop_preds, min_propensity=min_propensity
    )
    if warn_on_clip and diagnostics["n_clipped"]:
        warnings.warn(
            f"Propensity scores were clipped below min_propensity={diagnostics['min_propensity']:g} "
            f"for {diagnostics['n_clipped']} values across treatment arms "
            f"({_format_clip_counts(diagnostics)}). Clipping is a numerical stabilization signal and does not "
            "by itself prove that identification failed.",
            UserWarning,
            stacklevel=2,
        )
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
    Calculate uplift curve points, integral, and errors on both points and integral.

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
