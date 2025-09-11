import numpy as np

def _weighted_quantile_1d(values, quantiles, sample_weight):
    """
    Compute weighted quantiles or percentiles for 1D arrays using binary search + local interpolation.

    Parameters
    ----------
    values : array-like
        Input data (1-D array).
    quantiles : array-like
        Quantiles to compute (must be in [0, 1]).
    sample_weight : array-like
        Weights associated with each value (1-D array of the same length as `values`).

    Returns
    -------
    array-like
        Weighted quantile values.
    """
    sorter = np.argsort(values)
    values_sorted = values[sorter]
    weights_sorted = sample_weight[sorter]

    cdf = np.cumsum(weights_sorted) - 0.5 * weights_sorted
    cdf /= np.sum(weights_sorted)

    idx = np.searchsorted(cdf, quantiles, side="right")

    results = []
    for q, i in zip(quantiles, idx):
        if i == 0:
            results.append(values_sorted[0])
        elif i == len(values_sorted):
            results.append(values_sorted[-1])
        else:
            cdf_lo, cdf_hi = cdf[i-1], cdf[i]
            val_lo, val_hi = values_sorted[i-1], values_sorted[i]
            t = (q - cdf_lo) / (cdf_hi - cdf_lo)
            results.append(val_lo + t * (val_hi - val_lo))
    return np.array(results)

def weighted_stat(values, q, sample_weight=None, axis=None, keepdims=False, mode="quantile"):
    """
    Compute weighted quantiles or percentiles along a given axis using binary search + local interpolation.

    Parameters
    ----------
    values : array-like
        Input data (N-D array).
    q : array-like or float
        Quantiles or percentiles to compute.
        - If mode="quantile", q must be in [0, 1].
        - If mode="percentile", q must be in [0, 100].
    sample_weight : array-like or None, optional
        Weights associated with each value. Must be broadcastable to `values`.
        If None, equal weights are assumed.
    axis : int or None, optional
        Axis along which the computation is performed. Default is None (flatten the array).
    keepdims : bool, optional
        If True, the reduced axes are left in the result as dimensions with size one.
    mode : {"quantile", "percentile"}, default="quantile"
        Whether `q` is specified in quantiles ([0,1]) or percentiles ([0,100]).

    Returns
    -------
    result : ndarray
        Weighted quantile/percentile values.
    """
    values = np.asarray(values)
    q = np.atleast_1d(q)

    if mode == "percentile":
        q = q / 100.0
    elif mode != "quantile":
        raise ValueError("mode must be either 'quantile' or 'percentile'")

    if sample_weight is None:
        sample_weight = np.ones_like(values, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)

    # Flatten if axis=None
    if axis is None:
        values = values.ravel()
        sample_weight = sample_weight.ravel()
        return _weighted_quantile_1d(values, q, sample_weight)

    # Move axis to front, then iterate slice by slice
    values = np.moveaxis(values, axis, 0)
    sample_weight = np.moveaxis(sample_weight, axis, 0)

    result = []
    for v, w in zip(values, sample_weight):
        result.append(_weighted_quantile_1d(v.ravel(), q, w.ravel()))
    result = np.stack(result, axis=0)

    if not keepdims:
        result = np.moveaxis(result, 0, axis)
    else:
        shape = list(values.shape)
        shape[0] = 1
        result = result.reshape([len(q)] + shape)

    return result

