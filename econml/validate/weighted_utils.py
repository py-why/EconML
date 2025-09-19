import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import check_cv
from joblib import Parallel, delayed

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

def weighted_se(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the standard error of the weighted mean.

    Parameters
    ----------
    values : array-like
        Data values (x_i).
    weights : array-like
        Sample weights (w_i), must be non-negative and same shape as values.

    Returns
    -------
    float
        Standard error of the weighted mean.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    if values.shape != weights.shape:
        raise ValueError("values and weights must have the same shape")

    W_sum = np.sum(weights)
    if W_sum == 0:
        return np.nan

    wmean = np.average(values, weights=weights)
    diff = values - wmean

    num = np.sum((weights**2) * (diff**2))
    den = W_sum**2

    return np.sqrt(num / den)

def cross_val_predict_with_weights(
    estimator, X, y, *, sample_weight=None, cv=5, method="predict", n_jobs=None, verbose=0, fit_params=None
):
    """
    Cross-validation predictions with support for sample_weight.

    Works like sklearn.model_selection.cross_val_predict, but passes sample_weight to estimator.fit.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like
        Input features.
    y : array-like
        Target variable.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights to pass to fit.
    cv : int, cross-validation generator, or iterable, default=5
        Determines the cross-validation splitting strategy.
    method : str, default='predict'
        Invokes the passed method name of the estimator.
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        Verbosity level.
    fit_params : dict, default=None
        Additional fit parameters.

    Returns
    -------
    predictions : ndarray
        Array of predictions aligned with the input data.
    """
    if fit_params is None:
        fit_params = {}

    X, y = indexable(X, y)
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    # Where predictions will be stored
    n_samples = _num_samples(X)
    predictions = None

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

    def _fit_and_predict(train, test):
        est = clone(estimator)
        fit_params_fold = fit_params.copy()

        # If sample_weight is provided, slice it for the train indices
        if sample_weight is not None:
            fit_params_fold = {**fit_params_fold, "sample_weight": sample_weight[train]}

        est.fit(X[train], y[train], **fit_params_fold)
        pred = getattr(est, method)(X[test])
        return test, pred

    results = parallel(
        delayed(_fit_and_predict)(train, test)
        for train, test in cv.split(X, y)
    )

    # Allocate predictions after we know their shape
    for test, pred in results:
        if predictions is None:
            # initialize array with right shape and dtype
            if pred.ndim == 1:
                predictions = np.empty(n_samples, dtype=pred.dtype)
            else:
                predictions = np.empty((n_samples, pred.shape[1]), dtype=pred.dtype)
        predictions[test] = pred

    return predictions
