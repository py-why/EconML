# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numbers
import numpy as np


def _get_n_samples_subsample(n_samples, max_samples):
    """
    Get the number of samples in a sub-sample without replacement.
    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.
    Returns
    -------
    n_samples_subsample : int
        The total number of samples to draw for the subsample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples <= 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(np.floor(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _accumulate_prediction(predict, X, out, lock, *args, **kwargs):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, *args, check_input=False, **kwargs)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


def _accumulate_prediction_var(predict, X, out, lock, *args, **kwargs):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    Accumulates the mean covariance of a tree prediction. predict is assumed to
    return an array of (n_samples, d) or a tuple of arrays. This method accumulates in the placeholder
    out[0] the (n_samples, d, d) covariance of the columns of the prediction across
    the trees and for each sample (or a tuple of covariances to be stored in each element
    of the list out).
    """
    prediction = predict(X, *args, check_input=False, **kwargs)
    with lock:
        if len(out) == 1:
            out[0] += np.einsum('ijk,ikm->ijm',
                                prediction.reshape(prediction.shape + (1,)),
                                prediction.reshape((-1, 1) + prediction.shape[1:]))
        else:
            for i in range(len(out)):
                pred_i = prediction[i]
                out[i] += np.einsum('ijk,ikm->ijm',
                                    pred_i.reshape(pred_i.shape + (1,)),
                                    pred_i.reshape((-1, 1) + pred_i.shape[1:]))


def _accumulate_prediction_and_var(predict, X, out, out_var, lock, *args, **kwargs):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    Combines `_accumulate_prediction` and `_accumulate_prediction_var` in a single
    parallel run, so that out will contain the mean of the predictions across trees
    and out_var the covariance.
    """
    prediction = predict(X, *args, check_input=False, **kwargs)
    with lock:
        if len(out) == 1:
            out[0] += prediction
            out_var[0] += np.einsum('ijk,ikm->ijm',
                                    prediction.reshape(prediction.shape + (1,)),
                                    prediction.reshape((-1, 1) + prediction.shape[1:]))
        else:
            for i in range(len(out)):
                pred_i = prediction[i]
                out[i] += prediction
                out_var[i] += np.einsum('ijk,ikm->ijm',
                                        pred_i.reshape(pred_i.shape + (1,)),
                                        pred_i.reshape((-1, 1) + pred_i.shape[1:]))


def _accumulate_oob_preds(tree, X, subsample_inds, alpha_hat, jac_hat, counts, lock):
    mask = np.ones(X.shape[0], dtype=bool)
    mask[subsample_inds] = False
    alpha, jac = tree.predict_alpha_and_jac(X[mask])
    with lock:
        alpha_hat[mask] += alpha
        jac_hat[mask] += jac
        counts[mask] += 1
