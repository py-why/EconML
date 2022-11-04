
import numpy as np


def long(x):
    """
    Reshape panel data to long format, i.e. (n_units * n_periods, d_x) or (n_units * n_periods,)

    Parameters
    ----------
    x : array_like
        Panel data in multi-dimensional format (n_units, n_periods, d_x) or (n_units, n_periods)

    Returns
    -------
    arr : array_like
        Reshaped panel data in long format"""
    n_units = x.shape[0]
    n_periods = x.shape[1]
    if np.ndim(x) == 2:
        return x.reshape(n_units * n_periods)
    else:
        return x.reshape(n_units * n_periods, -1)


def wide(x):
    """
    Reshape panel data to wide format, i.e. (n_units, n_periods * d_x) or (n_units, n_periods,).
    If the input is two-dimensional, this is a no-op because it is already considered to be in wide format.

    Parameters
    ----------
    x : array_like
        Panel data in multi-dimensional format (n_units, n_periods, d_x) or (n_units, n_periods).

    Returns
    -------
    arr : array_like
        Reshaped panel data in wide format"""
    n_units = x.shape[0]
    return x.reshape(n_units, -1)
