
import numpy as np
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as exn:
    from .utilities import MissingModule

    # make any access to matplotlib or plt throw an exception
    matplotlib = plt = MissingModule("matplotlib is no longer a dependency of the main econml package; "
                                     "install econml[plt] or econml[all] to require it, or install matplotlib "
                                     "separately, to use the tree interpreters", exn)


def long(x):
    """
    Reshape panel data to long format, i.e. (n_units * n_periods, d_x) or (n_units * n_periods,).

    Parameters
    ----------
    x : array_like
        Panel data in multi-dimensional format (n_units, n_periods, d_x) or (n_units, n_periods)

    Returns
    -------
    arr : array_like
    Reshaped panel data in long format
    """
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
    Reshaped panel data in wide format
    """
    n_units = x.shape[0]
    return x.reshape(n_units, -1)


# Auxiliary function for adding xticks and vertical lines when plotting results
# for dynamic dml vs ground truth parameters.
def add_vlines(n_periods, n_treatments, hetero_inds):
    locs, labels = plt.xticks([], [])
    locs += [- .501 + (len(hetero_inds) + 1) / 2]
    labels += ["\n\n$\\tau_{{{}}}$".format(0)]
    locs += [qx for qx in np.arange(len(hetero_inds) + 1)]
    labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
    for q in np.arange(1, n_treatments):
        plt.axvline(x=q * (len(hetero_inds) + 1) - .5,
                    linestyle='--', color='red', alpha=.2)
        locs += [q * (len(hetero_inds) + 1) - .501 + (len(hetero_inds) + 1) / 2]
        labels += ["\n\n$\\tau_{{{}}}$".format(q)]
        locs += [(q * (len(hetero_inds) + 1) + qx)
                 for qx in np.arange(len(hetero_inds) + 1)]
        labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
    locs += [- .501 + (len(hetero_inds) + 1) * n_treatments / 2]
    labels += ["\n\n\n\n$\\theta_{{{}}}$".format(0)]
    for t in np.arange(1, n_periods):
        plt.axvline(x=t * (len(hetero_inds) + 1) *
                    n_treatments - .5, linestyle='-', alpha=.6)
        locs += [t * (len(hetero_inds) + 1) * n_treatments - .501 +
                 (len(hetero_inds) + 1) * n_treatments / 2]
        labels += ["\n\n\n\n$\\theta_{{{}}}$".format(t)]
        locs += [t * (len(hetero_inds) + 1) *
                 n_treatments - .501 + (len(hetero_inds) + 1) / 2]
        labels += ["\n\n$\\tau_{{{}}}$".format(0)]
        locs += [t * (len(hetero_inds) + 1) * n_treatments +
                 qx for qx in np.arange(len(hetero_inds) + 1)]
        labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
        for q in np.arange(1, n_treatments):
            plt.axvline(x=t * (len(hetero_inds) + 1) * n_treatments + q * (len(hetero_inds) + 1) - .5,
                        linestyle='--', color='red', alpha=.2)
            locs += [t * (len(hetero_inds) + 1) * n_treatments + q *
                     (len(hetero_inds) + 1) - .501 + (len(hetero_inds) + 1) / 2]
            labels += ["\n\n$\\tau_{{{}}}$".format(q)]
            locs += [t * (len(hetero_inds) + 1) * n_treatments + (q * (len(hetero_inds) + 1) + qx)
                     for qx in np.arange(len(hetero_inds) + 1)]
            labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
    plt.xticks(locs, labels)
    plt.tight_layout()
