import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state

_ihdp_sim_file = os.path.join(os.path.dirname(__file__), "ihdp", "sim.csv")
_ihdp_sim_data = pd.read_csv(_ihdp_sim_file)


def ihdp_surface_A(random_state=None):
    """ Generates semi-synthetic, constant treatment effect data according to response surface A
        from Hill (2011).

        Parameters
        ----------
        random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
            If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
            by :mod:`np.random<numpy.random>`.

        Returns
        -------
        Y : array-like, shape (n, d_y)
            Outcome for the treatment policy.

        T : array-like, shape (n, d_t)
            Binary treatment policy.

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.
    """
    # Remove children with nonwhite mothers from the treatment group
    T, X = _process_ihdp_sim_data()
    n = X.shape[0]
    d_x = X.shape[1]
    random_state = check_random_state(random_state)
    beta = random_state.choice([0, 1, 2, 3, 4], size=d_x, replace=True, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    Y = np.dot(X, beta) + T * 4 + random_state.normal(0, 1, size=n)
    true_TE = np.ones(X.shape[0]) * 4
    return Y, T, X, true_TE


def ihdp_surface_B(random_state=None):
    """ Generates semi-synthetic, heterogeneous treatment effect data according to response surface B
        from Hill (2011).

        Parameters
        ----------
        random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
            If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
            by :mod:`np.random<numpy.random>`.

        Returns
        -------
        Y : array-like, shape (n, d_y)
            Outcome for the treatment policy.

        T : array-like, shape (n, d_t)
            Binary treatment policy.

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.
    """
    T, X = _process_ihdp_sim_data()
    n = X.shape[0]
    d_x = X.shape[1]
    random_state = check_random_state(random_state)
    beta = random_state.choice([0, 0.1, 0.2, 0.3, 0.4], size=d_x, replace=True, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    offset = np.concatenate((np.zeros((n, 1)), np.ones((n, d_x - 1)) * 0.5), axis=1)
    omega = np.mean((np.dot(X, beta) - np.exp(np.dot(X + offset, beta)))[T == 1]) - 4
    Y = (np.dot(X, beta) - omega) * T + np.exp(np.dot(X + offset, beta)) * (1 - T) + random_state.normal(0, 1, size=n)
    true_TE = ((np.dot(X, beta) - omega) - np.exp(np.dot(X + offset, beta)))
    return Y, T, X, true_TE


def _process_ihdp_sim_data():
    # Remove children with nonwhite mothers from the treatment group
    data_subset = _ihdp_sim_data[~((_ihdp_sim_data['treat'] == 1) & (_ihdp_sim_data['momwhite'] == 0))]
    T = data_subset['treat'].values
    # Select columns
    X = data_subset[['bw', 'b.head', 'preterm', 'birth.o', 'nnhealth', 'momage',
                     'sex', 'twin', 'b.marr', 'mom.lths', 'mom.hs', 'mom.scoll', 'cig',
                     'first', 'booze', 'drugs', 'work.dur', 'prenatal', 'site1', 'site2',
                     'site3', 'site4', 'site5', 'site6', 'site7']].values
    # Scale the numeric variables
    X[:, :6] = StandardScaler().fit_transform(X[:, :6])
    # Change the binary variable 'first' takes values in {1,2}
    X[:, 13] = X[:, 13] + 1
    # Append a column of ones as intercept
    X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
    return T, X
