# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import abc
import numpy as np
from econml.utilities import cross_product
from statsmodels.tools.tools import add_constant
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as exn:
    from .utilities import MissingModule

    # make any access to matplotlib or plt throw an exception
    matplotlib = plt = MissingModule("matplotlib is no longer a dependency of the main econml package; "
                                     "install econml[plt] or econml[all] to require it, or install matplotlib "
                                     "separately, to use the tree interpreters", exn)


class _BaseDynamicPanelDGP:

    def __init__(self, n_periods, n_treatments, n_x):
        self.n_periods = n_periods
        self.n_treatments = n_treatments
        self.n_x = n_x
        return

    @abc.abstractmethod
    def create_instance(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _gen_data_with_policy(self, n_units, policy_gen, random_seed=123):
        pass

    def static_policy_data(self, n_units, tau, random_seed=123):
        def policy_gen(Tpre, X, period):
            return tau[period]
        return self._gen_data_with_policy(n_units, policy_gen, random_seed=random_seed)

    def adaptive_policy_data(self, n_units, policy_gen, random_seed=123):
        return self._gen_data_with_policy(n_units, policy_gen, random_seed=random_seed)

    def static_policy_effect(self, tau, mc_samples=1000):
        Y_tau, _, _, _ = self.static_policy_data(mc_samples, tau)
        Y_zero, _, _, _ = self.static_policy_data(
            mc_samples, np.zeros((self.n_periods, self.n_treatments)))
        return np.mean(Y_tau[np.arange(Y_tau.shape[0]) % self.n_periods == self.n_periods - 1]) - \
            np.mean(Y_zero[np.arange(Y_zero.shape[0]) %
                           self.n_periods == self.n_periods - 1])

    def adaptive_policy_effect(self, policy_gen, mc_samples=1000):
        Y_tau, _, _, _ = self.adaptive_policy_data(mc_samples, policy_gen)
        Y_zero, _, _, _ = self.static_policy_data(
            mc_samples, np.zeros((self.n_periods, self.n_treatments)))
        return np.mean(Y_tau[np.arange(Y_tau.shape[0]) % self.n_periods == self.n_periods - 1]) - \
            np.mean(Y_zero[np.arange(Y_zero.shape[0]) %
                           self.n_periods == self.n_periods - 1])


class DynamicPanelDGP(_BaseDynamicPanelDGP):

    def __init__(self, n_periods, n_treatments, n_x):
        super().__init__(n_periods, n_treatments, n_x)

    def create_instance(self, s_x, sigma_x=.8, sigma_y=.1, conf_str=5, hetero_strength=.5, hetero_inds=None,
                        autoreg=.25, state_effect=.25, random_seed=123):
        np.random.seed(random_seed)
        self.s_x = s_x
        self.conf_str = conf_str
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.hetero_inds = hetero_inds.astype(
            int) if hetero_inds is not None else hetero_inds
        self.endo_inds = np.setdiff1d(
            np.arange(self.n_x), hetero_inds).astype(int)
        # The first s_x state variables are confounders. The final s_x variables are exogenous and can create
        # heterogeneity
        self.Alpha = np.random.uniform(-1, 1,
                                       size=(self.n_x, self.n_treatments))
        self.Alpha /= np.linalg.norm(self.Alpha, axis=1, ord=1, keepdims=True)
        self.Alpha *= state_effect
        if self.hetero_inds is not None:
            self.Alpha[self.hetero_inds] = 0

        self.Beta = np.zeros((self.n_x, self.n_x))
        for t in range(self.n_x):
            self.Beta[t, :] = autoreg * np.roll(np.random.uniform(low=4.0**(-np.arange(
                0, self.n_x)), high=4.0**(-np.arange(1, self.n_x + 1))), t)
        if self.hetero_inds is not None:
            self.Beta[np.ix_(self.endo_inds, self.hetero_inds)] = 0
            self.Beta[np.ix_(self.hetero_inds, self.endo_inds)] = 0

        self.epsilon = np.random.uniform(-1, 1, size=self.n_treatments)
        self.zeta = np.zeros(self.n_x)
        self.zeta[:self.s_x] = self.conf_str / self.s_x

        self.y_hetero_effect = np.zeros(self.n_x)
        self.x_hetero_effect = np.zeros(self.n_x)
        if self.hetero_inds is not None:
            self.y_hetero_effect[self.hetero_inds] = np.random.uniform(.5 * hetero_strength, 1.5 * hetero_strength) / \
                len(self.hetero_inds)
            self.x_hetero_effect[self.hetero_inds] = np.random.uniform(.5 * hetero_strength, 1.5 * hetero_strength) / \
                len(self.hetero_inds)

        self.true_effect = np.zeros((self.n_periods, self.n_treatments))
        # Invert indices to match latest API
        self.true_effect[self.n_periods - 1] = self.epsilon
        for t in np.arange(self.n_periods - 2, -1, -1):
            self.true_effect[t, :] = (self.zeta.reshape(
                1, -1) @ np.linalg.matrix_power(self.Beta, (self.n_periods - 1 - t) - 1) @ self.Alpha)

        self.true_hetero_effect = np.zeros(
            (self.n_periods, (self.n_x + 1) * self.n_treatments))
        self.true_hetero_effect[self.n_periods - 1, :] = cross_product(
            add_constant(self.y_hetero_effect.reshape(1, -1), has_constant='add'),
            self.epsilon.reshape(1, -1))
        for t in np.arange(self.n_periods - 2, -1, -1):
            # Invert indices to match latest API
            self.true_hetero_effect[t, :] = cross_product(
                add_constant(self.x_hetero_effect.reshape(1, -1), has_constant='add'),
                self.zeta.reshape(1, -1) @ np.linalg.matrix_power(
                    self.Beta, (self.n_periods - 1 - t) - 1) @ self.Alpha)
        return self

    def hetero_effect_fn(self, t, x):
        if t == self.n_periods - 1:
            return (np.dot(self.y_hetero_effect, x.flatten()) + 1) * self.epsilon
        else:
            return (np.dot(self.x_hetero_effect, x.flatten()) + 1) *\
                (self.zeta.reshape(1, -1) @ np.linalg.matrix_power(self.Beta, (self.n_periods - 1 - t) - 1)
                    @ self.Alpha).flatten()

    def _gen_data_with_policy(self, n_units, policy_gen, random_seed=123):
        np.random.seed(random_seed)
        Y = np.zeros(n_units * self.n_periods)
        T = np.zeros((n_units * self.n_periods, self.n_treatments))
        X = np.zeros((n_units * self.n_periods, self.n_x))
        groups = np.zeros(n_units * self.n_periods)
        for t in range(n_units * self.n_periods):
            period = t % self.n_periods
            if period == 0:
                X[t] = np.random.normal(0, self.sigma_x, size=self.n_x)
                const_x0 = X[t][self.hetero_inds]
                T[t] = policy_gen(np.zeros(self.n_treatments), X[t], period)
            else:
                X[t] = (np.dot(self.x_hetero_effect, X[t - 1]) + 1) * np.dot(self.Alpha, T[t - 1]) + \
                    np.dot(self.Beta, X[t - 1]) + \
                    np.random.normal(0, self.sigma_x, size=self.n_x)
                # The feature for heterogeneity stays constant
                X_t = X[t].copy()
                X_t[self.hetero_inds] = const_x0
                T[t] = policy_gen(T[t - 1], X[t], period)
            Y[t] = (np.dot(self.y_hetero_effect, X_t if period != 0 else X[t]) + 1) * np.dot(self.epsilon, T[t]) + \
                np.dot(X[t], self.zeta) + \
                np.random.normal(0, self.sigma_y)
            groups[t] = t // self.n_periods

        return Y, T, X[:, self.hetero_inds] if (self.hetero_inds is not None) else None, X[:, self.endo_inds], groups

    def observational_data(self, n_units, gamma=0, s_t=1, sigma_t=0.5, random_seed=123):
        """Generate observational data with some observational treatment policy parameters.

        Parameters
        ----------
        n_units : how many units to observe
        gamma : what is the degree of auto-correlation of the treatments across periods
        s_t : sparsity of treatment policy; how many states does it depend on
        sigma_t : what is the std of the exploration/randomness in the treatment
        """
        Delta = np.zeros((self.n_treatments, self.n_x))
        Delta[:, :s_t] = self.conf_str / s_t

        def policy_gen(Tpre, X, period):
            return gamma * Tpre + (1 - gamma) * np.dot(Delta, X) + \
                np.random.normal(0, sigma_t, size=self.n_treatments)
        return self._gen_data_with_policy(n_units, policy_gen, random_seed=random_seed)


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
