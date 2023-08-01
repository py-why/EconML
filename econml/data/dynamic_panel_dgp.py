import numpy as np
from econml.utilities import cross_product
from statsmodels.tools.tools import add_constant
import pandas as pd
import scipy as sp
from scipy.stats import expon
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib
import os


dir = os.path.dirname(__file__)

# covariance matrix


def new_cov_matrix(cov):
    p = cov.shape[0]
    # get eigen value and eigen vectors
    e_val, e_vec = sp.linalg.eigh(cov)
    start = [0, 35, 77, 86]
    end = [35, 77, 86, p]
    e_val_new = np.array([])
    for i, j in zip(start, end):
        e_val_new = np.append(e_val_new, linear_approximation(i, j, e_val))
    # simulate eigen vectors
    e_vec_new = np.zeros_like(e_vec)
    for i in range(p):
        w = np.zeros(p)  # , np.random.normal(0.01, 0.01, size=p)
        w[np.random.choice(p, 6)] += np.random.normal(0.01, 0.06, size=(6))
        e_vec_new[:, i] = w / np.linalg.norm(w)
    # keep the top 4 eigen value and corresponding eigen vector
    e_vec_new[:, -4:] = e_vec[:, -4:]
    e_val_new[-4:] = e_val[-4:]
    # replace the negative eigen values
    e_val_new[np.where(e_val_new < 0)] = e_val[np.where(e_val_new < 0)]
    # generate a new covariance matrix
    cov_new = e_vec_new.dot(np.diag(e_val_new)).dot(e_vec_new.T)
    return cov_new

# get linear approximation of eigen values


def linear_approximation(start, end, e_val):
    est = LinearRegression()
    X = np.arange(start, end).reshape(-1, 1)
    est.fit(X, e_val[start:end])
    pred = est.predict(X)
    return pred


# coefs
def generate_coefs(index, columns):
    simulated_coefs_df = pd.DataFrame(0, index=index, columns=columns)
    # get the indices of each group of features
    ind_demo = [columns.index(col) for col in columns if "demo" in col]
    ind_proxy = [columns.index(col) for col in columns if "proxy" in col]
    ind_investment = [columns.index(col)
                      for col in columns if "investment" in col]

    for i in range(7):
        outcome_name = simulated_coefs_df.index[i]
        if "proxy" in outcome_name:
            ind_same_proxy = [
                ind for ind in ind_proxy if outcome_name in columns[ind]]
            # print(ind_same_proxy)
            random_proxy_name = np.random.choice(
                [proxy for proxy in index[:4] if proxy != outcome_name]
            )
            ind_random_other_proxy = [
                ind for ind in ind_proxy if random_proxy_name in columns[ind]
            ]
            # demo
            simulated_coefs_df.iloc[
                i, np.random.choice(ind_demo, 2)
            ] = np.random.uniform(0.004, 0.05)
            # same proxy
            simulated_coefs_df.iloc[i, ind_same_proxy] = sorted(
                np.random.choice(expon.pdf(np.arange(10)) *
                                 5e-1, 6, replace=False)
            )
            simulated_coefs_df.iloc[i, ind_random_other_proxy] = sorted(
                np.random.choice(expon.pdf(np.arange(10)) *
                                 5e-2, 6, replace=False)
            )
        elif "investment" in outcome_name:
            ind_same_invest = [
                ind for ind in ind_investment if outcome_name in columns[ind]
            ]
            random_proxy_name = np.random.choice(index[:4])
            ind_random_other_proxy = [
                ind for ind in ind_proxy if random_proxy_name in columns[ind]
            ]
            simulated_coefs_df.iloc[
                i, np.random.choice(ind_demo, 2)
            ] = np.random.uniform(0.001, 0.05)
            simulated_coefs_df.iloc[i, ind_same_invest] = sorted(
                np.random.choice(expon.pdf(np.arange(10)) *
                                 5e-1, 6, replace=False)
            )
            simulated_coefs_df.iloc[i, ind_random_other_proxy] = sorted(
                np.random.choice(expon.pdf(np.arange(10)) *
                                 1e-1, 6, replace=False)
            )
    return simulated_coefs_df


# residuals


def simulate_residuals(ind):
    n, n_pos, n_neg = joblib.load(os.path.join(dir, f"input_dynamicdgp/n_{ind}.jbl"))
    # gmm
    est = joblib.load(os.path.join(dir, f"input_dynamicdgp/gm_{ind}.jbl"))
    x_new = est.sample(n - n_pos - n_neg)[0].flatten()

    # log normal on outliers
    if n_pos > 0:
        # positive outliers
        s, loc, scale = joblib.load(os.path.join(dir, f"input_dynamicdgp/lognorm_pos_{ind}.jbl"))
        fitted_pos_outliers = sp.stats.lognorm(
            s, loc=loc, scale=scale).rvs(size=n_pos)
    else:
        fitted_pos_outliers = np.array([])
    # negative outliers
    if n_neg > 0:
        s, loc, scale = joblib.load(os.path.join(dir, f"input_dynamicdgp/lognorm_neg_{ind}.jbl"))
        fitted_neg_outliers = - \
            sp.stats.lognorm(s, loc=loc, scale=scale).rvs(size=n_neg)
    else:
        fitted_neg_outliers = np.array([])
    x_new = np.concatenate((x_new, fitted_pos_outliers, fitted_neg_outliers))
    return x_new


def simulate_residuals_all(res_df):
    res_df_new = res_df.astype(dtype='float64', copy=True, errors='raise')
    for i in range(res_df.shape[1]):
        res_df_new.iloc[:, i] = simulate_residuals(i)
    # demean the new residual again
    res_df_new = res_df_new - res_df_new.mean(axis=0)
    return res_df_new

# generate data


def get_prediction(df, coef_matrix, residuals, thetas, n, intervention, columns, index, counterfactual):
    data_matrix = df[columns].values
    # sample residuals
    sample_residuals = residuals
    preds = np.matmul(data_matrix, coef_matrix.T)

    # get prediction for current investment
    if counterfactual:
        pred_inv = np.zeros(preds[:, 4:].shape)
    else:
        pred_inv = preds[:, 4:] + sample_residuals[:, 4:] + intervention
    df[index[4:]] = pd.DataFrame(pred_inv, index=df.index)

    # get prediction for current proxy
    pred_proxy = preds[:, :4] + sample_residuals[:, :4] + \
        np.matmul(pred_inv, thetas.T)
    df[index[:4]] = pd.DataFrame(pred_proxy, index=df.index)
    return df


def generate_dgp(
    cov_matrix,
    n_tpid,
    t_period,
    coef_matrix,
    residual_matrix,
    thetas,
    intervention,
    columns,
    index,
    counterfactual
):
    df_all = pd.DataFrame()
    # get first period prediction
    m = cov_matrix.shape[0]
    x = np.random.multivariate_normal(np.repeat(0, m), cov_matrix, size=n_tpid)
    df = pd.DataFrame(
        np.hstack(
            (np.arange(n_tpid).reshape(-1, 1),
             np.repeat(1, n_tpid).reshape(-1, 1), x)
        ),
        columns=["id", "datetime"] + columns,
    )
    df = get_prediction(df, coef_matrix, residual_matrix[0],
                        thetas, n_tpid, intervention, columns, index, False)
    df_all = pd.concat([df_all, df], axis=0)

    # iterate the step ahead contruction
    for t in range(2, t_period + 1):
        # prepare new x
        new_df = df.copy(deep=True)
        new_df["datetime"] = np.repeat(t, n_tpid)
        for name in index:
            for i in range(-6, -1):
                new_df[f"{name}_{i}"] = df[f"{name}_{i+1}"]
            new_df[f"{name}_-1"] = df[name]
        df = get_prediction(new_df, coef_matrix, residual_matrix[t - 1],
                            thetas, n_tpid, [0, 0, 0], columns, index, counterfactual)
        df_all = pd.concat([df_all, df])
    df_all = df_all.sort_values(["id", "datetime"])
    return df_all


class AbstracDynamicPanelDGP:

    def __init__(self, n_periods, n_treatments, n_x):
        self.n_periods = n_periods
        self.n_treatments = n_treatments
        self.n_x = n_x
        return

    def create_instance(self, *args, **kwargs):
        pass

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


class DynamicPanelDGP(AbstracDynamicPanelDGP):

    def __init__(self, n_periods, n_treatments, n_x):
        super().__init__(n_periods, n_treatments, n_x)

    def create_instance(self, s_x, sigma_x, sigma_y, conf_str, epsilon, Alpha_unnormalized,
                        hetero_strength=0, hetero_inds=None,
                        autoreg=.5, state_effect=.5, random_seed=123):
        random_state = np.random.RandomState(random_seed)
        self.s_x = s_x
        self.conf_str = conf_str
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.hetero_inds = hetero_inds.astype(
            int) if hetero_inds is not None else hetero_inds
        self.hetero_strength = hetero_strength
        self.autoreg = autoreg
        self.state_effect = state_effect
        self.random_seed = random_seed
        self.endo_inds = np.setdiff1d(
            np.arange(self.n_x), hetero_inds).astype(int)
        # The first s_x state variables are confounders. The final s_x variables are exogenous and can create
        # heterogeneity
        self.Alpha = Alpha_unnormalized
        self.Alpha /= np.linalg.norm(self.Alpha, axis=1, ord=1, keepdims=True)
        self.Alpha *= state_effect
        if self.hetero_inds is not None:
            self.Alpha[self.hetero_inds] = 0

        self.Beta = np.zeros((self.n_x, self.n_x))
        for t in range(self.n_x):
            self.Beta[t, :] = autoreg * np.roll(random_state.uniform(low=4.0**(-np.arange(
                0, self.n_x)), high=4.0**(-np.arange(1, self.n_x + 1))), t)
        if self.hetero_inds is not None:
            self.Beta[np.ix_(self.endo_inds, self.hetero_inds)] = 0
            self.Beta[np.ix_(self.hetero_inds, self.endo_inds)] = 0

        self.epsilon = epsilon
        self.zeta = np.zeros(self.n_x)
        self.zeta[:self.s_x] = self.conf_str / self.s_x

        self.y_hetero_effect = np.zeros(self.n_x)
        self.x_hetero_effect = np.zeros(self.n_x)
        if self.hetero_inds is not None:
            self.y_hetero_effect[self.hetero_inds] = random_state.uniform(.5 * hetero_strength,
                                                                          1.5 * hetero_strength) /\
                len(self.hetero_inds)
            self.x_hetero_effect[self.hetero_inds] = random_state.uniform(.5 * hetero_strength,
                                                                          1.5 * hetero_strength) / \
                len(self.hetero_inds)

        self.true_effect = np.zeros((self.n_periods, self.n_treatments))
        self.true_effect[0] = self.epsilon
        for t in np.arange(1, self.n_periods):
            self.true_effect[t, :] = (self.zeta.reshape(
                1, -1) @ np.linalg.matrix_power(self.Beta, t - 1) @ self.Alpha)

        self.true_hetero_effect = np.zeros(
            (self.n_periods, (self.n_x + 1) * self.n_treatments))
        self.true_hetero_effect[0, :] = cross_product(add_constant(self.y_hetero_effect.reshape(1, -1),
                                                                   has_constant='add'),
                                                      self.epsilon.reshape(1, -1))
        for t in np.arange(1, self.n_periods):
            self.true_hetero_effect[t, :] = cross_product(add_constant(self.x_hetero_effect.reshape(1, -1),
                                                                       has_constant='add'),
                                                          self.zeta.reshape(1, -1) @
                                                          np.linalg.matrix_power(self.Beta, t - 1) @ self.Alpha)

        return self

    def hetero_effect_fn(self, t, x):
        if t == 0:
            return (np.dot(self.y_hetero_effect, x.flatten()) + 1) * self.epsilon
        else:
            return (np.dot(self.x_hetero_effect, x.flatten()) + 1) *\
                (self.zeta.reshape(1, -1) @ np.linalg.matrix_power(self.Beta, t - 1)
                    @ self.Alpha).flatten()

    def _gen_data_with_policy(self, n_units, policy_gen, random_seed=123):
        random_state = np.random.RandomState(random_seed)
        Y = np.zeros(n_units * self.n_periods)
        T = np.zeros((n_units * self.n_periods, self.n_treatments))
        X = np.zeros((n_units * self.n_periods, self.n_x))
        groups = np.zeros(n_units * self.n_periods)
        for t in range(n_units * self.n_periods):
            period = t % self.n_periods
            if period == 0:
                X[t] = random_state.normal(0, self.sigma_x, size=self.n_x)
                T[t] = policy_gen(np.zeros(self.n_treatments), X[t], period, random_state)
            else:
                X[t] = (np.dot(self.x_hetero_effect, X[t - 1]) + 1) * np.dot(self.Alpha, T[t - 1]) + \
                    np.dot(self.Beta, X[t - 1]) + \
                    random_state.normal(0, self.sigma_x, size=self.n_x)
                T[t] = policy_gen(T[t - 1], X[t], period, random_state)
            Y[t] = (np.dot(self.y_hetero_effect, X[t]) + 1) * np.dot(self.epsilon, T[t]) + \
                np.dot(X[t], self.zeta) + \
                random_state.normal(0, self.sigma_y)
            groups[t] = t // self.n_periods

        return Y, T, X, groups

    def observational_data(self, n_units, gamma, s_t, sigma_t, random_seed=123):
        """ Generated observational data with some observational treatment policy parameters

        Parameters
        ----------
        n_units : how many units to observe
        gamma : what is the degree of auto-correlation of the treatments across periods
        s_t : sparsity of treatment policy; how many states does it depend on
        sigma_t : what is the std of the exploration/randomness in the treatment
        """
        Delta = np.zeros((self.n_treatments, self.n_x))
        Delta[:, :s_t] = self.conf_str / s_t

        def policy_gen(Tpre, X, period, random_state):
            return gamma * Tpre + (1 - gamma) * np.dot(Delta, X) + \
                random_state.normal(0, sigma_t, size=self.n_treatments)
        return self._gen_data_with_policy(n_units, policy_gen, random_seed=random_seed)


class SemiSynthetic:

    def create_instance(self):
        # get new covariance matrix
        self.cov_new = joblib.load(os.path.join(dir, f"input_dynamicdgp/cov_new.jbl"))

        # get coefs
        self.index = ["proxy1", "proxy2", "proxy3", "proxy4",
                      "investment1", "investment2", "investment3", ]
        self.columns = [f"{ind}_{i}" for ind in self.index for i in range(-6, 0)] +\
            [f"demo_{i}" for i in range(47)]

        self.coef_df = generate_coefs(self.index, self.columns)
        self.n_proxies = 4
        self.n_treatments = 3

        # get residuals
        res_df = pd.DataFrame(columns=self.index)
        self.new_res_df = simulate_residuals_all(res_df)

    def gen_data(self, n, n_periods, thetas, random_seed):
        random_state = np.random.RandomState(random_seed)
        n_proxies = self.n_proxies
        n_treatments = self.n_treatments
        coef_matrix = self.coef_df.values
        residual_matrix = self.new_res_df.values
        n_x = len(self.columns)
        # proxy 1 is the outcome
        outcome = "proxy1"

        # make fixed residuals
        all_residuals = []
        for t in range(n_periods):
            sample_residuals = []
            for i in range(7):
                sample_residuals.append(
                    random_state.choice(residual_matrix[:, i], n))
            sample_residuals = np.array(sample_residuals).T
            all_residuals.append(sample_residuals)
        all_residuals = np.array(all_residuals)

        fn_df_control = generate_dgp(self.cov_new, n, n_periods,
                                     coef_matrix, all_residuals, thetas,
                                     [0, 0, 0], self.columns, self.index, False)

        fn_df_cf_control = generate_dgp(self.cov_new, n, n_periods,
                                        coef_matrix, all_residuals, thetas,
                                        [0, 0, 0], self.columns, self.index, True)
        true_effect = np.zeros((n_periods, n_treatments))
        for i in range(n_treatments):
            intervention = [0, 0, 0]
            intervention[i] = 1
            fn_df_treated = generate_dgp(self.cov_new, n, n_periods,
                                         coef_matrix, all_residuals, thetas,
                                         intervention, self.columns, self.index, True)
            for t in range(n_periods):
                ate_control = fn_df_cf_control.loc[
                    fn_df_control["datetime"] == t + 1, outcome
                ].mean()
                ate_treated = fn_df_treated.loc[
                    fn_df_treated["datetime"] == t + 1, outcome
                ].mean()
                true_effect[t, i] = ate_treated - ate_control

        new_index = ["proxy1", "proxy2", "proxy3", "proxy4"]
        new_columns = [f"{ind}_{i}" for ind in new_index for i in range(-6, 0)] +\
            [f"demo_{i}" for i in range(47)]
        panelX = fn_df_control[new_columns].values.reshape(-1, n_periods, len(new_columns))
        panelT = fn_df_control[self.index[n_proxies:]
                               ].values.reshape(-1, n_periods, n_treatments)
        panelY = fn_df_control[outcome].values.reshape(-1, n_periods)
        panelGroups = fn_df_control["id"].values.reshape(-1, n_periods)
        return panelX, panelT, panelY, panelGroups, true_effect

    def plot_coefs(self):
        coef_df = self.coef_df
        plt.figure(figsize=(20, 20))
        for i in range(7):
            outcome = coef_df.index[i]
            plt.subplot(2, 4, i + 1)
            coef_list = coef_df.iloc[i]
            coef_list = coef_list[coef_list != 0]
            plt.plot(coef_list)
            plt.xticks(rotation=90)
            plt.title(f"outcome:{outcome}")
        plt.show()

    def plot_cov(self):
        plt.imshow(self.cov_new)
        plt.colorbar()
        plt.show()
