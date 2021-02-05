import os
import numpy as np
import joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, MultiTaskLasso, MultiTaskLassoCV, LinearRegression
import warnings
from dynamic_panel_dgp import DynamicPanelDGP, LongRangeDynamicPanelDGP
from hetero_panel_dynamic_dml import HeteroDynamicPanelDML
import scipy


def exp(exp_id, dgp, n_units, gamma, s_t, sigma_t, hetero_inds, test_policies):
    np.random.seed(exp_id)
    if exp_id % 100 == 0:
        print(exp_id)

    warnings.simplefilter('ignore')

    Y, T, X, groups = dgp.observational_data(
        n_units, gamma, s_t, sigma_t, random_seed=exp_id)

    # alpha_regs = [5e-3, 1e-2, 5e-2]

    # def lasso_model(): return LassoCV(cv=3, alphas=alpha_regs)

    # def mlasso_model(): return MultiTaskLassoCV(cv=3, alphas=alpha_regs)

    def lasso_model(): return Lasso(alpha=0.05)

    def mlasso_model(): return MultiTaskLasso(alpha=0.05)

    est = HeteroDynamicPanelDML(model_t=mlasso_model(),
                                model_y=lasso_model(),
                                n_cfit_splits=5).fit(Y, T, X, groups, hetero_inds=hetero_inds)

    param_hat = est.param
    conf_ints = est.param_interval(alpha=.05)
    stderrs = est.param_stderr

    policy_effect_hat = np.zeros(test_policies.shape[0])
    policy_effect_lower = np.zeros(test_policies.shape[0])
    policy_effect_upper = np.zeros(test_policies.shape[0])
    policy_effect_stderr = np.zeros(test_policies.shape[0])
    for q in range(test_policies.shape[0]):
        policy_point, policy_ints, policy_std = est.policy_effect(
            test_policies[q], X[:, hetero_inds], groups)
        policy_effect_hat[q] = policy_point
        policy_effect_lower[q] = policy_ints[0]
        policy_effect_upper[q] = policy_ints[1]
        policy_effect_stderr[q] = policy_std

    return (param_hat, conf_ints[:, 0], conf_ints[:, 1], stderrs), (policy_effect_hat, policy_effect_lower, policy_effect_upper, policy_effect_stderr)


def add_vlines(n_periods, n_treatments, hetero_inds):
    locs, labels = plt.xticks([], [])
    locs += [- .5 + (len(hetero_inds) + 1) / 2]
    labels += ["\n\n$\\tau_{{{}}}$".format(0)]
    locs += [qx for qx in np.arange(len(hetero_inds) + 1)]
    labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
    for q in np.arange(1, n_treatments):
        plt.axvline(x=q * (len(hetero_inds) + 1) - .5,
                    linestyle='--', color='red', alpha=.2)
        locs += [q * (len(hetero_inds) + 1) - .5 + (len(hetero_inds) + 1) / 2]
        labels += ["\n\n$\\tau_{{{}}}$".format(q)]
        locs += [(q * (len(hetero_inds) + 1) + qx)
                 for qx in np.arange(len(hetero_inds) + 1)]
        labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
    locs += [- .5 + (len(hetero_inds) + 1) * n_treatments / 2]
    labels += ["\n\n\n\n$\\theta_{{{}}}$".format(0)]
    for t in np.arange(1, n_periods):
        plt.axvline(x=t * (len(hetero_inds) + 1) *
                    n_treatments - .5, linestyle='-', alpha=.6)
        locs += [t * (len(hetero_inds) + 1) * n_treatments - .5 +
                 (len(hetero_inds) + 1) * n_treatments / 2]
        labels += ["\n\n\n\n$\\theta_{{{}}}$".format(t)]
        locs += [t * (len(hetero_inds) + 1) *
                 n_treatments - .5 + (len(hetero_inds) + 1) / 2]
        labels += ["\n\n$\\tau_{{{}}}$".format(0)]
        locs += [t * (len(hetero_inds) + 1) * n_treatments +
                 qx for qx in np.arange(len(hetero_inds) + 1)]
        labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
        for q in np.arange(1, n_treatments):
            plt.axvline(x=t * (len(hetero_inds) + 1) * n_treatments + q * (len(hetero_inds) + 1) - .5,
                        linestyle='--', color='red', alpha=.2)
            locs += [t * (len(hetero_inds) + 1) * n_treatments + q *
                     (len(hetero_inds) + 1) - .5 + (len(hetero_inds) + 1) / 2]
            labels += ["\n\n$\\tau_{{{}}}$".format(q)]
            locs += [t * (len(hetero_inds) + 1) * n_treatments + (q * (len(hetero_inds) + 1) + qx)
                     for qx in np.arange(len(hetero_inds) + 1)]
            labels += ["$1$"] + ["$x_{{{}}}$".format(qx) for qx in hetero_inds]
    plt.xticks(locs, labels)
    plt.tight_layout()


def run_mc(n_exps, n_units, n_x, s_x, n_periods, n_treatments, s_t, sigma_x, sigma_t, sigma_y, gamma):
    print("Running {} MC experiments with: n_units={}, n_dimensions_x={}, non_zero_coefs={}".format(n_exps,
                                                                                                    n_units, n_x, s_x))
    random_seed = 123
    np.random.seed(random_seed)
    conf_str = 1
    # subset of features that are exogenous and create heterogeneity
    true_hetero_inds = np.arange(n_x - 2 * s_x, n_x - s_x)
    # strength of heterogeneity wrt the exogenous variables (assumed to be the last s_x features)
    hetero_strength = 1
    # subset of features wrt we estimate heterogeneity
    hetero_inds = np.arange(n_x - 2 * s_x, n_x)
    n_test_policies = 10
    test_policies = np.random.binomial(1, .5, size=(
        n_test_policies, n_periods, n_treatments))

    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(os.path.join('results', 'long_range_hetero')):
        os.makedirs(os.path.join('results', 'long_range_hetero'))
    dirname = os.path.join('results', 'long_range_hetero')

    param_str = ("n_exps_{}_n_units_{}_n_periods_{}_n_t_{}_n_x_{}_s_x_{}_s_t_{}"
                 "_sigma_x_{}_sigma_t_{}_sigma_y_{}_conf_str_{}_gamma_{}_het_str_{}").format(
        n_exps, n_units, n_periods, n_treatments, n_x, s_x, s_t, sigma_x, sigma_t,
        sigma_y, conf_str, gamma, hetero_strength)

    joblib.dump(hetero_inds, os.path.join(
        dirname, "hetero_hetero_inds_{}.jbl".format(param_str)))
    joblib.dump(test_policies, os.path.join(
        dirname, "hetero_test_policies_{}.jbl".format(param_str)))

    dgp = LongRangeDynamicPanelDGP(n_periods, n_treatments, n_x).create_instance(s_x, sigma_x, sigma_y,
                                                                                 conf_str, hetero_strength, true_hetero_inds,
                                                                                 random_seed=random_seed)
    joblib.dump(dgp, os.path.join(
        dirname, "hetero_dgp_obj_{}.jbl".format(param_str)))

    results = Parallel(n_jobs=-1, max_nbytes=None)(delayed(exp)(i, dgp, n_units, gamma, s_t, sigma_t, hetero_inds, test_policies)
                                                   for i in range(n_exps))
    joblib.dump(results, os.path.join(
        dirname, "hetero_results_{}.jbl".format(param_str)))

    param_results = np.array([r[0] for r in results])
    points = param_results[:, 0]
    lowers = param_results[:, 1]
    uppers = param_results[:, 2]
    stderrs = param_results[:, 3]
    policy_results = np.array([r[1] for r in results])
    policy_effect_hat = policy_results[:, 0]
    policy_effect_lowers = policy_results[:, 1]
    policy_effect_uppers = policy_results[:, 2]
    policy_effect_stderrs = policy_results[:, 3]

    true_effect_inds = []
    for t in range(n_treatments):
        true_effect_inds += [t * (1 + n_x)] + \
            list(t * (1 + n_x) + 1 + hetero_inds)
    true_effect_params = dgp.true_hetero_effect[:, true_effect_inds].flatten()

    true_policy_effect = np.array([dgp.static_policy_effect(
        tau, mc_samples=1000) for tau in test_policies])

    plt.figure(figsize=(15, 5))
    inds = np.arange(points.shape[1])
    plt.violinplot(points, positions=inds, showmeans=True)
    plt.scatter(inds, true_effect_params, marker='o',
                color='#D43F3A', s=10, zorder=3, alpha=.5)
    add_vlines(n_periods, n_treatments, hetero_inds)
    plt.savefig(os.path.join(dirname, "hetero_dists_{}.png".format(param_str)))

    plt.figure(figsize=(15, 5))
    inds = np.arange(points.shape[1])
    plt.violinplot(stderrs, positions=inds, showmeans=True)
    true_std = np.std(points, axis=0)
    true_std_error = (true_std * (np.sqrt((n_exps - 1) / scipy.stats.chi2.ppf((1 - .05 / 2), n_exps - 1)) - 1),
                      true_std * (1 - np.sqrt((n_exps - 1) / scipy.stats.chi2.ppf((.05 / 2), n_exps - 1))))
    plt.errorbar(inds, true_std, yerr=true_std_error, fmt='o',
                 color='#D43F3A', elinewidth=2, alpha=.9, capthick=.5, uplims=True, lolims=True)
    add_vlines(n_periods, n_treatments, hetero_inds)
    plt.savefig(os.path.join(
        dirname, "hetero_stderrs_{}.png".format(param_str)))

    coverage = np.mean((true_effect_params.reshape(1, -1) <= uppers) & (
        true_effect_params.reshape(1, -1) >= lowers), axis=0)
    plt.figure(figsize=(15, 5))
    inds = np.arange(points.shape[1])
    plt.scatter(inds, coverage)
    add_vlines(n_periods, n_treatments, hetero_inds)
    plt.savefig(os.path.join(
        dirname, "hetero_coverage_{}.png".format(param_str)))

    for kappa in range(n_periods):
        for t in range(n_treatments * (len(hetero_inds) + 1)):
            param_ind = kappa * (len(hetero_inds) + 1) * n_treatments + t
            coverage = np.mean((true_effect_params[param_ind] <= uppers[:, param_ind]) & (
                true_effect_params[param_ind] >= lowers[:, param_ind]))
            print("Effect Lag={}, TX={}: Mean={:.3f}, Std={:.3f}, Mean-Stderr={:.3f}, Coverage={:.3f}, (Truth={:.3f})".format(kappa, t,
                                                                                                                              np.mean(
                                                                                                                                  points[:, param_ind]),
                                                                                                                              np.std(
                                                                                                                                  points[:, param_ind]),
                                                                                                                              np.mean(
                                                                                                                                  stderrs[:, param_ind]),
                                                                                                                              coverage,
                                                                                                                              true_effect_params[param_ind]))

    plt.figure(figsize=(15, 5))
    inds = np.arange(policy_effect_hat.shape[1])
    plt.violinplot(policy_effect_hat, positions=inds, showmeans=True)
    plt.scatter(inds, true_policy_effect, marker='o',
                color='#D43F3A', s=10, zorder=3, alpha=.5)
    plt.savefig(os.path.join(
        dirname, "hetero_policy_dists_{}.png".format(param_str)))

    plt.figure(figsize=(15, 5))
    inds = np.arange(policy_effect_hat.shape[1])
    plt.violinplot(policy_effect_stderrs, positions=inds, showmeans=True)
    true_std = np.std(policy_effect_hat, axis=0)
    true_std_error = (true_std * (np.sqrt((n_exps - 1) / scipy.stats.chi2.ppf((1 - .05 / 2), n_exps - 1)) - 1),
                      true_std * (1 - np.sqrt((n_exps - 1) / scipy.stats.chi2.ppf((.05 / 2), n_exps - 1))))
    plt.errorbar(inds, true_std, yerr=true_std_error, fmt='o',
                 color='#D43F3A', elinewidth=2, alpha=.9, capthick=.5, uplims=True, lolims=True)
    plt.savefig(os.path.join(
        dirname, "hetero_policy_stderrs_{}.png".format(param_str)))

    policy_coverage = np.mean((true_policy_effect.reshape(1, -1) <= policy_effect_uppers) & (
        true_policy_effect.reshape(1, -1) >= policy_effect_lowers), axis=0)
    plt.figure(figsize=(15, 5))
    inds = np.arange(policy_coverage.shape[0])
    plt.scatter(inds, policy_coverage)
    plt.savefig(os.path.join(
        dirname, "hetero_policy_coverage_{}.png".format(param_str)))

    for q in range(test_policies.shape[0]):
        print("Policy effect for treatment seq: \n {}\n Mean={:.3f}, Std={:.3f}, Mean-Stderr={:.3f}, Coverage={:.3f}, (Truth={:.3f})".format(test_policies[q],
                                                                                                                                             np.mean(
                                                                                                                                                 policy_effect_hat[:, q]),
                                                                                                                                             np.std(
                                                                                                                                                 policy_effect_hat[:, q]),
                                                                                                                                             np.mean(
                                                                                                                                                 policy_effect_stderrs[:, q]),
                                                                                                                                             policy_coverage[
                                                                                                                                                 q],
                                                                                                                                             true_policy_effect[q]))


if __name__ == "__main__":
    run_mc(n_exps=1000, n_units=500, n_x=450, s_x=2, n_periods=3,
           n_treatments=2, s_t=2, sigma_x=1, sigma_t=1, sigma_y=1, gamma=.2)
    run_mc(n_exps=1000, n_units=1000, n_x=450, s_x=2, n_periods=3,
           n_treatments=2, s_t=2, sigma_x=1, sigma_t=1, sigma_y=1, gamma=.2)
    run_mc(n_exps=1000, n_units=2000, n_x=450, s_x=2, n_periods=3,
           n_treatments=2, s_t=2, sigma_x=1, sigma_t=1, sigma_y=1, gamma=.2)
    run_mc(n_exps=1000, n_units=500, n_x=450, s_x=2, n_periods=3,
           n_treatments=2, s_t=2, sigma_x=1, sigma_t=.5, sigma_y=1, gamma=.2)
    run_mc(n_exps=1000, n_units=2000, n_x=450, s_x=2, n_periods=3,
           n_treatments=2, s_t=2, sigma_x=1, sigma_t=.5, sigma_y=1, gamma=.2)
    run_mc(n_exps=1000, n_units=1000, n_x=450, s_x=2, n_periods=3,
           n_treatments=2, s_t=2, sigma_x=1, sigma_t=.5, sigma_y=.1, gamma=.2)
    run_mc(n_exps=1000, n_units=2000, n_x=450, s_x=2, n_periods=3,
           n_treatments=2, s_t=2, sigma_x=1, sigma_t=.5, sigma_y=.1, gamma=.2)
