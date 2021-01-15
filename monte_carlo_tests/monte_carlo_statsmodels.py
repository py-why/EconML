import numpy as np
from econml.dml import LinearDML
from sklearn.linear_model import LinearRegression, MultiTaskLassoCV, MultiTaskLasso, Lasso
from econml.inference import StatsModelsInference
from econml.tests.test_statsmodels import _summarize
from econml.sklearn_extensions.linear_model import WeightedLasso, WeightedMultiTaskLassoCV, WeightedMultiTaskLasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tools.tools import add_constant
from econml.utilities import cross_product
import matplotlib.pyplot as plt
import os
import time
import argparse
import warnings
import joblib
from sklearn.model_selection import GridSearchCV
from statsmodels.tools.tools import add_constant
from econml.utilities import cross_product
from sklearn.multioutput import MultiOutputRegressor


class GridSearchCVList:

    def __init__(self, estimator_list, param_grid_list, scoring=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0, pre_dispatch='2*n_jobs',
                 error_score='raise-deprecating', return_train_score=False):
        self._gcv_list = [GridSearchCV(estimator, param_grid, scoring=scoring,
                                       n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
                                       pre_dispatch=pre_dispatch, error_score=error_score,
                                       return_train_score=return_train_score)
                          for estimator, param_grid in zip(estimator_list, param_grid_list)]
        return

    def fit(self, X, y, **fit_params):
        self.best_ind_ = np.argmax([gcv.fit(X, y, **fit_params).best_score_ for gcv in self._gcv_list])
        self.best_estimator_ = self._gcv_list[self.best_ind_].best_estimator_
        self.best_score_ = self._gcv_list[self.best_ind_].best_score_
        self.best_params_ = self._gcv_list[self.best_ind_].best_params_
        return self

    def predict(self, X):
        return self.best_estimator_.predict(X)


def _coverage_profile(est, X_test, alpha, true_coef, true_effect):
    cov = {}
    d_t = true_coef.shape[1] // (X_test.shape[1] + 1)
    d_y = true_coef.shape[0]
    coef_interval = est.coef__interval(alpha=alpha)
    intercept_interval = est.intercept__interval(alpha=alpha)
    true_coef = true_coef.flatten()
    est_coef = np.concatenate((est.intercept_[..., np.newaxis], est.coef_), axis=-1).flatten()
    est_coef_lb = np.concatenate((intercept_interval[0][..., np.newaxis], coef_interval[0]), axis=-1).flatten()
    est_coef_ub = np.concatenate((intercept_interval[1][..., np.newaxis], coef_interval[1]), axis=-1).flatten()
    cov['coef'] = est_coef
    cov['coef_lower'] = est_coef_lb
    cov['coef_upper'] = est_coef_ub
    cov['true_coef'] = true_coef
    cov['coef_stderr'] = est.model_final.coef_stderr_.flatten()
    cov['coef_sqerror'] = (est_coef - true_coef)**2
    cov['coef_cov'] = ((true_coef >= est_coef_lb) & (true_coef <= est_coef_ub))
    cov['coef_length'] = est_coef_ub - est_coef_lb
    effect_interval = est.effect_interval(X_test, T0=np.zeros(
        (X_test.shape[0], d_t)), T1=np.ones((X_test.shape[0], d_t)), alpha=alpha)
    true_eff = true_effect(X_test, np.ones((X_test.shape[0], d_t))).reshape(effect_interval[0].shape)
    est_effect = est.effect(X_test, T0=np.zeros((X_test.shape[0], d_t)), T1=np.ones((X_test.shape[0], d_t)))
    cov['x_test'] = np.repeat(X_test, d_y, axis=0)
    cov['effect'] = est_effect.flatten()
    cov['effect_lower'] = effect_interval[0].flatten()
    cov['effect_upper'] = effect_interval[1].flatten()
    cov['true_effect'] = true_eff.flatten()
    cov['effect_sqerror'] = ((est_effect - true_eff)**2).flatten()
    cov['effect_stderr'] = est.model_final.prediction_stderr(
        cross_product(add_constant(X_test), np.ones((X_test.shape[0], d_t)))).flatten()
    cov['effect_cov'] = ((true_eff >= effect_interval[0]) & (true_eff <= effect_interval[1])).flatten()
    cov['effect_length'] = (effect_interval[1] - effect_interval[0]).flatten()
    return cov


def _append_coverage(key, coverage, est, X_test, alpha, true_coef, true_effect):
    cov = _coverage_profile(est, X_test, alpha, true_coef, true_effect)
    if key not in coverage:
        coverage[key] = {}
        for cov_key, value in cov.items():
            coverage[key][cov_key] = [value]
    else:
        for cov_key, value in cov.items():
            coverage[key][cov_key].append(value)


def _agg_coverage(coverage, qs=np.array([.005, .025, .1, .9, .975, .995])):
    mean_coverage_est = {}
    std_coverage_est = {}
    q_coverage_est = {}
    for key, cov_dict in coverage.items():
        mean_coverage_est[key] = {}
        std_coverage_est[key] = {}
        q_coverage_est[key] = {}
        for cov_key, cov_list in cov_dict.items():
            mean_coverage_est[key][cov_key] = np.mean(cov_list, axis=0)
            std_coverage_est[key][cov_key] = np.std(cov_list, axis=0)
            q_coverage_est[key][cov_key] = np.percentile(cov_list, qs * 100, axis=0)
    return mean_coverage_est, std_coverage_est, q_coverage_est


def plot_coverage(coverage, cov_key, n, n_exp, hetero_coef_list, d_list, d_x_list, p_list, t_list,
                  cov_type_list, alpha_list, prefix="", folder="", print_matrix=False):
    if not os.path.exists('figures'):
        os.makedirs('figures')
    if not os.path.exists(os.path.join("figures", folder)):
        os.makedirs(os.path.join("figures", folder))
    joblib.dump(coverage, os.path.join("figures", folder, "{}data.jbl".format(prefix)))
    for hetero_coef in hetero_coef_list:
        for d in d_list:
            for d_x in d_x_list:
                if d_x > d:
                    continue
                for p in p_list:
                    for d_t in t_list:
                        for cov_type in cov_type_list:
                            for alpha in alpha_list:
                                key = "n_{}_n_exp_{}_hetero_{}_d_{}_d_x_{}_p_{}_d_t_{}_cov_type_{}_alpha_{}".format(
                                    n, n_exp, hetero_coef, d, d_x, p, d_t, cov_type, alpha)
                                if print_matrix:
                                    print(coverage[key][cov_key])
                                plt.figure()
                                plt.title("{}{}_{}".format(prefix, key, cov_key))
                                plt.hist(coverage[key][cov_key].flatten())
                                plt.savefig(os.path.join("figures", folder, "{}{}_{}.png".format(prefix,
                                                                                                 key,
                                                                                                 cov_key)))
                                plt.close()


def print_aggregate(mean_coverage, std_coverage, q_coverage, file_gen=lambda: None):
    for key, cov in mean_coverage.items():
        print(key, file=file_gen())
        with np.printoptions(formatter={'float': '{:.2f}'.format}, suppress=True):
            print("Mean Coef (True, RMSE) \t (Coverage, Mean Length) \t Mean StdErr (True, RMSE)\t "
                  "[Mean Lower (std), Mean Upper (std)]\t (True Quantiles)\r\n{}".format(
                      "\r\n".join(["{:.2f} ({:.2f}, {:.2f}) \t ({:.2f}, {:.2f}) \t {:.2f} ({:.2f}, {:.2f}) "
                                   " \t [{:.2f} ({:.2f}), {:.2f} ({:.2f})] \t {}".format(est,
                                                                                         true,
                                                                                         np.sqrt(
                                                                                             sqerr),
                                                                                         coverage,
                                                                                         length,
                                                                                         stderr,
                                                                                         true_stderr,
                                                                                         np.sqrt(
                                                                                             std_stderr**2 +
                                                                                             (stderr -
                                                                                              true_stderr)**2),
                                                                                         lower,
                                                                                         std_lower,
                                                                                         upper,
                                                                                         std_upper,
                                                                                         true_qs)
                                   for (est, true, sqerr, coverage, length, stderr, std_stderr,
                                        true_stderr, lower, std_lower, upper, std_upper, true_qs)
                                   in zip(cov['coef'],
                                          cov['true_coef'],
                                          cov['coef_sqerror'],
                                          cov['coef_cov'],
                                          cov['coef_length'],
                                          cov['coef_stderr'],
                                          std_coverage[key]['coef_stderr'],
                                          std_coverage[key]['coef'],
                                          cov['coef_lower'],
                                          std_coverage[key]['coef_lower'],
                                          cov['coef_upper'],
                                          std_coverage[key]['coef_upper'],
                                          q_coverage[key]['coef'].T)])), file=file_gen())
            print("Effect SqError: {}".format(np.mean(cov['effect_sqerror'])), file=file_gen())

            print("Point\t Mean Coef (True, RMSE) \t (Coverage, Mean Length)\t Mean StdErr (True, RMSE)\t "
                  " [Mean Lower (std), Mean Upper (std)]\t (True Quantiles)\r\n{}".format(
                      "\r\n".join(["{}\t {:.2f} ({:.2f}, {:.2f}) \t ({:.2f}, {:.2f}) \t {:.2f} ({:.2f}, {:.2f}) \t "
                                   "[{:.2f} ({:.2f}), {:.2f} ({:.2f})] \t "
                                   "{}".format(','.join(x.astype(int).astype(str)),
                                               est,
                                               true,
                                               np.sqrt(
                                       sqerr),
                                       coverage,
                                       length,
                                       stderr,
                                       true_stderr,
                                       np.sqrt(
                                       std_stderr**2 + (stderr - true_stderr)**2),
                                       lower,
                                       std_lower,
                                       upper,
                                       std_upper,
                                       true_qs)
                                   for (x, est, true, sqerr, coverage, length, stderr, std_stderr,
                                        true_stderr, lower, std_lower, upper, std_upper, true_qs)
                                   in zip(cov['x_test'],
                                          cov['effect'],
                                          cov['true_effect'],
                                          cov['effect_sqerror'],
                                          cov['effect_cov'],
                                          cov['effect_length'],
                                          cov['effect_stderr'],
                                          std_coverage[key]['effect_stderr'],
                                          std_coverage[key]['effect'],
                                          cov['effect_lower'],
                                          std_coverage[key]['effect_lower'],
                                          cov['effect_upper'],
                                          std_coverage[key]['effect_upper'],
                                          q_coverage[key]['effect'].T)])), file=file_gen())
            print("Effect SqError: {}".format(np.mean(cov['effect_sqerror'])), file=file_gen())


def run_all_mc(first_stage, folder, n_list, n_exp, hetero_coef_list, d_list,
               d_x_list, p_list, t_list, cov_type_list, alpha_list):

    if not os.path.exists("results"):
        os.makedirs('results')
    results_filename = os.path.join("results", "{}.txt".format(folder))

    np.random.seed(123)
    coverage_est = {}
    coverage_lr = {}
    n_tests = 0
    n_failed_coef = 0
    n_failed_effect = 0
    cov_tol = .04
    for n in n_list:
        for hetero_coef in hetero_coef_list:
            for d in d_list:
                for d_x in d_x_list:
                    if d_x > d:
                        continue
                    for p in p_list:
                        for d_t in t_list:
                            X_test = np.unique(np.random.binomial(1, .5, size=(20, d_x)), axis=0)
                            t0 = time.time()
                            for it in range(n_exp):
                                X = np.random.binomial(1, .8, size=(n, d))
                                T = np.hstack([np.random.binomial(1, .5 * X[:, 0] + .25,
                                                                  size=(n,)).reshape(-1, 1) for _ in range(d_t)])
                                true_coef = np.hstack([np.hstack([it + np.arange(p).reshape(-1, 1),
                                                                  it + np.ones((p, 1)), np.zeros((p, d_x - 1))])
                                                       for it in range(d_t)])

                                def true_effect(x, t):
                                    return cross_product(
                                        np.hstack([np.ones((x.shape[0], 1)), x[:, :d_x]]), t) @ true_coef.T
                                y = true_effect(X, T) + X[:, [0] * p] +\
                                    (hetero_coef * X[:, [0]] + 1) * np.random.normal(0, 1, size=(n, p))

                                XT = np.hstack([X, T])
                                X1, X2, y1, y2, X_final_first, X_final_sec, y_sum_first, y_sum_sec,\
                                    n_sum_first, n_sum_sec, var_first, var_sec = _summarize(XT, y)
                                X = np.vstack([X1, X2])
                                y = np.concatenate((y1, y2))
                                X_final = np.vstack([X_final_first, X_final_sec])
                                y_sum = np.concatenate((y_sum_first, y_sum_sec))
                                n_sum = np.concatenate((n_sum_first, n_sum_sec))
                                var_sum = np.concatenate((var_first, var_sec))
                                first_half_sum = len(y_sum_first)
                                first_half = len(y1)
                                for cov_type in cov_type_list:
                                    class SplitterSum:
                                        def __init__(self):
                                            return

                                        def split(self, X, T):
                                            return [(np.arange(0, first_half_sum),
                                                     np.arange(first_half_sum, X.shape[0])),
                                                    (np.arange(first_half_sum, X.shape[0]),
                                                     np.arange(0, first_half_sum))]

                                    est = LinearDML(model_y=first_stage(),
                                                    model_t=first_stage(),
                                                    cv=SplitterSum(),
                                                    linear_first_stages=False,
                                                    discrete_treatment=False)
                                    est.fit(y_sum,
                                            X_final[:, -d_t:],
                                            X_final[:, :d_x],
                                            X_final[:, d_x:-d_t],
                                            sample_weight=n_sum,
                                            sample_var=var_sum,
                                            inference=StatsModelsInference(cov_type=cov_type))

                                    class Splitter:
                                        def __init__(self):
                                            return

                                        def split(self, X, T):
                                            return [(np.arange(0, first_half), np.arange(first_half, X.shape[0])),
                                                    (np.arange(first_half, X.shape[0]), np.arange(0, first_half))]

                                    lr = LinearDML(model_y=first_stage(),
                                                   model_t=first_stage(),
                                                   cv=Splitter(),
                                                   linear_first_stages=False,
                                                   discrete_treatment=False)
                                    lr.fit(y, X[:, -d_t:], X=X[:, :d_x], W=X[:, d_x:-d_t],
                                           inference=StatsModelsInference(cov_type=cov_type))
                                    for alpha in alpha_list:
                                        key = ("n_{}_n_exp_{}_hetero_{}_d_{}_d_x_"
                                               "{}_p_{}_d_t_{}_cov_type_{}_alpha_{}").format(
                                            n, n_exp, hetero_coef, d, d_x, p, d_t, cov_type, alpha)
                                        _append_coverage(key, coverage_est, est, X_test,
                                                         alpha, true_coef, true_effect)
                                        _append_coverage(key, coverage_lr, lr, X_test,
                                                         alpha, true_coef, true_effect)
                                        if it == n_exp - 1:
                                            n_tests += 1
                                            mean_coef_cov = np.mean(coverage_est[key]['coef_cov'])
                                            mean_eff_cov = np.mean(coverage_est[key]['effect_cov'])
                                            mean_coef_cov_lr = np.mean(coverage_lr[key]['coef_cov'])
                                            mean_eff_cov_lr = np.mean(coverage_lr[key]['effect_cov'])
                                            [print("{}. Time: {:.2f}, Mean Coef Cov: ({:.4f}, {:.4f}), "
                                                   "Mean Effect Cov: ({:.4f}, {:.4f})".format(key,
                                                                                              time.time() - t0,
                                                                                              mean_coef_cov,
                                                                                              mean_coef_cov_lr,
                                                                                              mean_eff_cov,
                                                                                              mean_eff_cov_lr),
                                                   file=f)
                                             for f in [None, open(results_filename, "a")]]
                                            coef_cov_dev = mean_coef_cov - (1 - alpha)
                                            if np.abs(coef_cov_dev) >= cov_tol:
                                                n_failed_coef += 1
                                                [print("BAD coef coverage on "
                                                       "average: deviation = {:.4f}".format(coef_cov_dev), file=f)
                                                 for f in [None, open(results_filename, "a")]]
                                            eff_cov_dev = mean_eff_cov - (1 - alpha)
                                            if np.abs(eff_cov_dev) >= cov_tol:
                                                n_failed_effect += 1
                                                [print("BAD effect coverage on "
                                                       "average: deviation = {:.4f}".format(eff_cov_dev), file=f)
                                                 for f in [None, open(results_filename, "a")]]

    [print("Finished {} Monte Carlo Tests. Failed Coef Coverage Tests: {}/{}."
           "Failed Effect Coverage Tests: {}/{}. (Coverage Tolerance={})".format(n_tests,
                                                                                 n_failed_coef,
                                                                                 n_tests,
                                                                                 n_failed_effect,
                                                                                 n_tests,
                                                                                 cov_tol),
           file=f) for f in [None, open(results_filename, "a")]]

    agg_coverage_est, std_coverage_est, q_coverage_est = _agg_coverage(coverage_est)
    agg_coverage_lr, std_coverage_lr, q_coverage_lr = _agg_coverage(coverage_lr)

    [print("\nResults for: {}\n--------------------------\n".format(folder), file=f)
     for f in [None, open(results_filename, "a")]]

    plot_coverage(agg_coverage_est, 'coef_cov', n, n_exp, hetero_coef_list, d_list, d_x_list,
                  p_list, t_list, cov_type_list, alpha_list, prefix="sum_", folder=folder)
    plot_coverage(agg_coverage_lr, 'coef_cov', n, n_exp, hetero_coef_list, d_list, d_x_list,
                  p_list, t_list, cov_type_list, alpha_list, prefix="orig_", folder=folder)
    plot_coverage(agg_coverage_est, 'effect_cov', n, n_exp, hetero_coef_list, d_list, d_x_list,
                  p_list, t_list, cov_type_list, alpha_list, prefix="sum_", folder=folder)
    plot_coverage(agg_coverage_lr, 'effect_cov', n, n_exp, hetero_coef_list, d_list, d_x_list,
                  p_list, t_list, cov_type_list, alpha_list, prefix="orig_", folder=folder)

    [print("Summarized Data\n----------------", file=f) for f in [None, open(results_filename, "a")]]
    print_aggregate(agg_coverage_est, std_coverage_est, q_coverage_est)
    print_aggregate(agg_coverage_est, std_coverage_est, q_coverage_est, lambda: open(results_filename, "a"))
    [print("\nUn-Summarized Data\n-----------------", file=f) for f in [None, open(results_filename, "a")]]
    print_aggregate(agg_coverage_lr, std_coverage_lr, q_coverage_lr)
    print_aggregate(agg_coverage_lr, std_coverage_lr, q_coverage_lr, lambda: open(results_filename, "a"))


def monte_carlo(first_stage=lambda: LinearRegression(), folder='lr'):
    n_exp = 1000
    n_list = [500]
    hetero_coef_list = [0, 1]
    d_list = [1, 10]
    d_x_list = [1, 5]
    p_list = [1, 5]
    t_list = [1, 2]
    cov_type_list = ['HC1']
    alpha_list = [.01, .05, .2]
    run_all_mc(first_stage, folder, n_list, n_exp, hetero_coef_list,
               d_list, d_x_list, p_list, t_list, cov_type_list, alpha_list)


def monte_carlo_lasso(first_stage=lambda: WeightedLasso(alpha=0.01,
                                                        fit_intercept=True,
                                                        tol=1e-6, random_state=123), folder='lasso'):
    n_exp = 1000
    n_list = [500]
    hetero_coef_list = [1]
    d_list = [20]
    d_x_list = [5]
    p_list = [1, 2]
    t_list = [1, 3]
    cov_type_list = ['HC1']
    alpha_list = [.01, .05, .2]
    run_all_mc(first_stage, folder, n_list, n_exp, hetero_coef_list,
               d_list, d_x_list, p_list, t_list, cov_type_list, alpha_list)


def monte_carlo_rf(first_stage=lambda: RandomForestRegressor(n_estimators=100,
                                                             max_depth=3, min_samples_leaf=10), folder='rf'):
    n_exp = 1000
    n_list = [500, 5000]
    hetero_coef_list = [1]
    d_list = [20]
    d_x_list = [5]
    p_list = [1]
    t_list = [2]
    cov_type_list = ['HC1']
    alpha_list = [.01, .05, .2]
    run_all_mc(first_stage, folder, n_list, n_exp, hetero_coef_list,
               d_list, d_x_list, p_list, t_list, cov_type_list, alpha_list)


def monte_carlo_gcv(folder='gcv'):
    def first_stage():
        return GridSearchCVList([LinearRegression(),
                                 WeightedMultiTaskLasso(alpha=0.05, fit_intercept=True,
                                                        tol=1e-6, random_state=123),
                                 RandomForestRegressor(n_estimators=100, max_depth=3,
                                                       min_samples_leaf=10, random_state=123),
                                 MultiOutputRegressor(GradientBoostingRegressor(n_estimators=20,
                                                                                max_depth=3,
                                                                                min_samples_leaf=10, random_state=123))],
                                param_grid_list=[{},
                                                 {},
                                                 {},
                                                 {}],
                                cv=3,
                                iid=True)
    n_exp = 1000
    n_list = [1000, 5000]
    hetero_coef_list = [1]
    d_list = [20]
    d_x_list = [5]
    p_list = [1]
    t_list = [2]
    cov_type_list = ['HC1']
    alpha_list = [.01, .05, .2]
    run_all_mc(first_stage, folder, n_list, n_exp, hetero_coef_list,
               d_list, d_x_list, p_list, t_list, cov_type_list, alpha_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-e', '--exp', help='What experiment (default=all)', required=False, default='all')
    args = vars(parser.parse_args())
    if args['exp'] in ['lr', 'all']:
        monte_carlo()
    if args['exp'] in ['lasso', 'all']:
        monte_carlo_lasso()
    if args['exp'] in ['rf', 'all']:
        monte_carlo_rf()
    if args['exp'] in ['gcv', 'all']:
        monte_carlo_gcv()
