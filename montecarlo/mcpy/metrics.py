import numpy as np

def l1_error(x, y): return np.linalg.norm(x-y, ord=1)
def l2_error(x, y): return np.linalg.norm(x-y, ord=2)
def bias(x, y): return x - y
def raw_estimate(x, y): return x
def truth(x, y): return y
def raw_estimate_nonzero_truth(x, y): return x[y>0]

def rmse(param_estimates, true_params):
    X_test = param_estimates[0][0][0]
    pred = np.array([param_estimates[i][0][1] for i in range(len(param_estimates))])
    rmse = np.sqrt(np.mean(np.array(pred-true_params)**2, axis=0))
    return (X_test, rmse)

def std(param_estimates, true_params):
    X_test = param_estimates[0][0][0]
    pred = np.array([param_estimates[i][0][1] for i in range(len(param_estimates))])
    return (X_test, np.std(pred, axis=0))

def conf_length(param_estimates, true_params):
    X_test = param_estimates[0][0][0]
    lb = np.array([param_estimates[i][1][0] for i in range(len(param_estimates))])
    ub = np.array([param_estimates[i][1][1] for i in range(len(param_estimates))])
    conf_length = np.mean(np.array(ub-lb), axis=0)
    return (X_test, conf_length)

def coverage(param_estimates, true_params):
    X_test = param_estimates[0][0][0]
    coverage = []
    for i, param_estimate in enumerate(param_estimates):
        lb, ub = param_estimate[1]
        coverage.append((lb <= true_params[i]) & (true_params[i] <= ub))
    return (X_test, np.mean(coverage, axis=0))

def coverage_band(param_estimates, true_params):
    coverage_band = []
    for i, param_estimate in enumerate(param_estimates):
        lb, ub = param_estimate[1]
        covered = (lb<=true_params[i]).all() & (true_params[i]<=ub).all()
        coverage_band.append(covered)
    return np.array(coverage_band)

def transform_identity(x, dgp, method, metric, config):
    return x[dgp][method][metric]

def transform_diff(x, dgp, method, metric, config):
    return x[dgp][method][metric] - x[dgp][config['proposed_method']][metric]

def transform_ratio(x, dgp, method, metric, config):
    return 100 * (x[dgp][method][metric] - x[dgp][config['proposed_method']][metric]) / x[dgp][method][metric]

def transform_agg_mean(x, dgp, method, metric, config):
    return np.mean(x[dgp][method][metric], axis=1)

def transform_agg_median(x, dgp, method, metric, config):
    return np.median(x[dgp][method][metric], axis=1)

def transform_agg_max(x, dgp, method, metric, config):
    return np.max(x[dgp][method][metric], axis=1)

def transform_diff_positive(x, dgp, method, metric, config):
    return x[dgp][method][metric] - x[dgp][config['proposed_method']][metric] >= 0
