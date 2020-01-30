from econml.dml import DMLCateEstimator, LinearDMLCateEstimator, SparseLinearDMLCateEstimator, ForestDMLCateEstimator
import numpy as np
from itertools import product
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV,LinearRegression,MultiTaskElasticNet,MultiTaskElasticNetCV
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split

# Treatment effect function
def exp_te(x):
    return np.exp(2*x[0])

def instance_params(opts, seed):
    n_controls = opts['n_controls']
    n_features = opts['n_features']
    n_samples = opts['n_samples']
    support_size = opts['support_size']

    instance_params = {}
    instance_params['support_Y'] = np.random.choice(np.arange(n_controls), size=support_size, replace=False)
    instance_params['coefs_Y'] = np.random.uniform(0, 1, size=support_size)
    instance_params['support_T'] = instance_params['support_Y']
    instance_params['coefs_T'] = np.random.uniform(0, 1, size=support_size)
    return instance_params

def gen_data(opts, instance_params, seed):
    n_controls = opts['n_controls']
    n_features = opts['n_features']
    n_samples = opts['n_samples']
    # for k, v in instance_params.items():
    #     locals()[k]=v
    support_Y = instance_params['support_Y']
    coefs_Y = instance_params['coefs_Y']
    support_T = instance_params['support_T']
    coefs_T = instance_params['coefs_T']

    # Outcome support
    # support_Y = np.random.choice(np.arange(n_controls), size=support_size, replace=False)
    # coefs_Y = np.random.uniform(0, 1, size=support_size)
    epsilon_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Treatment support
    # support_T = support_Y
    # coefs_T = np.random.uniform(0, 1, size=support_size)
    eta_sample = lambda n: np.random.uniform(-1, 1, size=n)

    # Generate controls, covariates, treatments, and outcomes
    W = np.random.normal(0, 1, size=(n_samples, n_controls))
    X = np.random.uniform(0, 1, size=(n_samples, n_features))

    # Heterogenous treatment effects
    TE = np.array([exp_te(x) for x in X])
    T = np.dot(W[:, support_T], coefs_T) + eta_sample(n_samples)
    Y = TE * T + np.dot(W[:, support_T], coefs_Y) + epsilon_sample(n_samples)
    Y_train, Y_val, T_train, T_val, X_train, X_val, W_train, W_val = train_test_split(Y, T, X, W, test_size=.2)
    # why use train_test_split at all?

    # Generate test data
    X_test = np.array(list(product(np.arange(0, 1, 0.01), repeat=n_features)))
    expected_te = np.array([exp_te(x_i) for x_i in X_test])

    # data, true_param
    return (X_test, Y_train, T_train, X_train, W_train), expected_te

def linear_dml_fit(data, opts, seed):
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    inference = opts['inference']

    est = LinearDMLCateEstimator(model_y=model_y, model_t=model_t, random_state=seed)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as  est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def sparse_linear_dml_poly_fit(data, opts, seed):
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    featurizer = opts['featurizer']
    inference = opts['inference']

    est = SparseLinearDMLCateEstimator(model_y=model_y,
        model_t=model_t,
        featurizer=featurizer,
        random_state=seed)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def dml_poly_fit(data, opts ,seed):
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    featurizer = opts['featurizer']
    model_final = opts['model_final']
    inference = opts['inference']

    est = DMLCateEstimator(model_y=model_y,
        model_t=model_t,
        model_final=model_final,
        featurizer=featurizer,
        random_state=seed)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def forest_dml_fit(data, opts, seed):
    X_test, Y, T, X, W = data
    model_y = opts['model_y']
    model_t = opts['model_t']
    discrete_treatment = opts['discrete_treatment']
    n_estimators = opts['n_estimators']
    subsample_fr = opts['subsample_fr']
    min_samples_leaf = opts['min_samples_leaf']
    min_impurity_decrease = opts['min_impurity_decrease']
    verbose = opts['verbose']
    min_weight_fraction_leaf = opts['min_weight_fraction_leaf']
    inference = opts['inference']

    est = ForestDMLCateEstimator(model_y=model_y,
        model_t=model_t,
        discrete_treatment=discrete_treatment,
        n_estimators=n_estimators,
        subsample_fr=subsample_fr,
        min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease,
        verbose=verbose,
        min_weight_fraction_leaf=min_weight_fraction_leaf)
    est.fit(Y, T, X, W, inference=inference)
    const_marginal_effect = est.const_marginal_effect(X_test) # same as est.effect(X)
    lb, ub = est.const_marginal_effect_interval(X_test, alpha=0.05)

    return (X_test, const_marginal_effect), (lb, ub)

def main():
    opts = {
        'n_controls':30,
        'n_features':1,
        'n_samples':1000,
        'support_size':5
    }
    method_opts = {
        'model_y': RandomForestRegressor(),
        'model_t': RandomForestRegressor(),
        'inference': 'statsmodels'
    }
    mc_opts = {'seed':123}
    data, true_param = gen_data(opts, mc_opts)
    X_test, Y_train, T_train, X_train, W_train = data
    te_pred = linear_dml_fit(data, method_opts, mc_opts['seed'])
    import pdb; pdb.set_trace()
    print(te_pred[0][1])

    # print(lb[0:5])
    # print(ub[0:5])
    # print(np.array(ub-lb)[0:5])
    # print(np.mean(np.array(ub-lb)))

    # def l2_error(x, y): return np.linalg.norm(x-y, ord=2)
    # error = l2_error(X_train, te_pred)
    # print(error)
    # plt.figure(figsize=(10,6))
    # plt.plot(X_test, te_pred, label='DML default')
    # plt.plot(X_test, true_param, 'b--', label='True effect')
    # plt.ylabel('Treatment Effect')
    # plt.xlabel('x')
    # plt.legend()
    # plt.show()

if __name__=="__main__":
    main()
