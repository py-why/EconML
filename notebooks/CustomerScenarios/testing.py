# Some imports to get us started
import warnings

warnings.simplefilter('ignore')

# Utilities
import os
import pdb
import urllib.request

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, display
from networkx.drawing.nx_pydot import to_pydot
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
# Generic ML imports
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from econml.cate_interpreter import (SingleTreeCateInterpreter,
                                     SingleTreePolicyInterpreter)
# EconML imports
from econml.dml import CausalForestDML, LinearDML

matplotlib.use("TkAgg")

# Import the sample pricing data
file_url = "https://msalicedatapublic.blob.core.windows.net/datasets/Pricing/pricing_sample.csv"
train_data = pd.read_csv(file_url)

# Define estimator inputs
train_data["log_demand"] = np.log(train_data["demand"])
train_data["log_price"] = np.log(train_data["price"])

Y = train_data["log_demand"].values
T = train_data["log_price"].values
X = train_data[["income"]].values  # features
confounder_names = ["account_age", "age", "avg_hours", "days_visited", "friends_count", "has_membership", "is_US", "songs_purchased"]
W = train_data[confounder_names].values

# Get test data
X_test = np.linspace(0, 5, 100).reshape(-1, 1)
X_test_data = pd.DataFrame(X_test, columns=["income"])

# Create a LinearRegression object with default parameters
lr = LinearRegression()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Original 
# est = LinearDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor(), param_list_y='auto', param_list_t='auto', grid_folds=2,
#               featurizer=PolynomialFeatures(degree=2, include_bias=False))

# est = LinearDML(model_y='automl', model_t=['linear', lr], scaling=True, param_list=None, grid_folds=2,
#               featurizer=PolynomialFeatures(degree=2, include_bias=False))

# Picks default hyper parameters if possible (Linear changes to ElasticNet and then lr doesn't have any)
# est = LinearDML(model_y=GradientBoostingRegressor(), model_t=['linear', lr], scaling=True, param_list='auto',
#               featurizer=PolynomialFeatures(degree=2, include_bias=False))

# est = LinearDML(model_y=GradientBoostingRegressor(), model_t=['linear', lr], scaling=False, param_list='auto',
#               featurizer=PolynomialFeatures(degree=2, include_bias=False))

# Inputs list and we have key words, verbose shows more. 
# est = LinearDML(model_y=['linear', LassoCV(), RandomForestRegressor()], model_t=['poly', lr], param_list=None, verbose=10,
#               featurizer=PolynomialFeatures(degree=2, include_bias=False))

# Random state for models if they have a random state
est = LinearDML(model_y='linear', model_t=lr, cv=kf, param_list_y=None, param_list_t=None, verbose=10,
              featurizer=PolynomialFeatures(degree=2, include_bias=False), random_state=42)

# est = LinearDML(model_y='linear', model_t=lr, grid_folds=kf, param_list=None, verbose=10,
#               featurizer=PolynomialFeatures(degree=2, include_bias=False), random_state=42)

# est = LinearDML(model_y=['linear', LassoCV(), RandomForestRegressor()], model_t=['poly', lr], param_list=None, verbose=10,
#               featurizer=PolynomialFeatures(degree=2, include_bias=False), random_state=42)


# pdb.set_trace()
# fit through dowhy
est_dw = est.dowhy.fit(Y, T, X=X, W=W, outcome_names=["log_demand"], treatment_names=["log_price"], feature_names=["income"],
               confounder_names=confounder_names, inference="statsmodels")
# pdb.set_trace()

identified_estimand = est_dw.identified_estimand_
print(identified_estimand)

lineardml_estimate = est_dw.estimate_
print(lineardml_estimate)

# Get treatment effect and its confidence interval
te_pred = est_dw.effect(X_test).flatten()
te_pred_interval = est_dw.effect_interval(X_test)

# Define underlying treatment effect function given DGP
def gamma_fn(X):
    return -3 - 14 * (X["income"] < 1)

def beta_fn(X):
    return 20 + 0.5 * (X["avg_hours"]) + 5 * (X["days_visited"] > 4)

def demand_fn(data, T):
    Y = gamma_fn(data) * T + beta_fn(data)
    return Y

def true_te(x, n, stats):
    if x < 1:
        subdata = train_data[train_data["income"] < 1].sample(n=n, replace=True)
    else:
        subdata = train_data[train_data["income"] >= 1].sample(n=n, replace=True)
    te_array = subdata["price"] * gamma_fn(subdata) / (subdata["demand"])
    if stats == "mean":
        return np.mean(te_array)
    elif stats == "median":
        return np.median(te_array)
    elif isinstance(stats, int):
        return np.percentile(te_array, stats)

# Get the estimate and range of true treatment effect
truth_te_estimate = np.apply_along_axis(true_te, 1, X_test, 1000, "mean")  # estimate
truth_te_upper = np.apply_along_axis(true_te, 1, X_test, 1000, 95)  # upper level
truth_te_lower = np.apply_along_axis(true_te, 1, X_test, 1000, 5)  # lower level

# Compare the estimate and the truth
plt.figure(figsize=(10, 6))
plt.plot(X_test.flatten(), te_pred, label="Sales Elasticity Prediction")
plt.plot(X_test.flatten(), truth_te_estimate, "--", label="True Elasticity")
plt.fill_between(
    X_test.flatten(),
    te_pred_interval[0].flatten(),
    te_pred_interval[1].flatten(),
    alpha=0.2,
    label="95% Confidence Interval",
)
plt.fill_between(
    X_test.flatten(),
    truth_te_lower,
    truth_te_upper,
    alpha=0.2,
    label="True Elasticity Range",
)
plt.xlabel("Income")
plt.ylabel("Songs Sales Elasticity")
plt.title("Songs Sales Elasticity vs Income")
plt.legend(loc="lower right")
plt.show()