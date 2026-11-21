# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def gbr():
    return GradientBoostingRegressor(n_estimators=20, max_depth=2, random_state=0)


def ridge():
    return Ridge(alpha=1.0)


def lr():
    return LogisticRegression(max_iter=200)
