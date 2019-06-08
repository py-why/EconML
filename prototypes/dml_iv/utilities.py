# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Utility classes and functions.
"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone
import numpy as np
import copy
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# A wrapper of statsmodel linear regression, wrapped in a sklearn interface.
# We can use statsmodel for all hypothesis testing capabilities
class StatsModelLinearRegression:

    def __init__(self, fit_intercept=True, cov_type='nonrobust'):
        self.fit_intercept = fit_intercept
        self.cov_type = cov_type
        return

    def fit(self, X, y, sample_weight=None):
        if self.fit_intercept:
            X = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
        if sample_weight is not None:
            X = X * np.sqrt(sample_weight).reshape(-1, 1)
            y = y * np.sqrt(sample_weight)
        self.model = sm.OLS(y, X).fit(cov_type=self.cov_type)
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
        return self.model.predict(exog=X)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    @property
    def coef_(self):
        if self.fit_intercept:
            return self.model._results.params[1:]
        return self.model._results.params

    @property
    def intercept_(self):
        if self.fit_intercept:
            return self.model._results.params[0]
        return 0

# A wrapper of statsmodel linear regression, wrapped in a sklearn interface.
# We can use statsmodel for all hypothesis testing capabilities
class ConstantModel:

    def __init__(self):
        self.est = StatsModelLinearRegression(fit_intercept=False)
        return

    def fit(self, X, y, sample_weight=None):
        self.est.fit(np.ones((X.shape[0], 1)), y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.est.predict(np.ones((X.shape[0], 1)))

    def summary(self, *args, **kwargs):
        return self.est.summary(*args, **kwargs)

    @property
    def coef_(self):
        return None

    @property
    def intercept_(self):
        return self.est.coef_[0]

class SeparateModel:
    """ Splits the data based on the last feature and trains
    a separate model for each subsample. At fit time, it
    uses the last feature to choose which model to use
    to predict.
    """
    def __init__(self, model0, model1):
        self.model0 = model0
        self.model1 = model1
        return

    def fit(self, XZ, T):
        Z0 = (XZ[:, -1] == 0)
        Z1 = (XZ[:, -1] == 1)
        self.model0.fit(XZ[Z0, :-1], T[Z0])
        self.model1.fit(XZ[Z1, :-1], T[Z1])
        return self

    def predict(self, XZ):
        Z0 = (XZ[:, -1] == 0)
        Z1 = (XZ[:, -1] == 1)
        t_pred = np.zeros(XZ.shape[0])
        if np.sum(Z0) > 0:
            t_pred[Z0] = self.model0.predict(XZ[Z0, :-1])
        if np.sum(Z1) > 0:
            t_pred[Z1] = self.model1.predict(XZ[Z1, :-1])
        return t_pred

    @property
    def coef_(self):
        return np.concatenate((self.model0.coef_, self.model1.coef_))

class RegWrapper:
    """
    A simple wrapper that makes a binary classifier behave like a regressor.
    Essentially .fit, calls the fit method of the classifier and
    .predict calls the .predict_proba method of the classifier
    and returns the probability of label 1.
    """

    def __init__(self, clf):
        """
        Parameters
        ----------
        clf : the classifier model
        """
        self._clf = clf

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : features
        y : binary label
        """
        self._clf.fit(X, y)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : features
        """
        return self._clf.predict_proba(X)[:, 1]

    def __getattr__(self, name):
        if name == 'get_params':
            raise AttributeError("not sklearn")
        return getattr(self._clf, name)

    def __deepcopy__(self, memo):
        return RegWrapper(copy.deepcopy(self._clf, memo))

class SubsetWrapper:
    """
    A simple wrapper that fits the data on a subset of the
    features given by an index list.
    """

    def __init__(self, model, inds):
        """
        Parameters
        ----------
        model : an sklearn model
        inds : a subset of the input features to use
        """
        self._model = model
        self._inds = inds

    def fit(self, X, y, **kwargs):
        """
        Parameters
        ----------
        X : features
        y : binary label
        """
        self._model.fit(X[:, self._inds], y, **kwargs)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : subset of features that correspond to inds
        """
        return self._model.predict(X)
    
    def __getattr__(self, name):
        if name == 'get_params':
            raise AttributeError("not sklearn")
        return getattr(self._model, name)

    def __deepcopy__(self, memo):
        return SubsetWrapper(copy.deepcopy(self._model, memo), self._inds)

class WeightWrapper:
    """
    A simple wrapper that adds sample weight to fit of a linear model.
    TODO. Currently this does not seem right for any penalized regression
    as by the internal transformation that we perform we will be penalizing
    the intercept too. Something not intended by the original unweighted class
    So we need to avoid intercept penalization. We could do it with selective
    lasso internally.
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : a linear sklearn model with no intercept
        """
        self._model = clone(model, safe=False)
        self._fit_intercept = False
        if hasattr(self._model, 'fit_intercept') and self._model.fit_intercept:
            raise ValueError("Weight wrapper cannot wrap a linear model with an intercept!")

    def fit(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : features
        y : binary label
        sample_weight : sample weight
        """
        if sample_weight is not None:
            self._model.fit(X * np.sqrt(sample_weight).reshape(X.shape[0], 1), y * np.sqrt(sample_weight))
        else:
            self._model.fit(X, y)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : subset of features that correspond to inds
        """
        return self._model.predict(X)
    
    @property
    def coef_(self):
        if self._fit_intercept:
            return self._model.coef_[1:]
        return self._model.coef_

    @property
    def intercept_(self):
        if self._fit_intercept:
            return self._model.coef_[0]
        return 0

    def __getattr__(self, name):
        if name == 'get_params':
            raise AttributeError("not sklearn")
        return getattr(self._model, name)

    def __deepcopy__(self, memo):
        model_copy = copy.deepcopy(self._model, memo)
        model_copy.fit_intercept = self._fit_intercept
        return WeightWrapper(copy.deepcopy(self._model, memo))

class SelectiveLasso:
    
    def __init__(self, inds, lasso_model):
        self.inds = inds
        self.lasso_model = lasso_model
        self.model_Y_X = LinearRegression(fit_intercept=False)
        self.model_X1_X2 = LinearRegression(fit_intercept=False)
        self.model_X2 = LinearRegression(fit_intercept=False)
        

    def fit(self, X, y):
        self.n_feats = X.shape[1]
        inds = self.inds
        inds_c = np.setdiff1d(np.arange(self.n_feats), self.inds)
        self.inds_c = inds_c
        if len(inds_c)==0:
            self.lasso_model.fit(X, y)
            return self
        res_y = y - self.model_Y_X.fit(X[:, inds_c], y).predict(X[:, inds_c])
        res_X1 = X[:, inds] - self.model_X1_X2.fit(X[:, inds_c], X[:, inds]).predict(X[:, inds_c])
        self.lasso_model.fit(res_X1, res_y)
        self.model_X2.fit(X[:, inds_c], y - self.lasso_model.predict(X[:, inds]))
        return self
    
    def predict(self, X):
        inds = self.inds
        inds_c = self.inds_c
        if len(inds_c)==0:
            return self.lasso_model.predict(X)
        return self.lasso_model.predict(X[:, inds]) + self.model_X2.predict(X[:, inds_c])
    
    @property
    def model(self):
        return self.lasso_model

    @property
    def coef_(self):
        coef = np.zeros(self.n_feats)
        inds = self.inds
        inds_c = self.inds_c
        if len(inds_c)==0:
            return self.lasso_model.coef_
        coef[inds] = self.lasso_model.coef_
        coef[inds_c] = self.model_X2.coef_
        return coef
    
    @property
    def intercept_(self):
        return self.lasso_model.intercept_ + self.model_X2.intercept_


class HonestForest(RandomForestRegressor):
    """
    A simple implementation of an honest locally-linear forest on top of a sklearn tree. We split
    the data in half and fit an sklearn random forest. Then use the other half to
    calculate the leaf estimates. We also fit an elasticnet locally at each
    leaf node.
    """
    def __init__(self, forest, local_linear=False, alpha=0.1):
        """
        Parameters
        ----------
        forest : an sklearn forest model used to create the splits
        alpha : l1/l2 regularization weight for local elasticnet fit at each leaf
        """
        self.forest = forest
        self.alpha = alpha
        self.local_linear = local_linear
        self.mu_leafs = None
        return

    def fit(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        X : features
        y : label
        sample_weight : sample weights
        """
        if sample_weight is not None:
            sample_weight = sample_weight
            X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test = train_test_split(X, y, sample_weight, test_size=.5, shuffle=True)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, shuffle=True)
            sample_weight_train, sample_weight_test = None, None

        self.forest.fit(X_train, y_train, sample_weight=sample_weight_train)
        leaf_test = self.forest.apply(X_test)
        
        self.mu_leafs = []
        for tree in range(leaf_test.shape[1]):                
            leafs = np.unique(leaf_test[:, tree])
            self.mu_leafs.append({})
            for i in leafs:
                if self.local_linear:
                    if sample_weight is not None:
                        lr = WeightWrapper(Pipeline([('bias', PolynomialFeatures(degree=1, include_bias=True)),
                                        ('lasso',  SelectiveLasso(np.arange(1, X.shape[1]+1), ElasticNet(alpha=self.alpha)))]))
                        lr.fit(X_test[leaf_test[:, tree]==i], y_test[leaf_test[:, tree]==i], sample_weight=sample_weight_test[leaf_test[:, tree]==i])
                    else:
                        lr = ElasticNet(alpha=self.alpha).fit(X_test[leaf_test[:, tree]==i], y_test[leaf_test[:, tree]==i])
                    self.mu_leafs[tree][i] = lr
                else:
                    if sample_weight is not None:
                        self.mu_leafs[tree][i] = np.average(y_test[leaf_test[:, tree]==i], weights=sample_weight_test[leaf_test[:, tree]==i])
                    else:
                        self.mu_leafs[tree][i] = np.mean(y_test[leaf_test[:, tree]==i])
        return self

    def predict(self, X):
        """
        X : features
        """
        leaf_pred = self.forest.apply(X)
        if self.local_linear:
            y_pred = [np.array([self.mu_leafs[tree][leaf].predict(X[[sample]])[0] if leaf in  self.mu_leafs[tree] else np.nan
                            for sample, leaf in enumerate(leaf_pred[:, tree])])
                    for tree in range(len(self.mu_leafs))]
        else:
            y_pred = [np.array([self.mu_leafs[tree][leaf] if leaf in  self.mu_leafs[tree] else np.nan
                            for sample, leaf in enumerate(leaf_pred[:, tree])])
                    for tree in range(len(self.mu_leafs))]
        return np.nanmean(y_pred, axis=0)
    
    def predict_interval(self, X, lower=2.5, upper=97.5, little_bags=False):
        """
        X : features
        """
        leaf_pred = self.forest.apply(X)
        if self.local_linear:
            y_pred = [np.array([self.mu_leafs[tree][leaf].predict(X[[sample]])[0] if leaf in  self.mu_leafs[tree] else np.nan
                            for sample, leaf in enumerate(leaf_pred[:, tree])])
                    for tree in range(len(self.mu_leafs))]
        else:
            y_pred = [np.array([self.mu_leafs[tree][leaf] if leaf in  self.mu_leafs[tree] else np.nan
                            for sample, leaf in enumerate(leaf_pred[:, tree])])
                    for tree in range(len(self.mu_leafs))]
        if little_bags:
            subsets = [np.random.choice(leaf_pred.shape[1], int(np.ceil(np.sqrt(leaf_pred.shape[1]))))
                        for _ in range(leaf_pred.shape[1])]
            y_pred = [np.nanmean(np.array(y_pred)[s], axis=0) for s in subsets]
        return np.nanpercentile(y_pred, lower, axis=0), np.nanpercentile(y_pred, upper, axis=0)

    def __getattr__(self, name):
        if name == 'get_params':
            raise AttributeError("not sklearn")
        return getattr(self.forest, name)

    def __deepcopy__(self, memo):
        new_forest = HonestForest(copy.deepcopy(self.forest, memo), local_linear=self.local_linear, alpha=self.alpha)
        if self.mu_leafs is not None:
                new_forest.mu_leafs = copy.deepcopy(self.mu_leafs, memo)
        return new_forest
    