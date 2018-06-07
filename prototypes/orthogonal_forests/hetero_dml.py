import inspect
import numpy as np
import warnings
from joblib import Parallel, delayed
from sklearn.linear_model import LassoCV, Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor


def cross_product(X1, X2):
    """ Computes cross product of features.

    Parameters
    ----------
    X1 (n x d1 matrix). first matrix of n samples of d1 features
    X2 (n x d2 matrix). second matrix of n samples of d2 features
    Returns
    -------
    X12 (n x d1*d2 matrix). matrix of n samples of d1*d2 cross product features,
        arranged in form such that each row t of X12 contains:
        [X1[t,0]*X2[t,0], ..., X1[t,d1-1]*X2[t,0], X1[t,0]*X2[t,1], ..., X1[t,d1-1]*X2[t,1], ...]
    """
    assert np.shape(X1)[0] == np.shape(X2)[0]
    return np.array([np.dot(X1[t].reshape(-1, 1), X2[t].reshape(1, -1)).flatten('F').T for t in range(np.shape(X1)[0])])


class HeteroDML(object):
    
    def __init__(self, poly_degree=3,
                            model_T=LassoCV(),
                            model_Y=LassoCV()):
        self.poly_degree = poly_degree
        self.model_T = model_T
        self.model_Y = model_Y
        self.model_final = LinearRegression(fit_intercept=False)
    
    def fit(self, W, x, T, Y):
        poly_x = PolynomialFeatures(degree=self.poly_degree, include_bias=True).fit_transform(x)
        composite_W = cross_product(poly_x, W)

        res_T = np.zeros(W.shape[0])
        res_Y = np.zeros(W.shape[0])
        
        kf = KFold(n_splits=2)
        for train_index, test_index in kf.split(W):
            # Split the data in half, train and test
            composite_W_train, W_train, T_train, Y_train = composite_W[train_index], W[train_index], T[train_index], Y[train_index]
            composite_W_test, W_test, T_test, Y_test  = composite_W[test_index], W[test_index], T[test_index], Y[test_index]
            
            # Fit with LassoCV the treatment as a function of W and the outcome as
            # a function of W, using only the train fold
            self.model_T.fit(W_train, T_train)
            self.model_Y.fit(composite_W_train, Y_train)
            
            # Then compute residuals T-g(W) and Y-f(W) on test fold
            res_T[test_index] = (T_test - self.model_T.predict(W_test))
            res_Y[test_index] = (Y_test - self.model_Y.predict(composite_W_test))
        
        self.model_final.fit(cross_product(poly_x, res_T.reshape(-1, 1)), res_Y.flatten())

    
    def predict(self, x):
        poly_x = PolynomialFeatures(degree=self.poly_degree, include_bias=True).fit_transform(x)
        return np.dot(poly_x, self.model_final.coef_.reshape(-1, 1)).flatten()


class ForestHeteroDML(object):
    
    def __init__(self):
        self.model_T = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=10)
        self.model_Y = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=10)
        self.model_final = RandomForestRegressor(n_estimators=100, max_depth=10, min_impurity_split=10)
    
    def fit(self, W, x, T, Y):

        res_T = np.zeros(W.shape[0])
        res_Y = np.zeros(W.shape[0])
        
        kf = KFold(n_splits=2)
        for train_index, test_index in kf.split(W):
            # Split the data in half, train and test
            x_train, W_train, T_train, Y_train = x[train_index], W[train_index], T[train_index], Y[train_index]
            x_test, W_test, T_test, Y_test  = x[test_index], W[test_index], T[test_index], Y[test_index]
            
            # Fit with LassoCV the treatment as a function of W and the outcome as
            # a function of W, using only the train fold
            self.model_T.fit(W_train, T_train)
            self.model_Y.fit(np.concatenate((x_train, W_train), axis=1), Y_train)
            
            # Then compute residuals T-g(W) and Y-f(W) on test fold
            res_T[test_index] = (T_test - self.model_T.predict(W_test))
            res_Y[test_index] = (Y_test - self.model_Y.predict(np.concatenate((x_test, W_test), axis=1)))

        self.model_final.fit(np.concatenate((x, res_T.reshape(-1, 1)), axis=1), res_Y.flatten())
        self.res_T = res_T

    
    def predict(self, x):
        # We create fake treatment points from the same distribution as the residuals created during the fit process
        # For each target x, we evaluate the model_final.predict for each such treatment point, and then we average
        # over the predictions to get the prediction at x, i.e. \tau(x) = E_{res_T\sim D_train}[predict(x, res_T)]
        return np.mean([self.model_final.predict(np.concatenate((x, np.ones((x.shape[0],1))*t), axis=1)).flatten()/(t+0.00001) for  t in self.res_T.flatten()], axis=0).flatten()
