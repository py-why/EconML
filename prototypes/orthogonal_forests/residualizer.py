"""Example implementations of residualizers used by the Orthogonal Forest algorithm.
"""

import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

def dml(W, T, Y, model_T=LassoCV(alphas=[0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 5, 10]),
                 model_Y=LassoCV(alphas=[0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 5, 10])):
    '''
    This is the double ml process of Chernozhukov et al 2016, for computing a treatment 
    effect on any given sample. To be used as a black box by the forest to compute coefficients on splits.
    '''
    res_T = np.zeros(W.shape[0])
    res_Y = np.zeros(W.shape[0])
    
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(W):
        # Split the data in half, train and test
        W_train, T_train, Y_train = W[train_index], T[train_index], Y[train_index]
        W_test, T_test, Y_test  = W[test_index], T[test_index], Y[test_index]
        
        # Fit with LassoCV the treatment as a function of x and the outcome as
        # a function of x, using only the train fold
        model_T.fit(W_train, T_train)
        model_Y.fit(W_train, Y_train)
        
        # Then compute residuals p-g(x) and q-q(x) on test fold
        res_T[test_index] = (T_test - model_T.predict(W_test)).flatten()
        res_Y[test_index] = (Y_test - model_Y.predict(W_test)).flatten()

    # Compute coefficient by OLS on residuals
    return np.sum(np.multiply(res_Y, res_T))/np.sum(np.multiply(res_T, res_T)), res_T, res_Y

def second_order_dml(W, T, Y, model_T=LassoCV(alphas=[0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 5, 10]),
                              model_Y=LassoCV(alphas=[0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 5, 10])):
    '''
    This is the second order orthogonal estimation approach proposed in Mackey, Syrgkanis, Zadik, 2017
    which maintains unbiasedness under even slower first stage rates.
    '''
    res_T = np.zeros(W.shape[0])
    res_Y = np.zeros(W.shape[0])
    mult_T_est_split = np.zeros(W.shape[0])
    
    kf = KFold(n_splits=2)
    for train_index, test_index in kf.split(W):
        # Split the data in half, train and test
        W_train, T_train, Y_train = W[train_index], T[train_index], Y[train_index]
        W_test, T_test, Y_test  = W[test_index], T[test_index], Y[test_index]
        
        # Fit with LassoCV the treatment as a function of x and the outcome as
        # a function of x, using only the train fold
        model_T.fit(W_train, T_train)
        model_Y.fit(W_train, Y_train)
        
        # Then compute residuals p-g(x) and q-q(x) on test fold
        res_T[test_index] = (T_test - model_T.predict(W_test)).flatten()
        res_Y[test_index] = (Y_test - model_Y.predict(W_test)).flatten()

        # Estimate multipliers for second order orthogonal method 
        nested_kf = KFold(n_splits=2)
        for nested_train_index, nested_test_index in nested_kf.split(W):
            res_T_first = res_T[test_index[nested_train_index]]
            second_T_est = np.mean(res_T_first**2)
            cube_T_est = np.mean(res_T_first**3) - 3 * np.mean(res_T_first) * np.mean(res_T_first**2)
            res_T_second = res_T[test_index[nested_test_index]]
            mult_T_est_split[test_index[nested_test_index]] = res_T_second**3 - 3 * second_T_est * res_T_second - cube_T_est

    theta = np.mean(res_Y * mult_T_est_split) / np.mean(res_T * mult_T_est_split)
    return theta, res_T, res_Y
