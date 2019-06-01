# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Orthogonal instrumental variable estimation of ATE.

"""

import numpy as np
from econml.utilities import hstack
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression
import scipy.stats
from sklearn.base import clone


class DMLATEIV:
    """
    Implementation of the orthogonal/double ml method for ATE estimation with
    IV as described in 
    
    Double/Debiased Machine Learning for Treatment and Causal Parameters
    Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins
    https://arxiv.org/abs/1608.00060
    
    Requires that either co-variance of T, Z is independent of X or that effect
    is not heterogeneous in X for correct recovery. Otherwise it estimates
    a biased ATE.
    """

    def __init__(self, model_Y_X, model_T_X, model_Z_X, n_splits=2,
                 binary_instrument=False, binary_treatment=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]. In alt_fit, this model is also
            used to predict E[T | X, Z]
        model_Z_X : model to predict E[Z | X]
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        """
        self.model_Y_X = [clone(model_Y_X, safe=False) for _ in range(n_splits)]
        self.model_T_X = [clone(model_T_X, safe=False) for _ in range(n_splits)]
        self.model_Z_X = [clone(model_Z_X, safe=False) for _ in range(n_splits)]
        self.n_splits = n_splits
        self.binary_instrument = binary_instrument
        self.binary_treatment = binary_treatment

    def fit(self, y, T, X, Z):
        """
        Parameters
        ----------
        y : outcome
        T : treatment (single dimensional)
        X : features/controls
        Z : instrument (single dimensional)
        """
        if len(Z.shape) > 1 and Z.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional instrument")
        if len(T.shape) > 1 and T.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional treatment")
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional outcome")
        Z = Z.flatten()
        T = T.flatten()
        y = y.flatten()

        n_samples = y.shape[0]
        res_t = np.zeros(n_samples)
        res_z = np.zeros(n_samples)
        res_y = np.zeros(n_samples)

        if self.n_splits == 1:
            splits = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
        elif self.binary_instrument or self.binary_treatment:
            group = 2*T*self.binary_treatment + Z.flatten()*self.binary_instrument
            splits = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True).split(X, group)
        else:
            splits = KFold(n_splits=self.n_splits, shuffle=True).split(X)

        for idx, (train, test) in enumerate(splits):
            # Calculate residuals
            res_t[test] = T[test] - \
                self.model_T_X[idx].fit(X[train], T[train]).predict(X[test])
            res_z[test] = Z[test] - \
                self.model_Z_X[idx].fit(X[train], Z[train]).predict(X[test])
            res_y[test] = y[test] - \
                self.model_Y_X[idx].fit(X[train], y[train]).predict(X[test])

        # Estimate E[T_res | Z_res]
        self._effect = np.mean(res_y * res_z)/np.mean(res_t * res_z)

        self._std = np.std(res_y * res_z)/(np.sqrt(res_y.shape[0]) * np.abs(np.mean(res_t * res_z)))

        return self

    def effect(self, X=None):
        """
        Parameters
        ----------
        X : features
        """
        if X is None:
            return self._effect
        else:
            return self._effect * np.ones(X.shape[0])

    def normal_effect_interval(self, lower=5, upper=95):
        return (scipy.stats.norm.ppf(lower/100, loc=self._effect, scale=self._std),
                scipy.stats.norm.ppf(upper/100, loc=self._effect, scale=self._std))
    @property
    def std(self):
        return self._std

    @property
    def fitted_nuisances(self):
        return {'model_Y_X': self.model_Y_X,
                'model_T_X': self.model_T_X,
                'model_Z_X': self.model_Z_X}


class ProjectedDMLATEIV:
    """
    Implementation of the orthogonal/double ml method for ATE estimation with
    IV as described in 
    
    Double/Debiased Machine Learning for Treatment and Causal Parameters
    Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins
    https://arxiv.org/abs/1608.00060
    
    Here we also project the insturment on the treatment and use E[T|X,Z] as the instrument.
    Requires that either co-variance of T, Z is independent of X or that effect
    is not heterogeneous in X for correct recovery. Otherwise it estimates
    a biased ATE.
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ, n_splits=2,
                 binary_instrument=False, binary_treatment=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]
        model_T_XZ : model to predict E[T | X, Z]
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        """
        self.model_Y_X = [clone(model_Y_X, safe=False) for _ in range(n_splits)]
        self.model_T_X = [clone(model_T_X, safe=False) for _ in range(n_splits)]
        self.model_T_XZ = [clone(model_T_XZ, safe=False) for _ in range(n_splits)]
        self.n_splits = n_splits
        self.binary_instrument = binary_instrument
        self.binary_treatment = binary_treatment

    def fit(self, y, T, X, Z):
        """
        Parameters
        ----------
        y : outcome
        T : treatment (single dimensional)
        X : features/controls
        Z : instrument
        """
        if len(T.shape) > 1 and T.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional treatment")
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional outcome")
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        if (Z.shape[1] > 1) and self.binary_instrument:
            raise AssertionError("Binary instrument flag is True, but instrument is multi-dimensional")
        T = T.flatten()
        y = y.flatten()

        n_samples = y.shape[0]
        pred_t = np.zeros(n_samples)
        proj_t = np.zeros(n_samples)
        res_y = np.zeros(n_samples)

        if self.n_splits == 1:
            splits = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
        # TODO. Deal with multi-class instrument
        elif self.binary_instrument or self.binary_treatment:
            group = 2*T*self.binary_treatment + Z.flatten()*self.binary_instrument
            splits = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True).split(X, group)
        else:
            splits = KFold(n_splits=self.n_splits, shuffle=True).split(X)

        for idx, (train, test) in enumerate(splits):
            # Calculate nuisances
            pred_t[test] = self.model_T_X[idx].fit(
                X[train], T[train]).predict(X[test])
            proj_t[test] = self.model_T_XZ[idx].fit(hstack([X[train], Z[train]]),
                                               T[train]).predict(hstack([X[test], Z[test]]))
            res_y[test] = y[test] - \
                self.model_Y_X[idx].fit(X[train], y[train]).predict(X[test])

        # Estimate E[T_res | Z_res]
        res_z = proj_t - pred_t
        res_t = T - pred_t
        self._effect = np.mean(res_y * res_z)/np.mean(res_t * res_z)

        self._std = np.std(res_y * res_z)/(np.sqrt(res_y.shape[0]) * np.abs(np.mean(res_t * res_z)))

        return self

    def effect(self, X=None):
        """
        Parameters
        ----------
        X : features
        """
        if X is None:
            return self._effect
        else:
            return self._effect * np.ones(X.shape[0])

    def normal_effect_interval(self, lower=5, upper=95):
        return (scipy.stats.norm.ppf(lower/100, loc=self._effect, scale=self._std),
                scipy.stats.norm.ppf(upper/100, loc=self._effect, scale=self._std))

    @property
    def std(self):
        return self._std

    @property
    def fitted_nuisances(self):
        return {'model_Y_X': self.model_Y_X,
                'model_T_X': self.model_T_X,
                'model_T_XZ': self.model_T_XZ}

class SimpleATEIV:
    """
    A non-doubly robust simple approach that predicts T from X,Z
    and then runs a regression of Y on E[T | X, Z] and X. No cross-fitting
    is used.
    """

    def __init__(self, model_T_XZ, model_final):
        """
        Parameters
        ----------
        model_T_XZ : model to predict E[T | X, Z]
        model_final : final model for predicting Y from E[T|X,Z], X
        """
        self.model_T_XZ = model_T_XZ
        self.model_final = model_final

    def fit(self, y, T, X, Z):
        """
        Parameters
        ----------
        y : outcome
        T : treatment (single dimensional)
        X : features/controls
        Z : instrument
        """
        if len(T.shape) > 1 and T.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional treatment")
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional outcome")
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        T = T.flatten()
        y = y.flatten()

        pred_t = self.model_T_XZ.fit(hstack([X, Z]), T).predict(hstack([X, Z]))
        self.model_final.fit(hstack([pred_t.reshape(-1, 1), X]), y)

        return self

    def effect(self, X, T0=0, T1=1):
        """
        Parameters
        ----------
        X : features
        """
        if not hasattr(T0, "__len__"):
            T0 = np.ones(X.shape[0])*T0
        if not hasattr(T1, "__len__"):
            T1 = np.ones(X.shape[0])*T1

        X0 = hstack([T0.reshape(-1, 1), X])
        X1 = hstack([T1.reshape(-1, 1), X])
        return self.model_final.predict(X1) - self.model_final.predict(X0)

    @property
    def coef_(self):
        return self.model_final.coef_
