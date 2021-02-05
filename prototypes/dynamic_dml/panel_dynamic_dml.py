import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import GroupKFold
import scipy


class DynamicPanelDML:

    def __init__(self, model_t=LassoCV(cv=3),
                 model_y=LassoCV(cv=3),
                 n_cfit_splits=3):
        model_t_copy = clone(model_t, safe=False)
        model_y_copy = clone(model_y, safe=False)
        self._model_t_gen = lambda: clone(model_t_copy, safe=False)
        self._model_y_gen = lambda: clone(model_y_copy, safe=False)
        self._n_cfit_splits = n_cfit_splits
        return

    def fit_nuisances(self, Y, T, X, groups, n_periods):
        ''' Fits all the nuisance models and calculates all residuals for each period and information set
        '''
        resT = {}
        resY = {}
        for kappa in np.arange(n_periods):
            resT[kappa] = {}
            resY[kappa] = np.zeros(self._n_train_units)
            for tau in np.arange(kappa, n_periods):
                resT[kappa][tau] = np.zeros(
                    (self._n_train_units,) + T.shape[1:])

        for train, test in GroupKFold(self._n_cfit_splits).split(X, Y, groups):
            inds_train = train[np.arange(train.shape[0]) % n_periods == 0]
            inds_test = test[np.arange(test.shape[0]) % n_periods == 0]
            for kappa in np.arange(n_periods):
                for tau in np.arange(kappa, n_periods):
                    resT[kappa][tau][inds_test // n_periods] = T[inds_test + tau]\
                        - self._model_t_gen().fit(X[inds_train + kappa],
                                                  T[inds_train + tau]).predict(X[inds_test + kappa])
                    resY[kappa][inds_test // n_periods] = Y[inds_test + n_periods - 1]\
                        - self._model_y_gen().fit(X[inds_train + kappa],
                                                  Y[inds_train + n_periods - 1]).predict(X[inds_test + kappa])
        return resT, resY

    def _fit_cov_matrix(self, resT, resY, models):
        ''' Calculates the covariance (n_periods*n_treatments) x (n_periods*n_treatments) matrix for all the parameters
        '''
        n_periods = len(models)
        M = np.zeros((n_periods * self._n_treatments,
                      n_periods * self._n_treatments))
        Sigma = np.zeros((n_periods * self._n_treatments,
                          n_periods * self._n_treatments))
        for kappa in np.arange(n_periods):
            # Calculating the (kappa, kappa) block entry (of size n_treatments x n_treatments) of matrix Sigma
            period = n_periods - 1 - kappa
            res_epsilon = (resY[period]
                           - np.sum([models[tau].predict(resT[period][n_periods - 1 - tau].reshape(-1, self._n_treatments))
                                     for tau in np.arange(kappa + 1)], axis=0)).reshape(-1, 1, 1)

            cur_resT = resT[period][period]
            cov_cur_resT = np.matmul(cur_resT.reshape(-1, self._n_treatments, 1),
                                     cur_resT.reshape(-1, 1, self._n_treatments))
            sigma_kappa = np.mean((res_epsilon**2) * cov_cur_resT, axis=0)
            Sigma[kappa * self._n_treatments:(kappa + 1) * self._n_treatments,
                  kappa * self._n_treatments:(kappa + 1) * self._n_treatments] = sigma_kappa

            for tau in np.arange(kappa + 1):
                # Calculating the (kappa, tau) block entry (of size n_treatments x n_treatments) of matrix M
                m_kappa_tau = np.mean(
                    np.matmul(resT[period][n_periods - 1 - tau].reshape(-1, self._n_treatments, 1),
                              cur_resT.reshape(-1, 1, self._n_treatments)),
                    axis=0)
                M[kappa * self._n_treatments:(kappa + 1) * self._n_treatments,
                  tau * self._n_treatments:(tau + 1) * self._n_treatments] = m_kappa_tau
        self._cov = np.linalg.inv(M) @ Sigma @ np.linalg.inv(M).T
        self._M = M
        self._Sigma = Sigma
        return self

    def fit_final(self, Y, T, X, groups, resT, resY, n_periods):
        ''' Fits the final lag effect models
        '''
        models = {}
        for kappa in np.arange(n_periods):
            period = n_periods - 1 - kappa
            Y_cal = resY[period].copy()
            if kappa > 0:
                Y_cal -= np.sum([models[tau].predict(resT[period][n_periods - 1 - tau].reshape(-1, self._n_treatments))
                                 for tau in np.arange(kappa)],
                                axis=0)
            models[kappa] = LinearRegression(fit_intercept=False).fit(
                resT[period][period].reshape(-1, self._n_treatments), Y_cal)

        self._fit_cov_matrix(resT, resY, models)
        self.final_models = models
        return self

    def fit(self, Y, T, X, groups):
        u_periods = np.unique(np.bincount(groups.astype(int)))
        self._n_train_units = len(np.unique(groups))
        self._n_treatments = 1 if len(T.shape[1:]) == 0 else T.shape[1]
        if len(u_periods) > 1:
            raise AttributeError(
                "Imbalanced panel. Method currently expects only panels with equal number of periods. Pad your data")
        self._n_train_periods = u_periods[0]
        resT, resY = self.fit_nuisances(Y, T, X, groups, self._n_train_periods)
        self.fit_final(Y, T, X, groups, resT, resY, self._n_train_periods)
        return self

    @property
    def param(self):
        return np.array([model.coef_ for key, model in self.final_models.items()]).flatten()

    @property
    def param_cov(self):
        return self._cov

    @property
    def param_stderr(self):
        return np.sqrt(np.diag(self._cov) / self._n_train_units)

    def param_interval(self, alpha=.05):
        return np.array([(scipy.stats.norm.ppf(alpha / 2, loc=param, scale=std),
                          scipy.stats.norm.ppf(1 - alpha / 2, loc=param, scale=std)) if std > 0 else (param, param)
                         for param, std in zip(self.param, self.param_stderr)])

    def policy_effect(self, tau):
        return np.dot(self.param, tau[::-1].flatten())

    def policy_effect_var(self, tau):
        return (tau[::-1].flatten().reshape(1, -1) @ self.param_cov @ tau[::-1].flatten().reshape(-1, 1))[0, 0]

    def policy_effect_stderr(self, tau):
        return np.sqrt(self.policy_effect_var(tau) / self._n_train_units)

    def policy_effect_interval(self, tau, alpha=0.05):
        param = self.policy_effect(tau)
        std = self.policy_effect_stderr(tau)
        if std == 0:
            return (param, param)
        return (scipy.stats.norm.ppf(alpha / 2, loc=param, scale=std),
                scipy.stats.norm.ppf(1 - alpha / 2, loc=param, scale=std))

    def adaptive_policy_effect(self, X, groups, policy_gen, alpha=.05):
        """ Assumes that the policy is adaptive only on exogenous states that
        are not affected by the treatmnet.
        """
        u_periods = np.unique(np.bincount(groups.astype(int)))
        if len(u_periods) > 1 or u_periods[0] != self._n_train_periods:
            raise AttributeError("Invalid period lengths.")
        n_periods = u_periods[0]

        tau = np.zeros(
            (X.shape[0] // n_periods, n_periods, self._n_treatments))
        for period in range(n_periods):
            inds = (np.arange(X.shape[0]) % n_periods == period)
            if period == 0:
                tau_pre = np.zeros((len(inds), self._n_treatments))
            else:
                tau_pre = tau[:, period - 1, :]
            tau[:, period, :] = np.array([policy_gen(t_pre, x, period)
                                          for t_pre, x in zip(tau_pre, X[inds])])
        mean_tau = np.mean(tau, axis=0)
        param = self.policy_effect(mean_tau)
        std = self.policy_effect_stderr(mean_tau)
        if std == 0:
            return param, (param, param)
        return param, (scipy.stats.norm.ppf(alpha / 2, loc=param, scale=std),
                       scipy.stats.norm.ppf(1 - alpha / 2, loc=param, scale=std))
