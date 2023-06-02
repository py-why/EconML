import unittest

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from drtester import DRtester


class TestDRTester(unittest.TestCase):

    @staticmethod
    def _get_data(num_treatments=1):
        np.random.seed(576)

        N = 20000  # number of units
        K = 5  # number of covariates

        # Generate random Xs
        X_mu = np.zeros(5)  # Means of Xs
        # Random covariance matrix of Xs
        X_sig = np.diag(np.random.rand(5))
        X = st.multivariate_normal(X_mu, X_sig).rvs(N)

        # Effect of Xs on outcome
        X_beta = np.random.uniform(0, 5, K)
        # Effect of treatment on outcomes
        D_beta = np.arange(num_treatments + 1)
        # Effect of treatment on outcome conditional on X1
        DX1_beta = np.array([0] * num_treatments + [3])

        # Generate treatments based on X and random noise
        beta_treat = np.random.uniform(-1, 1, (num_treatments + 1, K))
        D1 = np.zeros((N, num_treatments + 1))
        for k in range(num_treatments + 1):
            D1[:, k] = X @ beta_treat[k, :] + np.random.gumbel(0, 1, N)
        D = np.array([np.where(D1[i, :] == np.max(D1[i, :]))[0][0] for i in range(N)])
        D_dum = pd.get_dummies(D)

        # Generate Y (based on X, D, and random noise)
        Y_sig = 1  # Variance of random outcome noise
        Y = X @ X_beta + (D_dum @ D_beta) + X[:, 1] * (D_dum @ DX1_beta) + np.random.normal(0, Y_sig, N)
        Y = Y.to_numpy()

        train_prop = .5
        train_N = np.ceil(train_prop * N)
        ind = np.array(range(N))
        train_ind = np.random.choice(N, int(train_N), replace=False)
        val_ind = ind[~np.isin(ind, train_ind)]

        Xtrain, Dtrain, Ytrain = X[train_ind], D[train_ind], Y[train_ind]
        Xval, Dval, Yval = X[val_ind], D[val_ind], Y[val_ind]

        return Xtrain, Dtrain, Ytrain, Xval, Dval, Yval

    def test_multi(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=2)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier()
        reg_y = GradientBoostingRegressor()
        reg_cate = GradientBoostingRegressor()

        # test the DR outcome difference
        my_dr_tester = DRtester(reg_y, reg_t).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )
        dr_outcomes = my_dr_tester.dr_val

        ates = dr_outcomes.mean(axis=0)
        for k in range(dr_outcomes.shape[1]):
            ate_errs = np.sqrt(((dr_outcomes[:, k] - ates[k]) ** 2).sum() / \
                      (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))

            self.assertLess(abs(ates[k] - (k + 1)), 2 * ate_errs)

        Ztrain = Xtrain[:, 1]
        Zval = Xval[:, 1]

        my_dr_tester.fit_cate(reg_cate, Zval, Ztrain)

        my_dr_tester = my_dr_tester.evaluate_all()

        for k in range(3):
            if k == 0:
                with self.assertRaises(Exception) as exc:
                    my_dr_tester.plot_cal(k)
                self.assertTrue(str(exc.exception) == 'Plotting only supported for treated units (not controls)')
            else:
                self.assertTrue(my_dr_tester.plot_cal(k) is not None)

        self.assertGreater(my_dr_tester.df_res.blp_pval.values[0], 0.1)  # no heterogeneity
        self.assertLess(my_dr_tester.df_res.blp_pval.values[1], 0.05)  # heterogeneity

        self.assertLess(my_dr_tester.df_res.cal_r_squared.values[0], 0.2)  # poor R2
        self.assertGreater(my_dr_tester.df_res.cal_r_squared.values[1], 0.5)  # good R2

        self.assertLess(my_dr_tester.df_res.qini_pval.values[1], my_dr_tester.df_res.qini_pval.values[0])

    def test_binary(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier()
        reg_y = GradientBoostingRegressor()
        reg_cate = GradientBoostingRegressor()

        # test the DR outcome difference
        my_dr_tester = DRtester(reg_y, reg_t).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )
        dr_outcomes = my_dr_tester.dr_val

        ate = dr_outcomes.mean(axis=0)
        ate_err = np.sqrt(((dr_outcomes - ate) ** 2).sum() / \
                               (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))
        truth = 1
        self.assertLess(abs(ate - truth), 2 * ate_err)

        Ztrain = Xtrain[:, 1]
        Zval = Xval[:, 1]

        my_dr_tester = my_dr_tester.evaluate_all(reg_cate, Zval, Ztrain)

        for k in range(2):
            if k == 0:
                with self.assertRaises(Exception) as exc:
                    my_dr_tester.plot_cal(k)
                self.assertTrue(str(exc.exception) == 'Plotting only supported for treated units (not controls)')
            else:
                self.assertTrue(my_dr_tester.plot_cal(k) is not None)
        self.assertLess(my_dr_tester.df_res.blp_pval.values[0], 0.05)  # heterogeneity
        self.assertGreater(my_dr_tester.df_res.cal_r_squared.values[0], 0.5)  # good R2
        self.assertLess(my_dr_tester.df_res.qini_pval.values[0], 0.05)  # heterogeneity

    def test_nuisance_val_fit(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier()
        reg_y = GradientBoostingRegressor()
        reg_cate = GradientBoostingRegressor()

        # test the DR outcome difference
        my_dr_tester = DRtester(reg_y, reg_t).fit_nuisance(
            Xval, Dval, Yval
        )
        dr_outcomes = my_dr_tester.dr_val

        ate = dr_outcomes.mean(axis=0)
        ate_err = np.sqrt(((dr_outcomes - ate) ** 2).sum() / \
                               (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))
        truth = 1
        self.assertLess(abs(ate - truth), 2 * ate_err)

        Zval = Xval[:, 1]
        Ztrain = Xtrain[:, 1]

        with self.assertRaises(Exception) as exc:
            my_dr_tester.fit_cate(reg_cate, Zval, Ztrain)
        self.assertTrue(
            str(exc.exception) == "Nuisance models cross-fit on validation sample but Ztrain is specified"
        )

        # use evaluate_blp to fit on validation only
        my_dr_tester = my_dr_tester.evaluate_blp(reg_cate, Zval)

        self.assertLess(my_dr_tester.blp_res.blp_pval.values[0], 0.05)  # heterogeneity

        for func in [my_dr_tester.evaluate_cal, my_dr_tester.evaluate_qini, my_dr_tester.evaluate_all]:
            for kwargs in [{}, {'reg_cate': reg_cate}, {'Zval': Zval}, {'reg_cate': reg_cate, 'Zval': Zval}]:
                with self.assertRaises(Exception) as exc:
                    func(kwargs)
                if func.__name__ == 'evaluate_cal':
                    self.assertTrue(
                        str(exc.exception) == "Must fit nuisance models on training sample data to use calibration test"
                    )
                else:
                    self.assertTrue(
                        str(exc.exception) == "CATE not fitted on training data - must provide CATE model and both Zval, Ztrain"
                    )

    def test_cate_val_fit(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier()
        reg_y = GradientBoostingRegressor()
        reg_cate = GradientBoostingRegressor()

        # test the DR outcome difference
        my_dr_tester = DRtester(reg_y, reg_t).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )

        Zval = Xval[:, 1]

        with self.assertRaises(Exception) as exc:
            my_dr_tester.fit_cate(reg_cate, Zval)
        self.assertTrue(str(exc.exception) == 'Nuisance models fit on training sample but Ztrain not specified')

    def test_exceptions(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        reg_cate = GradientBoostingRegressor()
        reg_t = RandomForestClassifier()
        reg_y = GradientBoostingRegressor()

        my_dr_tester = DRtester(reg_y, reg_t)

        # fit nothing
        for func in [my_dr_tester.evaluate_blp, my_dr_tester.evaluate_cal, my_dr_tester.evaluate_qini]:
            for kwargs in [{}, {'reg_cate': reg_cate}]:
                with self.assertRaises(Exception) as exc:
                    func(kwargs)
                if func.__name__ == 'evaluate_cal':
                    self.assertTrue(
                        str(exc.exception) == "Must fit nuisance models on training sample data to use calibration test"
                    )
                else:
                    self.assertTrue(str(exc.exception) == "Must fit nuisances before evaluating")

        my_dr_tester = my_dr_tester.fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )

        # fit nuisances, but not CATE
        for func in [my_dr_tester.evaluate_blp, my_dr_tester.evaluate_cal, my_dr_tester.evaluate_qini]:
            for kwargs in [{}, {'reg_cate': reg_cate}, {'Zval': Xval[:, 1]}]:
                with self.assertRaises(Exception) as exc:
                    func(kwargs)
                if func.__name__ in ['evaluate_cal', 'evaluate_qini']:
                    self.assertTrue(
                        str(exc.exception) == 'CATE not fitted on training data - must provide CATE model and both Zval, Ztrain'
                    )
                else:
                    self.assertTrue(str(exc.exception) == "CATE not yet fitted - must provide Zval and CATE estimator")
