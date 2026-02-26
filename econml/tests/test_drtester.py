import unittest

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

from econml.validate.drtester import DRTester
from econml.dml import DML


class TestDRTester(unittest.TestCase):

    @staticmethod
    def _get_data(num_treatments=1, use_sample_weights=False):
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

        # sample weights
        if use_sample_weights:
            sample_weights = np.random.randint(1, 1000, size=N)

        Xtrain, Dtrain, Ytrain = X[train_ind], D[train_ind], Y[train_ind]
        Xval, Dval, Yval = X[val_ind], D[val_ind], Y[val_ind]

        if use_sample_weights:
            sampleweightstrain = sample_weights[train_ind]
            sampleweightsval = sample_weights[val_ind]

        if use_sample_weights:
            return Xtrain, Dtrain, Ytrain, Xval, Dval, Yval, sampleweightstrain, sampleweightsval
        else:
            return Xtrain, Dtrain, Ytrain, Xval, Dval, Yval

    def test_multi(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=2)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier(random_state=0)
        reg_y = GradientBoostingRegressor(random_state=0)

        cate = DML(
            model_y=reg_y,
            model_t=reg_t,
            model_final=reg_y,
            discrete_treatment=True
        ).fit(Y=Ytrain, T=Dtrain, X=Xtrain)

        # test the DR outcome difference
        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        ).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )
        dr_outcomes = my_dr_tester.dr_val_

        ates = dr_outcomes.mean(axis=0)
        for k in range(dr_outcomes.shape[1]):
            ate_errs = np.sqrt(((dr_outcomes[:, k] - ates[k]) ** 2).sum() /
                               (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))

            self.assertLess(abs(ates[k] - (k + 1)), 2 * ate_errs)

        res = my_dr_tester.evaluate_all(Xval, Xtrain)
        res_df = res.summary()

        for k in range(4):
            if k in [0, 3]:
                self.assertRaises(ValueError, res.plot_cal, k)
                self.assertRaises(ValueError, res.plot_qini, k)
                self.assertRaises(ValueError, res.plot_toc, k)
            else:  # real treatments, k = 1 or 2
                self.assertTrue(res.plot_cal(k) is not None)
                self.assertTrue(res.plot_qini(k) is not None)
                self.assertTrue(res.plot_toc(k) is not None)

        self.assertGreater(res_df.blp_pval.values[0], 0.1)  # no heterogeneity
        self.assertLess(res_df.blp_pval.values[1], 0.05)  # heterogeneity

        self.assertLess(res_df.cal_r_squared.values[0], 0)  # poor R2
        self.assertGreater(res_df.cal_r_squared.values[1], 0)  # good R2

        self.assertLess(res_df.qini_pval.values[1], res_df.qini_pval.values[0])
        self.assertLess(res_df.autoc_pval.values[1], res_df.autoc_pval.values[0])

    def test_binary(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier(random_state=0)
        reg_y = GradientBoostingRegressor(random_state=0)

        cate = DML(
            model_y=reg_y,
            model_t=reg_t,
            model_final=reg_y,
            discrete_treatment=True
        ).fit(Y=Ytrain, T=Dtrain, X=Xtrain)

        # test the DR outcome difference
        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        ).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )
        dr_outcomes = my_dr_tester.dr_val_

        ate = dr_outcomes.mean(axis=0)
        ate_err = np.sqrt(((dr_outcomes - ate) ** 2).sum() /
                          (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))
        truth = 1
        self.assertLess(abs(ate - truth), 2 * ate_err)

        res = my_dr_tester.evaluate_all(Xval, Xtrain)
        res_df = res.summary()

        for k in range(3):
            if k in [0, 2]:
                self.assertRaises(ValueError, res.plot_cal, k)
                self.assertRaises(ValueError, res.plot_qini, k)
                self.assertRaises(ValueError, res.plot_toc, k)
            else:  # real treatment, k = 1
                self.assertTrue(res.plot_cal(k) is not None)
                self.assertTrue(res.plot_qini(k, 'ucb2') is not None)
                self.assertTrue(res.plot_toc(k, 'ucb1') is not None)

        self.assertLess(res_df.blp_pval.values[0], 0.05)  # heterogeneity
        self.assertGreater(res_df.cal_r_squared.values[0], 0)  # good R2
        self.assertLess(res_df.qini_pval.values[0], 0.05)  # heterogeneity
        self.assertLess(res_df.autoc_pval.values[0], 0.05)  # heterogeneity

    def test_nuisance_val_fit(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier(random_state=0)
        reg_y = GradientBoostingRegressor(random_state=0)

        cate = DML(
            model_y=reg_y,
            model_t=reg_t,
            model_final=reg_y,
            discrete_treatment=True
        ).fit(Y=Ytrain, T=Dtrain, X=Xtrain)

        # test the DR outcome difference
        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        ).fit_nuisance(Xval, Dval, Yval)

        dr_outcomes = my_dr_tester.dr_val_

        ate = dr_outcomes.mean(axis=0)
        ate_err = np.sqrt(((dr_outcomes - ate) ** 2).sum() /
                          (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))
        truth = 1
        self.assertLess(abs(ate - truth), 2 * ate_err)

        # use evaluate_blp to fit on validation only
        blp_res = my_dr_tester.evaluate_blp(Xval)

        self.assertLess(blp_res.pvals[0], 0.05)  # heterogeneity

        for kwargs in [{}, {'Xval': Xval}]:
            with self.assertRaises(Exception) as exc:
                my_dr_tester.evaluate_cal(kwargs)
            self.assertEqual(
                str(exc.exception), "Must fit nuisance models on training sample data to use calibration test"
            )

    def test_exceptions(self):
        Xtrain, Dtrain, Ytrain, Xval, Dval, Yval = self._get_data(num_treatments=1)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier(random_state=0)
        reg_y = GradientBoostingRegressor(random_state=0)

        cate = DML(
            model_y=reg_y,
            model_t=reg_t,
            model_final=reg_y,
            discrete_treatment=True
        ).fit(Y=Ytrain, T=Dtrain, X=Xtrain)

        # test the DR outcome difference
        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        )

        # fit nothing
        for func in [my_dr_tester.evaluate_blp, my_dr_tester.evaluate_cal, my_dr_tester.evaluate_uplift]:
            with self.assertRaises(Exception) as exc:
                func()
            if func.__name__ == 'evaluate_cal':
                self.assertEqual(
                    str(exc.exception), "Must fit nuisance models on training sample data to use calibration test"
                )
            else:
                self.assertEqual(str(exc.exception), "Must fit nuisances before evaluating")

        my_dr_tester = my_dr_tester.fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )

        for func in [
            my_dr_tester.evaluate_blp,
            my_dr_tester.evaluate_cal,
            my_dr_tester.evaluate_uplift,
            my_dr_tester.evaluate_all
        ]:
            with self.assertRaises(Exception) as exc:
                func()
            if func.__name__ == 'evaluate_blp':
                self.assertEqual(
                    str(exc.exception), "CATE predictions not yet calculated - must provide Xval"
                )
            else:
                self.assertEqual(str(exc.exception),
                                 "CATE predictions not yet calculated - must provide both Xval, Xtrain")

        for func in [
            my_dr_tester.evaluate_cal,
            my_dr_tester.evaluate_uplift,
            my_dr_tester.evaluate_all
        ]:
            with self.assertRaises(Exception) as exc:
                func(Xval=Xval)
            self.assertEqual(
                str(exc.exception), "CATE predictions not yet calculated - must provide both Xval, Xtrain")

        cal_res = my_dr_tester.evaluate_cal(Xval, Xtrain)
        self.assertGreater(cal_res.cal_r_squared[0], 0)  # good R2

        with self.assertRaises(Exception) as exc:
            my_dr_tester.evaluate_uplift(metric='blah')
        self.assertEqual(
            str(exc.exception), "Unsupported metric 'blah' - must be one of ['toc', 'qini']"
        )

        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        ).fit_nuisance(
            Xval, Dval, Yval, Xtrain, Dtrain, Ytrain
        )
        qini_res = my_dr_tester.evaluate_uplift(Xval, Xtrain)
        self.assertLess(qini_res.pvals[0], 0.05)

        with self.assertRaises(Exception) as exc:
            qini_res.plot_uplift(tmt=1, err_type='blah')
        self.assertEqual(
            str(exc.exception), "Invalid error type 'blah'; must be one of [None, 'ucb2', 'ucb1']"
        )

        autoc_res = my_dr_tester.evaluate_uplift(Xval, Xtrain, metric='toc')
        self.assertLess(autoc_res.pvals[0], 0.05)

    def test_multi_with_weights(self):
        (Xtrain,
         Dtrain,
         Ytrain,
         Xval,
         Dval,
         Yval,
         sampleweightstrain,
         sampleweightsval) = self._get_data(num_treatments=2, use_sample_weights=True)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier(random_state=0)
        reg_y = GradientBoostingRegressor(random_state=0)

        cate = DML(
            model_y=reg_y,
            model_t=reg_t,
            model_final=reg_y,
            discrete_treatment=True
        ).fit(Y=Ytrain,
              T=Dtrain,
              X=Xtrain,
              sample_weight=sampleweightstrain)

        # test the DR outcome difference
        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        ).fit_nuisance(
            Xval,
            Dval,
            Yval,
            Xtrain,
            Dtrain,
            Ytrain,
            sampleweightval=sampleweightsval,
            sampleweighttrain=sampleweightstrain
        )
        dr_outcomes = my_dr_tester.dr_val_

        ates = np.average(dr_outcomes, axis=0, weights=sampleweightsval)
        for k in range(dr_outcomes.shape[1]):
            ate_errs = np.sqrt(((dr_outcomes[:, k] - ates[k]) ** 2).sum() /
                               (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))

            self.assertLess(abs(ates[k] - (k + 1)), 2 * ate_errs)

        res = my_dr_tester.evaluate_all(Xval, Xtrain)
        res_df = res.summary()

        for k in range(4):
            if k in [0, 3]:
                self.assertRaises(ValueError, res.plot_cal, k)
                self.assertRaises(ValueError, res.plot_qini, k)
                self.assertRaises(ValueError, res.plot_toc, k)
            else:  # real treatments, k = 1 or 2
                self.assertTrue(res.plot_cal(k) is not None)
                self.assertTrue(res.plot_qini(k) is not None)
                self.assertTrue(res.plot_toc(k) is not None)

        self.assertGreater(res_df.blp_pval.values[0], 0.1)  # no heterogeneity
        self.assertLess(res_df.blp_pval.values[1], 0.05)  # heterogeneity

        self.assertLess(res_df.cal_r_squared.values[0], 0)  # poor R2
        self.assertGreater(res_df.cal_r_squared.values[1], 0)  # good R2

        self.assertLess(res_df.qini_pval.values[1], res_df.qini_pval.values[0])
        self.assertLess(res_df.autoc_pval.values[1], res_df.autoc_pval.values[0])

    def test_binary_with_weights(self):
        (Xtrain,
         Dtrain,
         Ytrain,
         Xval,
         Dval,
         Yval,
         sampleweightstrain,
         sampleweightsval) = self._get_data(num_treatments=1, use_sample_weights=True)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier(random_state=0)
        reg_y = GradientBoostingRegressor(random_state=0)

        cate = DML(
            model_y=reg_y,
            model_t=reg_t,
            model_final=reg_y,
            discrete_treatment=True
        ).fit(Y=Ytrain,
              T=Dtrain,
              X=Xtrain,
              sample_weight=sampleweightstrain)

        # test the DR outcome difference
        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        ).fit_nuisance(
            Xval,
            Dval,
            Yval,
            Xtrain,
            Dtrain,
            Ytrain,
            sampleweightval=sampleweightsval,
            sampleweighttrain=sampleweightstrain
        )
        dr_outcomes = my_dr_tester.dr_val_

        ate = np.average(dr_outcomes, axis=0, weights=sampleweightsval)
        ate_err = np.sqrt(((dr_outcomes - ate) ** 2).sum() /
                          (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))
        truth = 1
        self.assertLess(abs(ate - truth), 2 * ate_err)

        res = my_dr_tester.evaluate_all(Xval, Xtrain)
        res_df = res.summary()

        for k in range(3):
            if k in [0, 2]:
                self.assertRaises(ValueError, res.plot_cal, k)
                self.assertRaises(ValueError, res.plot_qini, k)
                self.assertRaises(ValueError, res.plot_toc, k)
            else:  # real treatment, k = 1
                self.assertTrue(res.plot_cal(k) is not None)
                self.assertTrue(res.plot_qini(k, 'ucb2') is not None)
                self.assertTrue(res.plot_toc(k, 'ucb1') is not None)

        self.assertLess(res_df.blp_pval.values[0], 0.05)  # heterogeneity
        self.assertGreater(res_df.cal_r_squared.values[0], 0)  # good R2
        self.assertLess(res_df.qini_pval.values[0], 0.05)  # heterogeneity
        self.assertLess(res_df.autoc_pval.values[0], 0.05)  # heterogeneity

    def test_nuisance_val_fit_with_weights(self):
        (Xtrain,
         Dtrain,
         Ytrain,
         Xval,
         Dval,
         Yval,
         sampleweightstrain,
         sampleweightsval) = self._get_data(num_treatments=1, use_sample_weights=True)

        # Simple classifier and regressor for propensity, outcome, and cate
        reg_t = RandomForestClassifier(random_state=0)
        reg_y = GradientBoostingRegressor(random_state=0)

        cate = DML(
            model_y=reg_y,
            model_t=reg_t,
            model_final=reg_y,
            discrete_treatment=True
        ).fit(Y=Ytrain,
              T=Dtrain,
              X=Xtrain,
              sample_weight=sampleweightstrain)

        # test the DR outcome difference
        my_dr_tester = DRTester(
            model_regression=reg_y,
            model_propensity=reg_t,
            cate=cate
        ).fit_nuisance(Xval,
                       Dval,
                       Yval,
                       sampleweightval=sampleweightsval)

        dr_outcomes = my_dr_tester.dr_val_

        ate = np.average(dr_outcomes, axis=0, weights=sampleweightsval)
        ate_err = np.sqrt(((dr_outcomes - ate) ** 2).sum() /
                          (dr_outcomes.shape[0] * (dr_outcomes.shape[0] - 1)))
        truth = 1
        self.assertLess(abs(ate - truth), 2 * ate_err)

        # use evaluate_blp to fit on validation only
        blp_res = my_dr_tester.evaluate_blp(Xval)

        self.assertLess(blp_res.pvals[0], 0.05)  # heterogeneity

        for kwargs in [{}, {'Xval': Xval}]:
            with self.assertRaises(Exception) as exc:
                my_dr_tester.evaluate_cal(kwargs)
            self.assertEqual(
                str(exc.exception), "Must fit nuisance models on training sample data to use calibration test"
            )
