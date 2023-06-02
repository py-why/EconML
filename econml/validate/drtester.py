from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from statsmodels.api import OLS
from statsmodels.tools import add_constant


class DRtester:

    """
    Validation tests for CATE models. Includes the best linear predictor (BLP) test as in Chernozhukov et al. (2022),
    the calibration test in Dwivedi et al. (2020), and the QINI coefficient as in Radcliffe (2007).

    **Best Linear Predictor (BLP)**

    Runs ordinary least squares (OLS) of doubly robust (DR) outcomes on DR outcome predictions from the CATE model
    (and a constant). If the CATE model captures true heterogeneity, then the OLS estimate on the CATE predictions
    should be positive and significantly different than 0.

    **Calibration**

    First, units are binned based on out-of-sample defined quantiles of the CATE predictions (s(Z)). Within each bin
    (k), the absolute difference between the mean CATE prediction and DR outcome is calculated, along with the
    absolute difference in the mean CATE prediction and the overall ATE. These measures are then summed across bins,
    weighted by a probability a unit is in each bin.

    Cal_G = \sum_k \pi(k) |E[s(Z) | k] - E[Y^{DR | k}|

    Cal_O = \sum_k \pi(k) |E[s(Z) | k] - E[Y^{DR}|

    The calibration r-squared is then defined as


    \mathcal{R^2}_C = 1 - \frac{Cal_G}{Cal_O}

    The calibration r-squared metric is similar to thestandard R-square score in that it can take any value
    $\leq 1$, with scores closer to 1 indicating a better calibrated CATE model.

    **QINI**


    Parameters
    ----------
    reg_outcome: estimator
        Nuisance model estimator used to fit the outcome to features. Must be able to implement `fit' and `predict'
        methods

    reg_t: estimator
        Nuisance model estimator used to fit the treatment assignment to features. Must be able to implement `fit'
        method and either `predict' (in the case of binary treatment) or `predict_proba' methods (in the case of
        multiple categorical treatments).

    n_splits: integer, default 5
        Number of splits used to generate cross-validated predictions

    References
    ----------


    [Chernozhukov2022] V. Chernozhukov et al.
    Generic Machine Learning Inference on Heterogeneous Treatment Effects in Randomized Experiments
    arXiv preprint arXiv:1712.04802, 2022.
    `<https://arxiv.org/abs/1712.04802>`_

    [Dwivedi2020] R. Dwivedi et al.
    Stable Discovery of Interpretable Subgroups via Calibration in Causal Studies
    arXiv preprint 	arXiv:2008.10109, 2020.
    `<https://arxiv.org/abs/2008.10109>`_


    [Radcliffe2007] N. Radcliffe
    Using control groups to target on predicted lift: Building and assessing uplift model.
    Direct Marketing Analytics Journal (2007), pages 14–21.
    """

    def __init__(
        self,
        reg_outcome,
        reg_t,
        n_splits: int = 5
    ):
        self.reg_outcome = reg_outcome
        self.reg_t = reg_t
        self.n_splits = n_splits
        self.dr_train = None
        self.cate_preds_train = None
        self.cate_preds_val = None
        self.dr_val = None

    def fit_nuisance(
        self,
        Xval: np.array,
        Dval: np.array,
        yval: np.array,
        Xtrain: np.array = None,
        Dtrain: np.array = None,
        ytrain: np.array = None,
    ):

        """

        Generates nuisance predictions and calculates doubly robust (DR) outcomes either by (1) cross-fitting in the
        validation sample, or (2) fitting in the training sample and applying to the validation sample. If Xtrain,
        Dtrain, and ytrain are all not None, then option (2) will be implemented, otherwise, option (1) will be
        implemented. In order to use the `evaluate_cal' method then Xtrain, Dtrain, and ytrain must all be specified.

        Parameters
        ----------
        Xval: (n_val x k) matrix or vector of length n
            Features used in nuisance models for validation sample
        Dval: vector of length n_val
            Treatment assignment of validation sample. Control status must be minimum value. It is recommended to have
            the control status be equal to 0, and all other treatments integers starting at 1.
        yval: vector of length n_val
            Outcomes for the validation sample
        Xtrain: (n_train x k) matrix or vector of length n, default ``None``
            Features used in nuisance models for training sample
        Dtrain: vector of length n_train, default ``None''
            Treatment assignment of training sample. Control status must be minimum value. It is recommended to have
            the control status be equal to 0, and all other treatments integers starting at 1.
        ytrain: vector of length n_train, defaul ``None``
            Outcomes for the training sample

        Returns
        ------
        self, with added attributes for the validation treatments (Dval), treatment values (tmts),
        number of treatments excluding control (n_treat), boolean flag for whether training data is provided
        (fit_on_train), doubly robust outcome values for the validation set (dr_val), and the DR ATE value (ate_val).
        If training data is provided, also adds attributes for the doubly robust outcomes for the training
        set (dr_train) and the training treatments (Dtrain)

        """

        self.Dval = Dval

        # Unique treatments (ordered, includes control)
        self.tmts = np.sort(np.unique(Dval))

        # Number of treatments (excluding control)
        self.n_treat = len(self.tmts) - 1

        # Indicator for whether
        self.fit_on_train = (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None)

        if self.fit_on_train:
            # Get DR outcomes in training sample
            reg_preds_train, prop_preds_train = self.fit_nuisance_cv(Xtrain, Dtrain, ytrain)
            self.dr_train = self.calculate_dr_outcomes(Dtrain, ytrain, reg_preds_train, prop_preds_train)

            # Get DR outcomes in validation sample
            reg_preds_val, prop_preds_val = self.fit_nuisance_train(Xtrain, Dtrain, ytrain, Xval)
            self.dr_val = self.calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)

            self.Dtrain = Dtrain

        else:
            # Get DR outcomes in validation sample
            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val = self.calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)

        # Calculate ATE in the validation sample
        self.ate_val = self.dr_val.mean(axis=0)

        return self

    def fit_cate(
        self,
        reg_cate,
        Zval: np.array,
        Ztrain: np.array = None
    ):

        """
        Fits CATE model and generates predictions. If Ztrain is None, then the predictions are generated using k-folds
        cross-validation on the validation set. If Ztrain is specified, then the CATE is fit on the training sample
        (where the DR outcomes were generated using k-folds CV), and then applied to the validation sample.

        Parameters
        ----------
        reg_cate: estimator
            CATE model. Must be able to implement `fit' and `predict' methods
        Zval: (n_val x n_treatment) matrix
            Validation set features to be used to predict (and potentially fit) DR outcomes in CATE model
        Ztrain (n_train x n_treatment) matrix, defaul ``None``
            Training set features used to fit CATE model

        Returns
        -------
        None, but adds attribute cate_preds_val for predicted CATE values on the validation set and, if training
        data is provided, attribute cate_preds_train for predicted CATE values on the training set

        """

        if (Ztrain is None) and self.fit_on_train:
            raise Exception("Nuisance models fit on training sample but Ztrain not specified")

        if (Ztrain is not None) and (not self.fit_on_train):
            raise Exception("Nuisance models cross-fit on validation sample but Ztrain is specified")

        if Ztrain is not None:
            self.cate_preds_train = self.fit_cate_cv(reg_cate, Ztrain, self.Dtrain, self.dr_train, self.n_splits)
            self.cate_preds_val = self.fit_cate_train(reg_cate, Ztrain, Zval)
        else:
            self.cate_preds_val = self.fit_cate_cv(reg_cate, Zval, self.Dval, self.dr_val, self.n_splits)

    def evaluate_cal(
        self,
        reg_cate=None,
        Zval: np.array = None,
        Ztrain: np.array = None,
        n_groups: int = 4
    ):

        """
        Implements calibration test as in [Dwivedi2020]

        Parameters
        ----------
        reg_cate: estimator, default ``None``
            CATE model. Must be able to implement `fit' and `predict' methods. If not specified, then fit_cate() must
            already have been implemented
        Zval: (n_cal x n_treatment) matrix, default ``None``
            Validation sample features for CATE model. If not specified, then `fit_cate' method must already have been
            implemented
        Ztrain: (n_train x n_treatment) matrix, default ``None``
            Training sample features for CATE model. If not specified, then `fit cate' method must already have been
            implemented (with Ztrain specified)
        n_groups: integer, default 4
            Number of quantile-based groups used to calculate calibration score.

        Returns
        -------
        self, with added attribute cal_r_squared showing the R^2 value of the calibration test and dataframe df_plot
        containing relevant results to plot calibration test gates

        """
        if self.dr_train is None:
            raise Exception("Must fit nuisance models on training sample data to use calibration test")

        # if CATE is given explicitly or has not been fitted at all previously, fit it now
        if (self.cate_preds_train is None) or (self.cate_preds_val is None) or (reg_cate is not None):
            if (Zval is None) or (Ztrain is None) or (reg_cate is None):
                raise Exception('CATE not fitted on training data - must provide CATE model and both Zval, Ztrain')
            self.fit_cate(reg_cate, Zval, Ztrain)

        self.cal_r_squared = np.zeros(self.n_treat)
        self.df_plot = pd.DataFrame()
        for k in range(self.n_treat):

            cuts = np.quantile(self.cate_preds_train[:, k], np.linspace(0, 1, n_groups + 1))
            probs = np.zeros(n_groups)
            g_cate = np.zeros(n_groups)
            se_g_cate = np.zeros(n_groups)
            gate = np.zeros(n_groups)
            se_gate = np.zeros(n_groups)
            for i in range(n_groups):
                # Assign units in validation set to groups
                ind = (self.cate_preds_val[:, k] >= cuts[i]) & (self.cate_preds_val[:, k] <= cuts[i + 1])
                # Proportion of validations set in group
                probs[i] = np.mean(ind)
                # Group average treatment effect (GATE) -- average of DR outcomes in group
                gate[i] = np.mean(self.dr_val[ind, k])
                se_gate[i] = np.std(self.dr_val[ind, k]) / np.sqrt(np.sum(ind))
                # Average of CATE predictions in group
                g_cate[i] = np.mean(self.cate_preds_val[ind, k])
                se_g_cate[i] = np.std(self.cate_preds_val[ind, k]) / np.sqrt(np.sum(ind))

            # Calculate group calibration score
            cal_score_g = np.sum(abs(gate - g_cate) * probs)
            # Calculate overall calibration score
            cal_score_o = np.sum(abs(gate - self.ate_val[k]) * probs)
            # Calculate R-square calibration score
            self.cal_r_squared[k] = 1 - (cal_score_g / cal_score_o)

            df_plot1 = pd.DataFrame({'ind': np.array(range(n_groups)),
                                     'gate': gate, 'se_gate': se_gate,
                                    'g_cate': g_cate, 'se_g_cate': se_g_cate})
            df_plot1['tmt'] = self.tmts[k + 1]
            self.df_plot = pd.concat((self.df_plot, df_plot1))

        return self

    def plot_cal(self, tmt: int):

        """
        Plots group average treatment effects (GATEs) and predicted GATEs by quantile-based group in validation sample.

        Parameters
        ----------
        tmt: integer
            Treatment level to plot

        Returns
        -------
        fig: matplotlib
            Plot with predicted GATE on x-axis and GATE (and 95% CI) on y-axis
        """
        if tmt == 0:
            raise Exception('Plotting only supported for treated units (not controls)')

        df = self.df_plot
        df = df[df.tmt == tmt].copy()
        rsq = round(self.cal_r_squared[np.where(self.tmts == tmt)[0][0] - 1], 3)
        df['95_err'] = 1.96 * df['se_gate']
        fig = df.plot(
            kind='scatter',
            x='g_cate',
            y='gate',
            yerr='95_err',
            xlabel = 'Group Mean CATE',
            ylabel = 'GATE',
            title=f"Treatment = {tmt}, Calibration R^2 = {rsq}"
        ).get_figure()

        return fig

    def evaluate_blp(
        self,
        reg_cate=None,
        Zval: np.array = None,
        Ztrain: np.array = None
    ):
        """
        Implements the best linear predictor (BLP) test as in [Chernozhukov2022]. `fit_nusiance' method must already
        be implemented.

        Parameters
        ----------
        reg_cate: estimator, default ``None''
            CATE model. Must be able to implement `fit' and `predict' methods. If not specified, then fit_cate() must
            already have been implemented
        Zval: (n_val x k) matrix, default ``None''
            Validation sample features for CATE model. If not specified, then `fit_cate' method must already have been
            implemented
        Ztrain: (n_train x k) matrix, default ``None''
            Training sample features for CATE model. If specified, then CATE is fitted on training sample and applied
            to Zval. If specified, then Xtrain, Dtrain, Ytrain must have been specified in `fit_nuisance' method (and
            vice-versa)

        Returns
        -------
        self, with added dataframe blp_res showing the results of the BLP test

        """

        if self.dr_val is None:
            raise Exception("Must fit nuisances before evaluating")

        # if CATE is given explicitly or has not been fitted at all previously, fit it now
        if (self.cate_preds_val is None) or (reg_cate is not None):
            if (Zval is None) or (reg_cate is None):  # need at least Zval and a CATE estimator to fit
                raise Exception('CATE not yet fitted - must provide Zval and CATE estimator')
            self.fit_cate(reg_cate, Zval, Ztrain)

        if self.n_treat == 1:  # binary treatment
            reg = OLS(self.dr_val, add_constant(self.cate_preds_val)).fit()
            params = [reg.params[1]]
            errs = [reg.bse[1]]
            pvals = [reg.pvalues[1]]
        else:  # categorical treatment
            params = []
            errs = []
            pvals = []
            for k in range(self.n_treat):  # run a separate regression for each
                reg = OLS(self.dr_val[:, k], add_constant(self.cate_preds_val[:, k])).fit(cov_type = 'HC1')
                params.append(reg.params[1])
                errs.append(reg.bse[1])
                pvals.append(reg.pvalues[1])

        self.blp_res = pd.DataFrame(
            {'treatment': self.tmts[1:], 'blp_est': params, 'blp_se': errs, 'blp_pval': pvals}
        ).round(3)

        return self

    def evaluate_all(
        self,
        reg_cate=None,
        Zval: np.array = None,
        Ztrain: np.array = None,
        n_groups: int = 4
    ):

        """
        Implements the best linear prediction (`evaluate_blp'), calibration (`evaluate_cal') and QINI coefficient
        (`evaluate_qini') methods.

        Parameters
        ----------
        reg_cate: estimator, default ``None''
            CATE model. Must be able to implement `fit' and `predict' methods. If not specified, then fit_cate() must
            already have been implemented
        Zval: (n_cal x k) matrix, default ``None''
            Validation sample features for CATE model. If not specified, then `fit_cate' method must already have been
            implemented
        Ztrain: (n_train x k) matrix, default ``None''
            Training sample features for CATE model. If not specified, then `fit_cate' method must already have been
            implemented

        Returns
        -------
        self, with added dataframe df_res summarizing the results of all tests
        """

        # if CATE is given explicitly or has not been fitted at all previously, fit it now
        if (self.cate_preds_val is None) or (self.cate_preds_train is None) or (reg_cate is not None):
            if (Zval is None) or (reg_cate is None) or (Ztrain is None):
                raise Exception('CATE not fitted on training data - must provide CATE model and both Zval, Ztrain')
            self.fit_cate(reg_cate, Zval, Ztrain)

        self.evaluate_blp()
        self.evaluate_cal(n_groups=n_groups)
        self.evaluate_qini()

        self.df_res = self.blp_res.merge(self.qini_res, on='treatment')
        self.df_res['cal_r_squared'] = np.around(self.cal_r_squared, 3)

        return self

    def fit_nuisance_train(
        self,
        Xtrain: np.array,
        Dtrain: np.array,
        ytrain: np.array,
        Xval: np.array
    ) -> Tuple[np.array, np.array]:

        """
        Fits nuisance models in training sample and applies to generate predictions in validation sample.

        Parameters
        ----------
        Xtrain: (n_train x k) matrix
            Training sample features used to predict both treatment status and outcomes
        Dtrain: array of length n_train
            Training sample treatment assignments. Should have integer values with the lowest-value corresponding to the
            control treatment. It is recommended to have the control take value 0 and all other treatments be integers
            starting at 1
        ytrain: array of length n_train
            Outcomes for training sample
        Xval: (n_train x k) matrix
            Validation sample features used to predict both treatment status and outcomes

        Returns
        -------
        2 (n_val x n_treatment + 1) arrays corresponding to the predicted outcomes under treatment status and predicted
        treatment probabilities, respectively. Both evaluated on validation set.
        """

        # Fit propensity in treatment
        reg_t_fitted = self.reg_t.fit(Xtrain, Dtrain)
        # Predict propensity scores
        prop_preds = reg_t_fitted.predict_proba(Xval)

        # Possible treatments (need to allow more than 2)
        n = Xval.shape[0]
        reg_preds = np.zeros((n, self.n_treat + 1))
        for i in range(self.n_treat + 1):
            reg_outcome_fitted = self.reg_outcome.fit(Xtrain[Dtrain == self.tmts[i]], ytrain[Dtrain == self.tmts[i]])
            reg_preds[:, i] = reg_outcome_fitted.predict(Xval)

        return reg_preds, prop_preds

    def fit_nuisance_cv(
        self,
        X: np.array,
        D: np.array,
        y: np.array,
        random_state: int = 123
    ) -> Tuple[np.array, np.array]:

        """
        Generates nuisance function predictions using k-folds cross validation.

        Parameters
        ----------
        X: (n x k) matrix
            Features used to predict treatment/outcome
        D: array of length n
            Treatment assignments. Should have integer values with the lowest-value corresponding to the
            control treatment. It is recommended to have the control take value 0 and all other treatments be integers
            starting at 1
        y: array of length n
            Outcomes
        random_state: int, default 123
            Random seed

        Returns
        -------
        2 (n x n_treatment + 1) arrays corresponding to the predicted outcomes under treatment status and predicted
        treatment probabilities, respectively.
        """

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(X, D))

        prop_preds = cross_val_predict(self.reg_t, X, D, cv=splits, method='predict_proba')

        # Predict outcomes
        # T-learner logic
        N = X.shape[0]
        reg_preds = np.zeros((N, self.n_treat + 1))
        for k in range(self.n_treat + 1):
            for train, test in splits:
                reg_outcome_fitted = self.reg_outcome.fit(X[train][D[train] == self.tmts[k]],
                                                          y[train][D[train] == self.tmts[k]])
                reg_preds[test, k] = reg_outcome_fitted.predict(X[test])

        return reg_preds, prop_preds

    @staticmethod
    def calculate_dr_outcomes(
        D: np.array,
        y: np.array,
        reg_preds: np.array,
        prop_preds: np.array
    ) -> np.array:

        """
        Calculates doubly-robust (DR) outcomes using predictions from nuisance models

        Parameters
        ----------
        D: vector of length n
            Treatment assignments. Should have integer values with the lowest-value corresponding to the
            control treatment. It is recommended to have the control take value 0 and all other treatments be integers
            starting at 1
        y: vector of length n
            Outcomes
        reg_preds: (n x n_treat) matrix
            Outcome predictions for each potential treatment
        prop_preds: (n x n_treat) matrix
            Propensity score predictions for each treatment

        Returns
        -------
        Doubly robust outcome values
        """

        # treat each treatment as a separate regression
        # here, prop_preds should be a matrix
        # with rows corresponding to units and columns corresponding to treatment statuses
        dr_vec = []
        d0_mask = np.where(D == 0, 1, 0)
        y_dr_0 = reg_preds[:, 0] + (d0_mask / np.clip(prop_preds[:, 0], .01, np.inf)) * (y - reg_preds[:, 0])
        for k in np.sort(np.unique(D)):  # pick a treatment status
            if k > 0:  # make sure it is not control
                dk_mask = np.where(D == k, 1, 0)
                y_dr_k = reg_preds[:, k] + (dk_mask / np.clip(prop_preds[:, k], .01, np.inf)) * (y - reg_preds[:, k])
                dr_k = y_dr_k - y_dr_0  # this is an n x 1 vector
                dr_vec.append(dr_k)
        dr = np.column_stack(dr_vec)  # this is an n x n_treatment matrix

        return dr

        # Fits CATE in training, predicts in validation
    def fit_cate_train(
        self,
        reg_cate,
        Ztrain: np.array,
        Zval: np.array
    ) -> np.array:

        """
        Fit provided CATE estimator on training dataset and return predicted out-of-sample CATE values for validation
        set.

        Parameters
        ----------
        reg_cate: estimator
            CATE model. Must be able to implement `fit' and `predict' methods
        Ztrain: (n_train x k) matrix
            Training sample features for CATE model
        Zval: (n_val x k) matrix
            Validation sample features for CATE model

        Returns
        -------
        n_val x n_treatment array of predicted CATE values for validation set.
        """

        if np.ndim(Zval) == 1:
            Zval = Zval.reshape(-1, 1)

        if np.ndim(Ztrain) == 1:
            Ztrain = Ztrain.reshape(-1, 1)

        if self.n_treat == 1:
            reg_cate_fitted = reg_cate.fit(Ztrain, np.squeeze(self.dr_train))
            cate_preds = np.expand_dims(reg_cate_fitted.predict(Zval), axis=1)
        else:
            cate_preds = []
            for k in range(self.n_treat):  # fit a separate cate model for each treatment status?
                reg_cate_fitted = reg_cate.fit(Ztrain, self.dr_train[:, k])
                cate_preds.append(reg_cate_fitted.predict(Zval))

            cate_preds = np.column_stack(cate_preds)

        return cate_preds

    # CV prediction of CATEs
    @staticmethod
    def fit_cate_cv(
        reg_cate,
        Z: np.array,
        D: np.array,
        dr: np.array,
        n_splits: int = 5,
        random_state: int = 123
    ) -> np.array:

        """
        Generates CATE predictions using k-folds cross validation.

        Parameters
        ----------

        reg_cate: estimator
            CATE model. Must be able to implement `fit' and `predict' methods
        Z: (n x k) matrix
            Features for CATE model
        D: vector of length n
            Treatment assignments
        dr: (n x n_treat) matrix
            Doubly robust outcomes for each treatment
        n_splits: int, default 5
            Number of folds to use for cross validation prediction
        random_state: int default 123
            Seed value

        Returns
        -------
       (n x n_treat) array of predicted DR outcomes for each treatment
        """

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(cv.split(Z, D))

        if np.ndim(Z) == 1:
            Z = Z.reshape(-1, 1)

        K = len(np.unique(D)) - 1

        N = Z.shape[0]
        cate_preds = np.zeros((N, K))
        for k in range(K):
            cate_preds[:, k] = cross_val_predict(reg_cate, Z, dr[:, k], cv = splits)

        return cate_preds

    @staticmethod
    def calc_qini_coeff(
        cate_preds_train: np.array,
        cate_preds_val: np.array,
        dr_val: np.array,
        percentiles: np.array
    ) -> Tuple[float, float]:

        """
        Helper function for QINI coefficient calculation. See documentation for "evaluate_qini" method
        for more details.

        Parameters
        ----------
        cate_preds_train: (n_train x n_treatment) matrix
            Predicted CATE values for the training sample.
        cate_preds_val: (n_val x n_treatment) matrix
            Predicted CATE values for the validation sample.
        dr_val: (n_val x n_treatment) matrix
            Doubly robust outcome values for each treatment status in validation sample. Each value is relative to
            control, e.g. for treatment k the value is Y(k) - Y(0), where 0 signifies no treatment.
        percentiles: one-dimensional array
            Array of percentiles over which the QINI curve should be constructed. Defaults to 5%-95% in intervals of 5%.

        Returns
        -------
        QINI coefficient and associated standard error.
        """
        qs = np.percentile(cate_preds_train, percentiles)
        toc, toc_std, group_prob = np.zeros(len(qs)), np.zeros(len(qs)), np.zeros(len(qs))
        toc_psi = np.zeros((len(qs), dr_val.shape[0]))
        n = len(dr_val)
        ate = np.mean(dr_val)
        for it in range(len(qs)):
            inds = (qs[it] <= cate_preds_val)  # group with larger CATE prediction than the q-th quantile
            group_prob = np.sum(inds) / n  # fraction of population in this group
            toc[it] = group_prob * (
                    np.mean(dr_val[inds]) - ate)  # tau(q) = q * E[Y(1) - Y(0) | tau(X) >= q[it]] - E[Y(1) - Y(0)]
            toc_psi[it, :] = np.squeeze(
                (dr_val - ate) * (inds - group_prob) - toc[it])  # influence function for the tau(q)
            toc_std[it] = np.sqrt(np.mean(toc_psi[it] ** 2) / n)  # standard error of tau(q)

        qini_psi = np.sum(toc_psi[:-1] * np.diff(percentiles).reshape(-1, 1) / 100, 0)
        qini = np.sum(toc[:-1] * np.diff(percentiles) / 100)
        qini_stderr = np.sqrt(np.mean(qini_psi ** 2) / n)

        return qini, qini_stderr

    def evaluate_qini(
        self,
        reg_cate=None,
        Zval: np.array = None,
        Ztrain: np.array = None,
        percentiles: np.array = np.linspace(5, 95, 50)
    ):

        """
        Calculates QINI coefficient for the given model as in Radcliffe (2007), where units are ordered by predicted
        CATE values and a running measure of the average treatment effect in each cohort is kept as we progress
        through ranks. The QINI coefficient is then the area under the resulting curve, with a value of 0 interpreted
        as corresponding to a model with randomly assigned CATE coefficients. All calculations are performed on
        validation dataset results, using the training set as input.

        Parameters
        ----------
        reg_cate: estimator, default ``None''
            CATE model. Must be able to implement `fit' and `predict' methods. If not specified, then fit_cate() must
            already have been implemented
        Zval: (n_val x k) matrix, default ``None''
            Validation sample features for CATE model. If not specified, then `fit_cate' method must already have been
            implemented
        Ztrain: (n_train x k) matrix, default ``None''
            Training sample features for CATE model. If specified, then CATE is fitted on training sample and applied
            to Zval. If specified, then Xtrain, Dtrain, Ytrain must have been specified in `fit_nuisance' method (and
            vice-versa)
        percentiles: one-dimensional array, default ``np.linspace(5, 95, 50)''
            Array of percentiles over which the QINI curve should be constructed. Defaults to 5%-95% in intervals of 5%.

        Returns
        -------
        self, with added dataframe qini_res showing the results of the QINI fit
        """
        if self.dr_val is None:
            raise Exception("Must fit nuisances before evaluating")

        # if CATE is given explicitly or has not been fitted at all previously, fit it now
        if (self.cate_preds_train is None) or (self.cate_preds_val is None) or (reg_cate is not None):
            if (Zval is None) or (Ztrain is None) or (reg_cate is None):
                raise Exception('CATE not fitted on training data - must provide CATE model and both Zval, Ztrain')
            self.fit_cate(reg_cate, Zval, Ztrain)

        if self.n_treat == 1:
            qini, qini_err = self.calc_qini_coeff(
                self.cate_preds_train,
                self.cate_preds_val,
                self.dr_val,
                percentiles
            )
            qinis = [qini]
            errs = [qini_err]
        else:
            qinis = []
            errs = []
            for k in range(self.n_treat):
                qini, qini_err = self.calc_qini_coeff(
                    self.cate_preds_train[:, k],
                    self.cate_preds_val[:, k],
                    self.dr_val[:, k],
                    percentiles
                )

                qinis.append(qini)
                errs.append(qini_err)

        pvals = [st.norm.sf(abs(q / e)) for q, e in zip(qinis, errs)]

        self.qini_res = pd.DataFrame(
            {'treatment': self.tmts[1:], 'qini_coeff': qinis, 'qini_se': errs, 'qini_pval': pvals},
        ).round(3)

        return self
