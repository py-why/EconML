from typing import Tuple, Union, List

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from statsmodels.api import OLS
from statsmodels.tools import add_constant
from sklearn.model_selection import check_cv
from utils import calculate_dr_outcomes, calc_qini_coeff


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

    .. math::

    Cal_G = \sum_k \pi(k) |E[s(Z) | k] - E[Y^{DR | k}|

    Cal_O = \sum_k \pi(k) |E[s(Z) | k] - E[Y^{DR}|

    The calibration r-squared is then defined as

    .. math::

    \mathcal{R^2}_C = 1 - \frac{Cal_G}{Cal_O}

    The calibration r-squared metric is similar to the standard R-square score in that it can take any value
    less than or equal to 1, with scores closer to 1 indicating a better calibrated CATE model.

    **QINI**

    Units are ordered by predicted CATE values and a running measure of the average treatment effect in each cohort is
    kept as we progress through ranks. The QINI coefficient is then the area under the resulting curve, with a value
    of 0 interpreted as corresponding to a model with randomly assigned CATE coefficients. All calculations are
    performed on validation dataset results, using the training set as input.

    More formally, the QINI curve is given by the following function:

    .. math::

    \tau_{QINI}(q) = Cov(Y^{DR}(g,p), 1\{\hat{\tau}(Z) \geq \hat{\mu}(q)\})

    Where q is the desired quantile, \hat{\mu} is the quantile function, and \hat{\tau} is the predicted CATE function.
    Y^{DR}(g,p) refers to the doubly robust outcome difference (relative to control) for the given observation.

    The QINI coefficient is then given by:

    QINI = \int_0^1 \tau_{QINI}(q) dq

    Parameters
    ----------
    model_regression: estimator
        Nuisance model estimator used to fit the outcome to features. Must be able to implement `fit' and `predict'
        methods

    model_propensity: estimator
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
        *,
        model_regression,
        model_propensity,
        cate,
        cv: Union[int, List] = 5
    ):
        self.model_regression = model_regression
        self.model_propensity = model_propensity
        self.cate = cate
        self.cv = cv
        self.dr_train = None
        self.cate_preds_train = None
        self.cate_preds_val = None
        self.dr_val = None
        self.cv_splitter = None

    def get_cv_splitter(self, random_state: int = 123):
        splitter = check_cv(self.cv, [0], classifier=True)
        if splitter != self.cv and isinstance(splitter, (KFold, StratifiedKFold)):
            splitter.shuffle = True
            splitter.random_state = random_state
        self.cv_splitter = splitter

    def get_cv_splits(self, a, b):
        if self.cv_splitter is None:
            self.get_cv_splitter()

        all_vars = [var if np.ndim(var) == 2 else var.reshape(-1, 1) for var in [a,b] if var is not None]
        to_split = np.hstack(all_vars)
        folds = self.cv_splitter.split(to_split)

        return list(folds)

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
        self.treatments = np.sort(np.unique(Dval))

        # Number of treatments (excluding control)
        self.n_treat = len(self.treatments) - 1

        # Indicator for whether
        self.fit_on_train = (Xtrain is not None) and (Dtrain is not None) and (ytrain is not None)

        if self.fit_on_train:
            # Get DR outcomes in training sample
            reg_preds_train, prop_preds_train = self.fit_nuisance_cv(Xtrain, Dtrain, ytrain)
            self.dr_train = calculate_dr_outcomes(Dtrain, ytrain, reg_preds_train, prop_preds_train)

            # Get DR outcomes in validation sample
            reg_preds_val, prop_preds_val = self.fit_nuisance_train(Xtrain, Dtrain, ytrain, Xval)
            self.dr_val = calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)

            self.Dtrain = Dtrain

        else:
            # Get DR outcomes in validation sample
            reg_preds_val, prop_preds_val = self.fit_nuisance_cv(Xval, Dval, yval)
            self.dr_val = calculate_dr_outcomes(Dval, yval, reg_preds_val, prop_preds_val)

        # Calculate ATE in the validation sample
        self.ate_val = self.dr_val.mean(axis=0)

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
        model_propensity_fitted = self.model_propensity.fit(Xtrain, Dtrain)
        # Predict propensity scores
        prop_preds = model_propensity_fitted.predict_proba(Xval)

        # Possible treatments (need to allow more than 2)
        n = Xval.shape[0]
        reg_preds = np.zeros((n, self.n_treat + 1))
        for i in range(self.n_treat + 1):
            model_regression_fitted = self.model_regression.fit(Xtrain[Dtrain == self.treatments[i]], ytrain[Dtrain == self.treatments[i]])
            reg_preds[:, i] = model_regression_fitted.predict(Xval)

        return reg_preds, prop_preds

    def fit_nuisance_cv(
        self,
        X: np.array,
        D: np.array,
        y: np.array
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

        splits = self.get_cv_splits(X, D)
        prop_preds = cross_val_predict(self.model_propensity, X, D, cv=splits, method='predict_proba')

        # Predict outcomes
        # T-learner logic
        N = X.shape[0]
        reg_preds = np.zeros((N, self.n_treat + 1))
        for k in range(self.n_treat + 1):
            for train, test in splits:
                model_regression_fitted = self.model_regression.fit(X[train][D[train] == self.treatments[k]],
                                                          y[train][D[train] == self.treatments[k]])
                reg_preds[test, k] = model_regression_fitted.predict(X[test])

        return reg_preds, prop_preds

    def get_cate_preds(
        self,
        Zval: np.array,
        Ztrain: np.array = None
    ):

        """
        Fits CATE model and generates predictions. If Ztrain is None, then the predictions are generated using k-folds
        cross-validation on the validation set. If Ztrain is specified, then the CATE is fit on the training sample
        (where the DR outcomes were generated using k-folds CV), and then applied to the validation sample.

        Parameters
        ----------
        Zval: (n_val x n_treatment) matrix
            Validation set features to be used to predict (and potentially fit) DR outcomes in CATE model
        Ztrain (n_train x n_treatment) matrix, defaul ``None``
            Training set features used to fit CATE model

        Returns
        -------
        None, but adds attribute cate_preds_val for predicted CATE values on the validation set and, if training
        data is provided, attribute cate_preds_train for predicted CATE values on the training set
        """
        base = self.treatments[0]
        vals = [self.cate.effect(Zval, TO=base, T1=t) for t in self.treatments[1:]]
        self.cate_preds_val = np.stack(vals).T

        if Ztrain is not None:
            trains = [self.cate.effect(Ztrain, TO=base, T1=t) for t in self.treatments[1:]]
            self.cate_preds_train = np.stack(trains).T

    def evaluate_cal(
        self,
        Zval: np.array = None,
        Ztrain: np.array = None,
        n_groups: int = 4
    ):

        """
        Implements calibration test as in [Dwivedi2020]

        Parameters
        ----------
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
        if (self.cate_preds_train is None) or (self.cate_preds_val is None):
            if (Zval is None) or (Ztrain is None):
                raise Exception('CATE not fitted on training data - must provide both Zval, Ztrain')
            self.get_cate_preds(Zval, Ztrain)

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
            df_plot1['tmt'] = self.treatments[k + 1]
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
        rsq = round(self.cal_r_squared[np.where(self.treatments == tmt)[0][0] - 1], 3)
        df['95_err'] = 1.96 * df['se_gate']
        fig = df.plot(
            kind='scatter',
            x='g_cate',
            y='gate',
            yerr='95_err',
            xlabel = 'Group Mean CATE',
            ylabel = 'GATE',
            title=f"Treatment = {tmt}, Calibration R^2 = {rsq}"
        )

        return fig

    def evaluate_blp(
        self,
        Zval: np.array = None,
        Ztrain: np.array = None
    ):
        """
        Implements the best linear predictor (BLP) test as in [Chernozhukov2022]. `fit_nusiance' method must already
        be implemented.

        Parameters
        ----------
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
        if self.cate_preds_val is None:
            if Zval is None:  # need at least Zval
                raise Exception('CATE predictions not yet calculated - must provide Zval')
            self.get_cate_preds(Zval, Ztrain)

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
                reg = OLS(self.dr_val[:, k], add_constant(self.cate_preds_val[:, k])).fit(cov_type='HC1')
                params.append(reg.params[1])
                errs.append(reg.bse[1])
                pvals.append(reg.pvalues[1])

        self.blp_res = pd.DataFrame(
            {'treatment': self.treatments[1:], 'blp_est': params, 'blp_se': errs, 'blp_pval': pvals}
        ).round(3)

        return self

    def evaluate_qini(
        self,
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
        if (self.cate_preds_train is None) or (self.cate_preds_val is None):
            if (Zval is None) or (Ztrain is None):
                raise Exception('CATE predictions not yet calculated - must provide CATE model and both Zval, Ztrain')
            self.get_cate_preds(Zval, Ztrain)

        if self.n_treat == 1:
            qini, qini_err = calc_qini_coeff(
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
                qini, qini_err = calc_qini_coeff(
                    self.cate_preds_train[:, k],
                    self.cate_preds_val[:, k],
                    self.dr_val[:, k],
                    percentiles
                )

                qinis.append(qini)
                errs.append(qini_err)

        pvals = [st.norm.sf(abs(q / e)) for q, e in zip(qinis, errs)]

        self.qini_res = pd.DataFrame(
            {'treatment': self.treatments[1:], 'qini_coeff': qinis, 'qini_se': errs, 'qini_pval': pvals},
        ).round(3)

        return self

    def evaluate_all(
        self,
        Zval: np.array = None,
        Ztrain: np.array = None,
        n_groups: int = 4
    ):
        """
        Implements the best linear prediction (`evaluate_blp'), calibration (`evaluate_cal') and QINI coefficient
        (`evaluate_qini') methods.

        Parameters
        ----------
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
        if (self.cate_preds_val is None) or (self.cate_preds_train is None):
            if (Zval is None) or (Ztrain is None):
                raise Exception('CATE predictions not yet calculated - must provide both Zval, Ztrain')
            self.get_cate_preds(Zval, Ztrain)

        self.evaluate_blp()
        self.evaluate_cal(n_groups=n_groups)
        self.evaluate_qini()

        self.df_res = self.blp_res.merge(self.qini_res, on='treatment')
        self.df_res['cal_r_squared'] = np.around(self.cal_r_squared, 3)

        return self
