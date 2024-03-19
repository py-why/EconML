import numpy as np
import pandas as pd

from typing import List, Dict, Any


class CalibrationEvaluationResults:
    """
    Results class for calibration test.

    Parameters
    ----------
    cal_r_squared: list or numpy array of floats
        Sequence of calibration R^2 values

    plot_data_dict: dict
        Dictionary mapping treatment levels to dataframes containing necessary
        data for plotting calibration test GATE results

    treatments: list or numpy array of floats
        Sequence of treatment labels
    """

    def __init__(
        self,
        cal_r_squared: np.array,
        plot_data_dict: Dict[Any, pd.DataFrame],
        treatments: np.array
    ):
        self.cal_r_squared = cal_r_squared
        self.plot_data_dict = plot_data_dict
        self.treatments = treatments

    def summary(self) -> pd.DataFrame:
        """
        Constructs dataframe summarizing the results of the calibration test.

        Parameters
        ----------
        None

        Returns
        -------
        pandas dataframe containing summary of calibration test results
        """

        res = pd.DataFrame({
            'treatment': self.treatments[1:],
            'cal_r_squared': self.cal_r_squared,
        }).round(3)
        return res

    def plot_cal(self, tmt: Any):
        """
        Plots group average treatment effects (GATEs) and predicted GATEs by quantile-based group in validation sample.

        Parameters
        ----------
        tmt: Any
            Name of treatment to plot

        Returns
        -------
        matplotlib plot with predicted GATE on x-axis and GATE (and 95% CI) on y-axis
        """
        if tmt not in self.treatments[1:]:
            raise ValueError(f'Invalid treatment; must be one of {self.treatments[1:]}')

        df = self.plot_data_dict[tmt].copy()
        rsq = round(self.cal_r_squared[np.where(self.treatments == tmt)[0][0] - 1], 3)
        df['95_err'] = 1.96 * df['se_gate']
        fig = df.plot(
            kind='scatter',
            x='g_cate',
            y='gate',
            yerr='95_err',
            xlabel='Group Mean CATE',
            ylabel='GATE',
            title=f"Treatment = {tmt}, Calibration R^2 = {rsq}"
        )

        return fig


class BLPEvaluationResults:
    """
    Results class for BLP test.

    Parameters
    ----------
    params: list or numpy array of floats
       Sequence of estimated coefficient values

    errs: list or numpy array of floats
       Sequence of estimated coefficient standard errors

    pvals: list or numpy array of floats
       Sequence of estimated coefficient p-values

    treatments: list or numpy array of floats
       Sequence of treatment labels
    """

    def __init__(
        self,
        params: List[float],
        errs: List[float],
        pvals: List[float],
        treatments: np.array
    ):
        self.params = params
        self.errs = errs
        self.pvals = pvals
        self.treatments = treatments

    def summary(self):
        """
        Constructs dataframe summarizing the results of the BLP test.

        Parameters
        ----------
        None

        Returns
        -------
        pandas dataframe containing summary of BLP test results
        """
        res = pd.DataFrame({
            'treatment': self.treatments[1:],
            'blp_est': self.params,
            'blp_se': self.errs,
            'blp_pval': self.pvals
        }).round(3)
        return res


class UpliftEvaluationResults:
    """
    Results class for uplift curve-based tests.

    Parameters
    ----------
    params: list or numpy array of floats
       Sequence of estimated QINI coefficient values

    errs: list or numpy array of floats
       Sequence of estimated QINI coefficient standard errors

    pvals: list or numpy array of floats
       Sequence of estimated QINI coefficient p-values

    treatments: list or numpy array of floats
       Sequence of treatment labels

    curve_data_dict: dict
        Dictionary mapping treatment levels to dataframes containing
        necessary data for plotting uplift curves
    """

    def __init__(
        self,
        params: List[float],
        errs: List[float],
        pvals: List[float],
        treatments: np.array,
        curve_data_dict: Dict[Any, pd.DataFrame]
    ):
        self.params = params
        self.errs = errs
        self.pvals = pvals
        self.treatments = treatments
        self.curves = curve_data_dict

    def summary(self):
        """
        Constructs dataframe summarizing the results of the QINI test.

        Parameters
        ----------
        None

        Returns
        -------
        pandas dataframe containing summary of QINI test results
        """
        res = pd.DataFrame({
            'treatment': self.treatments[1:],
            'est': self.params,
            'se': self.errs,
            'pval': self.pvals
        }).round(3)
        return res

    def plot_uplift(self, tmt: Any, err_type: str = None):
        """
        Plots uplift curves.

        Parameters
        ----------
        tmt: any (sortable)
            Name of treatment to plot.

        err_type: str
            Type of error to plot. Accepted values are normal (None), two-sided uniform confidence band ('ucb2'),
            or 1-sided uniform confidence band ('ucb1').

        Returns
        -------
        matplotlib plot with percentage treated on x-axis and uplift metric (and 95% CI) on y-axis
        """
        if tmt not in self.treatments[1:]:
            raise ValueError(f'Invalid treatment; must be one of {self.treatments[1:]}')

        df = self.curves[tmt].copy()

        if err_type is None:
            df['95_err'] = 1.96 * df['err']
        elif err_type == 'ucb2':
            df['95_err'] = df['uniform_critical_value'] * df['err']
        elif err_type == 'ucb1':
            df['95_err'] = df['uniform_one_side_critical_value'] * df['err']
        else:
            raise ValueError(f"Invalid error type {err_type!r}; must be one of [None, 'ucb2', 'ucb1']")

        res = self.summary()
        coeff = round(res.loc[res['treatment'] == tmt]['est'].values[0], 3)
        err = round(res.loc[res['treatment'] == tmt]['se'].values[0], 3)

        if err_type == 'ucb1':
            fig = df.plot(
                kind='scatter',
                x='Percentage treated',
                y='value',
                yerr=[[df['95_err'], np.zeros(len(df))]],
                ylabel='Gain over Random',
                title=f"Treatment = {tmt}, Integral = {coeff} +/- {err}"
            )
        else:
            fig = df.plot(
                kind='scatter',
                x='Percentage treated',
                y='value',
                yerr='95_err',
                ylabel='Gain over Random',
                title=f"Treatment = {tmt}, Integral = {coeff} +/- {err}"
            )

        return fig


class EvaluationResults:
    """
    Results class for combination of all tests.

    Parameters
    ----------
    cal_res: CalibrationEvaluationResults object
       Results object for calibration test

    blp_res: BLPEvaluationResults object
       Results object for BLP test

    qini_res: UpliftEvaluationResults object
       Results object for QINI test

    toc_res: UpliftEvaluationResults object
       Results object for TOC test
    """

    def __init__(
        self,
        cal_res: CalibrationEvaluationResults,
        blp_res: BLPEvaluationResults,
        qini_res: UpliftEvaluationResults,
        toc_res: UpliftEvaluationResults
    ):
        self.cal = cal_res
        self.blp = blp_res
        self.qini = qini_res
        self.toc = toc_res

    def summary(self):
        """
        Constructs dataframe summarizing the results of all 3 tests.

        Parameters
        ----------
        None

        Returns
        -------
        pandas dataframe containing summary of all test results
        """
        res = self.blp.summary().merge(
            self.qini.summary().rename({'est': 'qini_est', 'se': 'qini_se', 'pval': 'qini_pval'}, axis=1),
            on='treatment'
        ).merge(
            self.toc.summary().rename({'est': 'autoc_est', 'se': 'autoc_se', 'pval': 'autoc_pval'}, axis=1),
            on='treatment'
        ).merge(
            self.cal.summary(),
            on='treatment'
        )
        return res

    def plot_cal(self, tmt: int):
        """
        Plots group average treatment effects (GATEs) and predicted GATEs by quantile-based group in validation sample.

        Parameters
        ----------
        tmt: integer
            Treatment level to plot

        Returns
        -------
        matplotlib plot with predicted GATE on x-axis and GATE (and 95% CI) on y-axis
        """
        return self.cal.plot_cal(tmt)

    def plot_qini(self, tmt: int, err_type: str = None):
        """
        Plots QINI curves.

        Parameters
        ----------
        tmt: integer
            Treatment level to plot

        err_type: str
            Type of error to plot. Accepted values are normal (None), two-sided uniform confidence band ('ucb2'),
            or 1-sided uniform confidence band ('ucb1').

        Returns
        -------
        matplotlib plot with percentage treated on x-axis and QINI value (and 95% CI) on y-axis
        """
        return self.qini.plot_uplift(tmt, err_type)

    def plot_toc(self, tmt: int, err_type: str = None):
        """
        Plots TOC curves.

        Parameters
        ----------
        tmt: integer
            Treatment level to plot

        err_type: str
            Type of error to plot. Accepted values are normal (None), two-sided uniform confidence band ('ucb2'),
            or 1-sided uniform confidence band ('ucb1').

        Returns
        -------
        matplotlib plot with percentage treated on x-axis and TOC value (and 95% CI) on y-axis
        """
        return self.toc.plot_uplift(tmt, err_type)
