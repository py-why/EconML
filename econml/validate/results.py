import numpy as np
import pandas as pd

from typing import List


class CalibrationEvaluationResults:
    """
    Results class for calibration test.

    Parameters
    ----------
    cal_r_squared: list or numpy array of floats
        Sequence of calibration R^2 values

    df_plot: pandas dataframe
        Dataframe containing necessary data for plotting calibration test GATE results

    treatments: list or numpy array of floats
        Sequence of treatment labels
    """
    def __init__(
        self,
        cal_r_squared: np.array,
        df_plot: pd.DataFrame,
        treatments: np.array
    ):
        self.cal_r_squared = cal_r_squared
        self.df_plot = df_plot
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


class QiniEvaluationResults:
    """
    Results class for QINI test.

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
            'qini_est': self.params,
            'qini_se': self.errs,
            'qini_pval': self.pvals
        }).round(3)
        return res


class EvaluationResults:
    """
    Results class for combination of all tests.

    Parameters
    ----------
    cal_res: CalibrationEvaluationResults object
       Results object for calibration test

    blp_res: BLPEvaluationResults object
       Results object for BLP test

    qini_res: QiniEvaluationResults object
       Results object for QINI test
    """
    def __init__(
        self,
        cal_res: CalibrationEvaluationResults,
        blp_res: BLPEvaluationResults,
        qini_res: QiniEvaluationResults
    ):
        self.cal = cal_res
        self.blp = blp_res
        self.qini = qini_res

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
            self.qini.summary(),
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
