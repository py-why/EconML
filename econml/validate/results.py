import numpy as np
import pandas as pd


class CalibrationEvaluationResults:
    def __init__(self, cal_r_squared, df_plot, treatments):
        self.cal_r_squared = cal_r_squared
        self.df_plot = df_plot
        self.treatments = treatments

    def summary(self):
        res = pd.DataFrame({
            'treatment': self.treatments[1:],
            'Calibration R^2': self.cal_r_squared,
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
            xlabel='Group Mean CATE',
            ylabel='GATE',
            title=f"Treatment = {tmt}, Calibration R^2 = {rsq}"
        )

        return fig


class BLPEvaluationResults:
    def __init__(self, params, errs, pvals, treatments):
        self.params = params
        self.errs = errs
        self.pvals = pvals
        self.treatments = treatments

    def summary(self):
        res = pd.DataFrame({
            'treatment': self.treatments[1:],
            'blp_est': self.params,
            'blp_se': self.errs,
            'blp_pval': self.pvals
        }).round(3)
        return res


class QiniEvaluationResults:
    def __init__(self, params, errs, pvals, treatments):
        self.params = params
        self.errs = errs
        self.pvals = pvals
        self.treatments = treatments

    def summary(self):
        res = pd.DataFrame({
            'treatment': self.treatments[1:],
            'blp_est': self.params,
            'blp_se': self.errs,
            'blp_pval': self.pvals
        }).round(3)
        return res


class EvaluationResults:
    def __init__(self, cal_res, blp_res, qini_res):
        self.cal = cal_res
        self.blp = blp_res
        self.qini = qini_res

    def summary(self):
        res = self.blp.merge(self.qini, on='treatment')
        res['cal_r_squared'] = np.around(self.cal.cal_r_squared, 3)
        return res
