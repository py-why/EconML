# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pytest
import unittest
import numpy as np

from econml.data.dgps import StandardDGP
from econml.dml import LinearDML
from econml.iv.dml import OrthoIV, DMLIV

from itertools import product

from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

class TestDGP(unittest.TestCase):

    def test_standard_dgp(self):
        n_list = [10000]
        d_t_list = [1]
        d_y_list = [1]
        d_x_list = [1, 5]
        d_z_list = [None, 1]
        discrete_treatment_list=[False, True]
        discrete_instrument_list=[False, True]

        nuisance_Y_list = [
            None,
        ]

        nuisance_Y_list = [
            None,
        ]

        nuisance_T_list = [
            None,
        ]

        nuisance_TZ_list = [
            None,
        ]

        theta_list = [
            None, 
        ]

        y_of_t_list = [
            None,
        ]

        param_lists = [n_list, d_t_list, d_y_list, d_x_list, d_z_list, discrete_treatment_list, discrete_instrument_list, nuisance_Y_list, nuisance_T_list, nuisance_TZ_list, theta_list, y_of_t_list]

        results_list = []

        for i in range(10):
            for config_tuple in product(*param_lists):
            # for config_tuple in [config_tuple]:
                n, d_t, d_y, d_x, d_z, discrete_treatment, discrete_instrument, nuisance_Y, nuisance_T, nuisance_TZ, theta, y_of_t = config_tuple

                dgp = StandardDGP(n=n, d_t=d_t, d_y=d_y, d_x=d_x, d_z=d_z, discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument, nuisance_Y=nuisance_Y, nuisance_T=nuisance_T, nuisance_TZ=nuisance_TZ, theta=theta, y_of_t=y_of_t)

                data_dict = dgp.gen_data()

                if d_z:
                    est = OrthoIV(discrete_treatment=discrete_treatment, discrete_instrument=discrete_instrument)
                else:
                    est = LinearDML(
                        model_t = LogisticRegression() if discrete_treatment else LinearRegression(), 
                        model_y = LinearRegression(), 
                        discrete_treatment=discrete_treatment
                    )

                est.fit(**data_dict)


                X = data_dict['X']
                T = data_dict['T']
                actual_eff = dgp.effect(X, T0=0, T1=np.ones(shape=T.shape))
                eff_inf = est.effect_inference(X=X)
                eff_lb, eff_ub = eff_inf.conf_int(alpha=0.01)
                proportion_in_interval = ((eff_lb < actual_eff) & (actual_eff < eff_ub)).mean()
                results_list.append(proportion_in_interval)

                np.testing.assert_array_less(0.50, proportion_in_interval)
                
    
    
    






