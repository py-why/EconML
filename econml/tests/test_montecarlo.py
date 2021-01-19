# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.dml import (DML, LinearDML, SparseLinearDML, KernelDML, NonParamDML, ForestDML)
from econml.dr import (DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner)
from econml.iv.dml import (DMLATEIV, ProjectedDMLATEIV, DMLIV, NonParamDMLIV)
from econml.iv.dr import (IntentToTreatDRIV, LinearIntentToTreatDRIV)
import numpy as np


class TestMonteCarlo(unittest.TestCase):

    def test_montecarlo(self):
        """Test that we can perform nuisance averaging, and that it reduces the variance in a simple example."""
        y = np.random.normal(size=30) + [0, 1] * 15
        T = np.random.normal(size=(30,)) + y
        W = np.random.normal(size=(30, 3))
        est1 = LinearDML(model_y=LinearRegression(), model_t=LinearRegression())
        est2 = LinearDML(model_y=LinearRegression(), model_t=LinearRegression(), mc_iters=2)
        est3 = LinearDML(model_y=LinearRegression(), model_t=LinearRegression(), mc_iters=2, mc_agg='median')
        # Run ten experiments, recomputing the variance of 10 estimates of the effect in each experiment
        v1s = [np.var([est1.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        v2s = [np.var([est2.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        v3s = [np.var([est3.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        # The average variance should be lower when using monte carlo iterations
        assert np.mean(v2s) < np.mean(v1s)
        assert np.mean(v3s) < np.mean(v1s)

    def test_discrete_treatment(self):
        """Test that we can perform nuisance averaging, and that it reduces the variance in a simple example."""
        y = np.random.normal(size=30) + [0, 1] * 15
        T = np.random.binomial(1, .5, size=(30,))
        W = np.random.normal(size=(30, 3))
        est1 = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                         discrete_treatment=True)
        est2 = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                         discrete_treatment=True, mc_iters=2)
        est3 = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(),
                         discrete_treatment=True, mc_iters=2, mc_agg='median')
        # Run ten experiments, recomputing the variance of 10 estimates of the effect in each experiment
        v1s = [np.var([est1.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        v2s = [np.var([est2.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        v3s = [np.var([est3.fit(y, T, W=W).effect() for _ in range(10)]) for _ in range(10)]
        # The average variance should be lower when using monte carlo iterations
        assert np.mean(v2s) < np.mean(v1s)
        assert np.mean(v3s) < np.mean(v1s)

    def test_parameter_passing(self):
        for gen in [DML, NonParamDML]:
            est = gen(model_y=LinearRegression(), model_t=LinearRegression(),
                      model_final=LinearRegression(),
                      mc_iters=2, mc_agg='median')
            assert est.mc_iters == 2
            assert est.mc_agg == 'median'
        for gen in [LinearDML, SparseLinearDML, KernelDML, ForestDML]:
            est = gen(model_y=LinearRegression(), model_t=LinearRegression(),
                      mc_iters=2, mc_agg='median')
            assert est.mc_iters == 2
            assert est.mc_agg == 'median'
        for gen in [DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner]:
            est = gen(mc_iters=2, mc_agg='median')
            assert est.mc_iters == 2
            assert est.mc_agg == 'median'
        for gen in [DMLATEIV(model_Y_W=LinearRegression(),
                             model_T_W=LinearRegression(),
                             model_Z_W=LinearRegression(), mc_iters=2, mc_agg='median'),
                    ProjectedDMLATEIV(model_Y_W=LinearRegression(),
                                      model_T_W=LinearRegression(),
                                      model_T_WZ=LinearRegression(), mc_iters=2, mc_agg='median'),
                    DMLIV(model_Y_X=LinearRegression(),
                          model_T_X=LinearRegression(),
                          model_T_XZ=LinearRegression(),
                          model_final=LinearRegression(), mc_iters=2, mc_agg='median'),
                    NonParamDMLIV(model_Y_X=LinearRegression(),
                                  model_T_X=LinearRegression(),
                                  model_T_XZ=LinearRegression(),
                                  model_final=LinearRegression(), mc_iters=2, mc_agg='median'),
                    IntentToTreatDRIV(model_Y_X=LinearRegression(),
                                      model_T_XZ=LinearRegression(),
                                      flexible_model_effect=LinearRegression(), mc_iters=2, mc_agg='median'),
                    LinearIntentToTreatDRIV(model_Y_X=LinearRegression(),
                                            model_T_XZ=LinearRegression(),
                                            flexible_model_effect=LinearRegression(),
                                            mc_iters=2, mc_agg='median')]:
            assert est.mc_iters == 2
            assert est.mc_agg == 'median'
