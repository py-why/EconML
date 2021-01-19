# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

from econml.dml import DML, LinearDML, SparseLinearDML, NonParamDML
from econml.metalearners import XLearner, TLearner, SLearner, DomainAdaptationLearner
from econml.dr import DRLearner
from econml.score import RScorer


def _fit_model(name, model, Y, T, X):
    return name, model.fit(Y, T, X=X)


class TestRScorer(unittest.TestCase):

    def _get_data(self):
        X = np.random.normal(0, 1, size=(500, 2))
        T = np.random.binomial(1, .5, size=(500,))
        y = X[:, 0] * T + np.random.normal(size=(500,))
        return y, T, X, X[:, 0]

    def test_comparison(self):
        def reg():
            return LinearRegression()

        def clf():
            return LogisticRegression()

        y, T, X, true_eff = self._get_data()
        (X_train, X_val, T_train, T_val,
         Y_train, Y_val, _, true_eff_val) = train_test_split(X, T, y, true_eff, test_size=.4)

        models = [('ldml', LinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True,
                                     linear_first_stages=False, cv=3)),
                  ('sldml', SparseLinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True,
                                            featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                            linear_first_stages=False, cv=3)),
                  ('xlearner', XLearner(models=reg(), cate_models=reg(), propensity_model=clf())),
                  ('dalearner', DomainAdaptationLearner(models=reg(), final_models=reg(), propensity_model=clf())),
                  ('slearner', SLearner(overall_model=reg())),
                  ('tlearner', TLearner(models=reg())),
                  ('drlearner', DRLearner(model_propensity=clf(), model_regression=reg(),
                                          model_final=reg(), cv=3)),
                  ('rlearner', NonParamDML(model_y=reg(), model_t=clf(), model_final=reg(),
                                           discrete_treatment=True, cv=3)),
                  ('dml3dlasso', DML(model_y=reg(), model_t=clf(), model_final=reg(), discrete_treatment=True,
                                     featurizer=PolynomialFeatures(degree=3),
                                     linear_first_stages=False, cv=3))
                  ]

        models = Parallel(n_jobs=-1, verbose=1)(delayed(_fit_model)(name, mdl,
                                                                    Y_train, T_train, X_train)
                                                for name, mdl in models)

        scorer = RScorer(model_y=reg(), model_t=clf(),
                         discrete_treatment=True, cv=3, mc_iters=2, mc_agg='median')
        scorer.fit(Y_val, T_val, X=X_val)
        rscore = [scorer.score(mdl) for _, mdl in models]
        rootpehe_score = [np.sqrt(np.mean((true_eff_val.flatten() - mdl.effect(X_val).flatten())**2))
                          for _, mdl in models]
        assert LinearRegression().fit(np.array(rscore).reshape(-1, 1), np.array(rootpehe_score)).coef_ < 0.5
        mdl, _ = scorer.best_model([mdl for _, mdl in models])
        rootpehe_best = np.sqrt(np.mean((true_eff_val.flatten() - mdl.effect(X_val).flatten())**2))
        assert rootpehe_best < 1.2 * np.min(rootpehe_score)
        mdl, _ = scorer.ensemble([mdl for _, mdl in models])
        rootpehe_ensemble = np.sqrt(np.mean((true_eff_val.flatten() - mdl.effect(X_val).flatten())**2))
        assert rootpehe_ensemble < 1.2 * np.min(rootpehe_score)
