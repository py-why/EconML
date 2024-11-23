# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

from econml.dml import DML, LinearDML, SparseLinearDML, NonParamDML
from econml.metalearners import XLearner, TLearner, SLearner, DomainAdaptationLearner
from econml.dr import DRLearner
from econml.score import RScorer


def _fit_model(name, model, Y, T, X):
    return name, model.fit(Y, T, X=X)


class TestRScorer(unittest.TestCase):

    def _get_data(self, discrete_outcome=False):
        X = np.random.normal(0, 1, size=(100000, 2))
        T = np.random.binomial(1, .5, size=(100000,))
        if discrete_outcome:
            eps = np.random.normal(size=(100000,))
            log_odds = X[:, 0]*T + eps
            y_sigmoid = 1/(1 + np.exp(-log_odds))
            y = np.array([np.random.binomial(1, p) for p in y_sigmoid])
            # Difference in conditional probabilities P(y=1|X,T=1) - P(y=1|X,T=0)
            true_eff = (1 / (1 + np.exp(-(X[:, 0]+eps)))) - (1 / (1 + np.exp(-eps)))
        else:
            y = X[:, 0] * T + np.random.normal(size=(100000,))
            true_eff = X[:, 0]

        y = y.reshape(-1, 1)
        T = T.reshape(-1, 1)
        return y, T, X, true_eff

    def test_comparison(self):

        def reg():
            return LinearRegression()

        def clf():
            return LogisticRegression()

        test_cases = [
            {"name":"continuous_outcome", "discrete_outcome": False},
            {"name":"discrete_outcome", "discrete_outcome": True}
        ]

        for case in test_cases:
            with self.subTest(case["name"]):
                discrete_outcome = case["discrete_outcome"]

                if discrete_outcome:
                    y, T, X, true_eff = self._get_data(discrete_outcome=True)

                    models = [('ldml', LinearDML(model_y=clf(), model_t=clf(), discrete_treatment=True,
                                                 discrete_outcome=discrete_outcome, cv=3)),
                            ('sldml', SparseLinearDML(model_y=clf(), model_t=clf(), discrete_treatment=True,
                                                      discrete_outcome=discrete_outcome,
                                                      featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                                      cv=3)),
                            ('drlearner', DRLearner(model_propensity=clf(), model_regression=clf(), model_final=reg(),
                                                    discrete_outcome=discrete_outcome, cv=3)),
                            ('rlearner', NonParamDML(model_y=clf(), model_t=clf(), model_final=reg(),
                                                    discrete_treatment=True, discrete_outcome=discrete_outcome, cv=3)),
                            ('dml3dlasso', DML(model_y=clf(), model_t=clf(), model_final=reg(), discrete_treatment=True,
                                               discrete_outcome=discrete_outcome,
                                               featurizer=PolynomialFeatures(degree=3), cv=3)),
                            # SLearner as baseline for rootpehe score - not enough variation in rscore w/ above models
                            ('slearner', SLearner(overall_model=reg())),
                            ]

                else:
                    y, T, X, true_eff = self._get_data()

                    models = [('ldml', LinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True, cv=3)),
                            ('sldml', SparseLinearDML(model_y=reg(), model_t=clf(), discrete_treatment=True,
                                                        featurizer=PolynomialFeatures(degree=2, include_bias=False),
                                                        cv=3)),
                            ('xlearner', XLearner(models=reg(), cate_models=reg(), propensity_model=clf())),
                            ('dalearner', DomainAdaptationLearner(models=reg(), final_models=reg(),
                                                                  propensity_model=clf())),
                            ('slearner', SLearner(overall_model=reg())),
                            ('tlearner', TLearner(models=reg())),
                            ('drlearner', DRLearner(model_propensity=clf(), model_regression=reg(),
                                                    model_final=reg(), cv=3)),
                            ('rlearner', NonParamDML(model_y=reg(), model_t=clf(), model_final=reg(),
                                                    discrete_treatment=True, cv=3)),
                            ('dml3dlasso', DML(model_y=reg(), model_t=clf(), model_final=reg(),
                                               discrete_treatment=True, featurizer=PolynomialFeatures(degree=3), cv=3))
                            ]

                (X_train, X_val, T_train, T_val,
                Y_train, Y_val, _, true_eff_val) = train_test_split(X, T, y, true_eff, test_size=.4)

                models = Parallel(n_jobs=1, verbose=1)(delayed(_fit_model)(name, mdl,
                                                                        Y_train, T_train, X_train)
                                                    for name, mdl in models)

                if discrete_outcome:
                    scorer = RScorer(model_y=clf(), model_t=clf(),
                                    discrete_treatment=True, discrete_outcome=discrete_outcome,
                                    cv=3, mc_iters=2, mc_agg='median')
                else:
                    scorer = RScorer(model_y=reg(), model_t=clf(),
                                    discrete_treatment=True, cv=3,
                                    mc_iters=2, mc_agg='median')

                scorer.fit(Y_val, T_val, X=X_val)
                rscore = [scorer.score(mdl) for _, mdl in models]
                rootpehe_score = [np.sqrt(np.mean((true_eff_val.flatten() - mdl.effect(X_val).flatten())**2))
                                for _, mdl in models]
                # Checking neg corr between rscore and rootpehe (precision in estimating heterogeneous effects)
                assert LinearRegression().fit(np.array(rscore).reshape(-1, 1), np.array(rootpehe_score)).coef_ < 0.5
                mdl, _ = scorer.best_model([mdl for _, mdl in models])
                rootpehe_best = np.sqrt(np.mean((true_eff_val.flatten() - mdl.effect(X_val).flatten())**2))
                # Checking best model selection behaves as intended
                assert rootpehe_best < 1.5 * np.min(rootpehe_score) + 0.05
                mdl, _ = scorer.ensemble([mdl for _, mdl in models])
                rootpehe_ensemble = np.sqrt(np.mean((true_eff_val.flatten() - mdl.effect(X_val).flatten())**2))
                # Checking cate ensembling behaves as intended
                assert rootpehe_ensemble < 1.5 * np.min(rootpehe_score) + 0.05
