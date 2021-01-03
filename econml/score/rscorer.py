from ..dml import LinearDML
from sklearn.base import clone
import numpy as np
from scipy.special import softmax
from .ensemble_cate import EnsembleCateEstimator


class RScorer:
    """ Scorer based on the RLearner loss. Fits residual models at fit time and calculates
    residuals of the evaluation data, Yres, Tres. Then for any given cate model calculates
    the loss::

        loss(cate) = E_n[(Yres - cate(X)'Tres)^2]

    Also calculates a baseline loss based on a constant treatment effect model, i.e.::

        base_loss = min_{theta} E_n[(Yres - theta'Tres)^2]

    Returns an analogue of the R-square score for regression::

        score = 1 - loss(cate) / base_loss

    This corresponds to the extra variance of the outcome explained by introducing heterogeneity
    in the effect as captured by the cate model, as opposed to always predicting a constant effect.
    A negative score, means that the cate model performs even worse than a constant effect model
    and hints at overfitting during training of the cate model.
    """

    def __init__(self, *, model_y, model_t,
                 discrete_treatment=False, n_splits=2,
                 categories='auto', random_state=None,
                 monte_carlo_iterations=None):
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.discrete_treatment = discrete_treatment
        self.n_splits = n_splits
        self.categories = categories
        self.random_state = random_state
        self.monte_carlo_iterations = monte_carlo_iterations

    def fit(self, y, T, X=None, W=None, sample_weight=None):
        if X is None:
            raise ValueError("X cannot be None for the RScorer!")

        self.lineardml_ = LinearDML(model_y=self.model_y,
                                    model_t=self.model_t,
                                    n_splits=self.n_splits,
                                    discrete_treatment=self.discrete_treatment,
                                    categories=self.categories,
                                    random_state=self.random_state,
                                    monte_carlo_iterations=self.monte_carlo_iterations)
        self.lineardml_.fit(y, T, X=None, W=np.hstack([v for v in [X, W] if v is not None]),
                            sample_weight=sample_weight, cache_values=True)
        self.base_score_ = self.lineardml_.score_
        self.dx_ = X.shape[1]

    def score(self, cate_model):
        Y_res, T_res = self.lineardml_._cached_values.nuisances
        X = self.lineardml_._cached_values.W[:, :self.dx_]
        sample_weight = self.lineardml_._cached_values.sample_weight
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        effects = cate_model.const_marginal_effect(X).reshape((-1, Y_res.shape[1], T_res.shape[1]))
        Y_res_pred = np.einsum('ijk,ik->ij', effects, T_res).reshape(Y_res.shape)
        if sample_weight is not None:
            return 1 - np.mean(np.average((Y_res - Y_res_pred)**2, weights=sample_weight, axis=0)) / self.base_score_
        else:
            return 1 - np.mean((Y_res - Y_res_pred) ** 2) / self.base_score_

    def best_model(self, cate_models, return_scores=False):
        rscores = [self.score(mdl) for mdl in cate_models]
        best = np.argmax(rscores)
        if return_scores:
            return cate_models[best], rscores[best], rscores
        else:
            return cate_models[best], rscores[best]

    def ensemble(self, cate_models, eta=1000.0, return_scores=False):
        rscores = np.array([self.score(mdl) for mdl in cate_models])
        weights = softmax(eta * rscores)
        ensemble = EnsembleCateEstimator(cate_models=cate_models, weights=weights)
        ensemble_score = self.score(ensemble)
        if return_scores:
            return ensemble, ensemble_score, rscores
        else:
            return ensemble, ensemble_score
