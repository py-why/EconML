# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from ..dml import LinearDML
from sklearn.base import clone
import numpy as np
from scipy.special import softmax
from .ensemble_cate import EnsembleCateEstimator


class RScorer:
    """ Scorer based on the RLearner loss. Fits residual models at fit time and calculates
    residuals of the evaluation data in a cross-fitting manner::

        Yres = Y - E[Y|X, W]
        Tres = T - E[T|X, W]

    Then for any given cate model calculates the loss::

        loss(cate) = E_n[(Yres - <cate(X), Tres>)^2]

    Also calculates a baseline loss based on a constant treatment effect model, i.e.::

        base_loss = min_{theta} E_n[(Yres - <theta, Tres>)^2]

    Returns an analogue of the R-square score for regression::

        score = 1 - loss(cate) / base_loss

    This corresponds to the extra variance of the outcome explained by introducing heterogeneity
    in the effect as captured by the cate model, as opposed to always predicting a constant effect.
    A negative score, means that the cate model performs even worse than a constant effect model
    and hints at overfitting during training of the cate model.

    This method was also advocated in recent work of [Schuleretal2018]_ when compared among several alternatives
    for causal model selection and introduced in the work of [NieWager2017]_.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, optional (Default=2)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(concat[W, X], T)` to generate the splits. If all
        W, X are None, then we call `split(ones((T.shape[0], 1)), T)`.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    References
    ----------
    .. [NieWager2017] X. Nie and S. Wager.
        Quasi-Oracle Estimation of Heterogeneous Treatment Effects.
        arXiv preprint arXiv:1712.04912, 2017.
        `<https://arxiv.org/pdf/1712.04912.pdf>`_

    .. [Schuleretal2018] Alejandro Schuler, Michael Baiocchi, Robert Tibshirani, Nigam Shah.
        "A comparison of methods for model selection when estimating individual treatment effects."
        Arxiv, 2018
        `<https://arxiv.org/pdf/1804.05146.pdf>`_

    """

    def __init__(self, *,
                 model_y,
                 model_t,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.discrete_treatment = discrete_treatment
        self.cv = cv
        self.categories = categories
        self.random_state = random_state
        self.mc_iters = mc_iters
        self.mc_agg = mc_agg

    def fit(self, y, T, X=None, W=None, sample_weight=None, groups=None):
        """

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: optional (n × dₓ) matrix
            Features for each sample
        W: optional (n × d_w) matrix
            Controls for each sample
        sample_weight: optional (n,) vector
            Weights for each row
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.

        Returns
        -------
        self
        """
        if X is None:
            raise ValueError("X cannot be None for the RScorer!")

        self.lineardml_ = LinearDML(model_y=self.model_y,
                                    model_t=self.model_t,
                                    cv=self.cv,
                                    discrete_treatment=self.discrete_treatment,
                                    categories=self.categories,
                                    random_state=self.random_state,
                                    mc_iters=self.mc_iters,
                                    mc_agg=self.mc_agg)
        self.lineardml_.fit(y, T, X=None, W=np.hstack([v for v in [X, W] if v is not None]),
                            sample_weight=sample_weight, groups=groups, cache_values=True)
        self.base_score_ = self.lineardml_.score_
        self.dx_ = X.shape[1]
        return self

    def score(self, cate_model):
        """
        Parameters
        ----------
        cate_model : instance of fitted BaseCateEstimator

        Returns
        -------
        score : double
            An analogue of the R-square loss for the causal setting.
        """
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
        """ Chooses the best among a list of models

        Parameters
        ----------
        cate_models : list of instances of fitted BaseCateEstimator
        return_scores : bool, optional (default=False)
            Whether to return the list scores of each model
        Returns
        -------
        best_model : instance of fitted BaseCateEstimator
            The model that achieves the best score
        best_score : double
            The score of the best model
        scores : list of double
            The list of scores for each of the input models. Returned only if `return_scores=True`.
        """
        rscores = [self.score(mdl) for mdl in cate_models]
        best = np.nanargmax(rscores)
        if return_scores:
            return cate_models[best], rscores[best], rscores
        else:
            return cate_models[best], rscores[best]

    def ensemble(self, cate_models, eta=1000.0, return_scores=False):
        """ Ensembles a list of models based on their performance

        Parameters
        ----------
        cate_models : list of instances of fitted BaseCateEstimator
        eta : double, optional (default=1000)
            The soft-max parameter for the ensemble
        return_scores : bool, optional (default=False)
            Whether to return the list scores of each model
        Returns
        -------
        ensemble_model : instance of fitted EnsembleCateEstimator
            A fitted ensemble cate model that calculates effects based on a weighted
            version of the input cate models, weighted by a softmax of their score
            performance
        ensemble_score : double
            The score of the ensemble model
        scores : list of double
            The list of scores for each of the input models. Returned only if `return_scores=True`.
        """
        rscores = np.array([self.score(mdl) for mdl in cate_models])
        goodinds = np.isfinite(rscores)
        weights = softmax(eta * rscores[goodinds])
        goodmodels = [mdl for mdl, good in zip(cate_models, goodinds) if good]
        ensemble = EnsembleCateEstimator(cate_models=goodmodels, weights=weights)
        ensemble_score = self.score(ensemble)
        if return_scores:
            return ensemble, ensemble_score, rscores
        else:
            return ensemble, ensemble_score
