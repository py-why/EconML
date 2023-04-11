# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from ..dr import DRLearner
from sklearn.base import clone
import numpy as np
from scipy.special import softmax
from .ensemble_cate import EnsembleCateEstimator


class DRScorer:
    """ Scorer based on the DRLearner loss. Fits regression model g (using T-Learner) and propensity model p at fit time
    and calculates the regression and propensity of the evaluation data::

        g (model_regression) = E[Y | X, W, T]
        
        p (model_propensity) = Pr[T | X, W]

        Ydr(g,p) = g  + (Y - g ) / p * T

    Then for any given cate model calculates the loss::

        loss(cate) = E_n[(Ydr(g, p) - cate(X))^2]

    Also calculates a baseline loss based on a constant treatment effect model, i.e.::

        base_loss = min_{theta} E_n[(Ydr(g, p) - theta)^2]

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
    model_propensity : scikit-learn classifier or 'auto', default 'auto'
        Estimator for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.
        Must implement `fit` and `predict_proba` methods. The `fit` method must be able to accept X and T,
        where T is a shape (n, ) array.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV` will be chosen.

    model_regression : scikit-learn regressor or 'auto', default 'auto'
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment. Must implement `fit` and
        `predict` methods. If different models per treatment arm are desired, see the
        :class:`.MultiModelWrapper` helper class.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_final :
        estimator for the final cate model. Trained on regressing the doubly robust potential outcomes
        on (features X).

        - If X is None, then the fit method of model_final should be able to handle X=None.
        - If featurizer is not None and X is not None, then it is trained on the outcome of
          featurizer.fit_transform(X).
        - If multitask_model_final is True, then this model must support multitasking
          and it is trained by regressing all doubly robust target outcomes on (featurized) features simultanteously.
        - The output of the predict(X) of the trained model will contain the CATEs for each treatment compared to
          baseline treatment (lexicographically smallest). If multitask_model_final is False, it is assumed to be a
          mono-task model and a separate clone of the model is trained for each outcome. Then predict(X) of the t-th
          clone will be the CATE of the t-th lexicographically ordered treatment compared to the baseline.

    multitask_model_final : bool, default False
        Whether the model_final should be treated as a multi-task model. See description of model_final.

    featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    min_propensity : float, default ``1e-6``
        The minimum propensity at which to clip propensity estimates to avoid dividing by zero.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, default 2
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

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
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
                 model_propensity='auto',
                 model_regression='auto',
                 model_final=StatsModelsLinearRegression(),
                 multitask_model_final=False,
                 featurizer=None,
                 min_propensity=1e-6,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_propensity = clone(model_propensity, safe=False)
        self.model_regression = clone(model_regression, safe=False)
        self.model_final = clone(model_final, safe=False)
        self.multitask_model_final = multitask_model_final
        self.featurizer = clone(featurizer, safe=False)
        self.min_propensity = min_propensity
        self.categories = categories
        self.cv = cv
        self.mc_iters = mc_iters
        self.mc_agg = mc_agg
        self.random_state = random_state

    def fit(self, y, T, X=None, W=None, sample_weight=None, groups=None):
        """

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X:  (n × dₓ) matrix, optional
            Features for each sample
        W:  (n × d_w) matrix, optional
            Controls for each sample
        sample_weight:  (n,) vector, optional
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
            raise ValueError("X cannot be None for the DRScorer!")

        self.drlearner_ = DRLearner(model_propensity=self.model_propensity,
                                    model_regression=self.model_regression,
                                    model_final=self.model_final,
                                    multitask_model_final=self.multitask_model_final,
                                    featurizer=self.featurizer,
                                    min_propensity=self.min_propensity,
                                    categories=self.categories,
                                    cv=self.cv,
                                    mc_iters=self.mc_iters,
                                    mc_agg=self.mc_agg,
                                    random_state=self.random_state)
        self.drlearner_.fit(y, T, X=None, W=np.hstack([v for v in [X, W] if v is not None]),
                            sample_weight=sample_weight, groups=groups, cache_values=True)
        self.base_score_ = self.drlearner_.score_
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
            An analogue of the DR-square loss for the causal setting.
        """
        Ydr = self.drlearner_.model_final 
        X = self.drlearner_._cached_values.W[:, :self.dx_]
        sample_weight = self.drlearner_._cached_values.sample_weight
        if Ydr.ndim == 1:
            Ydr = Ydr.reshape((-1, 1))
            
        cate = cate_model.const_marginal_effect(X).reshape((-1, Ydr.shape[1]))

        if sample_weight is not None:
            return 1 - np.mean(np.average((Ydr - cate)**2, weights=sample_weight, axis=0)) / self.base_score_
        else:
            return 1 - np.mean((Ydr - cate) ** 2) / self.base_score_

    def best_model(self, cate_models, return_scores=False):
        """ Chooses the best among a list of models

        Parameters
        ----------
        cate_models : list of instance of fitted BaseCateEstimator
        return_scores : bool, default False
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
        cate_models : list of instance of fitted BaseCateEstimator
        eta : double, default 1000
            The soft-max parameter for the ensemble
        return_scores : bool, default False
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
        drscores = np.array([self.score(mdl) for mdl in cate_models])
        goodinds = np.isfinite(drscores)
        weights = softmax(eta * drscores[goodinds])
        goodmodels = [mdl for mdl, good in zip(cate_models, goodinds) if good]
        ensemble = EnsembleCateEstimator(cate_models=goodmodels, weights=weights)
        ensemble_score = self.score(ensemble)
        if return_scores:
            return ensemble, ensemble_score, drscores
        else:
            return ensemble, ensemble_score
