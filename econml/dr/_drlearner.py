# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Doubly Robust Learner. The method uses the doubly robust correction to construct doubly
robust estimates of all the potential outcomes of each samples. Then estimates a CATE model
by regressing the potential outcome differences on the heterogeneity features X.

References
----------

Dylan Foster, Vasilis Syrgkanis (2019).
    Orthogonal Statistical Learning.
    ACM Conference on Learning Theory. https://arxiv.org/abs/1901.09036

Robins, J.M., Rotnitzky, A., and Zhao, L.P. (1994).
    Estimation of regression coefficients when some regressors are not always observed.
    Journal of the American Statistical Association 89,846–866.

Bang, H. and Robins, J.M. (2005).
    Doubly robust estimation in missing data and causal inference models.
    Biometrics 61,962–972.

Tsiatis AA (2006).
    Semiparametric Theory and Missing Data.
    New York: Springer; 2006.

.. testcode::
    :hide:

    import numpy as np
    import scipy.special
    np.set_printoptions(suppress=True)

"""

from warnings import warn
from copy import deepcopy

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import (LassoCV, LinearRegression,
                                  LogisticRegressionCV)
from sklearn.ensemble import RandomForestRegressor

from .._ortho_learner import _OrthoLearner
from .._cate_estimator import (DebiasedLassoCateEstimatorDiscreteMixin,
                               ForestModelFinalCateEstimatorDiscreteMixin,
                               StatsModelsCateEstimatorDiscreteMixin, LinearCateEstimator)
from ..inference import GenericModelFinalInferenceDiscrete
from ..grf import RegressionForest
from ..sklearn_extensions.linear_model import (
    DebiasedLasso, StatsModelsLinearRegression, WeightedLassoCVWrapper)
from ..utilities import (_deprecate_positional, check_high_dimensional,
                         filter_none_kwargs, fit_with_groups, inverse_onehot)
from .._shap import _shap_explain_multitask_model_cate, _shap_explain_model_cate


class _ModelNuisance:
    def __init__(self, model_propensity, model_regression, min_propensity):
        self._model_propensity = model_propensity
        self._model_regression = model_regression
        self._min_propensity = min_propensity

    def _combine(self, X, W):
        return np.hstack([arr for arr in [X, W] if arr is not None])

    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, groups=None):
        if Y.ndim != 1 and (Y.ndim != 2 or Y.shape[1] != 1):
            raise ValueError("The outcome matrix must be of shape ({0}, ) or ({0}, 1), "
                             "instead got {1}.".format(len(X), Y.shape))
        if (X is None) and (W is None):
            raise AttributeError("At least one of X or W has to not be None!")
        if np.any(np.all(T == 0, axis=0)) or (not np.any(np.all(T == 0, axis=1))):
            raise AttributeError("Provided crossfit folds contain training splits that " +
                                 "don't contain all treatments")
        XW = self._combine(X, W)
        filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight)

        fit_with_groups(self._model_propensity, XW, inverse_onehot(T), groups=groups, **filtered_kwargs)
        fit_with_groups(self._model_regression, np.hstack([XW, T]), Y, groups=groups, **filtered_kwargs)
        return self

    def score(self, Y, T, X=None, W=None, *, sample_weight=None, groups=None):
        XW = self._combine(X, W)
        filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight)

        if hasattr(self._model_propensity, 'score'):
            propensity_score = self._model_propensity.score(XW, inverse_onehot(T), **filtered_kwargs)
        else:
            propensity_score = None
        if hasattr(self._model_regression, 'score'):
            regression_score = self._model_regression.score(np.hstack([XW, T]), Y, **filtered_kwargs)
        else:
            regression_score = None

        return propensity_score, regression_score

    def predict(self, Y, T, X=None, W=None, *, sample_weight=None, groups=None):
        XW = self._combine(X, W)
        propensities = np.maximum(self._model_propensity.predict_proba(XW), self._min_propensity)
        n = T.shape[0]
        Y_pred = np.zeros((T.shape[0], T.shape[1] + 1))
        T_counter = np.zeros(T.shape)
        Y_pred[:, 0] = self._model_regression.predict(np.hstack([XW, T_counter])).reshape(n)
        Y_pred[:, 0] += (Y.reshape(n) - Y_pred[:, 0]) * np.all(T == 0, axis=1) / propensities[:, 0]
        for t in np.arange(T.shape[1]):
            T_counter = np.zeros(T.shape)
            T_counter[:, t] = 1
            Y_pred[:, t + 1] = self._model_regression.predict(np.hstack([XW, T_counter])).reshape(n)
            Y_pred[:, t + 1] += (Y.reshape(n) - Y_pred[:, t + 1]) * (T[:, t] == 1) / propensities[:, t + 1]
        return Y_pred.reshape(Y.shape + (T.shape[1] + 1,))


class _ModelFinal:
    # Coding Remark: The reasoning around the multitask_model_final could have been simplified if
    # we simply wrapped the model_final with a MultiOutputRegressor. However, because we also want
    # to allow even for model_final objects whose fit(X, y) can accept X=None
    # (e.g. the StatsModelsLinearRegression), we cannot take that route, because the MultiOutputRegressor
    # checks that X is 2D array.
    def __init__(self, model_final, featurizer, multitask_model_final):
        self._model_final = clone(model_final, safe=False)
        self._featurizer = clone(featurizer, safe=False)
        self._multitask_model_final = multitask_model_final
        return

    def fit(self, Y, T, X=None, W=None, *, nuisances, sample_weight=None, sample_var=None):
        Y_pred, = nuisances
        self.d_y = Y_pred.shape[1:-1]  # track whether there's a Y dimension (must be a singleton)
        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.fit_transform(X)
        filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight, sample_var=sample_var)
        if self._multitask_model_final:
            ys = Y_pred[..., 1:] - Y_pred[..., [0]]  # subtract control results from each other arm
            if self.d_y:  # need to squeeze out singleton so that we fit on 2D array
                ys = ys.squeeze(1)
            self.model_cate = self._model_final.fit(X, ys, **filtered_kwargs)
        else:
            self.models_cate = [clone(self._model_final, safe=False).fit(X, Y_pred[..., t] - Y_pred[..., 0],
                                                                         **filtered_kwargs)
                                for t in np.arange(1, Y_pred.shape[-1])]
        return self

    def predict(self, X=None):
        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.transform(X)
        if self._multitask_model_final:
            pred = self.model_cate.predict(X)
            if self.d_y:  # need to reintroduce singleton Y dimension
                return pred[:, np.newaxis, :]
            return pred
        else:
            preds = np.array([mdl.predict(X).reshape((-1,) + self.d_y) for mdl in self.models_cate])
            return np.moveaxis(preds, 0, -1)  # move treatment dim to end

    def score(self, Y, T, X=None, W=None, *, nuisances, sample_weight=None, sample_var=None):
        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.transform(X)
        Y_pred, = nuisances
        if self._multitask_model_final:
            return np.mean(np.average((Y_pred[..., 1:] - Y_pred[..., [0]] - self.model_cate.predict(X))**2,
                                      weights=sample_weight, axis=0))
        else:
            return np.mean([np.average((Y_pred[..., t] - Y_pred[..., 0] -
                                        self.models_cate[t - 1].predict(X))**2,
                                       weights=sample_weight, axis=0)
                            for t in np.arange(1, Y_pred.shape[-1])])


class DRLearner(_OrthoLearner):
    """
    CATE estimator that uses doubly-robust correction techniques to account for
    covariate shift (selection bias) between the treatment arms. The estimator is a special
    case of an :class:`._OrthoLearner` estimator, so it follows the two
    stage process, where a set of nuisance functions are estimated in the first stage in a crossfitting
    manner and a final stage estimates the CATE model. See the documentation of
    :class:`._OrthoLearner` for a description of this two stage process.

    In this estimator, the CATE is estimated by using the following estimating equations. If we let:

    .. math ::
        Y_{i, t}^{DR} = E[Y | X_i, W_i, T_i=t]\
            + \\frac{Y_i - E[Y | X_i, W_i, T_i=t]}{Pr[T_i=t | X_i, W_i]} \\cdot 1\\{T_i=t\\}

    Then the following estimating equation holds:

    .. math ::
        E\\left[Y_{i, t}^{DR} - Y_{i, 0}^{DR} | X_i\\right] = \\theta_t(X_i)

    Thus if we estimate the nuisance functions :math:`h(X, W, T) = E[Y | X, W, T]` and
    :math:`p_t(X, W)=Pr[T=t | X, W]` in the first stage, we can estimate the final stage cate for each
    treatment t, by running a regression, regressing :math:`Y_{i, t}^{DR} - Y_{i, 0}^{DR}` on :math:`X_i`.

    The problem of estimating the nuisance function :math:`p` is a simple multi-class classification
    problem of predicting the label :math:`T` from :math:`X, W`. The :class:`.DRLearner`
    class takes as input the parameter ``model_propensity``, which is an arbitrary scikit-learn
    classifier, that is internally used to solve this classification problem.

    The second nuisance function :math:`h` is a simple regression problem and the :class:`.DRLearner`
    class takes as input the parameter ``model_regressor``, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem.

    The final stage is multi-task regression problem with outcomes the labels :math:`Y_{i, t}^{DR} - Y_{i, 0}^{DR}`
    for each non-baseline treatment t. The :class:`.DRLearner` takes as input parameter
    ``model_final``, which is any scikit-learn regressor that is internally used to solve this multi-task
    regresion problem. If the parameter ``multitask_model_final`` is False, then this model is assumed
    to be a mono-task regressor, and separate clones of it are used to solve each regression target
    separately.

    Parameters
    ----------
    model_propensity : scikit-learn classifier or 'auto', optional (default='auto')
        Estimator for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.
        Must implement `fit` and `predict_proba` methods. The `fit` method must be able to accept X and T,
        where T is a shape (n, ) array.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV` will be chosen.

    model_regression : scikit-learn regressor or 'auto', optional (default='auto')
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

    multitask_model_final : bool, optional, default False
        Whether the model_final should be treated as a multi-task model. See description of model_final.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    min_propensity : float, optional, default ``1e-6``
        The minimum propensity at which to clip propensity estimates to avoid dividing by zero.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, optional (default is 2)
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

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    Examples
    --------
    A simple example with the default models:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dr import DRLearner

        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.001
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner()
        est.fit(y, T, X=X, W=None)

    >>> est.const_marginal_effect(X[:2])
    array([[0.511640..., 1.144004...],
           [0.378140..., 0.613143...]])
    >>> est.effect(X[:2], T0=0, T1=1)
    array([0.511640..., 0.378140...])
    >>> est.score_
    5.11238581...
    >>> est.score(y, T, X=X)
    5.78673506...
    >>> est.model_cate(T=1).coef_
    array([0.434910..., 0.010226..., 0.047913...])
    >>> est.model_cate(T=2).coef_
    array([ 0.863723...,  0.086946..., -0.022288...])
    >>> est.cate_feature_names()
    <BLANKLINE>
    >>> [mdl.coef_ for mdl in est.models_regression]
    [array([ 1.472...,  0.001..., -0.011...,  0.698..., 2.049...]),
     array([ 1.455..., -0.002...,  0.005...,  0.677...,  1.998...])]
    >>> [mdl.coef_ for mdl in est.models_propensity]
    [array([[-0.747...,  0.153..., -0.018...],
           [ 0.083..., -0.110..., -0.076...],
           [ 0.663..., -0.043... ,  0.094...]]),
     array([[-1.048...,  0.000...,  0.032...],
           [ 0.019...,  0.124..., -0.081...],
           [ 1.029..., -0.124...,  0.049...]])]

    Beyond default models:

    .. testcode::

        from sklearn.linear_model import LassoCV
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from econml.dr import DRLearner

        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        sigma = 0.01
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(0, sigma, size=(1000,))
        est = DRLearner(model_propensity=RandomForestClassifier(n_estimators=100, min_samples_leaf=10),
                        model_regression=RandomForestRegressor(n_estimators=100, min_samples_leaf=10),
                        model_final=LassoCV(cv=3),
                        featurizer=None)
        est.fit(y, T, X=X, W=None)

    >>> est.score_
    1.7...
    >>> est.const_marginal_effect(X[:3])
    array([[0.68...,  1.10...],
           [0.56...,  0.79...],
           [0.34...,  0.10...]])
    >>> est.model_cate(T=2).coef_
    array([0.74..., 0.        , 0.        ])
    >>> est.model_cate(T=2).intercept_
    1.9...
    >>> est.model_cate(T=1).coef_
    array([0.24..., 0.00..., 0.        ])
    >>> est.model_cate(T=1).intercept_
    0.94...

    Attributes
    ----------
    score_ : float
        The MSE in the final doubly robust potential outcome regressions, i.e.

        .. math::
            \\frac{1}{n_t} \\sum_{t=1}^{n_t} \\frac{1}{n} \\sum_{i=1}^n (Y_{i, t}^{DR} - \\hat{\\theta}_t(X_i))^2

        where n_t is the number of treatments (excluding control).

        If `sample_weight` is not None at fit time, then a weighted average across samples is returned.


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
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_propensity = clone(model_propensity, safe=False)
        self.model_regression = clone(model_regression, safe=False)
        self.model_final = clone(model_final, safe=False)
        self.multitask_model_final = multitask_model_final
        self.featurizer = clone(featurizer, safe=False)
        self.min_propensity = min_propensity
        super().__init__(cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         discrete_treatment=True,
                         discrete_instrument=False,  # no instrument, so doesn't matter
                         categories=categories,
                         random_state=random_state)

    def _get_inference_options(self):
        options = super()._get_inference_options()
        if not self.multitask_model_final:
            options.update(auto=GenericModelFinalInferenceDiscrete)
        else:
            options.update(auto=lambda: None)
        return options

    def _gen_ortho_learner_model_nuisance(self):
        if self.model_propensity == 'auto':
            model_propensity = LogisticRegressionCV(cv=3, solver='lbfgs', multi_class='auto',
                                                    random_state=self.random_state)
        else:
            model_propensity = clone(self.model_propensity, safe=False)

        if self.model_regression == 'auto':
            model_regression = WeightedLassoCVWrapper(cv=3, random_state=self.random_state)
        else:
            model_regression = clone(self.model_regression, safe=False)

        return _ModelNuisance(model_propensity, model_regression, self.min_propensity)

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_ortho_learner_model_final(self):
        return _ModelFinal(self._gen_model_final(), self._gen_featurizer(), self.multitask_model_final)

    @_deprecate_positional("X and W should be passed by keyword only. In a future release "
                           "we will disallow passing X and W by position.", ['X', 'W'])
    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`).

        Returns
        -------
        self: DRLearner instance
        """
        # Replacing fit from _OrthoLearner, to enforce Z=None and improve the docstring
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    def score(self, Y, T, X=None, W=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted residual nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calculates the MSE of the final residual Y on residual T regression.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample

        Returns
        -------
        score: float
            The MSE of the final CATE model on the new data.
        """
        # Replacing score from _OrthoLearner, to enforce Z=None and improve the docstring
        return super().score(Y, T, X=X, W=W)

    @property
    def multitask_model_cate(self):
        """
        Get the fitted final CATE model.

        Returns
        -------
        multitask_model_cate: object of type(`model_final`)
            An instance of the model_final object that was fitted after calling fit which corresponds whose
            vector of outcomes correspond to the CATE model for each treatment, compared to baseline.
            Available only when multitask_model_final=True.
        """
        if not self.ortho_learner_model_final_._multitask_model_final:
            raise AttributeError("Separate CATE models were fitted for each treatment! Use model_cate.")
        return self.ortho_learner_model_final_.model_cate

    def model_cate(self, T=1):
        """
        Get the fitted final CATE model.

        Parameters
        ----------
        T: alphanumeric
            The treatment with respect to which we want the fitted CATE model.

        Returns
        -------
        model_cate: object of type(model_final)
            An instance of the model_final object that was fitted after calling fit which corresponds
            to the CATE model for treatment T=t, compared to baseline. Available when multitask_model_final=False.
        """
        if self.ortho_learner_model_final_._multitask_model_final:
            raise AttributeError("A single multitask model was fitted for all treatments! Use multitask_model_cate.")
        _, T = self._expand_treatments(None, T)
        ind = inverse_onehot(T).item() - 1
        assert ind >= 0, "No model was fitted for the control"
        return self.ortho_learner_model_final_.models_cate[ind]

    @property
    def models_propensity(self):
        """
        Get the fitted propensity models.

        Returns
        -------
        models_propensity: list of objects of type(`model_propensity`)
            A list of instances of the `model_propensity` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model_propensity for mdl in super().models_nuisance_]

    @property
    def models_regression(self):
        """
        Get the fitted regression models.

        Returns
        -------
        model_regression: list of objects of type(`model_regression`)
            A list of instances of the model_regression object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model_regression for mdl in super().models_nuisance_]

    @property
    def nuisance_scores_propensity(self):
        """Gets the score for the propensity model on out-of-sample training data"""
        return self.nuisance_scores_[0]

    @property
    def nuisance_scores_regression(self):
        """Gets the score for the regression model on out-of-sample training data"""
        return self.nuisance_scores_[1]

    @property
    def featurizer_(self):
        """
        Get the fitted featurizer.

        Returns
        -------
        featurizer: object of type(`featurizer`)
            An instance of the fitted featurizer that was used to preprocess X in the final CATE model training.
            Available only when featurizer is not None and X is not None.
        """
        return self.ortho_learner_model_final_._featurizer

    def cate_feature_names(self, feature_names=None):
        """
        Get the output feature names.

        Parameters
        ----------
        feature_names: list of strings of length X.shape[1] or None
            The names of the input features. If None and X is a dataframe, it defaults to the column names
            from the dataframe.

        Returns
        -------
        out_feature_names: list of strings or None
            The names of the output features :math:`\\phi(X)`, i.e. the features with respect to which the
            final CATE model for each treatment is linear. It is the names of the features that are associated
            with each entry of the :meth:`coef_` parameter. Available only when the featurizer is not None and has
            a method: `get_feature_names(feature_names)`. Otherwise None is returned.
        """
        if self._d_x is None:
            # Handles the corner case when X=None but featurizer might be not None
            return None
        if feature_names is None:
            feature_names = self._input_names["feature_names"]
        if self.featurizer_ is None:
            return feature_names
        elif hasattr(self.featurizer_, 'get_feature_names'):
            # This fails if X=None and featurizer is not None, but that case is handled above
            return self.featurizer_.get_feature_names(feature_names)
        else:
            raise AttributeError("Featurizer does not have a method: get_feature_names!")

    @property
    def model_final_(self):
        return self.ortho_learner_model_final_._model_final

    @property
    def fitted_models_final(self):
        return self.ortho_learner_model_final_.models_cate

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        if self.featurizer_ is not None:
            F = self.featurizer_.transform(X)
        else:
            F = X
        feature_names = self.cate_feature_names(feature_names)

        if self.ortho_learner_model_final_._multitask_model_final:
            return _shap_explain_multitask_model_cate(self.const_marginal_effect, self.multitask_model_cate, F,
                                                      self._d_t, self._d_y,
                                                      feature_names=feature_names,
                                                      treatment_names=treatment_names,
                                                      output_names=output_names,
                                                      input_names=self._input_names,
                                                      background_samples=background_samples)
        else:
            return _shap_explain_model_cate(self.const_marginal_effect, self.fitted_models_final,
                                            F, self._d_t, self._d_y,
                                            feature_names=feature_names,
                                            treatment_names=treatment_names,
                                            output_names=output_names,
                                            input_names=self._input_names,
                                            background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__


class LinearDRLearner(StatsModelsCateEstimatorDiscreteMixin, DRLearner):
    """
    Special case of the :class:`.DRLearner` where the final stage
    is a Linear Regression on a low dimensional set of features. In this case, inference
    can be performed via the asymptotic normal characterization of the estimated parameters.
    This is computationally faster than bootstrap inference. To do this, just leave the setting ``inference='auto'``
    unchanged, or explicitly set ``inference='statsmodels'`` or alter the covariance type calculation via
    ``inference=StatsModelsInferenceDiscrete(cov_type='HC1)``.

    More concretely, this estimator assumes that the final cate model for each treatment takes a linear form:

    .. math ::
        \\theta_t(X) = \\left\\langle \\theta_t, \\phi(X) \\right\\rangle + \\beta_t

    where :math:`\\phi(X)` is the outcome features of the featurizers, or `X` if featurizer is None. :math:`\\beta_t`
    is a an intercept of the CATE, which is included if ``fit_cate_intercept=True`` (Default). It fits this by
    running a standard ordinary linear regression (OLS), regressing the doubly robust outcome differences on X:

    .. math ::
        \\min_{\\theta_t, \\beta_t}\
        E_n\\left[\\left(Y_{i, t}^{DR} - Y_{i, 0}^{DR}\
            - \\left\\langle \\theta_t, \\phi(X_i) \\right\\rangle - \\beta_t\\right)^2\\right]

    Then inference can be performed via standard approaches for inference of OLS, via asympotic normal approximations
    of the estimated parameters. The default covariance estimator used is heteroskedasticity robust (HC1).
    For other methods see :class:`.StatsModelsInferenceDiscrete`. Use can invoke them by setting:
    ``inference=StatsModelsInferenceDiscrete(cov_type=...)``.

    This approach is valid even if the CATE model is not linear in :math:`\\phi(X)`. In this case it performs
    inference on the best linear approximation of the CATE model.

    Parameters
    ----------
    model_propensity : scikit-learn classifier or 'auto', optional (default='auto')
        Estimator for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.
        Must implement `fit` and `predict_proba` methods. The `fit` method must be able to accept X and T,
        where T is a shape (n, ) array.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV` will be chosen.

    model_regression : scikit-learn regressor or 'auto', optional (default='auto')
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment. Must implement `fit` and
        `predict` methods. If different models per treatment arm are desired, see the
        :class:`.MultiModelWrapper` helper class.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    min_propensity : float, optional, default ``1e-6``
        The minimum propensity at which to clip propensity estimates to avoid dividing by zero.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, optional (default is 2)
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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    Examples
    --------
    A simple example with the default models:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dr import DRLearner, LinearDRLearner

        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = LinearDRLearner()
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([ 0.409743...,  0.312604..., -0.127394...])
    >>> est.effect_interval(X[:3])
    (array([ 0.120682..., -0.102543..., -0.663246...]), array([0.698803..., 0.727753..., 0.408458...]))
    >>> est.coef_(T=1)
    array([ 0.450779..., -0.003214... ,  0.063884... ])
    >>> est.coef__interval(T=1)
    (array([ 0.202646..., -0.207195..., -0.104558...]), array([0.698911..., 0.200767..., 0.232326...]))
    >>> est.intercept_(T=1)
    0.88425066...
    >>> est.intercept__interval(T=1)
    (0.68655813..., 1.08194320...)

    Attributes
    ----------
    score_ : float
        The MSE in the final doubly robust potential outcome regressions, i.e.

        .. math::
            \\frac{1}{n_t} \\sum_{t=1}^{n_t} \\frac{1}{n} \\sum_{i=1}^n (Y_{i, t}^{DR} - \\hat{\\theta}_t(X_i))^2

        where n_t is the number of treatments (excluding control).

        If `sample_weight` is not None at fit time, then a weighted average across samples is returned.

    """

    def __init__(self, *,
                 model_propensity='auto',
                 model_regression='auto',
                 featurizer=None,
                 fit_cate_intercept=True,
                 min_propensity=1e-6,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.fit_cate_intercept = fit_cate_intercept
        super().__init__(model_propensity=model_propensity,
                         model_regression=model_regression,
                         model_final=None,
                         featurizer=featurizer,
                         multitask_model_final=False,
                         min_propensity=min_propensity,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=self.fit_cate_intercept)

    def _gen_ortho_learner_model_final(self):
        return _ModelFinal(self._gen_model_final(), self._gen_featurizer(), False)

    @_deprecate_positional("X and W should be passed by keyword only. In a future release "
                           "we will disallow passing X and W by position.", ['X', 'W'])
    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports ``'bootstrap'``
            (or an instance of :class:`.BootstrapInference`) and ``'statsmodels'``
            (or an instance of :class:`.StatsModelsInferenceDiscrete`).

        Returns
        -------
        self: DRLearner instance
        """
        # Replacing fit from DRLearner, to add statsmodels inference in docstring
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def fit_cate_intercept_(self):
        return self.model_final_.fit_intercept

    @property
    def multitask_model_cate(self):
        # Replacing this method which is invalid for this class, so that we make the
        # dosctring empty and not appear in the docs.
        return super().multitask_model_cate

    @property
    def multitask_model_final(self):
        return False

    @multitask_model_final.setter
    def multitask_model_final(self, value):
        if value:
            raise ValueError("Parameter `multitask_model_final` cannot change from `False` for this estimator!")

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class SparseLinearDRLearner(DebiasedLassoCateEstimatorDiscreteMixin, DRLearner):
    """
    Special case of the :class:`.DRLearner` where the final stage
    is a Debiased Lasso Regression. In this case, inference can be performed via the debiased lasso approach
    and its asymptotic normal characterization of the estimated parameters. This is computationally
    faster than bootstrap inference. Leave the default ``inference='auto'`` unchanged, or explicitly set
    ``inference='debiasedlasso'`` at fit time to enable inference via asymptotic normality.

    More concretely, this estimator assumes that the final cate model for each treatment takes a linear form:

    .. math ::
        \\theta_t(X) = \\left\\langle \\theta_t, \\phi(X) \\right\\rangle + \\beta_t

    where :math:`\\phi(X)` is the outcome features of the featurizers, or `X` if featurizer is None. :math:`\\beta_t`
    is a an intercept of the CATE, which is included if ``fit_cate_intercept=True`` (Default). It fits this by
    running a debiased lasso regression (i.e. :math:`\\ell_1`-penalized regression with debiasing),
    regressing the doubly robust outcome differences on X: i.e. first solves the penalized square loss problem

    .. math ::
        \\min_{\\theta_t, \\beta_t}\
        E_n\\left[\\left(Y_{i, t}^{DR} - Y_{i, 0}^{DR}\
            - \\left\\langle \\theta_t, \\phi(X_i) \\right\\rangle - \\beta_t\\right)^2\\right]\
                + \\lambda \\left\\lVert \\theta_t \\right\\rVert_1

    and then adds a debiasing correction to the solution. If alpha='auto' (recommended), then the penalty
    weight :math:`\\lambda` is set optimally via cross-validation.

    This approach is valid even if the CATE model is not linear in :math:`\\phi(X)`. In this case it performs
    inference on the best sparse linear approximation of the CATE model.

    Parameters
    ----------
    model_propensity : scikit-learn classifier or 'auto', optional (default='auto')
        Estimator for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.
        Must implement `fit` and `predict_proba` methods. The `fit` method must be able to accept X and T,
        where T is a shape (n, ) array.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV` will be chosen.

    model_regression : scikit-learn regressor or 'auto', optional (default='auto')
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment. Must implement `fit` and
        `predict` methods. If different models per treatment arm are desired, see the
        :class:`.MultiModelWrapper` helper class.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    alpha: string | float, optional., default 'auto'.
        CATE L1 regularization applied through the debiased lasso in the final model.
        'auto' corresponds to a CV form of the :class:`DebiasedLasso`.

    n_alphas : int, optional, default 100
        How many alphas to try if alpha='auto'

    alpha_cov : string | float, optional, default 'auto'
        The regularization alpha that is used when constructing the pseudo inverse of
        the covariance matrix Theta used to for correcting the final state lasso coefficient
        in the debiased lasso. Each such regression corresponds to the regression of one feature
        on the remainder of the features.

    n_alphas_cov : int, optional, default 10
        How many alpha_cov to try if alpha_cov='auto'.

    max_iter : int, optional, default 1000
        The maximum number of iterations in the Debiased Lasso

    tol : float, optional, default 1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    min_propensity : float, optional, default ``1e-6``
        The minimum propensity at which to clip propensity estimates to avoid dividing by zero.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, optional, default 2
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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    Examples
    --------
    A simple example with the default models:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dr import DRLearner, SparseLinearDRLearner

        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        T = np.random.binomial(2, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = SparseLinearDRLearner()
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([ 0.41...,  0.31..., -0.12...])
    >>> est.effect_interval(X[:3])
    (array([ 0.04..., -0.19..., -0.73...]), array([0.77..., 0.82..., 0.47...]))
    >>> est.coef_(T=1)
    array([ 0.45..., -0.00..., 0.06...])
    >>> est.coef__interval(T=1)
    (array([ 0.24... , -0.19..., -0.13...]), array([0.65..., 0.19..., 0.26...]))
    >>> est.intercept_(T=1)
    0.88...
    >>> est.intercept__interval(T=1)
    (0.68..., 1.08...)

    Attributes
    ----------
    score_ : float
        The MSE in the final doubly robust potential outcome regressions, i.e.

        .. math::
            \\frac{1}{n_t} \\sum_{t=1}^{n_t} \\frac{1}{n} \\sum_{i=1}^n (Y_{i, t}^{DR} - \\hat{\\theta}_t(X_i))^2

        where n_t is the number of treatments (excluding control).

        If `sample_weight` is not None at fit time, then a weighted average across samples is returned.

    """

    def __init__(self, *,
                 model_propensity='auto',
                 model_regression='auto',
                 featurizer=None,
                 fit_cate_intercept=True,
                 alpha='auto',
                 n_alphas=100,
                 alpha_cov='auto',
                 n_alphas_cov=10,
                 max_iter=1000,
                 tol=1e-4,
                 n_jobs=None,
                 min_propensity=1e-6,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.fit_cate_intercept = fit_cate_intercept
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.alpha_cov = alpha_cov
        self.n_alphas_cov = n_alphas_cov
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        super().__init__(model_propensity=model_propensity,
                         model_regression=model_regression,
                         model_final=None,
                         featurizer=featurizer,
                         multitask_model_final=False,
                         min_propensity=min_propensity,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return DebiasedLasso(alpha=self.alpha,
                             n_alphas=self.n_alphas,
                             alpha_cov=self.alpha_cov,
                             n_alphas_cov=self.n_alphas_cov,
                             fit_intercept=self.fit_cate_intercept,
                             max_iter=self.max_iter,
                             tol=self.tol,
                             n_jobs=self.n_jobs,
                             random_state=self.random_state)

    def _gen_ortho_learner_model_final(self):
        return _ModelFinal(self._gen_model_final(), self._gen_featurizer(), False)

    @_deprecate_positional("X and W should be passed by keyword only. In a future release "
                           "we will disallow passing X and W by position.", ['X', 'W'])
    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports ``'bootstrap'``
            (or an instance of :class:`.BootstrapInference`) and ``'debiasedlasso'``
            (or an instance of :class:`.LinearModelInferenceDiscrete`).

        Returns
        -------
        self: DRLearner instance
        """
        # Replacing fit from DRLearner, to add debiasedlasso inference in docstring
        # TODO: support sample_var
        if sample_weight is not None and inference is not None:
            warn("This estimator does not yet support sample variances and inference does not take "
                 "sample variances into account. This feature will be supported in a future release.")
        check_high_dimensional(X, T, threshold=5, featurizer=self.featurizer,
                               discrete_treatment=self.discrete_treatment,
                               msg="The number of features in the final model (< 5) is too small for a sparse model. "
                               "We recommend using the LinearDRLearner for this low-dimensional setting.")
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=None, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def fit_cate_intercept_(self):
        return self.model_final_.fit_intercept

    @property
    def multitask_model_final(self):
        return False

    @multitask_model_final.setter
    def multitask_model_final(self, value):
        if value:
            raise ValueError("Parameter `multitask_model_final` cannot change from `False` for this estimator!")

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class ForestDRLearner(ForestModelFinalCateEstimatorDiscreteMixin, DRLearner):
    """ Instance of DRLearner with a :class:`~econml.grf.RegressionForest`
    as a final model, so as to enable non-parametric inference.

    Parameters
    ----------
    model_propensity : scikit-learn classifier
        Estimator for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.
        Must implement `fit` and `predict_proba` methods. The `fit` method must be able to accept X and T,
        where T is a shape (n, ) array.

    model_regression : scikit-learn regressor
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment. Must implement `fit` and
        `predict` methods. If different models per treatment arm are desired, see the
        :class:`~econml.utilities.MultiModelWrapper` helper class.

    min_propensity : float, optional, default ``1e-6``
        The minimum propensity at which to clip propensity estimates to avoid dividing by zero.

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

    n_crossfit_splits: int or 'raise', optional (default='raise')
        Deprecated by parameter `cv` and will be removed in next version. Can be used
        interchangeably with `cv`.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    n_estimators : integer, optional (default=100)
        The total number of trees in the forest. The forest consists of a
        forest of sqrt(n_estimators) sub-forests, where each sub-forest
        contains sqrt(n_estimators) trees.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of splitting samples required to split an internal node.

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` splitting samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression. After construction the tree is also pruned
        so that there are at least min_samples_leaf estimation samples on
        each leaf.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        splitting samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided. After construction
        the tree is pruned so that the fraction of the sum total weight
        of the estimation samples contained in each leaf node is at
        least min_weight_fraction_leaf

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of split samples, ``N_t`` is the number of
        split samples at the current node, ``N_t_L`` is the number of split samples in the
        left child, and ``N_t_R`` is the number of split samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    subsample_fr : float or 'auto', optional (default='auto')
        The fraction of the half-samples that are used on each tree. Each tree
        will be built on subsample_fr * n_samples/2.

        If 'auto', then the subsampling fraction is set to::

            (n_samples/2)**(1-1/(2*n_features+2))/(n_samples/2)

        which is sufficient to guarantee asympotitcally valid inference.

    honest : boolean, optional (default=True)
        Whether to use honest trees, i.e. half of the samples are used for
        creating the tree structure and the other half for the estimation at
        the leafs. If False, then all samples are used for both parts.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, *,
                 model_regression="auto",
                 model_propensity="auto",
                 featurizer=None,
                 min_propensity=1e-6,
                 categories='auto',
                 cv=2,
                 n_crossfit_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 n_estimators=1000,
                 criterion='deprecated',
                 max_depth=None,
                 min_samples_split=5,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes='deprecated',
                 min_impurity_decrease=0.,
                 subsample_fr='deprecated',
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 subforest_size=4,
                 n_jobs=-1,
                 verbose=0,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.subforest_size = subforest_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.n_crossfit_splits = n_crossfit_splits
        if self.n_crossfit_splits != 'raise':
            cv = self.n_crossfit_splits
        self.subsample_fr = subsample_fr
        self.max_leaf_nodes = max_leaf_nodes
        self.criterion = criterion
        super().__init__(model_regression=model_regression,
                         model_propensity=model_propensity,
                         model_final=None,
                         featurizer=featurizer,
                         multitask_model_final=False,
                         min_propensity=min_propensity,
                         categories=categories,
                         cv=cv,
                         n_splits='raise',
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return RegressionForest(n_estimators=self.n_estimators,
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                max_features=self.max_features,
                                min_impurity_decrease=self.min_impurity_decrease,
                                max_samples=self.max_samples,
                                min_balancedness_tol=self.min_balancedness_tol,
                                honest=self.honest,
                                inference=True,
                                subforest_size=self.subforest_size,
                                n_jobs=self.n_jobs,
                                random_state=self.random_state,
                                verbose=self.verbose,
                                warm_start=False)

    def _gen_ortho_learner_model_final(self):
        return _ModelFinal(self._gen_model_final(), self._gen_featurizer(), False)

    @_deprecate_positional("X and W should be passed by keyword only. In a future release "
                           "we will disallow passing X and W by position.", ['X', 'W'])
    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

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
        sample_var: optional (n, n_y) vector
            Variance of sample, in case it corresponds to summary of many samples. Currently
            not in use by this method (as inference method does not require sample variance info).
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'blb'
            (for Bootstrap-of-Little-Bags based inference)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=None, groups=groups,
                           cache_values=cache_values, inference=inference)

    def multitask_model_cate(self):
        # Replacing to remove docstring
        super().multitask_model_cate()

    @property
    def multitask_model_final(self):
        return False

    @multitask_model_final.setter
    def multitask_model_final(self, value):
        if value:
            raise ValueError("Parameter `multitask_model_final` cannot change from `False` for this estimator!")

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")

    ####################################################################
    # Everything below should be removed once parameters are deprecated
    ####################################################################

    @property
    def n_crossfit_splits(self):
        return self.cv

    @n_crossfit_splits.setter
    def n_crossfit_splits(self, value):
        if value != 'raise':
            warn("Deprecated by parameter `n_splits` and will be removed in next version.")
        self.cv = value

    @property
    def criterion(self):
        return self.criterion

    @criterion.setter
    def criterion(self, value):
        if value != 'deprecated':
            warn("The parameter 'criterion' has been deprecated and will be removed in the next version. "
                 "Only the 'mse' criterion is supported.")

    @property
    def max_leaf_nodes(self):
        return self.max_leaf_nodes

    @max_leaf_nodes.setter
    def max_leaf_nodes(self, value):
        if value != 'deprecated':
            warn("The parameter 'max_leaf_nodes' has been deprecated and will be removed in the next version.")

    @property
    def subsample_fr(self):
        return 2 * self.max_samples

    @subsample_fr.setter
    def subsample_fr(self, value):
        if value != 'deprecated':
            warn("The parameter 'subsample_fr' has been deprecated and will be removed in the next version. "
                 "Use 'max_samples' instead, with the convention that "
                 "'subsample_fr=x' is equivalent to 'max_samples=x/2'.")
            self.max_samples = .45 if value == 'auto' else value / 2
