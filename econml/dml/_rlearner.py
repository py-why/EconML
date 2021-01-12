# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""

The R Learner is an approach for estimating flexible non-parametric models
of conditional average treatment effects in the setting with no unobserved confounders.
The method is based on the idea of Neyman orthogonality and estimates a CATE
whose mean squared error is robust to the estimation errors of auxiliary submodels
that also need to be estimated from data:

    1) the outcome or regression model
    2) the treatment or propensity or policy or logging policy model

References
----------

Xinkun Nie, Stefan Wager (2017). Quasi-Oracle Estimation of Heterogeneous Treatment Effects.
    https://arxiv.org/abs/1712.04912

Dylan Foster, Vasilis Syrgkanis (2019). Orthogonal Statistical Learning.
    ACM Conference on Learning Theory. https://arxiv.org/abs/1901.09036

Chernozhukov et al. (2017). Double/debiased machine learning for treatment and structural parameters.
    The Econometrics Journal. https://arxiv.org/abs/1608.00060
"""

from abc import abstractmethod
import numpy as np
import copy
from warnings import warn
from ..utilities import (shape, reshape, ndim, hstack, filter_none_kwargs, _deprecate_positional)
from sklearn.linear_model import LinearRegression
from sklearn.base import clone
from .._ortho_learner import _OrthoLearner


class _ModelNuisance:
    """
    Nuisance model fits the model_y and model_t at fit time and at predict time
    calculates the residual Y and residual T based on the fitted models and returns
    the residuals as two nuisance parameters.
    """

    def __init__(self, model_y, model_t):
        self._model_y = model_y
        self._model_t = model_t

    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        assert Z is None, "Cannot accept instrument!"
        self._model_t.fit(X, W, T, **filter_none_kwargs(sample_weight=sample_weight, groups=groups))
        self._model_y.fit(X, W, Y, **filter_none_kwargs(sample_weight=sample_weight, groups=groups))
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        if hasattr(self._model_y, 'score'):
            # note that groups are not passed to score because they are only used for fitting
            Y_score = self._model_y.score(X, W, Y, **filter_none_kwargs(sample_weight=sample_weight))
        else:
            Y_score = None
        if hasattr(self._model_t, 'score'):
            # note that groups are not passed to score because they are only used for fitting
            T_score = self._model_t.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight))
        else:
            T_score = None
        return Y_score, T_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        Y_pred = self._model_y.predict(X, W)
        T_pred = self._model_t.predict(X, W)
        if (X is None) and (W is None):  # In this case predict above returns a single row
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            T_pred = np.tile(T_pred.reshape(1, -1), (T.shape[0], 1))
        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = T - T_pred.reshape(T.shape)
        return Y_res, T_res


class _ModelFinal:
    """
    Final model at fit time, fits a residual on residual regression with a heterogeneous coefficient
    that depends on X, i.e.

        .. math ::
            Y - E[Y | X, W] = \\theta(X) \\cdot (T - E[T | X, W]) + \\epsilon

    and at predict time returns :math:`\\theta(X)`. The score method returns the MSE of this final
    residual on residual regression.
    """

    def __init__(self, model_final):
        self._model_final = model_final

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        Y_res, T_res = nuisances
        self._model_final.fit(X, T_res, Y_res, sample_weight=sample_weight, sample_var=sample_var)
        return self

    def predict(self, X=None):
        return self._model_final.predict(X)

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        Y_res, T_res = nuisances
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        effects = self._model_final.predict(X).reshape((-1, Y_res.shape[1], T_res.shape[1]))
        Y_res_pred = np.einsum('ijk,ik->ij', effects, T_res).reshape(Y_res.shape)
        if sample_weight is not None:
            return np.mean(np.average((Y_res - Y_res_pred)**2, weights=sample_weight, axis=0))
        else:
            return np.mean((Y_res - Y_res_pred) ** 2)


class _RLearner(_OrthoLearner):
    """
    Base class for CATE learners that residualize treatment and outcome and run residual on residual regression.
    The estimator is a special of an :class:`._OrthoLearner` estimator,
    so it follows the two
    stage process, where a set of nuisance functions are estimated in the first stage in a crossfitting
    manner and a final stage estimates the CATE model. See the documentation of
    :class:`._OrthoLearner` for a description of this two stage process.

    In this estimator, the CATE is estimated by using the following estimating equations:

    .. math ::
        Y - \\E[Y | X, W] = \\Theta(X) \\cdot (T - \\E[T | X, W]) + \\epsilon

    Thus if we estimate the nuisance functions :math:`q(X, W) = \\E[Y | X, W]` and
    :math:`f(X, W)=\\E[T | X, W]` in the first stage, we can estimate the final stage cate for each
    treatment t, by running a regression, minimizing the residual on residual square loss:

    .. math ::
        \\hat{\\theta} = \\arg\\min_{\\Theta}\
        \\E_n\\left[ (\\tilde{Y} - \\Theta(X) \\cdot \\tilde{T})^2 \\right]

    Where :math:`\\tilde{Y}=Y - \\E[Y | X, W]` and :math:`\\tilde{T}=T-\\E[T | X, W]` denotes the
    residual outcome and residual treatment.

    Parameters
    ----------
    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable
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

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    Examples
    --------

    The example code below implements a very simple version of the double machine learning
    method on top of the :class:`._RLearner` class, for expository purposes.
    For a more elaborate implementation of a Double Machine Learning child class of the class
    checkout :class:`.DML` and its child classes:

    .. testcode::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from econml.dml._rlearner import _RLearner
        from sklearn.base import clone
        class ModelFirst:
            def __init__(self, model):
                self._model = clone(model, safe=False)
            def fit(self, X, W, Y, sample_weight=None):
                self._model.fit(np.hstack([X, W]), Y)
                return self
            def predict(self, X, W):
                return self._model.predict(np.hstack([X, W]))
        class ModelFinal:
            def fit(self, X, T_res, Y_res, sample_weight=None, sample_var=None):
                self.model = LinearRegression(fit_intercept=False).fit(X * T_res.reshape(-1, 1),
                                                                       Y_res)
                return self
            def predict(self, X):
                return self.model.predict(X)
        class RLearner(_RLearner):
            def _gen_model_y(self):
                return ModelFirst(LinearRegression())
            def _gen_model_t(self):
                return ModelFirst(LinearRegression())
            def _gen_rlearner_model_final(self):
                return ModelFinal()
        np.random.seed(123)
        X = np.random.normal(size=(1000, 3))
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.01, size=(1000,))
        est = RLearner(cv=2, discrete_treatment=False, categories='auto', random_state=None)
        est.fit(y, X[:, 0], X=np.ones((X.shape[0], 1)), W=X[:, 1:])

    >>> est.const_marginal_effect(np.ones((1,1)))
    array([0.999631...])
    >>> est.effect(np.ones((1,1)), T0=0, T1=10)
    array([9.996314...])
    >>> est.score(y, X[:, 0], X=np.ones((X.shape[0], 1)), W=X[:, 1:])
    9.73638006...e-05
    >>> est.rlearner_model_final_.model
    LinearRegression(fit_intercept=False)
    >>> est.rlearner_model_final_.model.coef_
    array([0.999631...])
    >>> est.score_
    9.82623204...e-05
    >>> [mdl._model for mdl in est.models_y]
    [LinearRegression(), LinearRegression()]
    >>> [mdl._model for mdl in est.models_t]
    [LinearRegression(), LinearRegression()]

    Attributes
    ----------
    models_y: list of objects of type(model_y)
        A list of instances of the model_y object. Each element corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    models_t: list of objects of type(model_t)
        A list of instances of the model_t object. Each element corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    rlearner_model_final_ : object of type(model_final)
        An instance of the model_final object that was fitted after calling fit.
    score_ : float
        The MSE in the final residual on residual regression
    nuisance_scores_y : list of float
        The out-of-sample scores for each outcome model
    nuisance_scores_t : list of float
        The out-of-sample scores for each treatment model

        .. math::
            \\frac{1}{n} \\sum_{i=1}^n (Y_i - \\hat{E}[Y|X_i, W_i]\
                                        - \\hat{\\theta}(X_i)\\cdot (T_i - \\hat{E}[T|X_i, W_i]))^2

        If `sample_weight` is not None at fit time, then a weighted average is returned. If the outcome Y
        is multidimensional, then the average of the MSEs for each dimension of Y is returned.
    """

    def __init__(self, *, discrete_treatment, categories, cv, random_state,
                 n_splits='raise', mc_iters=None, mc_agg='mean'):
        super().__init__(discrete_treatment=discrete_treatment,
                         discrete_instrument=False,  # no instrument, so doesn't matter
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         random_state=random_state,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg)

    @abstractmethod
    def _gen_model_y(self):
        """
        Returns
        -------
        model_y: estimator of E[Y | X, W]
            The estimator for fitting the response to the features and controls. Must implement
            `fit` and `predict` methods.  Unlike sklearn estimators both methods must
            take an extra second argument (the controls), i.e. ::

                model_y.fit(X, W, Y, sample_weight=sample_weight)
                model_y.predict(X, W)
        """
        pass

    @abstractmethod
    def _gen_model_t(self):
        """
        Returns
        -------
        model_t: estimator of E[T | X, W]
            The estimator for fitting the treatment to the features and controls. Must implement
            `fit` and `predict` methods.  Unlike sklearn estimators both methods must
            take an extra second argument (the controls), i.e. ::

                model_t.fit(X, W, T, sample_weight=sample_weight)
                model_t.predict(X, W)
        """
        pass

    @abstractmethod
    def _gen_rlearner_model_final(self):
        """
        Returns
        -------
        model_final: estimator for fitting the response residuals to the features and treatment residuals
            Must implement `fit` and `predict` methods. Unlike sklearn estimators the fit methods must
            take an extra second argument (the treatment residuals). Predict, on the other hand,
            should just take the features and return the constant marginal effect. More, concretely::

                model_final.fit(X, T_res, Y_res,
                                sample_weight=sample_weight, sample_var=sample_var)
                model_final.predict(X)
        """
        pass

    def _gen_ortho_learner_model_nuisance(self):
        return _ModelNuisance(self._gen_model_y(), self._gen_model_t())

    def _gen_ortho_learner_model_final(self):
        return _ModelFinal(self._gen_rlearner_model_final())

    @_deprecate_positional("X, and should be passed by keyword only. In a future release "
                           "we will disallow passing X and W by position.", ['X', 'W'])
    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
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
        inference: string,:class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of:class:`.BootstrapInference`).

        Returns
        -------
        self: _RLearner instance
        """
        # Replacing fit from _OrthoLearner, to enforce Z=None and improve the docstring
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    def score(self, Y, T, X=None, W=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted residual nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calculates the MSE of the final residual Y on residual T regression.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
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
    def rlearner_model_final_(self):
        # NOTE: important to get parent's wrapped copy so that
        #       after training wrapped featurizer is also trained, etc.
        return self.ortho_learner_model_final_._model_final

    @property
    def models_y(self):
        return [mdl._model_y for mdl in super().models_nuisance_]

    @property
    def models_t(self):
        return [mdl._model_t for mdl in super().models_nuisance_]

    @property
    def nuisance_scores_y(self):
        return self.nuisance_scores_[0]

    @property
    def nuisance_scores_t(self):
        return self.nuisance_scores_[1]

    @property
    def residuals_(self):
        """
        A tuple (y_res, T_res, X, W), of the residuals from the first stage estimation
        along with the associated X and W. Samples are not guaranteed to be in the same
        order as the input order.
        """
        if not hasattr(self, '_cached_values'):
            raise AttributeError("Estimator is not fitted yet!")
        if self._cached_values is None:
            raise AttributeError("`fit` was called with `cache_values=False`. "
                                 "Set to `True` to enable residual storage.")
        Y_res, T_res = self._cached_values.nuisances
        return Y_res, T_res, self._cached_values.X, self._cached_values.W
