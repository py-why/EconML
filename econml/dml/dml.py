# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from warnings import warn

import numpy as np
from sklearn.base import TransformerMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import (ElasticNetCV, LassoCV, LogisticRegressionCV)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder,
                                   OneHotEncoder)
from sklearn.utils import check_random_state
import copy

from .._ortho_learner import _OrthoLearner
from ._rlearner import _RLearner
from .._cate_estimator import (DebiasedLassoCateEstimatorMixin,
                               ForestModelFinalCateEstimatorMixin,
                               LinearModelFinalCateEstimatorMixin,
                               StatsModelsCateEstimatorMixin,
                               LinearCateEstimator)
from ..inference import StatsModelsInference, GenericSingleTreatmentModelFinalInference
from ..sklearn_extensions.linear_model import (MultiOutputDebiasedLasso,
                                               StatsModelsLinearRegression,
                                               WeightedLassoCVWrapper)
from ..sklearn_extensions.model_selection import WeightedStratifiedKFold
from ..utilities import (_deprecate_positional, add_intercept,
                         broadcast_unit_treatments, check_high_dimensional,
                         cross_product, deprecated, fit_with_groups,
                         hstack, inverse_onehot, ndim, reshape,
                         reshape_treatmentwise_effects, shape, transpose)
from .._shap import _shap_explain_model_cate


class _FirstStageWrapper:
    def __init__(self, model, is_Y, featurizer, linear_first_stages, discrete_treatment):
        self._model = clone(model, safe=False)
        self._featurizer = clone(featurizer, safe=False)
        self._is_Y = is_Y
        self._linear_first_stages = linear_first_stages
        self._discrete_treatment = discrete_treatment

    def _combine(self, X, W, n_samples, fitting=True):
        if X is None:
            # if both X and W are None, just return a column of ones
            return (W if W is not None else np.ones((n_samples, 1)))
        XW = hstack([X, W]) if W is not None else X
        if self._is_Y and self._linear_first_stages:
            if self._featurizer is None:
                F = X
            else:
                F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
            return cross_product(XW, hstack([np.ones((shape(XW)[0], 1)), F]))
        else:
            return XW

    def fit(self, X, W, Target, sample_weight=None, groups=None):
        if (not self._is_Y) and self._discrete_treatment:
            # In this case, the Target is the one-hot-encoding of the treatment variable
            # We need to go back to the label representation of the one-hot so as to call
            # the classifier.
            if np.any(np.all(Target == 0, axis=0)) or (not np.any(np.all(Target == 0, axis=1))):
                raise AttributeError("Provided crossfit folds contain training splits that " +
                                     "don't contain all treatments")
            Target = inverse_onehot(Target)

        if sample_weight is not None:
            fit_with_groups(self._model, self._combine(X, W, Target.shape[0]), Target, groups=groups,
                            sample_weight=sample_weight)
        else:
            fit_with_groups(self._model, self._combine(X, W, Target.shape[0]), Target, groups=groups)
        return self

    def predict(self, X, W):
        n_samples = X.shape[0] if X is not None else (W.shape[0] if W is not None else 1)
        if (not self._is_Y) and self._discrete_treatment:
            return self._model.predict_proba(self._combine(X, W, n_samples, fitting=False))[:, 1:]
        else:
            return self._model.predict(self._combine(X, W, n_samples, fitting=False))

    def score(self, X, W, Target, sample_weight=None):
        if hasattr(self._model, 'score'):
            if (not self._is_Y) and self._discrete_treatment:
                # In this case, the Target is the one-hot-encoding of the treatment variable
                # We need to go back to the label representation of the one-hot so as to call
                # the classifier.
                Target = inverse_onehot(Target)
            if sample_weight is not None:
                return self._model.score(self._combine(X, W, Target.shape[0]), Target, sample_weight=sample_weight)
            else:
                return self._model.score(self._combine(X, W, Target.shape[0]), Target)
        else:
            return None


class _FinalWrapper:
    def __init__(self, model_final, fit_cate_intercept, featurizer, use_weight_trick):
        self._model = clone(model_final, safe=False)
        self._use_weight_trick = use_weight_trick
        self._original_featurizer = clone(featurizer, safe=False)
        if self._use_weight_trick:
            self._fit_cate_intercept = False
            self._featurizer = self._original_featurizer
        else:
            self._fit_cate_intercept = fit_cate_intercept
            if self._fit_cate_intercept:
                add_intercept_trans = FunctionTransformer(add_intercept,
                                                          validate=True)
                if featurizer:
                    self._featurizer = Pipeline([('featurize', self._original_featurizer),
                                                 ('add_intercept', add_intercept_trans)])
                else:
                    self._featurizer = add_intercept_trans
            else:
                self._featurizer = self._original_featurizer

    def _combine(self, X, T, fitting=True):
        if X is not None:
            if self._featurizer is not None:
                F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
            else:
                F = X
        else:
            if not self._fit_cate_intercept:
                if self._use_weight_trick:
                    raise AttributeError("Cannot use this method with X=None. Consider "
                                         "using the LinearDML estimator.")
                else:
                    raise AttributeError("Cannot have X=None and also not allow for a CATE intercept!")
            F = np.ones((T.shape[0], 1))
        return cross_product(F, T)

    def fit(self, X, T_res, Y_res, sample_weight=None, sample_var=None):
        # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T_res)[1:]
        self._d_y = shape(Y_res)[1:]
        if not self._use_weight_trick:
            fts = self._combine(X, T_res)
            if sample_weight is not None:
                if sample_var is not None:
                    self._model.fit(fts,
                                    Y_res, sample_weight=sample_weight, sample_var=sample_var)
                else:
                    self._model.fit(fts,
                                    Y_res, sample_weight=sample_weight)
            else:
                self._model.fit(fts, Y_res)

            self._intercept = None
            intercept = self._model.predict(np.zeros_like(fts[0:1]))
            if (np.count_nonzero(intercept) > 0):
                warn("The final model has a nonzero intercept for at least one outcome; "
                     "it will be subtracted, but consider fitting a model without an intercept if possible.",
                     UserWarning)
                self._intercept = intercept
        elif not self._fit_cate_intercept:
            if (np.ndim(T_res) > 1) and (self._d_t[0] > 1):
                raise AttributeError("This method can only be used with single-dimensional continuous treatment "
                                     "or binary categorical treatment.")
            F = self._combine(X, np.ones(T_res.shape[0]))
            self._intercept = None
            T_res = T_res.ravel()
            sign_T_res = np.sign(T_res)
            sign_T_res[(sign_T_res < 1) & (sign_T_res > -1)] = 1
            clipped_T_res = sign_T_res * np.clip(np.abs(T_res), 1e-5, np.inf)
            if np.ndim(Y_res) > 1:
                clipped_T_res = clipped_T_res.reshape(-1, 1)
            target = Y_res / clipped_T_res
            target_var = sample_var / clipped_T_res**2 if sample_var is not None else None

            if sample_weight is not None:
                if target_var is not None:
                    self._model.fit(F, target, sample_weight=sample_weight * T_res.flatten()**2,
                                    sample_var=target_var)
                else:
                    self._model.fit(F, target, sample_weight=sample_weight * T_res.flatten()**2)
            else:
                self._model.fit(F, target, sample_weight=T_res.flatten()**2)
        else:
            raise AttributeError("This combination is not a feasible one!")
        return self

    def predict(self, X):
        X2, T = broadcast_unit_treatments(X if X is not None else np.empty((1, 0)),
                                          self._d_t[0] if self._d_t else 1)
        # This works both with our without the weighting trick as the treatments T are unit vector
        # treatments. And in the case of a weighting trick we also know that treatment is single-dimensional
        prediction = self._model.predict(self._combine(None if X is None else X2, T, fitting=False))
        if self._intercept is not None:
            prediction -= self._intercept
        return reshape_treatmentwise_effects(prediction,
                                             self._d_t, self._d_y)


class _BaseDML(_RLearner):
    # A helper class that access all the internal fitted objects of a DML Cate Estimator. Used by
    # both Parametric and Non Parametric DML.

    @property
    def original_featurizer(self):
        # NOTE: important to use the rlearner_model_final_ attribute instead of the
        #       attribute so that the trained featurizer will be passed through
        return self.rlearner_model_final_._original_featurizer

    @property
    def featurizer_(self):
        # NOTE This is used by the inference methods and has to be the overall featurizer. intended
        # for internal use by the library
        return self.rlearner_model_final_._featurizer

    @property
    def model_final_(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        #      We need to use the rlearner's copy to retain the information from fitting
        return self.rlearner_model_final_._model

    @property
    def model_cate(self):
        """
        Get the fitted final CATE model.

        Returns
        -------
        model_cate: object of type(model_final)
            An instance of the model_final object that was fitted after calling fit which corresponds
            to the constant marginal CATE model.
        """
        return self.rlearner_model_final_._model

    @property
    def models_y(self):
        """
        Get the fitted models for E[Y | X, W].

        Returns
        -------
        models_y: list of objects of type(`model_y`)
            A list of instances of the `model_y` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model for mdl in super().models_y]

    @property
    def models_t(self):
        """
        Get the fitted models for E[T | X, W].

        Returns
        -------
        models_y: list of objects of type(`model_t`)
            A list of instances of the `model_y` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model for mdl in super().models_t]

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
            final constant marginal CATE model is linear. It is the names of the features that are associated
            with each entry of the :meth:`coef_` parameter. Not available when the featurizer is not None and
            does not have a method: `get_feature_names(feature_names)`. Otherwise None is returned.
        """
        if self._d_x is None:
            # Handles the corner case when X=None but featurizer might be not None
            return None
        if feature_names is None:
            feature_names = self._input_names["feature_names"]
        if self.original_featurizer is None:
            return feature_names
        elif hasattr(self.original_featurizer, 'get_feature_names'):
            # This fails if X=None and featurizer is not None, but that case is handled above
            return self.original_featurizer.get_feature_names(feature_names)
        else:
            raise AttributeError("Featurizer does not have a method: get_feature_names!")


class DML(LinearModelFinalCateEstimatorMixin, _BaseDML):
    """
    The base class for parametric Double ML estimators. The estimator is a special
    case of an :class:`._RLearner` estimator, which in turn is a special case
    of an :class:`_OrthoLearner` estimator, so it follows the two
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

    The DML estimator further assumes a linear parametric form for the cate, i.e. for each outcome
    :math:`i` and treatment :math:`j`:

    .. math ::
        \\Theta_{i, j}(X) =  \\phi(X)' \\cdot \\Theta_{ij}

    For some given feature mapping :math:`\\phi(X)` (the user can provide this featurizer via the `featurizer`
    parameter at init time and could be any arbitrary class that adheres to the scikit-learn transformer
    interface :class:`~sklearn.base.TransformerMixin`).

    The second nuisance function :math:`q` is a simple regression problem and the
    :class:`.DML`
    class takes as input the parameter `model_y`, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem.

    The problem of estimating the nuisance function :math:`f` is also a regression problem and
    the :class:`.DML`
    class takes as input the parameter `model_t`, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem. If the init flag `discrete_treatment` is set
    to `True`, then the parameter `model_t` is treated as a scikit-learn classifier. The input categorical
    treatment is one-hot encoded (excluding the lexicographically smallest treatment which is used as the
    baseline) and the `predict_proba` method of the `model_t` classifier is used to
    residualize the one-hot encoded treatment.

    The final stage is (potentially multi-task) linear regression problem with outcomes the labels
    :math:`\\tilde{Y}` and regressors the composite features
    :math:`\\tilde{T}\\otimes \\phi(X) = \\mathtt{vec}(\\tilde{T}\\cdot \\phi(X)^T)`.
    The :class:`.DML` takes as input parameter
    ``model_final``, which is any linear scikit-learn regressor that is internally used to solve this
    (multi-task) linear regresion problem.

    Parameters
    ----------
    model_y: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t: estimator or 'auto' (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods.  Must be a linear model for correctness
        when linear_first_stages is ``True``;
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/
        :class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

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
    """

    def __init__(self, *,
                 model_y, model_t, model_final,
                 featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies
        self.fit_cate_intercept = fit_cate_intercept
        self.linear_first_stages = linear_first_stages
        self.featurizer = clone(featurizer, safe=False)
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.model_final = clone(model_final, safe=False)
        super().__init__(discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        if self.model_y == 'auto':
            model_y = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_y = clone(self.model_y, safe=False)
        return _FirstStageWrapper(model_y, True, self._gen_featurizer(),
                                  self.linear_first_stages, self.discrete_treatment)

    def _gen_model_t(self):
        if self.model_t == 'auto':
            if self.discrete_treatment:
                model_t = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                               random_state=self.random_state)
            else:
                model_t = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_t = clone(self.model_t, safe=False)
        return _FirstStageWrapper(model_t, False, self._gen_featurizer(),
                                  self.linear_first_stages, self.discrete_treatment)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_rlearner_model_final(self):
        return _FinalWrapper(self._gen_model_final(), self.fit_cate_intercept, self._gen_featurizer(), False)

    # override only so that we can update the docstring to indicate support for `StatsModelsInference`
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
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.LinearModelFinalInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    @property
    def bias_part_of_coef(self):
        return self.rlearner_model_final_._fit_cate_intercept

    @property
    def fit_cate_intercept_(self):
        return self.rlearner_model_final_._fit_cate_intercept


class LinearDML(StatsModelsCateEstimatorMixin, DML):
    """
    The Double ML Estimator with a low-dimensional linear final stage implemented as a statsmodel regression.

    Parameters
    ----------
    model_y: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods;
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV` will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

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
    """

    def __init__(self, *,
                 model_y='auto', model_t='auto',
                 featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages=True,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=None,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         linear_first_stages=linear_first_stages,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,)

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=False)

    # override only so that we can update the docstring to indicate support for `StatsModelsInference`
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
        sample_var: (n,) vector, optional
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'statsmodels'
            (or an instance of :class:`.StatsModelsInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class SparseLinearDML(DebiasedLassoCateEstimatorMixin, DML):
    """
    A specialized version of the Double ML estimator for the sparse linear case.

    This estimator should be used when the features of heterogeneity are high-dimensional
    and the coefficients of the linear CATE function are sparse.

    The last stage is an instance of the
    :class:`.MultiOutputDebiasedLasso`

    Parameters
    ----------
    model_y: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods, and must be a
        linear model for correctness;
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/
        :class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    alpha: string | float, optional. Default='auto'.
        CATE L1 regularization applied through the debiased lasso in the final model.
        'auto' corresponds to a CV form of the :class:`MultiOutputDebiasedLasso`.

    n_alphas : int, optional, default 100
        How many alphas to try if alpha='auto'

    alpha_cov : string | float, optional, default 'auto'
        The regularization alpha that is used when constructing the pseudo inverse of
        the covariance matrix Theta used to for correcting the final state lasso coefficient
        in the debiased lasso. Each such regression corresponds to the regression of one feature
        on the remainder of the features.

    n_alphas_cov : int, optional, default 10
        How many alpha_cov to try if alpha_cov='auto'.

    max_iter : int, optional, default=1000
        The maximum number of iterations in the Debiased Lasso

    tol : float, optional, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

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
    """

    def __init__(self, *,
                 model_y='auto', model_t='auto',
                 alpha='auto',
                 n_alphas=100,
                 alpha_cov='auto',
                 n_alphas_cov=10,
                 max_iter=1000,
                 tol=1e-4,
                 n_jobs=None,
                 featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages=True,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.alpha_cov = alpha_cov
        self.n_alphas_cov = n_alphas_cov
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=None,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         linear_first_stages=linear_first_stages,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return MultiOutputDebiasedLasso(alpha=self.alpha,
                                        n_alphas=self.n_alphas,
                                        alpha_cov=self.alpha_cov,
                                        n_alphas_cov=self.n_alphas_cov,
                                        fit_intercept=False,
                                        max_iter=self.max_iter,
                                        tol=self.tol,
                                        n_jobs=self.n_jobs,
                                        random_state=self.random_state)

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
            not in use by this method but will be supported in a future release.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'debiasedlasso'
            (or an instance of :class:`.LinearModelFinalInference`)

        Returns
        -------
        self
        """
        # TODO: support sample_var
        if sample_var is not None and inference is not None:
            warn("This estimator does not yet support sample variances and inference does not take "
                 "sample variances into account. This feature will be supported in a future release.")
        check_high_dimensional(X, T, threshold=5, featurizer=self.featurizer,
                               discrete_treatment=self.discrete_treatment,
                               msg="The number of features in the final model (< 5) is too small for a sparse model. "
                               "We recommend using the LinearDML estimator for this low-dimensional setting.")
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=None, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class _RandomFeatures(TransformerMixin):
    def __init__(self, *, dim, bw, random_state):
        self.dim = dim
        self.bw = bw
        self.random_state = random_state

    def fit(self, X):
        random_state = check_random_state(self.random_state)
        self.omegas_ = random_state.normal(0, 1 / self.bw, size=(shape(X)[1], self.dim))
        self.biases_ = random_state.uniform(0, 2 * np.pi, size=(1, self.dim))
        self.dim_ = self.dim
        return self

    def transform(self, X):
        return np.sqrt(2 / self.dim_) * np.cos(np.matmul(X, self.omegas_) + self.biases_)


class KernelDML(DML):
    """
    A specialized version of the linear Double ML Estimator that uses random fourier features.

    Parameters
    ----------
    model_y: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods;
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/
        :class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    dim: int, optional (default is 20)
        The number of random Fourier features to generate

    bw: float, optional (default is 1.0)
        The bandwidth of the Gaussian used to generate features

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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

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
    """

    def __init__(self, model_y='auto', model_t='auto',
                 discrete_treatment=False, categories='auto',
                 fit_cate_intercept=True,
                 dim=20,
                 bw=1.0,
                 cv=2,
                 n_splits='raise',
                 mc_iters=None, mc_agg='mean',
                 random_state=None):
        self.dim = dim
        self.bw = bw
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=None,
                         featurizer=None,
                         fit_cate_intercept=fit_cate_intercept,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return ElasticNetCV(fit_intercept=False, random_state=self.random_state)

    def _gen_featurizer(self):
        return _RandomFeatures(dim=self.dim, bw=self.bw, random_state=self.random_state)

    @property
    def featurizer(self):
        return self._gen_featurizer()

    @featurizer.setter
    def featurizer(self, value):
        if value is not None:
            raise ValueError("Parameter `featurizer` cannot be altered for this estimator!")

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class NonParamDML(_BaseDML):
    """
    The base class for non-parametric Double ML estimators, that can have arbitrary final ML models of the CATE.
    Works only for single-dimensional continuous treatment or for binary categorical treatment and uses
    the re-weighting trick, reducing the final CATE estimation to a weighted square loss minimization.
    The model_final parameter must support the sample_weight keyword argument at fit time.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods. It can be an arbitrary scikit-learn regressor. The `fit` method
        must accept `sample_weight` as a keyword argument.

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

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
    """

    def __init__(self, *,
                 model_y, model_t, model_final,
                 featurizer=None,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):

        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.model_final = clone(model_final, safe=False)
        super().__init__(discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _get_inference_options(self):
        # add blb to parent's options
        options = super()._get_inference_options()
        options.update(auto=GenericSingleTreatmentModelFinalInference)
        return options

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        return _FirstStageWrapper(clone(self.model_y, safe=False), True,
                                  self._gen_featurizer(), False, self.discrete_treatment)

    def _gen_model_t(self):
        return _FirstStageWrapper(clone(self.model_t, safe=False), False,
                                  self._gen_featurizer(), False, self.discrete_treatment)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_rlearner_model_final(self):
        return _FinalWrapper(self._gen_model_final(), False, self._gen_featurizer(), True)

    # override only so that we can update the docstring to indicate
    # support for `GenericSingleTreatmentModelFinalInference`
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
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.GenericSingleTreatmentModelFinalInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        if self.featurizer_ is not None:
            F = self.featurizer_.transform(X)
        else:
            F = X
        feature_names = self.cate_feature_names(feature_names)

        return _shap_explain_model_cate(self.const_marginal_effect, self.model_cate, F, self._d_t, self._d_y,
                                        feature_names=feature_names,
                                        treatment_names=treatment_names,
                                        output_names=output_names,
                                        input_names=self._input_names,
                                        background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__


@deprecated("The ForestDML class has been deprecated by the CausalForestDML with parameter "
            "`criterion='mse'`; an upcoming release will remove support for the old class")
def ForestDML(model_y, model_t,
              discrete_treatment=False,
              categories='auto',
              cv=2,
              n_crossfit_splits='raise',
              mc_iters=None,
              mc_agg='mean',
              n_estimators=100,
              criterion="mse",
              max_depth=None,
              min_samples_split=2,
              min_samples_leaf=1,
              min_weight_fraction_leaf=0.,
              max_features="auto",
              max_leaf_nodes=None,
              min_impurity_decrease=0.,
              subsample_fr='auto',
              honest=True,
              n_jobs=None,
              verbose=0,
              random_state=None):
    """ Instance of NonParamDML with a
    :class:``~econml.grf.RegressionForest`
    as a final model, so as to enable non-parametric inference.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

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
    from . import CausalForestDML
    return CausalForestDML(model_y=model_y,
                           model_t=model_t,
                           discrete_treatment=discrete_treatment,
                           categories=categories,
                           cv=cv,
                           n_crossfit_splits=n_crossfit_splits,
                           mc_iters=mc_iters,
                           mc_agg=mc_agg,
                           n_estimators=n_estimators,
                           criterion="mse",
                           max_depth=max_depth,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf,
                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                           max_features=max_features,
                           min_impurity_decrease=min_impurity_decrease,
                           max_samples=.45 if subsample_fr == 'auto' else subsample_fr / 2,
                           honest=honest,
                           n_jobs=n_jobs,
                           verbose=verbose,
                           random_state=random_state)
