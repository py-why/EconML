# Copyright (c) PyWhy contributors. All rights reserved.
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
                         cross_product, deprecated,
                         hstack, inverse_onehot, ndim, reshape,
                         reshape_treatmentwise_effects, shape, transpose,
                         get_feature_names_or_default, filter_none_kwargs)
from .._shap import _shap_explain_model_cate
from ..sklearn_extensions.model_selection import get_selector, ModelSelector, SingleModelSelector


def _combine(X, W, n_samples):
    if X is None:
        # if both X and W are None, just return a column of ones
        return (W if W is not None else np.ones((n_samples, 1)))
    return hstack([X, W]) if W is not None else X


class _FirstStageWrapper:
    def __init__(self, model, discrete_target):
        self._model = model  # plain sklearn-compatible model, not a ModelSelector
        self._discrete_target = discrete_target

    def predict(self, X, W):
        n_samples = X.shape[0] if X is not None else (W.shape[0] if W is not None else 1)
        if self._discrete_target:
            if hasattr(self._model, 'predict_proba'):
                return self._model.predict_proba(_combine(X, W, n_samples))[:, 1:]
            else:
                warn('First stage model has discrete target but model is not a classifier!', UserWarning)
                return self._model.predict(_combine(X, W, n_samples))
        else:
            if hasattr(self._model, 'predict_proba'):
                raise AttributeError("Cannot use a classifier as a first stage model when the target is continuous!")
            return self._model.predict(_combine(X, W, n_samples))

    def score(self, X, W, Target, sample_weight=None):
        if hasattr(self._model, 'score'):
            if self._discrete_target:
                # In this case, the Target is the one-hot-encoding of the treatment variable
                # We need to go back to the label representation of the one-hot so as to call
                # the classifier.
                Target = inverse_onehot(Target)
            if sample_weight is not None:
                return self._model.score(_combine(X, W, Target.shape[0]), Target, sample_weight=sample_weight)
            else:
                return self._model.score(_combine(X, W, Target.shape[0]), Target)
        else:
            return None


class _FirstStageSelector(SingleModelSelector):
    def __init__(self, model: SingleModelSelector, discrete_target):
        self._model = clone(model, safe=False)
        self._discrete_target = discrete_target

    def train(self, is_selecting, folds, X, W, Target, sample_weight=None, groups=None):
        if self._discrete_target:
            # In this case, the Target is the one-hot-encoding of the treatment variable
            # We need to go back to the label representation of the one-hot so as to call
            # the classifier.
            if np.any(np.all(Target == 0, axis=0)) or (not np.any(np.all(Target == 0, axis=1))):
                raise AttributeError("Provided crossfit folds contain training splits that " +
                                     "don't contain all treatments")
            Target = inverse_onehot(Target)

        self._model.train(is_selecting, folds, _combine(X, W, Target.shape[0]), Target,
                          **filter_none_kwargs(groups=groups, sample_weight=sample_weight))
        return self

    @property
    def best_model(self):
        return _FirstStageWrapper(self._model.best_model, self._discrete_target)

    @property
    def best_score(self):
        return self._model.best_score


def _make_first_stage_selector(model, is_discrete, random_state):
    if model == 'auto':
        model = ['forest', 'linear']
    return _FirstStageSelector(get_selector(model,
                                            is_discrete=is_discrete,
                                            random_state=random_state),
                               discrete_target=is_discrete)


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
                # data is already validated at initial fit time
                add_intercept_trans = FunctionTransformer(add_intercept,
                                                          validate=False)
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

    def fit(self, X, T, T_res, Y_res, sample_weight=None, freq_weight=None, sample_var=None, groups=None):
        # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T_res)[1:]
        self._d_y = shape(Y_res)[1:]
        if not self._use_weight_trick:
            fts = self._combine(X, T_res)
            filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight,
                                                 freq_weight=freq_weight, sample_var=sample_var)
            self._model.fit(fts, Y_res, **filtered_kwargs)
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
                sample_weight = sample_weight * T_res.flatten()**2
            else:
                sample_weight = T_res.flatten()**2
            filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight,
                                                 freq_weight=freq_weight, sample_var=target_var)
            self._model.fit(F, target, **filtered_kwargs)
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
        models_y: nested list of objects of type(`model_y`)
            A nested list of instances of the `model_y` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._model for mdl in mdls] for mdls in super().models_y]

    @property
    def models_t(self):
        """
        Get the fitted models for E[T | X, W].

        Returns
        -------
        models_t: nested list of objects of type(`model_t`)
            A nested list of instances of the `model_y` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._model for mdl in mdls] for mdls in super().models_t]

    def cate_feature_names(self, feature_names=None):
        """
        Get the output feature names.

        Parameters
        ----------
        feature_names: list of str of length X.shape[1] or None
            The names of the input features. If None and X is a dataframe, it defaults to the column names
            from the dataframe.

        Returns
        -------
        out_feature_names: list of str or None
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
        return get_feature_names_or_default(self.original_featurizer, feature_names)


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
    model_y: estimator, default ``'auto'``
        Determines how to fit the outcome to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t: estimator, default ``'auto'``
        Determines how to fit the treatment to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    fit_cate_intercept : bool, default True
        Whether the linear CATE model should have a constant term.

    discrete_outcome: bool, default ``False``
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default ``False``
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    verbose: int, default 2
        The verbosity level of the output messages. Higher values indicate more verbosity.

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

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in X, W. If True, will need to supply model_y, model_t, and model_final
        that can handle missing values.

    use_ray: bool, default False
        Whether to use Ray to parallelize the cross-validation step. If True, Ray must be installed.

    ray_remote_func_options : dict, default None
        Options to pass to the remote function when using Ray.
        See https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html

    Examples
    --------
    A simple example with discrete treatment and a linear model_final (equivalent to LinearDML):

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dml import DML
        from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = DML(
            model_y=RandomForestRegressor(),
            model_t=RandomForestClassifier(),
            model_final=StatsModelsLinearRegression(fit_intercept=False),
            discrete_treatment=True
        )
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([0.63382..., 1.78225..., 0.71859...])
    >>> est.effect_interval(X[:3])
    (array([0.27937..., 1.27619..., 0.42091...]),
    array([0.98827... , 2.28831..., 1.01628...]))
    >>> est.coef_
    array([ 0.42857...,  0.04488..., -0.03317...,  0.02258..., -0.14875...])
    >>> est.coef__interval()
    (array([ 0.25179..., -0.10558..., -0.16723... , -0.11916..., -0.28759...]),
    array([ 0.60535...,  0.19536...,  0.10088...,  0.16434..., -0.00990...]))
    >>> est.intercept_
    1.01166...
    >>> est.intercept__interval()
    (0.87125..., 1.15207...)
    """

    def __init__(self, *,
                 model_y,
                 model_t,
                 model_final,
                 featurizer=None,
                 treatment_featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages="deprecated",
                 discrete_outcome=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False,
                 use_ray=False,
                 ray_remote_func_options=None
                 ):
        self.fit_cate_intercept = fit_cate_intercept
        if linear_first_stages != "deprecated":
            warn("The linear_first_stages parameter is deprecated and will be removed in a future version of EconML",
                 DeprecationWarning)
        self.featurizer = clone(featurizer, safe=False)
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.model_final = clone(model_final, safe=False)
        super().__init__(discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         treatment_featurizer=treatment_featurizer,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing,
                         use_ray=use_ray,
                         ray_remote_func_options=ray_remote_func_options)

    def _gen_allowed_missing_vars(self):
        return ['X', 'W'] if self.allow_missing else []

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        return _make_first_stage_selector(self.model_y, self.discrete_outcome, self.random_state)

    def _gen_model_t(self):
        return _make_first_stage_selector(self.model_t, self.discrete_treatment, self.random_state)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_rlearner_model_final(self):
        return _FinalWrapper(self._gen_model_final(), self.fit_cate_intercept, self._gen_featurizer(), False)

    # override only so that we can update the docstring to indicate support for `LinearModelFinalInference`
    def fit(self, Y, T, *, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

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
        sample_weight : (n,) array_like, optional
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array_like of int, optional
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(n,), (n, d_y)} nd array_like, optional
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.LinearModelFinalInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight,
                           sample_var=sample_var, groups=groups,
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
    model_y: estimator, default ``'auto'``
        Determines how to fit the outcome to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t: estimator, default ``'auto'``
        Determines how to fit the treatment to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    fit_cate_intercept : bool, default True
        Whether the linear CATE model should have a constant term.

    discrete_outcome: bool, default ``False``
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default ``False``
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply model_y, model_t that can handle
        missing values.

    enable_federation: bool, default False
        Whether to enable federation for the final model.  This has a memory cost so should be enabled only
        if this model will be aggregated with other models.

    use_ray: bool, default False
        Whether to use Ray to parallelize the cross-fitting step. If True, Ray must be installed.

    ray_remote_func_options : dict, default None
        Options to pass to the remote function when using Ray.
        See https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html

    Examples
    --------
    A simple example with the default models and discrete treatment:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dml import LinearDML

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = LinearDML(discrete_treatment=True)
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([0.49977..., 1.91668..., 0.70799...])
    >>> est.effect_interval(X[:3])
    (array([0.15122..., 1.40176..., 0.40954...]),
    array([0.84831..., 2.43159..., 1.00644...]))
    >>> est.coef_
    array([ 0.48825...,  0.00105...,  0.00244...,  0.02217..., -0.08471...])
    >>> est.coef__interval()
    (array([ 0.30469..., -0.13904..., -0.12790..., -0.11514..., -0.22505... ]),
    array([0.67180..., 0.14116..., 0.13278..., 0.15949..., 0.05562...]))
    >>> est.intercept_
    1.01247...
    >>> est.intercept__interval()
    (0.87480..., 1.15015...)
    """

    def __init__(self, *,
                 model_y='auto', model_t='auto',
                 featurizer=None,
                 treatment_featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages="deprecated",
                 discrete_outcome=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False,
                 enable_federation=False,
                 use_ray=False,
                 ray_remote_func_options=None
                 ):

        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=None,
                         featurizer=featurizer,
                         treatment_featurizer=treatment_featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         linear_first_stages=linear_first_stages,
                         discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing,
                         use_ray=use_ray,
                         ray_remote_func_options=ray_remote_func_options)
        self.enable_federation = enable_federation

    def _gen_allowed_missing_vars(self):
        return ['W'] if self.allow_missing else []

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=False, enable_federation=self.enable_federation)

    # override only so that we can update the docstring to indicate support for `StatsModelsInference`
    def fit(self, Y, T, *, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

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
        sample_weight : (n,) array_like, optional
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array_like of int, optional
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(n,), (n, d_y)} nd array_like, optional
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'statsmodels'
            (or an instance of :class:`.StatsModelsInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var, groups=groups,
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
    model_y: estimator, default ``'auto'``
        Determines how to fit the outcome to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t: estimator, default ``'auto'``
        Determines how to fit the treatment to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    alpha: str or float, default 'auto'
        CATE L1 regularization applied through the debiased lasso in the final model.
        'auto' corresponds to a CV form of the :class:`MultiOutputDebiasedLasso`.

    n_alphas : int, default 100
        How many alphas to try if alpha='auto'

    alpha_cov : str | float, default 'auto'
        The regularization alpha that is used when constructing the pseudo inverse of
        the covariance matrix Theta used to for correcting the final state lasso coefficient
        in the debiased lasso. Each such regression corresponds to the regression of one feature
        on the remainder of the features.

    n_alphas_cov : int, default 10
        How many alpha_cov to try if alpha_cov='auto'.

    max_iter : int, default 1000
        The maximum number of iterations in the Debiased Lasso

    tol : float, default 1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    n_jobs : int or None, optional
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

    featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    fit_cate_intercept : bool, default True
        Whether the linear CATE model should have a constant term.

    discrete_outcome: bool, default ``False``
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default ``False``
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply model_y, model_t that can handle
        missing values.

    use_ray: bool, default False
        Whether to use Ray to parallelize the cross-fitting step. If True, Ray must be installed.

    ray_remote_func_options : dict, default None
        Options to pass to the remote function when using Ray.
        See https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html

    Examples
    --------
    A simple example with the default models and discrete treatment:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dml import SparseLinearDML

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = SparseLinearDML(discrete_treatment=True)
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([0.50083..., 1.91663..., 0.70386...])
    >>> est.effect_interval(X[:3])
    (array([0.14616..., 1.40364..., 0.40674...]),
    array([0.85550...  , 2.42962... , 1.00099...]))
    >>> est.coef_
    array([ 0.49123...,  0.00495...,  0.00007...,  0.02302..., -0.08483...])
    >>> est.coef__interval()
    (array([ 0.31323..., -0.13848..., -0.13721..., -0.11141..., -0.22961...]),
    array([0.66923..., 0.14839... , 0.13735..., 0.15745..., 0.05993...]))
    >>> est.intercept_
    1.01476...
    >>> est.intercept__interval()
    (0.87620..., 1.15332...)
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
                 treatment_featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages="deprecated",
                 discrete_outcome=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False,
                 use_ray=False,
                 ray_remote_func_options=None):
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
                         treatment_featurizer=treatment_featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         linear_first_stages=linear_first_stages,
                         discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing,
                         use_ray=use_ray,
                         ray_remote_func_options=ray_remote_func_options
                         )

    def _gen_allowed_missing_vars(self):
        return ['W'] if self.allow_missing else []

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

    def fit(self, Y, T, *, X=None, W=None, sample_weight=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

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
        sample_weight : (n,) array_like or None
            Individual weights for each sample. If None, it assumes equal weight.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'debiasedlasso'
            (or an instance of :class:`.LinearModelFinalInference`)

        Returns
        -------
        self
        """
        # TODO: support freq_weight and sample_var in debiased lasso
        check_high_dimensional(X, T, threshold=5, featurizer=self.featurizer,
                               discrete_treatment=self.discrete_treatment,
                               msg="The number of features in the final model (< 5) is too small for a sparse model. "
                               "We recommend using the LinearDML estimator for this low-dimensional setting.")
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, groups=groups,
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
    model_y: estimator, default ``'auto'``
        Determines how to fit the outcome to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t: estimator, default ``'auto'``
        Determines how to fit the treatment to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    fit_cate_intercept : bool, default True
        Whether the linear CATE model should have a constant term.

    dim: int, default 20
        The number of random Fourier features to generate

    bw: float, default 1.0
        The bandwidth of the Gaussian used to generate features

    discrete_outcome: bool, default ``False``
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default ``False``
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply model_y, model_t that can handle
        missing values.

    use_ray: bool, default False
        Whether to use Ray to parallelize the cross-fitting step. If True, Ray must be installed.

    ray_remote_func_options : dict, default None
        Options to pass to the remote function when using Ray.
        See https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html

    Examples
    --------
    A simple example with the default models and discrete treatment:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dml import KernelDML

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = KernelDML(discrete_treatment=True, dim=10, bw=5)
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([0.63041..., 1.86098..., 0.74218...])
    """

    def __init__(self, model_y='auto', model_t='auto',
                 discrete_outcome=False,
                 discrete_treatment=False,
                 treatment_featurizer=None,
                 categories='auto',
                 fit_cate_intercept=True,
                 dim=20,
                 bw=1.0,
                 cv=2,
                 mc_iters=None, mc_agg='mean',
                 random_state=None,
                 allow_missing=False,
                 use_ray=False,
                 ray_remote_func_options=None):
        self.dim = dim
        self.bw = bw
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=None,
                         featurizer=None,
                         treatment_featurizer=treatment_featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing,
                         use_ray=use_ray,
                         ray_remote_func_options=ray_remote_func_options
                         )

    def _gen_allowed_missing_vars(self):
        return ['W'] if self.allow_missing else []

    def _gen_model_final(self):
        return ElasticNetCV(fit_intercept=False, random_state=self.random_state)

    def _gen_featurizer(self):
        return _RandomFeatures(dim=self.dim, bw=self.bw, random_state=self.random_state)

    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

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
        sample_weight : (n,) array_like or None
            Individual weights for each sample. If None, it assumes equal weight.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.LinearModelFinalInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, groups=groups,
                           cache_values=cache_values, inference=inference)

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
    model_y: estimator, default ``'auto'``
        Determines how to fit the outcome to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t: estimator, default ``'auto'``
        Determines how to fit the treatment to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods. It can be an arbitrary scikit-learn regressor. The `fit` method
        must accept `sample_weight` as a keyword argument.

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    discrete_outcome: bool, default ``False``
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default ``False``
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

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

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply model_y, model_t, and model_final
        that can handle missing values.

    use_ray: bool, default False
        Whether to use Ray to parallelize the cross-fitting step. If True, Ray must be installed.

    ray_remote_func_options : dict, default None
        Options to pass to the remote function when using Ray.
        See https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html

    Examples
    --------
    A simple example with a discrete treatment:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dml import NonParamDML
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = NonParamDML(
            model_y=RandomForestRegressor(min_samples_leaf=20),
            model_t=RandomForestClassifier(min_samples_leaf=20),
            model_final=RandomForestRegressor(min_samples_leaf=20),
            discrete_treatment=True
        )
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([0.35318..., 1.28760..., 0.83506...])
    """

    def __init__(self, *,
                 model_y, model_t, model_final,
                 featurizer=None,
                 discrete_outcome=False,
                 discrete_treatment=False,
                 treatment_featurizer=None,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False,
                 use_ray=False,
                 ray_remote_func_options=None):
        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.model_final = clone(model_final, safe=False)
        super().__init__(discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         treatment_featurizer=treatment_featurizer,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing,
                         use_ray=use_ray,
                         ray_remote_func_options=ray_remote_func_options
                         )

    def _gen_allowed_missing_vars(self):
        return ['X', 'W'] if self.allow_missing else []

    def _get_inference_options(self):
        # add blb to parent's options
        options = super()._get_inference_options()
        options.update(auto=GenericSingleTreatmentModelFinalInference)
        return options

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        return _make_first_stage_selector(self.model_y, is_discrete=self.discrete_outcome,
                                          random_state=self.random_state)

    def _gen_model_t(self):
        return _make_first_stage_selector(self.model_t, is_discrete=self.discrete_treatment,
                                          random_state=self.random_state)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_rlearner_model_final(self):
        return _FinalWrapper(self._gen_model_final(), False, self._gen_featurizer(), True)

    # override only so that we can update the docstring to indicate
    # support for `GenericSingleTreatmentModelFinalInference`
    def fit(self, Y, T, *, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

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
        sample_weight : (n,) array_like, optional
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array_like of int, optional
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(n,), (n, d_y)} nd array_like, optional
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.GenericSingleTreatmentModelFinalInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var,
                           groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        return _shap_explain_model_cate(self.const_marginal_effect, self.model_cate, X, self._d_t, self._d_y,
                                        featurizer=self.featurizer_,
                                        feature_names=feature_names,
                                        treatment_names=treatment_names,
                                        output_names=output_names,
                                        input_names=self._input_names,
                                        background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__
