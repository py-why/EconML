# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""

Orthogonal Machine Learning is a general approach to estimating causal models
by formulating them as minimizers of some loss function that depends on
auxiliary regression models that also need to be estimated from data. The
class in this module implements the general logic in a very versatile way
so that various child classes can simply instantiate the appropriate models
and save a lot of code repetition.

References
----------

Dylan Foster, Vasilis Syrgkanis (2019). Orthogonal Statistical Learning.
    ACM Conference on Learning Theory. https://arxiv.org/abs/1901.09036

Xinkun Nie, Stefan Wager (2017). Quasi-Oracle Estimation of Heterogeneous Treatment Effects.
    https://arxiv.org/abs/1712.04912

Chernozhukov et al. (2017). Double/debiased machine learning for treatment and structural parameters.
    The Econometrics Journal. https://arxiv.org/abs/1608.00060

"""

import copy
from collections import namedtuple
from warnings import warn
from abc import abstractmethod
from typing import List, Union
import inspect
from collections import defaultdict
import re

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold, check_cv
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder,
                                   OneHotEncoder)
from sklearn.utils import check_random_state

from ._cate_estimator import (BaseCateEstimator, LinearCateEstimator,
                              TreatmentExpansionMixin)
from .inference import BootstrapInference
from .utilities import (_deprecate_positional, check_input_arrays,
                        cross_product, filter_none_kwargs, one_hot_encoder, strata_from_discrete_arrays,
                        inverse_onehot, jacify_featurizer, ndim, reshape, shape, transpose)
from .sklearn_extensions.model_selection import ModelSelector

try:
    import ray
except ImportError as exn:
    from .utilities import MissingModule
    ray = MissingModule(
        "Ray is not a dependency of the base econml package; install econml[ray] or econml[all] to require it, "
        "or install ray separately, to use functionality that depends on ray", exn)


def _fit_fold(model, train_idxs, test_idxs, calculate_scores, args, kwargs):
    """
    Fits a single model on the training data and calculates the nuisance value on the test data.
    model:  object
        An object that supports fit and predict. Fit must accept all the args
        and the keyword arguments kwargs. Similarly predict must all accept
        all the args as arguments and kwards as keyword arguments. The fit
        function estimates a model of the nuisance function, based on the input
        data to fit. Predict evaluates the fitted nuisance function on the input
        data to predict.

    train_idxs (array-like): Indices for the training data.
    test_idxs (array-like): Indices for the test data.
    calculate_scores (bool): Whether to calculate scores after fitting.

    args : a sequence of (numpy matrices or None)
        Each matrix is a data variable whose first index corresponds to a sample
    kwargs : a sequence of key-value args, with values being (numpy matrices or None)
        Each keyword argument is of the form Var=x, with x a numpy array. Each
        of these arrays are data variables. The model fit and predict will be
        called with signature: `model.fit(*args, **kwargs)` and
        `model.predict(*args, **kwargs)`. Key-value arguments that have value
        None, are ommitted from the two calls. So all the args and the non None
        kwargs variables must be part of the models signature.
    Returns:
    --------
    -Tuple containing:
    nuisance_temp (tuple): Predictions or values of interest from the model.
    fitted_model: The fitted model after training.
    score_temp (tuple or None): Scores calculated after fitting if `calculate_scores` is True, otherwise None.
    """
    model = clone(model, safe=False)

    args_train = tuple(var[train_idxs] if var is not None else None for var in args)
    args_test = tuple(var[test_idxs] if var is not None else None for var in args)

    kwargs_train = {key: var[train_idxs] for key, var in kwargs.items()}
    kwargs_test = {key: var[test_idxs] for key, var in kwargs.items()}

    model.train(False, None, *args_train, **kwargs_train)
    nuisance_temp = model.predict(*args_test, **kwargs_test)

    if not isinstance(nuisance_temp, tuple):
        nuisance_temp = (nuisance_temp,)

    if calculate_scores:
        score_temp = model.score(*args_test, **kwargs_test)

        if not isinstance(score_temp, tuple):
            score_temp = (score_temp,)

    return nuisance_temp, model, (score_temp if calculate_scores else None)


def _crossfit(models: Union[ModelSelector, List[ModelSelector]], folds, use_ray, ray_remote_fun_option,
              *args, **kwargs):
    """
    General crossfit based calculation of nuisance parameters.

    Parameters
    ----------
    models : ModelSelector or List[ModelSelector]
        One or more objects that have train and predict methods.
        The train method must take an 'is_selecting' argument first, a set of folds second
        (which will be None when not selecting) and then accept positional arguments `args`
        and keyword arguments `kwargs`; the predict method just takes those `args` and `kwargs`.
        The train method selects or estimates a model of the nuisance function, based on the input
        data to fit. Predict evaluates the fitted nuisance function on the input
        data to predict.
    folds : list of tuple or None
        The crossfitting fold structure. Every entry in the list is a tuple whose
        first element are the training indices of the args and kwargs data and
        the second entry are the test indices. If the union of the test indices
        is not the full set of all indices, then the remaining nuisance parameters
        for the missing indices have value NaN.  If folds is None, then cross fitting
        is not performed; all indices are used for both model fitting and prediction
    use_ray: bool, default False
        Flag to indicate whether to use ray to parallelize the cross-fitting step.
    ray_remote_fun_option: dict, default None
        Options to pass to the ray.remote decorator.
    args : a sequence of (numpy matrices or None)
        Each matrix is a data variable whose first index corresponds to a sample
    kwargs : a sequence of key-value args, with values being (numpy matrices or None)
        Each keyword argument is of the form Var=x, with x a numpy array. Each
        of these arrays are data variables. The model fit and predict will be
        called with signature: `model.fit(*args, **kwargs)` and
        `model.predict(*args, **kwargs)`. Key-value arguments that have value
        None, are ommitted from the two calls. So all the args and the non None
        kwargs variables must be part of the models signature.

    Returns
    -------
    nuisances : tuple of array_like
        Each entry in the tuple is a nuisance parameter matrix. Each row i-th in the
        matrix corresponds to the value of the nuisance parameter for the i-th input
        sample.
    model_list : list of object of same type as input model
        The cloned and fitted models for each fold. Can be used for inspection of the
        variability of the fitted models across folds.
    fitted_inds : np array1d
        The indices of the arrays for which the nuisance value was calculated. This
        corresponds to the union of the indices of the test part of each fold in
        the input fold list.
    scores : tuple of list of float or None
        The out-of-sample model scores for each nuisance model

    Examples
    --------

    .. testcode::

        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.linear_model import Lasso
        from econml._ortho_learner import _crossfit
        class Wrapper:
            def __init__(self, model):
                self._model = model
            def train(self, is_selecting, folds, X, y, W=None):
                self._model.fit(X, y)
                return self
            def predict(self, X, y, W=None):
                return self._model.predict(X)
        np.random.seed(123)
        X = np.random.normal(size=(5000, 3))
        y = X[:, 0] + np.random.normal(size=(5000,))
        folds = list(KFold(2).split(X, y))
        model = Lasso(alpha=0.01)
        use_ray = False
        ray_remote_fun_option = {}
        nuisance, model_list, fitted_inds, scores = _crossfit(Wrapper(model),folds, use_ray, ray_remote_fun_option,
         X, y,W=y, Z=None)

    >>> nuisance
    (array([-1.105728... , -1.537566..., -2.451827... , ...,  1.106287...,
       -1.829662..., -1.782273...]),)
    >>> model_list
    [<Wrapper object at 0x...>, <Wrapper object at 0x...>]
    >>> fitted_inds
    array([   0,    1,    2, ..., 4997, 4998, 4999])

    """
    model_list = []

    kwargs = filter_none_kwargs(**kwargs)

    n = (args[0] if args else kwargs.items()[0][1]).shape[0]

    # fully materialize folds so that they can be reused across models
    # and precompute fitted indices so that we fail fast if there's an issue with them
    if folds is not None:
        folds = list(folds)
        fitted_inds = []
        for idx, (train_idxs, test_idxs) in enumerate(folds):
            if len(np.intersect1d(train_idxs, test_idxs)) > 0:
                raise AttributeError(f"Invalid crossfitting fold structure. Train and test indices of fold {idx+1} "
                                     f"are not disjoint: {train_idxs}, {test_idxs}")
            common_idxs = np.intersect1d(fitted_inds, test_idxs)
            if len(common_idxs) > 0:
                raise AttributeError(f"Invalid crossfitting fold structure. The indexes {common_idxs} in fold {idx+1} "
                                     f"have appeared in previous folds")
            fitted_inds = np.concatenate((fitted_inds, test_idxs))
        fitted_inds = np.sort(fitted_inds.astype(int))
    else:
        fold_vals = [(np.arange(n), np.arange(n))]
        fitted_inds = np.arange(n)

    accumulated_nuisances = ()
    # NOTE: if any model is missing scores we will just return None even if another model
    #       has scores. this is because we don't know how many scores are missing
    #       for the models that are missing them, so we don't know how to pad the array
    calculate_scores = True
    accumulated_scores = ()

    # for convenience we allos a single model to be passed in lieu of a singleton list
    # in that case, we will also unwrap the model output
    unwrap_model_output = False
    if not isinstance(models, list):
        unwrap_model_output = True
        models = [models]

    for model in models:
        # when there is more than one model, nuisances from previous models
        # come first as positional arguments
        accumulated_args = accumulated_nuisances + args
        model.train(True, fold_vals if folds is None else folds, *accumulated_args, **kwargs)

        calculate_scores &= hasattr(model, 'score')

        model_list.append([])  # add a new empty list of clones for this model

        if folds is None:  # skip crossfitting
            model_list[-1].append(clone(model, safe=False))
            model_list[-1][0].train(False, None, *accumulated_args, **kwargs)  # fit the selected model
            nuisances = model_list[-1][0].predict(*accumulated_args, **kwargs)
            if not isinstance(nuisances, tuple):
                nuisances = (nuisances,)

            if calculate_scores:
                scores = model_list[-1][0].score(*accumulated_args, **kwargs)
                if not isinstance(scores, tuple):
                    scores = (scores,)
                # scores entries should be lists of scores, so make each entry a singleton list
                scores = tuple([s] for s in scores)

        else:
            fold_refs = []
            if use_ray:
                # Adding the kwargs to ray object store to be used by remote functions
                # for each fold to avoid IO overhead
                ray_args = ray.put(kwargs)
                for idx, (train_idxs, test_idxs) in enumerate(folds):
                    fold_refs.append(
                        ray.remote(_fit_fold).options(**ray_remote_fun_option).remote(model, train_idxs, test_idxs,
                                                                                      calculate_scores,
                                                                                      accumulated_args, ray_args))
            for idx, (train_idxs, test_idxs) in enumerate(folds):
                if use_ray:
                    nuisance_temp, model_out, score_temp = ray.get(fold_refs[idx])
                else:
                    nuisance_temp, model_out, score_temp = _fit_fold(model, train_idxs, test_idxs,
                                                                     calculate_scores, accumulated_args, kwargs)

                if idx == 0:
                    nuisances = tuple([np.full((n,) + nuis.shape[1:], np.nan)
                                      for nuis in nuisance_temp])

                for it, nuis in enumerate(nuisance_temp):
                    nuisances[it][test_idxs] = nuis

                if calculate_scores:
                    if idx == 0:
                        scores = tuple([] for _ in score_temp)
                    for it, score in enumerate(score_temp):
                        scores[it].append(score)

                model_list[-1].append(model_out)

        accumulated_nuisances += nuisances
        if calculate_scores:
            accumulated_scores += scores
        else:
            accumulated_scores = None

    if unwrap_model_output:
        model_list = model_list[0]
    return accumulated_nuisances, model_list, fitted_inds, accumulated_scores


CachedValues = namedtuple('CachedValues', ['nuisances',
                                           'Y', 'T', 'X', 'W', 'Z', 'sample_weight', 'freq_weight',
                                           'sample_var', 'groups'])


class _OrthoLearner(TreatmentExpansionMixin, LinearCateEstimator):
    """
    Base class for all orthogonal learners. This class is a parent class to any method that has
    the following architecture:

    1.  The CATE :math:`\\theta(X)` is the minimizer of some expected loss function

        .. math ::
            \\mathbb{E}[\\ell(V; \\theta(X), h(V))]

        where :math:`V` are all the random variables and h is a vector of nuisance functions. Alternatively,
        the class would also work if :math:`\\theta(X)` is the solution to a set of moment equations that
        also depend on nuisance functions :math:`h`.

    2.  To estimate :math:`\\theta(X)` we first fit the h functions and calculate :math:`h(V_i)` for each sample
        :math:`i` in a crossfit manner:

            - Let (F1_train, F1_test), ..., (Fk_train, Fk_test) be any KFold partition
              of the data, where Ft_train, Ft_test are subsets of indices of the input samples and such that
              F1_train is disjoint from F1_test. The sets F1_test, ..., Fk_test form an incomplete partition
              of all the input indices, i.e. they are be disjoint and their union could potentially be a subset of
              all input indices. For instance, in a time series split F0_train could be a prefix of the data and
              F0_test the suffix. Typically, these folds will be created
              by a KFold split, i.e. if S1, ..., Sk is any partition of the data, then Ft_train is the set of
              all indices except St and Ft_test = St. If the union of the Ft_test is not all the data, then only the
              subset of the data in the union of the Ft_test sets will be used in the final stage.

            - Then for each t in [1, ..., k]

                - Estimate a model :math:`\\hat{h}_t` for :math:`h` using Ft_train
                - Evaluate the learned :math:`\\hat{h}_t` model on the data in Ft_test and use that value
                  as the nuisance value/vector :math:`\\hat{U}_i=\\hat{h}(V_i)` for the indices i in Ft_test

    3.  Estimate the model for :math:`\\theta(X)` by minimizing the empirical (regularized) plugin loss on
        the subset of indices for which we have a nuisance value, i.e. the union of {F1_test, ..., Fk_test}:

        .. math ::
            \\mathbb{E}_n[\\ell(V; \\theta(X), \\hat{h}(V))]\
            = \\frac{1}{n} \\sum_{i=1}^n \\ell(V_i; \\theta(X_i), \\hat{U}_i)

        The method is a bit more general in that the final step does not need to be a loss minimization step.
        The class takes as input a model for fitting an estimate of the nuisance h given a set of samples
        and predicting the value of the learned nuisance model on any other set of samples. It also
        takes as input a model for the final estimation, that takes as input the data and their associated
        estimated nuisance values from the first stage and fits a model for the CATE :math:`\\theta(X)`. Then
        at predict time, the final model given any set of samples of the X variable, returns the estimated
        :math:`\\theta(X)`.

    The method essentially implements all the crossfit and plugin logic, so that any child classes need
    to only implement the appropriate `model_nuisance` and `model_final` and essentially nothing more.
    It also implements the basic preprocessing logic behind the expansion of discrete treatments into
    one-hot encodings.

    Parameters
    ----------
    discrete_outcome: bool
        Whether the outcome should be treated as binary

    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    treatment_featurizer : :term:`transformer` or None
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    discrete_instrument: bool
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

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

        Unless an iterable is used, we call `split(concat[Z, W, X], T)` to generate the splits. If all
        Z, W, X are None, then we call `split(ones((T.shape[0], 1)), T)`.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    allow_missing: bool
        Whether to allow missing values in X, W. If True, will need to supply nuisance models that can handle
        missing values.

    use_ray: bool, default False
        Whether to use ray to parallelize the cross-fitting step.

    ray_remote_func_options: dict, default None
        Options to pass to the ray.remote decorator.

    Examples
    --------

    The example code below implements a very simple version of the double machine learning
    method on top of the :class:`._OrthoLearner` class, for expository purposes.
    For a more elaborate implementation of a Double Machine Learning child class of the class
    :class:`._OrthoLearner` check out :class:`.DML`
    and its child classes:

    .. testcode::

        import numpy as np
        from sklearn.linear_model import LinearRegression
        from econml._ortho_learner import _OrthoLearner
        class ModelNuisance:
            def __init__(self, model_t, model_y):
                self._model_t = model_t
                self._model_y = model_y
            def train(self, is_selecting, folds, Y, T, W=None):
                self._model_t.fit(W, T)
                self._model_y.fit(W, Y)
                return self
            def predict(self, Y, T, W=None):
                return Y - self._model_y.predict(W), T - self._model_t.predict(W)
        class ModelFinal:
            def __init__(self):
                return
            def fit(self, Y, T, W=None, nuisances=None):
                Y_res, T_res = nuisances
                self.model = LinearRegression(fit_intercept=False).fit(T_res.reshape(-1, 1), Y_res)
                return self
            def predict(self, X=None):
                return self.model.coef_[0]
            def score(self, Y, T, W=None, nuisances=None):
                Y_res, T_res = nuisances
                return np.mean((Y_res - self.model.predict(T_res.reshape(-1, 1)))**2)
        class OrthoLearner(_OrthoLearner):
            def _gen_ortho_learner_model_nuisance(self):
                return ModelNuisance(LinearRegression(), LinearRegression())
            def _gen_ortho_learner_model_final(self):
                return ModelFinal()
        np.random.seed(123)
        X = np.random.normal(size=(100, 3))
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, size=(100,))
        est = OrthoLearner(cv=2, discrete_outcome=False, discrete_treatment=False, treatment_featurizer=None,
                           discrete_instrument=False, categories='auto', random_state=None)
        est.fit(y, X[:, 0], W=X[:, 1:])

    >>> est.score_
    0.00756830...
    >>> est.const_marginal_effect()
    1.02364992...
    >>> est.effect()
    array([1.023649...])
    >>> est.effect(T0=0, T1=10)
    array([10.236499...])
    >>> est.score(y, X[:, 0], W=X[:, 1:])
    0.00727995...
    >>> est.ortho_learner_model_final_.model
    LinearRegression(fit_intercept=False)
    >>> est.ortho_learner_model_final_.model.coef_
    array([1.023649...])

    The following example shows how to do double machine learning with discrete treatments, using
    the _OrthoLearner:

    .. testcode::

        class ModelNuisance:
            def __init__(self, model_t, model_y):
                self._model_t = model_t
                self._model_y = model_y
            def train(self, is_selecting, folds, Y, T, W=None):
                self._model_t.fit(W, np.matmul(T, np.arange(1, T.shape[1]+1)))
                self._model_y.fit(W, Y)
                return self
            def predict(self, Y, T, W=None):
                return Y - self._model_y.predict(W), T - self._model_t.predict_proba(W)[:, 1:]
        class ModelFinal:
            def __init__(self):
                return
            def fit(self, Y, T, W=None, nuisances=None):
                Y_res, T_res = nuisances
                self.model = LinearRegression(fit_intercept=False).fit(T_res.reshape(-1, 1), Y_res)
                return self
            def predict(self):
                # theta needs to be of dimension (1, d_t) if T is (n, d_t)
                return np.array([[self.model.coef_[0]]])
            def score(self, Y, T, W=None, nuisances=None):
                Y_res, T_res = nuisances
                return np.mean((Y_res - self.model.predict(T_res.reshape(-1, 1)))**2)
        from sklearn.linear_model import LogisticRegression
        class OrthoLearner(_OrthoLearner):
            def _gen_ortho_learner_model_nuisance(self):
                return ModelNuisance(LogisticRegression(solver='lbfgs'), LinearRegression())
            def _gen_ortho_learner_model_final(self):
                return ModelFinal()
        np.random.seed(123)
        W = np.random.normal(size=(100, 3))
        import scipy.special
        T = np.random.binomial(1, scipy.special.expit(W[:, 0]))
        y = T + W[:, 0] + np.random.normal(0, 0.01, size=(100,))
        est = OrthoLearner(cv=2, discrete_outcome=False, discrete_treatment=True, discrete_instrument=False,
                           treatment_featurizer=None, categories='auto', random_state=None)
        est.fit(y, T, W=W)

    >>> est.score_
    0.00673015...
    >>> est.const_marginal_effect()
    array([[1.008401...]])
    >>> est.effect()
    array([1.008401...])
    >>> est.score(y, T, W=W)
    0.00310431...
    >>> est.ortho_learner_model_final_.model.coef_[0]
    1.00840170...

    Attributes
    ----------
    models_nuisance_: nested list of objects of type(model_nuisance)
        A nested list of instances of the model_nuisance object. The number of sublist equals to the
        number of monte carlo iterations. Each element in the sublist corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    ortho_learner_model_final_: object of type(model_final)
        An instance of the model_final object that was fitted after calling fit.
    score_ : float or array of floats
        If the model_final has a score method, then `score_` contains the outcome of the final model
        score when evaluated on the fitted nuisances from the first stage. Represents goodness of fit,
        of the final CATE model.
    nuisance_scores_ : tuple of list of list of float or None
        The out-of-sample scores from training each nuisance model
    """

    def __init__(self, *,
                 discrete_outcome,
                 discrete_treatment,
                 treatment_featurizer,
                 discrete_instrument,
                 categories,
                 cv,
                 random_state,
                 mc_iters=None,
                 mc_agg='mean',
                 allow_missing=False,
                 use_ray=False,
                 ray_remote_func_options=None):
        self.cv = cv
        self.discrete_outcome = discrete_outcome
        self.discrete_treatment = discrete_treatment
        self.treatment_featurizer = treatment_featurizer
        self.discrete_instrument = discrete_instrument
        self.random_state = random_state
        self.categories = categories
        self.mc_iters = mc_iters
        self.mc_agg = mc_agg
        self.allow_missing = allow_missing
        self.use_ray = use_ray
        self.ray_remote_func_options = ray_remote_func_options
        super().__init__()

    def _gen_allowed_missing_vars(self):
        return ['X', 'W'] if self.allow_missing else []

    @abstractmethod
    def _gen_ortho_learner_model_nuisance(self):
        """Must return a fresh instance of a nuisance model selector

        Returns
        -------
        model_nuisance: list of selector
            The selector(s) for fitting the nuisance function. The returned estimators must implement
            `train` and `predict` methods that both have signatures::

                model_nuisance.train(is_selecting, folds, Y, T, X=X, W=W, Z=Z,
                                sample_weight=sample_weight)
                model_nuisance.predict(Y, T, X=X, W=W, Z=Z,
                                    sample_weight=sample_weight)

            In fact we allow for the model method signatures to skip any of the keyword arguments
            as long as the class is always called with the omitted keyword argument set to ``None``.
            This can be enforced in child classes by re-implementing the fit and the various effect
            methods. If ``discrete_treatment=True``, then the input ``T`` to both above calls will be the
            one-hot encoding of the original input ``T``, excluding the first column of the one-hot.

            If the estimator also provides a score method with the same arguments as fit, it will be used to
            calculate scores during training.
        """
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def _gen_ortho_learner_model_final(self):
        """ Must return a fresh instance of a final model

        Returns
        -------
        model_final: estimator for fitting the response residuals to the features and treatment residuals
            Must implement `fit` and `predict` methods that must have signatures::

                model_final.fit(Y, T, X=X, W=W, Z=Z, nuisances=nuisances,
                                sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var)
                model_final.predict(X=X)

            Predict, should just take the features X and return the constant marginal effect. In fact we allow
            for the model method signatures to skip any of the keyword arguments as long as the class is always
            called with the omitted keyword argument set to ``None``. Moreover, the predict function of the final
            model can take no argument if the class is always called with ``X=None``. This can be enforced in child
            classes by re-implementing the fit and the various effect methods. If ``discrete_treatment=True``,
            then the input ``T`` to both above calls will be the one-hot encoding of the original input ``T``,
            excluding the first column of the one-hot.
        """
        raise NotImplementedError("Abstract method")

    def _check_input_dims(self, Y, T, X=None, W=None, Z=None, *other_arrays):
        assert shape(Y)[0] == shape(T)[0], "Dimension mis-match!"
        for arr in [X, W, Z, *other_arrays]:
            assert (arr is None) or (arr.shape[0] == Y.shape[0]), "Dimension mismatch"
        self._d_x = X.shape[1:] if X is not None else None
        self._d_w = W.shape[1:] if W is not None else None
        self._d_z = Z.shape[1:] if Z is not None else None

    def _check_fitted_dims(self, X):
        if X is None:
            assert self._d_x is None, "X was not None when fitting, so can't be none for score or effect"
        else:
            assert self._d_x == X.shape[1:], "Dimension mis-match of X with fitted X"

    def _check_fitted_dims_w_z(self, W, Z):
        if W is None:
            assert self._d_w is None, "W was not None when fitting, so can't be none for score"
        else:
            assert self._d_w == W.shape[1:], "Dimension mis-match of W with fitted W"

        if Z is None:
            assert self._d_z is None, "Z was not None when fitting, so can't be none for score"
        else:
            assert self._d_z == Z.shape[1:], "Dimension mis-match of Z with fitted Z"

    def _subinds_check_none(self, var, inds):
        return var[inds] if var is not None else None

    def _strata(self, Y, T, X=None, W=None, Z=None,
                sample_weight=None, freq_weight=None, sample_var=None, groups=None,
                cache_values=False, only_final=False, check_input=True):
        arrs = []
        if self.discrete_outcome:
            arrs.append(Y)
        if self.discrete_treatment:
            arrs.append(T)
        if self.discrete_instrument:
            arrs.append(Z)

        return strata_from_discrete_arrays(arrs)

    def _prefit(self, Y, T, *args, only_final=False, **kwargs):

        # generate an instance of the final model
        self._ortho_learner_model_final = self._gen_ortho_learner_model_final()
        if not only_final:
            # generate an instance of the nuisance model
            self._ortho_learner_model_nuisance = self._gen_ortho_learner_model_nuisance()

        super()._prefit(Y, T, *args, **kwargs)

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, *, X=None, W=None, Z=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference=None, only_final=False, check_input=True):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        X: (n, d_x) matrix, optional
            Features for each sample
        W: (n, d_w) matrix, optional
            Controls for each sample
        Z: (n, d_z) matrix, optional
            Instruments for each sample
        sample_weight : (n,) array_like, optional
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n, ) array_like of int, optional
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(n,), (n, d_y)} nd array_like, optional
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the cv argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache the inputs and computed nuisances, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`).
        only_final: bool, defaul False
            Whether to fit the nuisance models or use the existing cached values
            Note. This parameter is only used internally by the `refit` method and should not be exposed
            publicly by overwrites of the `fit` method in public classes.
        check_input: bool, default True
            Whether to check if the input is valid
            Note. This parameter is only used internally by the `refit` method and should not be exposed
            publicly by overwrites of the `fit` method in public classes.

        Returns
        -------
        self : object
        """
        self._random_state = check_random_state(self.random_state)
        assert (freq_weight is None) == (
            sample_var is None), "Sample variances and frequency weights must be provided together!"
        assert not (self.discrete_treatment and self.treatment_featurizer), "Treatment featurization " \
            "is not supported when treatment is discrete"
        if check_input:
            Y, T, Z, sample_weight, freq_weight, sample_var, groups = check_input_arrays(
                Y, T, Z, sample_weight, freq_weight, sample_var, groups)
            X, = check_input_arrays(
                X, force_all_finite='allow-nan' if 'X' in self._gen_allowed_missing_vars() else True)
            W, = check_input_arrays(
                W, force_all_finite='allow-nan' if 'W' in self._gen_allowed_missing_vars() else True)
            self._check_input_dims(Y, T, X, W, Z, sample_weight, freq_weight, sample_var, groups)

        if not only_final:

            if self.discrete_outcome:
                self.outcome_transformer = LabelEncoder()
                self.outcome_transformer.fit(Y)
                if Y.shape[1:] and Y.shape[1] > 1:
                    raise ValueError(
                        f"Only one outcome variable is supported when discrete_outcome=True. Got Y of shape {Y.shape}")
                if len(self.outcome_transformer.classes_) > 2:
                    raise AttributeError(
                        f"({len(self.outcome_transformer.classes_)} outcome classes detected. "
                        "Currently, only 2 outcome classes are allowed when discrete_outcome=True. "
                        f"Classes provided include {self.outcome_transformer.classes_[:5]}")
            else:
                self.outcome_transformer = None

            if self.discrete_treatment:
                categories = self.categories
                if categories != 'auto':
                    categories = [categories]  # OneHotEncoder expects a 2D array with features per column
                self.transformer = one_hot_encoder(categories=categories, drop='first')
                self.transformer.fit(reshape(T, (-1, 1)))
                self._d_t = (len(self.transformer.categories_[0]) - 1,)
            elif self.treatment_featurizer:
                self._original_treatment_featurizer = clone(self.treatment_featurizer, safe=False)
                self.transformer = jacify_featurizer(self.treatment_featurizer)
                output_T = self.transformer.fit_transform(T)
                self._d_t = np.shape(output_T)[1:]
            else:
                self.transformer = None

            if self.discrete_instrument:
                self.z_transformer = one_hot_encoder(categories='auto', drop='first')
                self.z_transformer.fit(reshape(Z, (-1, 1)))
            else:
                self.z_transformer = None

            all_nuisances = []
            fitted_inds = None
            if sample_weight is None:
                if freq_weight is not None:
                    sample_weight_nuisances = freq_weight
                else:
                    sample_weight_nuisances = None
            else:
                if freq_weight is not None:
                    sample_weight_nuisances = freq_weight * sample_weight
                else:
                    sample_weight_nuisances = sample_weight

            self._models_nuisance = []

            if self.use_ray:
                if not ray.is_initialized():
                    ray.init()
                self.ray_remote_func_options = self.ray_remote_func_options or {}

                # Define Ray remote function (Ray remote wrapper of the _fit_nuisances function)
                def _fit_nuisances(Y, T, X, W, Z, sample_weight, groups):
                    return self._fit_nuisances(Y, T, X, W, Z, sample_weight=sample_weight, groups=groups)

                # Create Ray remote jobs for parallel processing
                self.nuisances_ref = [ray.remote(_fit_nuisances).options(**self.ray_remote_func_options).remote(
                    Y, T, X, W, Z, sample_weight_nuisances, groups) for _ in range(self.mc_iters or 1)]

            for idx in range(self.mc_iters or 1):
                if self.use_ray:
                    nuisances, fitted_models, new_inds, scores = ray.get(self.nuisances_ref[idx])
                else:
                    nuisances, fitted_models, new_inds, scores = self._fit_nuisances(
                        Y, T, X, W, Z, sample_weight=sample_weight_nuisances, groups=groups)
                all_nuisances.append(nuisances)
                self._models_nuisance.append(fitted_models)
                if scores is None:
                    self.nuisance_scores_ = None
                else:
                    if idx == 0:
                        self.nuisance_scores_ = tuple([] for _ in scores)
                    for ind, score in enumerate(scores):
                        self.nuisance_scores_[ind].append(score)
                if fitted_inds is None:
                    fitted_inds = new_inds
                elif not np.array_equal(fitted_inds, new_inds):
                    raise AttributeError("Different indices were fit by different folds, so they cannot be aggregated")

            if self.mc_iters is not None:
                if self.mc_agg == 'mean':
                    nuisances = tuple(np.mean(nuisance_mc_variants, axis=0)
                                      for nuisance_mc_variants in zip(*all_nuisances))
                elif self.mc_agg == 'median':
                    nuisances = tuple(np.median(nuisance_mc_variants, axis=0)
                                      for nuisance_mc_variants in zip(*all_nuisances))
                else:
                    raise ValueError(
                        f"Parameter `mc_agg` must be one of {{'mean', 'median'}}. Got {self.mc_agg}")

            Y, T, X, W, Z, sample_weight, freq_weight, sample_var = (self._subinds_check_none(arr, fitted_inds)
                                                                     for arr in (Y, T, X, W, Z, sample_weight,
                                                                                 freq_weight, sample_var))
            nuisances = tuple([self._subinds_check_none(nuis, fitted_inds) for nuis in nuisances])
            self._cached_values = CachedValues(nuisances=nuisances,
                                               Y=Y, T=T, X=X, W=W, Z=Z,
                                               sample_weight=sample_weight,
                                               freq_weight=freq_weight,
                                               sample_var=sample_var,
                                               groups=groups) if cache_values else None
        else:
            nuisances = self._cached_values.nuisances
            # _d_t is altered by fit nuisances to what prefit does. So we need to perform the same
            # alteration even when we only want to fit_final.
            if self.transformer is not None:
                if self.discrete_treatment:
                    self._d_t = (len(self.transformer.categories_[0]) - 1,)
                else:
                    output_T = self.transformer.fit_transform(T)
                    self._d_t = np.shape(output_T)[1:]

        final_T = T
        if self.transformer:
            if (self.discrete_treatment):
                final_T = self.transformer.transform(final_T.reshape(-1, 1))
            else:  # treatment featurizer case
                final_T = output_T

        self._fit_final(Y=Y,
                        T=final_T,
                        X=X, W=W, Z=Z,
                        nuisances=nuisances,
                        sample_weight=sample_weight,
                        freq_weight=freq_weight,
                        sample_var=sample_var,
                        groups=groups)

        return self

    @property
    def _illegal_refit_inference_methods(self):
        return (BootstrapInference,)

    def refit_final(self, inference=None):
        """
        Estimate the counterfactual model using a new final model specification but with cached first stage results.

        In order for this to succeed, ``fit`` must have been called with ``cache_values=True``. This call
        will only refit the final model. This call we use the current setting of any parameters that change the
        final stage estimation. If any parameters that change how the first stage nuisance estimates
        has also been changed then it will have no effect. You need to call fit again to change the
        first stage estimation results.

        Parameters
        ----------
        inference : inference method, optional
            The string or object that represents the inference method

        Returns
        -------
        self : object
            This instance
        """
        assert self._cached_values, "Refit can only be called if values were cached during the original fit"
        if isinstance(self._get_inference(inference), self._illegal_refit_inference_methods):
            raise ValueError("The chosen inference method does not allow only for model final re-fitting.")
        cached = self._cached_values
        kwargs = filter_none_kwargs(
            Y=cached.Y, T=cached.T, X=cached.X, W=cached.W, Z=cached.Z,
            sample_weight=cached.sample_weight, freq_weight=cached.freq_weight, sample_var=cached.sample_var,
            groups=cached.groups,
        )
        _OrthoLearner.fit(self, **kwargs,
                          cache_values=True, inference=inference, only_final=True, check_input=False)
        return self

    def _fit_nuisances(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):

        # use a binary array to get stratified split in case of discrete treatment
        stratify = self.discrete_treatment or self.discrete_instrument or self.discrete_outcome
        strata = self._strata(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight, groups=groups)
        if strata is None:
            strata = T  # always safe to pass T as second arg to split even if we're not actually stratifying

        if self.transformer:
            if self.discrete_treatment:
                T = reshape(T, (-1, 1))
            T = self.transformer.transform(T)

        if self.discrete_instrument:
            Z = self.z_transformer.transform(reshape(Z, (-1, 1)))

        if self.discrete_outcome:
            Y = self.outcome_transformer.transform(Y).reshape(-1, 1)

        if self.cv == 1:  # special case, no cross validation
            folds = None
        else:
            splitter = check_cv(self.cv, [0], classifier=stratify)
            # if check_cv produced a new KFold or StratifiedKFold object, we need to set shuffle and random_state
            if splitter != self.cv and isinstance(splitter, (KFold, StratifiedKFold)):
                # upgrade to a GroupKFold or StratiGroupKFold if groups is not None
                if groups is not None:
                    if isinstance(splitter, KFold):
                        splitter = GroupKFold(n_splits=splitter.n_splits)
                    elif isinstance(splitter, StratifiedKFold):
                        splitter = StratifiedGroupKFold(n_splits=splitter.n_splits)
                splitter.shuffle = True
                splitter.random_state = self._random_state

            all_vars = [var if np.ndim(var) == 2 else var.reshape(-1, 1) for var in [Z, W, X] if var is not None]
            to_split = np.hstack(all_vars) if all_vars else np.ones((T.shape[0], 1))

            if groups is not None:
                # we won't have generated a KFold or StratifiedKFold ourselves when groups are passed,
                # but the user might have supplied one, which won't work
                if isinstance(splitter, (KFold, StratifiedKFold)):
                    raise TypeError("Groups were passed to fit while using a KFold or StratifiedKFold splitter. "
                                    "Instead you must initialize this object with a splitter that can handle groups.")
                folds = splitter.split(to_split, strata, groups=groups)
            else:
                folds = splitter.split(to_split, strata)

        nuisances, fitted_models, fitted_inds, scores = _crossfit(self._ortho_learner_model_nuisance, folds,
                                                                  self.use_ray, self.ray_remote_func_options, Y, T,
                                                                  X=X, W=W, Z=Z, sample_weight=sample_weight,
                                                                  groups=groups)
        return nuisances, fitted_models, fitted_inds, scores

    def _fit_final(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None,
                   freq_weight=None, sample_var=None, groups=None):
        self._ortho_learner_model_final.fit(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z,
                                                                       nuisances=nuisances,
                                                                       sample_weight=sample_weight,
                                                                       freq_weight=freq_weight,
                                                                       sample_var=sample_var,
                                                                       groups=groups))
        self.score_ = None
        if hasattr(self._ortho_learner_model_final, 'score'):
            self.score_ = self._ortho_learner_model_final.score(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z,
                                                                                           nuisances=nuisances,
                                                                                           sample_weight=sample_weight,
                                                                                           groups=groups))

    def const_marginal_effect(self, X=None):
        X, = check_input_arrays(X)
        self._check_fitted_dims(X)
        if X is None:
            return self._ortho_learner_model_final.predict()
        else:
            return self._ortho_learner_model_final.predict(X)

    const_marginal_effect.__doc__ = LinearCateEstimator.const_marginal_effect.__doc__

    def const_marginal_effect_interval(self, X=None, *, alpha=0.05):
        X, = check_input_arrays(X)
        self._check_fitted_dims(X)
        return super().const_marginal_effect_interval(X, alpha=alpha)

    const_marginal_effect_interval.__doc__ = LinearCateEstimator.const_marginal_effect_interval.__doc__

    def const_marginal_effect_inference(self, X=None):
        X, = check_input_arrays(X)
        self._check_fitted_dims(X)
        return super().const_marginal_effect_inference(X)

    const_marginal_effect_inference.__doc__ = LinearCateEstimator.const_marginal_effect_inference.__doc__

    def effect_interval(self, X=None, *, T0=0, T1=1, alpha=0.05):
        X, T0, T1 = check_input_arrays(X, T0, T1)
        self._check_fitted_dims(X)
        return super().effect_interval(X, T0=T0, T1=T1, alpha=alpha)

    effect_interval.__doc__ = LinearCateEstimator.effect_interval.__doc__

    def effect_inference(self, X=None, *, T0=0, T1=1):
        X, T0, T1 = check_input_arrays(X, T0, T1)
        self._check_fitted_dims(X)
        return super().effect_inference(X, T0=T0, T1=T1)

    effect_inference.__doc__ = LinearCateEstimator.effect_inference.__doc__

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds
        under different iterations. Then calls the score function of the model_final and
        returns the calculated score. The model_final model must have a score method.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        X: (n, d_x) matrix, optional
            Features for each sample
        W: (n, d_w) matrix, optional
            Controls for each sample
        Z: (n, d_z) matrix, optional
            Instruments for each sample
        sample_weight:(n,) vector, optional
            Weights for each samples
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.

        Returns
        -------
        score : float or (array of float)
            The score of the final CATE model on the new data. Same type as the return
            type of the model_final.score method.
        """
        if not hasattr(self._ortho_learner_model_final, 'score'):
            raise AttributeError("Final model does not have a score method!")
        Y, T, Z = check_input_arrays(Y, T, Z)
        X, = check_input_arrays(X, force_all_finite='allow-nan' if 'X' in self._gen_allowed_missing_vars() else True)
        W, = check_input_arrays(W, force_all_finite='allow-nan' if 'W' in self._gen_allowed_missing_vars() else True)
        self._check_fitted_dims(X)
        self._check_fitted_dims_w_z(W, Z)
        X, T = self._expand_treatments(X, T)
        if self.z_transformer is not None:
            Z = self.z_transformer.transform(reshape(Z, (-1, 1)))
        if self.discrete_outcome:
            Y = self.outcome_transformer.transform(Y).reshape(-1, 1)
        n_iters = len(self._models_nuisance)
        # self._models_nuisance will be a list of lists or a list of list of lists
        # so we use self._ortho_learner_model_nuisance to determine the nesting level
        expand_list = not isinstance(self._ortho_learner_model_nuisance, list)
        if expand_list:
            n_selectors = 1
            n_splits = len(self._models_nuisance[0])
        else:
            n_selectors = len(self._models_nuisance[0])
            n_splits = len(self._models_nuisance[0][0])

        kwargs = filter_none_kwargs(X=X, W=W, Z=Z, groups=groups)

        accumulated_nuisances = []

        for sel in range(n_selectors):
            accumulated_args = tuple(accumulated_nuisances) + (Y, T)
            # for each mc iteration
            for i, models_nuisances in enumerate(self._models_nuisance):
                if expand_list:
                    models_nuisances = [models_nuisances]

                # for each model under cross fit setting
                for j, mdl in enumerate(models_nuisances[sel]):
                    nuisance_temp = mdl.predict(*accumulated_args, **kwargs)
                    if not isinstance(nuisance_temp, tuple):
                        nuisance_temp = (nuisance_temp,)

                    if i == 0 and j == 0:
                        nuisances = [np.zeros((n_iters * n_splits,) + nuis.shape) for nuis in nuisance_temp]

                    for it, nuis in enumerate(nuisance_temp):
                        nuisances[it][j * n_iters + i] = nuis

            for it in range(len(nuisances)):
                nuisances[it] = np.mean(nuisances[it], axis=0)

            accumulated_nuisances += nuisances

        return self._ortho_learner_model_final.score(Y, T, nuisances=accumulated_nuisances,
                                                     **filter_none_kwargs(X=X, W=W, Z=Z,
                                                                          sample_weight=sample_weight, groups=groups))

    @property
    def ortho_learner_model_final_(self):
        if not hasattr(self, '_ortho_learner_model_final'):
            raise AttributeError("Model is not fitted!")
        return self._ortho_learner_model_final

    @property
    def models_nuisance_(self):
        if not hasattr(self, '_models_nuisance'):
            raise AttributeError("Model is not fitted!")
        return self._models_nuisance
