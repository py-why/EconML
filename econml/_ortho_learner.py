# Copyright (c) Microsoft Corporation. All rights reserved.
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
from warnings import warn

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.preprocessing import (FunctionTransformer, LabelEncoder,
                                   OneHotEncoder)
from sklearn.utils import check_random_state

from .cate_estimator import (BaseCateEstimator, LinearCateEstimator,
                             TreatmentExpansionMixin)
from .utilities import (_deprecate_positional, _EncoderWrapper, check_input_arrays,
                        cross_product, filter_none_kwargs,
                        inverse_onehot, ndim, reshape, shape, transpose)


def _crossfit(model, folds, *args, **kwargs):
    """
    General crossfit based calculation of nuisance parameters.

    Parameters
    ----------
    model : object
        An object that supports fit and predict. Fit must accept all the args
        and the keyword arguments kwargs. Similarly predict must all accept
        all the args as arguments and kwards as keyword arguments. The fit
        function estimates a model of the nuisance function, based on the input
        data to fit. Predict evaluates the fitted nuisance function on the input
        data to predict.
    folds : list of tuples or None
        The crossfitting fold structure. Every entry in the list is a tuple whose
        first element are the training indices of the args and kwargs data and
        the second entry are the test indices. If the union of the test indices
        is not the full set of all indices, then the remaining nuisance parameters
        for the missing indices have value NaN.  If folds is None, then cross fitting
        is not performed; all indices are used for both model fitting and prediction
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
    nuisances : tuple of numpy matrices
        Each entry in the tuple is a nuisance parameter matrix. Each row i-th in the
        matrix corresponds to the value of the nuisance parameter for the i-th input
        sample.
    model_list : list of objects of same type as input model
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
            def fit(self, X, y, W=None):
                self._model.fit(X, y)
                return self
            def predict(self, X, y, W=None):
                return self._model.predict(X)
        np.random.seed(123)
        X = np.random.normal(size=(5000, 3))
        y = X[:, 0] + np.random.normal(size=(5000,))
        folds = list(KFold(2).split(X, y))
        model = Lasso(alpha=0.01)
        nuisance, model_list, fitted_inds, scores = _crossfit(Wrapper(model), folds, X, y, W=y, Z=None)

    >>> nuisance
    (array([-1.105728... , -1.537566..., -2.451827... , ...,  1.106287...,
       -1.829662..., -1.782273...]),)
    >>> model_list
    [<Wrapper object at 0x...>, <Wrapper object at 0x...>]
    >>> fitted_inds
    array([   0,    1,    2, ..., 4997, 4998, 4999])

    """
    model_list = []
    fitted_inds = []
    calculate_scores = hasattr(model, 'score')

    # remove None arguments
    kwargs = filter_none_kwargs(**kwargs)

    if folds is None:  # skip crossfitting
        model_list.append(clone(model, safe=False))
        model_list[0].fit(*args, **kwargs)
        nuisances = model_list[0].predict(*args, **kwargs)
        scores = model_list[0].score(*args, **kwargs) if calculate_scores else None

        if not isinstance(nuisances, tuple):
            nuisances = (nuisances,)
        if not isinstance(scores, tuple):
            scores = (scores,)

        # scores entries should be lists of scores, so make each entry a singleton list
        scores = tuple([s] for s in scores)

        first_arr = args[0] if args else kwargs.items()[0][1]
        return nuisances, model_list, np.arange(first_arr.shape[0]), scores

    for idx, (train_idxs, test_idxs) in enumerate(folds):
        model_list.append(clone(model, safe=False))
        if len(np.intersect1d(train_idxs, test_idxs)) > 0:
            raise AttributeError("Invalid crossfitting fold structure." +
                                 "Train and test indices of each fold must be disjoint.")
        if len(np.intersect1d(fitted_inds, test_idxs)) > 0:
            raise AttributeError("Invalid crossfitting fold structure. The same index appears in two test folds.")
        fitted_inds = np.concatenate((fitted_inds, test_idxs))

        args_train = tuple(var[train_idxs] if var is not None else None for var in args)
        args_test = tuple(var[test_idxs] if var is not None else None for var in args)

        kwargs_train = {key: var[train_idxs] for key, var in kwargs.items()}
        kwargs_test = {key: var[test_idxs] for key, var in kwargs.items()}

        model_list[idx].fit(*args_train, **kwargs_train)

        nuisance_temp = model_list[idx].predict(*args_test, **kwargs_test)

        if not isinstance(nuisance_temp, tuple):
            nuisance_temp = (nuisance_temp,)

        if idx == 0:
            nuisances = tuple([np.full((args[0].shape[0],) + nuis.shape[1:], np.nan) for nuis in nuisance_temp])

        for it, nuis in enumerate(nuisance_temp):
            nuisances[it][test_idxs] = nuis

        if calculate_scores:
            score_temp = model_list[idx].score(*args_test, **kwargs_test)

            if not isinstance(score_temp, tuple):
                score_temp = (score_temp,)

            if idx == 0:
                scores = tuple([] for _ in score_temp)

            for it, score in enumerate(score_temp):
                scores[it].append(score)

    return nuisances, model_list, np.sort(fitted_inds.astype(int)), (scores if calculate_scores else None)


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
            = \\frac{1}{n} \\sum_{i=1}^n \\sum_i \\ell(V_i; \\theta(X_i), \\hat{U}_i)

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
    model_nuisance: estimator
        The estimator for fitting the nuisance function. Must implement
        `fit` and `predict` methods that both have signatures::

            model_nuisance.fit(Y, T, X=X, W=W, Z=Z,
                               sample_weight=sample_weight, sample_var=sample_var)
            model_nuisance.predict(Y, T, X=X, W=W, Z=Z,
                                   sample_weight=sample_weight, sample_var=sample_var)

        In fact we allow for the model method signatures to skip any of the keyword arguments
        as long as the class is always called with the omitted keyword argument set to ``None``.
        This can be enforced in child classes by re-implementing the fit and the various effect
        methods. If ``discrete_treatment=True``, then the input ``T`` to both above calls will be the
        one-hot encoding of the original input ``T``, excluding the first column of the one-hot.

        If the estimator also provides a score method with the same arguments as fit, it will be used to
        calculate scores during training.

    model_final: estimator for fitting the response residuals to the features and treatment residuals
        Must implement `fit` and `predict` methods that must have signatures::

            model_final.fit(Y, T, X=X, W=W, Z=Z, nuisances=nuisances,
                            sample_weight=sample_weight, sample_var=sample_var)
            model_final.predict(X=X)

        Predict, should just take the features X and return the constant marginal effect. In fact we allow for the
        model method signatures to skip any of the keyword arguments as long as the class is always called with the
        omitted keyword argument set to ``None``. Moreover, the predict function of the final model can take no
        argument if the class is always called with ``X=None``. This can be enforced in child classes by
        re-implementing the fit and the various effect methods. If ``discrete_treatment=True``, then the input ``T``
        to both above calls will be the one-hot encoding of the original input ``T``, excluding the first column of
        the one-hot.

    discrete_treatment: bool
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    discrete_instrument: bool
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    n_splits: int, cross-validation generator or an iterable
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`cv splitter`
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
            def fit(self, Y, T, W=None):
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
        np.random.seed(123)
        X = np.random.normal(size=(100, 3))
        y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, size=(100,))
        est = _OrthoLearner(ModelNuisance(LinearRegression(), LinearRegression()),
                            ModelFinal(),
                            n_splits=2, discrete_treatment=False, discrete_instrument=False,
                            categories='auto', random_state=None)
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
    >>> est.model_final.model
    LinearRegression(fit_intercept=False)
    >>> est.model_final.model.coef_
    array([1.023649...])

    The following example shows how to do double machine learning with discrete treatments, using
    the _OrthoLearner:

    .. testcode::

        class ModelNuisance:
            def __init__(self, model_t, model_y):
                self._model_t = model_t
                self._model_y = model_y

            def fit(self, Y, T, W=None):
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

        np.random.seed(123)
        W = np.random.normal(size=(100, 3))
        import scipy.special
        from sklearn.linear_model import LogisticRegression
        T = np.random.binomial(1, scipy.special.expit(W[:, 0]))
        y = T + W[:, 0] + np.random.normal(0, 0.01, size=(100,))
        est = _OrthoLearner(ModelNuisance(LogisticRegression(solver='lbfgs'), LinearRegression()),
                            ModelFinal(), n_splits=2, discrete_treatment=True, discrete_instrument=False,
                            categories='auto', random_state=None)
        est.fit(y, T, W=W)

    >>> est.score_
    0.00673015...
    >>> est.const_marginal_effect()
    array([[1.008401...]])
    >>> est.effect()
    array([1.008401...])
    >>> est.score(y, T, W=W)
    0.00310431...
    >>> est.model_final.model.coef_[0]
    1.00840170...

    Attributes
    ----------
    models_nuisance: list of objects of type(model_nuisance)
        A list of instances of the model_nuisance object. Each element corresponds to a crossfitting
        fold and is the model instance that was fitted for that training fold.
    model_final: object of type(model_final)
        An instance of the model_final object that was fitted after calling fit.
    score_ : float or array of floats
        If the model_final has a score method, then `score_` contains the outcome of the final model
        score when evaluated on the fitted nuisances from the first stage. Represents goodness of fit,
        of the final CATE model.
    nuisance_scores_ : tuple of lists of floats or None
        The out-of-sample scores from training each nuisance model
    """

    def __init__(self, model_nuisance, model_final, *,
                 discrete_treatment, discrete_instrument, categories, n_splits, random_state):
        self._model_nuisance = clone(model_nuisance, safe=False)
        self._models_nuisance = None
        self._model_final = clone(model_final, safe=False)
        self._n_splits = n_splits
        self._discrete_treatment = discrete_treatment
        self._discrete_instrument = discrete_instrument
        self._random_state = check_random_state(random_state)
        if discrete_treatment:
            if categories != 'auto':
                categories = [categories]  # OneHotEncoder expects a 2D array with features per column
            self._one_hot_encoder = OneHotEncoder(categories=categories, sparse=False, drop='first')
        super().__init__()

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

    def _strata(self, Y, T, X=None, W=None, Z=None, sample_weight=None, sample_var=None, groups=None):
        if self._discrete_instrument:
            Z = LabelEncoder().fit_transform(np.ravel(Z))

        if self._discrete_treatment:
            enc = LabelEncoder()
            T = enc.fit_transform(np.ravel(T))
            if self._discrete_instrument:
                return T + Z * len(enc.classes_)
            else:
                return T
        elif self._discrete_instrument:
            return Z
        else:
            return None

    @_deprecate_positional("X, W, and Z should be passed by keyword only. In a future release "
                           "we will disallow passing X, W, and Z by position.", ['X', 'W', 'Z'])
    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X=None, W=None, Z=None, *, sample_weight=None, sample_var=None, groups=None, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        X: optional (n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional (n, d_w) matrix or None (Default=None)
            Controls for each sample
        Z: optional (n, d_z) matrix or None (Default=None)
            Instruments for each sample
        sample_weight: optional (n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional (n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the n_splits argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`).

        Returns
        -------
        self : _OrthoLearner instance
        """
        Y, T, X, W, Z, sample_weight, sample_var, groups = check_input_arrays(
            Y, T, X, W, Z, sample_weight, sample_var, groups)
        self._check_input_dims(Y, T, X, W, Z, sample_weight, sample_var, groups)
        nuisances, fitted_inds = self._fit_nuisances(Y, T, X, W, Z, sample_weight=sample_weight, groups=groups)
        self._fit_final(self._subinds_check_none(Y, fitted_inds),
                        self._subinds_check_none(T, fitted_inds),
                        X=self._subinds_check_none(X, fitted_inds),
                        W=self._subinds_check_none(W, fitted_inds),
                        Z=self._subinds_check_none(Z, fitted_inds),
                        nuisances=tuple([self._subinds_check_none(nuis, fitted_inds) for nuis in nuisances]),
                        sample_weight=self._subinds_check_none(sample_weight, fitted_inds),
                        sample_var=self._subinds_check_none(sample_var, fitted_inds))
        return self

    def _fit_nuisances(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        # use a binary array to get stratified split in case of discrete treatment
        stratify = self._discrete_treatment or self._discrete_instrument
        strata = self._strata(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight, groups=groups)
        if strata is None:
            strata = T  # always safe to pass T as second arg to split even if we're not actually stratifying

        if self._discrete_treatment:
            T = self._one_hot_encoder.fit_transform(reshape(T, (-1, 1)))

        if self._discrete_instrument:
            z_enc = LabelEncoder()
            Z = z_enc.fit_transform(Z.ravel())
            z_ohe = OneHotEncoder(categories='auto', sparse=False, drop='first')
            Z = z_ohe.fit_transform(reshape(Z, (-1, 1)))
            self.z_transformer = FunctionTransformer(
                func=_EncoderWrapper(z_ohe, z_enc).encode,
                validate=False)
        else:
            self.z_transformer = None

        if self._n_splits == 1:  # special case, no cross validation
            folds = None
        else:
            splitter = check_cv(self._n_splits, [0], classifier=stratify)
            # if check_cv produced a new KFold or StratifiedKFold object, we need to set shuffle and random_state
            # TODO: ideally, we'd also infer whether we need a GroupKFold (if groups are passed)
            #       however, sklearn doesn't support both stratifying and grouping (see
            #       https://github.com/scikit-learn/scikit-learn/issues/13621), so for now the user needs to supply
            #       their own object that supports grouping if they want to use groups.
            if splitter != self._n_splits and isinstance(splitter, (KFold, StratifiedKFold)):
                splitter.shuffle = True
                splitter.random_state = self._random_state

            all_vars = [var if np.ndim(var) == 2 else var.reshape(-1, 1) for var in [Z, W, X] if var is not None]
            to_split = np.hstack(all_vars) if all_vars else np.ones((T.shape[0], 1))

            if groups is not None:
                if isinstance(splitter, (KFold, StratifiedKFold)):
                    raise TypeError("Groups were passed to fit while using a KFold or StratifiedKFold splitter. "
                                    "Instead you must initialize this object with a splitter that can handle groups.")
                folds = splitter.split(to_split, strata, groups=groups)
            else:
                folds = splitter.split(to_split, strata)

        if self._discrete_treatment:
            self._d_t = shape(T)[1:]
            self.transformer = FunctionTransformer(
                func=_EncoderWrapper(self._one_hot_encoder).encode,
                validate=False)

        nuisances, fitted_models, fitted_inds, scores = _crossfit(self._model_nuisance, folds,
                                                                  Y, T, X=X, W=W, Z=Z,
                                                                  sample_weight=sample_weight, groups=groups)
        self._models_nuisance = fitted_models
        self.nuisance_scores_ = scores
        return nuisances, fitted_inds

    def _fit_final(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        self._model_final.fit(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z,
                                                         nuisances=nuisances, sample_weight=sample_weight,
                                                         sample_var=sample_var))
        self.score_ = None
        if hasattr(self._model_final, 'score'):
            self.score_ = self._model_final.score(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z,
                                                                             nuisances=nuisances,
                                                                             sample_weight=sample_weight,
                                                                             sample_var=sample_var))

    def const_marginal_effect(self, X=None):
        X, = check_input_arrays(X)
        self._check_fitted_dims(X)
        if X is None:
            return self._model_final.predict()
        else:
            return self._model_final.predict(X)
    const_marginal_effect.__doc__ = LinearCateEstimator.const_marginal_effect.__doc__

    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        X, = check_input_arrays(X)
        self._check_fitted_dims(X)
        return super().const_marginal_effect_interval(X, alpha=alpha)
    const_marginal_effect_interval.__doc__ = LinearCateEstimator.const_marginal_effect_interval.__doc__

    def const_marginal_effect_inference(self, X=None):
        X, = check_input_arrays(X)
        self._check_fitted_dims(X)
        return super().const_marginal_effect_inference(X)
    const_marginal_effect_inference.__doc__ = LinearCateEstimator.const_marginal_effect_inference.__doc__

    def effect_interval(self, X=None, *, T0=0, T1=1, alpha=0.1):
        X, T0, T1 = check_input_arrays(X, T0, T1)
        self._check_fitted_dims(X)
        return super().effect_interval(X, T0=T0, T1=T1, alpha=alpha)
    effect_interval.__doc__ = LinearCateEstimator.effect_interval.__doc__

    def effect_inference(self, X=None, *, T0=0, T1=1):
        X, T0, T1 = check_input_arrays(X, T0, T1)
        self._check_fitted_dims(X)
        return super().effect_inference(X, T0=T0, T1=T1)
    effect_inference.__doc__ = LinearCateEstimator.effect_inference.__doc__

    def score(self, Y, T, X=None, W=None, Z=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calls the score function of the model_final and returns the calculated score.
        The model_final model must have a score method.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        X: optional (n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional (n, d_w) matrix or None (Default=None)
            Controls for each sample
        Z: optional (n, d_z) matrix or None (Default=None)
            Instruments for each sample

        Returns
        -------
        score : float or (array of float)
            The score of the final CATE model on the new data. Same type as the return
            type of the model_final.score method.
        """
        if not hasattr(self._model_final, 'score'):
            raise AttributeError("Final model does not have a score method!")
        Y, T, X, W, Z = check_input_arrays(Y, T, X, W, Z)
        self._check_fitted_dims(X)
        self._check_fitted_dims_w_z(W, Z)
        X, T = self._expand_treatments(X, T)
        if self.z_transformer is not None:
            Z = self.z_transformer.transform(Z)
        n_splits = len(self._models_nuisance)
        for idx, mdl in enumerate(self._models_nuisance):
            nuisance_temp = mdl.predict(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z))
            if not isinstance(nuisance_temp, tuple):
                nuisance_temp = (nuisance_temp,)

            if idx == 0:
                nuisances = [np.zeros((n_splits,) + nuis.shape) for nuis in nuisance_temp]

            for it, nuis in enumerate(nuisance_temp):
                nuisances[it][idx] = nuis

        for it in range(len(nuisances)):
            nuisances[it] = np.mean(nuisances[it], axis=0)

        return self._model_final.score(Y, T, **filter_none_kwargs(X=X, W=W, Z=Z, nuisances=nuisances))

    @property
    def model_final(self):
        return self._model_final

    @property
    def models_nuisance(self):
        return self._models_nuisance
