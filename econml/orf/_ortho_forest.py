# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Orthogonal Random Forest.

Orthogonal Random Forest (ORF) is an algorithm for heterogenous treatment effect
estimation. Orthogonal Random Forest combines orthogonalization,
a technique that effectively removes the confounding effect in two-stage estimation,
with generalized random forests, a flexible method for estimating treatment
effect heterogeneity.

This file consists of classes that implement the following variants of the ORF method:

- The :class:`DMLOrthoForest`, a two-forest approach for learning continuous or discrete treatment effects
  using kernel two stage estimation.

- The :class:`DROrthoForest`, a two-forest approach for learning discrete treatment effects
  using kernel two stage estimation.

For more details on these methods, see our paper [Oprescu2019]_.
"""

import abc
import inspect
import numpy as np
import warnings
from joblib import Parallel, delayed
from sklearn import clone
from scipy.stats import norm
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LassoCV, Lasso, LinearRegression, LogisticRegression, \
    LogisticRegressionCV, ElasticNet
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.utils import check_random_state, check_array, column_or_1d
from ..sklearn_extensions.linear_model import WeightedLassoCVWrapper
from .._cate_estimator import BaseCateEstimator, LinearCateEstimator, TreatmentExpansionMixin
from ._causal_tree import CausalTree
from ..inference import NormalInferenceResults
from ..inference._inference import Inference
from ..utilities import (one_hot_encoder, reshape, reshape_Y_T, MAX_RAND_SEED, check_inputs, _deprecate_positional,
                         cross_product, inverse_onehot, check_input_arrays, jacify_featurizer,
                         _RegressionWrapper, deprecated, ndim)
from sklearn.model_selection import check_cv
# TODO: consider working around relying on sklearn implementation details
from ..sklearn_extensions.model_selection import _cross_val_predict


def _build_tree_in_parallel(tree, Y, T, X, W,
                            nuisance_estimator, parameter_estimator, moment_and_mean_gradient_estimator):
    # Create splits of causal tree
    tree.create_splits(Y, T, X, W, nuisance_estimator, parameter_estimator, moment_and_mean_gradient_estimator)
    return tree


def _fit_weighted_pipeline(model_instance, X, y, sample_weight):
    weights_error_msg = (
        "Estimators of type {} do not accept weights. "
        "Consider using the class WeightedModelWrapper from econml.utilities to build a weighted model."
    )
    expected_error_msg = "fit() got an unexpected keyword argument 'sample_weight'"
    if not isinstance(model_instance, Pipeline):
        try:
            model_instance.fit(X, y, sample_weight=sample_weight)
        except TypeError as e:
            if expected_error_msg in str(e):
                # Make sure the correct exception is being rethrown
                raise TypeError(weights_error_msg.format(model_instance.__class__.__name__))
            else:
                raise e
    else:
        try:
            last_step_name = model_instance.steps[-1][0]
            model_instance.fit(X, y, **{"{0}__sample_weight".format(last_step_name): sample_weight})
        except TypeError as e:
            if expected_error_msg in str(e):
                raise TypeError(weights_error_msg.format(model_instance.steps[-1][1].__class__.__name__))
            else:
                raise e


def _cross_fit(model_instance, X, y, split_indices, sample_weight=None, predict_func_name='predict'):
    model_instance1 = clone(model_instance, safe=False)
    model_instance2 = clone(model_instance, safe=False)
    split_1, split_2 = split_indices
    predict_func1 = getattr(model_instance1, predict_func_name)
    predict_func2 = getattr(model_instance2, predict_func_name)
    if sample_weight is None:
        model_instance2.fit(X[split_2], y[split_2])
        pred_1 = predict_func2(X[split_1])
        model_instance1.fit(X[split_1], y[split_1])
        pred_2 = predict_func1(X[split_2])
    else:
        _fit_weighted_pipeline(model_instance2, X[split_2], y[split_2], sample_weight[split_2])
        pred_1 = predict_func2(X[split_1])
        _fit_weighted_pipeline(model_instance1, X[split_1], y[split_1], sample_weight[split_1])
        pred_2 = predict_func1(X[split_2])
    # Must make sure indices are merged correctly
    sorted_split_indices = np.argsort(np.concatenate(split_indices), kind='mergesort')
    return np.concatenate((pred_1, pred_2))[sorted_split_indices]


def _group_predict(X, n_groups, predict_func):
    """ Helper function that predicts using the predict function
    for every input argument that looks like [X; i] for i in range(n_groups). Used in
    DR moments, where we want to predict for each [X; t], for any value of the treatment t.
    Returns an (X.shape[0], n_groups) matrix of predictions for each row of X and each t in range(n_groups).

    Parameters
    ----------
    X : (n, m) array
    n_groups : int
    predict_func : fn

    Returns
    -------
    pred : (n, n_groups) array
    """
    group_pred = np.zeros((X.shape[0], n_groups))
    zero_t = np.zeros((X.shape[0], n_groups))
    for i in range(n_groups):
        zero_t[:, i] = 1
        group_pred[:, i] = predict_func(np.concatenate((X, zero_t), axis=1))
        zero_t[:, i] = 0
    # Convert rows to columns
    return group_pred


def _group_cross_fit(model_instance, X, y, t, split_indices, sample_weight=None, predict_func_name='predict'):
    # Require group assignment t to be one-hot-encoded
    model_instance1 = clone(model_instance, safe=False)
    model_instance2 = clone(model_instance, safe=False)
    split_1, split_2 = split_indices
    n_groups = t.shape[1]
    predict_func1 = getattr(model_instance1, predict_func_name)
    predict_func2 = getattr(model_instance2, predict_func_name)
    Xt = np.concatenate((X, t), axis=1)
    # Get predictions for the 2 splits
    if sample_weight is None:
        model_instance2.fit(Xt[split_2], y[split_2])
        pred_1 = _group_predict(X[split_1], n_groups, predict_func2)
        model_instance1.fit(Xt[split_1], y[split_1])
        pred_2 = _group_predict(X[split_2], n_groups, predict_func1)
    else:
        _fit_weighted_pipeline(model_instance2, Xt[split_2], y[split_2], sample_weight[split_2])
        pred_1 = _group_predict(X[split_1], n_groups, predict_func2)
        _fit_weighted_pipeline(model_instance1, Xt[split_1], y[split_1], sample_weight[split_1])
        pred_2 = _group_predict(X[split_2], n_groups, predict_func1)
    # Must make sure indices are merged correctly
    sorted_split_indices = np.argsort(np.concatenate(split_indices), kind='mergesort')
    return np.concatenate((pred_1, pred_2))[sorted_split_indices]


def _pointwise_effect(X_single, Y, T, X, W, w_nonzero, split_inds, slice_weights_list,
                      second_stage_nuisance_estimator, second_stage_parameter_estimator,
                      moment_and_mean_gradient_estimator, slice_len, n_slices, n_trees,
                      stderr=False):
    """Calculate the effect for a one data point with features X_single.

    Parameters
    ----------
    X_single : array_like, shape (d_x, )
        Feature vector that captures heterogeneity for one sample.

    stderr : bool, default False
        Whether to calculate the covariance matrix via bootstrap-of-little-bags.
    """
    # Crossfitting
    # Compute weighted nuisance estimates
    nuisance_estimates = second_stage_nuisance_estimator(Y, T, X, W, w_nonzero, split_indices=split_inds)
    parameter_estimate = second_stage_parameter_estimator(Y, T, X, nuisance_estimates, w_nonzero, X_single)
    # -------------------------------------------------------------------------------
    # Calculate the covariance matrix corresponding to the BLB inference
    #
    # 1. Calculate the moments and gradient of the training data w.r.t the test point
    # 2. Calculate the weighted moments for each tree slice to create a matrix
    #    U = (n_slices, n_T). The V = (U x grad^{-1}) matrix represents the deviation
    #    in that slice from the overall parameter estimate.
    # 3. Calculate the covariance matrix (V.T x V) / n_slices
    # -------------------------------------------------------------------------------
    if stderr:
        moments, mean_grad = moment_and_mean_gradient_estimator(Y, T, X, W, nuisance_estimates,
                                                                parameter_estimate)
        # Calclulate covariance matrix through BLB
        slice_weighted_moment_one = []
        slice_weighted_moment_two = []
        for slice_weights_one, slice_weights_two in slice_weights_list:
            slice_weighted_moment_one.append(
                np.average(moments[:len(split_inds[0])], axis=0, weights=slice_weights_one)
            )
            slice_weighted_moment_two.append(
                np.average(moments[len(split_inds[0]):], axis=0, weights=slice_weights_two)
            )
        U = np.vstack(slice_weighted_moment_one + slice_weighted_moment_two)
        inverse_grad = np.linalg.inv(mean_grad)
        cov_mat = inverse_grad.T @ U.T @ U @ inverse_grad / (2 * n_slices)
        return parameter_estimate, cov_mat
    return parameter_estimate


class BaseOrthoForest(TreatmentExpansionMixin, LinearCateEstimator):
    """Base class for the :class:`DMLOrthoForest` and :class:`DROrthoForest`."""

    def __init__(self,
                 nuisance_estimator,
                 second_stage_nuisance_estimator,
                 parameter_estimator,
                 second_stage_parameter_estimator,
                 moment_and_mean_gradient_estimator,
                 discrete_treatment=False,
                 treatment_featurizer=None,
                 categories='auto',
                 n_trees=500,
                 min_leaf_size=10, max_depth=10,
                 subsample_ratio=0.25,
                 bootstrap=False,
                 n_jobs=-1,
                 backend='loky',
                 verbose=3,
                 batch_size='auto',
                 random_state=None,
                 allow_missing=False):
        # Estimators
        self.nuisance_estimator = nuisance_estimator
        self.second_stage_nuisance_estimator = second_stage_nuisance_estimator
        self.parameter_estimator = parameter_estimator
        self.second_stage_parameter_estimator = second_stage_parameter_estimator
        self.moment_and_mean_gradient_estimator = moment_and_mean_gradient_estimator
        # OrthoForest parameters
        self.n_trees = n_trees
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.subsample_ratio = subsample_ratio
        self.n_jobs = n_jobs
        self.random_state = check_random_state(random_state)
        # Sub-forests
        self.forest_one_trees = None
        self.forest_two_trees = None
        self.forest_one_subsample_ind = None
        self.forest_two_subsample_ind = None
        # Auxiliary attributes
        self.n_slices = int(np.ceil((self.n_trees)**(1 / 2)))
        self.slice_len = int(np.ceil(self.n_trees / self.n_slices))
        # Fit check
        self.model_is_fitted = False
        self.discrete_treatment = discrete_treatment
        self.treatment_featurizer = treatment_featurizer
        self.backend = backend
        self.verbose = verbose
        self.batch_size = batch_size
        self.categories = categories
        self.allow_missing = allow_missing
        super().__init__()

    def _gen_allowed_missing_vars(self):
        return ['W'] if self.allow_missing else []

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, *, X, W=None, inference='auto'):
        """Build an orthogonal random forest from a training set (Y, T, X, W).

        Parameters
        ----------
        Y : array_like, shape (n, )
            Outcome for the treatment policy.

        T : array_like, shape (n, d_t)
            Treatment policy.

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array_like, shape (n, d_w), optional
            High-dimensional controls.

        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'blb' (or an instance of :class:`BLBInference`)

        Returns
        -------
        self: an instance of self.
        """
        Y, T, X, W = check_inputs(Y, T, X, W, multi_output_Y=False,
                                  force_all_finite_W='allow-nan' if 'W' in self._gen_allowed_missing_vars() else True)
        shuffled_inidces = self.random_state.permutation(X.shape[0])
        n = X.shape[0] // 2
        self.Y_one = Y[shuffled_inidces[:n]]
        self.Y_two = Y[shuffled_inidces[n:]]
        self.T_one = T[shuffled_inidces[:n]]
        self.T_two = T[shuffled_inidces[n:]]
        self.X_one = X[shuffled_inidces[:n]]
        self.X_two = X[shuffled_inidces[n:]]
        if W is not None:
            self.W_one = W[shuffled_inidces[:n]]
            self.W_two = W[shuffled_inidces[n:]]
        else:
            self.W_one = None
            self.W_two = None
        self.forest_one_subsample_ind, self.forest_one_trees = self._fit_forest(Y=self.Y_one,
                                                                                T=self.T_one,
                                                                                X=self.X_one,
                                                                                W=self.W_one)
        self.forest_two_subsample_ind, self.forest_two_trees = self._fit_forest(Y=self.Y_two,
                                                                                T=self.T_two,
                                                                                X=self.X_two,
                                                                                W=self.W_two)
        self.model_is_fitted = True
        return self

    def const_marginal_effect(self, X):
        """Calculate the constant marginal CATE θ(·) conditional on a vector of features X.

        Parameters
        ----------
        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        Returns
        -------
        Theta : matrix , shape (n, d_f_t) where d_f_t is \
            the dimension of the featurized treatment. If treatment_featurizer is None, d_f_t = d_t
            Constant marginal CATE of each treatment for each sample.
        """
        # TODO: Check performance
        return np.asarray(self._predict(X))

    def _predict(self, X, stderr=False):
        if not self.model_is_fitted:
            raise NotFittedError('This {0} instance is not fitted yet.'.format(self.__class__.__name__))
        X = check_array(X)
        results = Parallel(n_jobs=self.n_jobs, backend=self.backend,
                           batch_size=self.batch_size, verbose=self.verbose)(
            delayed(_pointwise_effect)(X_single, *self._pw_effect_inputs(X_single, stderr=stderr),
                                       self.second_stage_nuisance_estimator, self.second_stage_parameter_estimator,
                                       self.moment_and_mean_gradient_estimator, self.slice_len, self.n_slices,
                                       self.n_trees,
                                       stderr=stderr) for X_single in X)
        return results

    def _pw_effect_inputs(self, X_single, stderr=False):
        w1, w2 = self._get_weights(X_single)
        mask_w1 = (w1 != 0)
        mask_w2 = (w2 != 0)
        w1_nonzero = w1[mask_w1]
        w2_nonzero = w2[mask_w2]
        # Must normalize weights
        w_nonzero = np.concatenate((w1_nonzero, w2_nonzero))
        split_inds = (np.arange(len(w1_nonzero)), np.arange(len(w1_nonzero), len(w_nonzero)))
        slice_weights_list = []
        if stderr:
            slices = [
                (it * self.slice_len, min((it + 1) * self.slice_len, self.n_trees)) for it in range(self.n_slices)
            ]
            for slice_it in slices:
                slice_weights_one, slice_weights_two = self._get_weights(X_single, tree_slice=slice_it)
                slice_weights_list.append((slice_weights_one[mask_w1], slice_weights_two[mask_w2]))
        W_none = self.W_one is None
        return np.concatenate((self.Y_one[mask_w1], self.Y_two[mask_w2])), \
            np.concatenate((self.T_one[mask_w1], self.T_two[mask_w2])), \
            np.concatenate((self.X_one[mask_w1], self.X_two[mask_w2])), \
            np.concatenate((self.W_one[mask_w1], self.W_two[mask_w2])
                           ) if not W_none else None, \
            w_nonzero, \
            split_inds, slice_weights_list

    def _get_inference_options(self):
        # Override the CATE inference options
        # Add blb inference to parent's options
        options = super()._get_inference_options()
        options.update(blb=BLBInference)
        options.update(auto=BLBInference)
        return options

    def _fit_forest(self, Y, T, X, W=None):
        # Generate subsample indices
        subsample_ind = self._get_blb_indices(X)
        # Build trees in parallel
        trees = [CausalTree(self.min_leaf_size, self.max_depth, 1000, .4,
                            check_random_state(self.random_state.randint(MAX_RAND_SEED)))
                 for _ in range(len(subsample_ind))]
        return subsample_ind, Parallel(n_jobs=self.n_jobs, backend=self.backend,
                                       batch_size=self.batch_size, verbose=self.verbose, max_nbytes=None)(
            delayed(_build_tree_in_parallel)(tree,
                                             Y[s], T[s], X[s], W[s] if W is not None else None,
                                             self.nuisance_estimator,
                                             self.parameter_estimator,
                                             self.moment_and_mean_gradient_estimator)
            for s, tree in zip(subsample_ind, trees))

    def _get_weights(self, X_single, tree_slice=None):
        """Calculate weights for a single input feature vector over a subset of trees.

        The subset of trees is defined by the `tree_slice` tuple (start, end).
        The (start, end) tuple includes all trees from `start` to `end`-1.
        """
        w1 = np.zeros(self.Y_one.shape[0])
        w2 = np.zeros(self.Y_two.shape[0])
        if tree_slice is None:
            tree_range = range(self.n_trees)
        else:
            tree_range = range(*tree_slice)
        for t in tree_range:
            leaf = self.forest_one_trees[t].find_split(X_single)
            weight_indexes = self.forest_one_subsample_ind[t][leaf.est_sample_inds]
            leaf_weight = 1 / len(leaf.est_sample_inds)
            if self.bootstrap:
                # Bootstraping has repetitions in tree sample
                unique, counts = np.unique(weight_indexes, return_counts=True)
                w1[unique] += leaf_weight * counts
            else:
                w1[weight_indexes] += leaf_weight
        for t in tree_range:
            leaf = self.forest_two_trees[t].find_split(X_single)
            # Similar for `a` weights
            weight_indexes = self.forest_two_subsample_ind[t][leaf.est_sample_inds]
            leaf_weight = 1 / len(leaf.est_sample_inds)
            if self.bootstrap:
                # Bootstraping has repetitions in tree sample
                unique, counts = np.unique(weight_indexes, return_counts=True)
                w2[unique] += leaf_weight * counts
            else:
                w2[weight_indexes] += leaf_weight
        return (w1 / len(tree_range), w2 / len(tree_range))

    def _get_blb_indices(self, X):
        """Get  data indices for every tree under the little bags split."""
        # Define subsample size
        subsample_size = X.shape[0] // 2
        if not self.bootstrap:
            if self.subsample_ratio > 1.0:
                # Safety check
                warnings.warn("The argument 'subsample_ratio' must be between 0.0 and 1.0, " +
                              "however a value of {} was provided. The 'subsample_ratio' will be changed to 1.0.")
                self.subsample_ratio = 1.0
            subsample_size = int(self.subsample_ratio * subsample_size)
        subsample_ind = []
        # Draw points to create little bags
        for it in range(self.n_slices):
            half_sample_inds = self.random_state.choice(
                X.shape[0], X.shape[0] // 2, replace=False)
            for _ in np.arange(it * self.slice_len, min((it + 1) * self.slice_len, self.n_trees)):
                subsample_ind.append(half_sample_inds[self.random_state.choice(
                    X.shape[0] // 2, subsample_size, replace=self.bootstrap)])
        return np.asarray(subsample_ind)


class DMLOrthoForest(BaseOrthoForest):
    """OrthoForest for continuous or discrete treatments using the DML residual on residual moment function.

    A two-forest approach for learning heterogeneous treatment effects using
    kernel two stage estimation.

    Parameters
    ----------
    n_trees : int, default 500
        Number of causal estimators in the forest.

    min_leaf_size : int, default 10
        The minimum number of samples in a leaf.

    max_depth : int, default 10
        The maximum number of splits to be performed when expanding the tree.

    subsample_ratio : float, default 0.7
        The ratio of the total sample to be used when training a causal tree.
        Values greater than 1.0 will be considered equal to 1.0.
        Parameter is ignored when bootstrap=True.

    bootstrap : bool, default False
        Whether to use bootstrap subsampling.

    lambda_reg : float, default 0.01
        The regularization coefficient in the ell_2 penalty imposed on the
        locally linear part of the second stage fit. This is not applied to
        the local intercept, only to the coefficient of the linear component.

    model_T : estimator, default sklearn.linear_model.LassoCV(cv=3)
        The estimator for residualizing the continuous treatment at each leaf.
        Must implement `fit` and `predict` methods.

    model_Y :  estimator, default sklearn.linear_model.LassoCV(cv=3)
        The estimator for residualizing the outcome at each leaf. Must implement
        `fit` and `predict` methods.

    model_T_final : estimator, optional
        The estimator for residualizing the treatment at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to ``None``, it defaults to the
        value of `model_T` parameter.

    model_Y_final : estimator, optional
        The estimator for residualizing the outcome at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to ``None``, it defaults to the
        value of `model_Y` parameter.

    global_residualization : bool, default False
        Whether to perform a prior residualization of Y and T using the model_Y_final and model_T_final
        estimators, or whether to perform locally weighted residualization at each target point.
        Global residualization is computationally less intensive, but could lose some statistical
        power, especially when W is not None.

    global_res_cv : int, cross-validation generator or an iterable, default 2
        The specification of the CV splitter to be used for cross-fitting, when constructing
        the global residuals of Y and T.

    discrete_treatment : bool, default False
        Whether the treatment should be treated as categorical. If True, then the treatment T is
        one-hot-encoded and the model_T is treated as a classifier that must have a predict_proba
        method.

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    categories : array_like or 'auto', default 'auto'
        A list of pre-specified treatment categories. If 'auto' then categories are automatically
        recognized at fit time.

    n_jobs : int, default -1
        The number of jobs to run in parallel for both :meth:`fit` and :meth:`effect`.
        ``-1`` means using all processors. Since OrthoForest methods are
        computationally heavy, it is recommended to set `n_jobs` to -1.

    backend : 'threading' or 'loky', default 'loky'
        What backend should be used for parallelization with the joblib library.

    verbose : int, default 3
        Verbosity level

    batch_size : int or 'auto', default 'auto'
        Batch_size of jobs for parallelism

    random_state : int, RandomState instance, or None, default None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply nuisance_models
        that can handle missing values.

    """

    def __init__(self, *,
                 n_trees=500,
                 min_leaf_size=10, max_depth=10,
                 subsample_ratio=0.7,
                 bootstrap=False,
                 lambda_reg=0.01,
                 model_T='auto',
                 model_Y=WeightedLassoCVWrapper(cv=3),
                 model_T_final=None,
                 model_Y_final=None,
                 global_residualization=False,
                 global_res_cv=2,
                 discrete_treatment=False,
                 treatment_featurizer=None,
                 categories='auto',
                 n_jobs=-1,
                 backend='loky',
                 verbose=3,
                 batch_size='auto',
                 random_state=None,
                 allow_missing=False):
        # Copy and/or define models
        self.lambda_reg = lambda_reg
        if model_T == 'auto':
            if discrete_treatment:
                model_T = LogisticRegressionCV(cv=3)
            else:
                model_T = WeightedLassoCVWrapper(cv=3)
        self.model_T = model_T
        self.model_Y = model_Y
        self.model_T_final = model_T_final
        self.model_Y_final = model_Y_final
        # TODO: ideally the below private attribute logic should be in .fit but is needed in init
        # for nuisance estimator generation for parent class
        # should refactor later
        self._model_T = clone(model_T, safe=False)
        self._model_Y = clone(model_Y, safe=False)
        if self.model_T_final is None:
            self._model_T_final = clone(self.model_T, safe=False)
        else:
            self._model_T_final = clone(self.model_T_final, safe=False)
        if self.model_Y_final is None:
            self._model_Y_final = clone(self.model_Y, safe=False)
        else:
            self._model_Y_final = clone(self.model_Y_final, safe=False)
        if discrete_treatment:
            self._model_T = _RegressionWrapper(self._model_T)
            self._model_T_final = _RegressionWrapper(self._model_T_final)
        self.random_state = check_random_state(random_state)
        self.global_residualization = global_residualization
        self.global_res_cv = global_res_cv
        self.treatment_featurizer = treatment_featurizer
        # Define nuisance estimators
        nuisance_estimator = _DMLOrthoForest_nuisance_estimator_generator(
            self._model_T, self._model_Y, self.random_state, second_stage=False,
            global_residualization=self.global_residualization, discrete_treatment=discrete_treatment)
        second_stage_nuisance_estimator = _DMLOrthoForest_nuisance_estimator_generator(
            self._model_T_final, self._model_Y_final, self.random_state, second_stage=True,
            global_residualization=self.global_residualization, discrete_treatment=discrete_treatment)
        # Define parameter estimators
        parameter_estimator = _DMLOrthoForest_parameter_estimator_func
        second_stage_parameter_estimator = _DMLOrthoForest_second_stage_parameter_estimator_gen(
            self.lambda_reg)
        # Define
        moment_and_mean_gradient_estimator = _DMLOrthoForest_moment_and_mean_gradient_estimator_func

        super().__init__(
            nuisance_estimator,
            second_stage_nuisance_estimator,
            parameter_estimator,
            second_stage_parameter_estimator,
            moment_and_mean_gradient_estimator,
            n_trees=n_trees,
            min_leaf_size=min_leaf_size,
            max_depth=max_depth,
            subsample_ratio=subsample_ratio,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
            discrete_treatment=discrete_treatment,
            treatment_featurizer=treatment_featurizer,
            categories=categories,
            random_state=self.random_state,
            allow_missing=allow_missing)

    def _combine(self, X, W):
        if X is None:
            return W
        if W is None:
            return X
        return np.hstack([X, W])

    # Need to redefine fit here for auto inference to work due to a quirk in how
    # wrap_fit is defined
    def fit(self, Y, T, *, X, W=None, inference='auto'):
        """Build an orthogonal random forest from a training set (Y, T, X, W).

        Parameters
        ----------
        Y : array_like, shape (n, )
            Outcome for the treatment policy.

        T : array_like, shape (n, d_t)
            Treatment policy.

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array_like, shape (n, d_w), optional
            High-dimensional controls.

        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'blb' (or an instance of :class:`BLBInference`)

        Returns
        -------
        self: an instance of self.
        """
        self._set_input_names(Y, T, X, set_flag=True)
        Y, T, X, W = check_inputs(
            Y, T, X, W, force_all_finite_W='allow-nan' if 'W' in self._gen_allowed_missing_vars() else True)
        assert not (self.discrete_treatment and self.treatment_featurizer), "Treatment featurization " \
            "is not supported when treatment is discrete"

        if self.discrete_treatment:
            categories = self.categories
            if categories != 'auto':
                categories = [categories]  # OneHotEncoder expects a 2D array with features per column
            self.transformer = one_hot_encoder(categories=categories, drop='first')
            d_t_in = T.shape[1:]
            T = self.transformer.fit_transform(T.reshape(-1, 1))
            self._d_t = T.shape[1:]
        elif self.treatment_featurizer:
            self._original_treatment_featurizer = clone(self.treatment_featurizer, safe=False)
            self.transformer = jacify_featurizer(self.treatment_featurizer)
            d_t_in = T.shape[1:]
            T = self.transformer.fit_transform(T)
            self._d_t = np.shape(T)[1:]

        if self.global_residualization:
            cv = check_cv(self.global_res_cv, y=T, classifier=self.discrete_treatment)
            cv = list(cv.split(X=X, y=T))
            Y = Y - _cross_val_predict(self._model_Y_final, self._combine(X, W), Y, cv=cv, safe=False).reshape(Y.shape)
            T = T - _cross_val_predict(self._model_T_final, self._combine(X, W), T, cv=cv, safe=False).reshape(T.shape)

        super().fit(Y, T, X=X, W=W, inference=inference)

        # weirdness of wrap_fit. We need to store d_t_in. But because wrap_fit decorates the parent
        # fit, we need to set explicitly d_t_in here after super fit is called.
        if self.discrete_treatment or self.treatment_featurizer:
            self._d_t_in = d_t_in
        return self

    def const_marginal_effect(self, X):
        X = check_array(X)
        # Override to flatten output if T is flat
        effects = super().const_marginal_effect(X=X)
        return effects.reshape((-1,) + self._d_y + self._d_t)
    const_marginal_effect.__doc__ = BaseOrthoForest.const_marginal_effect.__doc__


class _DMLOrthoForest_nuisance_estimator_generator:
    """Generate nuissance estimator given model inputs from the class."""

    def __init__(self, model_T, model_Y, random_state=None, second_stage=True,
                 global_residualization=False, discrete_treatment=False):
        self.model_T = model_T
        self.model_Y = model_Y
        self.random_state = random_state
        self.second_stage = second_stage
        self.global_residualization = global_residualization
        self.discrete_treatment = discrete_treatment

    def __call__(self, Y, T, X, W, sample_weight=None, split_indices=None):
        if self.global_residualization:
            return 0
        if self.discrete_treatment:
            # Check that all discrete treatments are represented
            if len(np.unique(T @ np.arange(1, T.shape[1] + 1))) < T.shape[1] + 1:
                return None
        # Nuissance estimates evaluated with cross-fitting
        this_random_state = check_random_state(self.random_state)
        if (split_indices is None) and self.second_stage:
            if self.discrete_treatment:
                # Define 2-fold iterator
                kfold_it = StratifiedKFold(n_splits=2, shuffle=True, random_state=this_random_state).split(X, T)
                # Check if there is only one example of some class
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        split_indices = list(kfold_it)[0]
                    except Warning as warn:
                        msg = str(warn)
                        if "The least populated class in y has only 1 members" in msg:
                            return None
            else:
                # Define 2-fold iterator
                kfold_it = KFold(n_splits=2, shuffle=True, random_state=this_random_state).split(X)
                split_indices = list(kfold_it)[0]
        if W is not None:
            X_tilde = np.concatenate((X, W), axis=1)
        else:
            X_tilde = X

        try:
            if self.second_stage:
                T_hat = _cross_fit(self.model_T, X_tilde, T, split_indices, sample_weight=sample_weight)
                Y_hat = _cross_fit(self.model_Y, X_tilde, Y, split_indices, sample_weight=sample_weight)
            else:
                # need safe=False when cloning for WeightedModelWrapper
                T_hat = clone(self.model_T, safe=False).fit(X_tilde, T).predict(X_tilde)
                Y_hat = clone(self.model_Y, safe=False).fit(X_tilde, Y).predict(X_tilde)
        except ValueError as exc:
            raise ValueError("The original error: {0}".format(str(exc)) +
                             " This might be caused by too few sample in the tree leafs." +
                             " Try increasing the min_leaf_size.")
        return Y_hat, T_hat


def _DMLOrthoForest_parameter_estimator_func(Y, T, X,
                                             nuisance_estimates,
                                             sample_weight=None):
    """Calculate the parameter of interest for points given by (Y, T) and corresponding nuisance estimates."""
    # Compute residuals
    Y_res, T_res = _DMLOrthoForest_get_conforming_residuals(Y, T, nuisance_estimates)
    # Compute coefficient by OLS on residuals
    param_estimate = LinearRegression(fit_intercept=False).fit(
        T_res, Y_res, sample_weight=sample_weight
    ).coef_
    # Parameter returned by LinearRegression is (d_T, )
    return param_estimate


class _DMLOrthoForest_second_stage_parameter_estimator_gen:
    """
    For the second stage parameter estimation we add a local linear correction. So
    we fit a local linear function as opposed to a local constant function. We also penalize
    the linear part to reduce variance.
    """

    def __init__(self, lambda_reg):
        self.lambda_reg = lambda_reg

    def __call__(self, Y, T, X,
                 nuisance_estimates,
                 sample_weight,
                 X_single):
        """Calculate the parameter of interest for points given by (Y, T) and corresponding nuisance estimates.

        The parameter is calculated around the feature vector given by `X_single`. `X_single` can be used to do
        local corrections on a preliminary parameter estimate.
        """
        # Compute residuals
        Y_res, T_res = _DMLOrthoForest_get_conforming_residuals(Y, T, nuisance_estimates)
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
        XT_res = cross_product(T_res, X_aug)
        # Compute coefficient by OLS on residuals
        if sample_weight is not None:
            weighted_XT_res = sample_weight.reshape(-1, 1) * XT_res
        else:
            weighted_XT_res = XT_res / XT_res.shape[0]
        # ell_2 regularization
        diagonal = np.ones(XT_res.shape[1])
        diagonal[:T_res.shape[1]] = 0
        reg = self.lambda_reg * np.diag(diagonal)
        # Ridge regression estimate
        linear_coef_estimate = np.linalg.lstsq(np.matmul(weighted_XT_res.T, XT_res) + reg,
                                               np.matmul(weighted_XT_res.T, Y_res.reshape(-1, 1)),
                                               rcond=None)[0].flatten()
        X_aug = np.append([1], X_single)
        linear_coef_estimate = linear_coef_estimate.reshape((X_aug.shape[0], -1)).T
        # Parameter returned is of shape (d_T, )
        return np.dot(linear_coef_estimate, X_aug)


def _DMLOrthoForest_moment_and_mean_gradient_estimator_func(Y, T, X, W,
                                                            nuisance_estimates,
                                                            parameter_estimate):
    """Calculate the moments and mean gradient at points given by (Y, T, X, W)."""
    # Return moments and gradients
    # Compute residuals
    Y_res, T_res = _DMLOrthoForest_get_conforming_residuals(Y, T, nuisance_estimates)
    # Compute moments
    # Moments shape is (n, d_T)
    moments = (Y_res - np.matmul(T_res, parameter_estimate)).reshape(-1, 1) * T_res
    # Compute moment gradients
    mean_gradient = - np.matmul(T_res.T, T_res) / T_res.shape[0]
    return moments, mean_gradient


def _DMLOrthoForest_get_conforming_residuals(Y, T, nuisance_estimates):
    if nuisance_estimates == 0:
        return reshape_Y_T(Y, T)
    # returns shape-conforming residuals
    Y_hat, T_hat = reshape_Y_T(*nuisance_estimates)
    Y, T = reshape_Y_T(Y, T)
    Y_res, T_res = Y - Y_hat, T - T_hat
    return Y_res, T_res


class DROrthoForest(BaseOrthoForest):
    """
    OrthoForest for discrete treatments using the doubly robust moment function.

    A two-forest approach for learning heterogeneous treatment effects using
    kernel two stage estimation.

    Parameters
    ----------
    n_trees : int, default 500
        Number of causal estimators in the forest.

    min_leaf_size : int, default 10
        The minimum number of samples in a leaf.

    max_depth : int, default 10
        The maximum number of splits to be performed when expanding the tree.

    subsample_ratio : float, default 0.7
        The ratio of the total sample to be used when training a causal tree.
        Values greater than 1.0 will be considered equal to 1.0.
        Parameter is ignored when bootstrap=True.

    bootstrap : bool, default False
        Whether to use bootstrap subsampling.

    lambda_reg : float, default 0.01
        The regularization coefficient in the ell_2 penalty imposed on the
        locally linear part of the second stage fit. This is not applied to
        the local intercept, only to the coefficient of the linear component.

    propensity_model : estimator, default sklearn.linear_model.LogisticRegression(penalty='l1', \
                                                                                  solver='saga', \
                                                                                  multi_class='auto')
        Model for estimating propensity of treatment at each leaf.
        Will be trained on features and controls (concatenated). Must implement `fit` and `predict_proba` methods.

    model_Y :  estimator, default sklearn.linear_model.LassoCV(cv=3)
        Estimator for learning potential outcomes at each leaf.
        Will be trained on features, controls and one hot encoded treatments (concatenated).
        If different models per treatment arm are desired, see the :class:`.MultiModelWrapper`
        helper class. The model(s) must implement `fit` and `predict` methods.

    propensity_model_final : estimator, optional
        Model for estimating propensity of treatment at at prediction time.
        Will be trained on features and controls (concatenated). Must implement `fit` and `predict_proba` methods.
        If parameter is set to ``None``, it defaults to the value of `propensity_model` parameter.

    model_Y_final : estimator, optional
        Estimator for learning potential outcomes at prediction time.
        Will be trained on features, controls and one hot encoded treatments (concatenated).
        If different models per treatment arm are desired, see the :class:`.MultiModelWrapper`
        helper class. The model(s) must implement `fit` and `predict` methods.
        If parameter is set to ``None``, it defaults to the value of `model_Y` parameter.

    categories: 'auto' or list
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    n_jobs : int, default -1
        The number of jobs to run in parallel for both :meth:`fit` and :meth:`effect`.
        ``-1`` means using all processors. Since OrthoForest methods are
        computationally heavy, it is recommended to set `n_jobs` to -1.

    backend : 'threading' or 'loky', default 'loky'
        What backend should be used for parallelization with the joblib library.

    verbose : int, default 3
        Verbosity level

    batch_size : int or 'auto', default 'auto'
        Batch_size of jobs for parallelism

    random_state : int, RandomState instance, or None, default None
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply nuisance_models
        that can handle missing values.
    """

    def __init__(self, *,
                 n_trees=500,
                 min_leaf_size=10, max_depth=10,
                 subsample_ratio=0.7,
                 bootstrap=False,
                 lambda_reg=0.01,
                 propensity_model=LogisticRegression(penalty='l1', solver='saga',
                                                     multi_class='auto'),  # saga solver supports l1
                 model_Y=WeightedLassoCVWrapper(cv=3),
                 propensity_model_final=None,
                 model_Y_final=None,
                 categories='auto',
                 n_jobs=-1,
                 backend='loky',
                 verbose=3,
                 batch_size='auto',
                 random_state=None,
                 allow_missing=False):
        self.lambda_reg = lambda_reg
        # Copy and/or define models
        self.propensity_model = clone(propensity_model, safe=False)
        self.model_Y = clone(model_Y, safe=False)
        self.propensity_model_final = clone(propensity_model_final, safe=False)
        self.model_Y_final = clone(model_Y_final, safe=False)
        if self.propensity_model_final is None:
            self.propensity_model_final = clone(self.propensity_model, safe=False)
        if self.model_Y_final is None:
            self.model_Y_final = clone(self.model_Y, safe=False)
        self.random_state = check_random_state(random_state)

        nuisance_estimator = DROrthoForest.nuisance_estimator_generator(
            self.propensity_model, self.model_Y, self.random_state, second_stage=False)
        second_stage_nuisance_estimator = DROrthoForest.nuisance_estimator_generator(
            self.propensity_model_final, self.model_Y_final, self.random_state, second_stage=True)
        # Define parameter estimators
        parameter_estimator = DROrthoForest.parameter_estimator_func
        second_stage_parameter_estimator = DROrthoForest.second_stage_parameter_estimator_gen(
            self.lambda_reg)
        # Define moment and mean gradient estimator
        moment_and_mean_gradient_estimator = DROrthoForest.moment_and_mean_gradient_estimator_func
        super().__init__(
            nuisance_estimator,
            second_stage_nuisance_estimator,
            parameter_estimator,
            second_stage_parameter_estimator,
            moment_and_mean_gradient_estimator,
            discrete_treatment=True,
            categories=categories,
            n_trees=n_trees,
            min_leaf_size=min_leaf_size,
            max_depth=max_depth,
            subsample_ratio=subsample_ratio,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
            random_state=self.random_state,
            allow_missing=allow_missing)

    def fit(self, Y, T, *, X, W=None, inference='auto'):
        """Build an orthogonal random forest from a training set (Y, T, X, W).

        Parameters
        ----------
        Y : array_like, shape (n, )
            Outcome for the treatment policy. Must be a vector.

        T : array_like, shape (n, )
            Discrete treatment policy vector. The treatment policy should be a set of consecutive integers
            starting with `0`, where `0` denotes the control group. Otherwise, the treatment policies
            will be ordered lexicographically, with the smallest value being considered the control group.

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array_like, shape (n, d_w), optional
            High-dimensional controls.

        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'blb' (or an instance of :class:`BLBInference`)

        Returns
        -------
        self: an instance of self.
        """
        self._set_input_names(Y, T, X, set_flag=True)
        Y, T, X, W = check_inputs(
            Y, T, X, W, force_all_finite_W='allow-nan' if 'W' in self._gen_allowed_missing_vars() else True)
        # Check that T is shape (n, )
        # Check T is numeric
        T = self._check_treatment(T)
        d_t_in = T.shape[1:]
        # Train label encoder
        categories = self.categories
        if categories != 'auto':
            categories = [categories]  # OneHotEncoder expects a 2D array with features per column
        self.transformer = one_hot_encoder(categories=categories, drop='first')
        d_t_in = T.shape[1:]
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = T.shape[1:]

        # Call `fit` from parent class
        super().fit(Y, T, X=X, W=W, inference=inference)

        # weirdness of wrap_fit. We need to store d_t_in. But because wrap_fit decorates the parent
        # fit, we need to set explicitly d_t_in here after super fit is called.
        self._d_t_in = d_t_in
        return self

    # override only so that we can exclude treatment featurization verbiage in docstring
    def const_marginal_effect(self, X):
        """Calculate the constant marginal CATE θ(·) conditional on a vector of features X.

        Parameters
        ----------
        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        Returns
        -------
        Theta : matrix , shape (n, d_t)
            Constant marginal CATE of each treatment for each sample.
        """
        X = check_array(X)
        # Override to flatten output if T is flat
        effects = super().const_marginal_effect(X=X)
        return effects.reshape((-1,) + self._d_y + self._d_t)

    # override only so that we can exclude treatment featurization verbiage in docstring
    def const_marginal_ate(self, X=None):
        """
        Calculate the average constant marginal CATE :math:`E_X[\\theta(X)]`.

        Parameters
        ----------
        X: (m, d_x) matrix, optional
            Features for each sample.

        Returns
        -------
        theta: (d_y, d_t) matrix
            Average constant marginal CATE of each treatment on each outcome.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return super().const_marginal_ate(X=X)

    @staticmethod
    def nuisance_estimator_generator(propensity_model, model_Y, random_state=None, second_stage=False):
        """Generate nuissance estimator given model inputs from the class."""
        def nuisance_estimator(Y, T, X, W, sample_weight=None, split_indices=None):
            # Expand one-hot encoding to include the zero treatment
            ohe_T = np.hstack([np.all(1 - T, axis=1, keepdims=True), T])
            # Test that T contains all treatments. If not, return None
            T = ohe_T @ np.arange(ohe_T.shape[1])
            if len(np.unique(T)) < ohe_T.shape[1]:
                return None
            # Nuissance estimates evaluated with cross-fitting
            this_random_state = check_random_state(random_state)
            if (split_indices is None) and second_stage:
                # Define 2-fold iterator
                kfold_it = StratifiedKFold(n_splits=2, shuffle=True, random_state=this_random_state).split(X, T)
                # Check if there is only one example of some class
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        split_indices = list(kfold_it)[0]
                    except Warning as warn:
                        msg = str(warn)
                        if "The least populated class in y has only 1 members" in msg:
                            return None
            if W is not None:
                X_tilde = np.concatenate((X, W), axis=1)
            else:
                X_tilde = X
            try:
                if not second_stage:
                    # No need to crossfit for internal nodes
                    propensity_model_clone = clone(propensity_model, safe=False)
                    propensity_model_clone.fit(X_tilde, T)
                    propensities = propensity_model_clone.predict_proba(X_tilde)
                    Y_hat = _group_predict(X_tilde, ohe_T.shape[1],
                                           clone(model_Y, safe=False).fit(np.hstack([X_tilde, ohe_T]), Y).predict)
                else:
                    propensities = _cross_fit(propensity_model, X_tilde, T, split_indices,
                                              sample_weight=sample_weight, predict_func_name='predict_proba')
                    Y_hat = _group_cross_fit(model_Y, X_tilde, Y, ohe_T, split_indices, sample_weight=sample_weight)
            except ValueError as exc:
                raise ValueError("The original error: {0}".format(str(exc)) +
                                 " This might be caused by too few sample in the tree leafs." +
                                 " Try increasing the min_leaf_size.")
            return Y_hat, propensities
        return nuisance_estimator

    @staticmethod
    def parameter_estimator_func(Y, T, X,
                                 nuisance_estimates,
                                 sample_weight=None):
        """Calculate the parameter of interest for points given by (Y, T) and corresponding nuisance estimates."""
        # Compute partial moments
        pointwise_params = DROrthoForest._partial_moments(Y, T, nuisance_estimates)
        param_estimate = np.average(pointwise_params, weights=sample_weight, axis=0)
        # If any of the values in the parameter estimate is nan, return None
        return param_estimate

    @staticmethod
    def second_stage_parameter_estimator_gen(lambda_reg):
        """
        For the second stage parameter estimation we add a local linear correction. So
        we fit a local linear function as opposed to a local constant function. We also penalize
        the linear part to reduce variance.
        """
        def parameter_estimator_func(Y, T, X,
                                     nuisance_estimates,
                                     sample_weight,
                                     X_single):
            """Calculate the parameter of interest for points given by (Y, T) and corresponding nuisance estimates.

            The parameter is calculated around the feature vector given by `X_single`. `X_single` can be used to do
            local corrections on a preliminary parameter estimate.
            """
            # Compute partial moments
            pointwise_params = DROrthoForest._partial_moments(Y, T, nuisance_estimates)
            X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            # Compute coefficient by OLS on residuals
            if sample_weight is not None:
                weighted_X_aug = sample_weight.reshape(-1, 1) * X_aug
            else:
                weighted_X_aug = X_aug / X_aug.shape[0]
            # ell_2 regularization
            diagonal = np.ones(X_aug.shape[1])
            diagonal[0] = 0
            reg = lambda_reg * np.diag(diagonal)
            # Ridge regression estimate
            linear_coef_estimate = np.linalg.lstsq(np.matmul(weighted_X_aug.T, X_aug) + reg,
                                                   np.matmul(weighted_X_aug.T, pointwise_params),
                                                   rcond=None)[0].flatten()
            X_aug = np.append([1], X_single)
            linear_coef_estimate = linear_coef_estimate.reshape((X_aug.shape[0], -1)).T
            # Parameter returned is of shape (d_T, )
            return np.dot(linear_coef_estimate, X_aug)

        return parameter_estimator_func

    @staticmethod
    def moment_and_mean_gradient_estimator_func(Y, T, X, W,
                                                nuisance_estimates,
                                                parameter_estimate):
        """Calculate the moments and mean gradient at points given by (Y, T, X, W)."""
        # Return moments and gradients
        # Compute partial moments
        partial_moments = DROrthoForest._partial_moments(Y, T, nuisance_estimates)
        # Compute moments
        # Moments shape is (n, d_T-1)
        moments = partial_moments - parameter_estimate
        # Compute moment gradients
        n_T = nuisance_estimates[0].shape[1] - 1
        mean_gradient = np.diag(np.ones(n_T) * (-1))
        return moments, mean_gradient

    @staticmethod
    def _partial_moments(Y, T, nuisance_estimates):
        Y_hat, propensities = nuisance_estimates
        partial_moments = np.zeros((len(Y), Y_hat.shape[1] - 1))
        T = T @ np.arange(1, T.shape[1] + 1)
        mask_0 = (T == 0)
        for i in range(0, Y_hat.shape[1] - 1):
            # Need to calculate this in an elegant way for when propensity is 0
            partial_moments[:, i] = Y_hat[:, i + 1] - Y_hat[:, 0]
            mask_i = (T == (i + 1))
            partial_moments[:, i][mask_i] += (Y - Y_hat[:, i + 1])[mask_i] / propensities[:, i + 1][mask_i]
            partial_moments[:, i][mask_0] -= (Y - Y_hat[:, 0])[mask_0] / propensities[:, 0][mask_0]
        return partial_moments

    def _check_treatment(self, T):
        try:
            # This will flatten T
            T = column_or_1d(T)
        except Exception as exc:
            raise ValueError("Expected array of shape ({n}, ), but got {T_shape}".format(n=len(T), T_shape=T.shape))
        # Check that T is numeric
        try:
            T.astype(float)
        except Exception as exc:
            raise ValueError("Expected numeric array but got non-numeric types.")
        return T


class BLBInference(Inference):
    """
    Bootstrap-of-Little-Bags inference implementation for the OrthoForest classes.

    This class can only be used for inference with any estimator derived from :class:`BaseOrthoForest`.

    Parameters
    ----------
    estimator : :class:`BaseOrthoForest`
        Estimator to perform inference on. Must be a child class of :class:`BaseOrthoForest`.
    """

    def fit(self, estimator, *args, **kwargs):
        """
        Fits the inference model.

        This is called after the estimator's fit.
        """
        self._estimator = estimator
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self.d_t = self._d_t[0] if self._d_t else 1
        self.d_y = self._d_y[0] if self._d_y else 1
        # Test whether the input estimator is supported
        if not hasattr(self._estimator, "_predict"):
            raise TypeError("Unsupported estimator of type {}.".format(self._estimator.__class__.__name__) +
                            " Estimators must implement the '_predict' method with the correct signature.")
        return self

    def const_marginal_effect_interval(self, X=None, *, alpha=0.05):
        """ Confidence intervals for the quantities :math:`\\theta(X)` produced
        by the model. Available only when ``inference`` is ``blb`` or ``auto``, when
        calling the fit method.

        Parameters
        ----------
        X: (m, d_x) matrix, optional
            Features for each sample

        alpha:  float in [0, 1], default 0.05
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`const_marginal_effect(X)<const_marginal_effect>` ,\
                             type of :meth:`const_marginal_effect(X)<const_marginal_effect>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        X = check_array(X)
        params_and_cov = self._predict_wrapper(X)
        # Calculate confidence intervals for the parameter (marginal effect)
        lower = alpha / 2
        upper = 1 - alpha / 2
        param_lower = [param + np.apply_along_axis(lambda s: norm.ppf(lower, scale=s), 0, np.sqrt(np.diag(cov_mat)))
                       for (param, cov_mat) in params_and_cov]
        param_upper = [param + np.apply_along_axis(lambda s: norm.ppf(upper, scale=s), 0, np.sqrt(np.diag(cov_mat)))
                       for (param, cov_mat) in params_and_cov]
        param_lower, param_upper = np.asarray(param_lower), np.asarray(param_upper)
        return param_lower.reshape((-1,) + self._estimator._d_y + self._estimator._d_t), \
            param_upper.reshape((-1,) + self._estimator._d_y + self._estimator._d_t)

    def const_marginal_effect_inference(self, X=None):
        """ Inference results for the quantities :math:`\\theta(X)` produced
        by the model. Available only when ``inference`` is ``blb`` or ``auto``, when
        calling the fit method.

        Parameters
        ----------
        X: (m, d_x) matrix, optional
            Features for each sample

        Returns
        -------
        InferenceResults: instance of :class:`~econml.inference.NormalInferenceResults`
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        X = check_array(X)
        params, cov = zip(*(self._predict_wrapper(X)))
        params = np.array(params).reshape((-1,) + self._estimator._d_y + self._estimator._d_t)
        stderr = np.sqrt(np.diagonal(np.array(cov), axis1=1, axis2=2))
        stderr = stderr.reshape((-1,) + self._estimator._d_y + self._estimator._d_t)
        return NormalInferenceResults(d_t=self._estimator._d_t[0] if self._estimator._d_t else 1,
                                      d_y=self._estimator._d_y[0] if self._estimator._d_y else 1,
                                      pred=params, pred_stderr=stderr, mean_pred_stderr=None, inf_type='effect',
                                      feature_names=self._estimator.cate_feature_names(),
                                      output_names=self._estimator.cate_output_names(),
                                      treatment_names=self._estimator.cate_treatment_names())

    def _effect_inference_helper(self, X, T0, T1):
        X, T0, T1 = self._estimator._expand_treatments(*check_input_arrays(X, T0, T1))
        dT = (T1 - T0) if T0.ndim == 2 else (T1 - T0).reshape(-1, 1)
        params_and_cov = self._predict_wrapper(X)
        # Calculate confidence intervals for the effect
        # Calculate the effects
        eff = np.asarray([np.dot(params_and_cov[i][0], dT[i]) for i in range(X.shape[0])])
        # Calculate the standard deviations for the effects
        scales = np.asarray([np.sqrt(dT[i] @ params_and_cov[i][1] @ dT[i]) for i in range(X.shape[0])])
        return eff.reshape((-1,) + self._estimator._d_y), scales.reshape((-1,) + self._estimator._d_y)

    def effect_interval(self, X=None, *, T0=0, T1=1, alpha=0.05):
        """ Confidence intervals for the quantities :math:`\\tau(X, T0, T1)` produced
        by the model. Available only when ``inference`` is ``blb`` or ``auto``, when
        calling the fit method.

        Parameters
        ----------
        X:  (m, d_x) matrix, optional
            Features for each sample
        T0:  (m, d_t) matrix or vector of length m, default 0
            Base treatments for each sample
        T1:  (m, d_t) matrix or vector of length m, default 1
            Target treatments for each sample
        alpha:  float in [0, 1], default 0.05
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`effect(X, T0, T1)<effect>`, type of :meth:`effect(X, T0, T1))<effect>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        eff, scales = self._effect_inference_helper(X, T0, T1)
        lower = alpha / 2
        upper = 1 - alpha / 2
        effect_lower = eff + np.apply_along_axis(lambda s: norm.ppf(lower, scale=s), 0, scales)
        effect_upper = eff + np.apply_along_axis(lambda s: norm.ppf(upper, scale=s), 0, scales)
        return effect_lower, effect_upper

    def effect_inference(self, X=None, *, T0=0, T1=1):
        """ Inference results for the quantities :math:`\\tau(X, T0, T1)` produced
        by the model. Available only when ``inference`` is ``blb`` or ``auto``, when
        calling the fit method.

        Parameters
        ----------
        X:  (m, d_x) matrix, optional
            Features for each sample
        T0:  (m, d_t) matrix or vector of length m, default 0
            Base treatments for each sample
        T1:  (m, d_t) matrix or vector of length m, default 1
            Target treatments for each sample

        Returns
        -------
        InferenceResults: instance of :class:`~econml.inference.NormalInferenceResults`
            The inference results instance contains prediction and prediction standard error and
            can on demand calculate confidence interval, z statistic and p value. It can also output
            a dataframe summary of these inference results.
        """
        eff, scales = self._effect_inference_helper(X, T0, T1)

        # d_t=None here since we measure the effect across all Ts
        return NormalInferenceResults(d_t=None, d_y=self._estimator._d_y[0] if self._estimator._d_y else 1,
                                      pred=eff, pred_stderr=scales, mean_pred_stderr=None, inf_type='effect',
                                      feature_names=self._estimator.cate_feature_names(),
                                      output_names=self._estimator.cate_output_names(),
                                      treatment_names=self._estimator.cate_treatment_names())

    def _marginal_effect_inference_helper(self, T, X):
        if not self._estimator._original_treatment_featurizer:
            return self.const_marginal_effect_inference(X)

        X, T = check_input_arrays(X, T)
        X, T = self._estimator._expand_treatments(X, T, transform=False)

        feat_T = self._estimator.transformer.transform(T)

        jac_T = self._estimator.transformer.jac(T)

        params, cov = zip(*(self._predict_wrapper(X)))
        params = np.array(params)
        cov = np.array(cov)

        eff_einsum_str = 'mf, mtf-> mt'

        # conditionally expand jacobian dimensions to align with einsum str
        jac_index = [slice(None), slice(None), slice(None)]
        if ndim(T) == 1:
            jac_index[1] = None
        if ndim(feat_T) == 1:
            jac_index[2] = None

        # Calculate the effects
        eff = np.einsum(eff_einsum_str, params, jac_T[tuple(jac_index)])

        # Calculate the standard deviations for the effects
        d_t_orig = T.shape[1:]
        d_t_orig = d_t_orig[0] if d_t_orig else 1
        self.d_t_orig = d_t_orig
        output_shape = [X.shape[0]]
        if T.shape[1:]:
            output_shape.append(T.shape[1])
        scales = np.zeros(shape=output_shape)

        for i in range(d_t_orig):
            # conditionally index multiple dimensions depending on shapes of T, Y and feat_T
            jac_index = [slice(None)]
            me_index = [slice(None)]
            if T.shape[1:]:
                jac_index.append(i)
                me_index.append(i)

            if feat_T.shape[1:]:  # if featurized T is not a vector
                jac_index.append(slice(None))
            else:
                jac_index.append(None)

            jac = jac_T[tuple(jac_index)]
            final = np.einsum('mj, mjk, mk -> m', jac, cov, jac)
            scales[tuple(me_index)] = final

        eff = eff.reshape((-1,) + self._d_y + T.shape[1:])
        scales = scales.reshape((-1,) + self._d_y + T.shape[1:])
        return eff, scales

    def marginal_effect_inference(self, T, X):
        if self._estimator._original_treatment_featurizer is None:
            return self.const_marginal_effect_inference(X)

        eff, scales = self._marginal_effect_inference_helper(T, X)

        d_y = self._d_y[0] if self._d_y else 1
        d_t = self._d_t[0] if self._d_t else 1

        return NormalInferenceResults(d_t=self.d_t_orig, d_y=d_y,
                                      pred=eff, pred_stderr=scales, mean_pred_stderr=None, inf_type='effect',
                                      feature_names=self._estimator.cate_feature_names(),
                                      output_names=self._estimator.cate_output_names(),
                                      treatment_names=self._estimator.cate_treatment_names())

    def marginal_effect_interval(self, T, X, *, alpha=0.05):
        return self.marginal_effect_inference(T, X).conf_int(alpha=alpha)

    def _predict_wrapper(self, X=None):
        return self._estimator._predict(X, stderr=True)
