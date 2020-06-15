# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Orthogonal Random Forest.

Orthogonal Random Forest (ORF) is an algorithm for heterogenous treatment effect
estimation. Orthogonal Random Forest combines orthogonalization,
a technique that effectively removes the confounding effect in two-stage estimation,
with generalized random forests, a flexible method for estimating treatment
effect heterogeneity.

This file consists of classes that implement the following variants of the ORF method:

- The :class:`ContinuousTreatmentOrthoForest`, a two-forest approach for learning continuous treatment effects
  using kernel two stage estimation.

- The :class:`DiscreteTreatmentOrthoForest`, a two-forest approach for learning discrete treatment effects
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
from .sklearn_extensions.linear_model import WeightedLassoCVWrapper
from .cate_estimator import BaseCateEstimator, LinearCateEstimator, TreatmentExpansionMixin
from .causal_tree import CausalTree
from .inference import Inference
from .utilities import (reshape, reshape_Y_T, MAX_RAND_SEED, check_inputs,
                        cross_product, inverse_onehot, _EncoderWrapper)


def _build_tree_in_parallel(Y, T, X, W,
                            nuisance_estimator,
                            parameter_estimator,
                            moment_and_mean_gradient_estimator,
                            min_leaf_size, max_depth, random_state):
    tree = CausalTree(nuisance_estimator=nuisance_estimator,
                      parameter_estimator=parameter_estimator,
                      moment_and_mean_gradient_estimator=moment_and_mean_gradient_estimator,
                      min_leaf_size=min_leaf_size,
                      max_depth=max_depth,
                      random_state=random_state)
    # Create splits of causal tree
    tree.create_splits(Y, T, X, W)
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


def _group_cross_fit(model_instance, X, y, t, split_indices, sample_weight=None, predict_func_name='predict'):
    # Require group assignment t to be one-hot-encoded
    model_instance1 = clone(model_instance, safe=False)
    model_instance2 = clone(model_instance, safe=False)
    split_1, split_2 = split_indices
    n_groups = t.shape[1]
    predict_func1 = getattr(model_instance1, predict_func_name)
    predict_func2 = getattr(model_instance2, predict_func_name)
    Xt = np.concatenate((X, t), axis=1)
    # Define an inner function that iterates over group predictions

    def group_predict(split, predict_func):
        group_pred = []
        zero_t = np.zeros((len(split), n_groups - 1))
        for i in range(n_groups):
            pred_i = predict_func(
                np.concatenate((X[split], np.insert(zero_t, i, 1, axis=1)), axis=1)
            )
            group_pred.append(pred_i)
        # Convert rows to columns
        return np.asarray(group_pred).T

    # Get predictions for the 2 splits
    if sample_weight is None:
        model_instance2.fit(Xt[split_2], y[split_2])
        pred_1 = group_predict(split_1, predict_func2)
        model_instance1.fit(Xt[split_1], y[split_1])
        pred_2 = group_predict(split_2, predict_func1)
    else:
        _fit_weighted_pipeline(model_instance2, Xt[split_2], y[split_2], sample_weight[split_2])
        pred_1 = group_predict(split_1, predict_func2)
        _fit_weighted_pipeline(model_instance1, Xt[split_1], y[split_1], sample_weight[split_1])
        pred_2 = group_predict(split_2, predict_func1)
    # Must make sure indices are merged correctly
    sorted_split_indices = np.argsort(np.concatenate(split_indices), kind='mergesort')
    return np.concatenate((pred_1, pred_2))[sorted_split_indices]


class BaseOrthoForest(TreatmentExpansionMixin, LinearCateEstimator):
    """Base class for the :class:`ContinuousTreatmentOrthoForest` and :class:`DiscreteTreatmentOrthoForest`."""

    def __init__(self,
                 nuisance_estimator,
                 second_stage_nuisance_estimator,
                 parameter_estimator,
                 second_stage_parameter_estimator,
                 moment_and_mean_gradient_estimator,
                 n_trees=500,
                 min_leaf_size=10, max_depth=10,
                 subsample_ratio=0.25,
                 bootstrap=False,
                 n_jobs=-1,
                 random_state=None):
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
        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X, W=None, inference=None):
        """Build an orthogonal random forest from a training set (Y, T, X, W).

        Parameters
        ----------
        Y : array-like, shape (n, )
            Outcome for the treatment policy.

        T : array-like, shape (n, d_t)
            Treatment policy.

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array-like, shape (n, d_w) or None (default=None)
            High-dimensional controls.

        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'blb' (or an instance of :class:`BLBInference`)

        Returns
        -------
        self: an instance of self.
        """
        Y, T, X, W = check_inputs(Y, T, X, W, multi_output_Y=False)
        if Y.ndim > 1 and Y.shape[1] > 1:
            raise ValueError(
                "The outcome matrix must be of shape ({0}, ) or ({0}, 1), instead got {1}.".format(len(X), Y.shape))
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
        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        Returns
        -------
        Theta : matrix , shape (n, d_t)
            Constant marginal CATE of each treatment for each sample.
        """
        if not self.model_is_fitted:
            raise NotFittedError('This {0} instance is not fitted yet.'.format(self.__class__.__name__))
        X = check_array(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=3, backend='threading')(
            delayed(self._pointwise_effect)(X_single) for X_single in X)
        # TODO: Check performance
        return np.asarray(results)

    def _get_inference_options(self):
        # Override the CATE inference options
        # Add blb inference to parent's options
        options = super()._get_inference_options()
        options.update(blb=BLBInference)
        return options

    def _pointwise_effect(self, X_single, stderr=False):
        """Calculate the effect for a one data point with features X_single.

        Parameters
        ----------
        X_single : array-like, shape (d_x, )
            Feature vector that captures heterogeneity for one sample.

        stderr : boolean (default=False)
            Whether to calculate the covariance matrix via bootstrap-of-little-bags.
        """
        w1, w2 = self._get_weights(X_single)
        mask_w1 = (w1 != 0)
        mask_w2 = (w2 != 0)
        w1_nonzero = w1[mask_w1]
        w2_nonzero = w2[mask_w2]
        # Must normalize weights
        w_nonzero = np.concatenate((w1_nonzero, w2_nonzero))
        W_none = self.W_one is None
        # Crossfitting
        # Compute weighted nuisance estimates
        nuisance_estimates = self.second_stage_nuisance_estimator(
            np.concatenate((self.Y_one[mask_w1], self.Y_two[mask_w2])),
            np.concatenate((self.T_one[mask_w1], self.T_two[mask_w2])),
            np.concatenate((self.X_one[mask_w1], self.X_two[mask_w2])),
            np.concatenate((self.W_one[mask_w1], self.W_two[mask_w2])) if not W_none else None,
            w_nonzero,
            split_indices=(np.arange(len(w1_nonzero)), np.arange(
                len(w1_nonzero), len(w_nonzero)))
        )
        parameter_estimate = self.second_stage_parameter_estimator(
            np.concatenate((self.Y_one[mask_w1], self.Y_two[mask_w2])),
            np.concatenate((self.T_one[mask_w1], self.T_two[mask_w2])),
            np.concatenate((self.X_one[mask_w1], self.X_two[mask_w2])),
            nuisance_estimates,
            w_nonzero,
            X_single
        )
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
            moments, mean_grad = self.moment_and_mean_gradient_estimator(
                np.concatenate((self.Y_one[mask_w1], self.Y_two[mask_w2])),
                np.concatenate((self.T_one[mask_w1], self.T_two[mask_w2])),
                np.concatenate((self.X_one[mask_w1], self.X_two[mask_w2])),
                np.concatenate((self.W_one[mask_w1], self.W_two[mask_w2])) if self.W_one is not None else None,
                nuisance_estimates,
                parameter_estimate)
            # Calclulate covariance matrix through BLB
            slices = [
                (it * self.slice_len, min((it + 1) * self.slice_len, self.n_trees)) for it in range(self.n_slices)
            ]
            slice_weighted_moment_one = []
            slice_weighted_moment_two = []
            for slice_it in slices:
                slice_weights_one, slice_weights_two = self._get_weights(X_single, tree_slice=slice_it)
                slice_weighted_moment_one.append(
                    np.average(moments[:len(w1_nonzero)], axis=0, weights=slice_weights_one[mask_w1])
                )
                slice_weighted_moment_two.append(
                    np.average(moments[len(w1_nonzero):], axis=0, weights=slice_weights_two[mask_w2])
                )
            U = np.vstack(slice_weighted_moment_one + slice_weighted_moment_two)
            inverse_grad = np.linalg.inv(mean_grad)
            cov_mat = inverse_grad.T @ U.T @ U @ inverse_grad / (2 * self.n_slices)
            return parameter_estimate, cov_mat
        return parameter_estimate

    def _fit_forest(self, Y, T, X, W=None):
        # Generate subsample indices
        subsample_ind = self._get_blb_indices(X)
        # Build trees in parallel
        return subsample_ind, Parallel(n_jobs=self.n_jobs, verbose=3, max_nbytes=None)(
            delayed(_build_tree_in_parallel)(
                Y[s], T[s], X[s], W[s] if W is not None else None,
                self.nuisance_estimator,
                self.parameter_estimator,
                self.moment_and_mean_gradient_estimator,
                self.min_leaf_size, self.max_depth,
                self.random_state.randint(MAX_RAND_SEED)) for s in subsample_ind)

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


class ContinuousTreatmentOrthoForest(BaseOrthoForest):
    """OrthoForest for continuous treatments.

    A two-forest approach for learning heterogeneous treatment effects using
    kernel two stage estimation.

    Parameters
    ----------
    n_trees : integer, optional (default=500)
        Number of causal estimators in the forest.

    min_leaf_size : integer, optional (default=10)
        The minimum number of samples in a leaf.

    max_depth : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree.

    subsample_ratio : float, optional (default=0.7)
        The ratio of the total sample to be used when training a causal tree.
        Values greater than 1.0 will be considered equal to 1.0.
        Parameter is ignored when bootstrap=True.

    bootstrap : boolean, optional (default=False)
        Whether to use bootstrap subsampling.

    lambda_reg : float, optional (default=0.01)
        The regularization coefficient in the ell_2 penalty imposed on the
        locally linear part of the second stage fit. This is not applied to
        the local intercept, only to the coefficient of the linear component.

    model_T : estimator, optional (default=sklearn.linear_model.LassoCV(cv=3))
        The estimator for residualizing the continuous treatment at each leaf.
        Must implement `fit` and `predict` methods.

    model_Y :  estimator, optional (default=sklearn.linear_model.LassoCV(cv=3)
        The estimator for residualizing the outcome at each leaf. Must implement
        `fit` and `predict` methods.

    model_T_final : estimator, optional (default=None)
        The estimator for residualizing the treatment at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to ``None``, it defaults to the
        value of `model_T` parameter.

    model_Y_final : estimator, optional (default=None)
        The estimator for residualizing the outcome at prediction time. Must implement
        `fit` and `predict` methods. If parameter is set to ``None``, it defaults to the
        value of `model_Y` parameter.

    n_jobs : int, optional (default=-1)
        The number of jobs to run in parallel for both :meth:`fit` and :meth:`effect`.
        ``-1`` means using all processors. Since OrthoForest methods are
        computationally heavy, it is recommended to set `n_jobs` to -1.

    random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    """

    def __init__(self,
                 n_trees=500,
                 min_leaf_size=10, max_depth=10,
                 subsample_ratio=0.7,
                 bootstrap=False,
                 lambda_reg=0.01,
                 model_T=WeightedLassoCVWrapper(cv=3),
                 model_Y=WeightedLassoCVWrapper(cv=3),
                 model_T_final=None,
                 model_Y_final=None,
                 n_jobs=-1,
                 random_state=None):
        # Copy and/or define models
        self.lambda_reg = lambda_reg
        self.model_T = model_T
        self.model_Y = model_Y
        self.model_T_final = model_T_final
        self.model_Y_final = model_Y_final
        if self.model_T_final is None:
            self.model_T_final = clone(self.model_T, safe=False)
        if self.model_Y_final is None:
            self.model_Y_final = clone(self.model_Y, safe=False)
        # Define nuisance estimators
        nuisance_estimator = ContinuousTreatmentOrthoForest.nuisance_estimator_generator(
            self.model_T, self.model_Y, random_state, second_stage=False)
        second_stage_nuisance_estimator = ContinuousTreatmentOrthoForest.nuisance_estimator_generator(
            self.model_T_final, self.model_Y_final, random_state, second_stage=True)
        # Define parameter estimators
        parameter_estimator = ContinuousTreatmentOrthoForest.parameter_estimator_func
        second_stage_parameter_estimator =\
            ContinuousTreatmentOrthoForest.second_stage_parameter_estimator_gen(self.lambda_reg)
        # Define
        moment_and_mean_gradient_estimator = ContinuousTreatmentOrthoForest.moment_and_mean_gradient_estimator_func
        super(ContinuousTreatmentOrthoForest, self).__init__(
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
            random_state=random_state)

    def const_marginal_effect(self, X):
        # Override to flatten output if T is flat
        effects = super(ContinuousTreatmentOrthoForest, self).const_marginal_effect(X=X)
        if not self._d_t:
            # T is flat
            return effects.flatten()
        return effects
    const_marginal_effect.__doc__ = BaseOrthoForest.const_marginal_effect.__doc__

    @staticmethod
    def nuisance_estimator_generator(model_T, model_Y, random_state=None, second_stage=True):
        """Generate nuissance estimator given model inputs from the class."""
        def nuisance_estimator(Y, T, X, W, sample_weight=None, split_indices=None):
            # Nuissance estimates evaluated with cross-fitting
            this_random_state = check_random_state(random_state)
            if split_indices is None:
                # Define 2-fold iterator
                kfold_it = KFold(n_splits=2, shuffle=True, random_state=this_random_state).split(X)
                split_indices = list(kfold_it)[0]
            if W is not None:
                X_tilde = np.concatenate((X, W), axis=1)
            else:
                X_tilde = X

            try:
                if second_stage:
                    T_hat = _cross_fit(model_T, X_tilde, T, split_indices, sample_weight=sample_weight)
                    Y_hat = _cross_fit(model_Y, X_tilde, Y, split_indices, sample_weight=sample_weight)
                else:
                    # need safe=False when cloning for WeightedModelWrapper
                    T_hat = clone(model_T, safe=False).fit(X_tilde, T).predict(X_tilde)
                    Y_hat = clone(model_Y, safe=False).fit(X_tilde, Y).predict(X_tilde)
            except ValueError as exc:
                raise ValueError("The original error: {0}".format(str(exc)) +
                                 " This might be caused by too few sample in the tree leafs." +
                                 " Try increasing the min_leaf_size.")
            return Y_hat, T_hat

        return nuisance_estimator

    @staticmethod
    def parameter_estimator_func(Y, T, X,
                                 nuisance_estimates,
                                 sample_weight=None):
        """Calculate the parameter of interest for points given by (Y, T) and corresponding nuisance estimates."""
        # Compute residuals
        Y_hat, T_hat = nuisance_estimates
        Y_res, T_res = reshape_Y_T(Y - Y_hat, T - T_hat)
        # Compute coefficient by OLS on residuals
        param_estimate = LinearRegression(fit_intercept=False).fit(
            T_res, Y_res, sample_weight=sample_weight
        ).coef_
        # Parameter returned by LinearRegression is (d_T, )
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
            # Compute residuals
            Y_hat, T_hat = nuisance_estimates
            Y_res, T_res = reshape_Y_T(Y - Y_hat, T - T_hat)
            X_aug = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
            XT_res = cross_product(T_res, X_aug)
            # Compute coefficient by OLS on residuals
            if sample_weight is not None:
                weighted_XT_res = sample_weight.reshape(-1, 1) * XT_res
            else:
                weighted_XT_res = XT_res / XT_res.shape[0]
            # ell_2 regularization
            diagonal = np.ones(XT_res.shape[1])
            diagonal[:T_res.shape[1]] = 0
            reg = lambda_reg * np.diag(diagonal)
            # Ridge regression estimate
            linear_coef_estimate = np.linalg.lstsq(np.matmul(weighted_XT_res.T, XT_res) + reg,
                                                   np.matmul(weighted_XT_res.T, Y_res.reshape(-1, 1)),
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
        # Compute residuals
        Y_hat, T_hat = nuisance_estimates
        Y_res, T_res = reshape_Y_T(Y - Y_hat, T - T_hat)
        # Compute moments
        # Moments shape is (n, d_T)
        moments = (Y_res - np.matmul(T_res, parameter_estimate)).reshape(-1, 1) * T_res
        # Compute moment gradients
        mean_gradient = - np.matmul(T_res.T, T_res) / T_res.shape[0]
        return moments, mean_gradient


class DiscreteTreatmentOrthoForest(BaseOrthoForest):
    """
    OrthoForest for discrete treatments.

    A two-forest approach for learning heterogeneous treatment effects using
    kernel two stage estimation.

    Parameters
    ----------
    n_trees : integer, optional (default=500)
        Number of causal estimators in the forest.

    min_leaf_size : integer, optional (default=10)
        The minimum number of samples in a leaf.

    max_depth : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree.

    subsample_ratio : float, optional (default=0.7)
        The ratio of the total sample to be used when training a causal tree.
        Values greater than 1.0 will be considered equal to 1.0.
        Parameter is ignored when bootstrap=True.

    bootstrap : boolean, optional (default=False)
        Whether to use bootstrap subsampling.

    lambda_reg : float, optional (default=0.01)
        The regularization coefficient in the ell_2 penalty imposed on the
        locally linear part of the second stage fit. This is not applied to
        the local intercept, only to the coefficient of the linear component.

    propensity_model : estimator, optional (default=sklearn.linear_model.LogisticRegression(penalty='l1',\
                                                                                             solver='saga',\
                                                                                             multi_class='auto'))
        Model for estimating propensity of treatment at each leaf.
        Will be trained on features and controls (concatenated). Must implement `fit` and `predict_proba` methods.

    model_Y :  estimator, optional (default=sklearn.linear_model.LassoCV(cv=3))
        Estimator for learning potential outcomes at each leaf.
        Will be trained on features, controls and one hot encoded treatments (concatenated).
        If different models per treatment arm are desired, see the :class:`.MultiModelWrapper`
        helper class. The model(s) must implement `fit` and `predict` methods.

    propensity_model_final : estimator, optional (default=None)
        Model for estimating propensity of treatment at at prediction time.
        Will be trained on features and controls (concatenated). Must implement `fit` and `predict_proba` methods.
        If parameter is set to ``None``, it defaults to the value of `propensity_model` parameter.

    model_Y_final : estimator, optional (default=None)
        Estimator for learning potential outcomes at prediction time.
        Will be trained on features, controls and one hot encoded treatments (concatenated).
        If different models per treatment arm are desired, see the :class:`.MultiModelWrapper`
        helper class. The model(s) must implement `fit` and `predict` methods.
        If parameter is set to ``None``, it defaults to the value of `model_Y` parameter.

    categories: 'auto' or list
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    n_jobs : int, optional (default=-1)
        The number of jobs to run in parallel for both :meth:`fit` and :meth:`effect`.
        ``-1`` means using all processors. Since OrthoForest methods are
        computationally heavy, it is recommended to set `n_jobs` to -1.

    random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.


    """

    def __init__(self,
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
                 random_state=None):
        # Copy and/or define models
        self.propensity_model = clone(propensity_model, safe=False)
        self.model_Y = clone(model_Y, safe=False)
        self.propensity_model_final = clone(propensity_model_final, safe=False)
        self.model_Y_final = clone(model_Y_final, safe=False)
        if self.propensity_model_final is None:
            self.propensity_model_final = clone(self.propensity_model, safe=False)
        if self.model_Y_final is None:
            self.model_Y_final = clone(self.model_Y, safe=False)

        # Nuisance estimators shall be defined during fitting because they need to know the number of distinct
        # treatments
        nuisance_estimator = None
        second_stage_nuisance_estimator = None
        # Define parameter estimators
        parameter_estimator = DiscreteTreatmentOrthoForest.parameter_estimator_func
        second_stage_parameter_estimator =\
            DiscreteTreatmentOrthoForest.second_stage_parameter_estimator_gen(lambda_reg)
        # Define moment and mean gradient estimator
        moment_and_mean_gradient_estimator =\
            DiscreteTreatmentOrthoForest.moment_and_mean_gradient_estimator_func
        if categories != 'auto':
            categories = [categories]  # OneHotEncoder expects a 2D array with features per column
        self._one_hot_encoder = OneHotEncoder(categories=categories, sparse=False, drop='first')

        super(DiscreteTreatmentOrthoForest, self).__init__(
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
            random_state=random_state)

    def fit(self, Y, T, X, W=None, inference=None):
        """Build an orthogonal random forest from a training set (Y, T, X, W).

        Parameters
        ----------
        Y : array-like, shape (n, )
            Outcome for the treatment policy. Must be a vector.

        T : array-like, shape (n, )
            Discrete treatment policy vector. The treatment policy should be a set of consecutive integers
            starting with `0`, where `0` denotes the control group. Otherwise, the treatment policies
            will be ordered lexicographically, with the smallest value being considered the control group.

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array-like, shape (n, d_w) or None (default=None)
            High-dimensional controls.

        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`)

        Returns
        -------
        self: an instance of self.
        """
        # Check that T is shape (n, )
        # Check T is numeric
        T = self._check_treatment(T)
        # Train label encoder
        T = self._one_hot_encoder.fit_transform(T.reshape(-1, 1))
        # Define number of classes
        self.n_T = T.shape[1] + 1  # add one to compensate for dropped first
        T = inverse_onehot(T)
        self.nuisance_estimator = DiscreteTreatmentOrthoForest.nuisance_estimator_generator(
            self.propensity_model, self.model_Y, self.n_T, self.random_state, second_stage=False)
        self.second_stage_nuisance_estimator = DiscreteTreatmentOrthoForest.nuisance_estimator_generator(
            self.propensity_model_final, self.model_Y_final, self.n_T, self.random_state, second_stage=True)
        self.transformer = FunctionTransformer(
            func=_EncoderWrapper(self._one_hot_encoder).encode,
            validate=False)
        # Call `fit` from parent class
        return super().fit(Y, T, X, W=W, inference=inference)

    @staticmethod
    def nuisance_estimator_generator(propensity_model, model_Y, n_T, random_state=None, second_stage=False):
        """Generate nuissance estimator given model inputs from the class."""
        def nuisance_estimator(Y, T, X, W, sample_weight=None, split_indices=None):
            # Test that T contains all treatments. If not, return None
            ohe_T = OneHotEncoder(sparse=False, categories='auto').fit_transform(T.reshape(-1, 1))
            if ohe_T.shape[1] < n_T:
                return None
            # Nuissance estimates evaluated with cross-fitting
            this_random_state = check_random_state(random_state)
            if split_indices is None:
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
        pointwise_params = DiscreteTreatmentOrthoForest._partial_moments(Y, T, nuisance_estimates)
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
            pointwise_params = DiscreteTreatmentOrthoForest._partial_moments(Y, T, nuisance_estimates)
            X_aug = PolynomialFeatures(degree=1, include_bias=True).fit_transform(X)
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
        partial_moments = DiscreteTreatmentOrthoForest._partial_moments(Y, T, nuisance_estimates)
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
        # Test whether the input estimator is supported
        if not hasattr(self._estimator, "_pointwise_effect"):
            raise TypeError("Unsupported estimator of type {}.".format(self._estimator.__class__.__name__) +
                            " Estimators must implement the '_pointwise_effect' method with the correct signature.")
        # Test expansion of treatment
        # If expanded treatments are a vector, flatten const_marginal_effect_interval
        _, T0, _ = self._estimator._expand_treatments(None, 0, 1)
        self._T_vec = (T0.ndim == 1)
        return self

    def const_marginal_effect_interval(self, X=None, *, alpha=0.1):
        """ Confidence intervals for the quantities :math:`\\theta(X)` produced
        by the model. Available only when ``inference`` is ``blb``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix or None (Default=None)
            Features for each sample

        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`const_marginal_effect(X)<const_marginal_effect>` ,\
                             type of :meth:`const_marginal_effect(X)<const_marginal_effect>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        params_and_cov = self._predict_wrapper(X)
        # Calculate confidence intervals for the parameter (marginal effect)
        lower = alpha / 2
        upper = 1 - alpha / 2
        param_lower = [param + np.apply_along_axis(lambda s: norm.ppf(lower, scale=s), 0, np.sqrt(np.diag(cov_mat)))
                       for (param, cov_mat) in params_and_cov]
        param_upper = [param + np.apply_along_axis(lambda s: norm.ppf(upper, scale=s), 0, np.sqrt(np.diag(cov_mat)))
                       for (param, cov_mat) in params_and_cov]
        param_lower, param_upper = np.asarray(param_lower), np.asarray(param_upper)
        if self._T_vec:
            # If T is a vector, preserve shape of the effect interval
            return param_lower.flatten(), param_upper.flatten()
        return param_lower, param_upper

    def effect_interval(self, X=None, *, T0=0, T1=1, alpha=0.1):
        """ Confidence intervals for the quantities :math:`\\tau(X, T0, T1)` produced
        by the model. Available only when ``inference`` is ``blb``, when
        calling the fit method.

        Parameters
        ----------
        X: optional (m, d_x) matrix
            Features for each sample

        T0: optional (m, d_t) matrix or vector of length m (Default=0)
            Base treatments for each sample

        T1: optional (m, d_t) matrix or vector of length m (Default=1)
            Target treatments for each sample

        alpha: optional float in [0, 1] (Default=0.1)
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        lower, upper : tuple(type of :meth:`effect(X, T0, T1)<effect>`, type of :meth:`effect(X, T0, T1))<effect>` )
            The lower and the upper bounds of the confidence interval for each quantity.
        """
        X, T0, T1 = self._estimator._expand_treatments(X, T0, T1)
        dT = (T1 - T0) if T0.ndim == 2 else (T1 - T0).reshape(-1, 1)
        params_and_cov = self._predict_wrapper(X)
        # Calculate confidence intervals for the effect
        lower = alpha / 2
        upper = 1 - alpha / 2
        # Calculate the effects
        eff = np.asarray([np.dot(params_and_cov[i][0], dT[i]) for i in range(X.shape[0])])
        # Calculate the standard deviations for the effects
        scales = [np.sqrt(dT[i] @ params_and_cov[i][1] @ dT[i]) for i in range(X.shape[0])]
        effect_lower = eff + np.apply_along_axis(lambda s: norm.ppf(lower, scale=s), 0, scales)
        effect_upper = eff + np.apply_along_axis(lambda s: norm.ppf(upper, scale=s), 0, scales)
        return effect_lower, effect_upper

    def _predict_wrapper(self, X=None):
        if not self._estimator.model_is_fitted:
            raise NotFittedError('This {0} instance is not fitted yet.'.format(self._estimator.__class__.__name__))
        X = check_array(X)
        # Get a list of (parameter, covariance matrix) pairs
        params_and_cov = Parallel(n_jobs=self._estimator.n_jobs, verbose=3, backend='threading')(
            delayed(self._estimator._pointwise_effect)(X_single, stderr=True) for X_single in X)
        return params_and_cov
