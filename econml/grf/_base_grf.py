# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code contains snippets of code from
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/_forest.py
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

import numbers
from warnings import catch_warnings, simplefilter, warn
from abc import ABCMeta, abstractmethod
import numpy as np
import threading
from ._ensemble import BaseEnsemble, _partition_estimators
from ..utilities import check_inputs, cross_product
from ..tree._tree import DTYPE, DOUBLE
from ._base_grftree import GRFTree
from joblib import Parallel, delayed
from scipy.sparse import hstack as sparse_hstack
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils import check_X_y
import scipy.stats
from scipy.special import erfc

__all__ = ["BaseGRF"]

MAX_INT = np.iinfo(np.int32).max


def _get_n_samples_subsample(n_samples, max_samples):
    """
    Get the number of samples in a sub-sample without replacement.
    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    max_samples : int or float
        The maximum number of samples to draw from the total available:
            - if float, this indicates a fraction of the total and should be
              the interval `(0, 1)`;
            - if int, this indicates the exact number of samples;
            - if None, this indicates the total number of samples.
    Returns
    -------
    n_samples_subsample : int
        The total number of samples to draw for the subsample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples <= 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _accumulate_prediction(predict, X, out, lock, *args, **kwargs):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, *args, check_input=False, **kwargs)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


def _accumulate_prediction_var(predict, X, out, lock, *args, **kwargs):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    Accumulates the mean covariance of a tree prediction. predict is assumed to
    return an array of (n_samples, d) or a tuple of arrays. This method accumulates in the placeholder
    out[0] the (n_samples, d, d) covariance of the columns of the prediction across
    the trees and for each sample (or a tuple of covariances to be stored in each element
    of the list out).
    """
    prediction = predict(X, *args, check_input=False, **kwargs)
    with lock:
        if len(out) == 1:
            out[0] += np.einsum('ijk,ikm->ijm',
                                prediction.reshape(prediction.shape + (1,)),
                                prediction.reshape((-1, 1) + prediction.shape[1:]))
        else:
            for i in range(len(out)):
                pred_i = prediction[i]
                out[i] += np.einsum('ijk,ikm->ijm',
                                    pred_i.reshape(pred_i.shape + (1,)),
                                    pred_i.reshape((-1, 1) + pred_i.shape[1:]))


def _accumulate_prediction_and_var(predict, X, out, out_var, lock, *args, **kwargs):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    Combines `_accumulate_prediction` and `_accumulate_prediction_var` in a single
    parallel run, so that out will contain the mean of the predictions across trees
    and out_var the covariance.
    """
    prediction = predict(X, *args, check_input=False, **kwargs)
    with lock:
        if len(out) == 1:
            out[0] += prediction
            out_var[0] += np.einsum('ijk,ikm->ijm',
                                    prediction.reshape(prediction.shape + (1,)),
                                    prediction.reshape((-1, 1) + prediction.shape[1:]))
        else:
            for i in range(len(out)):
                pred_i = prediction[i]
                out[i] += prediction
                out_var[i] += np.einsum('ijk,ikm->ijm',
                                        pred_i.reshape(pred_i.shape + (1,)),
                                        pred_i.reshape((-1, 1) + pred_i.shape[1:]))


# =============================================================================
# Base Generalized Random Forest
# =============================================================================


class BaseGRF(BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for Genearlized Random Forests for solving linear moment equations of
    the form::

        E[J * theta(x) - A | X = x] = 0

    where J is an (d, d) random matrix, A is an (d, 1) random vector and theta(x)
    is a local parameter to be estimated, which might contain both relevant and
    nuisance parameters.

    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 n_estimators=100, *,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 inference=True,
                 fit_intercept=True,
                 subforest_size=4,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=GRFTree(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "min_var_leaf", "min_var_leaf_on_val",
                              "max_features", "min_impurity_decrease", "honest",
                              "min_balancedness_tol",
                              "random_state"))

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_var_fraction_leaf = min_var_fraction_leaf
        self.min_var_leaf_on_val = min_var_leaf_on_val
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.inference = inference
        self.fit_intercept = fit_intercept
        self.subforest_size = subforest_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples

    def _get_alpha_and_pointJ(self, X, T, y, **kwargs):
        """ This function must be implemented by child class and given input variables
        X, T, y and any auxiliary variables passed as keyword only, should be calculating
        the point-wise random vector A and the point-wise jacobian random variable J of
        the linear moment equation for every sample in the input samples.

        Returns
        -------
        A : array of shape (n_samples, n_outputs)
            The A part of the moment equation for each sample
        J : array of shape (n_samples, n_outputs * n_outputs)
            The J matrix part of the moment equation, flattened in Fortran-contiguous format.
        """
        pass

    def _get_n_outputs_decomposition(self, X, T, y, **kwargs):
        """ This function must be implemented by child class and given input variables
        X, T, y and any auxiliary variables passed as keyword only, should return a tuple
        (n_outputs, n_relevant_outputs), which determines how many parameters is the moment
        estimating and what prefix of these parameters are the relevant ones that we care about.

        Returns
        -------
        n_outputs : int
            The number of parameters we are estimating
        n_relevant_outputs : int
            The length of the prefix of parameters that we care about (remainder are nuisance)
        """
        pass

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.
        """
        X = self._validate_X_predict(X)
        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend="threading")(
            delayed(tree.apply)(X, check_input=False)
            for tree in self.estimators_)

        return np.array(results).T

    def decision_path(self, X):
        """
        Return the decision path in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements indicates
            that the samples goes through the nodes. The matrix is of CSR
            format.
        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.
        """
        X = self._validate_X_predict(X)
        indicators = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='threading')(
            delayed(tree.decision_path)(X, check_input=False)
            for tree in self.estimators_)

        n_nodes = [0]
        n_nodes.extend([i.shape[1] for i in indicators])
        n_nodes_ptr = np.array(n_nodes).cumsum()

        return sparse_hstack(indicators).tocsr(), n_nodes_ptr

    def fit(self, X, T, y, *, sample_weight=None, sample_var=None, **kwargs):
        """
        Build a forest of trees from the training set (X, T, y) and any other auxiliary variables.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float64``.
        T : array-like of shape (n_samples, n_treatments)
            The treatment vector for each sample
        y : array-like of shape (n_samples,) or (n_samples, n_outcomes)
            The outcome values for each sample.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        sample_var : array-like of shape (n_samples,), default=None
            Currently ignored and raises a warning. Added for API consistency and for raising relevant errors.
        **kwargs : dictionary of array-like items of shape (n_samples, d_var)
            Auxiliary random variables that go into the moment function (e.g. instrument, censoring etc)
            Any of these variables will be passed on as is to the `get_pointJ` and
            `get_alpha` method of the children classes.

        Returns
        -------
        self : object
        """

        if sample_var is not None:
            warn("`sample_var` is currently being ingored by the estimator. All samples are "
                 "considered to have equal variance of y.")

        y, T, X, _ = check_inputs(y, T, X, W=None, multi_output_T=True, multi_output_Y=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_y_ = y.shape[1]

        T = np.atleast_1d(T)
        if T.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            T = np.reshape(T, (-1, 1))

        alpha, pointJ = self._get_alpha_and_pointJ(X, T, y, **kwargs)
        self.n_outputs_, self.n_relevant_outputs_ = self._get_n_outputs_decomposition(X, T, y, **kwargs)
        yaug = np.hstack([y, alpha, pointJ])

        if getattr(yaug, "dtype", None) != DOUBLE or not yaug.flags.contiguous:
            yaug = np.ascontiguousarray(yaug, dtype=DOUBLE)

        if getattr(X, "dtype", None) != DTYPE:
            X = X.astype(DTYPE)

        # Get subsample sample size
        n_samples_subsample = _get_n_samples_subsample(
            n_samples=n_samples,
            max_samples=self.max_samples
        )

        # Converting `min_var_fraction_leaf` to an absolute `min_var_leaf` that the GRFTree can handle
        if self.min_var_fraction_leaf is None:
            self.min_var_leaf = None
        elif (not isinstance(self.min_var_fraction_leaf, numbers.Real)) or (not (0 < self.min_var_fraction_leaf <= 1)):
            msg = "`min_var_fraction_leaf` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(self.min_var_fraction_leaf))
        else:
            # We calculate the min eigenvalue proxy that each criterion is considering
            # on the overall mean jacobian, to determine the absolute level of `min_var_leaf`
            jac = np.mean(pointJ, axis=0).reshape((self.n_outputs_, self.n_outputs_))
            min_var = np.min(np.abs(np.diag(jac)))
            if self.criterion == 'mse':
                for i in range(self.n_outputs_):
                    for j in range(self.n_outputs_):
                        if j != i:
                            det = np.sqrt(np.abs(jac[i, i] * jac[j, j] - jac[i, j] * jac[j, i]))
                            if det < min_var:
                                min_var = det
            self.min_var_leaf = min_var * self.min_var_fraction_leaf

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)
        # We re-initialize the subsample_random_seed_ only if we are not in warm_start mode or
        # if this is the first `fit` call of the warm start mode.
        if (not self.warm_start) or (not hasattr(self, 'subsample_random_seed_')):
            self.subsample_random_seed_ = random_state.randint(MAX_INT)
        else:
            random_state.randint(MAX_INT)  # just advance random_state
        subsample_random_state = check_random_state(self.subsample_random_seed_)

        if (self.warm_start and hasattr(self, 'inference_') and (self.inference != self.inference_)):
            raise ValueError("Parameter inference cannot be altered in between `fit` "
                             "calls when `warm_start=True`.")
        self.inference_ = self.inference

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.slices_ = []
            # the below are needed to replicate randomness of subsampling when warm_start=True
            self.slices_n_samples_ = []
            self.slices_n_samples_subsample_ = []
            self.n_samples_ = []
            self.n_samples_subsample_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.inference:
                if not isinstance(self.subforest_size, numbers.Integral):
                    raise ValueError("Parameter `subforest_size` must be "
                                     "an integer but got value {}.".format(self.subforest_size))
                if self.subforest_size < 2:
                    raise ValueError("Parameter `subforest_size` must be at least 2 if `inference=True`, "
                                     "but got value {}".format(self.subforest_size))
                if not (n_more_estimators % self.subforest_size == 0):
                    raise ValueError("The number of estimators to be constructed must be divisible "
                                     "the `subforest_size` parameter. Asked to build `n_estimators={}` "
                                     "with `subforest_size={}`.".format(n_more_estimators, self.subforest_size))
                if n_samples_subsample > n_samples // 2:
                    if isinstance(self.max_samples, numbers.Integral):
                        raise ValueError("Parameter `max_samples` must be in [1, n_samples // 2], "
                                         "if `inference=True`. "
                                         "Got values n_samples={}, max_samples={}".format(n_samples, self.max_samples))
                    else:
                        raise ValueError("Parameter `max_samples` must be in (0, .5], if `inference=True`. "
                                         "Got value {}".format(self.max_samples))

            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state).init()
                     for i in range(n_more_estimators)]

            if self.inference:
                if self.warm_start:
                    # Advancing subsample_random_state. Assumes each prior fit call has the same number of
                    # samples at fit time. If not then this would not exactly replicate a single batch execution,
                    # but would still advance randomness enough so that tree subsamples will be different.
                    for sl, n_, ns_ in zip(self.slices_, self.slices_n_samples_, self.slices_n_samples_subsample_):
                        subsample_random_state.choice(n_, n_ // 2, replace=False)
                        for _ in range(len(sl)):
                            subsample_random_state.choice(n_ // 2, ns_, replace=False)

                # Generating indices a priori before parallelism ended up being orders of magnitude
                # faster than how sklearn does it. The reason is that random samplers do not release the
                # gil it seems.
                n_groups = n_more_estimators // self.subforest_size
                new_slices = np.array_split(np.arange(len(self.estimators_),
                                                      len(self.estimators_) + n_more_estimators),
                                            n_groups)
                s_inds = []
                for sl in new_slices:
                    half_sample_inds = subsample_random_state.choice(n_samples, n_samples // 2, replace=False)
                    s_inds.extend([half_sample_inds[subsample_random_state.choice(n_samples // 2,
                                                                                  n_samples_subsample,
                                                                                  replace=False)]
                                   for _ in range(len(sl))])
            else:
                if self.warm_start:
                    # Advancing subsample_random_state. Assumes each prior fit call has the same number of
                    # samples at fit time. If not then this would not exactly replicate a single batch execution,
                    # but would still advance randomness enough so that tree subsamples will be different.
                    for _, n_, ns_ in range(len(self.estimators_), self.n_samples_, self.n_samples_subsample_):
                        subsample_random_state.choice(n_, ns_, replace=False)
                new_slices = []
                s_inds = [subsample_random_state.choice(n_samples, n_samples_subsample, replace=False)
                          for _ in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='threading')(
                delayed(t.fit)(X[s], yaug[s], self.n_y_, self.n_outputs_, self.n_relevant_outputs_,
                               sample_weight=sample_weight[s] if sample_weight is not None else None,
                               check_input=False)
                for t, s in zip(trees, s_inds))

            # Collect newly grown trees
            self.estimators_.extend(trees)
            self.n_samples_.extend([n_samples] * len(trees))
            self.n_samples_subsample_.extend([n_samples_subsample] * len(trees))
            self.slices_.extend(list(new_slices))
            self.slices_n_samples_.extend([n_samples] * len(new_slices))
            self.slices_n_samples_subsample_.extend([n_samples_subsample] * len(new_slices))

        return self

    def get_subsample_inds(self,):
        """ Re-generate the example same sample indices as those at fit time using same pseudo-randomness.
        """
        check_is_fitted(self)
        subsample_random_state = check_random_state(self.subsample_random_seed_)
        if self.inference_:
            s_inds = []
            for sl, n_, ns_ in zip(self.slices_, self.slices_n_samples_, self.slices_n_samples_subsample_):
                half_sample_inds = subsample_random_state.choice(n_, n_ // 2, replace=False)
                s_inds.extend([half_sample_inds[subsample_random_state.choice(n_ // 2, ns_, replace=False)]
                               for _ in range(len(sl))])
            return s_inds
        else:
            return [subsample_random_state.choice(n_, ns_, replace=False)
                    for n_, ns_ in zip(self.n_samples_, self.n_samples_subsample_)]

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        """
        The feature importances based on the amount of parameter heterogeneity they create.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized) total heterogeneity that the feature
        creates. For each tree and for each split that the feature was chosen adds::

            parent_weight * (left_weight * right_weight)
                * mean((value_left[k] - value_right[k])**2) / parent_weight**2

        to the importance of the feature. Each such quantity is also weighted by the depth of the split.
        These importances are normalized at the tree level and then averaged across trees.

        Parameters
        ----------
        max_depth : int, default=4
            Splits of depth larger than `max_depth` are not used in this calculation
        depth_decay_exponent: double, default=2.0
            The contribution of each split to the total score is re-weighted by 1 / (1 + `depth`)**2.0.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total parameter heterogeneity inducing importance of each feature
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(tree.feature_importances)(
                max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    @property
    def feature_importances_(self):
        return self.feature_importances()

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, and other predict methods."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    def predict_tree_average_full(self, X):
        """ Return the fitted local parameters for each X, i.e. theta(X). This
        method simply returns the average of the parameters estimated by each tree. `predict_full`
        should be preferred over `pred_tree_average_full`, as it performs a more stable averaging across
        trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        theta(X) : array-like of shape (n_samples, n_outputs)
            The estimated relevant parameters for each row of X
        """

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading', require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_full, X, [y_hat], lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat

    def predict_tree_average(self, X):
        """ Return the prefix of relevant fitted local parameters for each X, i.e. theta(X)[1..n_relevant_outputs].
        This method simply returns the average of the parameters estimated by each tree. `predict`
        should be preferred over `pred_tree_average`, as it performs a more stable averaging across
        trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        theta(X)[1, .., n_relevant_outputs] : array-like of shape (n_samples, n_relevant_outputs)
            The estimated relevant parameters for each row of X
        """
        y_hat = self.predict_tree_average_full(X)
        if self.n_relevant_outputs_ == self.n_outputs_:
            return y_hat
        return y_hat[:, :self.n_relevant_outputs_]

    def predict_moment_and_var(self, X, parameter, slice=None, parallel=True):
        """ Return the value of the conditional expected moment vector at each sample and
        for the given parameter estimate for each sample::

            M(x; theta(x)) := E[J | X=x] theta(x) - E[A | X=x]

        where conditional expectations are estimated based on the forest weights, i.e.::

            M_tree(x; theta(x)) := (1/ |leaf(x)|) sum_{val sample i in leaf(x)} w[i] (J[i] theta(x) - A[i])
            M(x; theta(x) = (1/n_trees) sum_{trees} M_tree(x; theta(x))

        where w[i] is the sample weight (1.0 if sample_weight is None), as well as the variance of the local
        moment vector across trees::

            Var(M_tree(x; theta(x))) = (1/n_trees) sum_{trees} M_tree(x; theta(x)) @ M_tree(x; theta(x)).T

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        parameter : array-like of shape (n_samples, n_outputs)
            An estimate of the parameter theta(x) for each sample x in X
        slice : list of int or None, default=None
            If not None, then only the trees with index in slice, will be used to calculate the mean
            and the variance.
        parallel : bool , default=True
            Whether the averaging should happen using parallelism or not. Parallelism adds some overhead
            but makes it faster with many trees.

        Returns
        -------
        moment : array-like of shape (n_samples, n_outputs)
            The estimated conditional moment M(x; theta(x)) for each sample x in X
        moment_var : array-like of shape (n_samples, n_outputs)
            The variance of the conditional moment Var(M_tree(x; theta(x))) across trees for each sample x
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        if slice is None:
            slice = np.arange(len(self.estimators_))

        moment_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        moment_var_hat = np.zeros((X.shape[0], self.n_outputs_, self.n_outputs_), dtype=np.float64)
        lock = threading.Lock()
        if parallel:
            n_jobs, _, _ = _partition_estimators(len(slice), self.n_jobs)
            verbose = self.verbose
            # Parallel loop
            Parallel(n_jobs=n_jobs, verbose=verbose, backend='threading', require="sharedmem")(
                delayed(_accumulate_prediction_and_var)(self.estimators_[t].predict_moment, X,
                                                        [moment_hat], [moment_var_hat], lock,
                                                        parameter)
                for t in slice)
        else:
            [_accumulate_prediction_and_var(self.estimators_[t].predict_moment, X,
                                            [moment_hat], [moment_var_hat], lock,
                                            parameter)
             for t in slice]

        moment_hat /= len(slice)
        moment_var_hat /= len(slice)

        return moment_hat, moment_var_hat

    def predict_alpha_and_jac(self, X, slice=None, parallel=True):
        """ Return the value of the conditional jacobian E[J | X=x] and the conditional alpha E[A | X=x]
        using the forest as kernel weights, i.e.::

            alpha(x) = (1/n_trees) sum_{trees} (1/ |leaf(x)|) sum_{val sample i in leaf(x)} w[i] A[i]
            jac(x) = (1/n_trees) sum_{trees} (1/ |leaf(x)|) sum_{val sample i in leaf(x)} w[i] J[i]

        where w[i] is the sample weight (1.0 if sample_weight is None).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        slice : list of int or None, default=None
            If not None, then only the trees with index in slice, will be used to calculate the mean
            and the variance.
        parallel : bool , default=True
            Whether the averaging should happen using parallelism or not. Parallelism adds some overhead
            but makes it faster with many trees.

        Returns
        -------
        alpha : array-like of shape (n_samples, n_outputs)
            The estimated conditional A, alpha(x) for each sample x in X
        jac : array-like of shape (n_samples, n_outputs, n_outputs)
            The estimated conditional J, jac(x) for each sample x in X
        """
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        if slice is None:
            slice = np.arange(len(self.estimators_))
        n_jobs = 1
        verbose = 0
        if parallel:
            n_jobs, _, _ = _partition_estimators(len(slice), self.n_jobs)
            verbose = self.verbose

        alpha_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        jac_hat = np.zeros((X.shape[0], self.n_outputs_**2), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=verbose, backend='threading', require="sharedmem")(
            delayed(_accumulate_prediction)(self.estimators_[t].predict_alpha_and_jac, X, [alpha_hat, jac_hat], lock)
            for t in slice)

        alpha_hat /= len(slice)
        jac_hat /= len(slice)

        return alpha_hat, jac_hat.reshape((-1, self.n_outputs_, self.n_outputs_))

    def _predict_point_and_var(self, X, full=False, point=True, var=False, project=False, projector=None):
        """ An internal private method that coordinates all prediction functionality and tries to share
        as much computation between different predict methods to avoid re-computation and re-spawining of
        parallel executions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        full : bool, default=False
            Whether to return the full estimated parameter or only the relevant part
        point : bool, default=True
            Whether to return the point estimate theta(x)
        var : bool, default=False
            Whether to return the co-variance of the point estimate V(theta(x))
        project : bool, default=False
            Whether to project the point estimate using an inner product with a projector, and also
            return the variance of the projection
        projector : array-like of shape (n_samples, n_outputs)
            The projection vector for each sample. The point estimate theta(x) for each sample will
            be projected and return the inner produce <theta(x), projector(x)> for each sample x.
            Also the variance information will be about the inner product as opposed to the parameter
            theta(x).

        Returns
        -------
        point : array-like of shape (n_samples, x)
            The point estimate of the parameter theta(x) or its inner product with projector(x) for each
            sample x in X.
            If `point=False`, this return value is omitted. If `project=True`, then `x=1`. If `project=False`
            and `full=True`, then `x=n_outputs`. If `project=False` and `full=False`, then `x=n_relevant_outputs`.
        var : array-like of shape (n_samples, x, x) or (n_samples, 1)
            The covariance of the parameter theta(x) or its inner product with projector(x) for each sample x in X.
            If `var=False`, this return value is omitted. If `project=True`, then return is of shape (n_samples, 1).
            If `project=False` and `full=True`, then `x=n_outputs`. If `project=False` and `full=False`,
            then `x=n_relevant_outputs`.
        """

        alpha, jac = self.predict_alpha_and_jac(X)
        invjac = np.linalg.pinv(jac)
        parameter = np.einsum('ijk,ik->ij', invjac, alpha)

        if var:
            if not self.inference:
                raise AttributeError("Inference not available. Forest was initiated with `inference=False`.")

            slices = self.slices_
            n_jobs, _, _ = _partition_estimators(len(slices), self.n_jobs)

            moment_bags, moment_var_bags = zip(*Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading')(
                delayed(self.predict_moment_and_var)(X, parameter, slice=sl, parallel=False) for sl in slices))

            moment = np.mean(moment_bags, axis=0)

            trans_moment_bags = np.moveaxis(moment_bags, 0, -1)
            sq_between = np.einsum('tij,tjk->tik', trans_moment_bags,
                                   np.transpose(trans_moment_bags, (0, 2, 1))) / len(slices)
            moment_sq = np.einsum('tij,tjk->tik',
                                  moment.reshape(moment.shape + (1,)),
                                  moment.reshape(moment.shape[:-1] + (1, moment.shape[-1])))
            var_between = sq_between - moment_sq
            pred_cov = np.einsum('ijk,ikm->ijm', invjac,
                                 np.einsum('ijk,ikm->ijm', var_between, np.transpose(invjac, (0, 2, 1))))

            if project:
                pred_var = np.einsum('ijk,ikm->ijm', projector.reshape((-1, 1, projector.shape[1])),
                                     np.einsum('ijk,ikm->ijm', pred_cov,
                                               projector.reshape((-1, projector.shape[1], 1))))[:, 0, 0]
            else:
                pred_var = np.diagonal(pred_cov, axis1=1, axis2=2)

            #####################
            # Variance correction
            #####################
            # Subtract the average within bag variance. This ends up being equal to the
            # overall (E_{all trees}[moment^2] - E_bags[ E[mean_bag_moment]^2 ]) / sizeof(bag).
            # The negative part is just sq_between.
            var_total = np.mean(moment_var_bags, axis=0)
            correction = (var_total - sq_between) / (len(slices[0]) - 1)
            pred_cov_correction = np.einsum('ijk,ikm->ijm', invjac,
                                            np.einsum('ijk,ikm->ijm', correction, np.transpose(invjac, (0, 2, 1))))
            if project:
                pred_var_correction = np.einsum('ijk,ikm->ijm', projector.reshape((-1, 1, projector.shape[1])),
                                                np.einsum('ijk,ikm->ijm', pred_cov_correction,
                                                          projector.reshape((-1, projector.shape[1], 1))))[:, 0, 0]
            else:
                pred_var_correction = np.diagonal(pred_cov_correction, axis1=1, axis2=2)
            # Objective bayes debiasing for the diagonals where we know a-prior they are positive
            # The off diagonals we have no objective prior, so no correction is applied.
            naive_estimate = pred_var - pred_var_correction
            se = np.maximum(pred_var, pred_var_correction) * np.sqrt(2.0 / len(slices))
            zstat = naive_estimate / np.clip(se, 1e-10, np.inf)
            numerator = np.exp(- (zstat**2) / 2) / np.sqrt(2.0 * np.pi)
            denominator = 0.5 * erfc(-zstat / np.sqrt(2.0))
            pred_var_corrected = naive_estimate + se * numerator / denominator

            # Finally correcting the pred_cov or pred_var
            if project:
                pred_var = pred_var_corrected
            else:
                pred_cov = pred_cov - pred_cov_correction
                for t in range(self.n_outputs_):
                    pred_cov[:, t, t] = pred_var_corrected[:, t]

        if project:
            if point:
                pred = np.sum(parameter * projector, axis=1)
                if var:
                    return pred, pred_var
                else:
                    return pred
            else:
                return pred_var
        else:
            n_outputs = self.n_outputs_ if full else self.n_relevant_outputs_
            if point and var:
                return (parameter[:, :n_outputs],
                        pred_cov[:, :n_outputs, :n_outputs],)
            elif point:
                return parameter[:, :n_outputs]
            else:
                return pred_cov[:, :n_outputs, :n_outputs]

    def predict_full(self, X, interval=False, alpha=0.05):
        """ Return the fitted local parameters for each x in X, i.e. theta(x).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        interval : bool, default=False
            Whether to return a confidence interval too
        alpha : float in (0, 1), default=0.05
            The confidence level of the confidence interval. Returns a symmetric (alpha/2, 1-alpha/2)
            confidence interval.

        Returns
        -------
        theta(x) : array-like of shape (n_samples, n_outputs)
            The estimated relevant parameters for each row x of X
        lb(x), ub(x) : array-like of shape (n_samples, n_outputs)
            The lower and upper end of the confidence interval for each parameter. Return value is omitted if
            `interval=False`.
        """
        if interval:
            point, pred_var = self._predict_point_and_var(X, full=True, point=True, var=True)
            lb, ub = np.zeros(point.shape), np.zeros(point.shape)
            for t in range(self.n_outputs_):
                lb[:, t] = scipy.stats.norm.ppf(alpha / 2, loc=point[:, t], scale=np.sqrt(pred_var[:, t, t]))
                ub[:, t] = scipy.stats.norm.ppf(1 - alpha / 2, loc=point[:, t], scale=np.sqrt(pred_var[:, t, t]))
            return point, lb, ub
        return self._predict_point_and_var(X, full=True, point=True, var=False)

    def predict(self, X, interval=False, alpha=0.05):
        """ Return the prefix of relevant fitted local parameters for each x in X,
        i.e. theta(x)[1..n_relevant_outputs].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        interval : bool, default=False
            Whether to return a confidence interval too
        alpha : float in (0, 1), default=0.05
            The confidence level of the confidence interval. Returns a symmetric (alpha/2, 1-alpha/2)
            confidence interval.

        Returns
        -------
        theta(X)[1, .., n_relevant_outputs] : array-like of shape (n_samples, n_relevant_outputs)
            The estimated relevant parameters for each row of X
        lb(x), ub(x) : array-like of shape (n_samples, n_relevant_outputs)
            The lower and upper end of the confidence interval for each parameter. Return value is omitted if
            `interval=False`.
        """
        if interval:
            y_hat, lb, ub = self.predict_full(X, interval=interval, alpha=alpha)
            if self.n_relevant_outputs_ == self.n_outputs_:
                return y_hat, lb, ub
            return (y_hat[:, :self.n_relevant_outputs_],
                    lb[:, :self.n_relevant_outputs_], ub[:, :self.n_relevant_outputs_])
        else:
            y_hat = self.predict_full(X, interval=False)
            if self.n_relevant_outputs_ == self.n_outputs_:
                return y_hat
            return y_hat[:, :self.n_relevant_outputs_]

    def predict_interval(self, X, alpha=0.05):
        """ Return the confidence interval for the relevant fitted local parameters for each x in X,
        i.e. theta(x)[1..n_relevant_outputs].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        alpha : float in (0, 1), default=0.05
            The confidence level of the confidence interval. Returns a symmetric (alpha/2, 1-alpha/2)
            confidence interval.

        Returns
        -------
        lb(x), ub(x) : array-like of shape (n_samples, n_relevant_outputs)
            The lower and upper end of the confidence interval for each parameter. Return value is omitted if
            `interval=False`.
        """
        _, lb, ub = self.predict(X, interval=True, alpha=alpha)
        return lb, ub

    def predict_and_var(self, X):
        """ Return the prefix of relevant fitted local parameters for each x in X,
        i.e. theta(x)[1..n_relevant_outputs] and their covariance matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        theta(x)[1, .., n_relevant_outputs] : array-like of shape (n_samples, n_relevant_outputs)
            The estimated relevant parameters for each row of X
        var(theta(x)) : array-like of shape (n_samples, n_relevant_outputs, n_relevant_outputs)
            The covariance of theta(x)[1, .., n_relevant_outputs]
        """
        return self._predict_point_and_var(X, full=False, point=True, var=True)

    def predict_var(self, X):
        """ Return the covariance matrix of the prefix of relevant fitted local parameters
        for each x in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        var(theta(x)) : array-like of shape (n_samples, n_relevant_outputs, n_relevant_outputs)
            The covariance of theta(x)[1, .., n_relevant_outputs]
        """
        return self._predict_point_and_var(X, full=False, point=False, var=True)

    def prediction_stderr(self, X):
        """ Return the standard deviation of each coordinate of the prefix of relevant fitted local parameters
        for each x in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.

        Returns
        -------
        std(theta(x)) : array-like of shape (n_samples, n_relevant_outputs)
            The standard deviation of each theta(x)[i] for i in {1, .., n_relevant_outputs}
        """
        return np.sqrt(np.diagonal(self.predict_var(X), axis1=1, axis2=2))

    def _check_projector(self, X, projector):
        """ validate the projector parameter
        """
        X, projector = check_X_y(X, projector, multi_output=True, y_numeric=True)
        if projector.ndim == 1:
            projector = projector.reshape((-1, 1))
        if self.n_outputs_ > self.n_relevant_outputs_:
            projector = np.hstack([projector,
                                   np.zeros((projector.shape[0], self.n_outputs_ - self.n_relevant_outputs_))])
        return X, projector

    def predict_projection_and_var(self, X, projector):
        """ Return the inner product of the prefix of relevant fitted local parameters for each x in X,
        i.e. theta(x)[1..n_relevant_outputs], with a projector vector projector(x), i.e.::

            mu(x) := <theta(x)[1..n_relevant_outputs], projector(x)>

        as well as the variance of mu(x).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        projector : array-like of shape (n_samples, n_relevant_outputs)
            The projector vector for each sample x in X

        Returns
        -------
        mu(x) : array-like of shape (n_samples, 1)
            The estimated inner product of the relevant parameters with the projector for each row x of X
        var(mu(x)) : array-like of shape (n_samples, 1)
            The variance of the estimated inner product
        """
        X, projector = self._check_projector(X, projector)
        return self._predict_point_and_var(X, full=False, point=True, var=True,
                                           project=True, projector=projector)

    def predict_projection(self, X, projector):
        """ Return the inner product of the prefix of relevant fitted local parameters for each x in X,
        i.e. theta(x)[1..n_relevant_outputs], with a projector vector projector(x), i.e.::

            mu(x) := <theta(x)[1..n_relevant_outputs], projector(x)>

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        projector : array-like of shape (n_samples, n_relevant_outputs)
            The projector vector for each sample x in X

        Returns
        -------
        mu(x) : array-like of shape (n_samples, 1)
            The estimated inner product of the relevant parameters with the projector for each row x of X
        """
        X, projector = self._check_projector(X, projector)
        return self._predict_point_and_var(X, full=False, point=True, var=False,
                                           project=True, projector=projector)

    def predict_projection_var(self, X, projector):
        """ Return the variance of the inner product of the prefix of relevant fitted local parameters
        for each x in X, i.e. theta(x)[1..n_relevant_outputs], with a projector vector projector(x), i.e.::

            Var(mu(x)) for mu(x) := <theta(x)[1..n_relevant_outputs], projector(x)>

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        projector : array-like of shape (n_samples, n_relevant_outputs)
            The projector vector for each sample x in X

        Returns
        -------
        var(mu(x)) : array-like of shape (n_samples, 1)
            The variance of the estimated inner product
        """
        X, projector = self._check_projector(X, projector)
        return self._predict_point_and_var(X, full=False, point=False, var=True,
                                           project=True, projector=projector)
