# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Subsampled honest forest extension to scikit-learn's forest methods.
"""

import numpy as np
import scipy.sparse
import threading
import sparse as sp
import itertools
from sklearn.utils import check_array, check_X_y, issparse
from sklearn.ensemble.forest import ForestRegressor, _accumulate_prediction
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from warnings import catch_warnings, simplefilter, warn
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.tree._tree import DTYPE, DOUBLE
from sklearn.utils import check_random_state, check_array, compute_sample_weight
from sklearn.utils._joblib import Parallel, delayed
from sklearn.utils.fixes import parallel_helper, _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.base import _partition_estimators

MAX_RAND_SEED = np.iinfo(np.int32).max
MAX_INT = np.iinfo(np.int32).max


def _parallel_add_trees(tree, forest, X, y, sample_weight, sample_var, s_inds, tree_idx, n_trees, verbose=0):
    """Private function used to fit a single subsampled honest tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    # Construct the subsample of data
    X, y = X[s_inds], y[s_inds]
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],), dtype=np.float64)
    else:
        sample_weight = sample_weight[s_inds]

    if sample_var is None:
        sample_var_none = True
        sample_var = np.zeros((X.shape[0],), dtype=np.float64)
    else:
        sampl_var_none = False

    # Split into estimation and splitting sample set
    if forest.honest:
        X_split, X_est, y_split, y_est,\
            sample_weight_split, sample_weight_est,\
            sample_var_split, sample_var_est = train_test_split(
                X, y, sample_weight, sample_var, test_size=.5, shuffle=True)
    else:
        X_split, X_est, y_split, y_est, sample_weight_split, sample_weight_est, sample_var_split, sample_var_est =\
            X, X, y, y, sample_weight, sample_weight, sample_var, sample_var

    # Fit the tree on the splitting sample
    tree.fit(X_split, y_split, sample_weight=sample_weight_split, check_input=False)

    if not sample_var_none or forest.honest:
        # Apply the trained tree on the estimation sample to get the path for every estimation sample
        path_est = tree.decision_path(X_est)

    sample_var_node = None
    if not sample_var_none:
        sample_var_node = np.array(scipy.sparse.csr_matrix(sample_var_est.reshape(1, -1)).dot(path_est))[0]

    # Set the estimation values based on the estimation split
    total_weight_est = np.sum(sample_weight_est)
    if forest.honest:
        # Calculate the total weight of estimation samples on each tree node
        weight_est = scipy.sparse.csr_matrix(sample_weight_est.reshape(1, -1)).dot(path_est)
        # Calculate the total number of estimation samples on each tree node
        count_est = np.sum(path_est, axis=0)
        # Calculate the weighted sum of responses on the estimation sample on each node
        value_est = scipy.sparse.csr_matrix((sample_weight_est.reshape(-1, 1) * y_est).T).dot(path_est)
        # Prune tree to remove leafs that don't satisfy the leaf requirements on the estimation sample
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_id = stack.pop()
            if weight_est[0, node_id] / total_weight_est < forest.min_weight_fraction_leaf\
                    or count_est[0, node_id] < forest.min_samples_leaf:
                tree.tree_.children_left[parent_id] = -1
                tree.tree_.children_right[parent_id] = -1
            else:
                for i in range(tree.n_outputs_):
                    tree.tree_.value[node_id, i] = value_est[i, node_id]
                tree.tree_.weighted_n_node_samples[node_id] = weight_est[0, node_id]
                tree.tree_.n_node_samples[node_id] = count_est[0, node_id]
                if (children_left[node_id] != children_right[node_id]):
                    stack.append((children_left[node_id], node_id))
                    stack.append((children_right[node_id], node_id))

    return tree, sample_var_node


class SubsampledHonestForest(ForestRegressor, RegressorMixin):
    """
    An implementation of a subsampled honest random forest regressor on top of an sklearn
    regression tree. Implements subsampling and honesty as described in [3],
    but uses a scikit-learn regression tree as a base.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is smaller than the original size and subsampling is
    performed without replacement. Each decision tree is built in an honest
    manner: half of the sub-sampled data are used for creating the tree structure
    (referred to as the splitting sample) and the other half for calculating the
    constant regression estimate at each leaf of the tree (referred to as the estimation sample).
    One difference with the algorithm proposed in [3] is that we do not ensure balancedness
    and we do not consider poisson sampling of the features, so that we guarantee
    that each feature has a positive probability of being selected on each split.
    Rather we use the original algorithm of Breiman [1], which selects the best split
    among a collection of candidate splits, as long as the max_depth is not reached
    and as long as there are not more than max_leafs and each child contains
    at least min_samples_leaf samples and total weight fraction of
    min_weight_fraction_leaf. Moreover, it allows the use of both mean squared error (MSE)
    and mean absoulte error (MAE) as the splitting criterion. Finally, we allow
    for early stopping of the splits if the criterion is not improved by more than
    min_impurity_decrease. These techniques that date back to the work of [1],
    should lead to finite sample performance improvements, especially for
    high dimensional features.

    The implementation also provides confidence intervals
    for each prediction using a bootstrap of little bags approach described in [3]:
    subsampling is performed at hierarchical level by first drawing a set of half-samples
    at random and then sub-sampling from each half-sample to build a forest
    of forests. All the trees are used for the point prediction and the distribution
    of predictions returned by each of the sub-forests is used for confidence
    interval construction.

    Parameters
    ----------
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
        `None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest. See :term:`the Glossary <warm_start>`.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
        Feature importances are calculate based on the reduction in the
        splitting criterion among the split samples (not the estimation samples).
        So it might contain some upwards bias.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    Examples
    --------
    >>> from econml.honestforest import SubsampledHonestForest
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
    ...                        random_state=0, shuffle=False)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
    >>> regr = SubsampledHonestForest(max_depth=None, random_state=0,
    ...                          n_estimators=1000)
    >>> regr.fit(X_train, y_train)
    SubsampledHonestForest(criterion='mse', honest=True, max_depth=None,
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=1000, n_jobs=None, random_state=0,
            subsample_fr=0.5757129508131623, verbose=0, warm_start=False)
    >>> print(regr.feature_importances_)
    [0.58333173 0.39696654 0.01039317 0.00930856]
    >>> print(regr.predict(np.ones((1, 4))))
    [114.09511749]
    >>> print(regr.predict_interval(np.ones((1, 4))), lower=2.5, upper=97.5)
    (array([102.25043604]), array([124.83995659]))
    >>> regr.score(X_test, y_test)
    0.944958960444053

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values. For valid inference, the trees
    are recommended to be fully grown.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    The default value ``max_features="auto"`` uses ``n_features``
    rather than ``n_features / 3``. The latter was originally suggested in
    [1], whereas the former was more recently justified empirically in [2].

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    .. [3] S. Athey, S. Wager, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests",
    Journal of the American Statistical Association 113.523 (2018): 1228-1242.

    """

    def __init__(self,
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
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(SubsampledHonestForest, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease",
                              "random_state"),
            bootstrap=False,
            oob_score=False,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.subsample_fr = subsample_fr
        self.honest = honest
        self.estimators_ = None
        self.vars_ = None

        return

    def fit(self, X, y, sample_weight=None, sample_var=None):
        """
        Parameters
        ----------
        X : features
        y : label
        sample_weight : sample weights
        """

        # Validate or convert input data
        X = check_array(X, accept_sparse="csc", dtype=DTYPE)
        y = check_array(y, accept_sparse='csc', ensure_2d=False, dtype=None)
        if sample_weight is not None:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if self.subsample_fr == 'auto':
            self.subsample_fr = (X.shape[0] / 2)**(1 - 1 / (2 * X.shape[1] + 2)) / (X.shape[0] / 2)

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.vars_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            self.n_slices = int(np.ceil((self.n_estimators)**(1 / 2)))
            self.slice_len = int(np.ceil(self.n_estimators / self.n_slices))
            s_inds = []
            # TODO. This slicing should ultimately be done inside the parallel function
            # so that we don't need to create a matrix of size roughly n_samples * n_estimators
            for it in range(self.n_slices):
                half_sample_inds = np.random.choice(X.shape[0], X.shape[0] // 2, replace=False)
                for _ in np.arange(it * self.slice_len, min((it + 1) * self.slice_len, self.n_estimators)):
                    s_inds.append(half_sample_inds[np.random.choice(X.shape[0] // 2,
                                                                    int(np.ceil(self.subsample_fr *
                                                                                (X.shape[0] // 2))),
                                                                    replace=False)])
            trees_plus_vars = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                                       **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_add_trees)(
                    t, self, X, y, sample_weight, sample_var, s_inds[i], i, len(trees),
                    verbose=self.verbose)
                for i, t in enumerate(trees))
            trees = [tree for tree, _ in trees_plus_vars]
            # Collect newly grown trees
            self.estimators_.extend(trees)

            if sample_var is not None:
                vars = [var for _, var in trees_plus_vars]
                self.vars_.extend(vars)

        return self

    def _weight(self, X):
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)
        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
        # avoid storing the output of every estimator by summing them here
        weight_hat = np.zeros((X.shape[0]), dtype=np.float64)
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(lambda x, check_input: e.tree_.weighted_n_node_samples[e.apply(x)],
                                            X, [weight_hat], lock)
            for e in self.estimators_)
        weight_hat /= len(self.estimators_)
        return weight_hat

    def predict(self, X):
        y_pred = super(SubsampledHonestForest, self).predict(X)
        weight_hat = self._weight(X)
        if len(y_pred.shape) > 1:
            weight_hat.reshape((y_pred.shape[0], 1))
        return y_pred / weight_hat

    def predict_interval(self, X, alpha=.1, normal=True):
        """
        Parameters
        ----------
        X : features
        alpha : the significance level of the interval
        normal : whether to use normal approximation to construct CI or perform
            a non-parametric bootstrap of little bags
        """
        lower = 100 * alpha / 2
        upper = 100 * (1 - alpha / 2)

        y_pred = np.array([tree.predict(X) for tree in self.estimators_])
        y_bags_pred = np.array([np.nanmean(y_pred[np.arange(it * self.slice_len,
                                                            min((it + 1) * self.slice_len, self.n_estimators))],
                                           axis=0)
                                for it in range(self.n_slices)])
        weight_hat = np.array([tree.tree_.weighted_n_node_samples[tree.apply(X)] for tree in self.estimators_])
        if self.vars_:
            var_hat = np.array([var[tree.apply(X)] for tree, var in zip(self.estimators_, self.vars_)])
        weight_hat_bags = np.array([np.nanmean(weight_hat[np.arange(it * self.slice_len,
                                                                    min((it + 1) * self.slice_len,
                                                                        self.n_estimators))], axis=0)
                                    for it in range(self.n_slices)])

        """
        The recommended approach in the GRF paper for estimating variance of estimate, specialized to this setup.

        .. math ::
            Var(\\theta(X)) ~\
                Var_{random half-samples S}[ \\sum_{b\\in S} w_b(x) (Y_i - \\theta(X)) ] / (\\sum_{b} w_b(x))^2
        """
        y_point_pred = np.sum(y_pred, axis=0) / np.sum(weight_hat, axis=0)
        bag_res = y_bags_pred - weight_hat_bags * y_point_pred.reshape(1, -1)
        if not self.vars_:
            std_pred = np.sqrt(np.nanmean(bag_res**2, axis=0)) / np.nanmean(weight_hat, axis=0)
        else:
            std_pred = np.sqrt(np.nanmean(bag_res**2, axis=0) + np.nanmean(var_hat, axis=0)) / \
                np.nanmean(weight_hat, axis=0)
        y_bags_pred /= weight_hat_bags

        if normal:
            upper_pred = scipy.stats.norm.ppf(upper / 100, loc=y_point_pred, scale=std_pred)
            lower_pred = scipy.stats.norm.ppf(lower / 100, loc=y_point_pred, scale=std_pred)
            return lower_pred, upper_pred
        else:
            return np.nanpercentile(y_bags_pred, lower, axis=0), np.nanpercentile(y_bags_pred, upper, axis=0)
