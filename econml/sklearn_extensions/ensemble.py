# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Subsampled honest forest extension to scikit-learn's forest methods. Contains pieces of code from
scikit-learn's random forest implementation.
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
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.base import _partition_estimators

MAX_INT = np.iinfo(np.int32).max


def _parallel_add_trees(tree, forest, X, y, sample_weight, s_inds, tree_idx, n_trees, verbose=0):
    """Private function used to fit a single subsampled honest tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    # Construct the subsample of data
    X, y = X[s_inds], y[s_inds]
    if sample_weight is None:
        sample_weight = np.ones((X.shape[0],), dtype=np.float64)
    else:
        sample_weight = sample_weight[s_inds]

    # Split into estimation and splitting sample set
    if forest.honest:
        X_split, X_est, y_split, y_est,\
            sample_weight_split, sample_weight_est = train_test_split(
                X, y, sample_weight, test_size=.5, shuffle=True, random_state=tree.random_state)
    else:
        X_split, X_est, y_split, y_est, sample_weight_split, sample_weight_est =\
            X, X, y, y, sample_weight, sample_weight

    # Fit the tree on the splitting sample
    tree.fit(X_split, y_split, sample_weight=sample_weight_split,
             check_input=False)

    # Set the estimation values based on the estimation split
    total_weight_est = np.sum(sample_weight_est)
    # Apply the trained tree on the estimation sample to get the path for every estimation sample
    path_est = tree.decision_path(X_est)
    # Calculate the total weight of estimation samples on each tree node:
    # \sum_i sample_weight[i] * 1{i \\in node}
    weight_est = sample_weight_est.reshape(1, -1) @ path_est
    # Calculate the total number of estimation samples on each tree node:
    # |node| = \sum_{i} 1{i \\in node}
    count_est = path_est.sum(axis=0)
    # Calculate the weighted sum of responses on the estimation sample on each node:
    # \sum_{i} sample_weight[i] 1{i \\in node} Y_i
    num_est = (sample_weight_est.reshape(-1, 1) * y_est).T @ path_est
    # Calculate the predicted value on each node based on the estimation sample:
    # weighted sum of responses / total weight
    value_est = num_est / weight_est

    # Calculate the criterion on each node based on the estimation sample and for each output dimension,
    # summing the impurity across dimensions.
    # First we calculate the difference of observed label y of each node and predicted value for each
    # node that the sample falls in: y[i] - value_est[node]
    impurity_est = np.zeros((1, path_est.shape[1]))
    for i in range(tree.n_outputs_):
        diff = path_est.multiply(y_est[:, [i]]) - path_est.multiply(value_est[[i], :])
        if tree.criterion == 'mse':
            # If criterion is mse then calculate weighted sum of squared differences for each node
            impurity_est_i = sample_weight_est.reshape(1, -1) @ diff.power(2)
        elif tree.criterion == 'mae':
            # If criterion is mae then calculate weighted sum of absolute differences for each node
            impurity_est_i = sample_weight_est.reshape(1, -1) @ np.abs(diff)
        else:
            raise AttributeError("Criterion {} not yet supported by SubsampledHonestForest!".format(tree.criterion))
        # Normalize each weighted sum of criterion for each node by the total weight of each node
        impurity_est += impurity_est_i / weight_est

    # Prune tree to remove leafs that don't satisfy the leaf requirements on the estimation sample
    # and for each un-pruned tree set the value and the weight appropriately.
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    numerator = np.empty_like(tree.tree_.value)
    denominator = np.empty_like(tree.tree_.weighted_n_node_samples)
    while len(stack) > 0:
        node_id, parent_id = stack.pop()
        # If minimum weight requirement or minimum leaf size requirement is not satisfied on estimation
        # sample, then prune the whole sub-tree
        if weight_est[0, node_id] / total_weight_est < forest.min_weight_fraction_leaf\
                or count_est[0, node_id] < forest.min_samples_leaf:
            tree.tree_.children_left[parent_id] = -1
            tree.tree_.children_right[parent_id] = -1
        else:
            for i in range(tree.n_outputs_):
                # Set the numerator of the node to: \sum_{i} sample_weight[i] 1{i \\in node} Y_i / |node|
                numerator[node_id, i] = num_est[i, node_id] / count_est[0, node_id]
                # Set the value of the node to:
                # \sum_{i} sample_weight[i] 1{i \\in node} Y_i / \sum_{i} sample_weight[i] 1{i \\in node}
                tree.tree_.value[node_id, i] = value_est[i, node_id]
            # Set the denominator of the node to: \sum_{i} sample_weight[i] 1{i \\in node} / |node|
            denominator[node_id] = weight_est[0, node_id] / count_est[0, node_id]
            # Set the weight of the node to: \sum_{i} sample_weight[i] 1{i \\in node}
            tree.tree_.weighted_n_node_samples[node_id] = weight_est[0, node_id]
            # Set the count to the estimation split count
            tree.tree_.n_node_samples[node_id] = count_est[0, node_id]
            # Set the node impurity to the estimation split impurity
            tree.tree_.impurity[node_id] = impurity_est[0, node_id]
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], node_id))
                stack.append((children_right[node_id], node_id))

    return tree, numerator, denominator


class SubsampledHonestForest(ForestRegressor, RegressorMixin):
    """
    An implementation of a subsampled honest random forest regressor on top of an sklearn
    regression tree. Implements subsampling and honesty as described in [3]_,
    but uses a scikit-learn regression tree as a base. It provides confidence intervals based on ideas
    described in [3]_ and [4]_

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is smaller than the original size and subsampling is
    performed without replacement. Each decision tree is built in an honest
    manner: half of the sub-sampled data are used for creating the tree structure
    (referred to as the splitting sample) and the other half for calculating the
    constant regression estimate at each leaf of the tree (referred to as the estimation sample).
    One difference with the algorithm proposed in [3]_ is that we do not ensure balancedness
    and we do not consider poisson sampling of the features, so that we guarantee
    that each feature has a positive probability of being selected on each split.
    Rather we use the original algorithm of Breiman [1]_, which selects the best split
    among a collection of candidate splits, as long as the max_depth is not reached
    and as long as there are not more than max_leafs and each child contains
    at least min_samples_leaf samples and total weight fraction of
    min_weight_fraction_leaf. Moreover, it allows the use of both mean squared error (MSE)
    and mean absoulte error (MAE) as the splitting criterion. Finally, we allow
    for early stopping of the splits if the criterion is not improved by more than
    min_impurity_decrease. These techniques that date back to the work of [1]_,
    should lead to finite sample performance improvements, especially for
    high dimensional features.

    The implementation also provides confidence intervals
    for each prediction using a bootstrap of little bags approach described in [3]_:
    subsampling is performed at hierarchical level by first drawing a set of half-samples
    at random and then sub-sampling from each half-sample to build a forest
    of forests. All the trees are used for the point prediction and the distribution
    of predictions returned by each of the sub-forests is used to calculate the standard error
    of the point prediction.

    In particular we use a variant of the standard error estimation approach proposed in [4]_,
    where, if :math:`\\theta(X)` is the point prediction at X, then the variance of :math:`\\theta(X)`
    is computed as:

    .. math ::
        Var(\\theta(X)) = \\frac{\\hat{V}}{\\left(\\frac{1}{B} \\sum_{b \\in [B], i\\in [n]} w_{b, i}(x)\\right)^2}

    where B is the number of trees, n the number of training points, and:

    .. math ::
        w_{b, i}(x) = \\text{sample\\_weight}[i] \\cdot \\frac{1\\{i \\in \\text{leaf}(x; b)\\}}{|\\text{leaf}(x; b)|}

    .. math ::
        \\hat{V} = \\text{Var}_{\\text{random half-samples } S}\\left[ \\frac{1}{B_S}\
            \\sum_{b\\in S, i\\in [n]} w_{b, i}(x) (Y_i - \\theta(X)) \\right]

    where :math:`B_S` is the number of trees in half sample S. The latter variance is approximated by:

    .. math ::
        \\hat{V} = \\frac{1}{|\\text{drawn half samples } S|} \\sum_{S} \\left( \\frac{1}{B_S}\
            \\sum_{b\\in S, i\\in [n]} w_{b, i}(x) (Y_i - \\theta(X)) \\right)^2

    This variance calculation does not contain the correction due to finite number of monte carlo half-samples
    used (as proposed in [4]_), hence can be a bit conservative when a small number of half samples is used.
    However, it is on the conservative side. We use ceil(sqrt(n_estimators)) half samples, and the forest associated
    with each such half-sample contains roughly sqrt(n_estimators) trees, amounting to a total of n_estimator trees
    overall.

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

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    subsample_fr_ : float
        The chosen subsample ratio. Eache tree was trained on ``subsample_fr_ * n_samples / 2``
        data points.

    Examples
    --------

    .. testcode::

        import numpy as np
        from econml.sklearn_extensions.ensemble import SubsampledHonestForest
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        np.random.seed(123)
        X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                               random_state=0, shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
        regr = SubsampledHonestForest(max_depth=None, random_state=0,
                                      n_estimators=1000)

    >>> regr.fit(X_train, y_train)
    SubsampledHonestForest(n_estimators=1000, random_state=0)
    >>> regr.feature_importances_
    array([0.64..., 0.33..., 0.01..., 0.01...])
    >>> regr.predict(np.ones((1, 4)))
    array([112.9...])
    >>> regr.predict_interval(np.ones((1, 4)), alpha=.05)
    (array([94.9...]), array([130.9...]))
    >>> regr.score(X_test, y_test)
    0.94...

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
    [1]_, whereas the former was more recently justified empirically in [2]_.

    References
    ----------

    .. [1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [2] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized
           trees", Machine Learning, 63(1), 3-42, 2006.

    .. [3] S. Athey, S. Wager, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests",
            Journal of the American Statistical Association 113.523 (2018): 1228-1242.

    .. [4] S. Athey, J. Tibshirani, and S. Wager, "Generalized random forests",
            The Annals of Statistics, 47(2), 1148-1178, 2019.

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
        super().__init__(
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
        self.random_state = random_state
        self.estimators_ = None
        self.vars_ = None
        self.subsample_fr_ = None

        return

    def fit(self, X, y, sample_weight=None, sample_var=None):
        """
        Fit the forest.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Input data.

        y : array, shape (n_samples, n_outputs)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample. Weights will not be normalized. The weighted square loss
            will be minimized by the forest.

        sample_var : numpy array of shape [n_samples, n_outputs]
            Variance of composite samples (not used here. Exists for API compatibility)

        Returns
        -------
        self
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
            self.subsample_fr_ = (
                X.shape[0] / 2)**(1 - 1 / (2 * X.shape[1] + 2)) / (X.shape[0] / 2)
        else:
            self.subsample_fr_ = self.subsample_fr

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.numerators_ = []
            self.denominators_ = []

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
                half_sample_inds = random_state.choice(
                    X.shape[0], X.shape[0] // 2, replace=False)
                for _ in np.arange(it * self.slice_len, min((it + 1) * self.slice_len, self.n_estimators)):
                    s_inds.append(half_sample_inds[random_state.choice(X.shape[0] // 2,
                                                                       int(np.ceil(self.subsample_fr_ *
                                                                                   (X.shape[0] // 2))),
                                                                       replace=False)])
            res = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                           **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_add_trees)(
                    t, self, X, y, sample_weight, s_inds[i], i, len(trees),
                    verbose=self.verbose)
                for i, t in enumerate(trees))
            trees, numerators, denominators = zip(*res)
            # Collect newly grown trees
            self.estimators_.extend(trees)
            self.numerators_.extend(numerators)
            self.denominators_.extend(denominators)

        return self

    def _mean_fn(self, X, fn, acc, slice=None):
        # Helper class that accumulates an arbitrary function in parallel on the accumulator acc
        # and calls the function fn on each tree e and returns the mean output. The function fn
        # should take as input a tree e and associated numerator n and denominator d structures and
        # return another function g_e, which takes as input X, check_input
        # If slice is not None, but rather a tuple (start, end), then a subset of the trees from
        # index start to index end will be used. The returned result is essentially:
        # (mean over e in slice)(g_e(X)).
        check_is_fitted(self, 'estimators_')
        # Check data
        X = self._validate_X_predict(X)

        if slice is None:
            estimator_slice = zip(self.estimators_, self.numerators_, self.denominators_)
            n_estimators = len(self.estimators_)
        else:
            estimator_slice = zip(self.estimators_[slice[0]:slice[1]], self.numerators_[slice[0]:slice[1]],
                                  self.denominators_[slice[0]:slice[1]])
            n_estimators = slice[1] - slice[0]

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(n_estimators, self.n_jobs)
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose,
                 **_joblib_parallel_args(require="sharedmem"))(
            delayed(_accumulate_prediction)(fn(e, n, d), X, [acc], lock)
            for e, n, d in estimator_slice)
        acc /= n_estimators
        return acc

    def _weight(self, X, slice=None):
        """
        Returns the cummulative sum of training data weights for each of a target set of samples

        Parameters
        ----------
        X : (n, d_x) array
            The target samples
        slice : tuple(int, int) or None
            (start, end) tree index of the slice to be used

        Returns
        -------
        W : (n,) array
            For each sample x in X, it returns the quantity:
            1/B_S \\sum_{b \\in S} \\sum_{i\\in [n]} sample\\_weight[i] * 1{i \\in leaf(x; b)} / |leaf(x; b)|.
            where S is the slice of estimators chosen. If slice is None, then all estimators are used else
            the slice start:end is used.
        """
        # Check data
        X = self._validate_X_predict(X)
        weight_hat = np.zeros((X.shape[0]), dtype=np.float64)
        return self._mean_fn(X, lambda e, n, d: (lambda x, check_input: d[e.apply(x)]),
                             weight_hat, slice=slice)

    def _predict(self, X, slice=None):
        """Construct un-normalized numerator of the prediction for taret X, which when divided by weights
        creates the point prediction. Allows for subselecting the set of trees to use.

        The predicted regression unnormalized target of an input sample is computed as the
        mean predicted regression unnormalized targets of the trees in the forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
        slice : tuple(int, int) or None
            (start, end) tree index of the slice to be used
        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted values based on the subset of estimators in the slice start:end
            (all estimators, if slice=None). This is equivalent to:
            1/B_S \\sum_{b\\in S} \\sum_{i\\in [n]} sample_weight[i] * 1{i \\in leaf(x; b)} * Y_i / |leaf(x; b)|
        """
        # Check data
        X = self._validate_X_predict(X)
        # avoid storing the output of every estimator by summing them here
        y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        y_hat = self._mean_fn(X, lambda e, n, d: (lambda x, check_input: n[e.apply(x), :, 0]), y_hat, slice=slice)
        if self.n_outputs_ == 1:
            y_hat = y_hat.flatten()

        return y_hat

    def _inference(self, X, stderr=False):
        """
        Returns the point prediction for a set of samples X and if stderr=True, also returns stderr of the prediction.
        For standard error calculation it implements the bootstrap of little bags approach proposed
        in the GRF paper for estimating variance of estimate, specialized to this setup.

        .. math ::
            Var(\\theta(X)) = \\frac{V_hat}{(1/B \\sum_{b \\in [B], i\\in [n]} w_{b, i}(x))^2}

        where B is the number of trees, n the number of training points,

        .. math ::
            w_{b, i}(x) = sample\\_weight[i] \\cdot 1{i \\in leaf(x; b)} / |leaf(x; b)|

        .. math ::
            V_hat = Var_{random half-samples S}[ 1/B_S \\sum_{b\\in S, i\\in [n]} w_{b, i}(x) (Y_i - \\theta(X)) ]
                  = E_S[(1/B_S \\sum_{b\\in S, i\\in [n]} w_{b, i}(x) (Y_i - \\theta(X)))^2]

        where B_S is the number of trees in half sample S. This variance calculation does not contain the
        correction due to finite number of monte carlo half-samples used, hence can be a bit conservative
        when a small number of half samples is used. However, it is on the conservative side.

        Parameters
        ----------
        X : (n, d_x) array
            The target samples
        stderr : bool, optional (default=2)
            Whether to return stderr information for each prediction

        Returns
        -------
        Y_pred : (n,) or (n, d_y) array
            For each sample x in X, it returns the point prediction
        stderr : (n,) or (n, d_y) array
            The standard error for each prediction. Returned only if stderr=True.
        """
        y_pred = self._predict(X)  # get 1/B \sum_{b, i} w_{b, i}(x) Y_i
        weight_hat = self._weight(X)  # get 1/B \sum_{b, i} w_{b, i}(x)
        if len(y_pred.shape) > 1:
            weight_hat = weight_hat[:, np.newaxis]

        y_point_pred = y_pred / weight_hat  # point prediction: \sum_{b, i} w_{b, i} Y_i / \sum_{b, i} w_{b, i}

        if stderr:
            def slice_inds(it):
                return (it * self.slice_len, min((it + 1) * self.slice_len, self.n_estimators))
            # Calculate for each slice S: 1/B_S \sum_{b\in S, i\in [n]} w_{b, i}(x) Y_i
            y_bags_pred = np.array([self._predict(X, slice=slice_inds(it))
                                    for it in range(self.n_slices)])
            # Calculate for each slice S: 1/B_S \sum_{b\in S, i\in [n]} w_{b, i}(x)
            weight_hat_bags = np.array([self._weight(X, slice=slice_inds(it))
                                        for it in range(self.n_slices)])
            if np.ndim(y_bags_pred) > 2:
                weight_hat_bags = weight_hat_bags[:, :, np.newaxis]

            # Calculate for each slice S: Q(S) = 1/B_S \sum_{b\in S, i\in [n]} w_{b, i}(x) (Y_i - \theta(X))
            # where \theta(X) is the point estimate using the whole forest
            bag_res = y_bags_pred - weight_hat_bags * \
                np.expand_dims(y_point_pred, axis=0)
            # Calculate the variance of the latter as E[Q(S)^2]
            std_pred = np.sqrt(np.nanmean(bag_res**2, axis=0)) / weight_hat

            return y_point_pred, std_pred

        return y_point_pred

    def predict(self, X):
        """
        Returns point prediction.

        Parameters
        ----------
        X : (n, d_x) array
            Features

        Returns
        -------
        y_pred : (n,) or (n, d_y) array
            Point predictions
        """
        return self._inference(X)

    def prediction_stderr(self, X):
        """
        Returns the standard deviation of the point prediction.

        Parameters
        ----------
        X : (n, d_x) array
            Features

        Returns
        -------
        pred_stderr : (n,) or (n, d_y) array
            The standard error for each point prediction
        """
        _, pred_stderr = self._inference(X, stderr=True)
        return pred_stderr

    def predict_interval(self, X, alpha=.1, normal=True):
        """
        Return the confidence interval of the prediction.

        Parameters
        ----------
        X : (n, d_x) array
            Features
        alpha : float
            The significance level of the interval

        Returns
        -------
        lb, ub : tuple(shape of :meth:`predict(X)<predict>`, shape of :meth:`predict(X)<predict>`)
            The lower and upper bound of an alpha-confidence interval for each prediction
        """
        y_point_pred, pred_stderr = self._inference(X, stderr=True)
        upper_pred = scipy.stats.norm.ppf(
            1 - alpha / 2, loc=y_point_pred, scale=pred_stderr)
        lower_pred = scipy.stats.norm.ppf(
            alpha / 2, loc=y_point_pred, scale=pred_stderr)
        return lower_pred, upper_pred
