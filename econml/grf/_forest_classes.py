import numbers
from warnings import catch_warnings, simplefilter, warn
from abc import ABCMeta, abstractmethod
import numpy as np
import threading
from ._ensemble import BaseEnsemble, _partition_estimators
from ..utilities import check_inputs, cross_product
from ..tree._tree import DTYPE, DOUBLE
from ._tree_classes import GRFTree
from joblib import Parallel, delayed
from scipy.sparse import hstack as sparse_hstack
from sklearn.utils import check_random_state, compute_sample_weight
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

MAX_INT = np.iinfo(np.int32).max


def _get_n_samples_subsample(n_samples, max_samples):
    """
    Get the number of samples in a bootstrap sample.
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
    n_samples_bootstrap : int
        The total number of samples to draw for the bootstrap sample.
    """
    if max_samples is None:
        return n_samples

    if isinstance(max_samples, numbers.Integral):
        if not (1 <= max_samples <= n_samples):
            msg = "`max_samples` must be in range 1 to {} but got value {}"
            raise ValueError(msg.format(n_samples, max_samples))
        return max_samples

    if isinstance(max_samples, numbers.Real):
        if not (0 < max_samples < 1):
            msg = "`max_samples` must be in range (0, 1) but got value {}"
            raise ValueError(msg.format(max_samples))
        return int(round(n_samples * max_samples))

    msg = "`max_samples` should be int or float, but got type '{}'"
    raise TypeError(msg.format(type(max_samples)))


def _generate_sample_indices(random_state, n_samples, n_samples_subsample):
    """
    Private function used to _parallel_build_trees function."""

    random_instance = check_random_state(random_state)
    sample_indices = random_instance.choice(n_samples, n_samples_subsample, replace=False)

    return sample_indices


def _parallel_build_trees(tree, forest, X, yaug, n_y, n_outputs, n_relevant_outputs,
                          sample_weight, tree_idx, n_trees,
                          verbose=0, n_samples_subsample=None):
    """
    Private function used to fit a single tree in parallel."""
    if verbose > 1:
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    n_samples = X.shape[0]
    if sample_weight is None:
        curr_sample_weight = np.ones((n_samples,), dtype=np.float64)
    else:
        curr_sample_weight = sample_weight.copy()

    indices = _generate_sample_indices(tree.random_state, n_samples, n_samples_subsample)
    sample_counts = np.zeros(n_samples)
    sample_counts[indices] = 1.0
    curr_sample_weight *= sample_counts

    tree.fit(X, yaug, n_y, n_outputs, n_relevant_outputs, sample_weight=curr_sample_weight, check_input=False)

    return tree


def _accumulate_prediction(predict, X, out, lock):
    """
    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    with lock:
        if len(out) == 1:
            out[0] += prediction
        else:
            for i in range(len(out)):
                out[i] += prediction[i]


# =============================================================================
# Base Generalized Random Forest
# =============================================================================

class BaseGRF(BaseEnsemble, metaclass=ABCMeta):
    """
    Base class for forests of CATE estimator trees.
    Warning: This class should not be used directly. Use derived classes
    instead.
    """

    def __init__(self,
                 n_estimators=100, *,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.9,
                 min_balancedness_tol=.3,
                 honest=True,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(
            base_estimator=GRFTree(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "min_impurity_decrease", "honest",
                              "min_balancedness_tol",
                              "random_state"))

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.max_samples = max_samples

    def get_alpha(self, X, T, y, **kwargs):
        pass

    def get_pointJ(self, X, T, y, **kwargs):
        pass

    def apply(self, X):
        """
        Apply trees in the forest to X, return leaf indices.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
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
        .. versionadded:: 0.18
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.
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

    def fit(self, X, T, y, *, sample_weight=None, **kwargs):
        """
        Build a forest of trees from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csc_matrix``.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. In the case of
            classification, splits are also ignored if they would result in any
            single class carrying a negative weight in either child node.
        Returns
        -------
        self : object
        """

        y, T, X = check_inputs(y, T, X, multi_output_T=True, multi_output_Y=False)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().", stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_y_ = y.shape[1]

        T = np.atleast_1d(T)
        if T.ndim == 2 and T.shape[1] == 1:
            warn("A column-vector T was passed when a 1d array was"
                 " expected. Please change the shape of T to "
                 "(n_samples,), for example using ravel().", stacklevel=2)

        if T.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            T = np.reshape(T, (-1, 1))

        self.n_outputs_ = T.shape[1] + 1
        self.n_relevant_outputs_ = T.shape[1]

        Taug = np.hstack([T, np.ones((T.shape[0], 1))])
        alpha = self.get_alpha(X, Taug, y, **kwargs)
        pointJ = self.get_pointJ(X, Taug, y, **kwargs)
        yaug = np.hstack([y, alpha, pointJ])

        if getattr(yaug, "dtype", None) != DOUBLE or not yaug.flags.contiguous:
            yaug = np.ascontiguousarray(yaug, dtype=DOUBLE)

        if getattr(X, "dtype", None) != DTYPE or not X.flags.c_contiguous:
            X = np.asfortranarray(X, dtype=DTYPE)

        # Get bootstrap sample size
        n_samples_subsample = _get_n_samples_subsample(
            n_samples=X.shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

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
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='threading')(
                delayed(_parallel_build_trees)(
                    t, self, X, yaug, self.n_y_, self.n_outputs_, self.n_relevant_outputs_,
                    sample_weight, i, len(trees),
                    verbose=self.verbose, n_samples_subsample=n_samples_subsample)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        return self

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    @property
    def feature_importances_(self):
        """
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(getattr)(tree, 'feature_importances_')
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    @property
    def feature_heterogeneity_importances_(self):
        """
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            The values of this array sum to 1, unless all trees are single node
            trees consisting of only the root node, in which case it will be an
            array of zeros.
        """
        check_is_fitted(self)

        all_importances = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(getattr)(tree, 'feature_heterogeneity_importances_')
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    def predict_tree_average(self, X):

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_relevant_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading', require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict, X, [y_hat], lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat

    def predict(self, X):

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        jac_hat = np.zeros((X.shape[0], self.n_outputs_**2), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading', require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_jac, X, [jac_hat], lock)
            for e in self.estimators_)

        jac_hat /= len(self.estimators_)

        alpha_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading', require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_alpha, X, [alpha_hat], lock)
            for e in self.estimators_)

        alpha_hat /= len(self.estimators_)

        y_hat = np.einsum('ijk,ik->ij',
                          np.linalg.pinv(jac_hat.reshape((-1, self.n_outputs_, self.n_outputs_))),
                          alpha_hat)

        return y_hat[:, :self.n_relevant_outputs_]

# =============================================================================
# Instantiations of Generalized Random Forest
# =============================================================================


class CausalForest(BaseGRF):

    def get_alpha(self, X, T, y):
        return y * T

    def get_pointJ(self, X, T, y):
        return cross_product(T, T)


class CausalIVForest(BaseGRF):

    def get_alpha(self, X, T, y, *, Z):
        Z = np.atleast_1d(Z)
        if Z.ndim == 2 and Z.shape[1] == 1:
            warn("A column-vector Z was passed when a 1d array was"
                 " expected. Please change the shape of Z to "
                 "(n_samples,), for example using ravel().", stacklevel=2)

        if Z.ndim == 1:
            Z = np.reshape(Z, (-1, 1))

        return y * np.hstack([Z, np.ones((Z.shape[0], 1))])

    def get_pointJ(self, X, T, y, *, Z):
        if Z.ndim == 1:
            Z = np.reshape(Z, (-1, 1))
        return cross_product(np.hstack([Z, np.ones((Z.shape[0], 1))]), T)
