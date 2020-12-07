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
import scipy.stats

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
                 min_balancedness_tol=.45,
                 honest=True,
                 inference=True,
                 fit_intercept=True,
                 n_subforests='auto',
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
        self.inference = inference
        self.fit_intercept = fit_intercept
        self.n_subforests = n_subforests
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

        y, T, X = check_inputs(y, T, X, multi_output_T=True, multi_output_Y=True)

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        # Remap output
        n_samples, self.n_features_ = X.shape

        y = np.atleast_1d(y)
        if y.ndim == 1:
            warn("A 1d vector y was passed when a 2d column-vector was"
                 " expected. Please change the shape of y to "
                 "(n_samples, 1). It will be treated as such.", stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_y_ = y.shape[1]

        T = np.atleast_1d(T)
        if T.ndim == 1:
            warn("A 1d vector T was passed when a 2d column-vector was"
                 " expected. Please change the shape of T to "
                 "(n_samples, 1). It will be treated as such.", stacklevel=2)

        if T.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            T = np.reshape(T, (-1, 1))

        self.n_relevant_outputs_ = T.shape[1]
        if self.fit_intercept:
            Taug = np.hstack([T, np.ones((T.shape[0], 1))])
            self.n_outputs_ = T.shape[1] + 1
        else:
            Taug = T
            self.n_outputs_ = T.shape[1]

        alpha = self.get_alpha(X, Taug, y, **kwargs)
        pointJ = self.get_pointJ(X, Taug, y, **kwargs)
        yaug = np.hstack([y, alpha, pointJ])

        if getattr(yaug, "dtype", None) != DOUBLE or not yaug.flags.contiguous:
            yaug = np.ascontiguousarray(yaug, dtype=DOUBLE)

        if getattr(X, "dtype", None) != DTYPE:
            X = X.astype(DTYPE)

        # Get bootstrap sample size
        n_samples_subsample = _get_n_samples_subsample(
            n_samples=n_samples,
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []
            self.slices_ = []
            self.halfsample_seeds_ = []

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

            if self.inference:
                # Generating indices a priori before parallelism ended up being orders of magnitude
                # faster than how sklearn does it. The reason is that random samplers do not release the
                # gil it seems.
                if self.n_subforests == 'auto':
                    n_subforests = int(np.ceil(n_more_estimators**(1 / 4)))
                elif isinstance(self.n_subforests, numbers.Integral):
                    n_subforests = self.n_subforests
                else:
                    raise ValueError("Parameter n_subforests must be an integer or 'auto'.")
                new_slices = np.array_split(np.arange(len(self.estimators_),
                                                      len(self.estimators_) + n_more_estimators),
                                            n_subforests)
                s_inds = []
                for sl in new_slices:
                    half_sample_inds = random_state.choice(n_samples, n_samples // 2, replace=False)
                    s_inds.extend([half_sample_inds[random_state.choice(n_samples // 2,
                                                                        n_samples_subsample // 2,
                                                                        replace=False)]
                                   for _ in range(len(sl))])
            else:
                new_slices = []
                s_inds = [random_state.choice(n_samples, n_samples_subsample, replace=False)
                          for _ in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose, backend='threading')(
                delayed(t.fit)(X[s], yaug[s], self.n_y_, self.n_outputs_, self.n_relevant_outputs_,
                               sample_weight[s] if sample_weight is not None else None)
                for t, s in zip(trees, s_inds))

            # Collect newly grown trees
            self.estimators_.extend(trees)
            self.slices_.extend(list(new_slices))

        return self

    def _validate_X_predict(self, X):
        """
        Validate X whenever one tries to predict, apply, predict_proba."""
        check_is_fitted(self)

        return self.estimators_[0]._validate_X_predict(X, check_input=True)

    def feature_importances(self, max_depth=None, depth_decay_exponent=.0):
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
            delayed(tree.feature_importances)(max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    def feature_heterogeneity_importances(self, max_depth=None, depth_decay_exponent=.0):
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
            delayed(tree.feature_heterogeneity_importances)(
                max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
            for tree in self.estimators_ if tree.tree_.node_count > 1)

        if not all_importances:
            return np.zeros(self.n_features_, dtype=np.float64)

        all_importances = np.mean(all_importances,
                                  axis=0, dtype=np.float64)
        return all_importances / np.sum(all_importances)

    def predict_tree_average_full(self, X):

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # avoid storing the output of every estimator by summing them here
        if self.n_outputs_ > 1:
            y_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        else:
            y_hat = np.zeros((X.shape[0]), dtype=np.float64)

        # Parallel loop
        lock = threading.Lock()
        Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading', require="sharedmem")(
            delayed(_accumulate_prediction)(e.predict_full, X, [y_hat], lock)
            for e in self.estimators_)

        y_hat /= len(self.estimators_)

        return y_hat

    def predict_tree_average(self, X):
        y_hat = self.predict_tree_average_full(X)
        if self.n_relevant_outputs_ == self.n_outputs_:
            return y_hat
        return y_hat[:, :self.n_relevant_outputs_]

    def predict_jac(self, X, slice=None, parallel=True):
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        if slice is None:
            slice = np.arange(len(self.estimators_))

        # avoid storing the output of every estimator by summing them here
        jac_hat = np.zeros((X.shape[0], self.n_outputs_**2), dtype=np.float64)
        lock = threading.Lock()
        if parallel:
            n_jobs, _, _ = _partition_estimators(len(slice), self.n_jobs)
            verbose = self.verbose
            # Parallel loop
            Parallel(n_jobs=n_jobs, verbose=verbose, backend='threading', require="sharedmem")(
                delayed(_accumulate_prediction)(self.estimators_[t].predict_jac, X, [jac_hat], lock)
                for t in slice)
        else:
            [_accumulate_prediction(self.estimators_[t].predict_jac, X, [jac_hat], lock)
             for t in slice]

        jac_hat /= len(slice)

        return jac_hat.reshape((-1, self.n_outputs_, self.n_outputs_))

    def predict_alpha(self, X, slice=None, parallel=True):

        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        if slice is None:
            slice = np.arange(len(self.estimators_))

        alpha_hat = np.zeros((X.shape[0], self.n_outputs_), dtype=np.float64)
        lock = threading.Lock()
        if parallel:
            n_jobs, _, _ = _partition_estimators(len(slice), self.n_jobs)
            verbose = self.verbose
            # Parallel loop
            Parallel(n_jobs=n_jobs, verbose=verbose, backend='threading', require="sharedmem")(
                delayed(_accumulate_prediction)(self.estimators_[t].predict_alpha, X, [alpha_hat], lock)
                for t in slice)
        else:
            [_accumulate_prediction(self.estimators_[t].predict_alpha, X, [alpha_hat], lock)
             for t in slice]

        alpha_hat /= len(slice)

        return alpha_hat

    def predict_moment_var(self, X, parameter, slice=None, parallel=True):
        check_is_fitted(self)
        # Check data
        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        if slice is None:
            slice = np.arange(len(self.estimators_))

        moment_var_hat = np.zeros((X.shape[0], self.n_outputs_, self.n_outputs_), dtype=np.float64)
        lock = threading.Lock()
        if parallel:
            n_jobs, _, _ = _partition_estimators(len(slice), self.n_jobs)
            verbose = self.verbose
            # Parallel loop
            Parallel(n_jobs=n_jobs, verbose=verbose, backend='threading', require="sharedmem")(
                delayed(_accumulate_prediction_var)(self.estimators_[t].predict_moment, X, [moment_var_hat], lock,
                                                    parameter)
                for t in slice)
        else:
            [_accumulate_prediction_var(self.estimators_[t].predict_moment, X, [moment_var_hat], lock,
                                        parameter)
             for t in slice]

        moment_var_hat /= len(slice)

        return moment_var_hat

    def predict_alpha_and_jac(self, X, slice=None, parallel=True):

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

    def _predict_point_and_cov(self, X, full=False, point=True, cov=False, var_correction=False):
        n_outputs = self.n_outputs_ if full else self.n_relevant_outputs_
        alpha, jac = self.predict_alpha_and_jac(X)
        invjac = np.linalg.pinv(jac)
        parameter = np.einsum('ijk,ik->ij', invjac, alpha)
        moment = alpha - np.einsum('ijk,ik->ij', jac, parameter)

        if cov:
            if not self.inference:
                raise AttributeError("Inference not available. Forest was initiated with `inference=False`.")
            slices = self.slices_
            n_jobs, _, _ = _partition_estimators(len(slices), self.n_jobs)
            alpha_bags, jac_bags = zip(*Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading')(
                delayed(self.predict_alpha_and_jac)(X, slice=sl, parallel=False) for sl in slices))
            alpha_bags = np.array(alpha_bags)
            jac_bags = np.array(jac_bags)
            moment_bags = alpha_bags - np.einsum('tijk,ik->tij', jac_bags, parameter)
            centered_moment_bags = moment_bags - moment
            centered_moment_bags = np.moveaxis(centered_moment_bags, 0, -1)
            param_cov = np.einsum('tij,tjk->tik', centered_moment_bags,
                                  np.transpose(centered_moment_bags, (0, 2, 1))) / len(slices)
            pred_cov = np.einsum('ijk,ikm->ijm', invjac,
                                 np.einsum('ijk,ikm->ijm', param_cov, np.transpose(invjac, (0, 2, 1))))
            if var_correction:
                moment_var_bags = Parallel(n_jobs=n_jobs, verbose=self.verbose, backend='threading')(
                    delayed(self.predict_moment_var)(X, parameter, slice=sl, parallel=False) for sl in slices)
                moment_var_bags -= np.einsum('tijk,tikm->tijm',
                                             moment_bags.reshape(moment_bags.shape + (1,)),
                                             moment_bags.reshape(moment_bags.shape[:-1] + (1, moment_bags.shape[-1]))
                                             )
                correction = np.mean(moment_var_bags, axis=0) / (len(slices[0]) - 1)
                pred_cov -= correction
        if point and cov:
            return (parameter[:, :n_outputs],
                    pred_cov[:, :n_outputs, :n_outputs],)
        elif point:
            return parameter[:, :n_outputs]
        else:
            return pred_cov[:, :n_outputs, :n_outputs]

    def predict_full(self, X, interval=False, alpha=0.05, var_correction=False):
        if interval:
            point, pred_cov = self._predict_point_and_cov(X, full=True, point=True,
                                                          cov=True, var_correction=var_correction)
            lb, ub = np.zeros(point.shape), np.zeros(point.shape)
            for t in range(self.n_outputs_):
                lb[:, t] = scipy.stats.norm.ppf(alpha / 2, loc=point[:, t], scale=np.sqrt(pred_cov[:, t, t]))
                ub[:, t] = scipy.stats.norm.ppf(1 - alpha / 2, loc=point[:, t], scale=np.sqrt(pred_cov[:, t, t]))
            return point, lb, ub
        return self._predict_point_and_cov(X, full=True, point=True, cov=False)

    def predict(self, X, interval=False, alpha=0.05, var_correction=False):
        if interval:
            y_hat, lb, ub = self.predict_full(X, interval=interval, alpha=alpha, var_correction=var_correction)
            if self.n_relevant_outputs_ == self.n_outputs_:
                return y_hat, lb, ub
            return (y_hat[:, :self.n_relevant_outputs_],
                    lb[:, :self.n_relevant_outputs_], ub[:, :self.n_relevant_outputs_])
        else:
            y_hat = self.predict_full(X, interval=False)
            if self.n_relevant_outputs_ == self.n_outputs_:
                return y_hat
            return y_hat[:, :self.n_relevant_outputs_]

    def predict_cov(self, X, var_correction=False):
        return self._predict_point_and_cov(X, full=False, point=False, cov=True, var_correction=var_correction)

    def predict_interval(self, X, alpha=.05, var_correction=False):
        _, lb, ub = self.predict(X, interval=True, alpha=alpha, var_correction=var_correction)
        return lb, ub

    def prediction_stderr(self, X, var_correction=False):
        return np.diagonal(self.predict_cov(X, var_correction=var_correction), axis1=1, axis2=2)


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
        if Z.ndim == 1:
            warn("A 1d vector Z was passed when a 2d column-vector was"
                 " expected. Please change the shape of Z to "
                 "(n_samples, 1). It will be treated as such.", stacklevel=2)

        if Z.ndim == 1:
            Z = np.reshape(Z, (-1, 1))

        if self.fit_intercept:
            return y * np.hstack([Z, np.ones((Z.shape[0], 1))])
        return y * Z

    def get_pointJ(self, X, T, y, *, Z):
        if Z.ndim == 1:
            Z = np.reshape(Z, (-1, 1))
        if self.fit_intercept:
            return cross_product(np.hstack([Z, np.ones((Z.shape[0], 1))]), T)
        return cross_product(Z, T)
