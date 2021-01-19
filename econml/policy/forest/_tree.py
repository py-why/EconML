# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code contains snippets of code from:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_classes.py
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

import numpy as np
import numbers
from math import ceil
from ...tree import Tree
from ...tree._criterion import Criterion
from ...tree._splitter import Splitter, BestSplitter
from ...tree import DepthFirstTreeBuilder
from ...tree import _tree
from ..._tree_exporter import _SingleTreeExporterMixin, _PolicyTreeDOTExporter, _PolicyTreeMPLExporter
from ._criterion import LinearPolicyCriterion
from . import _criterion
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_is_fitted
import copy

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_POLICY = {"neg_welfare": LinearPolicyCriterion}

SPLITTERS = {"best": BestSplitter, }

MAX_INT = np.iinfo(np.int32).max

# =============================================================================
# Base Policy tree
# =============================================================================


class PolicyTree(_SingleTreeExporterMixin, BaseEstimator):
    """ TODO Enable inference on `predict_value` with leaf-wise normality

    Parameters
    ----------
    criterion : {``'neg_welfare'``}, default='neg_welfare'
        The criterion type

    splitter : {"best"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=10
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=5
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
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

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    min_balancedness_tol: float in [0, .5], default=.45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest: bool, default=True
        Whether the data should be split in two equally sized samples, such that the one half-sample
        is used to determine the optimal split at each node and the other sample is used to determine
        the value of every node.

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances based on the amount of parameter heterogeneity they create.
        The higher, the more important the feature.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    n_samples_ : int
        The number of training samples when ``fit`` is performed.

    honest_ : int
        Whether honesty was enabled when ``fit`` was performed

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(econml.tree._tree.Tree)`` for attributes of Tree object.

    """

    def __init__(self, *,
                 criterion='neg_welfare',
                 splitter="best",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 min_impurity_decrease=0.,
                 min_balancedness_tol=0.45,
                 honest=True):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest

    def get_depth(self):
        """Return the depth of the decision tree.
        The depth of a tree is the maximum distance between the root
        and any leaf.

        Returns
        -------
        self.tree_.max_depth : int
            The maximum depth of the tree.
        """
        check_is_fitted(self)
        return self.tree_.max_depth

    def get_n_leaves(self):
        """Return the number of leaves of the decision tree.

        Returns
        -------
        self.tree_.n_leaves : int
            Number of leaves.
        """
        check_is_fitted(self)
        return self.tree_.n_leaves

    def init(self,):
        return self

    def fit(self, X, y, *, sample_weight=None, check_input=True):
        """ Fit the tree from the data

        Parameters
        ----------
        X : (n, d) array
            The features to split on

        y : (n, m) array
            All the variables required to calculate the criterion function, evaluate splits and
            estimate local values, i.e. all the values that go into the moment function except X.

        sample_weight : (n,) array, default=None
            The sample weights

        check_input : bool, defaul=True
            Whether to check the input parameters for validity. Should be set to False to improve
            running time in parallel execution, if the variables have already been checked by the
            forest class that spawned this tree.
        """

        self.random_seed_ = self.random_state
        random_state = check_random_state(self.random_seed_)

        # Determine output settings
        if check_input:
            X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        n_samples, self.n_features_ = X.shape
        self.n_outputs_ = 1 if y.ndim == 1 else y.shape[1]
        self.n_samples_ = n_samples
        self.honest_ = self.honest

        # Important: This must be the first invocation of the random state at fit time, so that
        # train/test splits are re-generatable from an external object simply by knowing the
        # random_state parameter of the tree.
        inds = np.arange(n_samples, dtype=np.intp)
        if self.honest:
            random_state.shuffle(inds)
            samples_train, samples_val = inds[:n_samples // 2], inds[n_samples // 2:]
        else:
            samples_train, samples_val = inds, inds

        if check_input:
            if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
                y = np.ascontiguousarray(y, dtype=DOUBLE)
            y = np.atleast_1d(y)
            if y.ndim == 1:
                # reshape is necessary to preserve the data contiguity against vs
                # [:, np.newaxis] that does not.
                y = np.reshape(y, (-1, 1))
            if len(y) != n_samples:
                raise ValueError("Number of labels=%d does not match "
                                 "number of samples=%d" % (len(y), n_samples))

            if (sample_weight is not None):
                sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

        # Check parameters
        max_depth = (np.iinfo(np.int32).max if self.max_depth is None
                     else self.max_depth)

        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = self.min_samples_leaf
        else:  # float
            if not 0. < self.min_samples_leaf <= 0.5:
                raise ValueError("min_samples_leaf must be at least 1 "
                                 "or in (0, 0.5], got %s"
                                 % self.min_samples_leaf)
            min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the integer %s"
                                 % self.min_samples_split)
            min_samples_split = self.min_samples_split
        else:  # float
            if not 0. < self.min_samples_split <= 1.:
                raise ValueError("min_samples_split must be an integer "
                                 "greater than 1 or a float in (0.0, 1.0]; "
                                 "got the float %s"
                                 % self.min_samples_split)
            min_samples_split = int(ceil(self.min_samples_split * n_samples))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                max_features = self.n_features_
            elif self.max_features == "sqrt":
                max_features = max(1, int(np.sqrt(self.n_features_)))
            elif self.max_features == "log2":
                max_features = max(1, int(np.log2(self.n_features_)))
            else:
                raise ValueError("Invalid value for max_features. "
                                 "Allowed string values are 'auto', "
                                 "'sqrt' or 'log2'.")
        elif self.max_features is None:
            max_features = self.n_features_
        elif isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        else:  # float
            if self.max_features > 0.0:
                max_features = max(1,
                                   int(self.max_features * self.n_features_))
            else:
                max_features = 0

        self.max_features_ = max_features

        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if max_depth < 0:
            raise ValueError("max_depth must be greater than or equal to zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not 0 <= self.min_balancedness_tol <= 0.5:
            raise ValueError("min_balancedness_tol must be in [0, 0.5]")

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        # Build tree

        # We calculate the maximum number of samples from each half-split that any node in the tree can
        # hold. Used by criterion for memory space savings.
        max_train = len(samples_train) if sample_weight is None else np.count_nonzero(sample_weight[samples_train])
        if self.honest:
            max_val = len(samples_val) if sample_weight is None else np.count_nonzero(sample_weight[samples_val])
        # Initialize the criterion object and the criterion_val object if honest.
        if self.criterion == 'neg_welfare':
            criterion = CRITERIA_POLICY['neg_welfare'](
                self.n_outputs_, self.n_outputs_, self.n_features_, self.n_outputs_, n_samples, max_train,
                random_state.randint(np.iinfo(np.int32).max))
            if self.honest:
                criterion_val = CRITERIA_POLICY['neg_welfare'](
                    self.n_outputs_, self.n_outputs_, self.n_features_, self.n_outputs_, n_samples, max_val,
                    random_state.randint(np.iinfo(np.int32).max))
            else:
                criterion_val = criterion
        else:
            raise ValueError("Only `criterion='neg_welfare'` is currently supported.")

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion, criterion_val,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                self.min_balancedness_tol,
                                                self.honest,
                                                -1.0,
                                                False,
                                                random_state.randint(np.iinfo(np.int32).max))

        self.tree_ = Tree(self.n_features_, self.n_outputs_, self.n_outputs_, store_jac=False)

        builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        max_depth,
                                        self.min_impurity_decrease)
        builder.build(self.tree_, X, y, samples_train, samples_val,
                      sample_weight=sample_weight,
                      store_jac=False)

        self.tree_model_ = self
        self.policy_value_ = np.mean(np.max(self.predict_value(X), axis=1))
        self.always_treat_value_ = np.mean(y, axis=0)
        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, or any other of the prediction
        related methods. """
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=False)

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X

    def get_train_test_split_inds(self,):
        """ Regenerate the train_test_split of input sample indices that was used for the training
        and the evaluation split of the honest tree construction structure. Uses the same random seed
        that was used at ``fit`` time and re-generates the indices.
        """
        check_is_fitted(self)
        random_state = check_random_state(self.random_seed_)
        inds = np.arange(self.n_samples_, dtype=np.intp)
        if self.honest_:
            random_state.shuffle(inds)
            return inds[:self.n_samples_ // 2], inds[self.n_samples_ // 2:]
        else:
            return inds, inds

    def predict(self, X, check_input=True):
        """

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        return np.argmax(pred, axis=1)

    def predict_value(self, X, check_input=True):
        """

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        return pred

    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        X_leaves : array-like of shape (n_samples,)
            For each datapoint x in X, return the index of the leaf x
            ends up in. Leaves are numbered within
            ``[0; self.tree_.node_count)``, possibly with gaps in the
            numbering.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.apply(X)

    def decision_path(self, X, check_input=True):
        """Return the decision path in the tree.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        indicator : sparse matrix of shape (n_samples, n_nodes)
            Return a node indicator CSR matrix where non zero elements
            indicates that the samples goes through the nodes.
        """
        X = self._validate_X_predict(X, check_input)
        return self.tree_.decision_path(X)

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        """

        Parameters
        ----------
        max_depth : int, default=4
            Splits of depth larger than `max_depth` are not used in this calculation
        depth_decay_exponent: double, default=2.0
            The contribution of each split to the total score is re-weighted by ``1 / (1 + `depth`)**2.0``.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total parameter heterogeneity inducing importance of each feature
        """
        check_is_fitted(self)

        return self.tree_.compute_feature_importances(normalize=True, max_depth=max_depth,
                                                      depth_decay=depth_decay_exponent)

    @property
    def feature_importances_(self):
        return self.feature_importances()

    def _make_dot_exporter(self, *, out_file, feature_names, treatment_names, max_depth, filled,
                           leaves_parallel, rotate, rounded,
                           special_characters, precision):
        title = "Average policy gains over no treatment: {} \n".format(np.around(self.policy_value_, precision))
        title += "Average policy gains over constant treatment policies for each treatment: {}".format(
            np.around(self.policy_value_ - self.always_treat_value_, precision))
        return _PolicyTreeDOTExporter(out_file=out_file, title=title,
                                      treatment_names=treatment_names, feature_names=feature_names,
                                      max_depth=max_depth,
                                      filled=filled, leaves_parallel=leaves_parallel, rotate=rotate,
                                      rounded=rounded, special_characters=special_characters,
                                      precision=precision)

    def _make_mpl_exporter(self, *, title, feature_names, treatment_names, max_depth, filled,
                           rounded, precision, fontsize):
        title = "" if title is None else title
        title += "Average policy gains over no treatment: {} \n".format(np.around(self.policy_value_, precision))
        title += "Average policy gains over constant treatment policies for each treatment: {}".format(
            np.around(self.policy_value_ - self.always_treat_value_, precision))
        return _PolicyTreeMPLExporter(treatment_names=treatment_names, title=title,
                                      feature_names=feature_names, max_depth=max_depth,
                                      filled=filled,
                                      rounded=rounded,
                                      precision=precision, fontsize=fontsize)
