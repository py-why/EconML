# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
#
# This code contains snippets of code from:
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_classes.py
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
from ...tree import BaseTree
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_X_y
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_is_fitted
import copy

# =============================================================================
# Types and constants
# =============================================================================

CRITERIA_POLICY = {"neg_welfare": LinearPolicyCriterion}

# =============================================================================
# Base Policy tree
# =============================================================================


class PolicyTree(_SingleTreeExporterMixin, BaseTree):
    """ Welfare maximization policy tree. Trains a tree to maximize the objective:
    :math:`1/n \\sum_i \\sum_j a_j(X_i) * y_{ij}`, where, where :math:`a(X)` is constrained
    to take value of 1 only on one coordinate and zero otherwise. This corresponds to a policy
    optimization problem.

    Parameters
    ----------
    criterion : {``'neg_welfare'``}, default 'neg_welfare'
        The criterion type

    splitter : {"best"}, default "best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split.

    max_depth : int, default None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default 10
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default 5
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default 0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"auto", "sqrt", "log2"}, or None, default None
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

    random_state : int, RandomState instance, or None, default None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    min_impurity_decrease : float, default 0.0
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

    min_balancedness_tol: float in [0, .5], default .45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest: bool, default True
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

    n_features_in_ : int
        The number of features when ``fit`` is performed.

    n_samples_ : int
        The number of training samples when ``fit`` is performed.

    honest_ : int
        Whether honesty was enabled when ``fit`` was performed

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(econml.tree._tree.Tree)`` for attributes of Tree object.

    policy_value_ : float
        The value achieved by the recommended policy

    always_treat_value_ : float
        The value of the policy that treats all samples

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
        super().__init__(criterion=criterion,
                         splitter=splitter,
                         max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features,
                         random_state=random_state,
                         min_impurity_decrease=min_impurity_decrease,
                         min_balancedness_tol=min_balancedness_tol,
                         honest=honest)

    def _get_valid_criteria(self):
        return CRITERIA_POLICY

    def _get_store_jac(self):
        return False

    def init(self,):
        return self

    def fit(self, X, y, *, sample_weight=None, check_input=True):
        """ Fit the tree from the data

        Parameters
        ----------
        X : (n, n_features) array
            The features to split on

        y : (n, n_treatments) array
            The reward for each of the m treatments (including baseline treatment)

        sample_weight : (n,) array, default None
            The sample weights

        check_input : bool, defaul=True
            Whether to check the input parameters for validity. Should be set to False to improve
            running time in parallel execution, if the variables have already been checked by the
            forest class that spawned this tree.

        Returns
        -------
        self : object instance
        """

        self.random_seed_ = self.random_state
        self.random_state_ = check_random_state(self.random_seed_)
        if check_input:
            X, y = check_X_y(X, y, multi_output=True, y_numeric=True, ensure_min_features=0)
        n_y = 1 if y.ndim == 1 else y.shape[1]
        super().fit(X, y, n_y, n_y, n_y,
                    sample_weight=sample_weight, check_input=check_input)

        # The values below are required and utilitized by methods in the _SingleTreeExporterMixin
        self.tree_model_ = self
        self.policy_value_ = np.mean(np.max(self.predict_value(X), axis=1))
        self.always_treat_value_ = np.mean(y, axis=0)
        return self

    def predict(self, X, check_input=True):
        """ Predict the best treatment for each sample

        Parameters
        ----------
        X : {array_like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        treatment : array_like of shape (n_samples)
            The recommded treatment, i.e. the treatment index with the largest reward for each sample
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        return np.argmax(pred, axis=1)

    def predict_proba(self, X, check_input=True):
        """ Predict the probability of recommending each treatment

        Parameters
        ----------
        X : {array_like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        treatment_proba : array_like of shape (n_samples, n_treatments)
            The probability of each treatment recommendation
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        proba = np.zeros(pred.shape)
        proba[np.arange(X.shape[0]), np.argmax(pred, axis=1)] = 1
        return proba

    def predict_value(self, X, check_input=True):
        """ Predict the expected value of each treatment for each sample

        Parameters
        ----------
        X : {array_like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``.
        check_input : bool, default True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        welfare : array_like of shape (n_samples, n_treatments)
            The conditional average welfare for each treatment for the group of each sample defined by the tree
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        return pred

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        """

        Parameters
        ----------
        max_depth : int, default 4
            Splits of depth larger than `max_depth` are not used in this calculation
        depth_decay_exponent: double, default 2.0
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


# HACK: sklearn 1.3 enforces that the input to plot_tree is a DecisionTreeClassifier or DecisionTreeRegressor
#       This is a hack to get around that restriction by declaring that PolicyTree inherits from DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
DecisionTreeClassifier.register(PolicyTree)
