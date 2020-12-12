import numpy as np
import numbers
from math import ceil
from ..tree import Tree
from ._criterion import LinearMomentGRFCriterionMSE, LinearMomentGRFCriterion
from ..tree._criterion import Criterion
from ..tree._splitter import Splitter, BestSplitter
from ..tree import DepthFirstTreeBuilder
from . import _criterion
from ..tree import _tree
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils.validation import check_is_fitted
import copy

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_GRF = {"het": LinearMomentGRFCriterion,
                "mse": LinearMomentGRFCriterionMSE}

SPLITTERS = {"best": BestSplitter, }


# =============================================================================
# Base GRF tree
# =============================================================================

class GRFTree(BaseEstimator):

    def __init__(self, *,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 min_var_leaf=None,
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
        self.min_var_leaf = min_var_leaf
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

    def fit(self, X, y, n_y, n_outputs, n_relevant_outputs, sample_weight=None, check_input=True):

        random_state = self.random_state  # if check_input=False, this is assumed to be an instance of RandomState

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        self.n_outputs_ = n_outputs
        self.n_relevant_outputs_ = n_relevant_outputs
        self.n_y_ = n_y

        if check_input:
            random_state = check_random_state(self.random_state)

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
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")
        if not 0 <= self.min_balancedness_tol <= 0.5:
            raise ValueError("min_balancedness_tol must be in [0, 0.5]")

        if self.min_var_leaf is None:
            min_var_leaf = -1.0
        elif isinstance(self.min_var_leaf, numbers.Real) and (self.min_var_leaf >= 0.0):
            min_var_leaf = self.min_var_leaf
        else:
            raise ValueError("min_var_leaf must be either None or a real in [0, infinity). "
                             "Got {}".format(self.min_var_leaf))

        # Set min_weight_leaf from min_weight_fraction_leaf
        if sample_weight is None:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               n_samples)
        else:
            min_weight_leaf = (self.min_weight_fraction_leaf *
                               np.sum(sample_weight))

        inds = np.arange(n_samples, dtype=np.intp)
        if self.honest:
            random_state.shuffle(inds)
            samples_train, samples_val = inds[:n_samples // 2], inds[n_samples // 2:]
        else:
            samples_train, samples_val = inds, inds

        # Build tree
        max_train = len(samples_train) if sample_weight is None else np.count_nonzero(sample_weight[samples_train])
        if self.honest:
            max_val = len(samples_val) if sample_weight is None else np.count_nonzero(sample_weight[samples_val])
        if callable(self.criterion):
            criterion = self.criterion(self.n_outputs_, self.n_relevant_outputs_, self.n_features_, self.n_y_,
                                       n_samples, max_train,
                                       random_state.randint(np.iinfo(np.int32).max))
            if not isinstance(criterion, Criterion):
                raise ValueError("Input criterion is not a valid criterion")
            if self.honest:
                criterion_val = self.criterion(self.n_outputs_, self.n_relevant_outputs_, self.n_features_, self.n_y_,
                                               n_samples, max_val,
                                               random_state.randint(np.iinfo(np.int32).max))
            else:
                criterion_val = criterion
        else:
            criterion = CRITERIA_GRF[self.criterion](
                self.n_outputs_, self.n_relevant_outputs_, self.n_features_, self.n_y_, n_samples, max_train,
                random_state.randint(np.iinfo(np.int32).max))
            if self.honest:
                criterion_val = CRITERIA_GRF[self.criterion](
                    self.n_outputs_, self.n_relevant_outputs_, self.n_features_, self.n_y_, n_samples, max_val,
                    random_state.randint(np.iinfo(np.int32).max))
            else:
                criterion_val = criterion

        if (min_var_leaf >= 0.0 and (not isinstance(criterion, LinearMomentGRFCriterion))
                and (not isinstance(criterion_val, LinearMomentGRFCriterion))):
            raise ValueError("This criterion does not support min_var_leaf constraint!")

        splitter = self.splitter
        if not isinstance(self.splitter, Splitter):
            splitter = SPLITTERS[self.splitter](criterion, criterion_val,
                                                self.max_features_,
                                                min_samples_leaf,
                                                min_weight_leaf,
                                                self.min_balancedness_tol,
                                                self.honest,
                                                min_var_leaf,
                                                random_state.randint(np.iinfo(np.int32).max))

        self.tree_ = Tree(self.n_features_, self.n_outputs_, self.n_relevant_outputs_, store_jac=True)

        builder = DepthFirstTreeBuilder(splitter, min_samples_split,
                                        min_samples_leaf,
                                        min_weight_leaf,
                                        max_depth,
                                        self.min_impurity_decrease)

        builder.build(self.tree_, X, y, samples_train, samples_val,
                      sample_weight=sample_weight,
                      store_jac=True)

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse=False)

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features_, n_features))

        return X

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of shape (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        return pred

    def predict_full(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of shape (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict_full(X)
        return pred

    def predict_jac(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.predict_jac(X)

    def predict_alpha(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.predict_precond(X)

    def predict_alpha_and_jac(self, X, check_input=True):
        """Predict class or regression value for X.
        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes, or the predict values.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.predict_precond_and_jac(X)

    def predict_moment(self, X, parameter, check_input=True):
        alpha, jac = self.predict_alpha_and_jac(X)
        return alpha - np.einsum('ijk,ik->ij', jac.reshape((-1, self.n_outputs_, self.n_outputs_)), parameter)

    def predict_alpha_and_jac_and_moment(self, X, parameter, check_input=True):
        alpha, jac = self.predict_alpha_and_jac(X)
        return (alpha, jac,
                np.einsum('ijk,ik->ij', jac.reshape((-1, self.n_outputs_, self.n_outputs_)), parameter) - alpha)

    def apply(self, X, check_input=True):
        """Return the index of the leaf that each sample is predicted as.
        .. versionadded:: 0.17
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
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
        .. versionadded:: 0.18
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
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

    def feature_importances(self, max_depth=None, depth_decay_exponent=.0):
        """Return the feature importances.
        The importance of a feature is computed as the (normalized) total
        reduction of the criterion brought by that feature.
        It is also known as the Gini importance.
        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.
        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total reduction of criteria by feature
            (Gini importance).
        """
        check_is_fitted(self)

        return self.tree_.compute_feature_heterogeneity_importances(normalize=True, max_depth=max_depth,
                                                                    depth_decay=depth_decay_exponent)
