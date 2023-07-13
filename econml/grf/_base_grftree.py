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
from ._criterion import LinearMomentGRFCriterionMSE, LinearMomentGRFCriterion
from ..tree import BaseTree
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
import copy

# =============================================================================
# Types and constants
# =============================================================================


CRITERIA_GRF = {"het": LinearMomentGRFCriterion,
                "mse": LinearMomentGRFCriterionMSE}

# =============================================================================
# Base GRF tree
# =============================================================================


class GRFTree(BaseTree):
    """A tree of a Generalized Random Forest [grftree1]. This method should be used primarily
    through the BaseGRF forest class and its derivatives and not as a standalone
    estimator. It fits a tree that solves the local moment equation problem::

        E[ m(Z; theta(x)) | X=x] = 0

    For some moment vector function m, that takes as input random samples of a random variable Z
    and is parameterized by some unknown parameter theta(x). Each node in the tree
    contains a local estimate of the parameter theta(x), for every region of X that
    falls within that leaf.

    Parameters
    ----------
    criterion : {``'mse'``, ``'het'``}, default 'mse'
        The function to measure the quality of a split. Supported criteria
        are ``'mse'`` for the mean squared error in a linear moment estimation tree and ``'het'`` for
        heterogeneity score. These criteria solve any linear moment problem of the form::

            E[J * theta(x) - A | X = x] = 0

        - The ``'mse'`` criterion finds splits that maximize the score:

          .. code-block::

            sum_{child} weight(child) * theta(child).T @ E[J | X in child] @ theta(child)

            - In the case of a causal tree, this coincides with minimizing the MSE:

              .. code-block::

                sum_{child} E[(Y - <theta(child), T>)^2 | X=child] weight(child)

            - In the case of an IV tree, this roughly coincides with minimize the projected MSE::

              .. code-block::

                sum_{child} E[(Y - <theta(child), E[T|Z]>)^2 | X=child] weight(child)

          Internally, for the case of more than two treatments or for the case of one treatment with
          ``fit_intercept=True`` then this criterion is approximated by computationally simpler variants for
          computationaly purposes. In particular, it is replaced by::

              sum_{child} weight(child) * rho(child).T @ E[J | X in child] @ rho(child)

          where:

          .. code-block::

              rho(child) := J(parent)^{-1} E[A - J * theta(parent) | X in child]

          This can be thought as a heterogeneity inducing score, but putting more weight on scores
          with a large minimum eigenvalue of the child jacobian ``E[J | X in child]``, which leads to smaller
          variance of the estimate and stronger identification of the parameters.

        - The ``'het'`` criterion finds splits that maximize the pure parameter heterogeneity score:

          .. code-block::

            sum_{child} weight(child) * rho(child).T @ rho(child)

          This can be thought as an approximation to the ideal heterogeneity score:

          .. code-block::

              weight(left) * weight(right) || theta(left) - theta(right)||_2^2 / weight(parent)^2

          as outlined in [grftree1]_

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

    min_var_leaf : None or double in (0, infinity), default None
        A constraint on the minimum degree of identification of the parameter of interest. This avoids performing
        splits where either the variance of the treatment is small or the correlation of the instrument with the
        treatment is small, or the variance of the instrument is small. Generically for any linear moment problem
        this translates to conditions on the leaf jacobian matrix J(leaf) that are proxies for a well-conditioned
        matrix, which leads to smaller variance of the local estimate. The proxy of the well-conditioning is
        different for different criterion, primarily for computational efficiency reasons.

        - If ``criterion='het'``, then the diagonal entries of J(leaf) are constraint to have absolute
          value at least `min_var_leaf`:

          .. code-block::

            for all i in {1, ..., n_outputs}: abs(J(leaf)[i, i]) > `min_var_leaf`

          In the context of a causal tree, when residual treatment is passed
          at fit time, then, this translates to a requirement on Var(T[i]) for every treatment coordinate i.
          In the context of an IV tree, with residual instruments and residual treatments passed at fit time
          this translates to ``Cov(T[i], Z[i]) > min_var_leaf`` for each coordinate i of the instrument and the
          treatment.

        - If ``criterion='mse'``, because the criterion stores more information about the leaf jacobian for
          every candidate split, then we impose further constraints on the pairwise determininants of the
          leaf jacobian, as they come at small extra computational cost, i.e.::

            for all i neq j:
                sqrt(abs(J(leaf)[i, i] * J(leaf)[j, j] - J(leaf)[i, j] * J(leaf)[j, i])) > `min_var_leaf`

          In the context of a causal tree, when residual treatment is passed at fit time, then this
          translates to a constraint on the pearson correlation coefficient on any two coordinates
          of the treatment within the leaf, i.e.::

            for all i neq j:
                sqrt( Var(T[i]) * Var(T[j]) * (1 - rho(T[i], T[j])^2) ) ) > `min_var_leaf`

          where rho(X, Y) is the Pearson correlation coefficient of two random variables X, Y. Thus this
          constraint also enforces that no two pairs of treatments be very co-linear within a leaf. This
          extra constraint primarily has bite in the case of more than two input treatments.

    min_var_leaf_on_val : bool, default False
        Whether the `min_var_leaf` constraint should also be enforced to hold on the validation set of the
        honest split too. If `min_var_leaf=None` then this flag does nothing. Setting this to True should
        be done with caution, as this partially violates the honesty structure, since parts of the variables
        other than the X variable (e.g. the variables that go into the jacobian J of the linear model) are
        used to inform the split structure of the tree. However, this is a benign dependence and for instance
        in a causal tree or an IV tree does not use the label y. It only uses the treatment T and the instrument
        Z and their local correlation structures to decide whether a split is feasible.

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
        The importance of a feature is computed as the (normalized) total heterogeneity that the feature
        creates. Each split that the feature was chosen adds::

            parent_weight * (left_weight * right_weight)
                * mean((value_left[k] - value_right[k])**2) / parent_weight**2

        to the importance of the feature. Each such quantity is also weighted by the depth of the split.
        By default splits below `max_depth=4` are not used in this calculation and also each split
        at depth `depth`, is re-weighted by 1 / (1 + `depth`)**2.0. See the method ``feature_importances``
        for a method that allows one to change these defaults.

    max_features_ : int
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    n_relevant_outputs_ : int
        The first `n_relevant_outputs_` where the ones we cared about when ``fit`` was performed.

    n_y_ : int
        The raw label dimension when ``fit`` is performed.

    n_samples_ : int
        The number of training samples when ``fit`` is performed.

    honest_ : int
        Whether honesty was enabled when ``fit`` was performed

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(econml.tree._tree.Tree)`` for attributes of Tree object.

    References
    ----------
    .. [grftree1] Athey, Susan, Julie Tibshirani, and Stefan Wager. "Generalized random forests."
        The Annals of Statistics 47.2 (2019): 1148-1178
        https://arxiv.org/pdf/1610.01271.pdf

    """

    def __init__(self, *,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_leaf=None,
                 min_var_leaf_on_val=False,
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
                         min_var_leaf=min_var_leaf,
                         min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features,
                         random_state=random_state,
                         min_impurity_decrease=min_impurity_decrease,
                         min_balancedness_tol=min_balancedness_tol,
                         honest=honest)

    def _get_valid_criteria(self):
        return CRITERIA_GRF

    def _get_valid_min_var_leaf_criteria(self):
        return (LinearMomentGRFCriterion,)

    def _get_store_jac(self):
        return True

    def init(self,):
        """ This method should be called before fit. We added this pre-fit step so that this step
        can be executed without parallelism as it contains code that holds the gil and can hinder
        parallel execution. We also did not merge this step to ``__init__`` as we want ``__init__`` to just
        be storing the parameters for easy cloning. We also don't want to directly pass a RandomState
        object as random_state, as we want to keep the starting seed to be able to replicate the
        randomness of the object outside the object.
        """
        self.random_seed_ = self.random_state
        self.random_state_ = check_random_state(self.random_seed_)
        return self

    def fit(self, X, y, n_y, n_outputs, n_relevant_outputs, sample_weight=None, check_input=True):
        """ Fit the tree from the data

        Parameters
        ----------
        X : (n, d) array
            The features to split on

        y : (n, m) array
            All the variables required to calculate the criterion function, evaluate splits and
            estimate local values, i.e. all the values that go into the moment function except X.

        n_y, n_outputs, n_relevant_outputs : auxiliary info passed to the criterion objects that
            help the object parse the variable y into each separate variable components.

            - In the case when `isinstance(criterion, LinearMomentGRFCriterion)`, then the first
              n_y columns of y are the raw outputs, the next n_outputs columns contain the A part
              of the moment and the next n_outputs * n_outputs columnts contain the J part of the moment
              in row contiguous format. The first n_relevant_outputs parameters of the linear moment
              are the ones that we care about. The rest are nuisance parameters.

        sample_weight : (n,) array, default None
            The sample weights

        check_input : bool, defaul=True
            Whether to check the input parameters for validity. Should be set to False to improve
            running time in parallel execution, if the variables have already been checked by the
            forest class that spawned this tree.
        """

        return super().fit(X, y, n_y, n_outputs, n_relevant_outputs,
                           sample_weight=sample_weight, check_input=check_input)

    def predict(self, X, check_input=True):
        """Return the prefix of relevant fitted local parameters for each X, i.e. theta(X).

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
        theta(X)[:n_relevant_outputs] : array_like of shape (n_samples, n_relevant_outputs)
            The estimated relevant parameters for each row of X
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict(X)
        return pred

    def predict_full(self, X, check_input=True):
        """Return the fitted local parameters for each X, i.e. theta(X).

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
        theta(X) : array_like of shape (n_samples, n_outputs)
            All the estimated parameters for each row of X
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        pred = self.tree_.predict_full(X)
        return pred

    def predict_alpha_and_jac(self, X, check_input=True):
        """Predict the local jacobian ``E[J | X=x]`` and the local alpha ``E[A | X=x]`` of
        a linear moment equation.

        Parameters
        ----------
        X : {array_like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``
        check_input : bool, default True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        alpha : array_like of shape (n_samples, n_outputs)
            The local alpha E[A | X=x] for each sample x
        jac : array_like of shape (n_samples, n_outputs * n_outputs)
            The local jacobian E[J | X=x] flattened in a C contiguous format
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self.tree_.predict_precond_and_jac(X)

    def predict_moment(self, X, parameter, check_input=True):
        """
        Predict the local moment value for each sample and at the given parameter::

            E[J | X=x] theta(x) - E[A | X=x]

        Parameters
        ----------
        X : {array_like} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float64``
        parameter : {array_like} of shape (n_samples, n_outputs)
            A parameter estimate for each sample
        check_input : bool, default True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        moment : array_like of shape (n_samples, n_outputs)
            The local moment E[J | X=x] theta(x) - E[A | X=x] for each sample x
        """
        alpha, jac = self.predict_alpha_and_jac(X)
        return alpha - np.einsum('ijk,ik->ij', jac.reshape((-1, self.n_outputs_, self.n_outputs_)), parameter)

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        """The feature importances based on the amount of parameter heterogeneity they create.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized) total heterogeneity that the feature
        creates. Each split that the feature was chosen adds::

            parent_weight * (left_weight * right_weight)
                * mean((value_left[k] - value_right[k])**2) / parent_weight**2

        to the importance of the feature. Each such quantity is also weighted by the depth of the split.

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

        return self.tree_.compute_feature_heterogeneity_importances(normalize=True, max_depth=max_depth,
                                                                    depth_decay=depth_decay_exponent)

    @property
    def feature_importances_(self):
        return self.feature_importances()
