# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from warnings import warn
from ..utilities import cross_product
from ._base_grf import BaseGRF
from ..utilities import check_inputs
from sklearn.base import BaseEstimator, clone
from sklearn.utils import check_X_y

__all__ = ["MultiOutputGRF",
           "CausalForest",
           "CausalIVForest",
           "RegressionForest"]

# =============================================================================
# A MultOutputWrapper for GRF classes
# =============================================================================


class MultiOutputGRF(BaseEstimator):
    """ Simple wrapper estimator that enables multiple outcome labels for all the
    grf estimators that only accept a single outcome. Similar to MultiOutputRegressor.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, T, y, *, sample_weight=None, **kwargs):
        y, T, X, _ = check_inputs(y, T, X, W=None, multi_output_T=True, multi_output_Y=True)
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))
        self.estimators_ = [clone(self.estimator) for _ in range(y.shape[1])]
        [estimator.fit(X, T, y[:, [it]], sample_weight=sample_weight, **kwargs)
         for it, estimator in enumerate(self.estimators_)]
        return self

    def predict(self, X, interval=False, alpha=0.05):
        if interval:
            pred, lb, ub = zip(*[estimator.predict(X, interval=interval, alpha=alpha)
                                 for estimator in self.estimators_])
            return np.moveaxis(np.array(pred), 0, 1), np.moveaxis(np.array(lb), 0, 1), np.moveaxis(np.array(ub), 0, 1)
        else:
            pred = [estimator.predict(X, interval=interval, alpha=alpha) for estimator in self.estimators_]
            return np.moveaxis(np.array(pred), 0, 1)

    def predict_and_var(self, X):
        pred, var = zip(*[estimator.predict_and_var(X) for estimator in self.estimators_])
        return np.moveaxis(np.array(pred), 0, 1), np.moveaxis(np.array(var), 0, 1)

    def predict_projection_and_var(self, X, projector):
        pred, var = zip(*[estimator.predict_projection_and_var(X, projector) for estimator in self.estimators_])
        return np.moveaxis(np.array(pred), 0, 1), np.moveaxis(np.array(var), 0, 1)

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        res = [estimator.feature_importances(max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
               for estimator in self.estimators_]
        return np.array(res)

    @property
    def feature_importances_(self):
        return self.feature_importances()

    def __len__(self):
        """Return the number of estimators in the ensemble for each target y."""
        return len(self.estimators_[0].estimators_)

    def __getitem__(self, index):
        """Return a list of the index'th estimator in the ensemble for each target y."""
        return [forest[index] for forest in self.estimators_]

    def __iter__(self):
        """Return iterator over tuples of estimators for each target y in the ensemble."""
        return iter(zip(*self.estimators_))

# =============================================================================
# Instantiations of Generalized Random Forest
# =============================================================================


class CausalForest(BaseGRF):
    """
    A Causal Forest [cf1]_. It fits a forest that solves the local moment equation problem:

    .. code-block::

        E[ (Y - <theta(x), T> - beta(x)) (T;1) | X=x] = 0

    Each node in the tree contains a local estimate of the parameter theta(x), for every region of X that
    falls within that leaf.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees

    criterion : {``"mse"``, ``"het"``}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error in a linear moment estimation tree and "het" for
        heterogeneity score.

        - The "mse" criterion finds splits that minimize the score:

          .. code-block::

            sum_{child} E[(Y - <theta(child), T> - beta(child))^2 | X=child] weight(child)

          Internally, for the case of more than two treatments or for the case of one treatment with
          ``fit_intercept=True`` then this criterion is approximated by computationally simpler variants for
          computationaly purposes. In particular, it is replaced by::

              sum_{child} weight(child) * rho(child).T @ E[(T;1) @ (T;1).T | X in child] @ rho(child)

          where:

          .. code-block::

            rho(child) := E[(T;1) @ (T;1).T | X in parent]^{-1}
                                    * E[(Y - <theta(x), T> - beta(x)) (T;1) | X in child]

          This can be thought as a heterogeneity inducing score, but putting more weight on scores
          with a large minimum eigenvalue of the child jacobian ``E[(T;1) @ (T;1).T | X in child]``,
          which leads to smaller variance of the estimate and stronger identification of the parameters.

        - The "het" criterion finds splits that maximize the pure parameter heterogeneity score:

          .. code-block::

            sum_{child} weight(child) * rho(child)[:n_T].T @ rho(child)[:n_T]

          This can be thought as an approximation to the ideal heterogeneity score:

          .. code-block::

              weight(left) * weight(right) || theta(left) - theta(right)||_2^2 / weight(parent)^2

          as outlined in [cf1]_

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

    min_var_fraction_leaf : None or float in (0, 1], default=None
        A constraint on some proxy of the variation of the treatment vector that should be contained within each
        leaf as a percentage of the total variance of the treatment vector on the whole sample. This avoids
        performing splits where either the variance of the treatment is small and hence the local parameter
        is not well identified and has high variance. The proxy of variance is different for different criterion,
        primarily for computational efficiency reasons.

        - If ``criterion='het'``, then this constraint translates to:

          .. code-block::

            for all i in {1, ..., T.shape[1]}:
                E[T[i]^2 | X in leaf] > `min_var_fraction_leaf` * E[T[i]^2]

          When ``T`` is the residual treatment (i.e. centered), this translates to a requirement that

          .. code-block::

            for all i in {1, ..., T.shape[1]}:
                Var(T[i] | X in leaf) > `min_var_fraction_leaf` * Var(T[i])

        - If ``criterion='mse'``, because the criterion stores more information about the leaf for
          every candidate split, then this constraint imposes further constraints on the pairwise correlations
          of different coordinates of each treatment, i.e.:

          .. code-block::

            for all i neq j:
              sqrt(Var(T[i]|X in leaf) * Var(T[j]|X in leaf) * (1 - rho(T[i], T[j]| in leaf)^2))
                  > `min_var_fraction_leaf` sqrt(Var(T[i]) * Var(T[j]) * (1 - rho(T[i], T[j])^2))

          where rho(X, Y) is the Pearson correlation coefficient of two random variables X, Y. Thus this
          constraint also enforces that no two pairs of treatments be very co-linear within a leaf. This
          extra constraint primarily has bite in the case of more than two input treatments and also avoids
          leafs where the parameter estimate has large variance due to local co-linearities of the treatments.

    min_var_leaf_on_val : bool, default=False
        Whether the `min_var_fraction_leaf` constraint should also be enforced to hold on the validation set of the
        honest split too. If `min_var_leaf=None` then this flag does nothing. Setting this to True should
        be done with caution, as this partially violates the honesty structure, since the treatment variable
        of the validation set is used to inform the split structure of the tree. However, this is a benign
        dependence as it only uses local correlation structure of the treatment T to decide whether
        a split is feasible.

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

    max_samples : int or float in (0, 1], default=.45,
        The number of samples to use for each subsample that is used to train each tree:

        - If int, then train each tree on `max_samples` samples, sampled without replacement from all the samples
        - If float, then train each tree on ceil(`max_samples` * `n_samples`), sampled without replacement
          from all the samples.

        If ``inference=True``, then `max_samples` must either be an integer smaller than `n_samples//2` or a float
        less than or equal to .5.

    min_balancedness_tol: float in [0, .5], default=.45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : bool, default=True
        Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal
        sized subsets, the train and the val set. All samples in train are used to create the split structure
        and all samples in val are used to calculate the value of each node in the tree.

    inference : bool, default=True
        Whether inference (i.e. confidence interval construction and uncertainty quantification of the estimates)
        should be enabled. If `inference=True`, then the estimator uses a bootstrap-of-little-bags approach
        to calculate the covariance of the parameter vector, with am objective Bayesian debiasing correction
        to ensure that variance quantities are positive.

    fit_intercept : bool, default=True
        Whether we should fit an intercept nuisance parameter beta(x). If `fit_intercept=False`, then no (;1) is
        appended to the treatment variable in all calculations in this docstring. If `fit_intercept=True`, then
        the constant treatment of `(;1)` is appended to each treatment vector and the coefficient in front
        of this constant treatment is the intercept beta(x). beta(x) is treated as a nuisance and not returned
        by the predict(X), predict_and_var(X) or the predict_var(X) methods.
        Use predict_full(X) to recover the intercept term too.

    subforest_size : int, default=4,
        The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation.
        The parameter `n_estimators` must be divisible by `subforest_size`. Should typically be a small constant.

    n_jobs : int or None, default=-1
        The number of parallel jobs to be used for parallelism; follows joblib semantics.
        ``n_jobs=-1`` means all available cpu cores. ``n_jobs=None`` means no parallelism.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=``False``
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

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
        By default splits below ``max_depth=4`` are not used in this calculation and also each split
        at depth `depth`, is re-weighted by ``1 / (1 + `depth`)**2.0``. See the method ``feature_importances``
        for a method that allows one to change these defaults.

    estimators_ : list of objects of type :class:`~econml.grf.GRFTree`
        The fitted trees.

    References
    ----------
    .. [cf1] Athey, Susan, Julie Tibshirani, and Stefan Wager. "Generalized random forests."
        The Annals of Statistics 47.2 (2019): 1148-1178
        https://arxiv.org/pdf/1610.01271.pdf

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
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf, min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features, min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples, min_balancedness_tol=min_balancedness_tol,
                         honest=honest, inference=inference, fit_intercept=fit_intercept,
                         subforest_size=subforest_size, n_jobs=n_jobs, random_state=random_state, verbose=verbose,
                         warm_start=warm_start)

    def fit(self, X, T, y, *, sample_weight=None, sample_var=None):
        """
        Build a causal forest of trees from the training set (X, T, y).

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

        Returns
        -------
        self : object
        """
        return super().fit(X, T, y, sample_weight=sample_weight, sample_var=sample_var)

    def _get_alpha_and_pointJ(self, X, T, y):
        # Append a constant treatment if `fit_intercept=True`, the coefficient
        # in front of the constant treatment is the intercept in the moment equation.
        if self.fit_intercept:
            T = np.hstack([T, np.ones((T.shape[0], 1))])
        return y * T, cross_product(T, T)

    def _get_n_outputs_decomposition(self, X, T, y):
        n_relevant_outputs = T.shape[1]
        n_outputs = n_relevant_outputs
        if self.fit_intercept:
            n_outputs = n_relevant_outputs + 1
        return n_outputs, n_relevant_outputs


class CausalIVForest(BaseGRF):
    """A Causal IV Forest [cfiv1]_. It fits a forest that solves the local moment equation problem:

    .. code-block

        E[ (Y - <theta(x), T> - beta(x)) (Z;1) | X=x] = 0

    Each node in the tree contains a local estimate of the parameter theta(x), for every region of X that
    falls within that leaf.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees

    criterion : {``"mse"``, ``"het"``}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error in a linear moment estimation tree and "het" for
        heterogeneity score.

        - The "mse" criterion finds splits that approximately minimize the score:

          .. code-block::

            sum_{child} E[(Y - <theta(child), E[T|Z]> - beta(child))^2 | X=child] weight(child)

          Though we note that the local estimate is still estimated by solving the local moment equation for samples
          that fall within the node and not by minimizing this loss. Internally, for the case of more than two
          treatments or for the case of one treatment with `fit_intercept=True` then this criterion is approximated
          by computationally simpler variants for computationaly purposes. In particular, it is replaced by:

          .. code-block::

              sum_{child} weight(child) * rho(child).T @ E[(T;1) @ (Z;1).T | X in child] @ rho(child)

          where:

          .. code-block::

              rho(child) := E[(T;1) @ (Z;1).T | X in parent]^{-1}
                                * E[(Y - <theta(x), T> - beta(x)) (Z;1) | X in child]

          This can be thought as a heterogeneity inducing score, but putting more weight on scores
          with a large minimum eigenvalue of the child jacobian E[(T;1) @ (Z;1).T | X in child], which leads to smaller
          variance of the estimate and stronger identification of the parameters.

        - The ``"het"`` criterion finds splits that maximize the pure parameter heterogeneity score:

          .. code-block::

            sum_{child} weight(child) * rho(child)[:n_T].T @ rho(child)[:n_T]

          This can be thought as an approximation to the ideal heterogeneity score:

          .. code-block::

              weight(left) * weight(right) || theta(left) - theta(right)||_2^2 / weight(parent)^2

          as outlined in [cfiv1]_

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

    min_var_fraction_leaf : None or float in (0, 1], default=None
        A constraint on some proxy of the variation of the covariance of the treatment vector with the instrument
        vector that should be contained within each leaf as a percentage of the total cov-variance of the treatment
        and instrument on the whole sample. This avoids performing splits where either the variance of the treatment
        is small or the variance of the instrument is small or the strength of the instrument on the treatment is
        locally weak and hence the local parameter is not well identified and has high variance.
        The proxy of variance is different for different criterion, primarily for computational efficiency reasons.

        - If ``criterion='het'``, then this constraint translates to:

          .. code-block::

            for all i in {1, ..., T.shape[1]}:
                E[T[i] Z[i] | X in leaf] > `min_var_fraction_leaf` * E[T[i] Z[i]]

          When `T` is the residual treatment and `Z` the residual instrument (i.e. centered),
          this translates to a requirement that:

          .. code-block::

            for all i in {1, ..., T.shape[1]}:
                Cov(T[i], Z[i] | X in leaf) > `min_var_fraction_leaf` * Cov(T[i], Z[i])

        - If ``criterion='mse'``, because the criterion stores more information about the leaf for
          every candidate split, then this constraint imposes further constraints on the pairwise correlations
          of different coordinates of each treatment. For instance, when the instrument and treatment are both
          residualized (centered) then this constraint translates to:

          .. code-block::

            for all i neq j:
                E[T[i]Z[i]] E[T[j]Z[j]] - E[T[i] Z[j]]
                sqrt(Cov(T[i], Z[i] |X in leaf) * Cov(T[j], Z[j]|X in leaf)
                        * (1 - rho(T[i], Z[j]|X in leaf) * rho(T[j], Z[i]|X in leaf)))
                  > `min_var_fraction_leaf` * sqrt(Cov(T[i], Z[i]) * Cov(T[j], Z[j])
                                                    * (1 - rho(T[i], Z[j]) * rho(T[j], Z[i])))

          where rho(X, Y) is the Pearson correlation coefficient of two random variables X, Y. Thus this
          constraint also enforces that no two pairs of treatments and instruments be very co-linear within a leaf.
          This extra constraint primarily has bite in the case of more than two input treatments and also avoids
          leafs where the parameter estimate has large variance due to local co-linearities of the treatments.

    min_var_leaf_on_val : bool, default=False
        Whether the `min_var_fraction_leaf` constraint should also be enforced to hold on the validation set of the
        honest split too. If `min_var_leaf=None` then this flag does nothing. Setting this to True should
        be done with caution, as this partially violates the honesty structure, since parts of the variables
        other than the X variable (e.g. the variables that go into the jacobian J of the linear model) are
        used to inform the split structure of the tree. However, this is a benign dependence as it only uses
        the treatment T its local correlation structure to decide whether a split is feasible.

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

    max_samples : int or float in (0, 1], default=.45,
        The number of samples to use for each subsample that is used to train each tree:

        - If int, then train each tree on `max_samples` samples, sampled without replacement from all the samples
        - If float, then train each tree on ceil(`max_samples` * `n_samples`), sampled without replacement
          from all the samples.

        If ``inference=True``, then `max_samples` must either be an integer smaller than `n_samples//2` or a float
        less than or equal to .5.

    min_balancedness_tol: float in [0, .5], default=.45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : bool, default=True
        Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal
        sized subsets, the train and the val set. All samples in train are used to create the split structure
        and all samples in val are used to calculate the value of each node in the tree.

    inference : bool, default=True
        Whether inference (i.e. confidence interval construction and uncertainty quantification of the estimates)
        should be enabled. If ``inference=True``, then the estimator uses a bootstrap-of-little-bags approach
        to calculate the covariance of the parameter vector, with am objective Bayesian debiasing correction
        to ensure that variance quantities are positive.

    fit_intercept : bool, default=True
        Whether we should fit an intercept nuisance parameter beta(x). If `fit_intercept=False`, then no (;1) is
        appended to the treatment variable in all calculations in this docstring. If `fit_intercept=True`, then
        the constant treatment of `(;1)` is appended to each treatment vector and the coefficient in front
        of this constant treatment is the intercept beta(x). beta(x) is treated as a nuisance and not returned
        by the predict(X), predict_and_var(X) or the predict_var(X) methods.
        Use predict_full(X) to recover the intercept term too.

    subforest_size : int, default=4,
        The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation.
        The parameter `n_estimators` must be divisible by `subforest_size`. Should typically be a small constant.

    n_jobs : int or None, default=-1
        The number of parallel jobs to be used for parallelism; follows joblib semantics.
        `n_jobs=-1` means all available cpu cores. `n_jobs=None` means no parallelism.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

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
        at depth `depth`, is re-weighted by ``1 / (1 + `depth`)**2.0``. See the method ``feature_importances``
        for a method that allows one to change these defaults.

    estimators_ : list of objects of type :class:`~econml.grf.GRFTree`
        The fitted trees.

    References
    ----------
    .. [cfiv1] Athey, Susan, Julie Tibshirani, and Stefan Wager. "Generalized random forests."
        The Annals of Statistics 47.2 (2019): 1148-1178
        https://arxiv.org/pdf/1610.01271.pdf

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
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=min_var_fraction_leaf, min_var_leaf_on_val=min_var_leaf_on_val,
                         max_features=max_features, min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples, min_balancedness_tol=min_balancedness_tol,
                         honest=honest, inference=inference, fit_intercept=fit_intercept,
                         subforest_size=subforest_size, n_jobs=n_jobs, random_state=random_state, verbose=verbose,
                         warm_start=warm_start)

    def fit(self, X, T, y, *, Z, sample_weight=None, sample_var=None):
        """
        Build an IV forest of trees from the training set (X, T, y, Z).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float64``.
        T : array-like of shape (n_samples, n_treatments)
            The treatment vector for each sample
        y : array-like of shape (n_samples,) or (n_samples, n_outcomes)
            The outcome values for each sample.
        Z : array-like of shape (n_samples, n_treatments)
            The instrument vector. This method requires an equal amount of instruments and
            treatments, i.e. an exactly identified IV regression. For low variance, use
            the optimal instruments by project the instrument on the treatment vector, i.e.
            Z -> E[T | Z], in a first stage estimation.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        sample_var : array-like of shape (n_samples,), default=None
            Currently ignored and raises a warning. Added for API consistency and for raising relevant errors.

        Returns
        -------
        self : object
        """
        return super().fit(X, T, y, Z=Z, sample_weight=sample_weight, sample_var=sample_var)

    def _get_alpha_and_pointJ(self, X, T, y, *, Z):
        # Append a constant treatment and constant instrument if `fit_intercept=True`,
        # the coefficient in front of the constant treatment is the intercept in the moment equation.
        _, Z = check_X_y(X, Z, y_numeric=True, multi_output=True, accept_sparse=False)
        Z = np.atleast_1d(Z)
        if Z.ndim == 1:
            Z = np.reshape(Z, (-1, 1))

        if not Z.shape[1] == T.shape[1]:
            raise ValueError("The dimension of the instrument should match the dimension of the treatment. "
                             "This method handles only exactly identified instrumental variable regression. "
                             "Preprocess your instrument by projecting it to the treatment space.")

        if self.fit_intercept:
            T = np.hstack([T, np.ones((T.shape[0], 1))])
            Z = np.hstack([Z, np.ones((Z.shape[0], 1))])

        return y * Z, cross_product(Z, T)

    def _get_n_outputs_decomposition(self, X, T, y, *, Z):
        n_relevant_outputs = T.shape[1]
        n_outputs = n_relevant_outputs
        if self.fit_intercept:
            n_outputs = n_relevant_outputs + 1
        return n_outputs, n_relevant_outputs


class RegressionForest(BaseGRF):
    """
    An implementation of a subsampled honest random forest regressor on top of an sklearn
    regression tree. Implements subsampling and honesty as described in [rf3]_,
    but uses a scikit-learn regression tree as a base. It provides confidence intervals based on ideas
    described in [rf3]_ and [rf4]_

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and uses averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is smaller than the original size and subsampling is
    performed without replacement. Each decision tree is built in an honest
    manner: half of the sub-sampled data are used for creating the tree structure
    (referred to as the splitting sample) and the other half for calculating the
    constant regression estimate at each leaf of the tree (referred to as the estimation sample).
    One difference with the algorithm proposed in [rf3]_ is that we do not ensure balancedness
    and we do not consider poisson sampling of the features, so that we guarantee
    that each feature has a positive probability of being selected on each split.
    Rather we use the original algorithm of Breiman [rf1]_, which selects the best split
    among a collection of candidate splits, as long as the max_depth is not reached
    and as long as there are not more than max_leafs and each child contains
    at least min_samples_leaf samples and total weight fraction of
    min_weight_fraction_leaf. Moreover, it allows the use of both mean squared error (MSE)
    and mean absoulte error (MAE) as the splitting criterion. Finally, we allow
    for early stopping of the splits if the criterion is not improved by more than
    min_impurity_decrease. These techniques that date back to the work of [rf1]_,
    should lead to finite sample performance improvements, especially for
    high dimensional features.

    The implementation also provides confidence intervals
    for each prediction using a bootstrap of little bags approach described in [rf3]_:
    subsampling is performed at hierarchical level by first drawing a set of half-samples
    at random and then sub-sampling from each half-sample to build a forest
    of forests. All the trees are used for the point prediction and the distribution
    of predictions returned by each of the sub-forests is used to calculate the standard error
    of the point prediction.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees

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

    max_samples : int or float in (0, 1], default=.45,
        The number of samples to use for each subsample that is used to train each tree:

        - If int, then train each tree on `max_samples` samples, sampled without replacement from all the samples
        - If float, then train each tree on ceil(`max_samples` * `n_samples`), sampled without replacement
          from all the samples.

        If `inference=True`, then `max_samples` must either be an integer smaller than `n_samples//2` or a float
        less than or equal to .5.

    min_balancedness_tol: float in [0, .5], default=.45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : bool, default=True
        Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal
        sized subsets, the train and the val set. All samples in train are used to create the split structure
        and all samples in val are used to calculate the value of each node in the tree.

    inference : bool, default=True
        Whether inference (i.e. confidence interval construction and uncertainty quantification of the estimates)
        should be enabled. If `inference=True`, then the estimator uses a bootstrap-of-little-bags approach
        to calculate the covariance of the parameter vector, with am objective Bayesian debiasing correction
        to ensure that variance quantities are positive.

    subforest_size : int, default=4,
        The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation.
        The parameter `n_estimators` must be divisible by `subforest_size`. Should typically be a small constant.

    n_jobs : int or None, default=-1
        The number of parallel jobs to be used for parallelism; follows joblib semantics.
        `n_jobs=-1` means all available cpu cores. `n_jobs=None` means no parallelism.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

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
        at depth `depth`, is re-weighted by ``1 / (1 + `depth`)**2.0``. See the method ``feature_importances``
        for a method that allows one to change these defaults.

    estimators_ : list of objects of type :class:`~econml.grf.GRFTree`
        The fitted trees.


    Examples
    --------

    .. testcode::

        import numpy as np
        from econml.grf import RegressionForest
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        np.set_printoptions(suppress=True)
        np.random.seed(123)
        X, y = make_regression(n_samples=1000, n_features=4, n_informative=2,
                               random_state=0, shuffle=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)
        regr = RegressionForest(max_depth=None, random_state=0,
                                n_estimators=1000)

    >>> regr.fit(X_train, y_train)
    RegressionForest(n_estimators=1000, random_state=0)
    >>> regr.feature_importances_
    array([0.88..., 0.11..., 0.00..., 0.00...])
    >>> regr.predict(np.ones((1, 4)), interval=True, alpha=.05)
    (array([[121.0...]]), array([[103.6...]]), array([[138.3...]]))

    References
    ----------

    .. [rf1] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    .. [rf3] S. Athey, S. Wager, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests",
            Journal of the American Statistical Association 113.523 (2018): 1228-1242.

    .. [rf4] S. Athey, J. Tibshirani, and S. Wager, "Generalized random forests",
            The Annals of Statistics, 47(2), 1148-1178, 2019.

    """

    def __init__(self,
                 n_estimators=100, *,
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 inference=True,
                 subforest_size=4,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super().__init__(n_estimators=n_estimators, criterion='het', max_depth=max_depth,
                         min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                         min_var_fraction_leaf=None, min_var_leaf_on_val=False,
                         max_features=max_features, min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples, min_balancedness_tol=min_balancedness_tol,
                         honest=honest, inference=inference, fit_intercept=False,
                         subforest_size=subforest_size, n_jobs=n_jobs, random_state=random_state, verbose=verbose,
                         warm_start=warm_start)

    def fit(self, X, y, *, sample_weight=None, sample_var=None):
        """
        Build an IV forest of trees from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. Internally, its dtype will be converted
            to ``dtype=np.float64``.
        y : array-like of shape (n_samples,) or (n_samples, n_outcomes)
            The outcome values for each sample.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node.
        sample_var : array-like of shape (n_samples,), default=None
            Currently ignored and raises a warning. Added for API consistency and for raising relevant errors.

        Returns
        -------
        self : object
        """
        return super().fit(X, y, np.ones((len(X), 1)), sample_weight=sample_weight, sample_var=sample_var)

    def _get_alpha_and_pointJ(self, X, y, T):
        jac = np.eye(y.shape[1]).reshape((1, -1))
        return y, np.tile(jac, (X.shape[0], 1))

    def _get_n_outputs_decomposition(self, X, y, T):
        return y.shape[1], y.shape[1]
