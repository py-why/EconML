import numpy as np
from warnings import warn
from ..utilities import cross_product
from ._base_grf import BaseGRF

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


class RegressionForest(BaseGRF):
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
                         max_features=max_features, min_impurity_decrease=min_impurity_decrease,
                         max_samples=max_samples, min_balancedness_tol=min_balancedness_tol,
                         honest=honest, inference=inference, fit_intercept=False,
                         subforest_size=subforest_size, n_jobs=n_jobs, random_state=random_state, verbose=verbose,
                         warm_start=warm_start)

    def fit(self, X, y):
        return super().fit(X, y, np.ones((len(X), 1)))

    def get_alpha(self, X, y, T):
        return y

    def get_pointJ(self, X, y, T):
        jac = np.eye(y.shape[1]).reshape((1, -1))
        return np.tile(jac, (X.shape[0], 1))
