# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Subsampled honest forest extension to scikit-learn's forest methods. Contains pieces of code from
scikit-learn's random forest implementation.
"""
from ..grf import RegressionForest
from ..utilities import deprecated


@deprecated("The SubsampledHonestForest class has been deprecated by the grf.RegressionForest class; "
            "an upcoming release will remove support for the this class.")
def SubsampledHonestForest(n_estimators=100,
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
    """
    An implementation of a subsampled honest random forest regressor on top of an sklearn
    regression tree. Implements subsampling and honesty as described in [3]_,
    but uses a scikit-learn regression tree as a base. It provides confidence intervals based on ideas
    described in [3]_ and [4]_

    Parameters
    ----------
    n_estimators : integer, optional (default=100)
        The total number of trees in the forest. The forest consists of a
        forest of sqrt(n_estimators) sub-forests, where each sub-forest
        contains sqrt(n_estimators) trees.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion.

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

    References
    ----------

    .. [3] S. Athey, S. Wager, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests",
            Journal of the American Statistical Association 113.523 (2018): 1228-1242.

    .. [4] S. Athey, J. Tibshirani, and S. Wager, "Generalized random forests",
            The Annals of Statistics, 47(2), 1148-1178, 2019.

    """
    return RegressionForest(n_estimators=n_estimators,
                            criterion=criterion,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            min_weight_fraction_leaf=min_weight_fraction_leaf,
                            max_features=max_features,
                            min_impurity_decrease=min_impurity_decrease,
                            max_samples=.45 if subsample_fr == 'auto' else subsample_fr / 2,
                            honest=honest,
                            n_jobs=n_jobs,
                            random_state=random_state,
                            verbose=verbose,
                            warm_start=warm_start)
