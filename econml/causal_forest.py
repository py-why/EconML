# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .utilities import LassoCVWrapper, deprecated
from sklearn.linear_model import LogisticRegressionCV
from .dml import CausalForestDML


@deprecated("The CausalForest class has been deprecated by the econml.dml.CausalForestDML; "
            "an upcoming release will remove support for the old class")
def CausalForest(n_trees=500,
                 min_leaf_size=10,
                 max_depth=10,
                 subsample_ratio=0.7,
                 lambda_reg=0.01,
                 model_T='auto',
                 model_Y=LassoCVWrapper(cv=3),
                 cv=2,
                 discrete_treatment=False,
                 categories='auto',
                 n_jobs=-1,
                 backend='threading',
                 verbose=0,
                 batch_size='auto',
                 random_state=None):
    """CausalForest for continuous treatments. To apply to discrete
    treatments, first one-hot-encode your treatments and then pass the one-hot-encoding.

    Parameters
    ----------
    n_trees : integer, optional (default=500)
        Number of causal estimators in the forest.

    min_leaf_size : integer, optional (default=10)
        The minimum number of samples in a leaf.

    max_depth : integer, optional (default=10)
        The maximum number of splits to be performed when expanding the tree.

    subsample_ratio : float, optional (default=0.7)
        The ratio of the total sample to be used when training a causal tree.
        Values greater than 1.0 will be considered equal to 1.0.

    lambda_reg : float, optional (default=0.01)
        The regularization coefficient in the ell_2 penalty imposed on the
        locally linear part of the second stage fit. This is not applied to
        the local intercept, only to the coefficient of the linear component.

    model_T : estimator, optional (default=sklearn.linear_model.LassoCV(cv=3))
        The estimator for residualizing the continuous treatment.
        Must implement `fit` and `predict` methods.

    model_Y :  estimator, optional (default=sklearn.linear_model.LassoCV(cv=3)
        The estimator for residualizing the outcome. Must implement
        `fit` and `predict` methods.

    cv : int, cross-validation generator or an iterable, optional (default=2)
        The specification of the CV splitter to be used for cross-fitting, when constructing
        the global residuals of Y and T.

    discrete_treatment : bool, optional (default=False)
        Whether the treatment should be treated as categorical. If True, then the treatment T is
        one-hot-encoded and the model_T is treated as a classifier that must have a predict_proba
        method.

    categories : array like or 'auto', optional (default='auto')
        A list of pre-specified treatment categories. If 'auto' then categories are automatically
        recognized at fit time.

    n_jobs : int, optional (default=-1)
        The number of jobs to run in parallel for both :meth:`fit` and :meth:`effect`.
        ``-1`` means using all processors. Since OrthoForest methods are
        computationally heavy, it is recommended to set `n_jobs` to -1.

    backend : 'threading' or 'multiprocessing'
        What backend should be used for parallelization with the joblib library.

    random_state : int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    """

    return CausalForestDML(
        model_t=model_T,
        model_y=model_Y,
        cv=cv,
        discrete_treatment=discrete_treatment,
        categories=categories,
        n_estimators=n_trees,
        criterion='het',
        min_samples_leaf=min_leaf_size,
        max_depth=max_depth,
        max_samples=subsample_ratio / 2,
        min_balancedness_tol=.3,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state
    )
