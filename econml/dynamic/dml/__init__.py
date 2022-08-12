# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Double Machine Learning for Dynamic Treatment Effects.

A Double/Orthogonal machine learning approach to estimation of heterogeneous
treatment effect in the dynamic treatment regime. For the theoretical
foundations of these methods see: [dynamicdml]_.

References
----------

.. [dynamicdml] Greg Lewis and Vasilis Syrgkanis.
    Double/Debiased Machine Learning for Dynamic Treatment Effects.
    `<https://arxiv.org/abs/2002.07285>`_, 2021.
"""

import econml.panel.dml
from econml.utilities import deprecated


@deprecated("The DynamicDML class has been moved to econml.panel.dml.DynamicDML; "
            "an upcoming release will remove the econml.panel package, please update references to the new location")
def DynamicDML(*,
               model_y='auto', model_t='auto',
               featurizer=None,
               fit_cate_intercept=True,
               linear_first_stages=False,
               discrete_treatment=False,
               categories='auto',
               cv=2,
               mc_iters=None,
               mc_agg='mean',
               random_state=None):
    """CATE estimator for dynamic treatment effect estimation.

    This estimator is an extension of the Double ML approach for treatments assigned sequentially
    over time periods.

    The estimator is a special case of an :class:`_OrthoLearner` estimator, so it follows the two
    stage process, where a set of nuisance functions are estimated in the first stage in a crossfitting
    manner and a final stage estimates the CATE model. See the documentation of
    :class:`._OrthoLearner` for a description of this two stage process.

    Parameters
    ----------
    model_y: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods;
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV` will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, optional (Default=2)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.
          Iterables should make sure a group belongs to a single split.

        For integer/None inputs, :class:`~sklearn.model_selection.GroupKFold` is used

        Unless an iterable is used, we call `split(X, T, groups)` to generate the splits.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    Examples
    --------
    A simple example with default models:

    .. testcode::
        :hide:

        import numpy as np
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.panel.dml import DynamicDML

        np.random.seed(123)

        n_panels = 100  # number of panels
        n_periods = 3  # number of time periods per panel
        n = n_panels * n_periods
        groups = np.repeat(a=np.arange(n_panels), repeats=n_periods, axis=0)
        X = np.random.normal(size=(n, 1))
        T = np.random.normal(size=(n, 2))
        y = np.random.normal(size=(n, ))
        est = DynamicDML()
        est.fit(y, T, X=X, W=None, groups=groups, inference="auto")

    >>> est.const_marginal_effect(X[:2])
    array([[-0.336..., -0.048..., -0.061...,  0.042..., -0.204...,
         0.00667271],
        [-0.101...,  0.433...,  0.054..., -0.217..., -0.101...,
         -0.159...]])
    >>> est.effect(X[:2], T0=0, T1=1)
    array([-0.601..., -0.091...])
    >>> est.effect(X[:2], T0=np.zeros((2, n_periods*T.shape[1])), T1=np.ones((2, n_periods*T.shape[1])))
    array([-0.601..., -0.091...])
    >>> est.coef_
    array([[ 0.112...],
       [ 0.231...],
       [ 0.055...],
       [-0.125...],
       [ 0.049...],
       [-0.079...]])
    >>> est.coef__interval()
    (array([[-0.063...],
           [-0.009...],
           [-0.114...],
           [-0.413...],
           [-0.117...],
           [-0.262...]]), array([[0.289...],
           [0.471...],
           [0.225...],
           [0.163...],
           [0.216...],
           [0.103...]]))
    """
    return econml.panel.dml.DynamicDML(
        model_y=model_y, model_t=model_t,
        featurizer=featurizer,
        fit_cate_intercept=fit_cate_intercept,
        linear_first_stages=linear_first_stages,
        discrete_treatment=discrete_treatment,
        categories=categories,
        cv=cv,
        mc_iters=mc_iters,
        mc_agg=mc_agg,
        random_state=random_state)
