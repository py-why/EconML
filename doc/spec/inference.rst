=================
Inference
=================

\ 

Bootstrap Inference
====================

Every estimator can provide bootstrap based confidence intervals by passing ``inference='bootstrap'`` or
``inference=BootstrapInference(n_bootstrap_samples=100, n_jobs=-1)`` (see :class:`.BootstrapInference`).
These intervals are calculated by training multiple versions of the original estimator on bootstrap subsamples
with replacement. Then the intervals are calculated based on the quantiles of the estimate distribution
across the multiple clones. See also :class:`.BootstrapEstimator` for more details on this.

For instance:

.. testsetup::

    import numpy as np
    X = np.random.choice(np.arange(5), size=(100,3))
    Y = np.random.normal(size=(100,2))
    y = np.random.normal(size=(100,))
    T = T0 = T1 = np.random.choice(np.arange(3), size=(100,2))
    t = t0 = t1 = T[:,0]
    W = np.random.normal(size=(100,2))

.. testcode::

    from econml.dml import NonParamDML
    from sklearn.ensemble import RandomForestRegressor
    est = NonParamDML(model_y=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                                model_t=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                                model_final=RandomForestRegressor(n_estimators=10, min_samples_leaf=10))
    est.fit(y, t, X=X, W=W, inference='bootstrap')
    point = est.const_marginal_effect(X)
    lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)



OLS Inference
====================

For estimators where the final stage CATE estimate is based on an Ordinary Least Squares regression, then we offer
normality-based confidence intervals by default (leaving the setting ``inference='auto'`` unchanged), or by
explicitly setting ``inference='statsmodels'``, or dependent on the estimator one can alter the covariance type calculation via
``inference=StatsModelsInference(cov_type='HC1)`` or ``inference=StatsModelsInferenceDiscrete(cov_type='HC1)``.
See :class:`.StatsModelsInference` and :class:`.StatsModelsInferenceDiscrete` for more details.
This for instance holds for the :class:`.LinearDML` and the :class:`.LinearDRLearner`, e.g.:

.. testcode::

    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor
    est = LinearDML(model_y=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                                 model_t=RandomForestRegressor(n_estimators=10, min_samples_leaf=10))
    est.fit(y, t, X=X, W=W)
    point = est.const_marginal_effect(X)
    lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)

.. testcode::

    from econml.drlearner import LinearDRLearner
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    est = LinearDRLearner(model_regression=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                          model_propensity=RandomForestClassifier(n_estimators=10, min_samples_leaf=10))
    est.fit(y, t, X=X, W=W)
    point = est.effect(X)
    lb, ub = est.effect_interval(X, alpha=0.05)

This inference is enabled by our :class:`.StatsModelsLinearRegression` extension to the scikit-learn 
:class:`~sklearn.linear_model.LinearRegression`.

Debiased Lasso Inference
=========================

For estimators where the final stage CATE estimate is based on a high dimensional linear model with a sparsity
constraint, then we offer confidence intervals using the debiased lasso technique. This for instance
holds for the :class:`.SparseLinearDML` and the :class:`.SparseLinearDRLearner`. You can enable such
intervals by default (leaving the setting ``inference='auto'`` unchanged), or by
explicitly setting ``inference='debiasedlasso'``, e.g.:

.. testcode::

    from econml.dml import SparseLinearDML
    from sklearn.ensemble import RandomForestRegressor
    est = SparseLinearDML(model_y=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                                       model_t=RandomForestRegressor(n_estimators=10, min_samples_leaf=10))
    est.fit(y, t, X=X, W=W)
    point = est.const_marginal_effect(X)
    lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)

.. testcode::

    from econml.drlearner import SparseLinearDRLearner
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    est = SparseLinearDRLearner(model_regression=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                                model_propensity=RandomForestClassifier(n_estimators=10, min_samples_leaf=10))
    est.fit(y, t, X=X, W=W)
    point = est.effect(X)
    lb, ub = est.effect_interval(X, alpha=0.05)


This inference is enabled by our implementation of the :class:`.DebiasedLasso` extension to the scikit-learn
:class:`~sklearn.linear_model.Lasso`.


Subsampled Honest Forest Inference
===================================

For estimators where the final stage CATE estimate is a non-parametric model based on a Random Forest, we offer
confidence intervals via the bootstrap-of-little-bags approach (see [Athey2019]_) for estimating the uncertainty of
an Honest Random Forest. This for instance holds for the :class:`.ForestDML`
and the :class:`.ForestDRLearner`. Such intervals are enabled by leaving inference at its default setting of ``'auto'``
or by explicitly setting ``inference='blb'``, e.g.:

.. testcode::

    from econml.dml import ForestDML
    from sklearn.ensemble import RandomForestRegressor
    est = ForestDML(model_y=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                                 model_t=RandomForestRegressor(n_estimators=10, min_samples_leaf=10))
    est.fit(y, t, X=X, W=W)
    point = est.const_marginal_effect(X)
    lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)

.. testcode::

    from econml.drlearner import ForestDRLearner
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    est = ForestDRLearner(model_regression=RandomForestRegressor(n_estimators=10, min_samples_leaf=10),
                          model_propensity=RandomForestClassifier(n_estimators=10, min_samples_leaf=10))
    est.fit(y, t, X=X, W=W)
    point = est.effect(X)
    lb, ub = est.effect_interval(X, alpha=0.05)

This inference is enabled by our implementation of the :class:`.SubsampledHonestForest` extension to the scikit-learn
:class:`~sklearn.ensemble.RandomForestRegressor`.


OrthoForest Bootstrap of Little Bags Inference
==============================================

For the Orthogonal Random Forest estimators (see :class:`.ContinuousTreatmentOrthoForest`, :class:`.DiscreteTreatmentOrthoForest`), 
we provide confidence intervals built via the bootstrap-of-little-bags approach ([Athey2019]_). This technique is well suited for
estimating the uncertainty of the honest causal forests underlying the OrthoForest estimators. Such intervals are enabled by leaving 
inference at its default setting of ``'auto'`` or by explicitly setting ``inference='blb'``, e.g.:

.. testcode::

    from econml.ortho_forest import ContinuousTreatmentOrthoForest
    from econml.sklearn_extensions.linear_model import WeightedLasso
    est = ContinuousTreatmentOrthoForest(n_trees=10,
                                         min_leaf_size=3,
                                         model_T=WeightedLasso(alpha=0.01),
                                         model_Y=WeightedLasso(alpha=0.01))
    est.fit(y, t, X=X, W=W)
    point = est.const_marginal_effect(X)
    lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)

.. todo::    
    * Subsampling
    * Doubly Robust Gradient Inference
