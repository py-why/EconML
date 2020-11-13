.. _druserguide:

======================
Doubly Robust Learning
======================

What is it?
==================================

Doubly Robust Learning, similar to Double Machine Learning, is a method for estimating (heterogeneous) treatment effects when
the treatment is categorical and all potential confounders/controls (factors that simultaneously had a direct effect on the treatment decision in the
collected data and the observed outcome) are observed, but are either too many (high-dimensional) for
classical statistical approaches to be applicable or their effect on 
the treatment and outcome cannot be satisfactorily modeled by parametric functions (non-parametric).
Both of these latter problems can be addressed via machine learning techniques (see e.g. [Chernozhukov2016]_, [Foster2019]_).
The method dates back to the early works of [Robins1994]_, [Bang]_ (see [Tsiatis]_ for more details), which applied
the method primarily for the estimation of average treatment effects. In this library we implement recent modifications
to the doubly robust approach that allow for the estimation of heterogeneous treatment effects (see e.g. [Foster2019]_).
The method has also been recently heavily used in the context of policy learning (see e.g. [Dudik2014]_, [Athey2017]_).

It reduces the problem to first estimating *two predictive tasks*: 

    1) predicting the outcome from the treatment and controls,
    2) predicting the treatment from the controls;

Thus unlike Double Machine Learning the first model predicts the outcome from both the treatment and the controls as
opposed to just the controls. Then the method combines these two predictive models in a final stage estimation so as to create a
model of the heterogeneous treatment efffect. The approach allows for *arbitrary Machine Learning algorithms* to be
used for the two predictive tasks, while maintaining many favorable statistical properties related to the final
model (e.g. small mean squared error, asymptotic normality, construction of confidence intervals). The latter
favorable statsitical properties hold if either the first or the second of the two predictive tasks achieves small mean
squared error (hence the name doubly robust).

Our package offers several variants for the final model estimation. Many of these variants also
provide *valid inference* (confidence interval construction) for measuring the uncertainty of the learned model.


What are the relevant estimator classes?
========================================

This section describes the methodology implemented in the classes, :class:`.DRLearner`,
:class:`.LinearDRLearner`,
:class:`.SparseLinearDRLearner`, :class:`.ForestDRLearner`.
Click on each of these links for a detailed module documentation and input parameters of each class.


When should you use it?
==================================

Suppose you have observational (or experimental from an A/B test) historical data, where some treatment/intervention/action
:math:`T` from among a finite set of treatments was chosen and some outcome(s) :math:`Y` was observed and all the variables :math:`W` that could have
potentially gone into the choice of :math:`T`, and simultaneously could have had a direct effect on the outcome
:math:`Y` (aka controls or confounders) are also recorder in the dataset.

If your goal is to understand what was the effect of each of the treatments on the outcome as a function of a set of observable
characteristics :math:`X` of the treated samples, then one can use this method. For instance call:

.. testsetup::

    import numpy as np
    X = np.random.choice(np.arange(5), size=(100,3))
    Y = np.random.normal(size=(100,2))
    y = np.random.normal(size=(100,))
    T = np.random.choice(np.arange(2), size=(100,))
    t0 = T0 = np.zeros((100,))
    t1 = T1 = np.ones((100,))
    W = np.random.normal(size=(100,2))

.. testcode::

    from econml.drlearner import LinearDRLearner
    est = LinearDRLearner()
    est.fit(y, T, X=X, W=W)
    est.effect(X, T0=t0, T1=t1)

This way an optimal treatment policy can be learned, by simply inspecting for which :math:`X` the effect was positive.


Overview of Formal Methodology
==================================

The model's assumpitons are better explained in the language of potential outcomes. If we denote with :math:`Y^{(t)}` the potential outcome that
we would have observed had we treated the sample with treatment :math:`T=t`, then the approach assumes that:

.. math::

    Y^{(t)} =~& g_t(X, W) + \epsilon_t ~~~&~~~ \E[\epsilon | X, W] = 0 \\
    \Pr[T = t | X, W] =~& p_t(X, W) & \\
    \{Y^{(t)}\}_{t=1}^{n_t} \perp T | X, W

It makes no further structural assumptions on :math:`g_t` and :math:`p_t` and estimates them 
non-parametrically using arbitrary non-parametric Machine Learning methods. Our goal is to estimate
the CATE associated with each possible treatment :math:`t \in \{1, \ldots, n_t\}`, as compared to some baseline
treatment :math:`t=0`, i.e.: 

.. math::

    \theta_t(X) = \E[Y^{(t)} - Y^{(0)} | X] = \E[g_t(X, W) - g_0(X, W) | X]

One way to estimate :math:`\theta_t(X)` is the *Direct Method* (DM) approach,
where we simply estimate a regression,
regresstin :math:`Y` on :math:`T, X, W` to learn a model
of :math:`g_T(X, W) = \E[Y | T, X, W]` and then evaluate :math:`\theta_t(X)` by regressing

.. math::

    Y_{i, t}^{DM} = g_t(X_i, W_i) - g_0(X_i, W_i)

on :math:`X`. The main problem with this approach is that it is heavily dependend
on the model-based extrapolation that is implicitly done via the model that is fitted in the regression. Essentially,
when we evaluate :math:`g_t(X, W)` on a sample with features :math:`X, W` for which we gave some other treatment
:math:`T=t'`, then we are extrapolating from other samples with similar :math:`X, W`, which received the treatment
:math:`T=t`. However, the definition of "similarity" is very model based and in some cases we might even be extrapolating
from very far away points (e.g. if we fit linear regression models).

An alternative approach that does not suffer from the aforementioned problems is the *Inverse Propensity Score* (IPS)
approach. This method starts from the realization that, due to the unconfoundedness assumption, we can create
an unbiased estimate of every potential outcome by re-weighting each sample by the inverse probability of that
sample receiving the treatment we observed (i.e. up-weighting samples that have "surprising" treatment assignments).
More concretely, if we let:

.. math::

    Y_{i, t}^{IPS} = \frac{Y_i 1\{T_i=t\}}{\Pr[T_i=t | X_i, W_i]} = \frac{Y_i 1\{T_i=t\}}{p_t(X_i, W_i)} 

then it holds that:

.. math::

    \E[Y_{i, t}^{IPS} | X, W] =~& \E\left[\frac{Y_i 1\{T_i=t\}}{p_t(X_i, W_i)} | X_i, W_i\right] = \E\left[\frac{Y_i^{(t)} 1\{T_i=t\}}{p_t(X_i, W_i)} | X_i, W_i\right]\\
    =~&  \E\left[\frac{Y_i^{(t)} \E[1\{T_i=t\} | X_i, W_i]}{p_t(X_i, W_i)} | X_i, W_i\right] = \E\left[Y_i^{(t)} | X_i, W_i\right]


Thus we can estimate a :math:`\theta_t(X)` by regressing :math:`Y_{i, t}^{IPS} - Y_{i, 0}^{IPS}` on :math:`X`. This
method has two drawbacks: 1) first, even if we knew the probability of treatment :math:`p_t(X, W)`, the approach has
high variance, because we are dividing the observation by a relatively small number (especially if some regions
of :math:`X, W`, some treatments are quite unlikely), 2) second, in observational data we typically don't know
the probability of treatment and thereby we also need to estimate a model for the probability of treatment.
This corresponds to a multi-class classification task, which when :math:`X, W` are high dimensional or when we
use non-linear models like random forests, could have slow estimation rates. This method will inherit these rates.
Moreover, if we use ML to fit these propensity models, then it is hard to characterize what the limit distribution
of our estimate will be so as to provide valid confidence intervals.

The *Doubly Robust* approach, avoids the above drawbacks by combining the two methods. In particular, it fits 
a direct regression model, but then debiases that model, by applying an Inverse Propensity approach to the
residual of that model, i.e. it constructs the following estimates of the potential outcomes:

.. math::
    Y_{i, t}^{DR} = g_t(X_i, W_i) + \frac{Y_i -g_t(X_i, W_i)}{p_t(X_i, W_i)} \cdot 1\{T_i=t\}

Then we can learn :math:`\theta_t(X)` by regressing :math:`Y_{i, t}^{DR} - Y_{i, 0}^{DR}` on :math:`X_i`.

This yields the overall algorithm: first learn a **regression model** :math:`\hat{g}_t(X, W)`, by running a regression
of :math:`Y` on :math:`T, X, W` and a **propensity model** :math:`\hat{p}_t(X, W)`, by running a classification to predict
:math:`T` from :math:`X, W`. Then construct the doubly robust random variables as described above and regress them on
:math:`X`.

The main advantage of the Doubly Robust method is that the mean squared error of the final estimate :math:`\theta_t(X)`,
is only affected by the product of the mean squared errors of the regression estimate :math:`\hat{g}_t(X, W)` and
the propensity estimate :math:`\hat{p}_t(X, W)`. Thus as long as one of them is accurate then the final model is correct.
For instance, as long as neither of them converges at a rate slower than :math:`n^{-1/4}`, then the final model achieves
parametric rates of :math:`n^{-1/2}`. Moreover, under some further assumption on what estimation algorithm
was used in the final stage, then the final estimate is asymptotically normal and valid confidence intervals can be constructed.
For this theorem to hold, the nuisance
estimates need to be fitted in a cross-fitting manner (see :class:`._OrthoLearner`).
The latter robustness property follows from the fact that the moment equations that correspond to the final 
least squares estimation (i.e. the gradient of the squared loss), satisfy a Neyman orthogonality condition with respect to the
nuisance parameters :math:`q, f`. For a more detailed exposition of how Neyman orthogonality 
leads to robustness we refer the reader to [Chernozhukov2016]_, [Mackey2017]_, [Nie2017]_, [Chernozhukov2017]_,
[Chernozhukov2018]_, [Foster2019]_. In fact, the doubly robust estimator satisfies a slightly stronger property
then Neyman orthogonality, which is why it possess the stronger robustness guarantee that only the product
of the two mean squared errors of the first stage models, matter for the error and the distributional properties
of the final estimator.

The other advantage of the Doubly Robust method compared to the DML method, is that the final regression is meaningful
even if the space of functions over which we minimize the final regression loss does not contain the true CATE function.
In that case, the method will estimate the projection of the CATE function onto the space of models over which
we optimize in the final regression. For instance, this allows one to perform inference on the best linear projection
of the CATE function or to perform inference on the best CATE function on a subset of the features that could potentially be
creating heterogeneity. For instance, one can use the DR method with a non-parametric final model like an Honest
Forest and perform inference of the marginal treatment effect heterogeneity with respect to a single feature, without
making any further assumptions on how that treatment effect heterogeneity looks like.

The downside of the DR method over DML is that it typically has higher variance, especially when there are regions
of the control space, :math:`X, W`, in which some treatment has a small probability of being assigned (typically referred
to as "small overlap" in the literature). In such settings, the DML method could potentially extrapolate better, as it only
requires good overlap "on-average" to achieve good mean squared error.


Class Hierarchy Structure
==================================

In this library we implement several variants of the Doubly Robust method, dependent on what type of estimation algorithm
is chosen for the final stage. The user can choose any regression/classification method for the first stage models
in all these variants. The hierarchy
structure of the implemented CATE estimators is as follows.

    .. inheritance-diagram:: econml.drlearner.DRLearner econml.drlearner.LinearDRLearner econml.drlearner.SparseLinearDRLearner econml.drlearner.ForestDRLearner
        :parts: 1
        :private-bases:
        :top-classes: econml._ortho_learner._OrthoLearner, econml.cate_estimator.StatsModelsCateEstimatorDiscreteMixin, econml.cate_estimator.DebiasedLassoCateEstimatorDiscreteMixin

Below we give a brief description of each of these classes:

    * **DRLearner.** The class :class:`.DRLearner` makes no assumption on the effect model for each outcome :math:`i`
      and treatment :math:`t`. Any scikit-learn regressor can be used for the final stage estimation. Similarly, any
      scikit-learn regressor can be used for the *regression model* and any scikit-learn classifier can be used
      for the *propensity model*:

      .. testcode::

        from econml.drlearner import DRLearner
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        est = DRLearner(model_regression=GradientBoostingRegressor(),
                        model_propensity=GradientBoostingClassifier(),
                        model_final=GradientBoostingRegressor())
        est.fit(y, T, X=X, W=W)
        point = est.effect(X, T0=T0, T1=T1)

      Examples of models include Random Forests (:class:`~sklearn.ensemble.RandomForestRegressor`),
      Gradient Boosted Forests (:class:`~sklearn.ensemble.GradientBoostingRegressor`) and
      Support Vector Machines (:class:`~sklearn.svm.SVC`). Moreover, one can even use cross validated estimators
      that perform automatic model selection for each of these models:

      .. testcode::

        from econml.drlearner import DRLearner
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        model_reg = lambda: GridSearchCV(
                        estimator=RandomForestRegressor(),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 50, 100)
                            }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        model_clf = lambda: GridSearchCV(
                        estimator=RandomForestClassifier(min_samples_leaf=10),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 50, 100)
                            }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        est = DRLearner(model_regression=model_reg(), model_propensity=model_clf(),
                        model_final=model_reg(), n_splits=5)
        est.fit(y, T, X=X, W=W)
        point = est.effect(X, T0=T0, T1=T1)

      From that respect this estimator is also a *Meta-Learner*, since all steps of the estimation use out-of-the-box ML algorithms. For more information,
      check out :ref:`Meta Learners User Guide <metalearnersuserguide>`. This general method was proposed in [Foster2019]_.

        - **LinearDRLearner.** The child class  :class:`.LinearDRLearner`, uses an unregularized final linear model and  
          essentially works only when the feature vector :math:`\phi(X)` is low dimensional. Given that it is an unregularized
          low dimensional final model, this class also offers confidence intervals via asymptotic normality 
          arguments. This is achieved by essentially using the :class:`.StatsModelsLinearRegression`
          (which is an extension of the scikit-learn LinearRegression estimator, that also supports inference
          functionalities) as a final model. The theoretical foundations of this class essentially follow the arguments in [Chernozhukov2016]_.
          For instance, to get confidence intervals on the effect of going
          from the baseline treatment (assumed to be treatment 0) to any other treatment T1, one can simply call:

          .. testcode::

            from econml.drlearner import LinearDRLearner
            est = LinearDRLearner()
            est.fit(y, T, X=X, W=W)
            point = est.effect(X, T1=t1)
            lb, ub = est.effect_interval(X, T1=t1, alpha=0.05)
            # Get CATE for all treatments
            point = est.const_marginal_effect(X)
            lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)

          One could also construct bootstrap based confidence intervals by setting `inference='bootstrap'`.

        - **SparseLinearDRLearner.** The child class :class:`.SparseLinearDRLearner`, uses an :math:`\ell_1`-regularized final    
          model. In particular, it uses our implementation of the DebiasedLasso algorithm [Buhlmann2011]_ (see :class:`.DebiasedLasso`).
          Using the asymptotic normality properties
          of the debiased lasso, this class also offers asymptotically normal based confidence intervals.
          The theoretical foundations of this class essentially follow the arguments in [Chernozhukov2017]_, [Chernozhukov2018]_.
          For instance, to get confidence intervals on the effect of going
          from any treatment T0 to any other treatment T1, one can simply call:

          .. testcode::

            from econml.drlearner import SparseLinearDRLearner
            est = SparseLinearDRLearner()
            est.fit(y, T, X=X, W=W)
            point = est.effect(X, T1=T1)
            lb, ub = est.effect_interval(X, T1=T1, alpha=0.05)
            # Get CATE for all treatments
            point = est.const_marginal_effect(X)
            lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)

        - **ForestDRLearner.** The child class :class:`.ForestDRLearner` uses a Subsampled Honest Forest regressor
          as a final model (see [Wager2018]_ and [Athey2019]_). The subsampled honest forest is implemented in our library as a scikit-learn extension
          of the :class:`~sklearn.ensemble.RandomForestRegressor`, in the class :class:`.SubsampledHonestForest`. This estimator
          offers confidence intervals via the Bootstrap-of-Little-Bags as described in [Athey2019]_.
          Using this functionality we can also construct confidence intervals for the CATE:

          .. testcode::

            from econml.drlearner import ForestDRLearner
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            est = ForestDRLearner(model_regression=GradientBoostingRegressor(),
                                  model_propensity=GradientBoostingClassifier())
            est.fit(y, T, X=X, W=W)
            point = est.effect(X, T0=T0, T1=T1)
            lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=0.05)

          This method is related to the :class:`.DiscreteTreatmentOrthoForest` and you can check [Oprescu2019]_ for more technical details;
          the main difference being how the nuisance models are being constructed for the CATE estimation at some
          target :math:`X=x`. Check out :ref:`Forest Estimators User Guide <orthoforestuserguide>` for more information on forest based CATE models and other
          alternatives to the :class:`.ForestDML`.


Usage FAQs
==========

- **What if I want confidence intervals?**

    For valid confidence intervals use the :class:`.LinearDRLearner` if the number of features :math:`X`,
    that you want to use for heterogeneity are small compared to the number of samples that you have,
    e.g.:

    .. testcode::

        from econml.drlearner import LinearDRLearner
        est = LinearDRLearner()
        est.fit(y, T, X=X, W=W)
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
        lb, ub = est.coef__interval(T=1, alpha=.05)
        lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=.05)

    If the number of features is comparable or even larger than the number of samples, then use :class:`.SparseLinearDRLearner`,
    with ``inference='debiasedlasso``. If you
    want non-linear models then use :class:`.ForestDRLearner` with ``inference='blb'``.

- **What if I have no idea how heterogeneity looks like?**

    Either use a flexible featurizer, e.g. a polynomial featurizer with many degrees and use
    the :class:`.SparseLinearDRLearner`:

    .. testcode::

        from econml.drlearner import SparseLinearDRLearner
        from sklearn.preprocessing import PolynomialFeatures
        est = SparseLinearDRLearner(featurizer=PolynomialFeatures(degree=3, include_bias=False))
        est.fit(y, T, X=X, W=W)
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
        lb, ub = est.coef__interval(T=1, alpha=.05)
        lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=.05)

    Alternatively, you can also use a forest based estimator such as :class:`.ForestDRLearner`. This 
    estimator can also handle many features, albeit typically smaller number of features than the sparse linear DRLearner.
    Moreover, this estimator essentially performs automatic featurization and can fit non-linear models.

    .. testcode::

        from econml.drlearner import ForestDRLearner
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        est = ForestDRLearner(model_regression=GradientBoostingRegressor(),
                              model_propensity=GradientBoostingClassifier())
        est.fit(y, T, X=X, W=W)
        point = est.effect(X, T0=T0, T1=T1)
        lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=0.05)
        lb, ub = est.const_marginal_effect_interval(X, alpha=0.05)

    If you care more about mean squared error than confidence intervals and hypothesis testing, then use the
    :class:`.DRLearner` class and choose a cross-validated final model (checkout the 
    `Forest Learners Jupyter notebook <https://github.com/microsoft/EconML/blob/master/notebooks/ForestLearners%20Basic%20Example.ipynb>`_ 
    for such an example).
    Also the check out the :ref:`Orthogonal Random Forest User Guide <orthoforestuserguide>` or the
    :ref:`Meta Learners User Guide <metalearnersuserguide>`.

- **What if I have too many features that can create heterogeneity?**

    Use the :class:`.SparseLinearDRLearner` or :class:`.ForestDRLearner` or :class:`.DRLearner`. (see above).

- **What if I have too many features I want to control for?**

    Use first stage models that work well with high dimensional features. For instance, the Lasso or the 
    ElasticNet or gradient boosted forests are all good options (the latter allows for 
    non-linearities in the model but can typically handle fewer features than the former), e.g.:

    .. testcode::

        from econml.drlearner import SparseLinearDRLearner
        from sklearn.linear_model import LassoCV, LogisticRegressionCV, ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = SparseLinearDRLearner(model_regression=LassoCV(),
                                    model_propensity=LogisticRegressionCV())
        est = SparseLinearDRLearner(model_regression=ElasticNetCV(),
                                    model_propensity=LogisticRegressionCV())
        est = SparseLinearDRLearner(model_regression=GradientBoostingRegressor(),
                                    model_propensity=GradientBoostingClassifier())

    The confidence intervals will still be valid, provided that these first stage models achieve small
    mean squared error.

- **What should I use for first stage estimation?**

    See above. The first stage problems are pure predictive tasks, so any ML approach that is relevant for your
    prediction problem is good.

- **How do I select the hyperparameters of the first stage models or the final model?**

    You can use cross-validated models that automatically choose the hyperparameters, e.g. the
    :class:`~sklearn.linear_model.LassoCV` instead of the :class:`~sklearn.linear_model.Lasso`. Similarly,
    for forest based estimators you can wrap them with a grid search CV, :class:`~sklearn.model_selection.GridSearchCV`, e.g.:

    .. testcode::

        from econml.drlearner import DRLearner
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.model_selection import GridSearchCV
        model_reg = lambda: GridSearchCV(
                        estimator=RandomForestRegressor(),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 50, 100)
                            }, cv=5, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        model_clf = lambda: GridSearchCV(
                        estimator=RandomForestClassifier(min_samples_leaf=10),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 50, 100)
                            }, cv=5, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        est = DRLearner(model_regression=model_reg(), model_propensity=model_clf(),
                        model_final=model_reg(), n_splits=5)
        est.fit(y, T, X=X, W=W)
        point = est.effect(X, T0=T0, T1=T1)

- **What if I have many treatments?**

    The method allows for multiple discrete (categorical) treatments and will estimate a CATE model for each treatment.

- **How can I assess the performance of the CATE model?**

    Each of the DRLearner classes have an attribute `score_` after they are fitted. So one can access that
    attribute and compare the performance accross different modeling parameters (lower score is better):

    .. testcode::

        from econml.drlearner import DRLearner
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        est = DRLearner(model_regression=RandomForestRegressor(oob_score=True),
                        model_propensity=RandomForestClassifier(min_samples_leaf=10, oob_score=True),
                        model_final=RandomForestRegressor())
        est.fit(y, T, X=X, W=W)
        est.score_

    This essentially measures the score based on the final stage loss. Moreover, one can assess the out-of-sample score by calling the `score` method on a separate validation sample that was not
    used for training::

        est.score(Y_val, T_val, X_val, W_val)

    Moreover, one can independently check the goodness of fit of the fitted first stage models by
    inspecting the fitted models. You can access the list of fitted first stage models (one for each
    fold of the crossfitting structure) via the methods: `models_t` and `models_y`. Then if those models
    also have a score associated attribute, that can be used as an indicator of performance of the first
    stage. For instance in the case of Random Forest first stages as in the above example, if the `oob_score`
    is set to `True`, then the estimator has a post-fit measure of performance::

        [mdl.oob_score_ for mdl in est.models_regression]

    If one uses cross-validated estimators as first stages, then model selection for the first stage models
    is performed automatically.

- **How should I set the parameter `n_splits`?**

    This parameter defines the number of data partitions to create in order to fit the first stages in a
    crossfittin manner (see :class:`._OrthoLearner`). The default is 2, which
    is the minimal. However, larger values like 5 or 6 can lead to greater statistical stability of the method,
    especially if the number of samples is small. So we advise that for small datasets, one should raise this
    value. This can increase the computational cost as more first stage models are being fitted.



Usage FAQs
==========

Check out the following Jupyter notebooks:

* `Meta Learners Jupyter Notebook <https://github.com/microsoft/EconML/blob/master/notebooks/Metalearners%20Examples.ipynb>`_ 
* `Forest Learners Jupyter Notebook <https://github.com/microsoft/EconML/blob/master/notebooks/ForestLearners%20Basic%20Example.ipynb>`_


