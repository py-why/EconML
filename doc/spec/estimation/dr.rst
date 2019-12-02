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
    T = T0 = T1 = np.random.choice(np.arange(3), size=(100,2))
    t = t0 = t1 = T[:,0]
    W = np.random.normal(size=(100,2))

.. testcode::

    from econml.dml import LinearDRLearner
    est = LinearDRLearner()
    est.fit(y, T, X, W)
    est.const_marginal_effect(X)

This way an optimal treatment policy can be learned, by simply inspecting for which :math:`X` the effect was positive.


Overview of Formal Methodology
==================================

The model's assumpitons are better explained in the language of potential outcomes. If we denote with :math:`Y^{(t)}` the potential outcome that
we would have observed had we treated the sample with treatment :math:`T=t`, then the approach assumes that:

.. math::

    Y^{(t)} = g_t(X, W) + \epsilon_t ~~~&~~~ \E[\epsilon | X, W] = 0 \\
    \Pr[T = t | X, W] =~& p_t(X, W) & \\
    \{Y^{(t)}\}_{t=1}^{n_t} \perp T | X, W

It makes no further structural assumptions on :math:`g_t` and :math:`p_t` and estimates them 
non-parametrically using arbitrary non-parametric Machine Learning methods. Our goal is to estimate
the CATE associated with each possible treatment :math:`t \in \{1, \ldots, n_t\}`, as compared to some baseline
treatment :math:`t=0`, i.e.: 

.. math::

    \theta_t(X) = \E[Y^{(t)} - Y^{(0)} | X] = \E[g_t(X, W) - g_0(X, W) | X]

One way to estimate :math:`\theta_t(X)` is the Direct Method (DM) approach, where we simply estimate a regression,
regresstin :math:`Y` on :math:`T, X, W` to learn a model
of :math:`g_T(X, W) = \E[Y | T, X, W]` and then evaluate :math:`\theta_t(X)` by regressing

.. math::

    Y_{i, t}^{DM} = g_t(X, W) - g_0(X, W)

on :math:`X`. The main problem with this approach is that it is heavily dependend
on the model-based extrapolation that is implicitly done via the model that is fitted in the regression. Essentially,
when we evaluate :math:`g_t(X, W)` on a sample with features :math:`X, W` for which we gave some other treatment
:math:`T=t'`, then we are extrapolating from other samples with similar :math:`X, W`, which received the treatment
:math:`T=t`. However, the definition of "similarity" is very model based and in some cases we might even be extrapolating
from very far away points (e.g. if we fit linear regression models).

An alternative approach that does not suffer from the aforementioned problems is the Inverse Propensity Score (IPS)
approach. This method starts from the realization that, due to the unconfoundedness assumption, we can create
an unbiased estimate of every potential outcome by re-weighting each sample by the inverse probability of that
sample receiving the treatment we observed (i.e. up-weighting samples that have "surprising" treatment assignments).
More concretely, if we let:

.. math::

    Y_{i, t}^{IPS} = \frac{Y 1\{T=t\}}{\Pr[T=t | X, W]}

then it holds that:

.. math::

    \E[Y_{i, t}^{IPS} | X, W] =& \E\left[\frac{Y 1\{T=t\}}{\Pr[T=t | X, W]} | X, W\right] \\
    =& \E\left[\frac{Y^{(t)} 1\{T=t\}}{\Pr[T=t | X, W]} | X, W\right] \\
    =&  \E\left[\frac{Y^{(t)} \E[1\{T=t\} | X, W]}{\Pr[T=t | X, W]} | X, W] = \E\left[\frac{Y^{(t)} | X, W]

Thus we can estimate a :math:`\theta_t(X)` by regressing :math:`Y_{i, t}^{IPS} - Y_{i, 0}^{IPS}` on :math:`X`.

The idea to estimate :math:`\theta(X)` is as follows:
In this estimator, the CATE is estimated by using the following estimating equations. If we let:

    .. math ::
        Y_{i, t}^{DR} = E[Y | X_i, W_i, T_i]\
            + \\sum_{t=0}^{n_t} \\frac{Y_i - E[Y | X_i, W_i, T_i]}{Pr[T=t | X_i, W_i]} \\cdot 1\\{T_i=t\\}

Then the following estimating equation holds:

    .. math ::
        E\\left[Y_{i, t}^{DR} - Y_{i, 0}^{DR} | X_i\\right] = \\theta_t(X_i)

Thus if we estimate the nuisance functions :math:`h(X, W, T) = E[Y | X, W, T]` and
:math:`p_t(X, W)=Pr[T=t | X, W]` in the first stage, we can estimate the final stage cate for each
treatment t, by running a regression, regressing :math:`Y_{i, t}^{DR} - Y_{i, 0}^{DR}` on :math:`X_i`.

The problem of estimating the nuisance function :math:`p` is a simple multi-class classification
problem of predicting the label :math:`T` from :math:`X, W`. The :class:`.DRLearner`
class takes as input the parameter ``model_propensity``, which is an arbitrary scikit-learn
classifier, that is internally used to solve this classification problem.

The second nuisance function :math:`h` is a simple regression problem and the :class:`.DRLearner`
class takes as input the parameter model_regressor, which is an arbitrary scikit-learn regressor that
is internally used to solve this regression problem.

The final stage is multi-task regression problem with outcomes the labels :math:`Y_{i, t}^{DR} - Y_{i, 0}^{DR}`
for each non-baseline treatment t. The :class:`.DRLearner` takes as input parameter
``model_final``, which is any scikit-learn regressor that is internally used to solve this multi-task
regresion problem. If the parameter ``multitask_model_final`` is False, then this model is assumed
to be a mono-task regressor, and separate clones of it are used to solve each regression target
separately.

The main advantage of DML is that if one makes parametric assumptions on :math:`\theta(X)`, then one achieves fast estimation rates and 
asymptotic normality on the second stage estimate :math:`\hat{\theta}`, even if the first stage estimates on :math:`q(X, W)` 
and :math:`f(X, W)` are only :math:`n^{1/4}` consistent, in terms of RMSE. For this theorem to hold, the nuisance
estimates need to be fitted in a cross-fitting manner (see :class:`._OrthoLearner`).
The latter robustness property follows from the fact that the moment equations that correspond to the final 
least squares estimation (i.e. the gradient of the squared loss), satisfy a Neyman orthogonality condition with respect to the
nuisance parameters :math:`q, f`. For a more detailed exposition of how Neyman orthogonality 
leads to robustness we refer the reader to [Chernozhukov2016]_, [Mackey2017]_, [Nie2017]_, [Chernozhukov2017]_,
[Chernozhukov2018]_, [Foster2019]_. 

Class Hierarchy Structure
==================================

In this library we implement variants of several of the approaches mentioned in the last section. The hierarchy
structure of the implemented CATE estimators is as follows.

    .. inheritance-diagram:: econml.drlearner.DRLearner econml.drlearner.LinearDRLearner econml.drlearner.SparseLinearDRLearner econml.drlearner.ForestDRLearner
        :parts: 1
        :private-bases:
        :top-classes: econml._ortho_learner._OrthoLearner, econml.cate_estimator.StatsModelsCateEstimatorDiscreteMixin, econml.cate_estimator.DebiasedLassoCateEstimatorDiscreteMixin

Below we give a brief description of each of these classes:

    * *DMLCateEstimator.* The class :class:`.DMLCateEstimator` assumes that the effect model for each outcome :math:`i` and treatment :math:`j` is linear, i.e. takes the form :math:`\theta_{ij}(X)=\langle \theta_{ij}, \phi(X)\rangle`, and allows for any arbitrary scikit-learn linear estimator to be defined as the final stage (e.g.    
      :class:`~sklearn.linear_model.ElasticNet`, :class:`~sklearn.linear_model.Lasso`, :class:`~sklearn.linear_model.LinearRegression` and their multi-task variations in the case where we have mulitple outcomes, i.e. :math:`Y` is a vector). The final linear model will be fitted on features that are derived by the Kronecker-product
      of the vectors :math:`T` and :math:`\phi(X)`, i.e. :math:`\tilde{T}\otimes \phi(X) = \mathtt{vec}(\tilde{T}\cdot \phi(X)^T)`. This regression will estimate the coefficients :math:`\theta_{ijk}` 
      for each outcome :math:`i`, treatment :math:`j` and feature :math:`k`. The final model is minimizing a regularized empirical square loss of the form:
      
      .. math::
    
            \hat{\alpha} = \arg\min_{\alpha} \E_n\left[ \left(\tilde{Y} - \Theta \cdot \tilde{T}\otimes \phi(X)\right)^2 \right] + \lambda R(\Theta)

      for some strongly convex regularizer :math:`R`, where :math:`\Theta` is the parameter matrix of dimensions (number of outcomes, number of treatments * number of features). For instance, if :math:`Y` is single dimensional and the lasso is used as model final, i.e.:

      .. testcode::
      
        from econml.dml import DMLCateEstimator
        from sklearn.linear_model import LassoCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                               model_t=GradientBoostingRegressor(),    
                               model_final=LassoCV())

      then :math:`R(\Theta) =\|\Theta\|_1`, 
      if ElasticNet is used as model final, i.e.:

      .. testcode::    

        from econml.dml import DMLCateEstimator
        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                               model_t=GradientBoostingRegressor(),
                               model_final=ElasticNetCV())

      then :math:`R(\Theta)=\kappa \|\Theta\|_2 + (1-\kappa)\|\Theta\|_1`. For multi-dimensional :math:`Y`, 
      one can impose several extensions to the matrix of parameters :math:`\alpha`, such as the one corresponding to the MultiTask Lasso 
      :math:`\sum_{j} \sum_{i} \theta_{ij}^2` or MultiTask ElasticNet or nuclear norm regularization  [Jaggi2010]_, which enforces low-rank 
      constraints on the matrix :math:`\Theta`.
      This essentially implements the techniques analyzed in [Chernozhukov2016]_, [Nie2017]_, [Chernozhukov2017]_, [Chernozhukov2018]_
        
        - *LinearDMLCateEstimator.* The child class  :class:`.LinearDMLCateEstimator`, uses an unregularized final linear model and  
          essentially works only when the feature vector :math:`\phi(X)` is low dimensional. Given that it is an unregularized
          low dimensional final model, this class also offers confidence intervals via asymptotic normality 
          arguments. This is achieved by essentially using the :class:`.StatsModelsLinearRegression`
          (which is an extension of the scikit-learn LinearRegression estimator, that also supports inference
          functionalities) as a final model. The theoretical foundations of this class essentially follow the arguments in [Chernozhukov2016]_.
          For instance, to get confidence intervals on the effect of going
          from any treatment T0 to any other treatment T1, one can simply call:

          .. testcode::

            est = LinearDMLCateEstimator()
            est.fit(y, T, X, W, inference='statsmodels')
            point = est.effect(X, T0=T0, T1=T1)
            lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=0.05)

          One could also construct bootstrap based confidence intervals by setting `inference='bootstrap'`.

        - *SparseLinearDMLCateEstimator.* The child class :class:`.SparseLinearDMLCateEstimator`, uses an :math:`\ell_1`-regularized final    
          model. In particular, it uses an implementation of the DebiasedLasso algorithm [Buhlmann2011]_ (see :class:`.DebiasedLasso`). Using the asymptotic normality properties
          of the debiased lasso, this class also offers asymptotically normal based confidence intervals.
          The theoretical foundations of this class essentially follow the arguments in [Chernozhukov2017]_, [Chernozhukov2018]_.
          For instance, to get confidence intervals on the effect of going
          from any treatment T0 to any other treatment T1, one can simply call:

          .. testcode::

            from econml.dml import SparseLinearDMLCateEstimator
            est = SparseLinearDMLCateEstimator()
            est.fit(y, T, X, W, inference='debiasedlasso')
            point = est.effect(X, T0=T0, T1=T1)
            lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=0.05)

        - *KernelDMLCateEstimator.* The child class :class:`.KernelDMLCateEstimator` performs a variant of the RKHS approach proposed in 
          [Nie2017]_. It approximates any function in the RKHS by creating random Fourier features. Then runs a ElasticNet
          regularized final model. Thus it approximately implements the results of [Nie2017], via the random fourier feature
          approximate representation of functions in the RKHS. Moreover, given that we use Random Fourier Features this class
          asssumes an RBF kernel.
    
    - *_RLearner.* The internal private class :class:`._RLearner` is a parent of the :class:`.DMLCateEstimator`
      and allows the user to specify any way of fitting a final model that takes as input the residual :math:`\tilde{T}`,
      the features :math:`X` and predicts the residual :math:`\tilde{Y}`. Moreover, the nuisance models take as input
      :math:`X` and :math:`W` and predict :math:`T` and :math:`Y` respectively. Since these models take non-standard
      input variables, one cannot use out-of-the-box scikit-learn estimators as inputs to this class. Hence, it is
      slightly more cumbersome to use, which is the reason why we designated it as private. However, if one wants to
      fit for instance a neural net model for :math:`\theta(X)`, then this class can be used (see the implementation
      of the :class:`.DMLCateEstimator` of how to wrap sklearn estimators and pass them as inputs to the
      :class:`._RLearner`. This private class essentially follows the general arguments and
      terminology of the RLearner presented in [Nie2017]_, and allows for the full flexibility of the final model
      estimation that is presented in [Foster2019]_.



Usage FAQs
==========

- **What if I want confidence intervals?**

    For valid confidence intervals use the :class:`.LinearDMLCateEstimator` if the number of features :math:`X`,
    that you want to use for heterogeneity are small compared to the number of samples that you have. If the number of
    features is comparable to the number of samples, then use :class:`.SparseLinearDMLCateEstimator`.
    e.g.:

    .. testcode::

        from econml.dml import LinearDMLCateEstimator
        est = LinearDMLCateEstimator()
        est.fit(y, T, X, W, inference='statsmodels')
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
        lb, ub = est.coef__interval(alpha=.05)
        lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=.05)

- **Why not just run a simple big linear regression with all the treatments, features and controls?**

    If you want to estimate an average treatment effect with accompanied confidence intervals then one
    potential approach one could take is simply run a big linear regression, regressing :math:`Y` on
    :math:`T, X, W` and then looking at the coefficient associated with the :math:`T` variable and
    the corresponding confidence interval (e.g. using statistical packages like
    :class:`~statsmodels.api.OLS`). However, this will not work if:

        1) The number of control variables :math:`X, W` that you have is large and comparable
        to the number of samples. This could for instance arise if one wants to control for
        unit fixed effects, in which case the number of controls is at least the number of units.
        In such high-dimensional settings, ordinary least squares (OLS) is not a reasonable approach.
        Typically, the covariance matrix of the controls, will be ill-posed and the inference
        will be invalid. The DML method bypasses this by using ML approaches to appropriately
        regularize the estimation and provide better models on how the controls affect the outcome,
        given the number of samples that you have.

        2) The effect of the variables :math:`X, W` on the outcome :math:`Y` is not linear.
        In this case, OLS will not provide a consistent model, which could lead to heavily
        biased effect results. The DML approach, when combined with non-linear first stage
        models, like Random Forests or Gradient Boosted Forests, can capture such non-linearities
        and provide unbiased estimates of the effect of :math:`T` on :math:`Y`. Moreover,
        it does so in a manner that is robust to the estimation mistakes that these ML algorithms
        might be making.
    
    Moreover, one may typically want to estimate treatment effect hetergoeneity,
    which the above OLS approach wouldn't provide. One potential way of providing such heterogeneity
    is to include product features of the form :math:`X\cdot T` in the OLS model. However, then
    one faces again the same problems as above:

        1) If effect heterogeneity does not have a linear form, then this approach is not valid.
        One might want to then create more complex featurization, in which case the problem could
        become too high-dimensional for OLS. Our :class:`.SparseLinearDMLCateEstimator`
        can handle such settings via the use of the debiased Lasso. Also see the :ref:`Orthogonal Random Forest User Guide <orthoforestuserguide>` or, if your treatment is categorical, then also check the :ref:`Meta Learners User Guide <metalearnersuserguide>`, if you want even more flexible CATE models.

        2) If the number of features :math:`X` is comparable to the number of samples, then even
        with a linear model, the OLS approach is not feasible or has very small statistical power.


- **What if I have no idea how heterogeneity looks like?**

    Either use a flexible featurizer, e.g. a polynomial featurizer with many degrees and use
    the :class:`.SparseLinearDMLCateEstimator`:

    .. testcode::

        from econml.dml import SparseLinearDMLCateEstimator
        from sklearn.preprocessing import PolynomialFeatures
        est = SparseLinearDMLCateEstimator(featurizer=PolynomialFeatures(degree=4))
        est.fit(y, T, X, W, inference='debiasedlasso')
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
    
    Alternatively, if your number of features :math:`X` is small compared to your number of samples, then
    you can look into the check out the :ref:`Orthogonal Random Forest User Guide <orthoforestuserguide>` or the
    :ref:`Meta Learners User Guide <metalearnersuserguide>`.

- **What if I have too many features that can create heterogeneity?**

    Use the :class:`.SparseLinearDMLCateEstimator` (see above).

- **What if I have too many features I want to control for?**

    Use first stage models that work well with high dimensional features. For instance, the Lasso or the 
    ElasticNet or gradient boosted forests are all good options (the latter allows for 
    non-linearities in the model but can typically handle fewer features than the former), e.g.:

    .. testcode::

        from econml.dml import SparseLinearDMLCateEstimator
        from sklearn.linear_model import LassoCV, ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = SparseLinearDMLCateEstimator(model_y=LassoCV(), model_t=LassoCV())
        est = SparseLinearDMLCateEstimator(model_y=ElasticNetCV(), model_t=ElasticNetCV())
        est = SparseLinearDMLCateEstimator(model_y=GradientBoostingRegressor(),
                                           model_t=GradientBoostingRegressor())
    
    The confidence intervals will still be valid, provided that these first stage models achieve small
    mean squared error.

- **What should I use for first stage estimation?**

    See above. The first stage problems are pure predictive tasks, so any ML approach that is relevant for your
    prediction problem is good.

- **How do I select the hyperparameters of the first stage models?**

    You can use cross-validated models that automatically choose the hyperparameters, e.g. the
    :class:`~sklearn.linear_model.LassoCV` instead of the :class:`~sklearn.linear_model.Lasso`. Similarly,
    for forest based estimators you can wrap them with a grid search CV, :class:`~sklearn.model_selection.GridSearchCV`, e.g.:

    .. testcode::

        from econml.dml import DMLCateEstimator
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        first_stage = lambda: GridSearchCV(
                        estimator=RandomForestRegressor(),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),
                                'max_features': (2,4,6)
                            }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        est = SparseLinearDMLCateEstimator(model_y=first_stage(), model_t=first_stage())

- **How do I select the hyperparameters of the final model (if any)?**

    You can use cross-validated classes for the final model too. Our default debiased lasso performs cross validation
    for hyperparameter selection. For custom final models you can also use CV versions, e.g.:

    .. testcode::

        from econml.dml import DMLCateEstimator
        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = DMLCateEstimator(model_y=GradientBoostingRegressor(),
                               model_t=GradientBoostingRegressor(),
                               model_final=ElasticNetCV())
        est.fit(y, t, X, W)
        point = est.const_marginal_effect(X)
        point = est.effect(X, T0=t0, T1=t1)

- **What if I have many treatments?**

    The method is going to assume that each of these treatments enters linearly into the model. So it cannot capture complementarities or substitutabilities
    of the different treatments. For that you can also create composite treatments that look like the product 
    of two base treatments. Then these product will enter in the model and an effect for that product will be estimated.
    This effect will be the substitute/complement effect of both treatments being present, i.e.:

    .. testcode::

        from econml.dml import LinearDMLCateEstimator
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        est = LinearDMLCateEstimator()
        T_composite = poly.fit_transform(T)
        est.fit(y, T_composite, X, W)
        point = est.const_marginal_effect(X)
        est.effect(X, T0=poly.transform(T0), T1=poly.transform(T1)) 

    If your treatments are too many, then you can use the :class:`.SparseLinearDMLCateEstimator`. However,
    this method will essentially impose a regularization that only a small subset of them has any effect.

- **What if my treatments are continuous and don't have a linear effect on the outcome?**

    You can create composite treatments and add them as extra treatment variables (see above). This would require
    imposing a particular form of non-linearity.

- **What if my treatment is categorical/binary?**

    You can simply set `discrete_treatment=True` in the parameters of the class. Then use any classifier for 
    `model_t`, that has a `predict_proba` method:

    .. testcode::

        from econml.dml import LinearDMLCateEstimator
        from sklearn.linear_model import LogisticRegressionCV
        est = LinearDMLCateEstimator(model_t=LogisticRegressionCV(), discrete_treatment=True)
        est.fit(y, t, X, W)
        point = est.const_marginal_effect(X)
        est.effect(X, T0=t0, T1=t1)

- **How can I assess the performance of the CATE model?**

    Each of the DML classes have an attribute `score_` after they are fitted. So one can access that
    attribute and compare the performance accross different modeling parameters (lower score is better):

    .. testcode::

        from econml.dml import DMLCateEstimator
        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import RandomForestRegressor
        est = DMLCateEstimator(model_y=RandomForestRegressor(oob_score=True),
                               model_t=RandomForestRegressor(oob_score=True),
                               model_final=ElasticNetCV(), featurizer=PolynomialFeatures(degree=1))
        est.fit(y, T, X, W)
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

        [mdl.oob_score_ for mdl in est.models_y]

    If one uses cross-validated estimators as first stages, then model selection for the first stage models
    is performed automatically.

- **How should I set the parameter `n_splits`?**

    This parameter defines the number of data partitions to create in order to fit the first stages in a
    crossfittin manner (see :class:`._OrthoLearner`). The default is 2, which
    is the minimal. However, larger values like 5 or 6 can lead to greater statistical stability of the method,
    especially if the number of samples is small. So we advise that for small datasets, one should raise this
    value. This can increase the computational cost as more first stage models are being fitted.


Usage Examples
==================================

