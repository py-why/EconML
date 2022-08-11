==================================
Scoring
==================================

What is it?
==================================

Scorer based on the RLearner loss. Fits residual models at fit time and calculates residuals of the evaluation data in a cross-fitting manner::

    Yres = Y - E[Y|X, W]
    Tres = T - E[T|X, W]

Then for any given cate model calculates the loss::

    loss(cate) = E_n[(Yres - <cate(X), Tres>)^2]

Also calculates a baseline loss based on a constant treatment effect model, i.e.::

    base_loss = min_{theta} E_n[(Yres - <theta, Tres>)^2]

Returns an analogue of the R-square score for regression::

    score = 1 - loss(cate) / base_loss

This corresponds to the extra variance of the outcome explained by introducing heterogeneity in the effect as captured by the cate model, as opposed to always predicting a constant effect. A negative score, means that the cate model performs even worse than a constant effect model and hints at overfitting during training of the cate model.


What are the relevant estimator classes?
========================================

This section describes the methodology implemented in the class :class:`.RScorer`
Click on each of these links for a detailed module documentation and input parameters of each class.


When should you use it?
==================================

Consider the case where you have just run some causal estimation on observational data with several different estimators. You are interested in comparing the effectiveness of these estimators against each other in order to select the best one. 
In this scenario, the :class:`.RScorer` class can help you get a sense of the quality of fit for each estimator. 

Below is some example code that showcases how you would score a LinearDML estimator.

.. testsetup::

    # DML
    import numpy as np
    from sklearn.model_selection import train_test_split

    X = np.random.choice(np.arange(5), size=(100,3))
    Y = np.random.normal(size=(100,2))
    y = np.random.normal(size=(100,))
    T = T0 = T1 = np.random.choice(np.arange(3), size=(100,2))
    t = t0 = t1 = T[:,0]
    W = np.random.normal(size=(100,2))

    X_train, X_val, T_train, T_val, Y_train, Y_val = train_test_split(X, T, Y, test_size=.4)


.. testcode::

    from econml.dml import LinearDML
    from econml.score import RScorer
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


    reg = lambda: RandomForestRegressor(min_samples_leaf=10)
    clf = lambda: RandomForestClassifier(min_samples_leaf=10)

    est = LinearDML(model_y=RandomForestRegressor(), model_t=RandomForestClassifier(), discrete_treatment=True,
                             linear_first_stages=False, cv=3)
    est.fit(Y_train, T_train, X=X_train)

    scorer = RScorer(
        model_y=RandomForestRegressor(), 
        model_t=RandomForestClassifier(),
        discrete_treatment=True,
        cv=3,
        mc_iters=3, 
        mc_agg='median'
    )
    scorer.fit(Y_val, T_val, X=X_val)

    scorer.score(est)



Overview of Formal Methodology
==================================

x



Usage FAQs
==========

- **What if I want confidence intervals?**

    For valid confidence intervals use the :class:`.LinearDML` if the number of features :math:`X`,
    that you want to use for heterogeneity are small compared to the number of samples that you have. If the number of
    features is comparable to the number of samples, then use :class:`.SparseLinearDML`.
    e.g.:

    .. testcode::

        from econml.dml import LinearDML
        est = LinearDML()
        est.fit(y, T, X=X, W=W)
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
        lb, ub = est.coef__interval(alpha=.05)
        lb, ub = est.effect_interval(X, T0=T0, T1=T1, alpha=.05)
    
    If you have a single dimensional continuous treatment or a binary treatment, then you can also fit non-linear
    models and have confidence intervals by using the :class:`.CausalForestDML`. This class will also
    perform well with high dimensional features, as long as only few of these features are actually relevant.

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
        become too high-dimensional for OLS. Our :class:`.SparseLinearDML`
        can handle such settings via the use of the debiased Lasso. Our :class:`.CausalForestDML` does not
        even need explicit featurization and learns non-linear forest based CATE models, automatically. Also see the
        :ref:`Forest Estimators User Guide <orthoforestuserguide>` and the :ref:`Meta Learners User Guide <metalearnersuserguide>`,
        if you want even more flexible CATE models.

        2) If the number of features :math:`X` is comparable to the number of samples, then even
        with a linear model, the OLS approach is not feasible or has very small statistical power.


- **What if I have no idea how heterogeneity looks like?**

    Either use a flexible featurizer, e.g. a polynomial featurizer with many degrees and use
    the :class:`.SparseLinearDML`:

    .. testcode::

        from econml.dml import SparseLinearDML
        from sklearn.preprocessing import PolynomialFeatures
        est = SparseLinearDML(featurizer=PolynomialFeatures(degree=4, include_bias=False))
        est.fit(y, T, X=X, W=W)
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
    
    Alternatively, you can also use a forest based estimator such as :class:`.CausalForestDML`. This 
    estimator can also handle many features, albeit typically smaller number of features than the sparse linear DML.
    Moreover, this estimator essentially performs automatic featurization and can fit non-linear models.

    .. testcode::

        from econml.dml import CausalForestDML
        from sklearn.ensemble import GradientBoostingRegressor
        est = CausalForestDML(model_y=GradientBoostingRegressor(),
                        model_t=GradientBoostingRegressor())
        est.fit(y, t, X=X, W=W)
        lb, ub = est.const_marginal_effect_interval(X, alpha=.05)
    
    Also the check out the :ref:`Orthogonal Random Forest User Guide <orthoforestuserguide>` or the
    :ref:`Meta Learners User Guide <metalearnersuserguide>`.

- **What if I have too many features that can create heterogeneity?**

    Use the :class:`.SparseLinearDML` or :class:`.CausalForestDML` (see above).

- **What if I have too many features I want to control for?**

    Use first stage models that work well with high dimensional features. For instance, the Lasso or the 
    ElasticNet or gradient boosted forests are all good options (the latter allows for 
    non-linearities in the model but can typically handle fewer features than the former), e.g.:

    .. testcode::

        from econml.dml import SparseLinearDML
        from sklearn.linear_model import LassoCV, ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = SparseLinearDML(model_y=LassoCV(), model_t=LassoCV())
        est = SparseLinearDML(model_y=ElasticNetCV(), model_t=ElasticNetCV())
        est = SparseLinearDML(model_y=GradientBoostingRegressor(),
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

        from econml.dml import SparseLinearDML
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        first_stage = lambda: GridSearchCV(
                        estimator=RandomForestRegressor(),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 30, 50, 100, 200),
                                'max_features': (1,2,3)
                            }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        est = SparseLinearDML(model_y=first_stage(), model_t=first_stage())

    Alternatively, you can pick the best first stage models outside of the EconML framework and pass in the selected models to EconML. 
    This can save on runtime and computational resources. Furthermore, it is statistically more stable since all data is being used for
    hyper-parameter tuning rather than a single fold inside of the DML algorithm (as long as the number of hyperparameter values
    that you are selecting over is not exponential in the number of samples, this approach is statistically valid). E.g.:

    .. testcode::

        from econml.dml import LinearDML
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        first_stage = lambda: GridSearchCV(
                        estimator=RandomForestRegressor(),
                        param_grid={
                                'max_depth': [3, None],
                                'n_estimators': (10, 30, 50, 100, 200),
                                'max_features': (1,2,3)
                            }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                        )
        model_y = first_stage().fit(X, Y).best_estimator_
        model_t = first_stage().fit(X, T).best_estimator_
        est = LinearDML(model_y=model_y, model_t=model_t)


- **How do I select the hyperparameters of the final model (if any)?**

    You can use cross-validated classes for the final model too. Our default debiased lasso performs cross validation
    for hyperparameter selection. For custom final models you can also use CV versions, e.g.:

    .. testcode::

        from econml.dml import DML
        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import GradientBoostingRegressor
        est = DML(model_y=GradientBoostingRegressor(),
                  model_t=GradientBoostingRegressor(),
                  model_final=ElasticNetCV(fit_intercept=False))
        est.fit(y, t, X=X, W=W)
        point = est.const_marginal_effect(X)
        point = est.effect(X, T0=t0, T1=t1)
    
    In the case of :class:`.NonParamDML` you can also use non-linear cross-validated models as model_final:

    .. testcode::

        from econml.dml import NonParamDML
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV
        cv_reg = lambda: GridSearchCV(
                    estimator=RandomForestRegressor(),
                    param_grid={
                            'max_depth': [3, None],
                            'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),
                            'max_features': (1,2,3)
                        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
                    )
        est = NonParamDML(model_y=cv_reg(), model_t=cv_reg(), model_final=cv_reg())


- **What if I have many treatments?**

    The method is going to assume that each of these treatments enters linearly into the model. So it cannot capture complementarities or substitutabilities
    of the different treatments. For that you can also create composite treatments that look like the product 
    of two base treatments. Then these product will enter in the model and an effect for that product will be estimated.
    This effect will be the substitute/complement effect of both treatments being present, i.e.:

    .. testcode::

        from econml.dml import LinearDML
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        est = LinearDML()
        T_composite = poly.fit_transform(T)
        est.fit(y, T_composite, X=X, W=W)
        point = est.const_marginal_effect(X)
        est.effect(X, T0=poly.transform(T0), T1=poly.transform(T1)) 

    If your treatments are too many, then you can use the :class:`.SparseLinearDML`. However,
    this method will essentially impose a regularization that only a small subset of them has any effect.

- **What if my treatments are continuous and don't have a linear effect on the outcome?**

    You can create composite treatments and add them as extra treatment variables (see above). This would require
    imposing a particular form of non-linearity.

- **What if my treatment is categorical/binary?**

    You can simply set `discrete_treatment=True` in the parameters of the class. Then use any classifier for 
    `model_t`, that has a `predict_proba` method:

    .. testcode::

        from econml.dml import LinearDML
        from sklearn.linear_model import LogisticRegressionCV
        est = LinearDML(model_t=LogisticRegressionCV(), discrete_treatment=True)
        est.fit(y, t, X=X, W=W)
        point = est.const_marginal_effect(X)
        est.effect(X, T0=t0, T1=t1)

- **How can I assess the performance of the CATE model?**

    Each of the DML classes have an attribute `score_` after they are fitted. So one can access that
    attribute and compare the performance accross different modeling parameters (lower score is better):

    .. testcode::

        from econml.dml import DML
        from sklearn.linear_model import ElasticNetCV
        from sklearn.ensemble import RandomForestRegressor
        est = DML(model_y=RandomForestRegressor(),
                  model_t=RandomForestRegressor(),
                  model_final=ElasticNetCV(fit_intercept=False), featurizer=PolynomialFeatures(degree=1))
        est.fit(y, T, X=X, W=W)
        est.score_

    This essentially measures the score based on the final stage loss. Moreover, one can assess the out-of-sample score by calling the `score` method on a separate validation sample that was not
    used for training::

        est.score(Y_val, T_val, X_val, W_val)

    Moreover, one can independently check the goodness of fit of the fitted first stage models by
    inspecting the fitted models. You can access the nested list of fitted first stage models (one for each
    fold of the crossfitting structure) via the methods: `models_t` and `models_y`. Then if those models
    also have a score associated attribute, that can be used as an indicator of performance of the first
    stage. For instance in the case of Random Forest first stages as in the above example, if the `oob_score`
    is set to `True`, then the estimator has a post-fit measure of performance::

        [mdl.oob_score_ for mdls in est.models_y for mdl in mdls]

    If one uses cross-validated estimators as first stages, then model selection for the first stage models
    is performed automatically.

- **How should I set the parameter `cv`?**

    This parameter defines the number of data partitions to create in order to fit the first stages in a
    crossfitting manner (see :class:`._OrthoLearner`). The default is 2, which
    is the minimal. However, larger values like 5 or 6 can lead to greater statistical stability of the method,
    especially if the number of samples is small. So we advise that for small datasets, one should raise this
    value. This can increase the computational cost as more first stage models are being fitted.


Usage Examples
==================================

For more extensive examples check out the following notebooks:
`DML Examples Jupyter Notebook <https://github.com/microsoft/EconML/blob/main/notebooks/Double%20Machine%20Learning%20Examples.ipynb>`_,
`Forest Learners Jupyter Notebook <https://github.com/microsoft/EconML/blob/main/notebooks/ForestLearners%20Basic%20Example.ipynb>`_.

.. rubric:: Single Outcome, Single Treatment

We consider some example use cases of the library when :math:`Y` and :math:`T` are :math:`1`-dimensional.

**Random Forest First Stages.**
A classical non-parametric regressor for the first stage estimates is a Random Forest. Using RandomForests in our API is as simple as:

.. testcode::

    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor
    est = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor())
    est.fit(y, T, X=X, W=W)
    pnt_effect = est.const_marginal_effect(X)
    lb_effect, ub_effect = est.const_marginal_effect_interval(X, alpha=.05)
    pnt_coef = est.coef_
    lb_coef, ub_coef = est.coef__interval(alpha=.05)


**Polynomial Features for Heterogeneity.**
Suppose that we believe that the treatment effect is a polynomial of :math:`X`, i.e.

.. math::
    
    Y = (\alpha_0 + \alpha_1 X + \alpha_2 X^2 + \ldots) \cdot T + g(X, W, \epsilon)

Then we can estimate the coefficients :math:`\alpha_i` by running:

.. testcode::

    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import PolynomialFeatures
    est = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor(),
                    featurizer=PolynomialFeatures(degree=4, include_bias=True))
    est.fit(y, T, X=X, W=W)

    # To get the coefficients of the polynomial fitted in the final stage we can
    # access the `coef_` attribute of the fitted second stage model. This would 
    # return the coefficients in front of each term in the vector T⊗ϕ(X).
    est.coef_


**Fixed Effects.**
To add fixed effect heterogeneity, we can create one-hot encodings of the id, which is assumed to be part of the input:

.. testcode::

    from econml.dml import LinearDML
    from sklearn.preprocessing import OneHotEncoder
    # removing one id to avoid colinearity, as is standard for fixed effects
    X_oh = OneHotEncoder(sparse=False).fit_transform(X)[:, 1:]

    est = LinearDML(model_y=RandomForestRegressor(),
                                 model_t=RandomForestRegressor())
    est.fit(y, T, X=X_oh, W=W)
    # The latter will fit a model for θ(x) of the form ̂α_0 + ̂α_1 𝟙{id=1} + ̂α_2 𝟙{id=2} + ...
    # The vector of α can be extracted as follows
    est.coef_

**Custom Features.**
One can also define a custom featurizer, as long as it supports the fit\_transform interface of sklearn.

.. testcode::

    from sklearn.ensemble import RandomForestRegressor
    class LogFeatures(object):
        """Augments the features with logarithmic features and returns the augmented structure"""
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return np.concatenate((X, np.log(1+X)), axis=1)
        def fit_transform(self, X, y=None):
            return self.fit(X).transform(X)

    est = LinearDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestRegressor(),
                    featurizer=LogFeatures())
    est.fit(y, T, X=X, W=W)

We can even create a Pipeline or Union of featurizers that will apply multiply featurizations, e.g. first creating log features and then adding polynomials of them:

.. testcode::

    from econml.dml import LinearDML
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    est = LinearDML(model_y=RandomForestRegressor(), 
                    model_t=RandomForestRegressor(),
                    featurizer=Pipeline([('log', LogFeatures()), 
                                         ('poly', PolynomialFeatures(degree=3))]))
    est.fit(y, T, X=X, W=W)


.. rubric:: Single Outcome, Multiple Treatments

Suppose that we believed that our treatment was affecting the outcome in a non-linear manner. 
Then we could expand the treatment vector to contain also polynomial features:

.. testcode::

    import numpy as np
    est = LinearDML()
    est.fit(y, np.concatenate((T, T**2), axis=1), X=X, W=W)

.. rubric:: Multiple Outcome, Multiple Treatments

In settings like demand estimation, we might want to fit the demand of multiple products as a function of the price of each one of them, i.e. fit the matrix of cross price elasticities. The latter can be done, by simply setting :math:`Y` to be the vector of demands and :math:`T` to be the vector of prices. Then we can recover the 
matrix of cross price elasticities as:

.. testcode::

    from sklearn.linear_model import MultiTaskElasticNet
    est = LinearDML(model_y=MultiTaskElasticNet(alpha=0.1),
                    model_t=MultiTaskElasticNet(alpha=0.1))
    est.fit(Y, T, X=None, W=W)

    # a_hat[i,j] contains the elasticity of the demand of product i on the price of product j
    a_hat = est.const_marginal_effect()

If we have too many products then the cross-price elasticity matrix contains many parameters and we need
to regularize. Given that we want to estimate a matrix, it makes sense in this application to consider
the case where this matrix has low rank: all the products can be embedded in some low dimensional feature
space and the cross-price elasticities is a linear function of these low dimensional embeddings. This corresponds
to well-studied latent factor models in pricing. Our framework can easily handle this by using 
a nuclear norm regularized multi-task regression in the final stage. For instance the 
lightning package implements such a class::

    from econml.dml import DML
    from sklearn.preprocessing import PolynomialFeatures
    from lightning.regression import FistaRegressor
    from sklearn.linear_model import MultiTaskElasticNet

    est = DML(model_y=MultiTaskElasticNet(alpha=0.1),
              model_t=MultiTaskElasticNet(alpha=0.1),
              model_final=FistaRegressor(penalty='trace', C=0.0001),
              fit_cate_intercept=False)
    est.fit(Y, T, X=X, W=W)
    te_pred = est.const_marginal_effect(np.median(X, axis=0, keepdims=True))
    print(te_pred)
    print(np.linalg.svd(te_pred[0]))
