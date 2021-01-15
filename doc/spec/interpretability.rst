Interpretability
================

Our package offers multiple interpretability tools to better understand the final model CATE.


Tree Interpreter
----------------

Tree Interpreter provides a presentation-ready summary of the key features that explain the biggest differences in responsiveness to an intervention.

:class:`.SingleTreeCateInterpreter` trains a single shallow decision tree for the treatment effect :math:`\theta(X)` you learned from any of
our available CATE estimators on a small set of feature :math:`X` that you are interested to learn heterogeneity from. The model will split on the cutoff
points that maximize the treatment effect difference in each leaf. Finally each leaf will be a subgroup of samples that respond to a treatment differently
from other leaves. 

For instance: 

.. testsetup::

    import numpy as np
    X = np.random.choice(np.arange(5), size=(100,3))
    Y = np.random.normal(size=(100,2))
    y = np.random.normal(size=(100,))
    T = np.random.choice(np.arange(3), size=(100,2))
    t = T[:,0]
    W = np.random.normal(size=(100,2))
    

.. testcode::

    from econml.cate_interpreter import SingleTreeCateInterpreter
    from econml.dml import LinearDML
    est = LinearDML()
    est.fit(y, t, X=X, W=W)
    intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=2, min_samples_leaf=10)
    # We interpret the CATE model's behavior based on the features used for heterogeneity
    intrp.interpret(est, X)
    # Plot the tree
    intrp.plot(feature_names=['A', 'B', 'C'], fontsize=12)

Policy Interpreter
------------------
Policy Interpreter offers similar functionality but taking cost into consideration. 

Instead of fitting a tree to learn groups that have a different treatment effect, :class:`.SingleTreePolicyInterpreter` tries to split the samples into different treatment groups.
So in the case of binary treatments it tries to create sub-groups such that all samples within the group have either all positive effect or all negative effect. Thus it tries to
separate responders from non-responders, as opposed to trying to find groups that have different levels of response.

This way you can construct an interpretable personalized policy where you treat the groups with a postive effect and don't treat the group with a negative effect.
Our policy tree provides the recommended treatment at each leaf node.


For instance: 

.. testcode::

    from econml.cate_interpreter import SingleTreePolicyInterpreter
    # We find a tree-based treatment policy based on the CATE model
    # sample_treatment_costs is the cost of treatment. Policy will treat if effect is above this cost.
    intrp = SingleTreePolicyInterpreter(risk_level=None, max_depth=2, min_samples_leaf=1,min_impurity_decrease=.001)
    intrp.interpret(est, X, sample_treatment_costs=0.02)
    # Plot the tree
    intrp.plot(feature_names=['A', 'B', 'C'], fontsize=12)


SHAP
----

`SHAP <https://shap.readthedocs.io/en/latest/>`_ is a popular open source library for interpreting black-box machine learning
models using the Shapley values methodology (see e.g. [Lundberg2017]_).

Similar to how black-box predictive machine learning models can be explained with SHAP, we can also explain black-box effect
heterogeneity models. This approach provides an explanation as to why a heterogeneous causal effect model produced larger or
smaller effect values for particular segments of the population. Which were the features that lead to such differentiation?
This question is easy to address when the model is succinctly described, such as the case of linear heterogneity models, 
where one can simply investigate the coefficients of the model. However, it becomes hard when one starts using more expressive
models, such as Random Forests and Causal Forests to model effect hetergoeneity. SHAP values can be of immense help to
understand the leading factors of effect hetergoeneity that the model picked up from the training data.

Our package offers seamless integration with the SHAP library. Every CATE estimator has a method `shap_values`, which returns the
SHAP value explanation of the estimators output for every treatment and outcome pair. These values can then be visualized with
the plethora of visualizations that the SHAP library offers. Moreover, whenever possible our library invokes fast specialized
algorithms from the SHAP library, for each type of final model, which can greatly reduce computation times.

For instance:

.. testcode::

    import shap
    from econml.dml import LinearDML
    est = LinearDML()
    est.fit(y, t, X=X, W=W)
    shap_values = est.shap_values(X)
    # local view: explain hetergoeneity for a given observation
    ind=0
    shap.plots.force(shap_values["Y0"]["T0"][ind], matplotlib=True)
    # global view: explain hetergoeneity for a sample of dataset
    shap.summary_plot(shap_values['Y0']['T0'])
