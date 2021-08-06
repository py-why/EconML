.. _dynamicdmluserguide:

===============================
Dynamic Double Machine Learning
===============================

What is it?
==================================

Dynamic Double Machine Learning is a method for estimating (heterogeneous) treatment effects when
treatments are offered over time via an adaptive dynamic policy. It applies to the case when
all potential dynamic confounders/controls (factors that simultaneously had a direct effect on the adaptive treatment
decision in the collected data and the observed outcome) are observed, but are either too many (high-dimensional) for
classical statistical approaches to be applicable or their effect on 
the treatment and outcome cannot be satisfactorily modeled by parametric functions (non-parametric).
Both of these latter problems can be addressed via machine learning techniques (see e.g. [Lewis2021]_).


What are the relevant estimator classes?
========================================

This section describes the methodology implemented in the class
:class:`.DynamicDML`.
Click on each of these links for a detailed module documentation and input parameters of each class.


When should you use it?
==================================

Suppose you have observational (or experimental from an A/B test) historical data, where multiple treatment(s)/intervention(s)/action(s) 
:math:`T` were offered over time to each of the units and some final outcome(s) :math:`Y` was observed and all the variables :math:`W` that could have
potentially gone into the choice of :math:`T`, and simultaneously could have had a direct effect on the outcome :math:`Y` (aka controls or confounders) are also recorder in the dataset.

If your goal is to understand what was the effect of the treatment on the outcome as a function of a set of observable
characteristics :math:`X` of the treated samples, then one can use this method. For instance call:

.. testsetup::

    # DynamicDML
    import numpy as np
    groups = np.repeat(a=np.arange(100), repeats=3, axis=0)
    W_dyn = np.random.normal(size=(300, 1))
    X_dyn = np.random.normal(size=(300, 1))
    T_dyn = np.random.normal(size=(300, 2))
    y_dyn = np.random.normal(size=(300, ))

.. testcode::

    from econml.dynamic.dml import DynamicDML
    est = DynamicDML()
    est.fit(y_dyn, T_dyn, X=X_dyn, W=W_dyn, groups=groups)


Class Hierarchy Structure
==================================

In this library we implement variants of several of the approaches mentioned in the last section. The hierarchy
structure of the implemented CATE estimators is as follows.

    .. inheritance-diagram:: econml.dynamic.dml.DynamicDML
        :parts: 1
        :private-bases:
        :top-classes: econml._OrthoLearner, econml._cate_estimator.LinearModelFinalCateEstimatorMixin

Below we give a brief description of each of these classes:

    * **DynamicDML.** The class :class:`.DynamicDML` is an extension of the Double ML approach for treatments assigned sequentially over time periods.
      This estimator will adjust for treatments that can have causal effects on future outcomes. The data corresponds to a Markov decision process :math:`\{X_t, W_t, T_t, Y_t\}_{t=1}^m`,
      where :math:`X_t, W_t` corresponds to the state at time :math:`t`, :math:`T_t` is the treatment at time :math:`t` and :math:`Y_t` is the observed outcome at time :math:`t`.

      The model makes the following structural equation assumptions on the data generating process:

      .. math::

        XW_t =~& A \cdot T_{t-1} + B \cdot XW_{t-1} + \eta_t\\ 
        T_t =~& p(T_{t-1}, XW_t, \zeta_t) \\
        Y_t =~& \theta_0(X_0)'T_t + \mu'XW_t + \epsilon_t

      where :math:`XW` is the concatenation of the :math:`X` and :math:`W` variables.
      For more details about this model and underlying assumptions, see [Lewis2021]_.

      To learn the treatment effects of treatments in the different periods on the last period outcome, one can simply call:

      .. testcode::

        from econml.dynamic.dml import DynamicDML
        est = DynamicDML()
        est.fit(y_dyn, T_dyn, X=X_dyn, W=W_dyn, groups=groups)



Usage FAQs
==========

See our FAQ section in :ref:`DML User Guide <dmluserguide>`
