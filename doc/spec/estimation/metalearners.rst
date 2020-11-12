.. _metalearnersuserguide:

==============
Meta-Learners
==============


What is it?
==================================

Metalearners are discrete treatment CATE estimators that model either two response surfaces, :math:`Y(0)` and :math:`Y(1)`, or
multiple response surfaces, :math:`Y(0)` to :math:`Y(K)` separately. For a detailed overview of these methods,
see [Kunzel2017]_. We also describe here more generally all our estimator classes where each
stage of estimation can be an arbitrary ML method (e.g. the DRLearner and the NonParamDML).
Moreover, we also introduce a new meta-learner that uses ideas from Domain Adaptation in Machine Learning (the DomainAdaptationLearner).
These methods fall into the meta-learner category because they simply combine ML methods in a black box manner
so as to get a final stage estimate and do not introduce new estimation components.

For examples of how to use our implemented metelearners check out this
`Metalearners Jupyter notebook <https://github.com/Microsoft/EconML/blob/master/notebooks/Metalearners%20Examples.ipynb>`_. The examples
and documents here are only based on binary treatment setting, but all of these estimators are applicable to multiple treatment settings as well.


What are the relevant estimator classes?
========================================

This section describes the methodology implemented in the classes, :class:`.SLearner`,
:class:`.TLearner`, :class:`.XLearner`, :class:`.DomainAdaptationLearner`, :class:`.NonParamDML`, :class:`.DRLearner`.
Click on each of these links for a detailed module documentation and input parameters of each class.

When should you use it?
==================================

These methods are particularly valuable when one wants full flexibility on what estimation method to use at each 
stage. Moreover, they allow for the user to perform cross-validation for more data-adaptive estimation at each
stage. Hence, they allow the user to do model selection both for nuisance quantities and for the final CATE model.
However, due to their unrestricted flexibility, they typically do not offer valid confidence intervals, since
it is not clear how arbitrary ML methods trade off bias and variance. So one should use these methods primarily
if the target goal is estimating a CATE with a small mean squared error and performing automatic model selection
via cross-validated estimators.

Overview of Formal Methodology
==============================

We present here the reasoning of each estimator for the case of binary treatment. Our package works even for multiple
categorical treatments and each method has a natural extension to multiple treatments, which we omit for succinctness.

T-Learner
-----------------

The T-Learner models :math:`Y(0)`, :math:`Y(1)` separately. The estimated CATE is given by:

.. math::

    \hat{\tau}(x) & = E[Y(1)-Y(0)\mid X=x] \\
                & = E[Y(1)\mid X=x] - E[Y(0)\mid X=x] \\
                & = \hat{\mu}_1(x) - \hat{\mu}_0(x)

where :math:`\hat{\mu}_0 = M_0(Y^0\sim X^0),\; \hat{\mu}_1 = M_1(Y^1\sim X^1)` are the outcome models for the control and treatment group, respectively. Here, :math:`M_0`, :math:`M_1` can be any suitable machine learning algorithms that can learn the relationship between features and outcome.

The EconML package provides the following implementation of the T-Learner:
:class:`.TLearner`

S-Learner
-----------

The S-Learner models :math:`Y(0)` and :math:`Y(1)` through one model that receives the treatment assignment :math:`T` as an input feature (along with the features :math:`X`). The estimated CATE is given by:

.. math::

    \hat{\tau}(x) & = E[Y \mid X=x, T=1] - E[Y\mid X=x, T=0] \\
    & = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)

where :math:`\hat{\mu}=M(Y \sim (X, T))` is the outcome model for features :math:`X, T`. Here, :math:`M` is any suitable machine learning algorithm.
 
The EconML package provides the following implementation of the S-Learner: 
:class:`.SLearner`

X-Learner
-----------

The X-Learner models :math:`Y(1)` and :math:`Y(0)` separately in order to estimate the CATT (Conditional Average Treatment Effect on the Treated) and CATC (Conditional Average Treatment Effect on the Controls). The CATE estimate for a new point :math:`x` is given by the propensity-weighted average of CATT and CATC. A sketch of the X-Learner procedure is given below:

.. math::

    \hat{\mu}_0 & = M_1(Y^0 \sim X^0) \\
    \hat{\mu}_1 & = M_2(Y^1 \sim X^1) \\
    \hat{D}^1 & = Y^1 - \hat{\mu}_0(X^1) \\
    \hat{D}^0 & = \hat{\mu}_1(X^0) - Y^0 \\
    \hat{\tau}_0 & = M_3(\hat{D}^0 \sim X^0) \\
    \hat{\tau}_1 & = M_4(\hat{D}^1 \sim X^1) \\
    \hat{\tau} & = g(x)\hat{\tau}_0(x) + (1-g(x))  \hat{\tau}_1(x)

where :math:`g(x)` is an estimation of :math:`P[T=1| X]` and :math:`M_1, M_2, M_3, M_4` are suitable machine learning algorithms. 

The EconML package provides the following implementation of the X-Learner: 
:class:`.XLearner`


Domain Adaptation Learner
-------------------------

The Domain Adaptation Learner is a variation of the :math:`X`-learner that uses domain adaptation techniques to estimate the 
outcome models :math:`\hat{\mu}_0` and :math:`\hat{\mu}_1`. The underlying assumption of the Domain Adaptation methodology is that 
the probability distributions :math:`P(X^0)` and :math:`P(X^1)` are different. This requires weighting the :math:`X^0` samples by how 
similar they are to :math:`X^1` samples when training a model on :math:`X^0` that is unbiased on :math:`X^1`. A sketch of the 
Domain Adaptation Learner procedure is given below:

.. math::

    \hat{\mu}_0 & = M_1\left(Y^0 \sim X^0, \text{weights}=\frac{g(X^0)}{1-g(X^0)}\right) \\
    \hat{\mu}_1 & = M_2\left(Y^1 \sim X^1, \text{weights}=\frac{1-g(X^1)}{g(X^1)}\right) \\
    \hat{D}^1 & = Y^1 - \hat{\mu}_0(X^1) \\
    \hat{D}^0 & = \hat{\mu}_1(X^0) - Y^0 \\
    \hat{\tau} & = M_3(\hat{D}^0|\hat{D}^1 \sim X^0|X^1)

where :math:`g(x)` is an estimation of :math:`P[T=1| X]`, :math:`M_1, M_2, M_3` are suitable machine learning algorithms, and :math:`|` denotes 
dataset concatenation. 

The EconML package provides the following implementation of the Domain Adaptation Learner: 
:class:`.DomainAdaptationLearner`


Doubly Robust Learner
---------------------

See :ref:`Doubly Robust Learning User Guide <druserguide>`.

Non-Parametric Double Machine Learning
--------------------------------------

See :ref:`Double Machine Learning User Guid <dmluserguide>`.


Class Hierarchy Structure
==================================

.. inheritance-diagram:: econml.metalearners.SLearner econml.metalearners.TLearner econml.metalearners.XLearner econml.metalearners.DomainAdaptationLearner econml.drlearner.DRLearner econml.dml.DML
        :parts: 1
        :private-bases:
        :top-classes: econml._ortho_learner._OrthoLearner, econml.cate_estimator.LinearCateEstimator, econml.cate_estimator.TreatmentExpansionMixin


Usage Examples
==================================

Check out the following notebooks:

    * `Metalearners Jupyter notebook <https://github.com/Microsoft/EconML/blob/master/notebooks/Metalearners%20Examples.ipynb>`_.
    * `DML Examples Jupyter Notebook <https://github.com/microsoft/EconML/blob/master/notebooks/Double%20Machine%20Learning%20Examples.ipynb>`_,
    * `Forest Learners Jupyter Notebook <https://github.com/microsoft/EconML/blob/master/notebooks/ForestLearners%20Basic%20Example.ipynb>`_.


.. todo::
    * Synthetic Controls via Matchings
    * Regression Discontinuity Estimators



