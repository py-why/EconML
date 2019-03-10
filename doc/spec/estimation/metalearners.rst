Metalearners
============

Metalearners are binary treatment CATE estimators that model the two 
response surfaces, :math:`Y(0)` and :math:`Y(1)`, separately. For a detailed overview of these methods,
see [Kunzel2017]_ and [Foster2019]_.

For examples of how to use our implemented metelearners check out this
`Metalearners Jypyter notebook <https://github.com/Microsoft/EconML/blob/master/notebooks/Metalearners%20Examples.ipynb>`_

T-Learner
-----------------

The T-Learner models :math:`Y(0)`, :math:`Y(1)` separately. The estimated CATE is given by:

.. math::

    \hat{\tau}(x) & = E[Y(1)-Y(0)\mid X=x] \\
                & = E[Y(1)\mid X=x] - E[Y(0)\mid X=x] \\
                & = \hat{\mu}_1(x) - \hat{\mu}_0(x)

where :math:`\hat{\mu}_0 = M_0(Y^0\sim X^0),\; \hat{\mu}_1 = M_1(Y^1\sim X^1)` are the outcome models for the control and treatment group, respectively. Here, :math:`M_0`, :math:`M_1` can be any suitable machine learning algorithms that can learn the relationship between features and outcome.

The EconML package provides the following implementation of the T-Learner:
:py:class:`~econml.metalearners.TLearner`

S-Learner
-----------

The S-Learner models :math:`Y(0)` and :math:`Y(1)` through one model that receives the treatment assignment :math:`T` as an input feature (along with the features :math:`X`). The estimated CATE is given by:

.. math::

    \hat{\tau}(x) & = E[Y \mid X=x, T=1] - E[Y\mid X=x, T=0] \\
    & = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)

where :math:`\hat{\mu}=M(Y \sim (X, T))` is the outcome model for features :math:`X, T`. Here, :math:`M` is any suitable machine learning algorithm.
 
The EconML package provides the following implementation of the S-Learner: 
:py:class:`~econml.metalearners.SLearner`

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
:py:class:`~econml.metalearners.XLearner`


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
:py:class:`~econml.metalearners.DomainAdaptationLearner`


Doubly Robust Learner
---------------------

The Doubly Robust Learner estimates the treatment effect by running a regression between 
the doubly robust unbiased estimates of the counterfactual outcomes and the target features :math:`X`, i.e.
it first constructs the proxies:

.. math::
    
    Y_{i,t}^{DR} = \hat{\E}[Y \mid T=t, x, W_i] - 1\{T_i=t\} \frac{Y_i - \hat{\E}[Y \mid T=t, x, W_i]}{\hat{\E}[1\{T=t\} \mid x, W_i]} 

and then runs a regression between :math:`Y_{i, 1}^{DR} - Y_{i, 0}^{DR}` and :math:`X`.

The EconML package provides the following implementation of the Doubly Robust Learner: 
:py:class:`~econml.metalearners.DoublyRobustLearner`


.. todo::
    * Synthetic Controls via Matchings
    * Regression Discontinuity Estimators



