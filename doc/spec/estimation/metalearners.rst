Metalearners
============

Metalearners are binary treatment CATE estimators that model the two 
response surfaces, :math:`Y(0)` and :math:`Y(1)`, separately. For a detailed overview of these methods, see [Kunzel2017]_.

T-Learner
-----------------

The T-Learner models :math:`Y(0)`, :math:`Y(1)` separately. The estimated CATE is given by:

.. math::

    \hat{\tau}(x) & = E[Y(1)-Y(0)\mid X=x] \\
                & = E[Y(1)\mid X=x] - E[Y(0)\mid X=x] \\
                & = \hat{\mu}_1(x) - \hat{\mu}_0(x)

where :math:`\hat{\mu}_0 = M_0(Y^0\sim X^0),\; \hat{\mu}_1 = M_1(Y^1\sim X^1)` are the outcome models for the control and treatment group, respectively. Here, :math:`M_0`, :math:`M_1` can be any suitable machine learning algorithms that can learn the relationship between features and outcome.

The CATE package provides the following implementation of the T-Learner:
 

.. code-block:: python3
    :caption: TLearner Class Implementation

    class TLearner(BaseCateEstimator):
        """Conditional mean regression estimator.
        
        Parameters
        ----------
        controls_model : outcome estimator for control units
            Must implement `fit` and `predict` methods.
        
        treated_model : outcome estimator for treated units
            Must implement `fit` and `predict` methods.
        """
        
        def __init__(self, controls_model, treated_model):
            ...
        
        def fit(self, Y, T, X):
            """Build an instance of SLearner.
            
            Parameters
            ----------
            Y : array-like, shape (n, ) or (n, d_y)
                Outcome(s) for the treatment policy.
            
            T : array-like, shape (n, ) or (n, 1)
                Treatment policy. Only binary treatments are accepted as input.
                T will be flattened if shape is (n, 1).
            
            X : array-like, shape (n, d_x)
                Feature vector that captures heterogeneity.
        
            Returns
            -------
            self: an instance of self.
            """
            ...

        def effect(self, X):
            """Calculates the heterogeneous treatment effect on a vector
            of features for each sample.
            
            Parameters
            ----------
            X : matrix, shape (m x d_x)
                Matrix of features for each sample.
            
            Returns
            -------
            τ_hat : array-like, shape (m, )
                Matrix of heterogeneous treatment effects for each sample.
            """
            ...
            
        def marginal_effect(self, X):
            """Calculates the heterogeneous marginal treatment effect. For binary
            treatment, it returns the same as `effect`.
            """
            ...

S-Learner
-----------

The S-Learner models :math:`Y(0)` and :math:`Y(1)` through one model that receives the treatment assignment :math:`T` as an input feature (along with the features :math:`X`). The estimated CATE is given by:

.. math::

    \hat{\tau}(x) & = E[Y \mid X=x, T=1] - E[Y\mid X=x, T=0] \\
    & = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)

where :math:`\hat{\mu}=M(Y \sim (X, T))` is the outcome model for features :math:`X, T`. Here, :math:`M` is any suitable machine learning algorithm.
 
The CATE package provides the following implementation of the S-Learner: 

.. code-block:: python3
    :caption: SLearner Class Implementation

    class SLearner(BaseCateEstimator):
        """Conditional mean regression estimator where the treatment
        assignment is taken as a feature in the ML model.
        
        Parameters
        ----------
        overall_model : outcome estimator for all units
            Model will be trained on X|T where '|' denotes concatenation.
            Must implement `fit` and `predict` methods.
        """

        def __init__(self, overall_model):
            ...

        def fit(self, Y, T, X):
            """Build an instance of SLearner.
            """
            ...

        def effect(self, X):
            """Calculates the heterogeneous treatment effect on a vector
            of features for each sample.
            """
            ...
            
        def marginal_effect(self, X):
            """Calculates the heterogeneous marginal treatment effect. For binary
            treatment, it returns the same as `effect`.
            """
            ...

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

The CATE package provides the following implementation of the X-Learner: 

.. code-block:: python3
    :caption: XLearner Class Implementation

    class XLearner(BaseCateEstimator):
        """Meta-algorithm proposed by Kunzel et al. that performs best in settings
        where the number of units in one treatment arm is much larger than in the other.
        
        Parameters
        ----------
        controls_model : outcome estimator for control units
            Must implement `fit` and `predict` methods.
        
        treated_model : outcome estimator for treated units
            Must implement `fit` and `predict` methods.
        
        cate_controls_model : estimator for pseudo-treatment effects on the controls
            Must implement `fit` and `predict` methods.
        
        cate_treated_model : estimator for pseudo-treatment effects on the treated
            Must implement `fit` and `predict` methods.
        
        propensity_model : estimator for the propensity function
            Must implement `fit` and `predict_proba` methods. The `fit` method must
            be able to accept X and T, where T is a shape (n, 1) array.
            Ignored when `propensity_func` is provided.
        
        propensity_func : propensity function
            Must accept an array of feature vectors and return an array of probabilities.
            If provided, the value for `propensity_model` (if any) will be ignored.
        """
        def __init__(self, controls_model,
                        treated_model,
                        cate_controls_model=None,
                        cate_treated_model=None,
                        propensity_model=LogisticRegression(),
                        propensity_func=None):
            ...

        def fit(self, Y, T, X):
            """Build an instance of XLearner.
            """
            ...
    
        def effect(self, X):
            """Calculates the heterogeneous treatment effect on a vector
            of features for each sample.
            """
            ...
        
        def marginal_effect(self, X):
            """Calculates the heterogeneous marginal treatment effect. For binary
            treatment, it returns the same as `effect`.
            """
            ...

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

The CATE package provides the following implementation of the Domain Adaptation Learner: 

.. code-block:: python3
    :caption: DomainAdaptationLearner Class Implementation

    class DomainAdaptationLearner(BaseCateEstimator):
        """Meta-algorithm that uses domain adaptation techniques to account for
        covariate shift (selection bias) between the treatment arms.
        
        Parameters
        ----------
        controls_model : outcome estimator for control units
            Must implement `fit` and `predict` methods.
            The `fit` method must accept the `sample_weight` parameter.
        
        treated_model : outcome estimator for treated units
            Must implement `fit` and `predict` methods.
            The `fit` method must accept the `sample_weight` parameter.
        
        overall_model : estimator for pseudo-treatment effects
            Must implement `fit` and `predict` methods.
        
        propensity_model : estimator for the propensity function
            Must implement `fit` and `predict_proba` methods. The `fit` method must
            be able to accept X and T, where T is a shape (n, 1) array.
            Ignored when `propensity_func` is provided.
        
        propensity_func : propensity function
            Must accept an array of feature vectors and return an array of probabilities.
            If provided, the value for `propensity_model` (if any) will be ignored.
        """

        def __init__(self, controls_model,
                        treated_model,
                        overall_model,
                        propensity_model=LogisticRegression(),
                        propensity_func=None):
            ...
                
        def fit(self, Y, T, X):
            """Build an instance of XLearner.
            """
            ...
        
        def effect(self, X):
            """Calculates the heterogeneous treatment effect on a vector
            of features for each sample.
            """
            ...
        
        def marginal_effect(self, X):
            """Calculates the heterogeneous marginal treatment effect. For binary
            treatment, it returns the same as `effect`.
            """
            ...

.. todo::
    * Synthetic Controls via Matchings
    * Regression Discontinuity Estimators
