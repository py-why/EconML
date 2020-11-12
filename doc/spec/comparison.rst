=============================
Detailed estimator comparison
=============================


+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| Estimator                                   | | Treatment  | | Requires   | | Delivers Conf. | | Linear    | | Linear        | | Mulitple | | Multiple   | | High-Dimensional |
|                                             | | Type       | | Instrument | | Intervals      | | Treatment | | Heterogeneity | | Outcomes | | Treatments | | Features         |
+=============================================+==============+==============+==================+=============+=================+============+==============+====================+
| :class:`.NonparametricTwoStageLeastSquares` | Any          | Yes          |                  | Yes         | Assumed         | Yes        | Yes          |                    |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.DeepIVEstimator`                   | Any          | Yes          |                  |             |                 | Yes        | Yes          |                    |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.SparseLinearDML`                   | Any          |              | Yes              | Yes         | Assumed         | Yes        | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.SparseLinearDRLearner`             | Categorical  |              | Yes              |             | Projected       |            | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.LinearDML`                         | Any          |              | Yes              | Yes         | Assumed         | Yes        | Yes          |                    |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.LinearDRLearner`                   | Categorical  |              | Yes              |             | Projected       |            | Yes          |                    |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.ForestDML`                         | 1-d/Binary   |              | Yes              | Yes         |                 | Yes        |              | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.ForestDRLearner`                   | Categorical  |              | Yes              |             |                 | Yes        | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.ContinuousTreatmentOrthoForest`    | Continuous   |              | Yes              | Yes         |                 |            | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.DiscreteTreatmentOrthoForest`      | Categorical  |              | Yes              |             |                 |            | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :mod:`~econml.metalearners`                 | Categorical  |              |                  |             |                 |            | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.DRLearner`                         | Categorical  |              |                  |             |                 |            | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.DML`                               | Any          |              |                  | Yes         | Assumed         | Yes        | Yes          | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+
| :class:`.NonParamDML`                       | 1-d/Binary   |              |                  | Yes         |                 | Yes        |              | Yes                |
+---------------------------------------------+--------------+--------------+------------------+-------------+-----------------+------------+--------------+--------------------+


Treatment Type
    Some estimators can only estimate effects of particular kinds of treatments. 
    *Discrete* treatments can be described by a finite number of comprehensive categories (for example, 
    group A received a 10% discount on product 1, group B received a 10% discount on product 2, group C 
    received no discounts). *Binary* treatments are a special case of discrete treatments with only two 
    categories. *Continuous* treatments can take on any value along the number line (for example, minutes of 
    exercise per week).  

Requires Instrument
    Some estimators identify the causal effect of a treatment by considering only a subset of the variation in 
    treatment intensity that is conditionally random given other data features. This subset of the variation 
    is driven by an instrument, which is usually some kind of randomization (i.e. an earlier experiment or a 
    lottery). See the Instrumental Variable Regression section for more information on picking a good 
    instrument.  

Delivers Confidence Intervals
    Many estimators can deliver analytic confidence intervals for the final treatment effects. These 
    confidence intervals correctly adjust for the reuse of data across multiple stages of estimation. EconML 
    cannot deliver analytic confidence intervals in cases where this multi-stage estimation is too complex or 
    for estimators such as the MetaLearners that trade honest confidence intervals for model selection and 
    regularization. In these cases it is still possible to get bootstrap confidence intervals, but this 
    process is slow and may not be statistically valid. 

Linear Treatment
    Some estimators impose the assumption that the outcome is a linear function of the treatment. These 
    estimators can also estimate a non-linear relationship between a treatment and the outcome if the 
    structure of the relationship is known and additively separable (for example, the linear function could 
    include both treatment and treatment-squared for continuous treatments). These linear functions can also 
    include specified interactions between treatments. However, these estimators cannot estimate a fully 
    flexible non-parametric relationship between treatments and the outcome (for example, the relationship 
    cannot be modeled by a forest). 

Linear Heterogeneity
    The CATE function determines how the size of a user’s response to the treatment varies by user features. 
    Some estimators impose the *assumption* that effect size is a linear function of user features. A few models 
    estimate a more flexible relationship between effect size and user features and then *project* that flexible
    function onto a linear model. This second approach delivers a better-fitting linear approximation of a 
    non-linear relationship, but is less efficient in cases where you are confident assuming the true 
    relationship is linear. Finally, some estimation models allow a fully flexible relationship between 
    effect size and user features with no linearity structure. 

Multiple Outcomes
    Some estimation models allow joint estimation of the effects of treatment(s) on multiple outcomes. Other 
    models only accommodate a single outcome. 

Multiple Treatments
    Some estimation models allow joint estimation of the effects of multiple treatments on outcome(s). Other 
    models only accommodate a single treatment. 

High-Dimensional Features
    Many estimators only behave well with a small set of specified features, X, that affect the size of a 
    user’s response to the treatment. If you do not already know which few features might reasonably affect 
    the user’s response, use one of our sparse estimators that can handle large feature sets and penalize them 
    to discover the features that are most correlated with treatment effect heterogeneity. 

