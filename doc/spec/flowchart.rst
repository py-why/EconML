==================
Library Flow Chart
==================


.. raw:: html

    <object data="../map.svg" type="image/svg+xml"></object>


.. glossary::
    EXPERIMENT
        Did you explicitly randomize treatment across a targeted audience? 

    COMPLIANCE
        Did everyone in the experiment receive the treatment to which they were assigned? For example, 
        if you are interested in the causal effect of joining a loyalty program a useful experiment might 
        randomly send some customers an email prompting them to join the program. Only some of these 
        targeted customers will join the loyalty program, so compliance is imperfect. In a medical 
        experiment where patients come into an office each week and receive either a novel drug or a 
        placebo, compliance with the assigned group is likely to be perfect. 

    TREATMENT ASSIGNMENT
        If treatment was not an outcome of an experiment (i.e. controlled), there will be multiple reasons
        that people received the treatment.  Suppose you are interested in the causal effect of minutes 
        of exercise per week (the treatment variable) on body fat percentage (the outcome). Some people 
        exercise more minutes than others.  Reasons for this (controls) may include having more flexible 
        work schedules or being more health conscious.  Some of these controls may be confounders (i.e. 
        they also affect the outcome directly).  In this example, being more health conscious is probably
        a confounding factor (it affects nutrition, which affects body fat), whereas having a more flexible
        work schedule is probably not. The question posed in this bubble is whether all confounding controls
        are measured in the data.     

    Can set W
        If you run an experiment, there are not concerns about confoundedness. Any features included in the
        control set will help with the efficiency of the estimate in smaller samples, but are not necessary
        to identify the causal effect. 

    CAUTION
        Bias is eliminated when all confounders are measurable. See Orthogonal/Double Machine Learning
        section for guidance on when this assumption holds. 

    Set Z to intended treatment
        With imperfect compliance, the indicator of the assigned treatment category is not equivalent to
        the indicator for actually receiving the treatment. If you are interested in the causal effect of
        the treatment, you should use assignment as an instrument for treatment, rather than simply treating
        assignment as the treatment itself. 

    TREATMENT RESPONSIVENESS
        Many estimators only behave well with a small set of specified features X that affect the size of a
        user’s response to the treatment. If you do not already know which few features might reasonably
        affect the user’s response, use one of our sparse estimators that can handle large feature sets and
        penalize them to discover the features that are most correlated with treatment effect heterogeneity. 

    INSTRUMENT
        Some estimators identify the causal effect of a treatment by considering only a subset of the
        variation in treatment intensity that is conditionally random given other data features. This
        subset of the variation is driven by an instrument, which is often some kind of randomization (i.e.
        an earlier experiment or a lottery). See the Instrumental Variable Regression section for more
        information on picking a good instrument. 

    LINEAR TREATMENT EFFECTS
        Some estimators impose the assumption that the outcome is a linear function of the treatment. These
        estimators can also estimate a non-linear relationship between a treatment and the outcome if the
        structure of the relationship is known and additively separable (for example, the linear function
        could include both treatment and treatment-squared for continuous treatments). These linear functions
        can also include specified interactions between treatments. However, these estimators cannot estimate
        a fully flexible non-parametric relationship between treatments and the outcome (for example, the
        relationship cannot be modeled by a forest). 

    LINEAR HETEROGENEITY
        The CATE function determines how the size of a user’s response to the treatment varies by user
        features. Some estimators impose the assumption that effect size is a linear function of user features.  

    CONFIDENCE INTERVALS/MODEL SELECTION
        The MetaLearner and DRLearner estimators offer the choice of any ML estimation model in all stages
        and allows for model selection via cross validation. This enhances flexibility, but because the
        sample data is used to choose among models it is impossible to calculate honest analytic confidence
        intervals. Moreover, most ML estimation approaches introduce bias for regularization purposes, so as
        to optimally balance bias and variance. Hence, confidence intervals based on such biased estimates
        will be invalid. For these models it is still possible to construct bootstrap confidence intervals,
        but this process is slow, may not be accurate in small samples and these intervals only capture the 
        variance but not the bias of the model. 
