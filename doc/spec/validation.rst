Validation
======================

Validating causal estimates is inherently challenging, as the true counterfactual outcome for a given treatment is
unobservable. However, there are several checks and tools available in EconML to help assess the credibility of causal
estimates.


Sensitivity Analysis
---------------------

For many EconML estimators, unobserved confounding can lead to biased causal estimates.
Moreover, it is impossible to prove the absence of unobserved confounders.
This is a fundamental problem for observational causal inference.

To mitigate this problem, EconML provides a suite of sensitivity analysis tools,
based on [Chernozhukov2022]_,
to assess the robustness of causal estimates to unobserved confounding. 

Specifically, select estimators (subclasses of :class:`.DML` and :class:`.DRLearner`)
have access to ``sensitivity_analysis``, ``robustness_value``, and ``sensitivity_summary`` methods.

``sensitivity_analysis`` provides an updated confidence interval for the ATE based on a specified level of unobserved confounding.


``robustness_value`` computes the minimum level of unobserved confounding required
so that confidence intervals around the ATE would begin to include the given point (0 by default).


``sensitivity_summary`` provides a summary of the the two above methods.

DRTester
----------------

EconML provides the :class:`.DRTester` class, which implements Best Linear Predictor (BLP), calibration r-squared,
and uplift modeling methods for validation.

See an example notebook `here <https://github.com/py-why/EconML/blob/main/notebooks/CATE%20validation.ipynb>`__.

Scoring
-------

Many EconML estimators implement a ``.score`` method to evaluate the goodness-of-fit of the final model. While it may be 
difficult to make direct sense of results from ``.score``, EconML offers the :class:`RScorer` class to facilitate model 
selection based on scoring.

:class:`RScorer` enables comparison and selection among different causal models.

See an example notebook `here
<https://github.com/py-why/EconML/blob/main/notebooks/Causal%20Model%20Selection%20with%20the%20RScorer.ipynb>`__.

Confidence Intervals and Inference
----------------------------------

Most EconML estimators allow for inference, including standard errors, confidence intervals, and p-values for
estimated effects. A common validation approach is to check whether the p-values are below a chosen significance level
(e.g., 0.05). If not, the null hypothesis that the causal effect is zero cannot be rejected.

**Note:** Inference results are only valid if the model specification is correct. For example, if a linear model is used
but the true data-generating process is nonlinear, the inference may not be reliable. It is generally not possible to
guarantee correct specification, so p-value inspection should be considered a surface-level check.

DoWhy Refutation Tests
----------------------

The DoWhy library, which complements EconML, includes several refutation tests for validating causal estimates. These
tests work by comparing the original causal estimate to estimates obtained from perturbed versions of the data, helping
to assess the robustness of causal conclusions.