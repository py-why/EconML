Estimation Methods with Instruments
===================================

This section contains methods for estimating (heterogeneous) treatment effects,
even when there are unobserved confounders (factors that simultaneously
had a direct effect on the treatment decision in the collected data and the observed outcome). However, the
methods assumes that we have access to an instrumental variable: a random variable that had an effect on the
treatment, but did not have any direct effect on the outcome, other than through the treatment.

The package offers two IV methods for 
estimating heterogeneous treatment effects: deep instrumental variables [Hartford2017]_ and the two-stage basis expansion approach 
of [Newey2003]_.  

.. toctree::
    :maxdepth: 2

    estimation/deepiv.rst
    estimation/two_sls.rst
    estimation/orthoiv.rst