EconML User Guide
=================

Causal machine learning applies the power of machine learning techniques to answer causal questions.  

* Decision-makers need estimates of causal impacts to answer what-if questions about shifts in policy - such as changes in product pricing for businesses or new treatments for health professionals.

* Most current machine learning tools are designed to forecast what will happen next under the present strategy, but cannot be interpreted to predict the effects of particular changes in behavior. 

* Existing solutions to answer what-if questions are expensive. Decision-makers can engage in active experimentation like A/B testing or employ highly trained economists who use traditional statistical models to infer causal effects from previously collected data. 

The EconML Python SDK, developed by the ALICE team at MSR New England, incorporates individual machine learning steps into interpretable causal models. By reducing the need for expert judgment, these innovations improve the reliability of what-if predictions and empower data scientists without extensive economic training to conduct causal analysis using existing data. 


.. toctree::
    motivation
    api
    flowchart
    comparison
    estimation
    estimation_iv
    estimation_dynamic
    inference
    interpretability
    references
    faq

.. todo::
    benchmark
    Panel data wrapper
