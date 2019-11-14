EconML Specification
====================

Causal machine learning applies the power of machine learning techniques to answer causal questions.  

* Business decision-makers need estimates of causal impacts to answer what-if questions about shifts in strategy - such as new investments from a sales team or a change to product pricing.  

* Most current machine learning tools are designed to forecast what will happen next under the present strategy, but cannot be interpreted to predict the effects of particular changes in behavior. 

* Existing solutions to answer what-if questions are expensive. Executives can engage in active experimentation like A/B testing or employ highly trained economists who use traditional statistical models to infer causal effects from previously collected data. 

The EconML Python SDK, developed by the ALICE team at MSR New England, incorporates individual machine learning steps into interpretable causal models. By reducing the need for expert judgment, these innovations improve the reliability of what-if predictions and empower data scientists without extensive economic training to conduct causal analysis using existing data. 


.. toctree::
    motivation
    api
    estimation
    inference
    references

.. todo::
    benchmark
    Panel data wrapper