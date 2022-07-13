EconML User Guide
=================

Causal machine learning applies the power of machine learning techniques to answer causal questions.  

* Decision-makers need estimates of causal impacts to answer what-if questions about shifts in policy - such as changes in product pricing for businesses or new treatments for health professionals.

* Most current machine learning tools are designed to forecast what will happen next under the present strategy, but cannot be interpreted to predict the effects of particular changes in behavior. 

* Existing solutions to answer what-if questions are expensive. Decision-makers can engage in active experimentation like A/B testing or employ highly trained economists who use traditional statistical models to infer causal effects from previously collected data. 

EconML is a Python package that applies the power of machine learning techniques to estimate individualized causal responses from observational or experimental data. The suite of estimation methods provided in EconML represents the latest advances in causal machine learning. By incorporating individual machine learning steps into interpretable causal models, these methods improve the reliability of what-if predictions and make causal analysis quicker and easier for a broad set of users.

.. toctree::
    motivation
    causal_intro
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
