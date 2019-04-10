[![Build Status](https://dev.azure.com/ms/EconML/_apis/build/status/Microsoft.EconML?branchName=master)](https://dev.azure.com/ms/EconML/_build/latest?definitionId=49&branchName=master)
[![PyPI version](https://img.shields.io/pypi/v/econml.svg)](https://pypi.org/project/econml/)
[![PyPI wheel](https://img.shields.io/pypi/wheel/econml.svg)](https://pypi.org/project/econml/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/econml.svg)](https://pypi.org/project/econml/)



<h1><img src="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/MSR-ALICE-HeaderGraphic-1920x720_1-800x550.jpg" width="130px" align="left" style="margin-right: 10px;"> EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation</h1>

The [ALICE project](https://www.microsoft.com/en-us/research/project/alice/) at Microsoft Research is 
aimed at applying Artificial Intelligence concepts to economic decision making.  The Microsoft EconML 
package is part of that project, providing a toolkit that combines state-of-the-art machine learning 
techniques with econometrics in order to bring automation to complex causal inference problems.  This 
toolkit is designed to measure the causal effect of some treatment variable(s) `T` on an outcome 
variable `Y`, controlling for a set of features `X`.  For more information about how to use this package, 
consult the documentation at https://econml.azurewebsites.net/.

<details>
<summary><strong><em>Table of Contents</em></strong></summary>

- [Introduction](#introduction)
  - [About Treatment Effect Estimation](#about-treatment-effect-estimation)
  - [Real-world Use Cases](#real-world-use-cases)
- [News](#news)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage Examples](#usage-examples)
- [For Developers](#for-developers)
  - [Running the tests](#running-the-tests)
  - [Generating the documentation](#generating-the-documentation)
- [Blogs and Publications](#blogs-and-publications)
- [Contributing and Feedback](#contributing-and-feedback)
- [References](#references)

</details>

# Introduction

## About Treatment Effect Estimation

One of the biggest promises of machine learning is the automation of decision making in a multitude of application domains. A core problem that arises in most data-driven personalized decision scenarios is the estimation of heterogeneous treatment effects: what is the effect of an intervention on an outcome of interest as a function of a set of observable characteristics of the treated sample? For instance, this problem arises in personalized pricing, where the goal is to estimate the effect of a price discount on the demand as a function of characteristics of the consumer. Similarly it arises in medical trials where the goal is to estimate the effect of a drug treatment on the clinical response of a patient as a function of patient characteristics. In many such settings we have an abundance of observational data, where the treatment was chosen via some unknown policy and the ability to run control A/B tests is limited.

## Real-world Use Cases

Customer Targeting/ Segmentation 

Personalized Pricing

Clinical Trials

Click-through Rates

# News

**04/10/2019:** Release v0.2, see release notes [here](release_notes_link).

**03/06/2019:** Release v0.1, welcome to have a try and provide feedback.

# Getting Started

## Installation

Install the latest release from [PyPI](https://pypi.org/project/econml/):
```
pip install econml
```
To install from source, see [For Developers](#for-developers) section below.

## Usage Examples

* [Double Machine Learning](#references)

  ```Python
  from econml import DMLCateEstimator
  from sklearn.linear_model import LassoCV
  
  est = DMLCateEstimator(model_y=LassoCV(), model_t=LassoCV)
  est.fit(Y, T, X, W) # W -> high-dimensional confounders, X -> features
  treatment_effects = est.const_marginal_effect(X_test)
  ```

* [Orthogonal Random Forests](#references)

  ```Python
  from econml import ContinuousTreatmentOrthoForest
  # Use defaults
  est = ContinuousTreatmentOrthoForest()
  # Or specify hyperparameters
  est = ContinuousTreatmentOrthoForest(n_trees=500, min_leaf_size=10, max_depth=10, 
                                       subsample_ratio=0.7, lambda_reg=0.01,
                                       model_T=LassoCV(cv=3), model_Y=LassoCV(cv=3)
                                       )
  est.fit(Y, T, X, W)
  treatment_effects = est.const_marginal_effect(X_test)
    ```

* [Deep Instrumental Variables](#references)
  
  ```Python
  from econml import DeepIVEstimator

  est = DeepIVEstimator(n_components=10, # Number of gaussians in the mixture density networks)
                        m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])), # Treatment model
                        h=lambda t, x: response_model(keras.layers.concatenate([t, x])), # Response model
                        n_samples=1 # Number of samples used to estimate the response
                        )
  est.fit(Y, T, X, Z) # Z -> instrumental variables
  treatment_effects = est.effect(T0, T1, X_test)
  ```


* Bootstrap Confidence Intervals
  ```Python
  from econml.bootstrap import BootstrapEstimator

  # Bootstrap estimator wrapper
  boot_est = BootstrapEstimator(est, n_bootstrap_samples=10)
  boot_est.fit(Y, T, X, W)
  treatment_effect_interval = boot_est.const_marginal_effect_interval(X_test, lower=1, upper=99)
  ```

To see more complex examples, go to the [notebooks](https://github.com/Microsoft/EconML/tree/master/notebooks) section of the repository. For a more detailed description of the treatment effect estimation algorithms, see the EconML [documentation](https://econml.azurewebsites.net/).

# For Developers

You can get starting by cloning this repository. We use 
[setuptools](https://setuptools.readthedocs.io/en/latest/index.html) for building and distributing our package.
We rely on some recent features of setuptools, so make sure to upgrade to a recent version with
`pip install setuptools --upgrade`.  Then from your local copy of the repository you can run `python setup.py develop` to get started.

## Running the tests

This project uses [pytest](https:docs.pytest.org/) for testing.  To run tests locally after installing the package, 
you can use `python setup.py pytest`.

## Generating the documentation

This project's documentation is generated via [Sphinx](https://www.sphinx-doc.org/en/master/index.html).  To generate a local copy
of the documentation from a clone of this repository, just run `python setup.py build_sphinx`, which will build the documentation and place it
under the `build/sphinx/html` path.

The reStructuredText files that make up the documentation are stored in the [docs directory](docs/); module documentation is automatically generated by the Sphinx build process.

# Blogs and Publications

* May 2019: [Open Data Science Conference Workshop](https://staging5.odsc.com/training/portfolio/machine-learning-estimation-of-heterogeneous-treatment-effect-the-microsoft-econml-library) 

* 2018: [Orthogonal Random Forests paper](https://arxiv.org/abs/1806.03467)

* 2017: [DeepIV paper](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf)

# Contributing and Feedback

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# References

M. Oprescu, V. Syrgkanis and Z. S. Wu.
**Orthogonal Random Forest for Causal Inference.**
[*ArXiv preprint arXiv:1806.03467*](http://arxiv.org/abs/1806.03467), 2018.

Jason Hartford, Greg Lewis, Kevin Leyton-Brown, and Matt Taddy. **Deep IV: A flexible approach for counterfactual prediction.** [*Proceedings of the 34th International Conference on Machine Learning*](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf), 2017.

V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, and a. W. Newey. **Double Machine Learning for Treatment and Causal Parameters.** [*ArXiv preprint arXiv:1608.00060*](https://arxiv.org/abs/1608.00060), 2016.