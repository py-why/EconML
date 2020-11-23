[![Build Status](https://dev.azure.com/ms/EconML/_apis/build/status/Microsoft.EconML?branchName=master)](https://dev.azure.com/ms/EconML/_build/latest?definitionId=49&branchName=master)
[![PyPI version](https://img.shields.io/pypi/v/econml.svg)](https://pypi.org/project/econml/)
[![PyPI wheel](https://img.shields.io/pypi/wheel/econml.svg)](https://pypi.org/project/econml/)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/econml.svg)](https://pypi.org/project/econml/)



<h1><img src="https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/MSR-ALICE-HeaderGraphic-1920x720_1-800x550.jpg" width="130px" align="left" style="margin-right: 10px;"> EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation</h1>

**EconML** is a Python package for estimating heterogeneous treatment effects from observational data via machine learning. This package was designed and built as part of the [ALICE project](https://www.microsoft.com/en-us/research/project/alice/) at Microsoft Research with the goal to combine state-of-the-art machine learning 
techniques with econometrics to bring automation to complex causal inference problems. The promise of EconML:

* Implement recent techniques in the literature at the intersection of econometrics and machine learning
* Maintain flexibility in modeling the effect heterogeneity (via techniques such as random forests, boosting, lasso and neural nets), while preserving the causal interpretation of the learned model and often offering valid confidence intervals
* Use a unified API
* Build on standard Python packages for Machine Learning and Data Analysis

One of the biggest promises of machine learning is to automate decision making in a multitude of domains. At the core of many data-driven personalized decision scenarios is the estimation of heterogeneous treatment effects: what is the causal effect of an intervention on an outcome of interest for a sample with a particular set of features? In a nutshell, this toolkit is designed to measure the causal effect of some treatment variable(s) `T` on an outcome 
variable `Y`, controlling for a set of features `X, W` and how does that effect vary as a function of `X`. The methods implemented are applicable even with observational (non-experimental or historical) datasets. For the estimation results to have a causal interpretation, some methods assume no unobserved confounders (i.e. there is no unobserved variable not included in `X, W` that simultaneously has an effect on both `T` and `Y`), while others assume access to an instrument `Z` (i.e. an observed variable `Z` that has an effect on the treatment `T` but no direct effect on the outcome `Y`). Most methods provide confidence intervals and inference results.

For detailed information about the package, consult the documentation at https://econml.azurewebsites.net/.

For information on use cases and background material on causal inference and heterogeneous treatment effects see our webpage at https://www.microsoft.com/en-us/research/project/econml/

<details>
<summary><strong><em>Table of Contents</em></strong></summary>

- [News](#news)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage Examples](#usage-examples)
    - [Estimation Methods](#estimation-methods)
    - [Interpretability](#interpretability)
    - [Inference](#inference)
- [For Developers](#for-developers)
  - [Running the tests](#running-the-tests)
  - [Generating the documentation](#generating-the-documentation)
- [Blogs and Publications](#blogs-and-publications)
- [Citation](#citation)
- [Contributing and Feedback](#contributing-and-feedback)
- [References](#references)

</details>

# News

**November 20, 2020:** Release v0.8.1, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.8.1)

<details><summary>Previous releases</summary>

**November 18, 2020:** Release v0.8.0, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.8.0)

**September 4, 2020:** Release v0.8.0b1, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.8.0b1)

**March 6, 2020:** Release v0.7.0, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.7.0)

**February 18, 2020:** Release v0.7.0b1, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.7.0b1)

**January 10, 2020:** Release v0.6.1, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.6.1)

**December 6, 2019:** Release v0.6, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.6)

**November 21, 2019:** Release v0.5, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.5). 

**June 3, 2019:** Release v0.4, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.4). 

**May 3, 2019:** Release v0.3, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.3).

**April 10, 2019:** Release v0.2, see release notes [here](https://github.com/Microsoft/EconML/releases/tag/v0.2).

**March 6, 2019:** Release v0.1, welcome to have a try and provide feedback.

</details>

# Getting Started

## Installation

Install the latest release from [PyPI](https://pypi.org/project/econml/):
```
pip install econml
```
To install from source, see [For Developers](#for-developers) section below.

## Usage Examples
### Estimation Methods

<details>
  <summary>Double Machine Learning (aka RLearner) (click to expand)</summary>

  * Linear final stage

  ```Python
  from econml.dml import LinearDML
  from sklearn.linear_model import LassoCV
  from econml.inference import BootstrapInference

  est = LinearDML(model_y=LassoCV(), model_t=LassoCV())
  ### Estimate with OLS confidence intervals
  est.fit(Y, T, X=X, W=W) # W -> high-dimensional confounders, X -> features
  treatment_effects = est.effect(X_test)
  lb, ub = est.effect_interval(X_test, alpha=0.05) # OLS confidence intervals

  ### Estimate with bootstrap confidence intervals
  est.fit(Y, T, X=X, W=W, inference='bootstrap')  # with default bootstrap parameters
  est.fit(Y, T, X=X, W=W, inference=BootstrapInference(n_bootstrap_samples=100))  # or customized
  lb, ub = est.effect_interval(X_test, alpha=0.05) # Bootstrap confidence intervals
  ```

  * Sparse linear final stage

  ```Python
  from econml.dml import SparseLinearDML
  from sklearn.linear_model import LassoCV

  est = SparseLinearDML(model_y=LassoCV(), model_t=LassoCV())
  est.fit(Y, T, X=X, W=W) # X -> high dimensional features
  treatment_effects = est.effect(X_test)
  lb, ub = est.effect_interval(X_test, alpha=0.05) # Confidence intervals via debiased lasso
  ```
  
  * Forest last stage
  
  ```Python
  from econml.dml import ForestDML
  from sklearn.ensemble import GradientBoostingRegressor

  est = ForestDML(model_y=GradientBoostingRegressor(), model_t=GradientBoostingRegressor())
  est.fit(Y, T, X=X, W=W) 
  treatment_effects = est.effect(X_test)
  # Confidence intervals via Bootstrap-of-Little-Bags for forests
  lb, ub = est.effect_interval(X_test, alpha=0.05)
  ```
  
  * Generic Machine Learning last stage
  
  ```Python
  from econml.dml import NonParamDML
  from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

  est = NonParamDML(model_y=RandomForestRegressor(),
                    model_t=RandomForestClassifier(),
                    model_final=RandomForestRegressor(),
                    discrete_treatment=True)
  est.fit(Y, T, X=X, W=W) 
  treatment_effects = est.effect(X_test)
  ```

</details>

<details>
  <summary>Causal Forests (click to expand)</summary>

  ```Python
  from econml.causal_forest import CausalForest
  from sklearn.linear_model import LassoCV
  # Use defaults
  est = CausalForest()
  # Or specify hyperparameters
  est = CausalForest(n_trees=500, min_leaf_size=10, 
                     max_depth=10, subsample_ratio=0.7,
                     lambda_reg=0.01,
                     discrete_treatment=False,
                     model_T=LassoCV(), model_Y=LassoCV())
  est.fit(Y, T, X=X, W=W)
  treatment_effects = est.effect(X_test)
  # Confidence intervals via Bootstrap-of-Little-Bags for forests
  lb, ub = est.effect_interval(X_test, alpha=0.05)
  ```
</details>


<details>
  <summary>Orthogonal Random Forests (click to expand)</summary>

  ```Python
  from econml.ortho_forest import DMLOrthoForest, DROrthoForest
  from econml.sklearn_extensions.linear_model import WeightedLasso, WeightedLassoCV
  # Use defaults
  est = DMLOrthoForest()
  est = DROrthoForest()
  # Or specify hyperparameters
  est = DMLOrthoForest(n_trees=500, min_leaf_size=10,
                       max_depth=10, subsample_ratio=0.7,
                       lambda_reg=0.01,
                       discrete_treatment=False,
                       model_T=WeightedLasso(alpha=0.01), model_Y=WeightedLasso(alpha=0.01),
                       model_T_final=WeightedLassoCV(cv=3), model_Y_final=WeightedLassoCV(cv=3))
  est.fit(Y, T, X=X, W=W)
  treatment_effects = est.effect(X_test)
  # Confidence intervals via Bootstrap-of-Little-Bags for forests
  lb, ub = est.effect_interval(X_test, alpha=0.05)
  ```
</details>

<details>

<summary>Meta-Learners (click to expand)</summary>
  
  * XLearner

  ```Python
  from econml.metalearners import XLearner
  from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

  est = XLearner(models=GradientBoostingRegressor(),
                propensity_model=GradientBoostingClassifier(),
                cate_models=GradientBoostingRegressor())
  est.fit(Y, T, X=np.hstack([X, W]))
  treatment_effects = est.effect(np.hstack([X_test, W_test]))

  # Fit with bootstrap confidence interval construction enabled
  est.fit(Y, T, X=np.hstack([X, W]), inference='bootstrap')
  treatment_effects = est.effect(np.hstack([X_test, W_test]))
  lb, ub = est.effect_interval(np.hstack([X_test, W_test]), alpha=0.05) # Bootstrap CIs
  ```
  
  * SLearner

  ```Python
  from econml.metalearners import SLearner
  from sklearn.ensemble import GradientBoostingRegressor

  est = SLearner(overall_model=GradientBoostingRegressor())
  est.fit(Y, T, X=np.hstack([X, W]))
  treatment_effects = est.effect(np.hstack([X_test, W_test]))
  ```

  * TLearner

  ```Python
  from econml.metalearners import TLearner
  from sklearn.ensemble import GradientBoostingRegressor

  est = TLearner(models=GradientBoostingRegressor())
  est.fit(Y, T, X=np.hstack([X, W]))
  treatment_effects = est.effect(np.hstack([X_test, W_test]))
  ```
</details>

<details>
<summary>Doubly Robust Learners (click to expand)
</summary>

* Linear final stage

```Python
from econml.drlearner import LinearDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

est = LinearDRLearner(model_propensity=GradientBoostingClassifier(),
                      model_regression=GradientBoostingRegressor())
est.fit(Y, T, X=X, W=W)
treatment_effects = est.effect(X_test)
lb, ub = est.effect_interval(X_test, alpha=0.05)
```

* Sparse linear final stage

```Python
from econml.drlearner import SparseLinearDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

est = SparseLinearDRLearner(model_propensity=GradientBoostingClassifier(),
                            model_regression=GradientBoostingRegressor())
est.fit(Y, T, X=X, W=W)
treatment_effects = est.effect(X_test)
lb, ub = est.effect_interval(X_test, alpha=0.05)
```

* Nonparametric final stage

```Python
from econml.drlearner import ForestDRLearner
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

est = ForestDRLearner(model_propensity=GradientBoostingClassifier(),
                      model_regression=GradientBoostingRegressor())
est.fit(Y, T, X=X, W=W) 
treatment_effects = est.effect(X_test)
lb, ub = est.effect_interval(X_test, alpha=0.05)
```
</details>

<details>
<summary>Orthogonal Instrumental Variables (click to expand)</summary>

* Intent to Treat Doubly Robust Learner (discrete instrument, discrete treatment)

```Python
from econml.ortho_iv import LinearIntentToTreatDRIV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression

est = LinearIntentToTreatDRIV(model_Y_X=GradientBoostingRegressor(),
                              model_T_XZ=GradientBoostingClassifier(),
                              flexible_model_effect=GradientBoostingRegressor())
est.fit(Y, T, Z=Z, X=X) # OLS inference by default
treatment_effects = est.effect(X_test)
lb, ub = est.effect_interval(X_test, alpha=0.05) # OLS confidence intervals
```

</details>

<details>
<summary>Deep Instrumental Variables (click to expand)</summary>

```Python
import keras
from econml.deepiv import DeepIVEstimator

treatment_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(2,)),
                                    keras.layers.Dropout(0.17),
                                    keras.layers.Dense(64, activation='relu'),
                                    keras.layers.Dropout(0.17),
                                    keras.layers.Dense(32, activation='relu'),
                                    keras.layers.Dropout(0.17)])
response_model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(2,)),
                                  keras.layers.Dropout(0.17),
                                  keras.layers.Dense(64, activation='relu'),
                                  keras.layers.Dropout(0.17),
                                  keras.layers.Dense(32, activation='relu'),
                                  keras.layers.Dropout(0.17),
                                  keras.layers.Dense(1)])
est = DeepIVEstimator(n_components=10, # Number of gaussians in the mixture density networks)
                      m=lambda z, x: treatment_model(keras.layers.concatenate([z, x])), # Treatment model
                      h=lambda t, x: response_model(keras.layers.concatenate([t, x])), # Response model
                      n_samples=1 # Number of samples used to estimate the response
                      )
est.fit(Y, T, X=X, Z=Z) # Z -> instrumental variables
treatment_effects = est.effect(X_test)
```
</details>

 See the <a href="#references">References</a> section for more details.

### Interpretability
<details>
  <summary>Tree Interpreter of the CATE model (click to expand)</summary>
  
  ```Python
  from econml.cate_interpreter import SingleTreeCateInterpreter
  intrp = SingleTreeCateInterpreter(include_model_uncertainty=True, max_depth=2, min_samples_leaf=10)
  # We interpret the CATE model's behavior based on the features used for heterogeneity
  intrp.interpret(est, X)
  # Plot the tree
  plt.figure(figsize=(25, 5))
  intrp.plot(feature_names=['A', 'B', 'C', 'D'], fontsize=12)
  plt.show()
  ```
  ![image](notebooks/images/dr_cate_tree.png)
  
</details>

<details>
  <summary>Policy Interpreter of the CATE model (click to expand)</summary>
  
  ```Python
  from econml.cate_interpreter import SingleTreePolicyInterpreter
  # We find a tree-based treatment policy based on the CATE model
  intrp = SingleTreePolicyInterpreter(risk_level=0.05, max_depth=2, min_samples_leaf=1,min_impurity_decrease=.001)
  intrp.interpret(est, X, sample_treatment_costs=0.2)
  # Plot the tree
  plt.figure(figsize=(25, 5))
  intrp.plot(feature_names=['A', 'B', 'C', 'D'], fontsize=12)
  plt.show()
  ```
  ![image](notebooks/images/dr_policy_tree.png)
  
</details>

### Inference

Whenever inference is enabled, then one can get a more structure `InferenceResults` object with more elaborate inference information, such
as p-values and z-statistics. When the CATE model is linear and parametric, then a `summary()` method is also enabled. For instance:

  ```Python
  from econml.dml import LinearDML
  # Use defaults
  est = LinearDML()
  est.fit(Y, T, X=X, W=W)
  # Get the effect inference summary, which includes the standard error, z test score, p value, and confidence interval given each sample X[i]
  est.effect_inference(X_test).summary_frame(alpha=0.05, value=0, decimals=3)
  # Get the population summary for the entire sample X
  est.effect_inference(X_test).population_summary(alpha=0.1, value=0, decimals=3, tol=0.001)
  #  Get the parameter inference summary for the final model
  est.summary()
  ```
  
  <details><summary>Example Output (click to expand)</summary>
  
  ```Python
  # Get the effect inference summary, which includes the standard error, z test score, p value, and confidence interval given each sample X[i]
  est.effect_inference(X_test).summary_frame(alpha=0.05, value=0, decimals=3)
  ```
  ![image](notebooks/images/summary_frame.png)
  
  ```Python
  # Get the population summary for the entire sample X
  est.effect_inference(X_test).population_summary(alpha=0.1, value=0, decimals=3, tol=0.001)
  ```
  ![image](notebooks/images/population_summary.png)
  
  ```Python
  #  Get the parameter inference summary for the final model
  est.summary()
  ```
  ![image](notebooks/images/summary.png)
  
  </details>
  
To see more complex examples, go to the [notebooks](https://github.com/Microsoft/EconML/tree/master/notebooks) section of the repository. For a more detailed description of the treatment effect estimation algorithms, see the EconML [documentation](https://econml.azurewebsites.net/).

# For Developers

You can get started by cloning this repository. We use 
[setuptools](https://setuptools.readthedocs.io/en/latest/index.html) for building and distributing our package.
We rely on some recent features of setuptools, so make sure to upgrade to a recent version with
`pip install setuptools --upgrade`.  Then from your local copy of the repository you can run `python setup.py develop` to get started.

## Running the tests

This project uses [pytest](https://docs.pytest.org/) for testing.  To run tests locally after installing the package, 
you can use `python setup.py pytest`.

## Generating the documentation

This project's documentation is generated via [Sphinx](https://www.sphinx-doc.org/en/master/index.html).  Note that we use [graphviz](https://graphviz.org/)'s 
`dot` application to produce some of the images in our documentation, so you should make sure that `dot` is installed and in your path.

To generate a local copy of the documentation from a clone of this repository, just run `python setup.py build_sphinx -W -E -a`, which will build the documentation and place it under the `build/sphinx/html` path. 

The reStructuredText files that make up the documentation are stored in the [docs directory](https://github.com/Microsoft/EconML/tree/master/doc); module documentation is automatically generated by the Sphinx build process.

# Blogs and Publications

* June 2019: [Treatment Effects with Instruments paper](https://arxiv.org/pdf/1905.10176.pdf)

* May 2019: [Open Data Science Conference Workshop](https://odsc.com/speakers/machine-learning-estimation-of-heterogeneous-treatment-effect-the-microsoft-econml-library/) 

* 2018: [Orthogonal Random Forests paper](http://proceedings.mlr.press/v97/oprescu19a.html)

* 2017: [DeepIV paper](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf)

# Citation

If you use EconML in your research, please cite us as follows:

   Microsoft Research. **EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation.** https://github.com/microsoft/EconML, 2019. Version 0.x.

BibTex:

```
@misc{econml,
  author={Microsoft Research},
  title={{EconML}: {A Python Package for ML-Based Heterogeneous Treatment Effects Estimation}},
  howpublished={https://github.com/microsoft/EconML},
  note={Version 0.x},
  year={2019}
}
```

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

X Nie, S Wager.
**Quasi-Oracle Estimation of Heterogeneous Treatment Effects.**
[*Biometrika*](https://doi.org/10.1093/biomet/asaa076), 2020

V. Syrgkanis, V. Lei, M. Oprescu, M. Hei, K. Battocchi, G. Lewis.
**Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments.**
[*Proceedings of the 33rd Conference on Neural Information Processing Systems (NeurIPS)*](https://arxiv.org/abs/1905.10176), 2019
**(Spotlight Presentation)**

D. Foster, V. Syrgkanis.
**Orthogonal Statistical Learning.**
[*Proceedings of the 32nd Annual Conference on Learning Theory (COLT)*](https://arxiv.org/pdf/1901.09036.pdf), 2019
**(Best Paper Award)**

M. Oprescu, V. Syrgkanis and Z. S. Wu.
**Orthogonal Random Forest for Causal Inference.**
[*Proceedings of the 36th International Conference on Machine Learning (ICML)*](http://proceedings.mlr.press/v97/oprescu19a.html), 2019.

S. Künzel, J. Sekhon, J. Bickel and B. Yu.
**Metalearners for estimating heterogeneous treatment effects using machine learning.**
[*Proceedings of the national academy of sciences, 116(10), 4156-4165*](https://www.pnas.org/content/116/10/4156), 2019.

S. Athey, J. Tibshirani, S. Wager.
**Generalized random forests.**
[*Annals of Statistics, 47, no. 2, 1148--1178*](https://projecteuclid.org/euclid.aos/1547197251), 2019.

V. Chernozhukov, D. Nekipelov, V. Semenova, V. Syrgkanis.
**Plug-in Regularized Estimation of High-Dimensional Parameters in Nonlinear Semiparametric Models.**
[*Arxiv preprint arxiv:1806.04823*](https://arxiv.org/abs/1806.04823), 2018.

S. Wager, S. Athey.
**Estimation and Inference of Heterogeneous Treatment Effects using Random Forests.**
[*Journal of the American Statistical Association, 113:523, 1228-1242*](https://www.tandfonline.com/doi/citedby/10.1080/01621459.2017.1319839), 2018.

Jason Hartford, Greg Lewis, Kevin Leyton-Brown, and Matt Taddy. **Deep IV: A flexible approach for counterfactual prediction.** [*Proceedings of the 34th International Conference on Machine Learning, ICML'17*](http://proceedings.mlr.press/v70/hartford17a/hartford17a.pdf), 2017.

V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, and a. W. Newey. **Double Machine Learning for Treatment and Causal Parameters.** [*ArXiv preprint arXiv:1608.00060*](https://arxiv.org/abs/1608.00060), 2016.
