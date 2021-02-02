# Introduction

Code for recreating figures and numbers from paper: "Double/Debiased Machine Learning for Dynamic Treatment Effects"

# Generating paper figures

To generate the paper figures run

```
./all_coverage.sh
```

This will produce a set of `.png` files that contain distributions of point estimates, distribution of standard error estimates,
coverage probabilities of confidence intervals (for both the estimated parameters and for a set of target counterfactual
policies).

The results for constant effects will be in the newly created folder:

`results/long_range_constant`

and the results for heterogeneous effects in the folder:

`results/long_range_hetero`

The code assumes Python 3 and requires the standard packages: statsmodels, numpy, scipy, scikit-learn, matplotlib. All can be pip installed.

To create the figure that contains the benchmark performance run the jupyter notebook: `high_dim_state_any_m_panel.ipynb`.

The jupyter notebook `high_dim_state_any_panel_hetero.ipynb` compares performance with benchmarks when there is effect heterogeneity.


# Files

## Estimator Classes

* `panel_dynamic_dml.py` : Contains the estimator `DynamicPanelDML` that estimates dynamic treatment effects without heterogeneity.
* `hetero_panel_dynamic_dml.py`: Contains the estimator `HeteroDynamicPanelDML` that estimates heterogeneous dynamic treatment effects.


## Expository Notebooks

* `high_dim_state_any_m_panel.ipynb`: examples of runnign the estimators with constant effects
* `high_dim_state_any_m_panel_hetero.ipynb`: examples of running the estiamtors with heterogeneous effects


## Data Generating Processes

* `dynamic_panel_dgp.py`: Contains several data generating processes for dynamic treatment effect estimation


## Coverage Experiments

* `coverage_panel.py`: Runs coverage experiments for contant dynamic effects
* `coverage_panel_hetero.py`: Runs coverage experiments for heterogeneous dynamic effects
* `all_coverage.sh`: Shell script that runs all coverage experiments
* `postprocess_panel.ipynb`: Post-processing notebook that reads the coverage results for constant effects so as to produce more plots later on
* `postprocess_panel_hetero.ipynb`: Post-processing notebook that reads the coverage results for heterogeneous effects so as to produce more plots later on
