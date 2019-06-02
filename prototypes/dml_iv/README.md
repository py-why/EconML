An orthogonal machine learning approach to estimation of heterogeneous
treatment effects with an endogenous treatment and an instrument. It
implements the algorithms presented in the paper:

**Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments**

_Vasilis Syrgkanis, Victor Lei, Miruna Oprescu, Maggie Hei, Keith Battocchi, Greg Lewis_

[https://arxiv.org/abs/1905.10176](https://arxiv.org/abs/1905.10176)

# Guide to the Files

- dml_iv.py : contains the classes that implement the DMLIV algorithm (and variants) 
- dr_iv.py : contains the classes that implement the DRIV algorithm (and variants)
- deep_dml_iv.py : contains children class of DMLIV that use keras neural net models as treatment effect models
- dml_ate_iv.py : contains the class that implements DMLATEIV for ATE estimation under the assumption of neither compliance heterogeneity nor effect heterogeneity
- utilities.py : some utility classes and wrappers that were used
- xgb_utilities.py : some utility wrapper classes for XGBoost
- NLSYM_Linear.ipynb, NLSYM_GBM.ipynb : a notebook that applies the methods to the real world data from Card's paper: [Using Geographic Variation in College Proximity to Estimate the Return to Schooling](http://davidcard.berkeley.edu/papers/geo_var_schooling.pdf) 
- NLSYM_Semi_Synthetic_Linear.ipynb, NLSYM_Semi_Synthetic_GBM.ipynb : semi-synthetic monte carlo, that uses the co-variates from Card's paper, but generates treatments and outcomes in a known manner.
- TA_DGP_Analysis.ipynb : a notebook that applies the methods to data that are drawn from a DGP similar to the private TripAdvisor data set.
- coverage_experiment.py : a monte carlo experiment of the coverage of the CIs of DRIV on a DGP that is similar to the TripAdvisor data
- post_processing.ipynb : a notebook that reads from the results of the coverage_experiment.py and plots several figures.

