# About

An orthogonal machine learning approach to estimation of heterogeneous
treatment effect with an endogenous treatment and an instrument. Based on the paper:

**Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments**  
_Vasilis Syrgkanis, Victor Lei, Miruna Oprescu, Maggie Hei, Keith Battocchi, Greg Lewis_  
[https://arxiv.org/abs/1905.10176](https://arxiv.org/abs/1905.10176)

# Guide to the Files

- dml_iv.py : contains the classes that implement the DMLIV algorithm (and variants) 
- dr_iv.py : contains the classes that implement the DRIV algorithm (and variants)
- deep_dml_iv.py, deep_dr_iv.py : contains children classes of DMLIV and DRIV that use keras neural net models as treatment effect models
- dml_ate_iv.py : contains the class that implements DMLATEIV for ATE estimation under the assumption of neither compliance heterogeneity nor effect heterogeneity
- utilities.py : some utility classes that were used
- xgb_utilities.py : some utility classes that were used related to wrapping xgboost
- NLSYM_Linear.ipynb, NLSYM_GBM.ipynb, NLSYM_Semi_Synthetic_Linear.ipynb, NLSYM_Semi_Synthetic_GBM.ipynb : a notebook that applies the methods to the real world data from Card's paper: [Using Geographic Variation in College Proximity to Estimate the Return to Schooling](http://davidcard.berkeley.edu/papers/geo_var_schooling.pdf) 
- TA_DGP_analysis.ipynb, coverage.py, post_processing.ipynb : synthetic data experiments with a DGP that resembles the TripAdvisor experiment data
