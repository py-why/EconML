# Guide to the Files

- DML-IV-binary.ipynb, DML-IV-Continuous : notebook that shows how to use the main classes in the repo and which runs some basic monte carlo simulations to judge its performance. For binary and continuous treatments/instruments correspondingly.
- dml_iv.py : contains the classes that implement the DMLIV algorithm in the write-up (and variants) 
- dr_iv.py : contains the classes that implement the DRIV algorithm in the write-up (and variants)
- deep_dml_iv.py : contains children classes of DMLIV and DRIV that use keras neural net models as treatment effect models
- dml_ate_iv.py : contains the class that implements DMLATEIV for ATE estimation under the assumption of neither compliance heterogeneity nor effect heterogeneity
- ortho_dml_iv.py : implements the orthogonal correction of the DMLIV algorithm
- ortho_linear_regression.py : a base class that minimizes penalized square loss plus a linear gradient correction term. used in OrthoDMLIV
- utilities.py : some utility classes that were used
- NLSYM.ipynb : a notebook that applies the methods to the real world data from Card's paper: [Using Geographic Variation in College Proximity to Estimate the Return to Schooling](http://davidcard.berkeley.edu/papers/geo_var_schooling.pdf) 
- Angrist_IV.ipynb : a notebook that applies the methods to the real world data from Angrist and Krueger's paper: [Does Compulsory School Attendance Affect Schooling and Earnings?](https://economics.mit.edu/faculty/angrist/data1/data/angkru1991)
