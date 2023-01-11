import numpy as np
import scipy
from econml.dml import LinearDML
from econml.inference import BootstrapInference

np.random.seed(123)
X = np.random.normal(size=(1000, 5))
T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
est = LinearDML(discrete_treatment=True)
est.fit(y, T, X=X, W=None, inference=BootstrapInference(10))

