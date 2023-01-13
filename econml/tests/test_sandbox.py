import numpy as np
import scipy
from econml.dml import LinearDML
from econml.inference import BootstrapInference
import time
import matplotlib.pyplot as plt

np.random.seed(123)
X = np.random.normal(size=(1000, 5))
T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
all_n_bootstrap_samples = [2**i for i in range(8)]
all_bootstrapping_times = []
final_bootstrapping_times = []

for n_bootstrap_samples in all_n_bootstrap_samples:
    est = LinearDML(discrete_treatment=True)
    start_time = time.time()
    est.fit(y, T, X=X, W=None, inference=BootstrapInference(n_bootstrap_samples=n_bootstrap_samples, only_final=False))
    end_time = time.time()
    elapsed_time = end_time - start_time
    all_bootstrapping_times.append(elapsed_time)
    print("Time elapsed: ", elapsed_time)

    est = LinearDML(discrete_treatment=True)
    start_time = time.time()
    est.fit(y, T, X=X, W=None, inference=BootstrapInference(n_bootstrap_samples=n_bootstrap_samples, only_final=True))
    end_time = time.time()
    elapsed_time = end_time - start_time
    final_bootstrapping_times.append(elapsed_time)
    print("Time elapsed: ", elapsed_time)

plt.plot(all_bootstrapping_times, color='r', label='bootstrap all')
plt.plot(final_bootstrapping_times, color='b', label='bootstrap final')
plt.xlabel("log(samples)")
plt.ylabel("times")
plt.title("bootstrap time comparison")
plt.legend()
plt.show()

