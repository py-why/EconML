import numpy as np
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

class LinearRegressionWrapper:
    def __init__(self, fit_intercept=True, cov_type="HC0"):
        self.model = StatsModelsLinearRegression(fit_intercept=fit_intercept, cov_type=cov_type)

    def fit(self, X, y, sample_weight=None, freq_weight=None, sample_var=None):
        return self.model.fit(X, y, sample_weight, freq_weight, sample_var)

    def compute_A(self, X, freq_weight):
        return self.model.compute_A(X, freq_weight)

    def compute_B(self, X, y, freq_weight):
        return self.model.compute_B(X, y, freq_weight)

# Example usage:
linear_regression = LinearRegressionWrapper()
linear_regression.fit(X, y, sample_weight, freq_weight, sample_var)

A = linear_regression.compute_A(X, freq_weight)
B = linear_regression.compute_B(X, y, freq_weight)

# Now you can export A and B outside of the LinearRegressionWrapper class and aggregate them


