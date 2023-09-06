import numpy as np

class FederalLearner:
    def __init__(self):
        self.A_i = None
        self.B_i = None
        self.A = None
        self.B = None
        self.theta_hat = None
        self.variance_matrix = None

    def fit(self, X_i, y_i):
        self.A_i = X_i.T @ X_i
        self.B_i = X_i.T @ y_i

    def initialize_from_existing(self, models):
        self.A = sum([model.StatsModelsLinearRegression.compute_A() for model in models])
        self.B = sum([model.StatsModelsLinearRegression.compute_B() for model in models])

    def solve_linear_equation(self):
        self.theta_hat = np.linalg.solve(self.A, self.B)
        residuals = self.B - self.A @ self.theta_hat
        self.variance_matrix = np.linalg.inv(self.A) @ residuals.T @ residuals @ np.linalg.inv(self.A.T)

    @classmethod
    def from_aggregated_values(cls, A, B):
        instance = cls()
        instance.A_i = A
        instance.B_i = B
        instance.solve_linear_equation()
        return instance