# test_placebo.py
# Copyright (c) PyWhy contributors.
# Licensed under the MIT License.

import unittest
import numpy as np
from econml.dr import LinearDRLearner
from econml.inference import BootstrapInference
from sklearn.linear_model import Lasso, LogisticRegression

class TestPlaceboEconML(unittest.TestCase):
    def test_placebo_effect(self):
        """
        This test implements a placebo test in the spirit of the whitepaper design.

        Steps:
            1. Simulate confounders X1 and X2.
            2. Assign treatment T using a logistic function of the confounders.
            3. Generate a placebo outcome (Y_placebo) that depends on the confounders but NOT on T.
               (Thus, any estimated treatment effect should be zero.)
            4. Fit a LinearDRLearner with BootstrapInference.
            5. Check that the estimated average treatment effect (ATE) is close to zero,
               and that its confidence interval contains zero.
            6. Verify that the p-value is > 0.05, meaning we fail to reject the null hypothesis.
        """
        # -----------------------------
        # 1. Data Simulation
        # -----------------------------
        np.random.seed(123)  # for reproducibility
        n = 1000  # sample size

        # Generate two confounders: X1 and X2 ~ N(0,1)
        X1 = np.random.normal(0, 1, n)
        X2 = np.random.normal(0, 1, n)
        X = np.column_stack([X1, X2])  # Combine into a matrix

        # Define a helper function for the logistic transformation (sigmoid)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # Treatment assignment probability
        p_treat = sigmoid(0.5 * X1 - 0.25 * X2)
        T = np.random.binomial(1, p_treat, size=n)

        # -----------------------------
        # 2. Generate Placebo Outcome
        # -----------------------------
        noise = np.random.normal(0, 1, n)
        Y_placebo = 1.5 * X1 - 2.0 * X2 + noise  # Does NOT depend on T

        # Check correlation between treatment and placebo outcome
        corr = np.corrcoef(T, Y_placebo)[0,1]
        print(f"Placebo Outcome - Correlation between T and Y_placebo: {corr:.4f}")

        # -----------------------------
        # 3. Fit LinearDRLearner with BootstrapInference
        # -----------------------------
        est = LinearDRLearner(
            model_regression=Lasso(alpha=0.01),
            model_propensity=LogisticRegression(solver='lbfgs'),
        )

        est.fit(Y_placebo, T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))

        # -----------------------------
        # 4. Extract ATE, Confidence Interval, and p-value
        # -----------------------------
        ate = est.ate(X)
        ci_lower, ci_upper = est.ate_interval(X, alpha=0.05)
        p_value = est.ate_inference(X).pvalue() 
        print(f"Placebo Outcome - p_value: {p_value:.4f}") 

        # -----------------------------
        # 5. Assertions (Quality Check)
        # -----------------------------
        self.assertAlmostEqual(ate, 0.0, delta=0.1,
                               msg=f"Estimated ATE on the placebo outcome should be near 0, but got {ate:.4f}.")
        self.assertTrue(ci_lower <= 0 <= ci_upper,
                        msg=f"The 95% CI for the ATE should contain 0, but got ({ci_lower:.4f}, {ci_upper:.4f}).")
        self.assertGreater(p_value, 0.05,
                           msg=f"P-value should be > 0.05 to confirm no effect, but got {p_value:.4f}.")

if __name__ == '__main__':
    unittest.main()
