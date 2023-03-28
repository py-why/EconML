import unittest

import numpy as np
from model_selection import *
from model_selection_utils import *
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler


class TestIsDataScaled(unittest.TestCase):

    def test_scaled_data(self):
        # Test with data that is already centered and scaled
        X = np.array([[0.0, -1.0], [1.0, 0.0], [-1.0, 1.0]])
        scale = StandardScaler()
        scaled_X = scale.fit_transform(X)
        self.assertTrue(is_data_scaled(scaled_X))

    def test_unscaled_data(self):
        # Test with data that is not centered and scaled
        X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        self.assertFalse(is_data_scaled(X))

    def test_large_scaled_data(self):
        # Test with a larger dataset that is already centered and scaled
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        scale = StandardScaler()
        scaled_X = scale.fit_transform(X)
        self.assertTrue(is_data_scaled(scaled_X))

    def test_large_unscaled_data(self):
        np.random.seed(42)
        X = np.random.randn(1000, 5)
        self.assertFalse(is_data_scaled(X))

    def test_is_data_scaled_with_scaled_iris_dataset(self):
        X, y = load_iris(return_X_y=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert is_data_scaled(X_scaled) == True

    def test_is_data_scaled_with_unscaled_iris_dataset(self):
        X, y = load_iris(return_X_y=True)
        assert is_data_scaled(X) == False
        
    def test_is_data_scaled_with_scaled_california_housing_dataset(self):
        X, y = housing = fetch_california_housing(return_X_y=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert is_data_scaled(X_scaled) == True
        
    def test_is_data_scaled_with_unscaled_california_housing_dataset(self):
        X, y = fetch_california_housing(return_X_y=True)
        assert is_data_scaled(X) == False



if __name__ == '__main__':
    unittest.main()