import unittest

import numpy as np
from model_selection import *
from model_selection_utils import *
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

class TestSearchEstimatorList(unittest.TestCase):
    def test_initialization(self):
        with self.assertRaises(ValueError):
            SearchEstimatorList(search='invalid_search')

    def test_auto_param_grid_discrete(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        search_estimator_list = SearchEstimatorList(is_discrete=True)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)

    def test_auto_param_grid_continuous(self):
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        search_estimator_list = SearchEstimatorList(is_discrete=False)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)
        print("Best estimator: ", search_estimator_list.best_estimator_)
        print("Best score: ", search_estimator_list.best_score_)
        print("Best parameters: ", search_estimator_list.best_params_)

    def test_random_forest_discrete(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestClassifier()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]
        search_estimator_list = SearchEstimatorList(estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=True)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)
        print("Best estimator: ", search_estimator_list.best_estimator_)
        print("Best score: ", search_estimator_list.best_score_)
        print("Best parameters: ", search_estimator_list.best_params_)

    def test_random_forest_continuous(self):
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestRegressor()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]
        search_estimator_list = SearchEstimatorList(estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=False)
        search_estimator_list.select(X_train, y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)
        print("Best estimator: ", search_estimator_list.best_estimator_)
        print("Best score: ", search_estimator_list.best_score_)
        print("Best parameters: ", search_estimator_list.best_params_)
    
    def test_warning(self):
        X, y = fetch_california_housing(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        estimator_list = [RandomForestRegressor()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]
        with self.assertWarns(UserWarning):
            search_estimator_list = SearchEstimatorList(estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=False)
        
if __name__ == '__main__':
    unittest.main()