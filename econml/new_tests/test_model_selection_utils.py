import unittest

import numpy as np
from econml.sklearn_extensions.model_selection import *
from econml.sklearn_extensions.model_selection_utils import *
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV


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


class TestFlattenList(unittest.TestCase):

    def test_flatten_empty_list(self):
        input = []
        expected_output = []
        self.assertEqual(flatten_list(input), expected_output)

    def test_flatten_simple_list(self):
        input = [1, 10, 15]
        expected_output = [1, 10, 15]
        self.assertEqual(flatten_list(input), expected_output)

    def test_flatten_nested_list(self):
        input = [1, [10, 15], [20, [25, 30]]]
        expected_output = [1, 10, 15, 20, 25, 30]
        self.assertEqual(flatten_list(input), expected_output)

    # Check functionality for below
    # def test_flatten_none_list(self):
    #     input = [[1, 10, None], 15, None]
    #     expected_output = [1, 10, None, 15, None]
    #     self.assertEqual(flatten_list(input), expected_output)

    def test_flatten_iris_dataset(self):
        X = load_iris()
        input = X.data.tolist()
        expected_output = sum(X.data.tolist(), [])
        self.assertEqual(flatten_list(input), expected_output)

    def test_flatten_california_housing_dataset(self):
        X = fetch_california_housing()
        input = X.data.tolist()
        expected_output = sum(X.data.tolist(), [])
        self.assertEqual(flatten_list(input), expected_output)


class TestIsPolynomialPipeline(unittest.TestCase):

    def test_is_polynomial_pipeline_true(self):
        X = np.array([[5, 10], [15, 20], [25, 30], [35, 40], [45, 50]])
        y = np.array([15, 29, 38, 47, 55])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', ElasticNetCV())
        ])
        model.fit(X_scaled, y)
        assert is_polynomial_pipeline(model) == True

    def test_is_polynomial_pipeline_false(self):
        model = ElasticNetCV()
        assert is_polynomial_pipeline(model) == False

    def test_is_polynomial_pipeline_false_step_number(self):
        X, y = load_iris(return_X_y=True)
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LogisticRegressionCV()),
            ('step_false', '')
        ])
        assert is_polynomial_pipeline(model) == False

    def test_is_polynomial_pipeline_interchange_steps(self):
        X, y = load_iris(return_X_y=True)
        model = Pipeline([
            ('poly', LogisticRegressionCV()),
            ('linear', PolynomialFeatures(degree=2)),
        ])
        assert is_polynomial_pipeline(model) == False

    # Cross-check functionaity - can the 'poly' keyword be changed to something else
    def test_is_polynomial_pipeline_false_first_step(self):
        X, y = fetch_california_housing(return_X_y=True)
        model = Pipeline([
            ('not_poly', PolynomialFeatures(degree=2)),
            ('linear', ElasticNetCV())
        ])
        assert is_polynomial_pipeline(model) == True


class TestCheckListType(unittest.TestCase):

    def test_check_list_type_true(self):
        list = ['linear', LogisticRegressionCV(), KFold()]
        assert check_list_type(list) == True

    def test_check_list_type_false_string(self):
        list = [18, LogisticRegressionCV(), KFold()]
        try:
            check_list_type(list)
        except TypeError as e:
            assert str(e) == "The list must contain only strings, sklearn model objects, and sklearn model selection objects."

    def test_check_list_type_empty(self):
        list = []
        try:
            check_list_type(list)
        except ValueError as e:
            assert str(e) == "Estimator list is empty. Please add some models or use some of the defaults provided."

    def test_check_list_type_all_strings(self):
        list = ['linear', 'lasso', 'forest']
        assert check_list_type(list) == True

    def test_check_list_type_all_models(self):
        list = [LogisticRegressionCV(), ElasticNetCV()]
        assert check_list_type(list) == True

    def test_check_list_duplicate_models_strings(self):
        list = [LogisticRegressionCV(), LogisticRegressionCV(), 'linear', 'linear']
        assert check_list_type(list) == True


class TestSelectContinuousEstimator(unittest.TestCase):

    def test_select_continuous_estimator_valid(self):
        assert isinstance(select_continuous_estimator('linear'), ElasticNetCV)
        assert isinstance(select_continuous_estimator('forest'), RandomForestRegressor)
        assert isinstance(select_continuous_estimator('gbf'), GradientBoostingRegressor)
        assert isinstance(select_continuous_estimator('nnet'), MLPRegressor)
        assert isinstance(select_continuous_estimator('poly'), Pipeline)

    def test_select_continuous_estimator_invalid(self):
        try:
            select_continuous_estimator('ridge')
        except ValueError as e:
            assert str(e) == 'Unsupported estimator type: ridge'


class TestSelectDiscreteEstimator(unittest.TestCase):

    def test_select_discrete_estimator_valid(self):
        assert isinstance(select_discrete_estimator('linear'), LogisticRegressionCV)
        assert isinstance(select_discrete_estimator('forest'), RandomForestClassifier)
        assert isinstance(select_discrete_estimator('gbf'), GradientBoostingClassifier)
        assert isinstance(select_discrete_estimator('nnet'), MLPClassifier)
        assert isinstance(select_discrete_estimator('poly'), Pipeline)

    def test_select_discrete_estimator_invalid(self):
        try:
            select_discrete_estimator('lasso')
        except ValueError as e:
            assert str(e) == 'Unsupported estimator type: lasso'


class TestSelectEstimator(unittest.TestCase):

    def test_select_estimator_valid(self):
        assert isinstance(select_estimator('linear', is_discrete=False), ElasticNetCV)
        assert isinstance(select_estimator('forest', is_discrete=False), RandomForestRegressor)
        assert isinstance(select_estimator('gbf', is_discrete=False), GradientBoostingRegressor)
        assert isinstance(select_estimator('nnet', is_discrete=False), MLPRegressor)
        assert isinstance(select_estimator('poly', is_discrete=False), Pipeline)

        assert isinstance(select_estimator('linear', is_discrete=True), LogisticRegression)
        assert isinstance(select_estimator('forest', is_discrete=True), RandomForestClassifier)
        assert isinstance(select_estimator('gbf', is_discrete=True), GradientBoostingClassifier)
        assert isinstance(select_estimator('nnet', is_discrete=True), MLPClassifier)
        assert isinstance(select_estimator('poly', is_discrete=True), Pipeline)

    def test_select_estimator_invalid_estimator(self):
        try:
            select_estimator('lasso', is_discrete=True)
        except ValueError as e:
            assert str(e) == 'Unsupported estimator type: lasso'

    def test_select_estimator_invalid(self):
        try:
            select_estimator('linear', is_discrete=None)
        except ValueError as e:
            assert str(e) == 'Unsupported target type: None'


if __name__ == '__main__':
    unittest.main()
