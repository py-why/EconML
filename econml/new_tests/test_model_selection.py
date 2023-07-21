import unittest

import numpy as np
from econml.sklearn_extensions.model_selection import *
from econml.sklearn_extensions.model_selection_utils import *
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR


class TestSearchEstimatorListClassifier(unittest.TestCase):
    def setUp(self):
        self.expected_accuracy = 0.9
        self.expected_f1_score = 0.9
        self.accuracy_tolerance = 0.05
        self.f1_score_tolerance = 0.05
        self.is_discrete = True
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def test_initialization(self):
        with self.assertRaises(ValueError):
            SearchEstimatorList(estimator_list='invalid_estimator')

    def test_auto_param_grid_discrete(self):

        search_estimator_list = SearchEstimatorList(is_discrete=self.is_discrete, scaling=False)
        search_estimator_list.fit(self.X_train, self.y_train)
        self.assertIsNotNone(search_estimator_list.best_estimator_)
        self.assertIsNotNone(search_estimator_list.best_score_)
        self.assertIsNotNone(search_estimator_list.best_params_)

    def test_linear_estimator(self):
        search = SearchEstimatorList(estimator_list='linear', is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertIsInstance(search.complete_estimator_list[0], LogisticRegressionCV)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_poly_estimator(self):
        search = SearchEstimatorList(estimator_list='poly', is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertTrue(is_polynomial_pipeline(search.complete_estimator_list[0]))

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_forest_estimator(self):
        search = SearchEstimatorList(estimator_list='forest', is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertIsInstance(search.complete_estimator_list[0], RandomForestClassifier)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_gbf_estimator(self):
        search = SearchEstimatorList(estimator_list='gbf', is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertIsInstance(search.complete_estimator_list[0], GradientBoostingClassifier)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_nnet_estimator(self):
        search = SearchEstimatorList(estimator_list='nnet', is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertIsInstance(search.complete_estimator_list[0], MLPClassifier)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_linear_and_forest_estimators(self):
        search = SearchEstimatorList(estimator_list=['linear', 'forest'], is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 2)
        self.assertEqual(len(search.param_grid_list), 2)
        self.assertIsInstance(search.complete_estimator_list[0], LogisticRegressionCV)
        self.assertIsInstance(search.complete_estimator_list[1], RandomForestClassifier)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_all_estimators(self):
        search = SearchEstimatorList(estimator_list=['linear', 'forest',
                                     'gbf', 'nnet', 'poly'], is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 5)
        self.assertEqual(len(search.param_grid_list), 5)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_logistic_regression_estimator(self):
        search = SearchEstimatorList(estimator_list=LogisticRegression(), is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_logistic_regression_cv_estimator(self):
        search = SearchEstimatorList(estimator_list=LogisticRegressionCV(),
                                     is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')
        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_empty_estimator_list(self):
        with self.assertRaises(ValueError):
            search = SearchEstimatorList(estimator_list=[], is_discrete=self.is_discrete, scaling=False)

    def test_invalid_regressor(self):
        with self.assertRaises(TypeError):
            estimator_list = [SVR(kernel='linear')]
            search = SearchEstimatorList(estimator_list=estimator_list, is_discrete=self.is_discrete)

    def test_polynomial_pipeline_regressor(self):
        with self.assertRaises(TypeError):
            estimator_list = [make_pipeline(PolynomialFeatures(), ElasticNetCV())]
            search = SearchEstimatorList(estimator_list=estimator_list, is_discrete=self.is_discrete)

    def test_mlp_regressor(self):
        with self.assertRaises(TypeError):
            estimator_list = [MLPRegressor()]
            search = SearchEstimatorList(estimator_list=estimator_list, is_discrete=self.is_discrete)

    def test_random_forest_regressor(self):
        with self.assertRaises(TypeError):
            estimator_list = [RandomForestRegressor()]
            search = SearchEstimatorList(estimator_list=estimator_list, is_discrete=self.is_discrete)

    def test_gradient_boosting_regressor(self):
        with self.assertRaises(TypeError):
            estimator_list = [GradientBoostingRegressor()]
            search = SearchEstimatorList(estimator_list=estimator_list, is_discrete=self.is_discrete)

    def test_combined_estimators(self):
        with self.assertRaises(TypeError):
            estimator_list = [LogisticRegression(), SVC(), GradientBoostingRegressor()]
            search = SearchEstimatorList(estimator_list=estimator_list, is_discrete=self.is_discrete)

    def test_random_forest_discrete(self):
        estimator_list = [RandomForestClassifier()]
        param_grid_list = [{'n_estimators': [10, 50, 100], 'max_depth': [3, 5, None]}]

        search = SearchEstimatorList(
            estimator_list=estimator_list, param_grid_list=param_grid_list, is_discrete=self.is_discrete, scaling=False)
        search.fit(self.X_train, self.y_train)

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)

        self.assertIsNotNone(search.best_estimator_)
        self.assertIsNotNone(search.best_score_)
        self.assertIsNotNone(search.best_params_)

    def test_data_scaling(self):
        search = SearchEstimatorList(estimator_list='linear', is_discrete=self.is_discrete, scaling=True)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertIsInstance(search.complete_estimator_list[0], LogisticRegressionCV)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    def test_custom_scoring_function(self):
        def custom_scorer(y_true, y_pred):
            return f1_score(y_true, y_pred, average='macro')

        search = SearchEstimatorList(estimator_list='linear', is_discrete=self.is_discrete,
                                     scaling=False, scoring=custom_scorer)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertIsInstance(search.complete_estimator_list[0], LogisticRegressionCV)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    # def test_refit_false(self):
    #     search = SearchEstimatorList(estimator_list='linear', is_discrete=self.is_discrete, scaling=False, refit=False)
    #     search.fit(self.X_train, self.y_train)
    #     with self.assertRaises(NotFittedError):
    #         y_pred = search.predict(self.X_test)

    def test_custom_random_state(self):
        search = SearchEstimatorList(estimator_list='linear', is_discrete=self.is_discrete,
                                     scaling=False, random_state=42)
        search.fit(self.X_train, self.y_train)
        y_pred = search.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='macro')

        self.assertEqual(len(search.complete_estimator_list), 1)
        self.assertEqual(len(search.param_grid_list), 1)
        self.assertIsInstance(search.complete_estimator_list[0], LogisticRegressionCV)

        self.assertGreaterEqual(acc, self.expected_accuracy)
        self.assertGreaterEqual(f1, self.expected_f1_score)

    
    def test_invalid_incorrect_scoring_numbers(self):
        with self.assertRaises(ValueError):
            search = SearchEstimatorList(estimator_list='linear', is_discrete=self.is_discrete,
                                         scaling=False, scoring=123)


if __name__ == '__main__':
    unittest.main()
