
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
from sklearn.base import BaseEstimator, is_regressor
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNet,
                                  ElasticNetCV, Lars, Lasso, LassoLars,
                                  LinearRegression, LogisticRegression,
                                  LogisticRegressionCV,
                                  OrthogonalMatchingPursuit, Ridge)
from sklearn.model_selection import BaseCrossValidator
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MaxAbsScaler, MinMaxScaler,
                                   PolynomialFeatures, RobustScaler,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold, StratifiedKFold, check_cv, GridSearchCV, BaseCrossValidator, RandomizedSearchCV
import warnings

# For regression problems
models_regression = [
    ElasticNetCV(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    MLPRegressor()
]


# For classification problems
models_classification = [
    LogisticRegressionCV(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    MLPClassifier()
]

scaling_lst =  [StandardScaler(), MinMaxScaler(), RobustScaler(), MaxAbsScaler()]
model_list = ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']     

def scale_pipeline(model):
    """
    Returns a pipeline that scales the input data using StandardScaler and applies the given model.

    Args:
        model : estimator object
            A model object that implements the scikit-learn estimator interface.

    Returns:
        pipe : Pipeline object
            A pipeline that scales the input data using StandardScaler and applies the given model.
    """
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    return pipe

def flatten_list(lst):
    """
    Flatten a list that may contain nested lists.
    
    Args:
        lst (list): The list to flatten.
    
    Returns:
        list: The flattened list.
    """
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened

def is_polynomial_pipeline(estimator):
    if not isinstance(estimator, Pipeline):
        return False
    steps = estimator.steps
    if len(steps) != 2:
        return False
    poly_step = steps[0]
    if not isinstance(poly_step[1], PolynomialFeatures):
        return False
    return True

def check_list_type(lst):
    """
    Checks if a list only contains strings, sklearn model objects, and sklearn model selection objects.

    Args:
        lst (list): A list to check.

    Returns:
        bool: True if the list only contains valid objects, False otherwise.

    Raises:
        TypeError: If the list contains objects other than strings, sklearn model objects, or sklearn model selection objects.

    Examples:
        >>> check_list_type(['linear', RandomForestRegressor(), KFold()])
        True
        >>> check_list_type([1, 'linear'])
        TypeError: The list must contain only strings, sklearn model objects, and sklearn model selection objects.
    """
    if len(lst) == 0:
        raise ValueError("Estimator list is empty. Please add some models or use some of the defaults provided.")
    
    for element in lst:
        if not isinstance(element, (str, BaseEstimator, BaseCrossValidator)):
            raise TypeError("The list must contain only strings, sklearn model objects, and sklearn model selection objects.")
    return True

def select_continuous_estimator(estimator_type):
    """
    Returns a continuous estimator object for the specified estimator type.

    Args:
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.

    Returns:
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """
    if estimator_type == 'linear':
        return (ElasticNetCV())
    elif estimator_type == 'forest':
        return RandomForestRegressor()
    elif estimator_type == 'gbf':
        return GradientBoostingRegressor()
    elif estimator_type == 'nnet':
        return (MLPRegressor())
    elif estimator_type == 'poly':
        poly = sklearn.preprocessing.PolynomialFeatures()
        linear = sklearn.linear_model.ElasticNetCV(cv=3) #Play around with precompute and tolerance
        return (Pipeline([('poly', poly), ('linear', linear)]))
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")

def select_discrete_estimator(estimator_type):
    """
    Returns a discrete estimator object for the specified estimator type.

    Args:
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.

    Returns:
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """
    if estimator_type == 'linear':
        return (LogisticRegressionCV(multi_class='auto'))
    elif estimator_type == 'forest':
        return RandomForestClassifier()
    elif estimator_type == 'gbf':
        return GradientBoostingClassifier()
    elif estimator_type == 'nnet':
        return (MLPClassifier())
    elif estimator_type == 'poly':
        poly = PolynomialFeatures()
        linear = LogisticRegressionCV(multi_class='auto')
        return (Pipeline([('poly', poly), ('linear', linear)]))
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")

def select_estimator(estimator_type, target_type):
    """
    Returns an estimator object for the specified estimator and target types.

    Args:
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'automl', 'all'.
        target_type (str): The type of target variable, one of: 'continuous', 'discrete'.

    Returns:
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator or target types are unsupported.
    """
    if target_type not in ['continuous', 'discrete']:
        raise ValueError(f"Unsupported target type: {target_type}")
    if target_type == 'continuous':
        return select_continuous_estimator(estimator_type=estimator_type)
    elif target_type == 'discrete':
        return select_discrete_estimator(estimator_type=estimator_type)

def get_complete_estimator_list(estimator_list, target_type):
    '''
    Returns a list of sklearn objects from an input list of str's, and sklearn objects.

    Args:
        estimator_list : List of estimators; can be sklearn object or str: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'automl', 'all'.

    Returns:
        object: A list of sklearn objects

    Raises:
        ValueError: If the estimator is not supported.

    '''
    if isinstance(estimator_list, str):
        if estimator_list in ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']:
            estimator_list = [estimator_list]
        else: 
            raise ValueError("Invalid estimator_list value. Please provide a valid value from the list of available estimators: ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']")
    
    if isinstance(estimator_list, BaseEstimator):
        estimator_list = [estimator_list]

    if not isinstance(estimator_list, list):
        raise ValueError(f"estimator_list should be of type list not: {type(estimator_list)}")

    # Throws error if incompatible elements exist
    check_list_type(estimator_list)
    # populate list of estimator objects
    temp_est_list = []

    # if 'all' or 'automl' chosen then create list of all estimators to search over
    if 'automl' in estimator_list or 'all' in estimator_list:
        estimator_list = ['linear', 'forest', 'gbf', 'nnet', 'poly']

    # loop over every estimator
    for estimator in estimator_list:
        # if sklearn object: add to list, else turn str into corresponding sklearn object and add to list
        if isinstance(estimator, (BaseEstimator, BaseCrossValidator)):
            temp_est_list.append(estimator)
        else:
            temp_est_list.append(select_estimator(estimator, target_type))

    temp_est_list = flatten_list(temp_est_list)
    return temp_est_list

def select_classification_hyperparameters(estimator):
    """
    Returns a hyperparameter grid for the specified classification model type.
    
    Args:
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.
    
    Returns:
        A dictionary representing the hyperparameter grid to search over.
    """
    
    if isinstance(estimator, LogisticRegressionCV):
        # Hyperparameter grid for linear classification model
        return {
            'Cs': [0.01, 0.1, 1],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }
    elif isinstance(estimator, RandomForestClassifier):
        # Hyperparameter grid for random forest classification model
        return {
            'n_estimators': [100, 500, 1000],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif isinstance(estimator, GradientBoostingClassifier):
        # Hyperparameter grid for gradient boosting classification model
        return {
            'n_estimators': [100, 500, 1000],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            
        }
    elif isinstance(estimator, MLPClassifier):
        # Hyperparameter grid for neural network classification model
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif is_polynomial_pipeline(estimator=estimator):
        # Hyperparameter grid for polynomial kernel classification model
        return {
            'poly__degree': [2, 3, 4],
            'linear__Cs': [1, 10, 20],
            'linear__max_iter': [100, 200],
            'linear__penalty': ['l2'],
            'linear__solver': ['saga', 'liblinear', 'lbfgs']
        }
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for LogisticRegressionCV, RandomForestClassifier, MLPClassifier, and the polynomial pipleine", category=UserWarning)
        return {}
        # raise ValueError("Invalid model type. Valid values are 'linear', 'forest', 'nnet', and 'poly'.")
    


def select_regression_hyperparameters(estimator):
    """
    Returns a dictionary of hyperparameters to be searched over for a regression model.

    Args:
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns:
        A dictionary of hyperparameters to be searched over using a grid search.
    """
    if isinstance(estimator, ElasticNetCV):
        return {
            'l1_ratio': [0.1, 0.5, 0.9],
            'max_iter': [1000],
        }
    elif isinstance(estimator, RandomForestRegressor):
        return {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 50],
            'min_samples_split': [2, 5, 10],
        }
    elif isinstance(estimator, MLPRegressor):
        # Hyperparameter grid for neural network classification model
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif isinstance(estimator, GradientBoostingRegressor):
        # Hyperparameter grid for gradient boosting regression model
        return {
            'n_estimators': [100],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [3, 5],
        }
    elif is_polynomial_pipeline(estimator=estimator):
        # Hyperparameter grid for polynomial kernel classification model
        return {
            'linear__l1_ratio': [0.1, 0.5, 0.9],
            'linear__max_iter': [1000],
            'poly__degree': [2, 3, 4]
        }
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for ElasticNetCV, RandomForestRegressor, MLPRegressor, and the polynomial pipeline.", category=UserWarning)
        return {}
        # raise ValueError("Invalid model type. Valid values are 'linear', 'forest', 'nnet', and 'poly'.")


def is_linear_model(estimator):
    """
    Check whether an estimator is a polynomial regression, logistic regression, linear SVM, or any other type of
    linear model.

    Args:
    estimator (scikit-learn estimator): The estimator to check.

    Returns:
    is_linear (bool): True if the estimator is a linear model, False otherwise.
    """

    # Check if the estimator is a polynomial regression
    if isinstance(estimator, Pipeline):
        has_poly_feature_step = any(isinstance(step[1], PolynomialFeatures) for step in estimator.steps)
        if has_poly_feature_step:
            return True

    # Check if the estimator is a linear regression or related model
    if hasattr(estimator, 'fit_intercept') and hasattr(estimator, 'coef_'):
        return True

    # Check if the estimator is a logistic regression or linear SVM
    if isinstance(estimator, (LogisticRegression, LinearSVC, SVC)):
        return True

    # Otherwise, the estimator is not a linear model
    return False


def is_data_scaled(X):
    """
    Check if the input data is already centered and scaled using StandardScaler.

    Args:
        X array-like of shape (n_samples, n_features): The input data.

    Returns:
        is_scaled (bool): Whether the input data is already centered and scaled using StandardScaler or not.

    """
    # Compute the mean and standard deviation of the scaled data
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Check if the mean is close to 0 and the standard deviation is close to 1
    is_scaled = np.allclose(mean, 0.0) and np.allclose(std, 1.0)

    return is_scaled

def auto_hyperparameters(estimator_list, is_discrete=True):
    """
    Selects hyperparameters for a list of estimators.
    
    Args:
    - estimator_list: list of scikit-learn estimators
    - is_discrete: boolean indicating whether the problem is classification or regression
    
    Returns:
    - param_list: list of parameter grids for the estimators
    """
    param_list = []
    for estimator in estimator_list:
        if is_discrete:
            param_list.append(select_classification_hyperparameters(estimator=estimator))
        else:
            param_list.append(select_regression_hyperparameters(estimator=estimator))
    return param_list


def set_search_hyperparameters(search_object, hyperparameters):
    if isinstance(search_object, (RandomizedSearchCV, GridSearchCV)):
        search_object.set_params(**hyperparameters)
    else:
        raise ValueError("Invalid search object")
    
