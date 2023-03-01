
import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


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
        return sklearn.linear_model.ElasticNetCV()
    elif estimator_type == 'forest':
        return sklearn.ensemble.RandomForestRegressor()
    elif estimator_type == 'gbf':
        return sklearn.ensemble.GradientBoostingRegressor()
    elif estimator_type == 'nnet':
        return sklearn.neural_network.MLPRegressor()
    elif estimator_type == 'poly':
        degrees = [2, 3, 4]
        models = []
        for degree in degrees:
            poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
            linear = sklearn.linear_model.ElasticNetCV(precompute=True, cv=3, tol=0.1, verbose=1)
            models.append((f"poly{degree}", sklearn.pipeline.Pipeline([('poly', poly), ('linear', linear)])))
        return sklearn.ensemble.VotingRegressor(estimators=models)
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
        return sklearn.linear_model.LogisticRegressionCV()
    elif estimator_type == 'forest':
        return sklearn.ensemble.RandomForestClassifier()
    elif estimator_type == 'gbf':
        return sklearn.ensemble.GradientBoostingClassifier()
    elif estimator_type == 'nnet':
        return sklearn.neural_network.MLPClassifier()
    elif estimator_type == 'poly':
        degrees = [2, 3, 4]
        models = []
        for degree in degrees:
            poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
            linear = sklearn.linear_model.LogisticRegressionCV()
            models.append((f"poly{degree}", sklearn.pipeline.Pipeline([('poly', poly), ('linear', linear)])))
        return sklearn.ensemble.VotingClassifier(estimators=models)
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

def get_estimator_type(estimator):
    """
    Returns the type of estimator. either 'discrete' or 'continuous'

    Args:

    """
    return 'continuous'

def get_complete_estimator_list(estimator_list):
    '''
    Returns a list of sklearn objects from an input list of str's, and sklearn objects.

    Args:
        estimator_list : List of estimators; can be sklearn object or str: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'automl', 'all'.

    Returns:
        object: A list of sklearn objects

    Raises:
        ValueError: If the estimator is not supported.

    '''
    # Throws error if incompatible elements exist
    check_list_type(estimator_list)
    # populate list of estimator objects
    temp_est_list = []
    for estimator in estimator_list:
        # if sklearn object: add to list, else turn str into corresponding sklearn object and add to list
        if isinstance(estimator, (BaseEstimator, BaseCrossValidator)):
            temp_est_list.append(estimator)
        else:
            temp_est_list.append(select_estimator(estimator, get_estimator_type(estimator)))
    return temp_est_list
