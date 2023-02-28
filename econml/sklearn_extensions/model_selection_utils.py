
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

def select_estimator(estimator_type, target_type):
    """
    Returns an estimator object for the specified estimator and target types.

    Args:
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'automl', 'all'.
        target_type (str): The type of target variable, one of: 'continuous', 'discrete'.

    Returns:
        object: An instance of the selected estimator class, configured with appropriate hyperparameters.

    Raises:
        ValueError: If the estimator or target types are unsupported.
    """
    if target_type not in ['continuous', 'discrete']:
        raise ValueError(f"Unsupported target type: {target_type}")
    if estimator_type == 'linear':
        if target_type == 'continuous':
            return sklearn.linear_model.ElasticNetCV()
        elif target_type == 'discrete':
            return sklearn.linear_model.LogisticRegressionCV()
    elif estimator_type == 'forest':
        if target_type == 'continuous':
            return sklearn.ensemble.RandomForestRegressor()
        elif target_type == 'discrete':
            return sklearn.ensemble.RandomForestClassifier()
    elif estimator_type == 'gbf':
        if target_type == 'continuous':
            return sklearn.ensemble.GradientBoostingRegressor()
        elif target_type == 'discrete':
            return sklearn.ensemble.GradientBoostingClassifier()
    elif estimator_type == 'nnet':
        if target_type == 'continuous':
            return sklearn.neural_network.MLPRegressor()
        elif target_type == 'discrete':
            return sklearn.neural_network.MLPClassifier()
    elif estimator_type == 'poly':
        degrees = [2, 3, 4]  # I don't think we want to take the voting of each but I'm not sure if we should return a list of models?
        models = []
        for degree in degrees:
            poly = sklearn.preprocessing.PolynomialFeatures(degree=degree)
            if target_type == 'continuous':
                linear = sklearn.linear_model.ElasticNetCV(precompute=True, cv=3, tol=0.1, verbose=1)
            elif target_type == 'discrete':
                linear = sklearn.linear_model.LogisticRegressionCV()
            else:
                raise ValueError(f"Unsupported target type: {target_type}")
            models.append((f"poly{degree}", sklearn.pipeline.Pipeline([('poly', poly), ('linear', linear)])))
        if target_type == 'continuous':
            return sklearn.ensemble.VotingRegressor(estimators=models)
        elif target_type == 'discrete':
            return sklearn.ensemble.VotingClassifier(estimators=models)
    elif estimator_type == 'automl':
        return    
    elif estimator_type == 'all':
        if target_type == 'continuous':
            return sklearn.ensemble.VotingRegressor(estimators=[
                ('linear', select_estimator('linear', target_type)),
                ('forest', select_estimator('forest', target_type)),
                ('gbf', select_estimator('gbf', target_type)),
                ('nnet', select_estimator('nnet', target_type)),
                ('poly', select_estimator('poly', target_type)),
            ])
        elif target_type == 'discrete':
            return sklearn.ensemble.VotingClassifier(estimators=[
                ('linear', select_estimator('linear', target_type)),
                ('forest', select_estimator('forest', target_type)),
                ('gbf', select_estimator('gbf', target_type)),
                ('nnet', select_estimator('nnet', target_type)),
                ('poly', select_estimator('poly', target_type)),
            ], voting='soft')
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}") 

