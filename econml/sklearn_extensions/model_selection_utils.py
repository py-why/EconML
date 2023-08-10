
import pdb
import warnings
from sklearn.exceptions import NotFittedError
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.preprocessing
from sklearn.base import BaseEstimator, is_regressor, is_classifier
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import (ElasticNetCV,
                                  LogisticRegression,
                                  LogisticRegressionCV, MultiTaskElasticNetCV)
from sklearn.model_selection import (BaseCrossValidator, GridSearchCV,
                                     RandomizedSearchCV,
                                     check_cv)
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (PolynomialFeatures,
                                   StandardScaler)
from sklearn.svm import SVC, LinearSVC
import inspect
from sklearn.exceptions import NotFittedError
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import KFold
import pandas as pd


def select_continuous_estimator(estimator_type, random_state):
    """
    Returns a continuous estimator object for the specified estimator type.

    Parameters
    ----------
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.
        TODO Add Random State for parameter
    Returns
    ----------
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """
    if estimator_type == 'linear':
        return (ElasticNetCV(random_state=random_state))
    elif estimator_type == 'forest':
        return RandomForestRegressor(random_state=random_state)
    elif estimator_type == 'gbf':
        return GradientBoostingRegressor(random_state=random_state)
    elif estimator_type == 'nnet':
        return (MLPRegressor(random_state=random_state))
    elif estimator_type == 'poly':
        poly = PolynomialFeatures()
        linear = ElasticNetCV(random_state=random_state)  # Play around with precompute and tolerance
        return (Pipeline([('poly', poly), ('linear', linear)]))
    elif estimator_type == 'weighted_lasso':
        from econml.sklearn_extensions.linear_model import WeightedLassoCVWrapper
        return WeightedLassoCVWrapper(random_state=random_state)
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")


def select_discrete_estimator(estimator_type, random_state):
    """
    Returns a discrete estimator object for the specified estimator type.

    Parameters
    ----------
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly'.
        TODO Add Random State for parameter
    Returns
    ----------
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator type is unsupported.
    """

    if estimator_type == 'linear':
        return (LogisticRegressionCV(cv=KFold(random_state=random_state),
                                     multi_class='auto', random_state=random_state))
    elif estimator_type == 'forest':
        return RandomForestClassifier(random_state=random_state)
    elif estimator_type == 'gbf':
        return GradientBoostingClassifier(random_state=random_state)
    elif estimator_type == 'nnet':
        return (MLPClassifier(random_state=random_state))
    elif estimator_type == 'poly':
        poly = PolynomialFeatures()
        linear = (LogisticRegressionCV(cv=KFold(random_state=random_state),
                                       multi_class='auto', random_state=random_state))
        return (Pipeline([('poly', poly), ('linear', linear)]))
    else:
        raise ValueError(f"Unsupported estimator type: {estimator_type}")


def select_estimator(estimator_type, is_discrete, random_state):
    """
    Returns an estimator object for the specified estimator and target types.

    Parameters
    ----------
        estimator_type (str): The type of estimator to use, one of: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'automl', 'all'.
        is_discrete (bool): The type of target variable, if true then it's discrete.
        TODO Add Random State for parameter
    Returns
    ----------
        object: An instance of the selected estimator class.

    Raises:
        ValueError: If the estimator or target types are unsupported.
    """
    if not isinstance(is_discrete, bool):
        raise ValueError(f"Unsupported target type: {type(is_discrete)}. is_discrete should be of type bool.")
    elif is_discrete:
        return select_discrete_estimator(estimator_type=estimator_type, random_state=random_state)
    else:
        return select_continuous_estimator(estimator_type=estimator_type, random_state=random_state)


def is_likely_estimator(estimator):
    """
    Check if an object is likely to be an estimator.

    This function checks if an object has 'fit' and 'predict' methods, or if it is an instance of BaseEstimator.

    Parameters
    ----------
    estimator : object
        The object to check.

    Returns
    -------
    bool
        True if the object is likely to be an estimator, False otherwise.
    """

    required_methods = ['fit', 'predict']
    return all(hasattr(estimator, method) for method in required_methods) or isinstance(estimator, BaseEstimator)


def check_list_type(lst):
    """
    Checks if a list only contains strings, sklearn model objects, and sklearn model selection objects.

    Parameters
    ----------
        lst (list): A list to check.

    Returns
    ----------
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

    # pdb.set_trace()
    for element in lst:
        if (not isinstance(element, (str, BaseCrossValidator))):
            if not is_likely_estimator(element):
                # pdb.set_trace()
                raise TypeError(
                    f"The list must contain only strings, sklearn model objects, and sklearn model selection objects. Invalid element: {element}")
    return True


def get_complete_estimator_list(estimator_list, is_discrete, random_state):
    '''
    Returns a list of sklearn objects from an input list of str's, and sklearn objects.

    Parameters
    ----------
        estimator_list : List of estimators; can be sklearn object or str: 'linear', 'forest', 'gbf', 'nnet', 'poly', 'auto', 'all'.
        is_discrete (bool): if target type is discrete or continuous.

    Returns
    ----------
        object: A list of sklearn objects

    Raises:
        ValueError: If the estimator is not supported.

    '''
    # pdb.set_trace()
    if isinstance(estimator_list, str):
        if 'all' == estimator_list:
            estimator_list = ['linear', 'forest', 'gbf', 'nnet', 'poly']
        elif 'auto' == estimator_list:
            estimator_list = ['linear']
        elif estimator_list in ['linear', 'forest', 'gbf', 'nnet', 'poly']:
            estimator_list = [estimator_list]
        else:
            raise ValueError(
                "Invalid estimator_list value. Please provide a valid value from the list of available estimators: ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']")
    elif isinstance(estimator_list, list):
        if 'auto' in estimator_list:
            for estimator in ['linear']:
                if estimator not in estimator_list:
                    estimator_list.append(estimator)
        if 'all' in estimator_list:
            for estimator in ['linear', 'forest', 'gbf', 'nnet', 'poly']:
                if estimator not in estimator_list:
                    estimator_list.append(estimator)

    elif is_likely_estimator(estimator_list):
        estimator_list = [estimator_list]
    else:
        raise ValueError(f"Incorrect type: {type(estimator_list)}")
    check_list_type(estimator_list)
    temp_est_list = []

    if not isinstance(estimator_list, list):
        raise ValueError(f"estimator_list should be of type list not: {type(estimator_list)}")

    # Set to remove duplicates
    for estimator in set(estimator_list):
        # if sklearn object: add to list, else turn str into corresponding sklearn object and add to list
        if isinstance(estimator, BaseCrossValidator) or is_likely_estimator(estimator):
            temp_est_list.append(estimator)
        else:
            temp_est_list.append(select_estimator(estimator_type=estimator,
                                 is_discrete=is_discrete, random_state=random_state))
    temp_est_list = flatten_list(temp_est_list)

    # Check that all types of models are matched towards the problem.
    # pdb.set_trace()
    for estimator in temp_est_list:
        if (isinstance(estimator, BaseEstimator)):
            if not is_regressor_or_classifier(estimator, is_discrete=is_discrete):
                raise TypeError("Invalid estimator type: {} - must be a regressor or classifier".format(type(estimator)))
    return temp_est_list


def select_classification_hyperparameters(estimator):
    """
    Returns a hyperparameter grid for the specified classification model type.

    Parameters
    ----------
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns
    ----------
        A dictionary representing the hyperparameter grid to search over.
    """

    if isinstance(estimator, LogisticRegressionCV):
        return {
            'Cs': [0.01, 0.1, 1],
            'cv': [3],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['lbfgs', 'liblinear', 'saga']
        }
    elif isinstance(estimator, RandomForestClassifier):
        return {
            'n_estimators': [100, 500],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    elif isinstance(estimator, GradientBoostingClassifier):
        return {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],

        }
    elif isinstance(estimator, MLPClassifier):
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif is_polynomial_pipeline(estimator=estimator):
        return {
            'poly__degree': [2, 3, 4],
            'linear__max_iter': [100, 200],
            'linear__penalty': ['l2'],
            'linear__solver': ['saga', 'lbfgs']
        }
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for LogisticRegressionCV, RandomForestClassifier, MLPClassifier, and the polynomial pipleine", category=UserWarning)
        return {}
        # raise ValueError("Invalid model type. Valid values are 'linear', 'forest', 'nnet', and 'poly'.")


def select_regression_hyperparameters(estimator):
    """
    Returns a dictionary of hyperparameters to be searched over for a regression model.

    Parameters
    ----------
        model_type (str): The type of model to be used. Valid values are 'linear', 'forest', 'nnet', and 'poly'.

    Returns
    ----------
        A dictionary of hyperparameters to be searched over using a grid search.
    """
    if isinstance(estimator, ElasticNetCV):
        return {
            'l1_ratio': [0.1, 0.5, 0.9],
            'cv': [3],
            'max_iter': [1000],
        }
    elif isinstance(estimator, RandomForestRegressor):
        return {
            'n_estimators': [100],
            'max_depth': [None, 10, 50],
            'min_samples_split': [2, 5, 10],
        }
    elif isinstance(estimator, MLPRegressor):
        return {
            'hidden_layer_sizes': [(10,), (50,), (100,)],
            'alpha': [0.0001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    elif isinstance(estimator, GradientBoostingRegressor):
        return {
            'n_estimators': [100, 500],
            'learning_rate': [0.01, 0.1, 0.05],
            'max_depth': [3, 5],
        }
    elif is_polynomial_pipeline(estimator=estimator):
        return {
            'linear__l1_ratio': [0.1, 0.5, 0.9],
            'linear__max_iter': [1000],
            'poly__degree': [2, 3, 4]
        }
    else:
        warnings.warn("No hyperparameters for this type of model. There are default hyperparameters for ElasticNetCV, RandomForestRegressor, MLPRegressor, and the polynomial pipeline.", category=UserWarning)
        return {}


def flatten_list(lst):
    """
    Flatten a list that may contain nested lists.

    Parameters
    ----------
        lst (list): The list to flatten.

    Returns
    ----------
        list: The flattened list.
    """
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened


def auto_hyperparameters(estimator_list, is_discrete=True):
    """
    Selects hyperparameters for a list of estimators.

    Parameters
    ----------
    - estimator_list: list of scikit-learn estimators
    - is_discrete: boolean indicating whether the problem is classification or regression

    Returns
    ----------
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


def is_mlp(estimator):
    return isinstance(estimator, (MLPClassifier, MLPRegressor))


def has_random_state(model):
    """
    Check if a model has a 'random_state' parameter.

    This function inspects the model's signature to check if it has a 'random_state' parameter.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if the model has a 'random_state' parameter, False otherwise.
    """

    if is_polynomial_pipeline(model):
        signature = inspect.signature(type(model['linear']))
    else:
        signature = inspect.signature(type(model))
    return ("random_state" in signature.parameters)


def supports_sample_weight(estimator):
    """
    Check if a model supports 'sample_weight'.

    This function inspects the signature of the model's 'fit' method to check if it supports 'sample_weight'.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if the model supports 'sample_weight', False otherwise.
    """

    fit_signature = inspect.signature(estimator.fit)
    return 'sample_weight' in fit_signature.parameters


def just_one_model_no_params(estimator_list, param_list):
    """
    Check if there is only one model and the parameter list is empty.

    This function checks if the length of the model and parameter list is 1 and 0 respectively.

    Parameters
    ----------
    estimator_list : list
        List of models.

    param_list : list
        List of parameters.

    Returns
    -------
    bool
        True if there is only one model and the parameter list is empty, False otherwise.
    """

    return (len(estimator_list) == 1) and (len(param_list) == 1) and (len(param_list[0]) == 0)


def param_grid_is_empty(param_grid):
    """
    Check if a parameter grid is empty.

    This function checks if the length of the parameter grid is 0.

    Parameters
    ----------
    param_grid : dict
        Parameter grid to check.

    Returns
    -------
    bool
        True if the parameter grid is empty, False otherwise.
    """

    return len(param_grid) == 0


def is_linear_model(estimator):
    """
    Check if a model is a linear model.

    This function checks if a model has 'fit_intercept' and 'coef_' attributes or if it is an instance of LogisticRegression, LinearSVC, or SVC.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if the model is a linear model, False otherwise.
    """

    if isinstance(estimator, Pipeline):
        has_poly_feature_step = any(isinstance(step[1], PolynomialFeatures) for step in estimator.steps)
        if has_poly_feature_step:
            return True

    if hasattr(estimator, 'fit_intercept') and hasattr(estimator, 'coef_'):
        return True

    if isinstance(estimator, (LogisticRegression, LinearSVC, SVC)):
        return True

    return False


def is_data_scaled(X):
    """
    Check if input data is scaled.

    This function checks if the input data is scaled by comparing its mean and standard deviation to 0 and 1 respectively.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.

    Returns
    -------
    bool
        True if the input data is scaled, False otherwise.

    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    is_scaled = np.allclose(mean, 0.0) and np.allclose(std, 1.0)

    return is_scaled


def is_regressor_or_classifier(model, is_discrete):
    """
    Check if a model is a regressor or classifier.

    This function checks if a model is a regressor or classifier depending on the 'is_discrete' parameter.

    Parameters
    ----------
    model : object
        The model to check.

    is_discrete : bool
        If True, checks if the model is a classifier. If False, checks if the model is a regressor.

    Returns
    -------
    bool
        True if the model matches the type specified by 'is_discrete', False otherwise.
    """

    if is_discrete:
        if is_polynomial_pipeline(model):
            return is_classifier(model[1])
        else:
            return is_classifier(model)
    else:
        if is_polynomial_pipeline(model):
            return is_regressor(model[1])
        else:
            return is_regressor(model)


def scale_pipeline(model):
    """
    Returns a pipeline that scales the input data using StandardScaler and applies the given model.

    Parameters
    ----------
        model : estimator object
            A model object that implements the scikit-learn estimator interface.

    Returns
    ----------
        pipe : Pipeline object
            A pipeline that scales the input data using StandardScaler and applies the given model.
    """
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    return pipe


def is_polynomial_pipeline(estimator):
    """
    Check if a model is a polynomial pipeline.

    This function checks if a model is a pipeline that includes a PolynomialFeatures step.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if the model is a polynomial pipeline, False otherwise.
    """

    if not isinstance(estimator, Pipeline):
        return False
    steps = estimator.steps
    if len(steps) != 2:
        return False
    poly_step = steps[0]
    if not isinstance(poly_step[1], PolynomialFeatures):
        return False
    return True


def is_likely_multi_task(y):
    """
    Check if a target array is likely multi-task.

    This function checks if a target array is likely to be multi-task by checking its shape.

    Parameters
    ----------
    y : array-like
        The target array to check.

    Returns
    -------
    bool
        True if the target array is likely multi-task, False otherwise.
    """

    if len(y.shape) == 2:
        if y.shape[1] > 1:
            return True
    return False


def can_handle_multitask(model, is_discrete=False):
    """
    Check if a model can handle multi-task output.

    This function checks if a model can handle multi-task output by trying to fit and predict on random data.

    Parameters
    ----------
    model : object
        The model to check.

    Returns
    -------
    bool
        True if the model can handle multi-task output, False otherwise.
    """

    X = np.random.rand(10, 3)
    if is_discrete:
        y = np.random.randint(0, 2, (10, 2))
    else:
        y = np.random.rand(10, 2)

    try:
        model.fit(X, y)
    except Exception as e:
        return False

    try:
        model.predict(X)
    except Exception as e:
        # warnings.warn(f"The model {model.__class__.__name__} is not properly fitted. Error: {e}")
        return False
    return True


def pipeline_convert_to_multitask(pipeline):
    """
    Convert a pipeline to handle multi-task output if possible.

    This function iterates over the steps in the input pipeline. If a step is a
    polynomial transformer, it adds the step to the new pipeline as is. If the
    step is an estimator, it attempts to convert it to handle multi-task output
    and adds the converted estimator to the new pipeline.

    Parameters
    ----------
    pipeline : sklearn.Pipeline
        The pipeline to convert.

    Returns
    -------
    sklearn.Pipeline
        The converted pipeline.

    Raises
    ------
    ValueError
        If an unknown error occurs when making model multi-task.
    """

    steps = list(pipeline.steps)
    if isinstance(steps[-1][1], (LogisticRegressionCV)):
        steps[-1] = ('linear', MultiOutputClassifier(steps[-1][1]))
    if isinstance(steps[-1][1], (ElasticNetCV)):
        steps[-1] = ('linear', MultiTaskElasticNetCV())
    new_pipeline = Pipeline(steps)

    return new_pipeline


def make_model_multi_task(model, is_discrete):
    """
    Convert a model to handle multi-task output if possible.

    This function converts a model to handle multi-task output if possible.

    Parameters
    ----------
    model : object
        The model to convert.

    is_discrete : bool
        If True, the model is treated as a classifier. If False, the model is treated as a regressor.

    Returns
    -------
    object
        The converted model if possible, raises an error otherwise.
    """

    try:
        if is_discrete:
            if is_polynomial_pipeline(model):
                return pipeline_convert_to_multitask(model)
            return MultiOutputClassifier(model)
        else:
            if isinstance(model, ElasticNetCV):
                return MultiTaskElasticNetCV()
            elif is_polynomial_pipeline(model):
                return pipeline_convert_to_multitask(model)
            else:
                return MultiOutputRegressor(model)
    except Exception as e:
        raise ValueError("An unknown error occurred when making model multitask.") from e


def make_param_multi_task(estimator, param_grid):
    """
    Convert the keys in a parameter grid to work with a multi-task model.

    This function converts the keys in a parameter grid to work with a multi-task model by prepending 'estimator__' to each key.

    Parameters
    ----------
    estimator : object
        The estimator the parameter grid is for.

    param_grid : dict
        The parameter grid to convert.

    Returns
    -------
    dict
        The converted parameter grid.
    """

    if isinstance(estimator, ElasticNetCV):
        return param_grid
    else:
        param_grid_multi = {f'estimator__{k}': v for k, v in param_grid.items()}
        return param_grid_multi


def preprocess_and_encode(data, cat_indices=None):
    """
    Detects categorical columns, one-hot encodes them, and returns the preprocessed data.

    Parameters:
    - data: pandas DataFrame or numpy array
    - cat_indices: list of column indices (or names for DataFrame) to be considered categorical

    Returns:
    - Preprocessed data in the format of the original input (DataFrame or numpy array)
    """
    was_numpy = False
    if isinstance(data, np.ndarray):
        was_numpy = True
        data = pd.DataFrame(data)

    # If cat_indices is None, detect categorical columns using object type as a heuristic
    if cat_indices is None:
        cat_columns = data.select_dtypes(['object']).columns.tolist()
    else:
        if all(isinstance(i, int) for i in cat_indices):  # if cat_indices are integer indices
            cat_columns = data.columns[cat_indices].tolist()
        else:  # assume cat_indices are column names
            cat_columns = cat_indices

    data_encoded = pd.get_dummies(data, columns=cat_columns)

    if was_numpy:
        return data_encoded.values
    else:
        return data_encoded
