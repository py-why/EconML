from bayes_opt import BayesianOptimization
import numbers
import warnings
import sklearn
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import type_of_target
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.base import clone, is_classifier
from sklearn.model_selection import KFold, StratifiedKFold, check_cv, GridSearchCV, BaseCrossValidator, RandomizedSearchCV
# TODO: conisder working around relying on sklearn implementation details
from sklearn.model_selection._validation import (_check_is_permutation,
                                                 _fit_and_predict)
from sklearn.exceptions import FitFailedWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import indexable, check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier



class BayesianOptimizationSearchListCV():
    # ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']    
    def __init__(self, estimator_list = ['linear', 'forest'], is_discrete = False):
        self.estimator_list = self.get_estimators(estimator_list, is_discrete)
        self.is_discrete = is_discrete
    def fit(self, X, y=None):
        pass
    
    def get_estimators(self, estimator_list, is_discrete):
        
        """
        Parameters: 
        ----------
        estimator_list: list of estimators. subset of ['linear', 'forest', 'gbf', 'nnet', 'poly', 'automl']  

        is_discrete: boolean for discrete or continuous output (regression or classification)

        Returns:
        --------
        estimators: corresponding methods that perform bayesian optimization over estimators hyperparameters


        Example:

        (['forest'], False) -> [random_forest_reg]

        """
        if not isinstance(estimator_list, list):
            raise ValueError(f"estimator_list should be of type list not: {type(estimator_list)}")

        # Throws error if incompatible elements exist
        check_list_type(estimator_list)
        # populate list of estimator objects
        temp_est_list = []

        # if 'all' or 'automl' chosen then create list of all estimators to search over
        if 'automl' in estimator_list or 'all' in estimator_list:
            estimator_list = ['linear', 'forest', 'gbf', 'nnet', 'poly']

        estimators = []
        for estimator in estimators:
            if estimator == 'linear':
                pass
                #estimators.append(logistic_reg if is_discrete else linear_reg)

    def random_forest_reg_score(self, max_depth, min_samples_split, min_weight_fraction_leaf, max_features, cv=5, scoring='neg_mean_squared_error'):
        """
        Parameters: 
        ----------
        max_depth: max depth of forest (hyperparameter for random forest)

        min_samples_split: The minimum number of samples required to split an internal node: (hyperparameter for random forest)

            If int, then consider min_samples_split as the minimum number.

            If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

        min_weight_fraction_leaf:

        max_features:

        cv: number of folds to perform cross-validation. Default: 5

        scoring: Scoring method. Default: 'neg_mean_squared_error'

        is_discrete: boolean for discrete or continuous output (regression or classification)

        Returns:
        --------
        estimators: corresponding methods that perform bayesian optimization over estimators hyperparameters


        Example:

        (['forest'], False) -> [random_forest_reg]

        """

        # parameterds to fit the model with
        params = {}

        # rounding may be necessary if the params need to be ints
        params['max_depth'] = round(max_depth)
        params['min_samples_split'] = round(min_samples_split)
        params['min_weight_fraction_leaf'] = min_weight_fraction_leaf
        #params['max_features'] = int(max_features)

        # scores the model with the above hyperparams
        scores = cross_val_score(RandomForestRegressor(random_state=123, **params),
                                X, y, scoring=scoring, cv=cv).mean()
        score = scores.mean()
        return score

    def random_forest_reg(self, random_state=123):
        # optimizes the hyperparameters for random forest regression

        # create dictionary for ranges to consider
        params ={
            'max_depth':(3, 10),
            'min_samples_split':(2,5),
            'min_weight_fraction_leaf':(0.0,0.2),
            'max_features':(1, 7)
        }

        # takes in a method to score the hyperparameters (implements above), the params with respective ranges
        rand_for = BayesianOptimization(self.random_forest_reg_score, params, random_state=random_state)

        # Optimize.
        """
        n_iter: How many steps of bayesian optimization you want to perform. 
                The more steps the more likely to find a good maximum you are.
        
        init_points: How many steps of random exploration you want to perform. 
                    Random exploration can help by diversifying the exploration space.
        """
        rand_for.maximize(init_points=10, n_iter=1)

        # return the best parameters found
        return rand_for.max['params']
    
    def random_forest_cls(self, random_state=123):
        pass

    # complete above two functions for each regression and classification task for: 

    # ['linear', 'forest', 'gbf', 'nnet', 'poly']

    # Linear regression optimizer and score.

    # Logistic regression optimizer and socre.

    # Gradient Boost regression optimizer and score.

    # Gradient Boost classification optimizer and score.

    # Neural Network regression optimizer and score.

    # Neural Network classifier optimizer and score.

    # Polynomial regression optimizer and score.

    # Polynomial classfication optimizer and score.