# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Module for assessing causal feature importance."""

import warnings
from collections import OrderedDict, namedtuple

import joblib
import lightgbm as lgb
from numba.core.utils import erase_traceback
import numpy as np
from numpy.lib.function_base import iterable
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.tree import _tree
from sklearn.utils.validation import column_or_1d
from ...cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter
from ...dml import LinearDML, CausalForestDML
from ...inference import NormalInferenceResults
from ...sklearn_extensions.linear_model import WeightedLasso
from ...sklearn_extensions.model_selection import GridSearchCVList
from ...utilities import _RegressionWrapper, get_feature_names_or_default, inverse_onehot, one_hot_encoder

# TODO: this utility is documented but internal; reimplement?
from sklearn.utils import _safe_indexing
# TODO: this utility is even less public...
from packaging.version import parse
import sklearn
if parse(sklearn.__version__) < parse("1.5"):
    from sklearn.utils import _get_column_indices
else:
    from sklearn.utils._indexing import _get_column_indices


class _CausalInsightsConstants:
    RawFeatureNameKey = 'raw_name'
    EngineeredNameKey = 'name'
    CategoricalColumnKey = 'cat'
    TypeKey = 'type'
    PointEstimateKey = 'point'
    StandardErrorKey = 'stderr'
    ZStatKey = 'zstat'
    ConfidenceIntervalLowerKey = 'ci_lower'
    ConfidenceIntervalUpperKey = 'ci_upper'
    PValueKey = 'p_value'
    Version = 'version'
    CausalComputationTypeKey = 'causal_computation_type'
    ConfoundingIntervalKey = 'confounding_interval'
    ViewKey = 'view'
    InitArgsKey = 'init_args'
    RowData = 'row_data'  # NOTE: RowData is mutually exclusive with the other data columns

    ALL = [RawFeatureNameKey,
           EngineeredNameKey,
           CategoricalColumnKey,
           TypeKey,
           PointEstimateKey,
           StandardErrorKey,
           ZStatKey,
           ConfidenceIntervalLowerKey,
           ConfidenceIntervalUpperKey,
           PValueKey,
           Version,
           CausalComputationTypeKey,
           ConfoundingIntervalKey,
           ViewKey,
           InitArgsKey,
           RowData]


def _get_default_shared_insights_output():
    """
    Dictionary elements shared among all analyses.

    In case of breaking changes to this dictionary output, the major version of this
    dictionary should be updated. In case of a change to this dictionary, the minor
    version should be updated.
    """
    return {
        _CausalInsightsConstants.RawFeatureNameKey: [],
        _CausalInsightsConstants.EngineeredNameKey: [],
        _CausalInsightsConstants.CategoricalColumnKey: [],
        _CausalInsightsConstants.TypeKey: [],
        _CausalInsightsConstants.Version: '1.0',
        _CausalInsightsConstants.CausalComputationTypeKey: "simple",
        _CausalInsightsConstants.ConfoundingIntervalKey: None,
        _CausalInsightsConstants.InitArgsKey: {}
    }


def _get_default_specific_insights(view):
    # keys should be mutually exclusive with shared keys, so that the dictionaries can be cleanly merged
    return {
        _CausalInsightsConstants.PointEstimateKey: [],
        _CausalInsightsConstants.StandardErrorKey: [],
        _CausalInsightsConstants.ZStatKey: [],
        _CausalInsightsConstants.ConfidenceIntervalLowerKey: [],
        _CausalInsightsConstants.ConfidenceIntervalUpperKey: [],
        _CausalInsightsConstants.PValueKey: [],
        _CausalInsightsConstants.ViewKey: view
    }


def _get_metadata_causal_insights_keys():
    return [_CausalInsightsConstants.Version,
            _CausalInsightsConstants.CausalComputationTypeKey,
            _CausalInsightsConstants.ConfoundingIntervalKey,
            _CausalInsightsConstants.ViewKey]


def _get_column_causal_insights_keys():
    return [_CausalInsightsConstants.RawFeatureNameKey,
            _CausalInsightsConstants.EngineeredNameKey,
            _CausalInsightsConstants.CategoricalColumnKey,
            _CausalInsightsConstants.TypeKey]


def _get_data_causal_insights_keys():
    return [_CausalInsightsConstants.PointEstimateKey,
            _CausalInsightsConstants.StandardErrorKey,
            _CausalInsightsConstants.ZStatKey,
            _CausalInsightsConstants.ConfidenceIntervalLowerKey,
            _CausalInsightsConstants.ConfidenceIntervalUpperKey,
            _CausalInsightsConstants.PValueKey]


def _first_stage_reg(X, y, *, automl=True, random_state=None, verbose=0):
    if automl:
        model = GridSearchCVList([LassoCV(random_state=random_state),
                                  RandomForestRegressor(
                                      n_estimators=100, random_state=random_state, min_samples_leaf=10),
                                  lgb.LGBMRegressor(num_leaves=32, random_state=random_state)],
                                 param_grid_list=[{},
                                                  {'min_weight_fraction_leaf':
                                                      [.001, .01, .1]},
                                                  {'learning_rate': [0.1, 0.3], 'max_depth': [3, 5]}],
                                 cv=3,
                                 scoring='r2',
                                 verbose=verbose)
        best_est = model.fit(X, y).best_estimator_
        if isinstance(best_est, LassoCV):
            return Lasso(alpha=best_est.alpha_, random_state=random_state)
        else:
            return best_est
    else:
        model = LassoCV(cv=5, random_state=random_state).fit(X, y)
        return Lasso(alpha=model.alpha_, random_state=random_state)


def _first_stage_clf(X, y, *, make_regressor=False, automl=True, min_count=None, random_state=None, verbose=0):
    # use same Cs as would be used by default by LogisticRegressionCV
    cs = np.logspace(-4, 4, 10)
    if min_count is None:
        min_count = _CAT_LIMIT  # we have at least this many instances
    if automl:
        # NOTE: we don't use LogisticRegressionCV inside the grid search because of the nested stratification
        #       which could affect how many times each distinct Y value needs to be present in the data

        model = GridSearchCVList([LogisticRegression(max_iter=1000,
                                                     random_state=random_state),
                                  RandomForestClassifier(n_estimators=100, min_samples_leaf=10,
                                                         random_state=random_state),
                                  lgb.LGBMClassifier(num_leaves=32, random_state=random_state)],
                                 param_grid_list=[{'C': cs},
                                                  {'max_depth': [3, None],
                                                   'min_weight_fraction_leaf': [.001, .01, .1]},
                                                  {'learning_rate': [0.1, 0.3], 'max_depth': [3, 5]}],
                                 cv=min(3, min_count),
                                 scoring='neg_log_loss',
                                 verbose=verbose)
        est = model.fit(X, y).best_estimator_
    else:
        model = LogisticRegressionCV(
            cv=min(5, min_count), max_iter=1000, Cs=cs, random_state=random_state).fit(X, y)
        est = LogisticRegression(C=model.C_[0], max_iter=1000, random_state=random_state)
    if make_regressor:
        return _RegressionWrapper(est)
    else:
        return est


def _final_stage(*, random_state=None, verbose=0):
    return GridSearchCVList([WeightedLasso(random_state=random_state),
                             RandomForestRegressor(n_estimators=100, random_state=random_state, verbose=verbose)],
                            param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                             {'max_depth': [3, 5],
                                              'min_samples_leaf': [10, 50]}],
                            cv=3,
                            scoring='neg_mean_squared_error',
                            verbose=verbose)


# simplification of sklearn's ColumnTransformer that encodes categoricals and passes through selected other columns
# but also supports get_feature_names with expected signature
class _ColumnTransformer(TransformerMixin):
    def __init__(self, categorical, passthrough):
        self.categorical = categorical
        self.passthrough = passthrough

    def fit(self, X):
        cat_cols = _safe_indexing(X, self.categorical, axis=1)
        if cat_cols.shape[1] > 0:
            self.has_cats = True
            # NOTE: set handle_unknown to 'ignore' so that we don't throw at runtime if given a novel value
            self.one_hot_encoder = one_hot_encoder(handle_unknown='ignore').fit(cat_cols)
        else:
            self.has_cats = False
        self.d_x = X.shape[1]
        return self

    def transform(self, X):
        rest = _safe_indexing(X, self.passthrough, axis=1)
        if self.has_cats:
            cats = self.one_hot_encoder.transform(_safe_indexing(X, self.categorical, axis=1))
            # NOTE: we rely on the passthrough columns coming first in the concatenated X;W
            #       when we pipeline scaling with our first stage models later, so the order here is important
            return np.hstack((rest, cats))
        else:
            return rest

    # TODO: remove once older sklearn support is no longer needed
    def get_feature_names(self, names=None):
        return self.get_feature_names_out(names)

    def get_feature_names_out(self, names=None):
        if names is None:
            names = [f"x{i}" for i in range(self.d_x)]
        rest = _safe_indexing(names, self.passthrough, axis=0)
        if self.has_cats:
            cats = get_feature_names_or_default(self.one_hot_encoder, _safe_indexing(names, self.categorical, axis=0))
            return np.concatenate((rest, cats))
        else:
            return rest


# Wrapper to make sure that we get a deep copy of the contents instead of clone returning an untrained copy
class _Wrapper:
    def __init__(self, item):
        self.item = item


class _FrozenTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, wrapper):
        self.wrapper = wrapper

    def fit(self, X, y):
        return self

    def transform(self, X):
        return self.wrapper.item.transform(X)


def _freeze(transformer):
    return _FrozenTransformer(_Wrapper(transformer))


# Convert python objects to (possibly nested) types that can easily be represented as literals
def _sanitize(obj):
    if obj is None or isinstance(obj, (bool, int, str, float)):
        return obj
    elif isinstance(obj, dict):
        return {_sanitize(key): _sanitize(obj[key]) for key in obj}
    else:
        try:
            return [_sanitize(item) for item in obj]
        except Exception:
            raise ValueError(f"Could not sanitize input {obj}")


# Convert SingleTreeInterpreter to a python dictionary
def _tree_interpreter_to_dict(interp, features, leaf_data=lambda t, n: {}):
    tree = interp.tree_model_.tree_
    node_dict = interp.node_dict_

    def recurse(node_id):
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return {'leaf': True, 'n_samples': tree.n_node_samples[node_id], **leaf_data(tree, node_id, node_dict)}
        else:
            return {'leaf': False, 'feature': features[tree.feature[node_id]], 'threshold': tree.threshold[node_id],
                    'left': recurse(tree.children_left[node_id]),
                    'right': recurse(tree.children_right[node_id])}

    return recurse(0)


class _PolicyOutput:
    """
    A type encapsulating various information related to a learned policy.

    Attributes
    ----------
    tree_dictionary:dict
        The policy tree represented as a dictionary,
    policy_value:float
        The average value of applying the recommended policy (over using the control),
    always_treat:dict of str to float
        A dictionary mapping each non-control treatment to the value of always treating with it (over control),
    control_name:str
        The name of the control treatment
    """

    def __init__(self, tree_dictionary, policy_value, always_treat, control_name):
        self.tree_dictionary = tree_dictionary
        self.policy_value = policy_value
        self.always_treat = always_treat
        self.control_name = control_name


# named tuple type for storing results inside CausalAnalysis class;
# must be lifted to module level to enable pickling
_result = namedtuple("_result", field_names=[
    "feature_index", "feature_name", "feature_baseline", "feature_levels", "hinds",
    "X_transformer", "W_transformer", "estimator", "global_inference", "treatment_value"])


def _process_feature(name, feat_ind, verbose, categorical_inds, categories, heterogeneity_inds, min_counts, y, X,
                     nuisance_models, h_model, random_state, model_y, cv, mc_iters):
    try:
        if verbose > 0:
            print(f"CausalAnalysis: Feature {name}")

        discrete_treatment = feat_ind in categorical_inds
        if discrete_treatment:
            cats = categories[categorical_inds.index(feat_ind)]
        else:
            cats = 'auto'  # just leave the setting at the default otherwise

        # the transformation logic here is somewhat tricky; we always need to encode the categorical columns,
        # whether they end up in X or in W.  However, for the continuous columns, we want to scale them all
        # when running the first stage models, but don't want to scale the X columns when running the final model,
        # since then our coefficients will have odd units and our trees will also have decisions using those units.
        #
        # we achieve this by pipelining the X scaling with the Y and T models (with fixed scaling, not refitting)

        hinds = heterogeneity_inds[feat_ind]
        WX_transformer = ColumnTransformer([('encode', one_hot_encoder(drop='first'),
                                             [ind for ind in categorical_inds
                                              if ind != feat_ind]),
                                            ('drop', 'drop', feat_ind)],
                                           remainder=StandardScaler())
        W_transformer = ColumnTransformer([('encode', one_hot_encoder(drop='first'),
                                            [ind for ind in categorical_inds
                                             if ind != feat_ind and ind not in hinds]),
                                           ('drop', 'drop', hinds),
                                           ('drop_feat', 'drop', feat_ind)],
                                          remainder=StandardScaler())

        X_cont_inds = [ind for ind in hinds
                       if ind != feat_ind and ind not in categorical_inds]

        # Use _ColumnTransformer instead of ColumnTransformer so we can get feature names
        X_transformer = _ColumnTransformer([ind for ind in categorical_inds
                                            if ind != feat_ind and ind in hinds],
                                           X_cont_inds)

        # Controls are all other columns of X
        WX = WX_transformer.fit_transform(X)
        # can't use X[:, feat_ind] when X is a DataFrame
        T = _safe_indexing(X, feat_ind, axis=1)

        # TODO: we can't currently handle unseen values of the feature column when getting the effect;
        #       we might want to modify OrthoLearner (and other discrete treatment classes)
        #       so that the user can opt-in to allowing unseen treatment values
        #       (and return NaN or something in that case)

        W = W_transformer.fit_transform(X)
        X_xf = X_transformer.fit_transform(X)

        # HACK: this is slightly ugly because we rely on the fact that DML passes [X;W] to the first stage models
        #       and so we can just peel the first columns off of that combined array for rescaling in the pipeline
        # TODO: consider addding an API to DML that allows for better understanding of how the nuisance inputs are
        #       built, such as model_y_feature_names, model_t_feature_names, model_y_transformer, etc., so that this
        #       becomes a valid approach to handling this
        X_scaler = ColumnTransformer([('scale', StandardScaler(),
                                       list(range(len(X_cont_inds))))],
                                     remainder='passthrough').fit(np.hstack([X_xf, W])).named_transformers_['scale']

        X_scaler_fixed = ColumnTransformer([('scale', _freeze(X_scaler),
                                             list(range(len(X_cont_inds))))],
                                           remainder='passthrough')

        if W.shape[1] == 0:
            # array checking routines don't accept 0-width arrays
            W = None

        if X_xf.shape[1] == 0:
            X_xf = None

        if verbose > 0:
            print("CausalAnalysis: performing model selection on T model")

        # perform model selection
        model_t = (_first_stage_clf(WX, T, automl=nuisance_models == 'automl',
                                    min_count=min_counts.get(feat_ind, None),
                                    random_state=random_state, verbose=verbose)
                   if discrete_treatment else _first_stage_reg(WX, T, automl=nuisance_models == 'automl',
                                                               random_state=random_state,
                                                               verbose=verbose))

        pipelined_model_t = Pipeline([('scale', X_scaler_fixed),
                                      ('model', model_t)])

        pipelined_model_y = Pipeline([('scale', X_scaler_fixed),
                                      ('model', model_y)])

        if X_xf is None and h_model == 'forest':
            warnings.warn(f"Using a linear model instead of a forest model for feature '{name}' "
                          "because forests don't support models with no heterogeneity indices")
            h_model = 'linear'

        if h_model == 'linear':
            est = LinearDML(model_y=pipelined_model_y,
                            model_t=pipelined_model_t,
                            discrete_treatment=discrete_treatment,
                            fit_cate_intercept=True,
                            categories=cats,
                            random_state=random_state,
                            cv=cv,
                            mc_iters=mc_iters)
        elif h_model == 'forest':
            est = CausalForestDML(model_y=pipelined_model_y,
                                  model_t=pipelined_model_t,
                                  discrete_treatment=discrete_treatment,
                                  n_estimators=4000,
                                  min_var_leaf_on_val=True,
                                  categories=cats,
                                  random_state=random_state,
                                  verbose=verbose,
                                  cv=cv,
                                  mc_iters=mc_iters)

            if verbose > 0:
                print("CausalAnalysis: tuning forest")
            est.tune(y, T, X=X_xf, W=W)
        if verbose > 0:
            print("CausalAnalysis: training causal model")
        est.fit(y, T, X=X_xf, W=W, cache_values=True)

        # Prefer ate__inference to const_marginal_ate_inference(X) because it is doubly-robust and not conservative
        if h_model == 'forest' and discrete_treatment:
            global_inference = est.ate__inference()
        else:
            # convert to NormalInferenceResults for consistency
            inf = est.const_marginal_ate_inference(X=X_xf)
            global_inference = NormalInferenceResults(d_t=inf.d_t, d_y=inf.d_y,
                                                      pred=inf.mean_point,
                                                      pred_stderr=inf.stderr_mean,
                                                      mean_pred_stderr=None,
                                                      inf_type='ate')

        # Set the dictionary values shared between local and global summaries
        if discrete_treatment:
            cats = est.transformer.categories_[0]
            baseline = cats[est.transformer.drop_idx_[0]]
            cats = cats[np.setdiff1d(np.arange(len(cats)),
                                     est.transformer.drop_idx_[0])]
            d_t = len(cats)
            insights = {
                _CausalInsightsConstants.TypeKey: ['cat'] * d_t,
                _CausalInsightsConstants.RawFeatureNameKey: [name] * d_t,
                _CausalInsightsConstants.CategoricalColumnKey: cats.tolist(),
                _CausalInsightsConstants.EngineeredNameKey: [
                    f"{name} (base={baseline}): {c}" for c in cats]
            }
            treatment_value = 1
        else:
            d_t = 1
            cats = ["num"]
            baseline = None
            insights = {
                _CausalInsightsConstants.TypeKey: ["num"],
                _CausalInsightsConstants.RawFeatureNameKey: [name],
                _CausalInsightsConstants.CategoricalColumnKey: [name],
                _CausalInsightsConstants.EngineeredNameKey: [name]
            }
            # calculate a "typical" treatment value, using the mean of the absolute value of non-zero treatments
            treatment_value = np.mean(np.abs(T[T != 0]))

        result = _result(feature_index=feat_ind,
                         feature_name=name,
                         feature_baseline=baseline,
                         feature_levels=cats,
                         hinds=hinds,
                         X_transformer=X_transformer,
                         W_transformer=W_transformer,
                         estimator=est,
                         global_inference=global_inference,
                         treatment_value=treatment_value)

        return insights, result
    except Exception as e:
        warnings.warn(f"Exception caught when training model for feature {name}: {e}")
        return e


# Unless we're opting into minimal cross-fitting, this is the minimum number of instances of each category
# required to fit a discrete DML model
_CAT_LIMIT = 10

# TODO: Add other nuisance model options, such as {'azure_automl', 'forests', 'boosting'} that will use particular
#       sub-cases of models or also integrate with azure autoML. (post-MVP)
# TODO: Add other heterogeneity model options, such as {'automl'} for performing
#       model selection for the causal effect, or {'sparse_linear'} for using a debiased lasso. (post-MVP)
# TODO: Enable multi-class classification (post-MVP)


class CausalAnalysis:
    """
    Note: this class is experimental and the API may evolve over our next few releases.

    Gets causal importance of features.

    Parameters
    ----------
    feature_inds: array_like of int, str, or bool
        The features for which to estimate causal effects, expressed as either column indices,
        column names, or boolean flags indicating which columns to pick
    categorical: array_like of int, str, or bool
        The features which are categorical in nature, expressed as either column indices,
        column names, or boolean flags indicating which columns to pick
    heterogeneity_inds: array_like of int, str, or bool, or None or list of array_like elements or None, default None
        If a 1d array, then whenever estimating a heterogeneous (local) treatment effect
        model, then only the features in this array will be used for heterogeneity. If a 2d
        array then its first dimension should be len(feature_inds) and whenever estimating
        a local causal effect for target feature feature_inds[i], then only features in
        heterogeneity_inds[i] will be used for heterogeneity. If heterogeneity_inds[i]=None, then all features
        are used for heterogeneity when estimating local causal effect for feature_inds[i], and likewise if
        heterogeneity_inds[i]=[] then no features will be used for heterogeneity. If heterogeneity_ind=None
        then all features are used for heterogeneity for all features, and if heterogeneity_inds=[] then
        no features will be.
    feature_names: list of str, optional
        The names for all of the features in the data.  Not necessary if the input will be a dataframe.
        If None and the input is a plain numpy array, generated feature names will be ['X1', 'X2', ...].
    upper_bound_on_cat_expansion: int, default 5
        The maximum number of categorical values allowed, because they are expanded via one-hot encoding. If a
        feature has more than this many values, then a causal effect model is not fitted for that target feature
        and a warning flag is raised. The remainder of the models are fitted.
    classification: bool, default False
        Whether this is a classification (as opposed to regression) task
    nuisance_models: one of {'linear', 'automl'}, default 'linear'
        The model class to use for nuisance estimation.  Separate nuisance models are trained to predict the
        outcome and also each individual feature column from all of the other columns in the dataset as a prerequisite
        step before computing the actual causal effect for that feature column. If 'linear', then
        :class:`~sklearn.linear_model.LassoCV` (for regression) or :class:`~sklearn.linear_model.LogisticRegressionCV`
        (for classification) is used for these models. If 'automl', then model selection picks the best-performing
        among several different model classes for each model being trained using k-fold cross-validation, which
        requires additional computation.
    heterogeneity_model: one of {'linear', 'forest'}, default 'linear'
        The type of model to use for the final heterogeneous treatment effect model. 'linear' means that a
        the estimated treatment effect for a column will be a linear function of the heterogeneity features for that
        column, while 'forest' means that a forest model will be trained to compute the effect from those heterogeneity
        features instead.
    categories: 'auto' or list of ('auto' or list of values), default 'auto'
        What categories to use for the categorical columns.  If 'auto', then the categories will be inferred for
        all categorical columns; otherwise this argument should have as many entries as there are categorical columns,
        and each entry should be either 'auto' to infer the values for that column or the list of values for the
        column. If explicit values are provided, the first value is treated as the "control" value for that column
        against which other values are compared.
    n_jobs: int, default -1
        Degree of parallelism to use when training models via joblib.Parallel
    verbose : int, default 0
        Controls the verbosity when fitting and predicting.
    cv: int, cross-validation generator or an iterable, default 5
        Determines the strategy for cross-fitting used when training causal models for each feature.
        Possible inputs for cv are:

        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).
    mc_iters: int, default 3
        The number of times to rerun the first stage models to reduce the variance of the causal model nuisances.
    skip_cat_limit_checks: bool, default False
        By default, categorical features need to have several instances of each category in order for a model to be
        fit robustly. Setting this to True will skip these checks (although at least 2 instances will always be
        required for linear heterogeneity models, and 4 for forest heterogeneity models even in that case).
    random_state : int, RandomState instance, or None, default None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    Attributes
    ----------
    nuisance_models_: str
        The nuisance models setting used for the most recent call to fit
    heterogeneity_model: str
        The heterogeneity model setting used for the most recent call to fit
    feature_names_: list of str
        The list of feature names from the data in the most recent call to fit
    trained_feature_indices_: list of int
        The list of feature indices where models were trained successfully
    untrained_feature_indices_: list of tuple of (int, str or Exception)
        The list of indices that were requested but not able to be trained succesfully,
        along with either a reason or caught Exception for each
    """

    def __init__(self, feature_inds, categorical, heterogeneity_inds=None, feature_names=None, classification=False,
                 upper_bound_on_cat_expansion=5, nuisance_models='linear', heterogeneity_model='linear', *,
                 categories='auto', n_jobs=-1, verbose=0, cv=5, mc_iters=3, skip_cat_limit_checks=False,
                 random_state=None):
        self.feature_inds = feature_inds
        self.categorical = categorical
        self.heterogeneity_inds = heterogeneity_inds
        self.feature_names = feature_names
        self.classification = classification
        self.upper_bound_on_cat_expansion = upper_bound_on_cat_expansion
        self.nuisance_models = nuisance_models
        self.heterogeneity_model = heterogeneity_model
        self.categories = categories
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.cv = cv
        self.mc_iters = mc_iters
        self.skip_cat_limit_checks = skip_cat_limit_checks
        self.random_state = random_state

    def fit(self, X, y, warm_start=False):
        """
        Fits global and local causal effect models for each feature in feature_inds on the data

        Parameters
        ----------
        X : array_like
            Feature data
        y : array_like of shape (n,) or (n,1)
            Outcome. If classification=True, then y should take two values. Otherwise an error is raised
            that only binary classification is implemented for now.
            TODO. enable multi-class classification for y (post-MVP)
        warm_start : bool, default False
            If False, train models for each feature in `feature_inds`.
            If True, train only models for features in `feature_inds` that had not already been trained by
            the previous call to `fit`, and for which neither the corresponding heterogeneity_inds, nor the
            automl flag have changed. If heterogeneity_inds have changed, then the final stage model of these features
            will be refit. If the automl flag has changed, then whole model is refit, despite the warm start flag.
        """

        # Validate inputs
        assert self.nuisance_models in ['automl', 'linear'], (
            "The only supported nuisance models are 'linear' and 'automl', "
            f"but was given {self.nuisance_models}")

        assert self.heterogeneity_model in ['linear', 'forest'], (
            "The only supported heterogeneity models are 'linear' and 'forest' but received "
            f"{self.heterogeneity_model}")

        assert np.ndim(X) == 2, f"X must be a 2-dimensional array, but here had shape {np.shape(X)}"

        assert iterable(self.feature_inds), f"feature_inds should be array_like, but got {self.feature_inds}"
        assert iterable(self.categorical), f"categorical should be array_like, but got {self.categorical}"
        assert self.heterogeneity_inds is None or iterable(self.heterogeneity_inds), (
            f"heterogeneity_inds should be None or array_like, but got {self.heterogeneity_inds}")
        assert self.feature_names is None or iterable(self.feature_names), (
            f"feature_names should be None or array_like, but got {self.feature_names}")
        assert self.categories == 'auto' or iterable(self.categories), (
            f"categories should be 'auto' or array_like, but got {self.categories}")

        # TODO: check compatibility of X and Y lengths

        if warm_start:
            if not hasattr(self, "_results"):
                # no previous fit, cancel warm start
                warm_start = False

            elif self._d_x != X.shape[1]:
                raise ValueError(
                    f"Can't warm start: previous X had {self._d_x} columns, new X has {X.shape[1]} columns")

        # work with numeric feature indices, so that we can easily compare with categorical ones
        train_inds = _get_column_indices(X, self.feature_inds)

        if len(train_inds) == 0:
            raise ValueError(
                "No features specified. At least one feature index must be specified so that a model can be trained.")

        heterogeneity_inds = self.heterogeneity_inds
        if heterogeneity_inds is None:
            heterogeneity_inds = [None for ind in train_inds]

        # if heterogeneity_inds is 1D, repeat it
        if heterogeneity_inds == [] or isinstance(heterogeneity_inds[0], (int, str, bool)):
            heterogeneity_inds = [heterogeneity_inds for _ in train_inds]

        # heterogeneity inds should be a 2D list of length same as train_inds
        elif heterogeneity_inds is not None and len(heterogeneity_inds) != len(train_inds):
            raise ValueError("Heterogeneity indexes should have the same number of entries, but here "
                             f" there were {len(heterogeneity_inds)} heterogeneity entries but "
                             f" {len(train_inds)} feature indices.")

        # replace None elements of heterogeneity_inds and ensure indices are numeric
        heterogeneity_inds = {ind: list(range(X.shape[1])) if hinds is None else _get_column_indices(X, hinds)
                              for ind, hinds in zip(train_inds, heterogeneity_inds)}

        if warm_start:
            train_y_model = False
            if self.nuisance_models != self.nuisance_models_:
                warnings.warn("warm_start will be ignored since the nuisance models have changed "
                              f"from {self.nuisance_models_} to {self.nuisance_models} since the previous call to fit")
                warm_start = False
                train_y_model = True

            if self.heterogeneity_model != self.heterogeneity_model_:
                warnings.warn("warm_start will be ignored since the heterogeneity model has changed "
                              f"from {self.heterogeneity_model_} to {self.heterogeneity_model} "
                              "since the previous call to fit")
                warm_start = False

            # TODO: bail out also if categorical columns, classification, random_state changed?
        else:
            train_y_model = True

        # TODO: should we also train a new model_y under any circumstances when warm_start is True?
        if warm_start:
            new_inds = [ind for ind in train_inds if (ind not in self._cache or
                                                      heterogeneity_inds[ind] != self._cache[ind][1].hinds)]
        else:
            new_inds = list(train_inds)

            self._cache = {}  # store mapping from feature to insights, results

            # train the Y model
            if train_y_model:
                # perform model selection for the Y model using all X, not on a per-column basis
                allX = ColumnTransformer([('encode',
                                           one_hot_encoder(drop='first'),
                                           self.categorical)],
                                         remainder=StandardScaler()).fit_transform(X)

                if self.verbose > 0:
                    print("CausalAnalysis: performing model selection on overall Y model")

                if self.classification:
                    self._model_y = _first_stage_clf(allX, y, automl=self.nuisance_models == 'automl',
                                                     make_regressor=True,
                                                     random_state=self.random_state, verbose=self.verbose)
                else:
                    self._model_y = _first_stage_reg(allX, y, automl=self.nuisance_models == 'automl',
                                                     random_state=self.random_state, verbose=self.verbose)

        if self.classification:
            # now that we've trained the classifier and wrapped it, ensure that y is transformed to
            # work with the regression wrapper

            # we use column_or_1d to treat pd.Series and pd.DataFrame objects the same way as arrays
            y = column_or_1d(y).reshape(-1, 1)

            # note that this needs to happen after wrapping to generalize to the multi-class case,
            # since otherwise we'll have too many columns to be able to train a classifier
            y = one_hot_encoder(drop='first').fit_transform(y)

        assert y.ndim == 1 or y.shape[1] == 1, ("Multiclass classification isn't supported" if self.classification
                                                else "Only a single outcome is supported")

        self._vec_y = y.ndim == 1
        self._d_x = X.shape[1]

        # start with empty results and default shared insights
        self._results = []
        self._shared = _get_default_shared_insights_output()
        self._shared[_CausalInsightsConstants.InitArgsKey] = {
            'feature_inds': _sanitize(self.feature_inds),
            'categorical': _sanitize(self.categorical),
            'heterogeneity_inds': _sanitize(self.heterogeneity_inds),
            'feature_names': _sanitize(self.feature_names),
            'classification': _sanitize(self.classification),
            'upper_bound_on_cat_expansion': _sanitize(self.upper_bound_on_cat_expansion),
            'nuisance_models': _sanitize(self.nuisance_models),
            'heterogeneity_model': _sanitize(self.heterogeneity_model),
            'categories': _sanitize(self.categories),
            'n_jobs': _sanitize(self.n_jobs),
            'verbose': _sanitize(self.verbose),
            'random_state': _sanitize(self.random_state)
        }

        # convert categorical indicators to numeric indices
        categorical_inds = _get_column_indices(X, self.categorical)

        categories = self.categories
        if categories == 'auto':
            categories = ['auto' for _ in categorical_inds]
        else:
            assert len(categories) == len(categorical_inds), (
                "If categories is not 'auto', it must contain one entry per categorical column.  Instead, categories"
                f"has length {len(categories)} while there are {len(categorical_inds)} categorical columns.")

        # check for indices over the categorical expansion bound
        invalid_inds = getattr(self, 'untrained_feature_indices_', [])

        # assume we'll be able to train former failures this time; we'll add them back if not
        invalid_inds = [(ind, reason) for (ind, reason) in invalid_inds if ind not in new_inds]

        self._has_column_names = True
        if self.feature_names is None:
            if hasattr(X, "iloc"):
                feature_names = X.columns
            else:
                self._has_column_names = False
                feature_names = [f"x{i}" for i in range(X.shape[1])]
        else:
            feature_names = self.feature_names
        self.feature_names_ = feature_names

        min_counts = {}
        for ind in new_inds:
            column_text = self._format_col(ind)

            if ind in categorical_inds:
                cats, counts = np.unique(_safe_indexing(X, ind, axis=1), return_counts=True)
                min_ind = np.argmin(counts)
                n_cat = len(cats)
                if n_cat > self.upper_bound_on_cat_expansion:
                    warnings.warn(f"{column_text} has more than {self.upper_bound_on_cat_expansion} "
                                  f"values (found {n_cat}) so no heterogeneity model will be fit for it; "
                                  "increase 'upper_bound_on_cat_expansion' to change this behavior.")
                    # can't remove in place while iterating over new_inds, so store in separate list
                    invalid_inds.append((ind, 'upper_bound_on_cat_expansion'))

                elif counts[min_ind] < _CAT_LIMIT:
                    if self.skip_cat_limit_checks and (counts[min_ind] >= 5 or
                                                       (counts[min_ind] >= 2 and
                                                        self.heterogeneity_model != 'forest')):
                        # train the model, but warn
                        warnings.warn(f"{column_text}'s value {cats[min_ind]} has only {counts[min_ind]} instances in "
                                      f"the training dataset, which is less than the lower limit ({_CAT_LIMIT}). "
                                      "A model will still be fit because 'skip_cat_limit_checks' is True, "
                                      "but this model may not be robust.")
                        min_counts[ind] = counts[min_ind]
                    elif counts[min_ind] < 2 or (counts[min_ind] < 5 and self.heterogeneity_model == 'forest'):
                        # no model can be trained in this case since we need more folds
                        warnings.warn(f"{column_text}'s value {cats[min_ind]} has only {counts[min_ind]} instances in "
                                      "the training dataset, but linear heterogeneity models need at least 2 and "
                                      "forest heterogeneity models need at least 5 instances, so no model will be fit "
                                      "for this column")
                        invalid_inds.append((ind, 'cat_limit'))
                    else:
                        # don't train a model, but suggest workaround since there are enough instances of least
                        # populated class
                        warnings.warn(f"{column_text}'s value {cats[min_ind]} has only {counts[min_ind]} instances in "
                                      f"the training dataset, which is less than the lower limit ({_CAT_LIMIT}), "
                                      "so no heterogeneity model will be fit for it. This check can be turned off by "
                                      "setting 'skip_cat_limit_checks' to True, but that may result in an inaccurate "
                                      "model for this feature.")
                        invalid_inds.append((ind, 'cat_limit'))

        for (ind, _) in invalid_inds:
            new_inds.remove(ind)
            # also remove from train_inds so we don't try to access the result later
            train_inds.remove(ind)
            if len(train_inds) == 0:
                raise ValueError("No features remain; increase the upper_bound_on_cat_expansion and ensure that there "
                                 "are several instances of each categorical value so that at least "
                                 "one feature model can be trained.")

        # extract subset of names matching new columns
        new_feat_names = _safe_indexing(feature_names, new_inds)

        cache_updates = dict(zip(new_inds,
                                 joblib.Parallel(
                                     n_jobs=self.n_jobs,
                                     verbose=self.verbose
                                 )(joblib.delayed(_process_feature)(
                                     feat_name, feat_ind,
                                     self.verbose, categorical_inds, categories, heterogeneity_inds, min_counts, y, X,
                                     self.nuisance_models, self.heterogeneity_model, self.random_state, self._model_y,
                                     self.cv, self.mc_iters)
                                     for feat_name, feat_ind in zip(new_feat_names, new_inds))))

        # track indices where an exception was thrown, since we can't remove from dictionary while iterating
        inds_to_remove = []
        for ind, value in cache_updates.items():
            if isinstance(value, Exception):
                # don't want to cache this failed result
                inds_to_remove.append(ind)
                train_inds.remove(ind)
                invalid_inds.append((ind, value))

        for ind in inds_to_remove:
            del cache_updates[ind]

        self._cache.update(cache_updates)

        for ind in train_inds:
            dict_update, result = self._cache[ind]
            self._results.append(result)
            for k in dict_update:
                self._shared[k] += dict_update[k]

        invalid_inds.sort()
        self.untrained_feature_indices_ = invalid_inds
        self.trained_feature_indices_ = train_inds

        self.nuisance_models_ = self.nuisance_models
        self.heterogeneity_model_ = self.heterogeneity_model
        return self

    def _format_col(self, ind):
        if self._has_column_names:
            return f"Column {ind} ({self.feature_names_[ind]})"
        else:
            return f"Column {ind}"

    # properties to return from effect InferenceResults
    @staticmethod
    def _point_props(alpha):
        return [(_CausalInsightsConstants.PointEstimateKey, 'point_estimate'),
                (_CausalInsightsConstants.StandardErrorKey, 'stderr'),
                (_CausalInsightsConstants.ZStatKey, 'zstat'),
                (_CausalInsightsConstants.PValueKey, 'pvalue'),
                (_CausalInsightsConstants.ConfidenceIntervalLowerKey, lambda inf: inf.conf_int(alpha=alpha)[0]),
                (_CausalInsightsConstants.ConfidenceIntervalUpperKey, lambda inf: inf.conf_int(alpha=alpha)[1])]

    # properties to return from PopulationSummaryResults
    @staticmethod
    def _summary_props(alpha):
        return [(_CausalInsightsConstants.PointEstimateKey, 'mean_point'),
                (_CausalInsightsConstants.StandardErrorKey, 'stderr_mean'),
                (_CausalInsightsConstants.ZStatKey, 'zstat'),
                (_CausalInsightsConstants.PValueKey, 'pvalue'),
                (_CausalInsightsConstants.ConfidenceIntervalLowerKey, lambda inf: inf.conf_int_mean(alpha=alpha)[0]),
                (_CausalInsightsConstants.ConfidenceIntervalUpperKey, lambda inf: inf.conf_int_mean(alpha=alpha)[1])]

    # Converts strings to property lookups or method calls as a convenience so that the
    # _point_props and _summary_props above can be applied to an inference object
    @staticmethod
    def _make_accessor(attr):
        if isinstance(attr, str):
            s = attr

            def attr(o):
                val = getattr(o, s)
                if callable(val):
                    return val()
                else:
                    return val
        return attr

    # Create a summary combining all results into a single output; this is used
    # by the various causal_effect and causal_effect_dict methods to generate either a dataframe
    # or a dictionary, respectively, based on the summary function passed into this method
    def _summarize(self, *, summary, get_inference, props, expand_arr, drop_sample):

        assert hasattr(self, "_results"), "This object has not been fit, so cannot get results"

        # ensure array has shape (m,y,t)
        def ensure_proper_dims(arr):
            if expand_arr:
                # population summary is missing sample dimension; add it for consistency
                arr = np.expand_dims(arr, 0)
            if self._vec_y:
                # outcome dimension is missing; add it for consistency
                arr = np.expand_dims(arr, axis=1)
            assert 2 <= arr.ndim <= 3
            # add singleton treatment dimension if missing
            return arr if arr.ndim == 3 else np.expand_dims(arr, axis=2)

        # store set of inference results so we don't need to recompute per-attribute below in summary/coalesce
        infs = [get_inference(res) for res in self._results]

        # each attr has dimension (m,y) or (m,y,t)
        def coalesce(attr):
            """Join together the arrays for each feature"""
            attr = self._make_accessor(attr)
            # concatenate along treatment dimension
            arr = np.concatenate([ensure_proper_dims(attr(inf))
                                  for inf in infs], axis=2)

            # for dictionary representation, want to remove unneeded sample dimension
            # in cohort and global results
            if drop_sample:
                arr = np.squeeze(arr, 0)

            return arr

        return summary([(key, coalesce(val)) for key, val in props])

    def _pandas_summary(self, get_inference, *, props, n,
                        expand_arr=False, keep_all_levels=False):
        """
        Summarizes results into a dataframe.

        Parameters
        ----------
        get_inference : lambda
            Method to get the relevant inference results from each result object
        props : list of (string, str or lambda)
            Set of column names and ways to get the corresponding values from the inference object
        n : int
            The number of samples in the dataset
        expand_arr : bool, default False
            Whether to add a synthetic sample dimension to the result arrays when performing internal computations
        keep_all_levels : bool, default False
            Whether to keep all levels, even when they don't take on more than one value;
            Note that regardless of this argument the "sample" level will only be present if expand_arr is False
        """
        def make_dataframe(props):

            to_include = OrderedDict([(key, value.reshape(-1))
                                      for key, value in props])

            # TODO: enrich outcome logic for multi-class classification when that is supported
            index = pd.MultiIndex.from_tuples([(i, outcome, res.feature_name, f"{lvl}v{res.feature_baseline}"
                                                if res.feature_baseline is not None
                                                else lvl)
                                               for i in range(n)
                                               for outcome in ["y0"]
                                               for res in self._results
                                               for lvl in res.feature_levels],
                                              names=["sample", "outcome", "feature", "feature_value"])

            if expand_arr:
                # There is no actual sample level in this data
                index = index.droplevel("sample")

            if not keep_all_levels:
                for lvl in index.levels:
                    if len(lvl) == 1:
                        if not isinstance(index, pd.MultiIndex):
                            # can't drop only level
                            index = pd.Index([self._results[0].feature_name], name="feature")
                        else:
                            index = index.droplevel(lvl.name)
            return pd.DataFrame(to_include, index=index)

        return self._summarize(summary=make_dataframe,
                               get_inference=get_inference,
                               props=props,
                               expand_arr=expand_arr,
                               drop_sample=False)  # dropping the sample dimension is handled above instead

    def _dict_summary(self, get_inference, *, n, props, kind, drop_sample=False, expand_arr=False, row_wise=False):
        """
        Summarizes results into a dictionary.

        Parameters
        ----------
        get_inference : lambda
            Method to get the relevant inference results from each result object
        n : int
            The number of samples in the dataset
        props : list of (string, str or lambda)
            Set of column names and ways to get the corresponding values from the inference object
        kind : str
            The kind of inference results to get (e.g. 'global', 'local', or 'cohort')
        drop_sample : bool, default False
            Whether to drop the sample dimension from each array
        expand_arr : bool, default False
            Whether to add an initial sample dimension to the result arrays
        row_wise : bool, default False
            Whether to return a list of dictionaries (one dictionary per row) instead of
            a dictionary of lists (one list per column)
        """
        def make_dict(props):
            # should be serialization-ready and contain no numpy arrays
            res = _get_default_specific_insights(kind)
            shared = self._shared

            if row_wise:
                row_data = {}
                # remove entries belonging to row data, since we're including them in the list of nested dictionaries
                for k in _get_data_causal_insights_keys():
                    del res[k]

                shared = shared.copy()  # copy so that we can modify without affecting shared state
                # TODO: Note that there's no column metadata for the sample number - should there be?
                for k in _get_column_causal_insights_keys():
                    # need to replicate the column info for each sample, then remove from the shared data
                    row_data[k] = shared[k] * n
                    del shared[k]

                # NOTE: the flattened order has the ouptut dimension before the feature dimension
                #       which may need to be revisited once we support multiclass
                row_data.update([(key, value.flatten()) for key, value in props])

                # get the length of the list corresponding to the first dictionary key
                # `list(row_data)` gets the keys as a list, since `row_data.keys()` can't be indexed into
                n_rows = len(row_data[list(row_data)[0]])
                res[_CausalInsightsConstants.RowData] = [{key: row_data[key][i]
                                                          for key in row_data} for i in range(n_rows)]
            else:
                res.update([(key, value.tolist()) for key, value in props])

            return {**shared, **res}

        return self._summarize(summary=make_dict,
                               get_inference=get_inference,
                               props=props,
                               expand_arr=expand_arr,
                               drop_sample=drop_sample)

    def global_causal_effect(self, *, alpha=0.05, keep_all_levels=False):
        """
        Get the global causal effect for each feature as a pandas DataFrame.

        Parameters
        ----------
        alpha : float, default 0.05
            The confidence level of the confidence interval
        keep_all_levels : bool, default False
            Whether to keep all levels of the output dataframe ('outcome', 'feature', and 'feature_level')
            even if there was only a single value for that level; by default single-valued levels are dropped.

        Returns
        -------
        global_effects : DataFrame
            DataFrame with the following structure:

            :Columns: ['point', 'stderr', 'zstat', 'pvalue', 'ci_lower', 'ci_upper']
            :Index: ['feature', 'feature_value']
            :Rows: For each feature that is numerical, we have an entry with index ['{feature_name}', 'num'], where
                    'num' is literally the string 'num' and feature_name is the input feature name.
                    For each feature that is categorical, we have an entry with index ['{feature_name}',
                    '{cat}v{base}'] where cat is the category value and base is the category used as baseline.
                    If all features are numerical then the feature_value index is dropped in the dataframe, but not
                    in the serialized dict.
        """
        # a global inference indicates the effect of that one feature on the outcome
        return self._pandas_summary(lambda res: res.global_inference, props=self._point_props(alpha),
                                    n=1, expand_arr=True, keep_all_levels=keep_all_levels)

    def _global_causal_effect_dict(self, *, alpha=0.05, row_wise=False):
        """
        Gets the global causal effect for each feature as dictionary.

        Dictionary entries for predictions, etc. will be nested lists of shape (d_y, sum(d_t))

        Only for serialization purposes to upload to AzureML
        """
        return self._dict_summary(lambda res: res.global_inference, props=self._point_props(alpha),
                                  kind='global', n=1, row_wise=row_wise, drop_sample=True, expand_arr=True)

    def _cohort_effect_inference(self, Xtest):
        assert np.ndim(Xtest) == 2 and np.shape(Xtest)[1] == self._d_x, (
            "Shape of Xtest must be compatible with shape of X, "
            f"but got shape {np.shape(Xtest)} instead of (n, {self._d_x})"
        )

        def inference_from_result(result):
            est = result.estimator
            X = result.X_transformer.transform(Xtest)
            if X.shape[1] == 0:
                X = None
            return est.const_marginal_ate_inference(X=X)
        return inference_from_result

    def cohort_causal_effect(self, Xtest, *, alpha=0.05, keep_all_levels=False):
        """
        Gets the average causal effects for a particular cohort defined by a population of X's.

        Parameters
        ----------
        Xtest : array_like
            The cohort samples for which to return the average causal effects within cohort
        alpha : float, default 0.05
            The confidence level of the confidence interval
        keep_all_levels : bool, default False
            Whether to keep all levels of the output dataframe ('outcome', 'feature', and 'feature_level')
            even if there was only a single value for that level; by default single-valued levels are dropped.

        Returns
        -------
        cohort_effects : DataFrame
            DataFrame with the following structure:

            :Columns: ['point', 'stderr', 'zstat', 'pvalue', 'ci_lower', 'ci_upper']
            :Index: ['feature', 'feature_value']
            :Rows: For each feature that is numerical, we have an entry with index ['{feature_name}', 'num'], where
              'num' is literally the string 'num' and feature_name is the input feature name.
              For each feature that is categorical, we have an entry with index ['{feature_name}', '{cat}v{base}']
              where cat is the category value and base is the category used as baseline.
              If all features are numerical then the feature_value index is dropped in the dataframe, but not
              in the serialized dict.
        """
        return self._pandas_summary(self._cohort_effect_inference(Xtest),
                                    props=self._summary_props(alpha), n=1,
                                    expand_arr=True, keep_all_levels=keep_all_levels)

    def _cohort_causal_effect_dict(self, Xtest, *, alpha=0.05, row_wise=False):
        """
        Gets the cohort causal effects for each feature as dictionary.

        Dictionary entries for predictions, etc. will be nested lists of shape (d_y, sum(d_t))

        Only for serialization purposes to upload to AzureML
        """
        return self._dict_summary(self._cohort_effect_inference(Xtest), props=self._summary_props(alpha),
                                  kind='cohort', n=1, row_wise=row_wise, expand_arr=True, drop_sample=True)

    def _local_effect_inference(self, Xtest):
        assert np.ndim(Xtest) == 2 and np.shape(Xtest)[1] == self._d_x, (
            "Shape of Xtest must be compatible with shape of X, "
            f"but got shape {np.shape(Xtest)} instead of (n, {self._d_x})"
        )

        def inference_from_result(result):
            est = result.estimator
            X = result.X_transformer.transform(Xtest)
            if X.shape[1] == 0:
                X = None
            eff = est.const_marginal_effect_inference(X=X)
            if X is None:
                # need to reshape the output to match the input
                eff = eff._expand_outputs(Xtest.shape[0])
            return eff
        return inference_from_result

    def local_causal_effect(self, Xtest, *, alpha=0.05, keep_all_levels=False):
        """
        Gets the local causal effect for each feature as a pandas DataFrame.

        Parameters
        ----------
        Xtest : array_like
            The samples for which to return the causal effects
        alpha : float, default 0.05
            The confidence level of the confidence interval
        keep_all_levels : bool, default False
            Whether to keep all levels of the output dataframe ('sample', 'outcome', 'feature', and 'feature_level')
            even if there was only a single value for that level; by default single-valued levels are dropped.
        Returns
        -------
        global_effect : DataFrame
            DataFrame with the following structure:

            :Columns: ['point', 'stderr', 'zstat', 'pvalue', 'ci_lower', 'ci_upper']
            :Index: ['sample', 'feature', 'feature_value']
            :Rows: For each feature that is numeric, we have an entry with index
                   ['{sampleid}', '{feature_name}', 'num'],
                   where 'num' is literally the string 'num' and feature_name is the input feature name and sampleid is
                   the index of the sample in Xtest.
                   For each feature that is categorical, we have an entry with index
                   ['{sampleid', '{feature_name}', '{cat}v{base}']
                   where cat is the category value and base is the category used as baseline.
                   If all features are numerical then the feature_value index is dropped in the dataframe, but not
                   in the serialized dict.
        """
        return self._pandas_summary(self._local_effect_inference(Xtest),
                                    props=self._point_props(alpha), n=Xtest.shape[0], keep_all_levels=keep_all_levels)

    def _local_causal_effect_dict(self, Xtest, *, alpha=0.05, row_wise=False):
        """
        Gets the local feature importance as dictionary

        Dictionary entries for predictions, etc. will be nested lists of shape (n_rows, d_y, sum(d_t))

        Only for serialization purposes to upload to AzureML
        """
        return self._dict_summary(self._local_effect_inference(Xtest), props=self._point_props(alpha),
                                  kind='local', n=Xtest.shape[0], row_wise=row_wise)

    def _safe_result_index(self, X, feature_index):
        assert hasattr(self, "_results"), "This instance has not yet been fitted"

        assert np.ndim(X) == 2 and np.shape(X)[1] == self._d_x, (
            "Shape of X must be compatible with shape of the fitted X, "
            f"but got shape {np.shape(X)} instead of (n, {self._d_x})"
        )

        (numeric_index,) = _get_column_indices(X, [feature_index])

        bad_inds = dict(self.untrained_feature_indices_)
        if numeric_index in bad_inds:
            error = bad_inds[numeric_index]
            col_text = self._format_col(numeric_index)
            if error == 'cat_limit':
                msg = f"{col_text} had a value with fewer than {_CAT_LIMIT} occurences, so no model was fit for it"
            elif error == 'upper_bound_on_cat_expansion':
                msg = (f"{col_text} had more distinct values than the setting of 'upper_bound_on_cat_expansion', "
                       "so no model was fit for it")
            else:
                msg = (f"{col_text} generated the following error during fitting, "
                       f"so no model was fit for it:\n{str(error)}")
            raise ValueError(msg)

        if numeric_index not in self.trained_feature_indices_:
            raise ValueError(f"{self._format_col(numeric_index)} was not passed as a feature index "
                             "so no model was fit for it")

        results = [res for res in self._results
                   if res.feature_index == numeric_index]

        assert len(results) == 1
        (result,) = results
        return result

    def _whatif_inference(self, X, Xnew, feature_index, y):
        assert not self.classification, "What-if analysis cannot be applied to classification tasks"

        assert np.shape(X)[0] == np.shape(Xnew)[0] == np.shape(y)[0], (
            "X, Xnew, and y must have the same length, but have shapes "
            f"{np.shape(X)}, {np.shape(Xnew)}, and {np.shape(y)}"
        )

        assert np.size(feature_index) == 1, f"Only one feature index may be changed, but got {np.size(feature_index)}"

        T0 = _safe_indexing(X, feature_index, axis=1)
        T1 = Xnew
        result = self._safe_result_index(X, feature_index)
        X = result.X_transformer.transform(X)
        if X.shape[1] == 0:
            X = None
        inf = result.estimator.effect_inference(X=X, T0=T0, T1=T1)

        # we want to offset the inference object by the baseline estimate of y
        inf.translate(y)

        return inf

    def whatif(self, X, Xnew, feature_index, y, *, alpha=0.05):
        """
        Get counterfactual predictions when feature_index is changed to Xnew from its observational counterpart.

        Note that this only applies to regression use cases; for classification what-if analysis is not supported.

        Parameters
        ----------
        X: array_like
            Features
        Xnew: array_like
            New values of a single column of X
        feature_index: int or str
            The index of the feature being varied to Xnew, either as a numeric index or
            the string name if the input is a dataframe
        y: array_like
            Observed labels or outcome of a predictive model for baseline y values
        alpha : float in [0, 1], default 0.05
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.

        Returns
        -------
        y_new: DataFrame
            The predicted outputs that would have been observed under the counterfactual features
        """
        return self._whatif_inference(X, Xnew, feature_index, y).summary_frame(alpha=alpha)

    def _whatif_dict(self, X, Xnew, feature_index, y, *, alpha=0.05, row_wise=False):
        """
        Get counterfactual predictions when feature_index is changed to Xnew from its observational counterpart.

        Note that this only applies to regression use cases; for classification what-if analysis is not supported.

        Parameters
        ----------
        X: array_like
            Features
        Xnew: array_like
            New values of a single column of X
        feature_index: int or str
            The index of the feature being varied to Xnew, either as a numeric index or
            the string name if the input is a dataframe
        y: array_like
            Observed labels or outcome of a predictive model for baseline y values
        alpha : float in [0, 1], default 0.05
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.
        row_wise : bool, default False
            Whether to return a list of dictionaries (one dictionary per row) instead of
            a dictionary of lists (one list per column)
        Returns
        -------
        dict : dict
            The counterfactual predictions, as a dictionary
        """

        inf = self._whatif_inference(X, Xnew, feature_index, y)
        props = self._point_props(alpha=alpha)
        res = _get_default_specific_insights('whatif')
        if row_wise:
            row_data = {}
            # remove entries belonging to row data, since we're including them in the list of nested dictionaries
            for k in _get_data_causal_insights_keys():
                del res[k]

            row_data.update([(key, self._make_accessor(attr)(inf).flatten()) for key, attr in props])

            # get the length of the list corresponding to the first dictionary key
            # `list(row_data)` gets the keys as a list, since `row_data.keys()` can't be indexed into
            n_rows = len(row_data[list(row_data)[0]])
            res[_CausalInsightsConstants.RowData] = [{key: row_data[key][i]
                                                      for key in row_data} for i in range(n_rows)]
        else:
            res.update([(key, self._make_accessor(attr)(inf).tolist()) for key, attr in props])
        return res

    def _tree(self, is_policy, Xtest, feature_index, *, treatment_costs=0,
              max_depth=3, min_samples_leaf=2, min_impurity_decrease=1e-4,
              include_model_uncertainty=False, alpha=0.05):

        result = self._safe_result_index(Xtest, feature_index)
        Xtest = result.X_transformer.transform(Xtest)
        if Xtest.shape[1] == 0:
            Xtest = None
        if result.feature_baseline is None:
            treatment_names = ['decrease', 'increase']
        else:
            treatment_names = [f"{result.feature_baseline}"] + \
                [f"{lvl}" for lvl in result.feature_levels]

        TreeType = SingleTreePolicyInterpreter if is_policy else SingleTreeCateInterpreter
        intrp = TreeType(include_model_uncertainty=include_model_uncertainty,
                         uncertainty_level=alpha,
                         max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         min_impurity_decrease=min_impurity_decrease,
                         random_state=self.random_state)

        if is_policy:
            intrp.interpret(result.estimator, Xtest,
                            sample_treatment_costs=treatment_costs)
            if result.feature_baseline is None:  # continuous treatment, so apply a treatment level 10% of typical
                treatment_level = result.treatment_value * 0.1

                # NOTE: this calculation is correct only if treatment costs are marginal costs,
                #       because then scaling the difference between treatment value and treatment costs is the
                #       same as scaling the treatment value and subtracting the scaled treatment cost.
                #
                #       Note also that unlike the standard outputs of the SinglePolicyTreeInterpreter, for
                #       continuous treatments, the policy value should include the benefit of decreasing treatments
                #       (rather than just not treating at all)
                #
                #       We can get the total by seeing that if we restrict attention to units where we would treat,
                #         2 * policy_value - always_treat
                #       includes exactly their contribution because policy_value and always_treat both include it
                #       and likewise restricting attention to the units where we want to decrease treatment,
                #         2 * policy_value - always-treat
                #       also computes the *benefit* of decreasing treatment, because their contribution to policy_value
                #       is zero and the contribution to always_treat is negative
                treatment_total = (2 * intrp.policy_value_ - intrp.always_treat_value_.item()) * treatment_level
                always_totals = intrp.always_treat_value_ * treatment_level
            else:
                treatment_total = intrp.policy_value_
                always_totals = intrp.always_treat_value_

            policy_values = treatment_total, always_totals
        else:  # no policy values for CATE trees
            intrp.interpret(result.estimator, Xtest)
            policy_values = None

        return intrp, result.X_transformer.get_feature_names_out(self.feature_names_), treatment_names, policy_values

    # TODO: it seems like it would be better to just return the tree itself rather than plot it;
    #       however, the tree can't store the feature and treatment names we compute here...
    def plot_policy_tree(self, Xtest, feature_index, *, treatment_costs=0,
                         max_depth=3, min_samples_leaf=2, min_value_increase=1e-4, include_model_uncertainty=False,
                         alpha=0.05):
        """
        Plot a recommended policy tree using matplotlib.

        Parameters
        ----------
        X : array_like
            Features
        feature_index
            Index of the feature to be considered as treament
        treatment_costs: array_like, default 0
            Cost of treatment, as a scalar value or per-sample. For continuous features this is the marginal cost per
            unit of treatment; for discrete features, this is the difference in cost between each of the non-default
            values and the default value (i.e., if non-scalar the array should have shape (n,d_t-1))
        max_depth : int, default 3
            maximum depth of the tree
        min_samples_leaf : int, default 2
            minimum number of samples on each leaf
        min_value_increase : float, default 1e-4
            The minimum increase in the policy value that a split needs to create to construct it
        include_model_uncertainty : bool, default False
            Whether to include confidence interval information when building a simplified model of the cate model.
        alpha : float in [0, 1], default 0.05
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.
        """
        intrp, feature_names, treatment_names, _ = self._tree(True, Xtest, feature_index,
                                                              treatment_costs=treatment_costs,
                                                              max_depth=max_depth,
                                                              min_samples_leaf=min_samples_leaf,
                                                              min_impurity_decrease=min_value_increase,
                                                              include_model_uncertainty=include_model_uncertainty,
                                                              alpha=alpha)
        return intrp.plot(feature_names=feature_names, treatment_names=treatment_names)

    def _policy_tree_output(self, Xtest, feature_index, *, treatment_costs=0,
                            max_depth=3, min_samples_leaf=2, min_value_increase=1e-4, alpha=0.05):
        """
        Get a tuple of policy outputs.

        The first item in the tuple is the recommended policy tree expressed as a dictionary.
        The second item is the per-unit-average value of applying the learned policy; if the feature is continuous this
        means the gain from increasing the treatment by 10% of the typical amount for units where the treatment should
        be increased and decreasing the treatment by 10% of the typical amount when not.
        The third item is the value of always treating.  This is a list, with one entry per non-control-treatment for
        discrete features, or just a single entry for continuous features, again increasing by 10% of a typical amount.

        Parameters
        ----------
        X : array_like
            Features
        feature_index
            Index of the feature to be considered as treament
        treatment_costs: array_like, default 0
            Cost of treatment, as a scalar value or per-sample. For continuous features this is the marginal cost per
            unit of treatment; for discrete features, this is the difference in cost between each of the non-default
            values and the default value (i.e., if non-scalar the array should have shape (n,d_t-1))
        max_depth : int, default 3
            maximum depth of the tree
        min_samples_leaf : int, default 2
            minimum number of samples on each leaf
        min_value_increase : float, default 1e-4
            The minimum increase in the policy value that a split needs to create to construct it
        alpha : float in [0, 1], default 0.05
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.

        Returns
        -------
        output : _PolicyOutput
        """

        (intrp, feature_names, treatment_names,
            (policy_val, always_trt)) = self._tree(True, Xtest, feature_index,
                                                   treatment_costs=treatment_costs,
                                                   max_depth=max_depth,
                                                   min_samples_leaf=min_samples_leaf,
                                                   min_impurity_decrease=min_value_increase,
                                                   alpha=alpha)

        def policy_data(tree, node_id, node_dict):
            return {'treatment': treatment_names[np.argmax(tree.value[node_id])]}
        return _PolicyOutput(_tree_interpreter_to_dict(intrp, feature_names, policy_data),
                             policy_val,
                             {treatment_names[i + 1]: val
                              for (i, val) in enumerate(always_trt.tolist())},
                             treatment_names[0])

    # TODO: it seems like it would be better to just return the tree itself rather than plot it;
    #       however, the tree can't store the feature and treatment names we compute here...
    def plot_heterogeneity_tree(self, Xtest, feature_index, *,
                                max_depth=3, min_samples_leaf=2, min_impurity_decrease=1e-4,
                                include_model_uncertainty=False,
                                alpha=0.05):
        """
        Plot an effect heterogeneity tree using matplotlib.

        Parameters
        ----------
        X : array_like
            Features
        feature_index
            Index of the feature to be considered as treament
        max_depth : int, default 3
            maximum depth of the tree
        min_samples_leaf : int, default 2
            minimum number of samples on each leaf
        min_impurity_decrease : float, default 1e-4
            The minimum decrease in the impurity/uniformity of the causal effect that a split needs to
            achieve to construct it
        include_model_uncertainty : bool, default False
            Whether to include confidence interval information when building a simplified model of the cate model.
        alpha : float in [0, 1], default 0.05
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.
        """

        intrp, feature_names, treatment_names, _ = self._tree(False, Xtest, feature_index,
                                                              max_depth=max_depth,
                                                              min_samples_leaf=min_samples_leaf,
                                                              min_impurity_decrease=min_impurity_decrease,
                                                              include_model_uncertainty=include_model_uncertainty,
                                                              alpha=alpha)
        return intrp.plot(feature_names=feature_names,
                          treatment_names=treatment_names)

    def _heterogeneity_tree_output(self, Xtest, feature_index, *,
                                   max_depth=3, min_samples_leaf=2, min_impurity_decrease=1e-4,
                                   include_model_uncertainty=False, alpha=0.05):
        """
        Get an effect heterogeneity tree expressed as a dictionary.

        Parameters
        ----------
        X : array_like
            Features
        feature_index
            Index of the feature to be considered as treament
        max_depth : int, default 3
            maximum depth of the tree
        min_samples_leaf : int, default 2
            minimum number of samples on each leaf
        min_impurity_decrease : float, default 1e-4
            The minimum decrease in the impurity/uniformity of the causal effect that a split needs to
            achieve to construct it
        include_model_uncertainty : bool, default False
            Whether to include confidence interval information when building a simplified model of the cate model.
        alpha : float in [0, 1], default 0.05
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.
        """

        intrp, feature_names, _, _ = self._tree(False, Xtest, feature_index,
                                                max_depth=max_depth,
                                                min_samples_leaf=min_samples_leaf,
                                                min_impurity_decrease=min_impurity_decrease,
                                                include_model_uncertainty=include_model_uncertainty,
                                                alpha=alpha)

        def hetero_data(tree, node_id, node_dict):
            if include_model_uncertainty:
                return {'effect': _sanitize(tree.value[node_id]),
                        'ci': _sanitize(node_dict[node_id]['ci'])}
            else:
                return {'effect': _sanitize(tree.value[node_id])}
        return _tree_interpreter_to_dict(intrp, feature_names, hetero_data)

    def individualized_policy(self, Xtest, feature_index, *, n_rows=None, treatment_costs=0, alpha=0.05):
        """
        Get individualized treatment policy based on the learned model for a feature, sorted by the predicted effect.

        Parameters
        ----------
        Xtest: array_like
            Features
        feature_index: int or str
            Index of the feature to be considered as treatment
        n_rows: int, optional
            How many rows to return (all rows by default)
        treatment_costs: array_like, default 0
            Cost of treatment, as a scalar value or per-sample. For continuous features this is the marginal cost per
            unit of treatment; for discrete features, this is the difference in cost between each of the non-default
            values and the default value (i.e., if non-scalar the array should have shape (n,d_t-1))
        alpha: float in [0, 1], default 0.05
            Confidence level of the confidence intervals
            A (1-alpha)*100% confidence interval is returned

        Returns
        -------
        output: DataFrame
            Dataframe containing recommended treatment, effect, confidence interval, sorted by effect
        """
        result = self._safe_result_index(Xtest, feature_index)

        # get dataframe with all but selected column
        orig_df = pd.DataFrame(Xtest, columns=self.feature_names_).rename(
            columns={self.feature_names_[result.feature_index]: 'Current treatment'})

        Xtest = result.X_transformer.transform(Xtest)
        if Xtest.shape[1] == 0:
            x_rows = Xtest.shape[0]
            Xtest = None

        if result.feature_baseline is None:
            # apply 10% of a typical treatment for this feature
            effect = result.estimator.effect_inference(Xtest, T1=result.treatment_value * 0.1)
        else:
            effect = result.estimator.const_marginal_effect_inference(Xtest)

        if Xtest is None:  # we got a scalar effect although our original X may have had more rows
            effect = effect._expand_outputs(x_rows)

        multi_y = (not self._vec_y) or self.classification

        if multi_y and result.feature_baseline is not None and np.ndim(treatment_costs) == 2:
            # we've got treatment costs of shape (n, d_t-1) so we need to add a y dimension to broadcast safely
            treatment_costs = np.expand_dims(treatment_costs, 1)

        effect.translate(-treatment_costs)

        est = effect.point_estimate
        est_lb = effect.conf_int(alpha)[0]
        est_ub = effect.conf_int(alpha)[1]

        if multi_y:  # y was an array, not a vector
            est = np.squeeze(est, 1)
            est_lb = np.squeeze(est_lb, 1)
            est_ub = np.squeeze(est_ub, 1)

        if result.feature_baseline is None:
            rec = np.empty(est.shape[0], dtype=object)
            rec[est > 0] = "increase"
            rec[est <= 0] = "decrease"
            # set the effect bounds; for positive treatments these agree with
            # the estimates; for negative treatments, we need to invert the interval
            eff_lb, eff_ub = est_lb, est_ub
            eff_lb[est <= 0], eff_ub[est <= 0] = -eff_ub[est <= 0], -eff_lb[est <= 0]
            # the effect is now always positive since we decrease treatment when negative
            eff = np.abs(est)
        else:
            # for discrete treatment, stack a zero result in front for control
            zeros = np.zeros((est.shape[0], 1))
            all_effs = np.hstack([zeros, est])
            eff_ind = np.argmax(all_effs, axis=1)
            treatment_arr = np.array([result.feature_baseline] + [lvl for lvl in result.feature_levels], dtype=object)
            rec = treatment_arr[eff_ind]
            # we need to call effect_inference to get the correct CI between the two treatment options
            effect = result.estimator.effect_inference(Xtest, T0=orig_df['Current treatment'], T1=rec)
            # we now need to construct the delta in the cost between the two treatments and translate the effect
            current_treatment = orig_df['Current treatment'].values
            if isinstance(current_treatment, pd.core.arrays.categorical.Categorical):
                current_treatment = current_treatment.to_numpy()
            if np.ndim(treatment_costs) >= 2:
                # remove third dimenions potentially added
                if multi_y:  # y was an array, not a vector
                    treatment_costs = np.squeeze(treatment_costs, 1)
                assert treatment_costs.shape[1] == len(treatment_arr) - 1, ("If treatment costs are an array, "
                                                                            " they must be of shape (n, d_t-1),"
                                                                            " where n is the number of samples"
                                                                            " and d_t the number of treatment"
                                                                            " categories.")
                all_costs = np.hstack([zeros, treatment_costs])
                # find cost of current treatment: equality creates a 2d array with True on each row,
                # only if its the location of the current treatment. Then we take the corresponding cost.
                current_cost = all_costs[current_treatment.reshape(-1, 1) == treatment_arr.reshape(1, -1)]
                target_cost = np.take_along_axis(all_costs, eff_ind.reshape(-1, 1), 1).reshape(-1)
            else:
                assert isinstance(treatment_costs, (int, float)), ("Treatments costs should either be float or "
                                                                   "a 2d array of size (n, d_t-1).")
                all_costs = np.array([0] + [treatment_costs] * (len(treatment_arr) - 1))
                # construct index of current treatment
                current_ind = (current_treatment.reshape(-1, 1) ==
                               treatment_arr.reshape(1, -1)) @ np.arange(len(treatment_arr))
                current_cost = all_costs[current_ind]
                target_cost = all_costs[eff_ind]
            delta_cost = current_cost - target_cost
            # add second dimension if needed for broadcasting during translation of effect
            if multi_y:
                delta_cost = np.expand_dims(delta_cost, 1)
            effect.translate(delta_cost)
            eff = effect.point_estimate
            eff_lb, eff_ub = effect.conf_int(alpha)
            if multi_y:  # y was an array, not a vector
                eff = np.squeeze(eff, 1)
                eff_lb = np.squeeze(eff_lb, 1)
                eff_ub = np.squeeze(eff_ub, 1)

        df = pd.DataFrame({'Treatment': rec,
                           'Effect of treatment': eff,
                           'Effect of treatment lower bound': eff_lb,
                           'Effect of treatment upper bound': eff_ub},
                          index=orig_df.index)

        return df.join(orig_df).sort_values('Effect of treatment',
                                            ascending=False).head(n_rows)

    def _individualized_policy_dict(self, Xtest, feature_index, *, n_rows=None, treatment_costs=0, alpha=0.05):
        """
        Get individualized treatment policy based on the learned model for a feature, sorted by the predicted effect.

        Parameters
        ----------
        Xtest: array_like
            Features
        feature_index: int or str
            Index of the feature to be considered as treatment
        n_rows: int, optional
            How many rows to return (all rows by default)
        treatment_costs: array_like, default 0
            Cost of treatment, as a scalar value or per-sample
        alpha: float in [0, 1], default 0.05
            Confidence level of the confidence intervals
            A (1-alpha)*100% confidence interval is returned

        Returns
        -------
        output: dictionary
            dictionary containing treatment policy, effects, and other columns
        """
        return self.individualized_policy(Xtest, feature_index,
                                          n_rows=n_rows,
                                          treatment_costs=treatment_costs,
                                          alpha=alpha).to_dict('list')

    def typical_treatment_value(self, feature_index):
        """
        Get the typical treatment value used for the specified feature

        Parameters
        ----------
        feature_index: int or str
            The index of the feature to be considered as treatment

        Returns
        -------
        treatment_value : float
            The treatment value considered 'typical' for this feature
        """
        result = [res for res in self._results if res.feature_index == feature_index]
        if len(result) == 0:
            if self._has_column_names:
                result = [res for res in self._results if res.feature_name == feature_index]
                assert len(result) == 1, f"Could not find feature with index/name {feature_index}"
                return result[0].treatment_value
            else:
                raise ValueError(f"No feature with index {feature_index}")
        return result[0].treatment_value
