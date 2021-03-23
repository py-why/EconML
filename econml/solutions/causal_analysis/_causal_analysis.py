# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Module for assessing causal feature importance."""

import warnings
from collections import OrderedDict, namedtuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.utils.validation import column_or_1d
from ...cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter
from ...dml import LinearDML
from ...sklearn_extensions.linear_model import WeightedLasso
from ...sklearn_extensions.model_selection import GridSearchCVList
from ...utilities import _RegressionWrapper, inverse_onehot

# TODO: this utility is documented but internal; reimplement?
from sklearn.utils import _safe_indexing
# TODO: this utility is even less public...
from sklearn.utils import _get_column_indices


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
           ViewKey]


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


def _first_stage_reg(X, y, *, automl=True):
    if automl:
        model = GridSearchCVList([LassoCV(),
                                  RandomForestRegressor(
                                      n_estimators=100, random_state=123, min_samples_leaf=10),
                                  lgb.LGBMRegressor()],
                                 param_grid_list=[{},
                                                  {'min_weight_fraction_leaf':
                                                      [.001, .01, .1]},
                                                  {'learning_rate': [0.1, 0.3], 'max_depth': [3, 5]}],
                                 cv=2,
                                 scoring='neg_mean_squared_error')
        best_est = model.fit(X, y).best_estimator_
        if isinstance(best_est, LassoCV):
            return Lasso(alpha=best_est.alpha_)
        return best_est
    else:
        model = LassoCV(cv=5).fit(X, y)
        return Lasso(alpha=model.alpha_)


def _first_stage_clf(X, y, *, make_regressor=False, automl=True):
    if automl:
        model = GridSearchCVList([LogisticRegression(),
                                  RandomForestClassifier(
                                      n_estimators=100, random_state=123),
                                  GradientBoostingClassifier(random_state=123)],
                                 param_grid_list=[{'C': [0.01, .1, 1, 10, 100]},
                                                  {'max_depth': [3, 5],
                                                   'min_samples_leaf': [10, 50]},
                                                  {'n_estimators': [50, 100],
                                                   'max_depth': [3],
                                                   'min_samples_leaf': [10, 30]}],
                                 cv=5,
                                 scoring='neg_log_loss')
        est = model.fit(X, y).best_estimator_
    else:
        model = LogisticRegressionCV(cv=5, max_iter=1000).fit(X, y)
        est = LogisticRegression(C=model.C_[0])
    if make_regressor:
        return _RegressionWrapper(est)
    else:
        return est


def _final_stage():
    return GridSearchCVList([WeightedLasso(),
                             RandomForestRegressor(n_estimators=100, random_state=123)],
                            param_grid_list=[{'alpha': [.001, .01, .1, 1, 10]},
                                             {'max_depth': [3, 5],
                                              'min_samples_leaf': [10, 50]}],
                            cv=5,
                            scoring='neg_mean_squared_error')


# TODO: make public and move to utilities
class _ConstantFeatures(TransformerMixin):
    """
    Tranformer that ignores its input, outputting a single constant column
    """

    def __init__(self, constant=1):
        self.constant = 1

    def fit(self, _):
        return self

    def transform(self, arr):
        n, _ = arr.shape
        return np.full((n, 1), self.constant)

    def get_feature_names(self, _=None):
        return ["constant"]

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
            self.one_hot_encoder = OneHotEncoder(
                drop='first', sparse=False).fit(cat_cols)
        else:
            self.has_cats = False
        self.d_x = X.shape[1]
        return self

    def transform(self, X):
        rest = _safe_indexing(X, self.passthrough, axis=1)
        if self.has_cats:
            cats = self.one_hot_encoder.transform(
                _safe_indexing(X, self.categorical, axis=1))
            return np.hstack((cats, rest))
        else:
            return rest

    def get_feature_names(self, names=None):
        if names is None:
            names = [f"x{i}" for i in range(self.d_x)]
        rest = _safe_indexing(names, self.passthrough, axis=0)
        if self.has_cats:
            cats = self.one_hot_encoder.get_feature_names(
                _safe_indexing(names, self.categorical, axis=0))
            return np.concatenate((cats, rest))
        else:
            return rest


class CausalAnalysis:
    """
    Gets causal importance of features

    Parameters
    ----------
    feature_inds: array-like of int, str, or bool
        The features for which to estimate causal effects, expressed as either column indices,
        column names, or boolean flags indicating which columns to pick
    categorical: array-like of int, str, or bool
        The features which are categorical in nature, expressed as either column indices,
        column names, or boolean flags indicating which columns to pick
    heterogeneity_inds: array-like of int, str, or bool, or None or list of array-like elements or None, default None
        If a 1d array, then whenever estimating a heterogeneous (local) treatment effect
        model, then only the features in this array will be used for heterogeneity. If a 2d
        array then its first dimension should be len(feature_inds) and whenever estimating
        a local causal effect for target feature feature_inds[i], then only features in
        heterogeneity_inds[i] will be used for heterogeneity. If heterogeneity_inds[i]=None, then all features
        are used for heterogeneity when estimating local causal effect for feature_inds[i]. If heterogeneity_ind=None
        then all features are used for heterogeneity for all features.
    feature_names: list of str, default None
        The names for all of the features in the data.  Not necessary if the input will be a dataframe.
        If None and the input is a plain numpy array, generated feature names will be ['X1', 'X2', ...].
    upper_bound_on_cat_expansion: int, default 5
        The maximum number of categorical values allowed, because they are expanded via one-hot encoding. If a
        feature has more than this many values, then a causal effect model is not fitted for that target feature
        and a warning flag is raised. The remainder of the models are fitted.
    classification: bool, default False
        Whether this is a classification (as opposed to regression) task
        TODO. Enable also multi-class classification (post-MVP)
    nuisance_models: one of {'linear', 'automl'}, optional (default='linear')
        What models to use for nuisance estimation (i.e. for estimating propensity models or models of how
        controls predict the outcome). If 'linear', then LassoCV (for regression) and LogisticRegressionCV
        (for classification) are used. If 'automl', then a kfold cross-validation and model selection is performed
        among several models and the best is chosen.
        TODO. Add other options, such as {'azure_automl', 'forests', 'boosting'} that will use particular sub-cases
        of models or also integrate with azure autoML. (post-MVP)
    heterogeneity_model: one of {'linear'}, optional (default='linear')
        What type of model to use for treatment effect heterogeneity. 'linear' means that a heterogeneity model
        of the form theta(X)=<a, X> will be used.
        TODO. Add other options, such as {'forest'}, for the use of a causal forest, or {'automl'} for performing
        model selection for the causal effect, or {'sparse_linear'} for using a debiased lasso. (post-MVP)
    automl: bool, default True
        Whether to automatically perform model selection over a variety of models
    n_jobs: int, default -1
        Degree of parallelism to use when training models via joblib.Parallel
    """

    _results = namedtuple("_results", field_names=[
        "feature_index", "feature_name", "feature_baseline", "feature_levels", "hinds",
        "X_transformer", "W_transformer", "estimator", "global_inference", "d_t"])

    def __init__(self, feature_inds, categorical, heterogeneity_inds=None, feature_names=None, classification=False,
                 upper_bound_on_cat_expansion=5, nuisance_models='linear', heterogeneity_model='linear', n_jobs=-1):
        self.feature_inds = feature_inds
        self.categorical = categorical
        self.heterogeneity_inds = heterogeneity_inds
        self.feature_names = feature_names
        self.classification = classification
        self.upper_bound_on_cat_expansion = upper_bound_on_cat_expansion
        self.nuisance_models = nuisance_models
        self.heterogeneity_model = heterogeneity_model
        self.n_jobs = n_jobs

    def fit(self, X, y, warm_start=False):
        """
        Fits global and local causal effect models for each feature in feature_inds on the data

        Parameters
        ----------
        X : array-like
            Feature data
        y : array-like of shape (n,) or (n,1)
            Outcome. If classification=True, then y should take two values. Otherwise an error is raised
            that only binary classification is implemented for now.
            TODO. enable multi-class classification for y (post-MVP)
        warm_start : boolean, default False
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

        assert self.heterogeneity_model in ['linear'], ("The only supported heterogeneity model is 'linear', "
                                                        f"but was given {self.heterogeneity_models}")

        assert np.ndim(X) == 2, f"X must be a 2-dimensional array, but here had shape {np.shape(X)}"

        if warm_start:
            if not hasattr(self, "_results"):
                raise ValueError(
                    "This object has not been fit, so warm_start does not apply")

            if self._d_x != X.shape[1]:
                raise ValueError(
                    f"Can't warm start: previous X had {self._d_x} columns, new X has {X.shape[1]} columns")

        # TODO: implement check for upper bound on categoricals

        # work with numeric feature indices, so that we can easily compare with categorical ones
        train_inds = _get_column_indices(X, self.feature_inds)

        heterogeneity_inds = self.heterogeneity_inds
        if heterogeneity_inds is None:
            heterogeneity_inds = [None for ind in train_inds]

        # if heterogeneity_inds is 1D, repeat it
        if heterogeneity_inds == [] or isinstance(heterogeneity_inds[0], (int, str, bool)):
            heterogeneity_inds = [heterogeneity_inds for _ in train_inds]

        # replace None elements of heterogeneity_inds and ensure indices are numeric
        heterogeneity_inds = {ind: list(range(X.shape[1])) if hinds is None else _get_column_indices(X, hinds)
                              for ind, hinds in zip(train_inds, heterogeneity_inds)}

        if warm_start and self.nuisance_models != self.nuisance_models_:
            warnings.warn("warm_start will be ignored since the nuisance models have changed"
                          f" from {self.nuisance_models_} to {self.nuisance_models} since the previous call to fit")
            new_inds = train_inds
        elif warm_start:
            new_inds = [ind for ind in train_inds if (ind not in self._cache or
                                                      heterogeneity_inds[ind] != self._cache[ind][1].hinds)]
        else:
            new_inds = train_inds

            self._cache = {}  # store mapping from feature to insights, results

            # train the Y model

            # perform model selection for the Y model using all X, not on a per-column basis
            self._x_transform = ColumnTransformer([('encode',
                                                    OneHotEncoder(
                                                        drop='first', sparse=False),
                                                    self.categorical)],
                                                  remainder='passthrough')
            allX = self._x_transform.fit_transform(X)

            if self.classification:
                self._model_y = _first_stage_clf(
                    allX, y, automl=self.nuisance_models == 'automl', make_regressor=True)
                # now that we've trained the classifier and wrapped it, ensure that y is transformed to
                # work with the regression wrapper
                # we use column_or_1d to treat pd.Series and pd.DataFrame objects the same way as arrays
                y = column_or_1d(y).reshape(-1, 1)
                y = OneHotEncoder(drop='first', sparse=False).fit_transform(y)
            else:
                self._model_y = _first_stage_reg(allX, y, automl=self.nuisance_models == 'automl')

        assert y.ndim == 1 or y.shape[1] == 1, ("Multiclass classification isn't supported" if self.classification
                                                else "Only a single outcome is supported")

        self._vec_y = y.ndim == 1
        self._d_x = X.shape[1]

        # start with empty results and default shared insights
        self._results = []
        self._shared = _get_default_shared_insights_output()

        # convert categorical indicators to numeric indices
        categorical_inds = _get_column_indices(X, self.categorical)

        def process_feature(name, feat_ind):
            discrete_treatment = feat_ind in categorical_inds
            hinds = heterogeneity_inds[feat_ind]
            WX_transformer = ColumnTransformer([('encode', OneHotEncoder(drop='first', sparse=False),
                                                 [ind for ind in categorical_inds
                                                  if ind != feat_ind]),
                                                ('drop', 'drop', feat_ind)],
                                               remainder='passthrough')
            W_transformer = ColumnTransformer([('encode', OneHotEncoder(drop='first', sparse=False),
                                                [ind for ind in categorical_inds
                                                 if ind != feat_ind and ind not in hinds]),
                                               ('drop', 'drop', hinds),
                                               ('drop_feat', 'drop', feat_ind)],
                                              remainder='passthrough')
            # Use _ColumnTransformer instead of ColumnTransformer so we can get feature names
            X_transformer = _ColumnTransformer([ind for ind in categorical_inds
                                                if ind != feat_ind and ind in hinds],
                                               [ind for ind in hinds
                                                if ind != feat_ind and ind not in categorical_inds])

            # Controls are all other columns of X
            WX = WX_transformer.fit_transform(X)
            # can't use X[:, feat_ind] when X is a DataFrame
            T = _safe_indexing(X, feat_ind, axis=1)

            # perform model selection
            model_t = (_first_stage_clf(WX, T, automl=self.nuisance_models == 'automl')
                       if discrete_treatment else _first_stage_reg(WX, T, automl=self.nuisance_models == 'automl'))

            # For the global model, use a constant featurizer to fit the ATE
            # Ideally, this could be PolynomialFeatures(degree=0, include_bias=True), but degree=0 is unsupported
            # So instead we'll use our own class
            featurizer = _ConstantFeatures()

            # TODO: support other types of heterogeneity via an initializer arg
            #       e.g. 'forest' -> ForestDML
            est = LinearDML(model_y=self._model_y,
                            model_t=model_t,
                            featurizer=featurizer,
                            discrete_treatment=discrete_treatment,
                            fit_cate_intercept=False,
                            linear_first_stages=False,
                            random_state=123)
            W = W_transformer.fit_transform(X)
            X_xf = X_transformer.fit_transform(X)
            if W.shape[1] == 0:
                # array checking routines don't accept 0-width arrays
                W = None
            est.fit(y, T, X=X_xf, W=W, cache_values=True)

            # effect doesn't depend on W, so only pass in first row
            global_inference = est.const_marginal_effect_inference(X=X_xf[0:1])
            # For the local model, change the featurizer to include X
            est.featurizer = PolynomialFeatures(degree=1, include_bias=True)
            est.refit_final()

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
            result = CausalAnalysis._results(feature_index=feat_ind,
                                             feature_name=name,
                                             feature_baseline=baseline,
                                             feature_levels=cats,
                                             hinds=hinds,
                                             X_transformer=X_transformer,
                                             W_transformer=W_transformer,
                                             estimator=est,
                                             d_t=d_t,
                                             global_inference=global_inference)

            return insights, result

        if self.feature_names is None:
            if hasattr(X, "iloc"):
                feature_names = X.columns
            else:
                feature_names = [f"x{i}" for i in range(X.shape[1])]

        self.feature_names_ = feature_names

        # extract subset matching new columns
        feature_names = _safe_indexing(feature_names, new_inds)

        cache_updates = dict(zip(new_inds,
                                 joblib.Parallel(n_jobs=self.n_jobs,
                                                 verbose=1)(joblib.delayed(process_feature)(feat_name, feat_ind)
                                                            for feat_name, feat_ind in zip(feature_names, new_inds))))

        self._cache.update(cache_updates)

        for ind in train_inds:
            dict_update, result = self._cache[ind]
            self._results.append(result)
            for k in dict_update:
                self._shared[k] += dict_update[k]

        self.nuisance_models_ = self.nuisance_models
        self.heterogeneity_model_ = self.heterogeneity_model
        return self

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

    def _summarize(self, *, summary, get_inference, props, n, expand_arr, drop_sample):

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

        # each attr has dimension (m,y) or (m,y,t)
        def coalesce(attr):
            """Join together the arrays for each feature"""
            if isinstance(attr, str):
                s = attr

                def attr(o):
                    val = getattr(o, s)
                    if callable(val):
                        return val()
                    else:
                        return val

            # concatenate along treatment dimension
            arr = np.concatenate([ensure_proper_dims(attr(get_inference(res)))
                                  for res in self._results], axis=2)

            # for dictionary representation, want to remove unneeded sample dimension
            # in cohort and global results
            if drop_sample:
                arr = np.squeeze(arr, 0)

            return arr

        return summary([(key, coalesce(val)) for key, val in props])

    def _pandas_summary(self, get_inference, props, n,
                        expand_arr=False):
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
            for lvl in index.levels:
                if len(lvl) == 1:
                    index = index.droplevel(lvl.name)
            return pd.DataFrame(to_include, index=index)

        return self._summarize(summary=make_dataframe,
                               get_inference=get_inference,
                               props=props,
                               n=n,
                               expand_arr=expand_arr,
                               drop_sample=False)  # dropping the sample dimension is handled above instead

    def _dict_summary(self, get_inference, *, props, n, kind, drop_sample=False, expand_arr=False):
        def make_dict(props):
            # should be serialization-ready and contain no numpy arrays
            res = _get_default_specific_insights(kind)
            res.update([(key, value.tolist()) for key, value in props])
            return {**self._shared, **res}

        return self._summarize(summary=make_dict,
                               get_inference=get_inference,
                               props=props,
                               n=n,
                               expand_arr=expand_arr,
                               drop_sample=drop_sample)

    def global_causal_effect(self, alpha=0.05):
        """
        Get the global causal effect for each feature as a pandas DataFrame.

        Parameters
        ----------
        alpha : the confidence level of the confidence interval

        Returns
        -------
        global_effects : pandas Dataframe
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
        return self._pandas_summary(lambda res: res.global_inference, props=self._point_props(alpha), n=1)

    def _global_causal_effect_dict(self, alpha=0.05):
        """
        Gets the global causal effect for each feature as dictionary.

        Dictionary entries for predictions, etc. will be nested lists of shape (d_y, sum(d_t))

        Only for serialization purposes to upload to AzureML
        """
        return self._dict_summary(lambda res: res.global_inference, props=self._point_props(alpha),
                                  n=1, kind='global', drop_sample=True)

    def _cohort_effect_inference(self, Xtest):
        assert np.ndim(Xtest) == 2 and np.shape(Xtest)[1] == self._d_x, (
            "Shape of Xtest must be compatible with shape of X, "
            f"but got shape {np.shape(Xtest)} instead of (n, {self._d_x})"
        )

        def inference_from_result(result):
            est = result.estimator
            X = result.X_transformer.transform(Xtest)
            return est.const_marginal_ate_inference(X=X)
        return inference_from_result

    def cohort_causal_effect(self, Xtest, alpha=0.05):
        """
        Gets the average causal effects for a particular cohort defined by a population of X's.

        Parameters
        ----------
        Xtest : the cohort samples for which to return the average causal effects within cohort
        alpha : the confidence level of the confidence interval

        Returns
        -------
        cohort_effects : pandas Dataframe
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
                                    expand_arr=True)

    def _cohort_causal_effect_dict(self, Xtest, alpha=0.05):
        """
        Gets the cohort causal effects for each feature as dictionary.

        Dictionary entries for predictions, etc. will be nested lists of shape (d_y, sum(d_t))

        Only for serialization purposes to upload to AzureML
        """
        return self._dict_summary(self._cohort_effect_inference(Xtest), props=self._summary_props(alpha),
                                  n=1, kind='cohort', expand_arr=True, drop_sample=True)

    def _local_effect_inference(self, Xtest):
        assert np.ndim(Xtest) == 2 and np.shape(Xtest)[1] == self._d_x, (
            "Shape of Xtest must be compatible with shape of X, "
            f"but got shape {np.shape(Xtest)} instead of (n, {self._d_x})"
        )

        def inference_from_result(result):
            est = result.estimator
            X = result.X_transformer.transform(Xtest)
            return est.const_marginal_effect_inference(X=X)
        return inference_from_result

    def local_causal_effect(self, Xtest, alpha=0.05):
        """
        Gets the local causal effect for each feature as a pandas DataFrame.

        Parameters
        ----------
        Xtest : the samples for which to return the causal effects
        alpha : the confidence level of the confidence interval

        Returns
        -------
        global_effect : pandas Dataframe
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
                                    props=self._point_props(alpha), n=Xtest.shape[0])

    def _local_causal_effect_dict(self, Xtest, alpha=0.05):
        """
        Gets the local feature importance as dictionary

        Dictionary entries for predictions, etc. will be nested lists of shape (n_rows, d_y, sum(d_t))

        Only for serialization purposes to upload to AzureML
        """
        return self._dict_summary(self._local_effect_inference(Xtest), props=self._point_props(alpha),
                                  kind='local', n=Xtest.shape[0])

    def _check_feature_index(self, X, feature_index):
        assert hasattr(self, "_results"), "This instance has not yet been fitted"

        assert np.ndim(X) == 2 and np.shape(X)[1] == self._d_x, (
            "Shape of X must be compatible with shape of the fitted X, "
            f"but got shape {np.shape(X)} instead of (n, {self._d_x})"
        )

        (numeric_index,) = _get_column_indices(X, [feature_index])
        results = [res for res in self._results
                   if res.feature_index == numeric_index]

        assert len(results) != 0, f"The feature index supplied was not fitted"
        (result,) = results
        return result

    def whatif(self, X, Xnew, feature_index, y):
        """
        Get counterfactual predictions when feature_index is changed to Xnew from its observational counterpart.

        Note that this only applies to regression use cases; for classification what-if analysis is not supported.

        Parameters
        ----------
        X: array-like
            Features
        Xnew: array-like
            New values of a single column of X
        feature_index: int or string
            The index of the feature being varied to Xnew, either as a numeric index or
            the string name if the input is a dataframe
        y: array-like
            Observed labels or outcome of a predictive model for baseline y values
        """

        assert not self.classification, "What-if analysis cannot be applied to classification tasks"

        assert np.shape(X)[0] == np.shape(Xnew)[0] == np.shape(y)[0], (
            "X, Xnew, and y must have the same length, but have shapes "
            f"{np.shape(X)}, {np.shape(Xnew)}, and {np.shape(y)}"
        )

        assert np.size(feature_index) == 1, f"Only one feature index may be changed, but got {np.size(feature_index)}"

        T0 = _safe_indexing(X, feature_index, axis=1)
        T1 = Xnew
        result = self._check_feature_index(X, feature_index)
        inf = result.estimator.effect_inference(
            X=result.X_transformer.transform(X), T0=T0, T1=T1)

        # we want to offset the inference object by the baseline estimate of y
        inf.translate(y)

        return inf

    def policy_tree(self, Xtest, feature_index, *, treatment_cost=0,
                    max_depth=3, min_samples_leaf=2, min_value_increase=1e-4, alpha=.1):
        """
        Get a recommended policy tree in graphviz format.

        Parameters
        ----------
        X : array-like
            Features
        feature_index
            Index of the feature to be considered as treament
        treatment_cost : int, or array-like of same length as number of rows of X, optional (default=0)
            Cost of treatment, or cost of treatment for each sample
        max_depth : int, optional (default=3)
            maximum depth of the tree
        min_samples_leaf : int, optional (default=2)
            minimum number of samples on each leaf
        min_value_increase : float, optional (default=1e-4)
            The minimum increase in the policy value that a split needs to create to construct it
        alpha : float in [0, 1], optional (default=.1)
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.
        """

        result = self._check_feature_index(Xtest, feature_index)
        Xtest = result.X_transformer.transform(Xtest)
        intrp = SingleTreePolicyInterpreter(include_model_uncertainty=True,
                                            uncertainty_level=alpha,
                                            max_depth=max_depth,
                                            min_samples_leaf=min_samples_leaf,
                                            min_impurity_decrease=min_value_increase)
        if result.feature_baseline is None:
            treatment_names = ['low', 'high']
        else:
            treatment_names = [f"{result.feature_baseline}"] + \
                [f"{lvl}" for lvl in result.feature_levels]
        intrp.interpret(result.estimator, Xtest,
                        sample_treatment_costs=treatment_cost)
        return intrp.export_graphviz(feature_names=result.X_transformer.get_feature_names(self.feature_names_),
                                     treatment_names=treatment_names)

    def heterogeneity_tree(self, Xtest, feature_index, *,
                           max_depth=3, min_samples_leaf=2, min_impurity_decrease=1e-4,
                           alpha=.1):
        """
        Get an effect hetergoeneity tree in graphviz format.

        Parameters
        ----------
        X : array-like
            Features
        feature_index
            Index of the feature to be considered as treament
        max_depth : int, optional (default=3)
            maximum depth of the tree
        min_samples_leaf : int, optional (default=2)
            minimum number of samples on each leaf
        min_impurity_decrease : float, optional (default=1e-4)
            The minimum decrease in the impurity/uniformity of the causal effect that a split needs to
            achieve to construct it
        alpha : float in [0, 1], optional (default=.1)
            Confidence level of the confidence intervals displayed in the leaf nodes.
            A (1-alpha)*100% confidence interval is displayed.
        """

        result = self._check_feature_index(Xtest, feature_index)
        Xtest = result.X_transformer.transform(Xtest)
        intrp = SingleTreeCateInterpreter(include_model_uncertainty=True,
                                          uncertainty_level=alpha,
                                          max_depth=max_depth,
                                          min_samples_leaf=min_samples_leaf,
                                          min_impurity_decrease=min_impurity_decrease)
        intrp.interpret(result.estimator, Xtest)
        return intrp.export_graphviz()

    @property
    def cate_models_(self):
        return [result.estimator for result in self._results]

    @property
    def X_transformers_(self):
        return [result.X_transformer for result in self._results]
