import numpy as np
from .dml import _BaseDML
from .dml import _FirstStageWrapper, _FinalWrapper
from ..sklearn_extensions.linear_model import WeightedLassoCVWrapper
from ..sklearn_extensions.model_selection import WeightedStratifiedKFold
from ..inference import Inference, NormalInferenceResults
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from ..utilities import add_intercept, shape, check_inputs
from ..grf import CausalForest, MultiOutputGRF


class _CausalForestFinalWrapper(_FinalWrapper):

    def _combine(self, X, fitting=True):
        if X is not None:
            if self._featurizer is not None:
                F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
            else:
                F = X
        else:
            raise AttributeError("Cannot use this method with X=None. Consider "
                                 "using the LinearDML estimator.")
        return F

    def fit(self, X, T_res, Y_res, sample_weight=None, sample_var=None):
        # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T_res)[1:]
        self._d_y = shape(Y_res)[1:]
        fts = self._combine(X)
        if sample_var is not None:
            raise ValueError("This estimator does not support sample_var!")
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        self._model.fit(fts, T_res, Y_res, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._model.predict(self._combine(X, fitting=False)).reshape((-1,) + self._d_y + self._d_t)


class GenericSingleOutcomeModelFinalWithCovInference(Inference):

    def prefit(self, estimator, *args, **kwargs):
        self.model_final = estimator.model_final
        self.featurizer = estimator.featurizer if hasattr(estimator, 'featurizer') else None

    def fit(self, estimator, *args, **kwargs):
        # once the estimator has been fit, it's kosher to store d_t here
        # (which needs to have been expanded if there's a discrete treatment)
        self._est = estimator
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self.d_t = self._d_t[0] if self._d_t else 1
        self.d_y = self._d_y[0] if self._d_y else 1

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
        return self.const_marginal_effect_inference(X).conf_int(alpha=alpha)

    def const_marginal_effect_inference(self, X):
        if X is None:
            raise ValueError("This inference method currently does not support X=None!")
        if self.featurizer is not None:
            X = self.featurizer.transform(X)
        pred, pred_var = self.model_final.predict_and_var(X)
        pred = pred.reshape((-1,) + self._d_y + self._d_t)
        pred_stderr = np.sqrt(np.diagonal(pred_var, axis1=2, axis2=3).reshape((-1,) + self._d_y + self._d_t))
        return NormalInferenceResults(d_t=self.d_t, d_y=self.d_y, pred=pred,
                                      pred_stderr=pred_stderr, inf_type='effect')

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
        return self.effect_inference(X, T0=T0, T1=T1).conf_int(alpha=alpha)

    def effect_inference(self, X, *, T0, T1):
        if X is None:
            raise ValueError("This inference method currently does not support X=None!")
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if self.featurizer is not None:
            X = self.featurizer.transform(X)
        dT = T1 - T0
        if dT.ndim == 1:
            dT = dT.reshape((-1, 1))
        pred, pred_var = self.model_final.predict_projection_and_var(X, dT)
        pred = pred.reshape((-1,) + self._d_y)
        pred_stderr = np.sqrt(pred_var.reshape((-1,) + self._d_y))
        return NormalInferenceResults(d_t=1, d_y=self.d_y, pred=pred,
                                      pred_stderr=pred_stderr, inf_type='effect')


class CausalForestDML(_BaseDML):

    def __init__(self, *,
                 model_y='auto',
                 model_t='auto',
                 featurizer=None,
                 linear_first_stages=False,
                 discrete_treatment=False,
                 categories='auto',
                 n_crossfit_splits=2,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=True,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 inference=True,
                 fit_intercept=True,
                 subforest_size=4,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):

        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies
        if model_y == 'auto':
            model_y = WeightedLassoCVWrapper(random_state=random_state)
        if model_t == 'auto':
            if discrete_treatment:
                model_t = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=random_state),
                                               random_state=random_state)
            else:
                model_t = WeightedLassoCVWrapper(random_state=random_state)
        self.bias_part_of_coef = False
        self.fit_cate_intercept = False
        model_final = MultiOutputGRF(CausalForest(n_estimators=n_estimators,
                                                  criterion=criterion,
                                                  max_depth=max_depth,
                                                  min_samples_split=min_samples_split,
                                                  min_samples_leaf=min_samples_leaf,
                                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
                                                  min_var_fraction_leaf=min_var_fraction_leaf,
                                                  min_var_leaf_on_val=min_var_leaf_on_val,
                                                  max_features=max_features,
                                                  min_impurity_decrease=min_impurity_decrease,
                                                  max_samples=max_samples,
                                                  min_balancedness_tol=min_balancedness_tol,
                                                  honest=honest,
                                                  inference=inference,
                                                  fit_intercept=fit_intercept,
                                                  subforest_size=subforest_size,
                                                  n_jobs=n_jobs,
                                                  random_state=random_state,
                                                  verbose=verbose,
                                                  warm_start=warm_start))
        super().__init__(model_y=_FirstStageWrapper(model_y, True,
                                                    featurizer, linear_first_stages, discrete_treatment),
                         model_t=_FirstStageWrapper(model_t, False,
                                                    featurizer, linear_first_stages, discrete_treatment),
                         model_final=_CausalForestFinalWrapper(model_final, False, featurizer, False),
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         n_splits=n_crossfit_splits,
                         random_state=random_state)

    def _get_inference_options(self):
        options = super()._get_inference_options()
        options.update(blb=GenericSingleOutcomeModelFinalWithCovInference)
        options.update(auto=GenericSingleOutcomeModelFinalWithCovInference)
        return options

    # override only so that we can update the docstring to indicate support for `blb`
    def fit(self, Y, T, *, X, W=None, sample_weight=None, sample_var=None, groups=None, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: (n × dₓ) matrix
            Features for each sample
        W: optional (n × d_w) matrix
            Controls for each sample
        sample_weight: optional (n,) vector
            Weights for each row
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) 'blb' 
            (or an instance of :class:`.GenericModelFinalWithCovInference`) and 'auto'
            (or an instance of :class:`.GenericModelFinalWithCovInference`)

        Returns
        -------
        self
        """
        if sample_var is not None:
            raise ValueError("This estimator does not support sample_var!")
        if X is None:
            raise ValueError("This estimator does not support X=None!")
        Y, T, X, W = check_inputs(Y, T, X, W=W, multi_output_T=True, multi_output_Y=True)
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           inference=inference)

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        imps = self.model_final.feature_importances(max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
        return imps.reshape(self._d_y + (-1,))

    @property
    def feature_importances_(self):
        return self.feature_importances()
