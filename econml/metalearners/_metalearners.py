# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Metalearners for heterogeneous treatment effects in the context of discrete treatments.

For more details on these CATE methods, see `<https://arxiv.org/abs/1706.03461>`_
(Künzel S., Sekhon J., Bickel P., Yu B.) on Arxiv.
"""

import numpy as np
import warnings
from .._cate_estimator import BaseCateEstimator, LinearCateEstimator, TreatmentExpansionMixin
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array, check_X_y
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from ..utilities import (check_inputs, check_models, broadcast_unit_treatments, reshape_treatmentwise_effects,
                         one_hot_encoder, inverse_onehot, transpose, _deprecate_positional)
from .._shap import _shap_explain_model_cate


class TLearner(TreatmentExpansionMixin, LinearCateEstimator):
    """Conditional mean regression estimator.

    Parameters
    ----------
    models : outcome estimators for both control units and treatment units
        It can be a single estimator applied to all the control and treatment units or a tuple/list of
        estimators with one estimator per treatment (including control).
        Must implement `fit` and `predict` methods.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    allow_missing: bool
        Whether to allow missing values in X. If True, will need to supply models
        that can handle missing values.

    Examples
    --------
    A simple example:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.metalearners import TLearner
        from sklearn.linear_model import LinearRegression

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = TLearner(models=LinearRegression())
        est.fit(y, T, X=X)

    >>> est.effect(X[:3])
    array([0.58547..., 1.82860..., 0.78379...])
    """

    def __init__(self, *,
                 models,
                 categories='auto',
                 allow_missing=False):
        self.models = clone(models, safe=False)
        self.categories = categories
        self.allow_missing = allow_missing
        super().__init__()

    def _gen_allowed_missing_vars(self):
        return ['X'] if self.allow_missing else []

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, *, X, inference=None):
        """Build an instance of TLearner.

        Parameters
        ----------
        Y : array_like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array_like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        inference : str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`)

        Returns
        -------
        self : an instance of self.

        """

        # Check inputs
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False,
                                  force_all_finite_X='allow-nan' if 'X' in self._gen_allowed_missing_vars() else True)
        categories = self.categories
        if categories != 'auto':
            categories = [categories]  # OneHotEncoder expects a 2D array with features per column
        self.transformer = one_hot_encoder(categories=categories, drop='first')
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = T.shape[1:]
        T = inverse_onehot(T)
        self.models = check_models(self.models, self._d_t[0] + 1)

        for ind in range(self._d_t[0] + 1):
            self.models[ind].fit(X[T == ind], Y[T == ind])

    def const_marginal_effect(self, X):
        """Calculate the constant marignal treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × d_x)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : matrix, shape (m, d_y, d_t)
            Constant marginal CATE of each treatment on each outcome for each sample X[i].
            Note that when Y is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
        """
        # Check inputs
        X = check_array(X)
        taus = []
        for ind in range(self._d_t[0]):
            taus.append(self.models[ind + 1].predict(X) - self.models[0].predict(X))
        taus = np.column_stack(taus).reshape((-1,) + self._d_t + self._d_y)  # shape as of m*d_t*d_y
        if self._d_y:
            taus = transpose(taus, (0, 2, 1))  # shape as of m*d_y*d_t
        return taus


class SLearner(TreatmentExpansionMixin, LinearCateEstimator):
    """Conditional mean regression estimator where the treatment assignment is taken as a feature in the ML model.

    Parameters
    ----------
    overall_model : outcome estimator for all units
        Model will be trained on X|T where '|' denotes concatenation.
        Must implement `fit` and `predict` methods.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    allow_missing: bool
        Whether to allow missing values in X. If True, will need to supply overall_model
        that can handle missing values.

    Examples
    --------
    A simple example:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.metalearners import SLearner
        from sklearn.ensemble import RandomForestRegressor

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = SLearner(overall_model=RandomForestRegressor())
        est.fit(y, T, X=X)

    >>> est.effect(X[:3])
    array([0.23577..., 1.62784... , 0.45946...])
    """

    def __init__(self, *,
                 overall_model,
                 categories='auto',
                 allow_missing=False):
        self.overall_model = clone(overall_model, safe=False)
        self.categories = categories
        self.allow_missing = allow_missing
        super().__init__()

    def _gen_allowed_missing_vars(self):
        return ['X'] if self.allow_missing else []

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, *, X=None, inference=None):
        """Build an instance of SLearner.

        Parameters
        ----------
        Y : array_like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array_like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array_like, shape (n, d_x), optional
            Feature vector that captures heterogeneity.

        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`)

        Returns
        -------
        self : an instance of self.
        """
        # Check inputs
        if X is None:
            X = np.zeros((Y.shape[0], 1))
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False,
                                  force_all_finite_X='allow-nan' if 'X' in self._gen_allowed_missing_vars() else True)
        categories = self.categories
        if categories != 'auto':
            categories = [categories]  # OneHotEncoder expects a 2D array with features per column
        self.transformer = one_hot_encoder(categories=categories, drop='first')
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = (T.shape[1], )
        # Note: unlike other Metalearners, we need the controls' encoded column for training
        # Thus, we append the controls column before the one-hot-encoded T
        # We might want to revisit, though, since it's linearly determined by the others
        feat_arr = np.concatenate((X, 1 - np.sum(T, axis=1).reshape(-1, 1), T), axis=1)
        self.overall_model.fit(feat_arr, Y)

    def const_marginal_effect(self, X=None):
        """Calculate the constant marginal treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × dₓ), optional
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : matrix, shape (m, d_y, d_t)
            Constant marginal CATE of each treatment on each outcome for each sample X[i].
            Note that when Y is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
        """
        # Check inputs
        if X is None:
            X = np.zeros((1, 1))
        X = check_array(X)
        Xs, Ts = broadcast_unit_treatments(X, self._d_t[0] + 1)
        feat_arr = np.concatenate((Xs, Ts), axis=1)
        prediction = self.overall_model.predict(feat_arr).reshape((-1, self._d_t[0] + 1,) + self._d_y)
        if self._d_y:
            prediction = transpose(prediction, (0, 2, 1))
            taus = (prediction - np.repeat(prediction[:, :, 0], self._d_t[0] + 1).reshape(prediction.shape))[:, :, 1:]
        else:
            taus = (prediction - np.repeat(prediction[:, 0], self._d_t[0] + 1).reshape(prediction.shape))[:, 1:]
        return taus


class XLearner(TreatmentExpansionMixin, LinearCateEstimator):
    """Meta-algorithm proposed by Kunzel et al. that performs best in settings
       where the number of units in one treatment arm is much larger than others.

    Parameters
    ----------
    models : outcome estimators for both control units and treatment units
        It can be a single estimator applied to all the control and treatment units or a tuple/list of
        estimators with one estimator per treatment (including control).
        Must implement `fit` and `predict` methods.

    cate_models : estimator for pseudo-treatment effects on control and treatments
        It can be a single estimator applied to all the control and treatments or a tuple/list of
        estimators with one estimator per treatment (including control).
        If None, it will be same models as the outcome estimators.
        Must implement `fit` and `predict` methods.

    propensity_model : estimator for the propensity function
        Must implement `fit` and `predict_proba` methods. The `fit` method must
        be able to accept X and T, where T is a shape (n, ) array.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    allow_missing: bool
        Whether to allow missing values in X. If True, will need to supply models, cate_models, and
        propensity_model that can handle missing values.

    Examples
    --------
    A simple example:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.metalearners import XLearner
        from sklearn.linear_model import LinearRegression

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = XLearner(models=LinearRegression())
        est.fit(y, T, X=X)

    >>> est.effect(X[:3])
    array([0.58547..., 1.82860..., 0.78379...])
    """

    def __init__(self, *,
                 models,
                 cate_models=None,
                 propensity_model=LogisticRegression(),
                 categories='auto',
                 allow_missing=False):
        self.models = clone(models, safe=False)
        self.cate_models = clone(cate_models, safe=False)
        self.propensity_model = clone(propensity_model, safe=False)
        self.categories = categories
        self.allow_missing = allow_missing
        super().__init__()

    def _gen_allowed_missing_vars(self):
        return ['X'] if self.allow_missing else []

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, *, X, inference=None):
        """Build an instance of XLearner.

        Parameters
        ----------
        Y : array_like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array_like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        inference : str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`)

        Returns
        -------
        self : an instance of self.
        """
        # Check inputs
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False,
                                  force_all_finite_X='allow-nan' if 'X' in self._gen_allowed_missing_vars() else True)
        if Y.ndim == 2 and Y.shape[1] == 1:
            Y = Y.flatten()
        categories = self.categories
        if categories != 'auto':
            categories = [categories]  # OneHotEncoder expects a 2D array with features per column
        self.transformer = one_hot_encoder(categories=categories, drop='first')
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = T.shape[1:]
        T = inverse_onehot(T)
        self.models = check_models(self.models, self._d_t[0] + 1)
        if self.cate_models is None:
            self.cate_models = [clone(model, safe=False) for model in self.models]
        else:
            self.cate_models = check_models(self.cate_models, self._d_t[0] + 1)
        self.propensity_models = []
        self.cate_treated_models = []
        self.cate_controls_models = []

        # Estimate response function
        for ind in range(self._d_t[0] + 1):
            self.models[ind].fit(X[T == ind], Y[T == ind])
        for ind in range(self._d_t[0]):
            self.cate_treated_models.append(clone(self.cate_models[ind + 1], safe=False))
            self.cate_controls_models.append(clone(self.cate_models[0], safe=False))
            self.propensity_models.append(clone(self.propensity_model, safe=False))
            imputed_effect_on_controls = self.models[ind + 1].predict(X[T == 0]) - Y[T == 0]
            imputed_effect_on_treated = Y[T == ind + 1] - self.models[0].predict(X[T == ind + 1])
            self.cate_controls_models[ind].fit(X[T == 0], imputed_effect_on_controls)
            self.cate_treated_models[ind].fit(X[T == ind + 1], imputed_effect_on_treated)
            X_concat = np.concatenate((X[T == 0], X[T == ind + 1]), axis=0)
            T_concat = np.concatenate((T[T == 0], T[T == ind + 1]), axis=0)
            self.propensity_models[ind].fit(X_concat, T_concat)

    def const_marginal_effect(self, X):
        """Calculate the constant marginal treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : matrix, shape (m, d_y, d_t)
            Constant marginal CATE of each treatment on each outcome for each sample X[i].
            Note that when Y is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
        """
        X = check_array(X)
        m = X.shape[0]
        taus = []
        for ind in range(self._d_t[0]):
            propensity_scores = self.propensity_models[ind].predict_proba(X)[:, 1:]
            tau_hat = propensity_scores * self.cate_controls_models[ind].predict(X).reshape(m, -1) \
                + (1 - propensity_scores) * self.cate_treated_models[ind].predict(X).reshape(m, -1)
            taus.append(tau_hat)
        taus = np.column_stack(taus).reshape((-1,) + self._d_t + self._d_y)  # shape as of m*d_t*d_y
        if self._d_y:
            taus = transpose(taus, (0, 2, 1))  # shape as of m*d_y*d_t
        return taus


class DomainAdaptationLearner(TreatmentExpansionMixin, LinearCateEstimator):
    """Meta-algorithm that uses domain adaptation techniques to account for
       covariate shift (selection bias) among the treatment arms.

    Parameters
    ----------
    models : outcome estimators for both control units and treatment units
        It can be a single estimator applied to all the control and treatment units or a tuple/list of
        estimators with one estimator per treatment (including control).
        Must implement `fit` and `predict` methods.
        The `fit` method must accept the `sample_weight` parameter.

    final_models : estimators for pseudo-treatment effects for each treatment
        It can be a single estimator applied to all the control and treatment units or a tuple/list of
        estimators with ones estimator per treatments (excluding control).
        Must implement `fit` and `predict` methods.

    propensity_model : estimator for the propensity function
        Must implement `fit` and `predict_proba` methods. The `fit` method must
        be able to accept X and T, where T is a shape (n, 1) array.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    allow_missing: bool
        Whether to allow missing values in X. If True, will need to supply models, final_models, and
        propensity_model that can handle missing values.

    Examples
    --------
    A simple example:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.metalearners import DomainAdaptationLearner
        from sklearn.linear_model import LinearRegression

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = DomainAdaptationLearner(
            models=LinearRegression(),
            final_models=LinearRegression()
        )
        est.fit(y, T, X=X)

    >>> est.effect(X[:3])
    array([0.51238..., 1.99864..., 0.68553...])
    """

    def __init__(self, *,
                 models,
                 final_models,
                 propensity_model=LogisticRegression(),
                 categories='auto',
                 allow_missing=False):
        self.models = clone(models, safe=False)
        self.final_models = clone(final_models, safe=False)
        self.propensity_model = clone(propensity_model, safe=False)
        self.categories = categories
        self.allow_missing = allow_missing
        super().__init__()

    def _gen_allowed_missing_vars(self):
        return ['X'] if self.allow_missing else []

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, *, X, inference=None):
        """Build an instance of DomainAdaptationLearner.

        Parameters
        ----------
        Y : array_like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array_like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array_like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        inference : str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`)

        Returns
        -------
        self : an instance of self.
        """
        # Check inputs
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False,
                                  force_all_finite_X='allow-nan' if 'X' in self._gen_allowed_missing_vars() else True)
        categories = self.categories
        if categories != 'auto':
            categories = [categories]  # OneHotEncoder expects a 2D array with features per column
        self.transformer = one_hot_encoder(categories=categories, drop='first')
        T = self.transformer.fit_transform(T.reshape(-1, 1))
        self._d_t = T.shape[1:]
        T = inverse_onehot(T)
        self.models = check_models(self.models, self._d_t[0] + 1)
        self.final_models = check_models(self.final_models, self._d_t[0])
        self.propensity_models = []
        self.models_control = []
        self.models_treated = []
        for ind in range(self._d_t[0]):
            self.models_control.append(clone(self.models[0], safe=False))
            self.models_treated.append(clone(self.models[ind + 1], safe=False))
            self.propensity_models.append(clone(self.propensity_model, safe=False))

            X_concat = np.concatenate((X[T == 0], X[T == ind + 1]), axis=0)
            T_concat = np.concatenate((T[T == 0], T[T == ind + 1]), axis=0)
            self.propensity_models[ind].fit(X_concat, T_concat)
            pro_scores = self.propensity_models[ind].predict_proba(X_concat)[:, 1]

            # Train model on controls. Assign higher weight to units resembling
            # treated units.
            self._fit_weighted_pipeline(self.models_control[ind], X[T == 0], Y[T == 0],
                                        sample_weight=pro_scores[T_concat == 0] / (1 - pro_scores[T_concat == 0]))
            # Train model on the treated. Assign higher weight to units resembling
            # control units.
            self._fit_weighted_pipeline(self.models_treated[ind], X[T == ind + 1], Y[T == ind + 1],
                                        sample_weight=(1 - pro_scores[T_concat == ind + 1]) /
                                        pro_scores[T_concat == ind + 1])
            imputed_effect_on_controls = self.models_treated[ind].predict(X[T == 0]) - Y[T == 0]
            imputed_effect_on_treated = Y[T == ind + 1] - self.models_control[ind].predict(X[T == ind + 1])

            imputed_effects_concat = np.concatenate((imputed_effect_on_controls, imputed_effect_on_treated), axis=0)
            self.final_models[ind].fit(X_concat, imputed_effects_concat)

    def const_marginal_effect(self, X):
        """Calculate the constant marginal treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : matrix, shape (m, d_y, d_t)
            Constant marginal CATE of each treatment on each outcome for each sample X[i].
            Note that when Y is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
        """
        X = check_array(X)
        taus = []
        for model in self.final_models:
            taus.append(model.predict(X))
        taus = np.column_stack(taus).reshape((-1,) + self._d_t + self._d_y)  # shape as of m*d_t*d_y
        if self._d_y:
            taus = transpose(taus, (0, 2, 1))  # shape as of m*d_y*d_t
        return taus

    def _fit_weighted_pipeline(self, model_instance, X, y, sample_weight):
        if not isinstance(model_instance, Pipeline):
            model_instance.fit(X, y, sample_weight)
        else:
            last_step_name = model_instance.steps[-1][0]
            model_instance.fit(X, y, **{"{0}__sample_weight".format(last_step_name): sample_weight})

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        return _shap_explain_model_cate(self.const_marginal_effect, self.final_models, X, self._d_t, self._d_y,
                                        featurizer=None,
                                        feature_names=feature_names,
                                        treatment_names=treatment_names,
                                        output_names=output_names,
                                        input_names=self._input_names,
                                        background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__
