# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Metalearners for heterogeneous treatment effects in the context of binary treatments.

For more details on these CATE methods, see <https://arxiv.org/abs/1706.03461>
(Künzel S., Sekhon J., Bickel P., Yu B.) on Arxiv.
"""

import numpy as np
import warnings
from .cate_estimator import BaseCateEstimator
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array, check_X_y
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from .utilities import check_inputs, check_models


class TLearner(BaseCateEstimator):
    """Conditional mean regression estimator.

    Parameters
    ----------
    controls_model : outcome estimator for control units
        Must implement `fit` and `predict` methods.

    treated_model : outcome estimator for treated units
        Must implement `fit` and `predict` methods.

    """

    def __init__(self, models):
        self.models = clone(models, safe=False)
        self._label_encoder = LabelEncoder()

        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X, inference=None):
        """Build an instance of TLearner.

        Parameters
        ----------
        Y : array-like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array-like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`)

        Returns
        -------
        self: an instance of self.

        """
        # Check inputs
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False)
        T = self._label_encoder.fit_transform(T)
        self.unique_T = self._label_encoder.classes_
        n_T = len(self.unique_T)
        self.models = check_models(self.models, n_T)

        for ind in range(n_T):
            self.models[ind].fit(X[T == ind], Y[T == ind])

    def effect(self, X, T0, T1):
        """Calculate the heterogeneous treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.
        T0: scaler
        T1: scaler

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        # Check inputs
        X = check_array(X)
        # Check treatment
        if T0 not in self.unique_T or T1 not in self.unique_T:
            raise ValueError(
                "T0 and T1 must be scalers in T you passed in at fitting time, "
                "please pass in values from {}".format(self.unique_T))
        ind1, = np.where(self.unique_T == T1)[0]
        ind0, = np.where(self.unique_T == T0)[0]
        tau_hat = self.models[ind1].predict(X) - self.models[ind0].predict(X)
        return tau_hat

    def marginal_effect(self, X, T1):
        """Calculate the heterogeneous marginal treatment effect.

        For binary treatments, it returns the same as `effect`.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.
        T1: scaler

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        return self.effect(X, T0=self.unique_T[0], T1=T1)


class SLearner(BaseCateEstimator):
    """Conditional mean regression estimator where the treatment assignment is taken as a feature in the ML model.

    Parameters
    ----------
    overall_model : outcome estimator for all units
        Model will be trained on X|T where '|' denotes concatenation.
        Must implement `fit` and `predict` methods.

    """

    def __init__(self, overall_model):
        self.overall_model = clone(overall_model, safe=False)
        self._one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X, inference=None):
        """Build an instance of SLearner.

        Parameters
        ----------
        Y : array-like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array-like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`)

        Returns
        -------
        self: an instance of self.
        """
        # Check inputs
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False)
        T = self._one_hot_encoder.fit_transform(T.reshape(-1, 1))[:, 1:]
        feat_arr = np.concatenate((X, T), axis=1)
        self.overall_model.fit(feat_arr, Y)

    def effect(self, X, T0, T1):
        """Calculate the heterogeneous treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        # Check inputs
        X = check_array(X)
        m = X.shape[0]
        X_controls = np.concatenate((X, self._one_hot_encoder.transform(np.ones((m, 1)) * T0)[:, 1:]), axis=1)
        X_treated = np.concatenate((X, self._one_hot_encoder.transform(np.ones((m, 1)) * T1)[:, 1:]), axis=1)
        tau_hat = self.overall_model.predict(X_treated) - self.overall_model.predict(X_controls)
        return tau_hat

    def marginal_effect(self, X, T1):
        """Calculate the heterogeneous marginal treatment effect.

        For binary treatments, it returns the same as `effect`.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        T0 = self._one_hot_encoder.categories_[0][0]
        return self.effect(X, T0=T0, T1=T1)


class XLearner(BaseCateEstimator):
    """Meta-algorithm proposed by Kunzel et al. that performs best in settings
       where the number of units in one treatment arm is much larger than in the other.

    Parameters
    ----------
    controls_model : outcome estimator for control units
        Must implement `fit` and `predict` methods.

    treated_model : outcome estimator for treated units
        Must implement `fit` and `predict` methods.

    cate_controls_model : estimator for pseudo-treatment effects on the controls
        Must implement `fit` and `predict` methods.

    cate_treated_model : estimator for pseudo-treatment effects on the treated
        Must implement `fit` and `predict` methods.

    propensity_model : estimator for the propensity function
        Must implement `fit` and `predict_proba` methods. The `fit` method must
        be able to accept X and T, where T is a shape (n, ) array.
        Ignored when `propensity_func` is provided.

    """

    def __init__(self, models,
                 cate_models=None,
                 propensity_model=LogisticRegression()):
        self.models = clone(models, safe=False)
        self.cate_models = clone(cate_models, safe=False)
        self.propensity_model = clone(propensity_model, safe=False)
        self._label_encoder = LabelEncoder()
        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X, inference=None):
        """Build an instance of XLearner.

        Parameters
        ----------
        Y : array-like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array-like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`)

        Returns
        -------
        self: an instance of self.
        """
        # Check inputs
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False)
        T = self._label_encoder.fit_transform(T)
        self.unique_T = self._label_encoder.classes_
        n_T = len(self.unique_T)
        self.models = check_models(self.models, n_T)
        if self.cate_models is None:
            self.cate_models = self.models
        else:
            self.cate_models = check_models(self.cate_models, n_T)
        self.propensity_models = []
        self.cate_treated_models = []
        self.cate_controls_models = []

        # Estimate response function
        for ind in range(n_T):
            self.models[ind].fit(X[T == ind], Y[T == ind])
        for ind in range(1, n_T):
            self.cate_treated_models.append(clone(self.cate_models[ind], safe=False))
            self.cate_controls_models.append(clone(self.cate_models[0], safe=False))
            self.propensity_models.append(clone(self.propensity_model, safe=False))
            imputed_effect_on_controls = self.models[ind].predict(X[T == 0]) - Y[T == 0]
            imputed_effect_on_treated = Y[T == ind] - self.models[0].predict(X[T == ind])
            self.cate_controls_models[ind - 1].fit(X[T == 0], imputed_effect_on_controls)
            self.cate_treated_models[ind - 1].fit(X[T == ind], imputed_effect_on_treated)
            X_concat = np.concatenate((X[T == 0], X[T == ind]), axis=0)
            T_concat = np.concatenate((T[T == 0], T[T == ind]), axis=0)
            self.propensity_models[ind - 1].fit(X_concat, T_concat)

    def effect(self, X, T0, T1):
        """Calculate the heterogeneous treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        # Check inputs
        if T0 == self.unique_T[0]:
            return self.marginal_effect(X, T1)
        tau_hat = self.marginal_effect(X, T1) - self.marginal_effect(X, T0)
        return tau_hat

    def marginal_effect(self, X, T1):
        """Calculate the heterogeneous marginal treatment effect.

        For binary treatments, it returns the same as `effect`.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        X = check_array(X)
        ind, = np.where(self.unique_T == T1)[0]
        propensity_scores = self.propensity_models[ind - 1].predict_proba(X)[:, 1]
        tau_hat = propensity_scores * self.cate_controls_models[ind - 1].predict(X) \
            + (1 - propensity_scores) * self.cate_treated_models[ind - 1].predict(X)
        return tau_hat


class DomainAdaptationLearner(BaseCateEstimator):
    """Meta-algorithm that uses domain adaptation techniques to account for
       covariate shift (selection bias) between the treatment arms.

    Parameters
    ----------
    controls_model : outcome estimator for control units
        Must implement `fit` and `predict` methods.
        The `fit` method must accept the `sample_weight` parameter.

    treated_model : outcome estimator for treated units
        Must implement `fit` and `predict` methods.
        The `fit` method must accept the `sample_weight` parameter.

    overall_model : estimator for pseudo-treatment effects
        Must implement `fit` and `predict` methods.

    propensity_model : estimator for the propensity function
        Must implement `fit` and `predict_proba` methods. The `fit` method must
        be able to accept X and T, where T is a shape (n, 1) array.
        Ignored when `propensity_func` is provided.

    """

    def __init__(self, models,
                 final_models,
                 propensity_model=LogisticRegression()):
        self.models = clone(models, safe=False)
        self.final_models = clone(final_models, safe=False)
        self.propensity_model = clone(propensity_model, safe=False)
        self._label_encoder = LabelEncoder()
        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X, inference=None):
        """Build an instance of DomainAdaptationLearner.

        Parameters
        ----------
        Y : array-like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array-like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`)

        Returns
        -------
        self: an instance of self.
        """
        # Check inputs
        Y, T, X, _ = check_inputs(Y, T, X, multi_output_T=False)
        T = self._label_encoder.fit_transform(T)
        self.unique_T = self._label_encoder.classes_
        n_T = len(self.unique_T)
        self.models = check_models(self.models, n_T)
        self.final_models = check_models(self.final_models, n_T - 1)
        self.propensity_models = []
        self.models_control = []
        self.models_treated = []
        for ind in range(1, n_T):
            self.models_control.append(clone(self.models[0], safe=False))
            self.models_treated.append(clone(self.models[ind], safe=False))
            self.propensity_models.append(clone(self.propensity_model, safe=False))

            X_concat = np.concatenate((X[T == 0], X[T == ind]), axis=0)
            T_concat = np.concatenate((T[T == 0], T[T == ind]), axis=0)
            self.propensity_models[ind - 1].fit(X_concat, T_concat)
            pro_scores = self.propensity_models[ind - 1].predict_proba(X_concat)[:, 1]

            # Train model on controls. Assign higher weight to units resembling
            # treated units.
            self._fit_weighted_pipeline(self.models_control[ind - 1], X[T == 0], Y[T == 0],
                                        sample_weight=pro_scores[T_concat == 0] / (1 - pro_scores[T_concat == 0]))
            # Train model on the treated. Assign higher weight to units resembling
            # control units.
            self._fit_weighted_pipeline(self.models_treated[ind - 1], X[T == ind], Y[T == ind],
                                        sample_weight=(1 - pro_scores[T_concat == ind]) / pro_scores[T_concat == ind])
            imputed_effect_on_controls = self.models_treated[ind - 1].predict(X[T == 0]) - Y[T == 0]
            imputed_effect_on_treated = Y[T == ind] - self.models_control[ind - 1].predict(X[T == ind])

            imputed_effects_concat = np.concatenate((imputed_effect_on_controls, imputed_effect_on_treated), axis=0)
            self.final_models[ind - 1].fit(X_concat, imputed_effects_concat)

    def effect(self, X, T0, T1):
        """Calculate the heterogeneous treatment effect on a vector of features for each sample.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        if T0 == self.unique_T[0]:
            return self.marginal_effect(X, T1)

        tau_hat = self.marginal_effect(X, T1) - self.marginal_effect(X, T0)
        return tau_hat

    def marginal_effect(self, X, T1):
        """Calculate the heterogeneous marginal treatment effect.

        For binary treatments, it returns the same as `effect`.

        Parameters
        ----------
        X : matrix, shape (m × dₓ)
            Matrix of features for each sample.

        Returns
        -------
        τ_hat : array-like, shape (m, )
            Matrix of heterogeneous treatment effects for each sample.
        """
        X = check_array(X)
        ind, = np.where(self.unique_T == T1)[0]
        tau_hat = self.final_models[ind - 1].predict(X)
        return tau_hat

    def _fit_weighted_pipeline(self, model_instance, X, y, sample_weight):
        if not isinstance(model_instance, Pipeline):
            model_instance.fit(X, y, sample_weight)
        else:
            last_step_name = model_instance.steps[-1][0]
            model_instance.fit(X, y, **{"{0}__sample_weight".format(last_step_name): sample_weight})
