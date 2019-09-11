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
from .utilities import check_inputs


class TLearner(BaseCateEstimator):
    """Conditional mean regression estimator.

    Parameters
    ----------
    controls_model : outcome estimator for control units
        Must implement `fit` and `predict` methods.

    treated_model : outcome estimator for treated units
        Must implement `fit` and `predict` methods.

    """

    def __init__(self, controls_model, treated_model):
        self.controls_model = clone(controls_model, safe=False)
        self.treated_model = clone(treated_model, safe=False)
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
        if not np.array_equal(np.unique(T), [0, 1]):
            raise ValueError("The treatments array (T) can only contain" +
                             "0 and 1.")
        self.controls_model.fit(X[T == 0], Y[T == 0])
        self.treated_model.fit(X[T == 1], Y[T == 1])

    def effect(self, X):
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
        tau_hat = self.treated_model.predict(X) - self.controls_model.predict(X)
        return tau_hat

    def marginal_effect(self, X):
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
        return self.effect(X)


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
        if not np.array_equal(np.unique(T), [0, 1]):
            raise ValueError("The treatments array (T) can only contain 0 and 1.")
        feat_arr = np.concatenate((X, T.reshape(-1, 1)), axis=1)
        self.overall_model.fit(feat_arr, Y)

    def effect(self, X):
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
        X_controls = np.concatenate((X, np.zeros((X.shape[0], 1))), axis=1)
        X_treated = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
        tau_hat = self.overall_model.predict(X_treated) - self.overall_model.predict(X_controls)
        return tau_hat

    def marginal_effect(self, X):
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
        return self.effect(X)


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

    propensity_func : propensity function
        Must accept an array of feature vectors and return an array of
        probabilities.
        If provided, the value for `propensity_model` (if any) will be ignored.

    """

    def __init__(self, controls_model,
                 treated_model,
                 cate_controls_model=None,
                 cate_treated_model=None,
                 propensity_model=LogisticRegression(),
                 propensity_func=None):
        self.controls_model = clone(controls_model, safe=False)
        self.treated_model = clone(treated_model, safe=False)
        self.cate_controls_model = clone(cate_controls_model, safe=False)
        self.cate_treated_model = clone(cate_treated_model, safe=False)

        if self.cate_controls_model is None:
            self.cate_controls_model = clone(self.controls_model, safe=False)
        if self.cate_treated_model is None:
            self.cate_treated_model = clone(self.treated_model, safe=False)

        self.propensity_func = clone(propensity_func, safe=False)
        self.propensity_model = clone(propensity_model, safe=False)
        self.has_propensity_func = self.propensity_func is not None
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
        if not np.array_equal(np.unique(T), [0, 1]):
            raise ValueError("The treatments array (T) can only contain 0 and 1.")

        self.controls_model.fit(X[T == 0], Y[T == 0])
        self.treated_model.fit(X[T == 1], Y[T == 1])
        imputed_effect_on_controls = self.treated_model.predict(X[T == 0]) - Y[T == 0]
        imputed_effect_on_treated = Y[T == 1] - self.controls_model.predict(X[T == 1])
        self.cate_controls_model.fit(X[T == 0], imputed_effect_on_controls)
        self.cate_treated_model.fit(X[T == 1], imputed_effect_on_treated)
        if not self.has_propensity_func:
            self.propensity_model.fit(X, T)
            self.propensity_func = lambda X_score: self.propensity_model.predict_proba(X_score)[:, 1]

    def effect(self, X):
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
        propensity_scores = self.propensity_func(X)
        tau_hat = propensity_scores * self.cate_controls_model.predict(X) \
            + (1 - propensity_scores) * self.cate_treated_model.predict(X)
        return tau_hat

    def marginal_effect(self, X):
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
        return self.effect(X)


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

    propensity_func : propensity function
        Must accept an array of feature vectors and return an array of probabilities.
        If provided, the value for `propensity_model` (if any) will be ignored.

    """

    def __init__(self, controls_model,
                 treated_model,
                 overall_model,
                 propensity_model=LogisticRegression(),
                 propensity_func=None):
        self.controls_model = clone(controls_model, safe=False)
        self.treated_model = clone(treated_model, safe=False)
        self.overall_model = clone(overall_model, safe=False)

        self.propensity_model = clone(propensity_model, safe=False)
        self.propensity_func = clone(propensity_func, safe=False)
        self.has_propensity_func = self.propensity_func is not None
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
        if not np.array_equal(np.unique(T), [0, 1]):
            raise ValueError("The treatments array (T) can only contain 0 and 1.")

        if not self.has_propensity_func:
            self.propensity_model.fit(X, T)
            self.propensity_func = lambda X_score: self.propensity_model.predict_proba(X_score)[:, 1]
        propensity_scores = self.propensity_func(X)
        # Train model on controls. Assign higher weight to units resembling
        # treated units.
        self._fit_weighted_pipeline(self.controls_model, X[T == 0], Y[T == 0],
                                    sample_weight=propensity_scores[T == 0] / (1 - propensity_scores[T == 0]))
        # Train model on the treated. Assign higher weight to units resembling
        # control units.
        self._fit_weighted_pipeline(self.treated_model, X[T == 1], Y[T == 1],
                                    sample_weight=(1 - propensity_scores[T == 1]) / propensity_scores[T == 1])
        imputed_effect_on_controls = self.treated_model.predict(X[T == 0]) - Y[T == 0]
        imputed_effect_on_treated = Y[T == 1] - self.controls_model.predict(X[T == 1])

        X_concat = np.concatenate((X[T == 0], X[T == 1]), axis=0)
        imputed_effects_concat = np.concatenate((imputed_effect_on_controls, imputed_effect_on_treated), axis=0)
        self.overall_model.fit(X_concat, imputed_effects_concat)

    def effect(self, X):
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
        tau_hat = self.overall_model.predict(X)
        return tau_hat

    def marginal_effect(self, X):
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
        return self.effect(X)

    def _fit_weighted_pipeline(self, model_instance, X, y, sample_weight):
        if not isinstance(model_instance, Pipeline):
            model_instance.fit(X, y, sample_weight)
        else:
            last_step_name = model_instance.steps[-1][0]
            model_instance.fit(X, y, **{"{0}__sample_weight".format(last_step_name): sample_weight})


class DoublyRobustLearner(BaseCateEstimator):
    """Meta-algorithm that uses doubly-robust correction techniques to account for
       covariate shift (selection bias) between the treatment arms.

    Parameters
    ----------
    outcome_model : outcome estimator for all data points
        Will be trained on features, controls and treatments (concatenated).
        If different models per treatment arm are desired, see the <econml.ortho_forest.MultiModelWrapper>
        helper class. The model(s) must implement `fit` and `predict` methods.

    pseudo_treatment_model : estimator for pseudo-treatment effects on the entire dataset
        Must implement `fit` and `predict` methods.

    propensity_model : estimator for the propensity function
        Must implement `fit` and `predict_proba` methods. The `fit` method must
        be able to accept X and T, where T is a shape (n, ) array.
        Ignored when `propensity_func` is provided.

    propensity_func : propensity function
        Must accept an array of feature vectors and return an array of
        probabilities.
        If provided, the value for `propensity_model` (if any) will be ignored.

    """

    def __init__(self,
                 outcome_model,
                 pseudo_treatment_model,
                 propensity_model=LogisticRegression(),
                 propensity_func=None):
        self.outcome_model = clone(outcome_model, safe=False)
        self.pseudo_treatment_model = clone(pseudo_treatment_model, safe=False)

        self.propensity_func = clone(propensity_func, safe=False)
        self.propensity_model = clone(propensity_model, safe=False)
        self.has_propensity_func = self.propensity_func is not None
        super().__init__()

    @BaseCateEstimator._wrap_fit
    def fit(self, Y, T, X, W=None, inference=None):
        """Build an instance of DoublyRobustLearner.

        Parameters
        ----------
        Y : array-like, shape (n, ) or (n, d_y)
            Outcome(s) for the treatment policy.

        T : array-like, shape (n, ) or (n, 1)
            Treatment policy. Only binary treatments are accepted as input.
            T will be flattened if shape is (n, 1).

        X : array-like, shape (n, d_x)
            Feature vector that captures heterogeneity.

        W : array-like, shape (n, d_w) or None (default=None)
            Controls (possibly high-dimensional).

        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of `BootstrapInference`)

        Returns
        -------
        self: an instance of self.
        """
        # Check inputs
        Y, T, X, W = check_inputs(Y, T, X, W, multi_output_T=False)
        Y = Y.flatten()
        if not np.array_equal(np.unique(T), [0, 1]):
            raise ValueError("The treatments array (T) can only contain 0 and 1.")
        if W is not None:
            XW = np.concatenate((X, W), axis=1)
        else:
            XW = X
        n = X.shape[0]
        # Fit outcome model on X||W||T (concatenated)
        self.outcome_model.fit(
            np.concatenate((XW, T.reshape(-1, 1)), axis=1),
            Y)
        if not self.has_propensity_func:
            self.propensity_model.fit(XW, T)
            self.propensity_func = lambda XW_score: self.propensity_model.predict_proba(XW_score)[:, 1]
        Y0 = self.outcome_model.predict(
            np.concatenate((XW, np.zeros((n, 1))), axis=1)
        )
        Y1 = self.outcome_model.predict(
            np.concatenate((XW, np.ones((n, 1))), axis=1)
        )
        propensities = self.propensity_func(XW)
        pseudo_te = Y1 - Y0
        pseudo_te[T == 0] -= (Y - Y0)[T == 0] / (1 - propensities)[T == 0]
        pseudo_te[T == 1] += (Y - Y1)[T == 1] / propensities[T == 1]
        self.pseudo_treatment_model.fit(X, pseudo_te)

    def effect(self, X):
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
        return self.pseudo_treatment_model.predict(X)

    def marginal_effect(self, X):
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
        return self.effect(X)
