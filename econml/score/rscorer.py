# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

from typing import List, Optional, Tuple, Union, Any
from ..dml import LinearDML
from sklearn.base import clone, BaseEstimator
import numpy as np
from scipy.special import softmax
from .ensemble_cate import EnsembleCateEstimator


class RScorer:
    """
    Scorer based on the RLearner loss.

    Fits residual models at fit time and calculates
    residuals of the evaluation data in a cross-fitting manner::

        Yres = Y - E[Y|X, W]
        Tres = T - E[T|X, W]

    Then for any given cate model calculates the loss::

        loss(cate) = E_n[(Yres - <cate(X), Tres>)^2]

    Also calculates a baseline loss based on a constant treatment effect model, i.e.::

        base_loss = min_{theta} E_n[(Yres - <theta, Tres>)^2]

    Returns an analogue of the R-square score for regression::

        score = 1 - loss(cate) / base_loss

    This corresponds to the extra variance of the outcome explained by introducing heterogeneity
    in the effect as captured by the cate model, as opposed to always predicting a constant effect.
    A negative score means that the cate model performs worse than a constant effect model
    and may indicate overfitting.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement `fit` and `predict`.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement `fit` and `predict`.

    discrete_treatment: bool, default=False
        Whether the treatment values should be treated as categorical.

    discrete_outcome: bool, default=False
        Whether the outcome should be treated as binary.

    categories: 'auto' or list, default='auto'
        Categories to use when encoding discrete treatments. 'auto' uses unique sorted values.
        The first category is treated as the control.

    cv: int, cross-validation generator or iterable, default=2
        Determines the cross-validation splitting strategy. See sklearn docs for options.

    mc_iters: int, optional
        Number of Monte Carlo iterations to reduce nuisance variance.

    mc_agg: {'mean', 'median'}, default='mean'
        How to aggregate nuisance values across MC iterations.

    random_state: int, RandomState instance or None, default=None
        Controls randomness for reproducibility.
    """

    def __init__(self, *,
                 model_y: BaseEstimator,
                 model_t: BaseEstimator,
                 discrete_treatment: bool = False,
                 discrete_outcome: bool = False,
                 categories: Union[str, List] = 'auto',
                 cv: Union[int, Any] = 2,
                 mc_iters: Optional[int] = None,
                 mc_agg: str = 'mean',
                 random_state: Optional[Union[int, np.random.RandomState]] = None):
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.discrete_treatment = discrete_treatment
        self.discrete_outcome = discrete_outcome
        self.categories = categories
        self.cv = cv
        self.mc_iters = mc_iters
        self.mc_agg = mc_agg
        self.random_state = random_state

        # Internal state
        self.lineardml_: Optional[LinearDML] = None
        self.base_score_: Optional[float] = None
        self.dx_: Optional[int] = None

    def fit(self,
            y: np.ndarray,
            T: np.ndarray,
            X: Optional[np.ndarray] = None,
            W: Optional[np.ndarray] = None,
            sample_weight: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None) -> 'RScorer':
        """
        Fit residual models and compute baseline score.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Outcome(s) for each sample.
        T : array-like of shape (n_samples,) or (n_samples, n_treatments)
            Treatment(s) for each sample.
        X : array-like of shape (n_samples, n_features), optional
            Features for heterogeneity.
        W : array-like of shape (n_samples, n_controls), optional
            Control variables.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        groups : array-like of shape (n_samples,), optional
            Group labels for grouped CV splits.

        Returns
        -------
        self : RScorer
            Fitted scorer.
        """
        if X is None:
            raise ValueError("X cannot be None for the RScorer!")

        # Combine X and W for controls in DML
        W_full = np.hstack([v for v in [X, W] if v is not None]) if W is not None or X is not None else None

        self.lineardml_ = LinearDML(
            model_y=self.model_y,
            model_t=self.model_t,
            cv=self.cv,
            discrete_treatment=self.discrete_treatment,
            discrete_outcome=self.discrete_outcome,
            categories=self.categories,
            random_state=self.random_state,
            mc_iters=self.mc_iters,
            mc_agg=self.mc_agg
        )

        self.lineardml_.fit(
            y, T, X=None, W=W_full,
            sample_weight=sample_weight, groups=groups, cache_values=True
        )

        if not hasattr(self.lineardml_, '_cached_values') or self.lineardml_._cached_values is None:
            raise RuntimeError("LinearDML did not cache values. Ensure cache_values=True.")

        self.base_score_ = self.lineardml_.score_
        if self.base_score_ <= 0:
            raise ValueError(f"Base score must be positive. Got {self.base_score_}.")
        self.dx_ = X.shape[1]

        return self

    def _get_X_from_cached_W(self) -> np.ndarray:
        """Extract X from cached W (first dx_ columns)."""
        if self.lineardml_ is None or self.dx_ is None:
            raise RuntimeError("Must call fit() before score().")
        W_cached = self.lineardml_._cached_values.W
        return W_cached[:, :self.dx_]

    def _compute_loss(self, Y_res: np.ndarray, T_res: np.ndarray, effects: np.ndarray,
                      sample_weight: Optional[np.ndarray] = None) -> float:
        """
        Compute mean squared error: E[(Yres - <effects, Tres>)^2]

        Parameters
        ----------
        Y_res : (n, d_y)
        T_res : (n, d_t)
        effects : (n, d_y, d_t)
        sample_weight : (n,), optional

        Returns
        -------
        loss : float
        """
        # Predicted residuals: sum over treatment dimension
        # einsum: 'ijk,ik->ij' => for each sample i, output j: sum_k effects[i,j,k] * T_res[i,k]
        Y_res_pred = np.einsum('ijk,ik->ij', effects, T_res)

        sq_errors = (Y_res - Y_res_pred) ** 2  # (n, d_y)

        if sample_weight is not None:
            # Weighted average over samples, then mean over outputs
            loss = np.mean(np.average(sq_errors, weights=sample_weight, axis=0))
        else:
            loss = np.mean(sq_errors)

        return loss

    def score(self, cate_model: Any) -> float:
        """
        Score a CATE model against the baseline.

        Parameters
        ----------
        cate_model : fitted estimator
            Must have `const_marginal_effect(X)` method returning (n, d_y, d_t) array.

        Returns
        -------
        score : float
            R-squared style score. Higher is better. Can be negative.
        """
        if self.lineardml_ is None or self.base_score_ is None:
            raise RuntimeError("Must call fit() before score().")

        # Validate cate_model interface
        if not hasattr(cate_model, 'const_marginal_effect'):
            raise ValueError("cate_model must implement 'const_marginal_effect(X)' method.")

        Y_res, T_res = self.lineardml_._cached_values.nuisances
        X = self._get_X_from_cached_W()
        sample_weight = self.lineardml_._cached_values.sample_weight

        # Ensure 2D
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape(-1, 1)
        if T_res.ndim == 1:
            T_res = T_res.reshape(-1, 1)

        effects = cate_model.const_marginal_effect(X)
        if effects.ndim != 3:
            raise ValueError(f"Expected 3D effects (n, d_y, d_t), got shape {effects.shape}")

        loss = self._compute_loss(Y_res, T_res, effects, sample_weight)

        # Guard against division by zero (shouldn't happen due to fit() check, but still)
        if self.base_score_ <= 0:
            return -np.inf if loss > 0 else 1.0

        return 1 - loss / self.base_score_

    def best_model(self,
                   cate_models: List[Any],
                   return_scores: bool = False
                   ) -> Union[Tuple[Any, float], Tuple[Any, float, List[float]]]:
        """
        Select the best model based on R-scores.

        Parameters
        ----------
        cate_models : list of fitted estimators
        return_scores : bool, default=False
            If True, also return list of scores.

        Returns
        -------
        best_model : estimator
        best_score : float
        scores : list of float, optional
        """
        if not cate_models:
            raise ValueError("cate_models list is empty.")

        rscores = [self.score(mdl) for mdl in cate_models]

        # Handle all-NaN case
        finite_scores = [s for s in rscores if np.isfinite(s)]
        if not finite_scores:
            raise ValueError("All model scores are invalid (NaN or inf).")

        best_idx = np.nanargmax(rscores)  # nanargmax ignores NaNs
        best_model = cate_models[best_idx]
        best_score = rscores[best_idx]

        if return_scores:
            return best_model, best_score, rscores
        else:
            return best_model, best_score

    def ensemble(self,
                 cate_models: List[Any],
                 eta: float = 1000.0,
                 return_scores: bool = False
                 ) -> Union[Tuple[EnsembleCateEstimator, float],
                            Tuple[EnsembleCateEstimator, float, np.ndarray]]:
        """
        Create a weighted ensemble of models using softmax weights based on scores.

        Parameters
        ----------
        cate_models : list of fitted estimators
        eta : float, default=1000.0
            Temperature parameter for softmax weighting.
        return_scores : bool, default=False
            If True, also return raw scores.

        Returns
        -------
        ensemble : EnsembleCateEstimator
        ensemble_score : float
        scores : array, optional
        """
        if not cate_models:
            raise ValueError("cate_models list is empty.")

        rscores = np.array([self.score(mdl) for mdl in cate_models])
        goodinds = np.isfinite(rscores)

        if not np.any(goodinds):
            raise ValueError("No valid (finite) scores to ensemble.")

        # Softmax weights on finite scores
        weights = softmax(eta * rscores[goodinds])
        goodmodels = [mdl for mdl, keep in zip(cate_models, goodinds) if keep]

        ensemble = EnsembleCateEstimator(cate_models=goodmodels, weights=weights)
        ensemble_score = self.score(ensemble)

        if return_scores:
            return ensemble, ensemble_score, rscores
        else:
            return ensemble, ensemble_score
