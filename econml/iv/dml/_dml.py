# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Orthogonal IV for Heterogeneous Treatment Effects.

A Double/Orthogonal machine learning approach to estimation of heterogeneous
treatment effect with an endogenous treatment and an instrument. It
implements the DMLIV and related algorithms from the paper:

Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments
Vasilis Syrgkanis, Victor Lei, Miruna Oprescu, Maggie Hei, Keith Battocchi, Greg Lewis
https://arxiv.org/abs/1905.10176

"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ..._ortho_learner import _OrthoLearner
from ..._cate_estimator import LinearModelFinalCateEstimatorMixin, StatsModelsCateEstimatorMixin
from ...inference import StatsModelsInference
from ...sklearn_extensions.linear_model import StatsModelsLinearRegression
from ...utilities import _deprecate_positional
from .._nuisance_wrappers import _FirstStageWrapper, _FinalWrapper


class _BaseDMLATEIVModelFinal:
    def __init__(self):
        self._first_stage = LinearRegression(fit_intercept=False)
        self._model_final = _FinalWrapper(LinearRegression(fit_intercept=False),
                                          fit_cate_intercept=True, featurizer=None, use_weight_trick=False)

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        Y_res, T_res, Z_res = nuisances
        if Z_res.ndim == 1:
            Z_res = Z_res.reshape(-1, 1)
        # DMLATEIV is just like 2SLS; first regress T_res on Z_res, then regress Y_res on predicted T_res
        T_res_pred = self._first_stage.fit(Z_res, T_res,
                                           sample_weight=sample_weight).predict(Z_res)
        # TODO: allow the final model to actually use X? Then we'd need to rename the class
        #       since we would actually be calculating a CATE rather than ATE.
        self._model_final.fit(X=None, T_res=T_res_pred, Y_res=Y_res, sample_weight=sample_weight)
        return self

    def predict(self, X=None):
        # TODO: allow the final model to actually use X?
        return self._model_final.predict(X=None)

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        Y_res, T_res, Z_res = nuisances
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        # TODO: allow the final model to actually use X?
        effects = self._model_final.predict(X=None).reshape((-1, Y_res.shape[1], T_res.shape[1]))
        Y_res_pred = np.einsum('ijk,ik->ij', effects, T_res).reshape(Y_res.shape)
        if sample_weight is not None:
            return np.mean(np.average((Y_res - Y_res_pred)**2, weights=sample_weight, axis=0))
        else:
            return np.mean((Y_res - Y_res_pred) ** 2)


class _BaseDMLATEIV(_OrthoLearner):
    def __init__(self, discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        super().__init__(discrete_treatment=discrete_treatment,
                         discrete_instrument=discrete_instrument,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_ortho_learner_model_final(self):
        return _BaseDMLATEIVModelFinal()

    @_deprecate_positional("W and Z should be passed by keyword only. In a future release "
                           "we will disallow passing W and Z by position.", ['W', 'Z'])
    def fit(self, Y, T, Z, W=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string,:class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of:class:`.BootstrapInference`).

        Returns
        -------
        self: _BaseDMLATEIV instance
        """
        # Replacing fit from _OrthoLearner, to enforce W=None and improve the docstring
        return super().fit(Y, T, W=W, Z=Z,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    def score(self, Y, T, Z, W=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted residual nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calculates the MSE of the final residual Y on residual T regression.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        Z: optional(n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample


        Returns
        -------
        score: float
            The MSE of the final CATE model on the new data.
        """
        # Replacing score from _OrthoLearner, to enforce X=None and improve the docstring
        return super().score(Y, T, W=W, Z=Z)


class _DMLATEIVModelNuisance:
    def __init__(self, model_Y_W, model_T_W, model_Z_W):
        self._model_Y_W = clone(model_Y_W, safe=False)
        self._model_T_W = clone(model_T_W, safe=False)
        self._model_Z_W = clone(model_Z_W, safe=False)

    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        assert X is None, "DML ATE IV does not accept features"
        self._model_Y_W.fit(X=X, W=W, Target=Y, sample_weight=sample_weight, groups=groups)
        self._model_T_W.fit(X=X, W=W, Target=T, sample_weight=sample_weight, groups=groups)
        self._model_Z_W.fit(X=X, W=W, Target=Z, sample_weight=sample_weight, groups=groups)
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        assert X is None, "DML ATE IV does not accept features"
        if hasattr(self._model_Y_W, 'score'):
            Y_X_score = self._model_Y_W.score(X=X, W=W, Target=Y, sample_weight=sample_weight)
        else:
            Y_X_score = None
        if hasattr(self._model_T_W, 'score'):
            T_X_score = self._model_T_W.score(X=X, W=W, Target=T, sample_weight=sample_weight)
        else:
            T_X_score = None
        if hasattr(self._model_Z_W, 'score'):
            Z_X_score = self._model_Z_W.score(X=X, W=W, Target=Z, sample_weight=sample_weight)
        else:
            Z_X_score = None
        return Y_X_score, T_X_score, Z_X_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        assert X is None, "DML ATE IV does not accept features"
        Y_pred = self._model_Y_W.predict(X=X, W=W)
        T_pred = self._model_T_W.predict(X=X, W=W)
        Z_pred = self._model_Z_W.predict(X=X, W=W)
        if W is None:  # In this case predict above returns a single row
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            T_pred = np.tile(T_pred.reshape(1, -1), (T.shape[0], 1))
            Z_pred = np.tile(Z_pred.reshape(1, -1), (Z.shape[0], 1))
        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = T - T_pred.reshape(T.shape)
        Z_res = Z - Z_pred.reshape(Z.shape)
        return Y_res, T_res, Z_res


class DMLATEIV(_BaseDMLATEIV):
    """
    Implementation of the orthogonal/double ml method for ATE estimation with
    IV as described in

    Double/Debiased Machine Learning for Treatment and Causal Parameters
    Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins
    https://arxiv.org/abs/1608.00060

    Requires that either co-variance of T, Z is independent of X or that effect
    is not heterogeneous in X for correct recovery. Otherwise it estimates
    a biased ATE.
    """

    def __init__(self, *,
                 model_Y_W,
                 model_T_W,
                 model_Z_W,
                 discrete_treatment=False,
                 discrete_instrument=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_Y_W = clone(model_Y_W, safe=False)
        self.model_T_W = clone(model_T_W, safe=False)
        self.model_Z_W = clone(model_Z_W, safe=False)
        super().__init__(discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_ortho_learner_model_nuisance(self):
        return _DMLATEIVModelNuisance(
            model_Y_W=_FirstStageWrapper(clone(self.model_Y_W, safe=False), discrete_target=False),
            model_T_W=_FirstStageWrapper(clone(self.model_T_W, safe=False), discrete_target=self.discrete_treatment),
            model_Z_W=_FirstStageWrapper(clone(self.model_Z_W, safe=False), discrete_target=self.discrete_instrument))


class _ProjectedDMLATEIVModelNuisance:

    def __init__(self, model_Y_W, model_T_W, model_T_WZ):
        self._model_Y_W = clone(model_Y_W, safe=False)
        self._model_T_W = clone(model_T_W, safe=False)
        self._model_T_WZ = clone(model_T_WZ, safe=False)

    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        assert X is None, "DML ATE IV does not accept features"
        self._model_Y_W.fit(X=X, W=W, Target=Y, sample_weight=sample_weight, groups=groups)
        self._model_T_W.fit(X=X, W=W, Target=T, sample_weight=sample_weight, groups=groups)
        self._model_T_WZ.fit(X=X, W=W, Z=Z, Target=T, sample_weight=sample_weight, groups=groups)
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        assert X is None, "DML ATE IV does not accept features"
        if hasattr(self._model_Y_W, 'score'):
            Y_X_score = self._model_Y_W.score(X=X, W=W, Target=Y, sample_weight=sample_weight)
        else:
            Y_X_score = None
        if hasattr(self._model_T_W, 'score'):
            T_X_score = self._model_T_W.score(X=X, W=W, Target=T, sample_weight=sample_weight)
        else:
            T_X_score = None
        if hasattr(self._model_T_WZ, 'score'):
            T_XZ_score = self._model_T_WZ.score(X=X, W=W, Z=Z, Target=T, sample_weight=sample_weight)
        else:
            T_XZ_score = None
        return Y_X_score, T_X_score, T_XZ_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        assert X is None, "DML ATE IV does not accept features"
        Y_pred = self._model_Y_W.predict(X, W)
        TX_pred = self._model_T_W.predict(X, W)
        TXZ_pred = self._model_T_WZ.predict(X, W, Z)
        if W is None:  # In this case predict above returns a single row
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            TX_pred = np.tile(TX_pred.reshape(1, -1), (T.shape[0], 1))
        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = T - TX_pred.reshape(T.shape)
        Z_res = TXZ_pred.reshape(T.shape) - TX_pred.reshape(T.shape)
        return Y_res, T_res, Z_res


class ProjectedDMLATEIV(_BaseDMLATEIV):

    def __init__(self, *,
                 model_Y_W,
                 model_T_W,
                 model_T_WZ,
                 discrete_treatment=False,
                 discrete_instrument=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_Y_W = clone(model_Y_W, safe=False)
        self.model_T_W = clone(model_T_W, safe=False)
        self.model_T_WZ = clone(model_T_WZ, safe=False)
        super().__init__(discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_ortho_learner_model_nuisance(self):
        return _ProjectedDMLATEIVModelNuisance(
            model_Y_W=_FirstStageWrapper(clone(self.model_Y_W, safe=False), discrete_target=False),
            model_T_W=_FirstStageWrapper(clone(self.model_T_W, safe=False), discrete_target=self.discrete_treatment),
            model_T_WZ=_FirstStageWrapper(clone(self.model_T_WZ, safe=False),
                                          discrete_target=self.discrete_treatment))


class _BaseDMLIVModelNuisance:
    """
    Nuisance model fits the three models at fit time and at predict time
    returns :math:`Y-\\E[Y|X]` and :math:`\\E[T|X,Z]-\\E[T|X]` as residuals.
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ):
        self._model_Y_X = clone(model_Y_X, safe=False)
        self._model_T_X = clone(model_T_X, safe=False)
        self._model_T_XZ = clone(model_T_XZ, safe=False)

    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        # TODO: would it be useful to extend to handle controls ala vanilla DML?
        assert W is None, "DML IV does not accept controls"
        self._model_Y_X.fit(X=X, W=None, Target=Y, sample_weight=sample_weight, groups=groups)
        self._model_T_X.fit(X=X, W=None, Target=T, sample_weight=sample_weight, groups=groups)
        self._model_T_XZ.fit(X=X, W=None, Z=Z, Target=T, sample_weight=sample_weight, groups=groups)
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        assert W is None, "DML IV does not accept controls"
        if hasattr(self._model_Y_X, 'score'):
            Y_X_score = self._model_Y_X.score(X=X, W=W, Target=Y, sample_weight=sample_weight)
        else:
            Y_X_score = None
        if hasattr(self._model_T_X, 'score'):
            T_X_score = self._model_T_X.score(X=X, W=W, Target=T, sample_weight=sample_weight)
        else:
            T_X_score = None
        if hasattr(self._model_T_XZ, 'score'):
            T_XZ_score = self._model_T_XZ.score(X=X, W=W, Z=Z, Target=T, sample_weight=sample_weight)
        else:
            T_XZ_score = None
        return Y_X_score, T_X_score, T_XZ_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        assert W is None, "DML IV does not accept controls"
        Y_pred = self._model_Y_X.predict(X, W)
        TXZ_pred = self._model_T_XZ.predict(X, W, Z)
        TX_pred = self._model_T_X.predict(X, W)
        if X is None:  # In this case predict above returns a single row
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            TX_pred = np.tile(TX_pred.reshape(1, -1), (T.shape[0], 1))
        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = TXZ_pred.reshape(T.shape) - TX_pred.reshape(T.shape)
        return Y_res, T_res


class _BaseDMLIVModelFinal:
    """
    Final model at fit time, fits a residual on residual regression with a heterogeneous coefficient
    that depends on X, i.e.

        .. math ::
            Y - \\E[Y | X] = \\theta(X) \\cdot (\\E[T | X, Z] - \\E[T | X]) + \\epsilon

    and at predict time returns :math:`\\theta(X)`. The score method returns the MSE of this final
    residual on residual regression.
    """

    def __init__(self, model_final):
        self._model_final = clone(model_final, safe=False)

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        Y_res, T_res = nuisances
        self._model_final.fit(X, T_res, Y_res, sample_weight=sample_weight, sample_var=sample_var)
        return self

    def predict(self, X=None):
        return self._model_final.predict(X)

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        Y_res, T_res = nuisances
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        effects = self._model_final.predict(X).reshape((-1, Y_res.shape[1], T_res.shape[1]))
        Y_res_pred = np.einsum('ijk,ik->ij', effects, T_res).reshape(Y_res.shape)
        if sample_weight is not None:
            return np.mean(np.average((Y_res - Y_res_pred)**2, weights=sample_weight, axis=0))
        else:
            return np.mean((Y_res - Y_res_pred)**2)


class _BaseDMLIV(_OrthoLearner):
    """
    The class _BaseDMLIV implements the base class of the DMLIV
    algorithm for estimating a CATE. It accepts three generic machine
    learning models:
    1) model_Y_X that estimates :math:`\\E[Y | X]`
    2) model_T_X that estimates :math:`\\E[T | X]`
    3) model_T_XZ that estimates :math:`\\E[T | X, Z]`
    These are estimated in a cross-fitting manner for each sample in the training set.
    Then it minimizes the square loss:

    .. math::
        \\sum_i (Y_i - \\E[Y|X_i] - \theta(X) * (\\E[T|X_i, Z_i] - \\E[T|X_i]))^2

    This loss is minimized by the model_final class, which is passed as an input.
    In the two children classes {DMLIV, GenericDMLIV}, we implement different strategies of how to invoke
    machine learning algorithms to minimize this final square loss.


    Parameters
    ----------
    model_Y_X : estimator
        model to estimate :math:`\\E[Y | X]`.  Must support `fit` and `predict` methods.

    model_T_X : estimator
        model to estimate :math:`\\E[T | X]`.  Must support `fit` and `predict` methods

    model_T_XZ : estimator
        model to estimate :math:`\\E[T | X, Z]`.  Must support `fit(X, Z, T, *, sample_weights)`
        and `predict(X, Z)` methods.

    model_final : estimator
        final model that at fit time takes as input :math:`(Y-\\E[Y|X])`, :math:`(\\E[T|X,Z]-\\E[T|X])` and X
        and supports method predict(X) that produces the CATE at X

    discrete_instrument: bool, optional, default False
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    discrete_treatment: bool, optional, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, optional, default 2
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(concat[W, X], T)` to generate the splits. If all
        W, X are None, then we call `split(ones((T.shape[0], 1)), T)`.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, discrete_instrument=False, discrete_treatment=False, categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None, mc_agg='mean',
                 random_state=None):
        super().__init__(discrete_treatment=discrete_treatment,
                         discrete_instrument=discrete_instrument,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    @_deprecate_positional("Z and X should be passed by keyword only. In a future release "
                           "we will disallow passing Z and X by position.", ['X', 'Z'])
    def fit(self, Y, T, Z, X=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string,:class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of:class:`.BootstrapInference`).

        Returns
        -------
        self: _BaseDMLIV
        """
        # Replacing fit from _OrthoLearner, to enforce W=None and improve the docstring
        return super().fit(Y, T, X=X, Z=Z,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    def score(self, Y, T, Z, X=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted residual nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calculates the MSE of the final residual Y on residual T regression.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        Z: optional(n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample


        Returns
        -------
        score: float
            The MSE of the final CATE model on the new data.
        """
        # Replacing score from _OrthoLearner, to enforce W=None and improve the docstring
        return super().score(Y, T, X=X, Z=Z)

    @property
    def original_featurizer(self):
        return self.ortho_learner_model_final_._model_final._original_featurizer

    @property
    def featurizer_(self):
        # NOTE This is used by the inference methods and has to be the overall featurizer. intended
        # for internal use by the library
        return self.ortho_learner_model_final_._model_final._featurizer

    @property
    def model_final_(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        return self.ortho_learner_model_final_._model_final._model

    @property
    def model_cate(self):
        """
        Get the fitted final CATE model.

        Returns
        -------
        model_cate: object of type(model_final)
            An instance of the model_final object that was fitted after calling fit which corresponds
            to the constant marginal CATE model.
        """
        return self.ortho_learner_model_final_._model_final._model

    @property
    def models_Y_X(self):
        """
        Get the fitted models for :math:`\\E[Y | X]`.

        Returns
        -------
        models_Y_X: list of objects of type(`model_Y_X`)
            A list of instances of the `model_Y_X` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model_Y_X._model for mdl in super().models_nuisance_]

    @property
    def models_T_X(self):
        """
        Get the fitted models for :math:`\\E[T | X]`.

        Returns
        -------
        models_T_X: list of objects of type(`model_T_X`)
            A list of instances of the `model_T_X` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model_T_X._model for mdl in super().models_nuisance_]

    @property
    def models_T_XZ(self):
        """
        Get the fitted models for :math:`\\E[T | X, Z]`.

        Returns
        -------
        models_T_XZ: list of objects of type(`model_T_XZ`)
            A list of instances of the `model_T_XZ` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model_T_XZ._model for mdl in super().models_nuisance_]

    @property
    def nuisance_scores_Y_X(self):
        """
        Get the scores for Y_X model on the out-of-sample training data
        """
        return self.nuisance_scores_[0]

    @property
    def nuisance_scores_T_X(self):
        """
        Get the scores for T_X model on the out-of-sample training data
        """
        return self.nuisance_scores_[1]

    @property
    def nuisance_scores_T_XZ(self):
        """
        Get the scores for T_XZ model on the out-of-sample training data
        """
        return self.nuisance_scores_[2]

    def cate_feature_names(self, feature_names=None):
        """
        Get the output feature names.

        Parameters
        ----------
        feature_names: list of strings of length X.shape[1] or None
            The names of the input features. If None and X is a dataframe, it defaults to the column names
            from the dataframe.

        Returns
        -------
        out_feature_names: list of strings or None
            The names of the output features :math:`\\phi(X)`, i.e. the features with respect to which the
            final constant marginal CATE model is linear. It is the names of the features that are associated
            with each entry of the :meth:`coef_` parameter. Not available when the featurizer is not None and
            does not have a method: `get_feature_names(feature_names)`. Otherwise None is returned.
        """
        if feature_names is None:
            feature_names = self._input_names["feature_names"]
        if self.original_featurizer is None:
            return feature_names
        elif hasattr(self.original_featurizer, 'get_feature_names'):
            return self.original_featurizer.get_feature_names(feature_names)
        else:
            raise AttributeError("Featurizer does not have a method: get_feature_names!")


class DMLIV(LinearModelFinalCateEstimatorMixin, _BaseDMLIV):
    """
    A child of the _BaseDMLIV class that specifies a particular effect model
    where the treatment effect is linear in some featurization of the variable X
    The features are created by a provided featurizer that supports fit_transform.
    Then an arbitrary model fits on the composite set of features.

    Concretely, it assumes that :math:`\\theta(X)=<\\theta, \\phi(X)>` for some features :math:`\\phi(X)`
    and runs a linear model regression of :math:`Y-\\E[Y|X]` on :math:`phi(X)*(\\E[T|X,Z]-\\E[T|X])`.
    The features are created by the featurizer provided by the user. The particular
    linear model regression is also specified by the user (e.g. Lasso, ElasticNet)

    Parameters
    ----------
    model_Y_X : estimator
        model to estimate :math:`\\E[Y | X]`.  Must support `fit` and `predict` methods.

    model_T_X : estimator
        model to estimate :math:`\\E[T | X]`.  Must support `fit` and either `predict` or `predict_proba` methods,
        depending on whether the treatment is discrete.

    model_T_XZ : estimator
        model to estimate :math:`\\E[T | X, Z]`.  Must support `fit` and either `predict` or `predict_proba` methods,
        depending on whether the treatment is discrete.

    model_final : estimator
        final linear model for predicting :math:`(Y-\\E[Y|X])` from :math:`\\phi(X) \\cdot (\\E[T|X,Z]-\\E[T|X])`
        Method is incorrect if this model is not linear (e.g. Lasso, ElasticNet, LinearRegression).

    featurizer: :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    cv: int, cross-validation generator or an iterable, optional, default 2
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(concat[W, X], T)` to generate the splits. If all
        W, X are None, then we call `split(ones((T.shape[0], 1)), T)`.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    discrete_instrument: bool, optional, default False
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    discrete_treatment: bool, optional, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, *,
                 model_Y_X,
                 model_T_X,
                 model_T_XZ,
                 model_final,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 discrete_instrument=False, discrete_treatment=False,
                 categories='auto', random_state=None):
        self.model_Y_X = clone(model_Y_X, safe=False)
        self.model_T_X = clone(model_T_X, safe=False)
        self.model_T_XZ = clone(model_T_XZ, safe=False)
        self.model_final = clone(model_final, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.fit_cate_intercept = fit_cate_intercept
        super().__init__(cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         random_state=random_state)

    def _gen_ortho_learner_model_nuisance(self):
        return _BaseDMLIVModelNuisance(_FirstStageWrapper(clone(self.model_Y_X, safe=False), False),
                                       _FirstStageWrapper(clone(self.model_T_X, safe=False), self.discrete_treatment),
                                       _FirstStageWrapper(clone(self.model_T_XZ, safe=False), self.discrete_treatment))

    def _gen_ortho_learner_model_final(self):
        return _BaseDMLIVModelFinal(_FinalWrapper(clone(self.model_final, safe=False),
                                                  fit_cate_intercept=self.fit_cate_intercept,
                                                  featurizer=clone(self.featurizer, safe=False),
                                                  use_weight_trick=False))

    @property
    def bias_part_of_coef(self):
        return self.ortho_learner_model_final_._model_final._fit_cate_intercept

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final_._model_final._fit_cate_intercept


class NonParamDMLIV(_BaseDMLIV):
    """
    A child of the _BaseDMLIV class that allows for an arbitrary square loss based ML
    method in the final stage of the DMLIV algorithm. The method has to support
    sample weights and the fit method has to take as input sample_weights (e.g. random forests), i.e.
    fit(X, y, sample_weight=None)
    It achieves this by re-writing the final stage square loss of the DMLIV algorithm as:

    .. math ::
        \\sum_i (\\E[T|X_i, Z_i] - \\E[T|X_i])^2 * ((Y_i - \\E[Y|X_i])/(\\E[T|X_i, Z_i] - \\E[T|X_i]) - \\theta(X))^2

    Then this can be viewed as a weighted square loss regression, where the target label is

    .. math ::
        \\tilde{Y}_i = (Y_i - \\E[Y|X_i])/(\\E[T|X_i, Z_i] - \\E[T|X_i])

    and each sample has a weight of

    .. math ::
        V(X_i) = (\\E[T|X_i, Z_i] - \\E[T|X_i])^2

    Thus we can call any regression model with inputs:

        fit(X, :math:`\\tilde{Y}_i`, sample_weight= :math:`V(X_i)`)

    Parameters
    ----------
    model_Y_X : estimator
        model to estimate :math:`\\E[Y | X]`.  Must support `fit` and `predict` methods.

    model_T_X : estimator
        model to estimate :math:`\\E[T | X]`.  Must support `fit` and either `predict` or `predict_proba` methods,
        depending on whether the treatment is discrete.

    model_T_XZ : estimator
        model to estimate :math:`\\E[T | X, Z]`.  Must support `fit` and either `predict` or `predict_proba` methods,
        depending on whether the treatment is discrete.

    model_final : estimator
        final model for predicting :math:`\\tilde{Y}` from X with sample weights V(X)

    cv: int, cross-validation generator or an iterable, optional, default 2
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(concat[W, X], T)` to generate the splits. If all
        W, X are None, then we call `split(ones((T.shape[0], 1)), T)`.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    discrete_instrument: bool, optional, default False
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    discrete_treatment: bool, optional, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    """

    def __init__(self, *,
                 model_Y_X,
                 model_T_X,
                 model_T_XZ,
                 model_final,
                 featurizer=None,
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 random_state=None):
        self.model_Y_X = clone(model_Y_X, safe=False)
        self.model_T_X = clone(model_T_X, safe=False)
        self.model_T_XZ = clone(model_T_XZ, safe=False)
        self.model_final = clone(model_final, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        super().__init__(cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         random_state=random_state)

    def _gen_ortho_learner_model_nuisance(self):
        return _BaseDMLIVModelNuisance(_FirstStageWrapper(clone(self.model_Y_X, safe=False), False),
                                       _FirstStageWrapper(clone(self.model_T_X, safe=False), self.discrete_treatment),
                                       _FirstStageWrapper(clone(self.model_T_XZ, safe=False), self.discrete_treatment))

    def _gen_ortho_learner_model_final(self):
        return _BaseDMLIVModelFinal(_FinalWrapper(clone(self.model_final, safe=False),
                                                  fit_cate_intercept=False,
                                                  featurizer=clone(self.featurizer, safe=False),
                                                  use_weight_trick=True))
