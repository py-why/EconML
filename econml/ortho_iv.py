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

from ._ortho_learner import _OrthoLearner
from .cate_estimator import LinearModelFinalCateEstimatorMixin, StatsModelsCateEstimatorMixin
from .dml import _FinalWrapper
from .inference import StatsModelsInference
from .sklearn_extensions.linear_model import StatsModelsLinearRegression
from .utilities import (_deprecate_positional, add_intercept, fit_with_groups, filter_none_kwargs,
                        hstack, inverse_onehot)


# A cut-down version of the DML first stage wrapper, since we don't need to support linear first stages
class _FirstStageWrapper:
    def __init__(self, model, discrete_target):
        self._model = clone(model, safe=False)
        self._discrete_target = discrete_target

    def _combine(self, X, W, Z, n_samples, fitting=True):
        # output is
        #   * a column of ones if X, W, and Z are all None
        #   * just X or W or Z if both of the others are None
        #   * hstack([arrs]) for whatever subset are not None otherwise

        # ensure Z is 2D
        if Z is not None:
            Z = Z.reshape(n_samples, -1)

        if X is None and W is None and Z is None:
            return np.ones((n_samples, 1))

        arrs = [arr for arr in [X, W, Z] if arr is not None]

        if len(arrs) == 1:
            return arrs[0]
        else:
            return hstack(arrs)

    def fit(self, *, X, W, Target, Z=None, sample_weight=None, groups=None):
        if self._discrete_target:
            # In this case, the Target is the one-hot-encoding of the treatment variable
            # We need to go back to the label representation of the one-hot so as to call
            # the classifier.
            if np.any(np.all(Target == 0, axis=0)) or (not np.any(np.all(Target == 0, axis=1))):
                raise AttributeError("Provided crossfit folds contain training splits that " +
                                     "don't contain all treatments")
            Target = inverse_onehot(Target)

        if sample_weight is not None:
            fit_with_groups(self._model, self._combine(X, W, Z, Target.shape[0]), Target,
                            groups=groups, sample_weight=sample_weight)
        else:
            fit_with_groups(self._model, self._combine(X, W, Z, Target.shape[0]), Target,
                            groups=groups)

    def score(self, *, X, W, Target, Z=None, sample_weight=None):
        if hasattr(self._model, 'score'):
            if self._discrete_target:
                # In this case, the Target is the one-hot-encoding of the treatment variable
                # We need to go back to the label representation of the one-hot so as to call
                # the classifier.
                if np.any(np.all(Target == 0, axis=0)) or (not np.any(np.all(Target == 0, axis=1))):
                    raise AttributeError("Provided crossfit folds contain training splits that " +
                                         "don't contain all treatments")
                Target = inverse_onehot(Target)

            if sample_weight is not None:
                return self._model.score(self._combine(X, W, Z, Target.shape[0]), Target, sample_weight=sample_weight)
            else:
                return self._model.score(self._combine(X, W, Z, Target.shape[0]), Target)
        else:
            return None

    def predict(self, X, W, Z=None):
        arrs = [arr for arr in [X, W, Z] if arr is not None]
        n_samples = arrs[0].shape[0] if arrs else 1
        if self._discrete_target:
            return self._model.predict_proba(self._combine(X, W, Z, n_samples, fitting=False))[:, 1:]
        else:
            return self._model.predict(self._combine(X, W, Z, n_samples, fitting=False))


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
                 n_splits=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        super().__init__(discrete_treatment=discrete_treatment,
                         discrete_instrument=discrete_instrument,
                         categories=categories,
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
            If groups is not None, the n_splits argument passed to this class's initializer
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

    def __init__(self, *, model_Y_W, model_T_W, model_Z_W,
                 discrete_treatment=False,
                 discrete_instrument=False,
                 categories='auto',
                 n_splits=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_Y_W = clone(model_Y_W, safe=False)
        self.model_T_W = clone(model_T_W, safe=False)
        self.model_Z_W = clone(model_Z_W, safe=False)
        super().__init__(discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
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

    def __init__(self, *, model_Y_W, model_T_W, model_T_WZ,
                 discrete_treatment=False,
                 discrete_instrument=False,
                 categories='auto',
                 n_splits=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_Y_W = clone(model_Y_W, safe=False)
        self.model_T_W = clone(model_T_W, safe=False)
        self.model_T_WZ = clone(model_T_WZ, safe=False)
        super().__init__(discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
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

    n_splits: int, cross-validation generator or an iterable, optional, default 2
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`cv splitter`
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
                 n_splits=2, mc_iters=None, mc_agg='mean', random_state=None):
        super().__init__(discrete_treatment=discrete_treatment,
                         discrete_instrument=discrete_instrument,
                         categories=categories,
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
            If groups is not None, the n_splits argument passed to this class's initializer
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
        return self.ortho_learner_model_final._model_final._original_featurizer

    @property
    def featurizer_(self):
        # NOTE This is used by the inference methods and has to be the overall featurizer. intended
        # for internal use by the library
        return self.ortho_learner_model_final._model_final._featurizer

    @property
    def model_final_(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        return self.ortho_learner_model_final._model_final._model

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
        return self.ortho_learner_model_final._model_final._model

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
        return [mdl._model_Y_X._model for mdl in super().models_nuisance]

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
        return [mdl._model_T_X._model for mdl in super().models_nuisance]

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
        return [mdl._model_T_XZ._model for mdl in super().models_nuisance]

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

    n_splits: int, cross-validation generator or an iterable, optional, default 2
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`cv splitter`
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

    def __init__(self, *, model_Y_X, model_T_X, model_T_XZ, model_final,
                 featurizer=None,
                 fit_cate_intercept=True,
                 n_splits=2,
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
        super().__init__(n_splits=n_splits,
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
        return self.ortho_learner_model_final._model_final._fit_cate_intercept

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final._model_final._fit_cate_intercept


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

    n_splits: int, cross-validation generator or an iterable, optional, default 2
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`cv splitter`
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

    def __init__(self, *, model_Y_X, model_T_X, model_T_XZ, model_final,
                 featurizer=None,
                 n_splits=2,
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
        super().__init__(n_splits=n_splits,
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


class _BaseDRIVModelFinal:
    """
    Final model at fit time, fits a residual on residual regression with a heterogeneous coefficient
    that depends on X, i.e.

        .. math ::
            Y - \\E[Y | X] = \\theta(X) \\cdot (\\E[T | X, Z] - \\E[T | X]) + \\epsilon

    and at predict time returns :math:`\\theta(X)`. The score method returns the MSE of this final
    residual on residual regression.
    """

    def __init__(self, model_final, featurizer,
                 discrete_treatment, discrete_instrument,
                 fit_cate_intercept, cov_clip, opt_reweighted):
        self._model_final = clone(model_final, safe=False)
        self._fit_cate_intercept = fit_cate_intercept
        self._original_featurizer = clone(featurizer, safe=False)
        self._discrete_treatment = discrete_treatment
        self._discrete_instrument = discrete_instrument
        if self._fit_cate_intercept:
            add_intercept_trans = FunctionTransformer(add_intercept,
                                                      validate=True)
            if featurizer:
                self._featurizer = Pipeline([('featurize', self._original_featurizer),
                                             ('add_intercept', add_intercept_trans)])
            else:
                self._featurizer = add_intercept_trans
        else:
            self._featurizer = self._original_featurizer
        self._cov_clip = cov_clip
        self._opt_reweighted = opt_reweighted

    def _effect_estimate(self, nuisances):
        prel_theta, res_t, res_y, res_z, cov = [nuisance.reshape(nuisances[0].shape) for nuisance in nuisances]

        # Estimate final model of theta(X) by minimizing the square loss:
        # (prel_theta(X) + (Y_res - prel_theta(X) * T_res) * Z_res / cov[T,Z | X] - theta(X))^2
        # We clip the covariance so that it is bounded away from zero, so as to reduce variance
        # at the expense of some small bias. For points with very small covariance we revert
        # to the model-based preliminary estimate and do not add the correction term.
        cov_sign = np.sign(cov)
        cov_sign[cov_sign == 0] = 1
        clipped_cov = cov_sign * np.clip(np.abs(cov),
                                         self._cov_clip, np.inf)
        return prel_theta + (res_y - prel_theta * res_t) * res_z / clipped_cov, clipped_cov

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        self.d_y = Y.shape[1:]
        self.d_t = nuisances[1].shape[1:]
        self.d_z = nuisances[3].shape[1:]

        # TODO: if opt_reweighted is False, we could change the logic to support multidimensional treatments,
        #       instruments, and outcomes
        if self.d_y and self.d_y[0] > 2:
            raise AttributeError("DRIV only supports a single outcome")

        if self.d_t and self.d_t[0] > 1:
            if self._discrete_treatment:
                raise AttributeError("DRIV only supports binary treatments")
            else:
                raise AttributeError("DRIV only supports single-dimensional continuous treatments")

        if self.d_z and self.d_z[0] > 1:
            if self._discrete_instrument:
                raise AttributeError("DRIV only supports binary instruments")
            else:
                raise AttributeError("DRIV only supports single-dimensional continuous instruments")

        theta_dr, clipped_cov = self._effect_estimate(nuisances)

        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.fit_transform(X)
        if self._opt_reweighted and (sample_weight is not None):
            sample_weight = sample_weight * clipped_cov.ravel()**2
        elif self._opt_reweighted:
            sample_weight = clipped_cov.ravel()**2
        self._model_final.fit(X, theta_dr, **filter_none_kwargs(sample_weight=sample_weight, sample_var=sample_var))

        return self

    def predict(self, X=None):
        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.transform(X)
        return self._model_final.predict(X).reshape((-1,) + self.d_y + self.d_t)

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None):
        theta_dr, clipped_cov = self._effect_estimate(nuisances)

        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.transform(X)

        if self._opt_reweighted and (sample_weight is not None):
            sample_weight = sample_weight * clipped_cov.ravel()**2
        elif self._opt_reweighted:
            sample_weight = clipped_cov.ravel()**2

        return np.average((theta_dr.ravel() - self._model_final.predict(X).ravel())**2,
                          weights=sample_weight, axis=0)


class _BaseDRIV(_OrthoLearner):

    """
    The _BaseDRIV algorithm for estimating CATE with IVs. It is the parent of the
    two public classes {DRIV, ProjectedDRIV}

    Parameters
    ----------
    nuisance_models : dictionary of nuisance models, with {'name_of_model' : EstimatorObject, ...}

    model_final : estimator
        model compatible with the sklearn regression API, used to fit the effect on X

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    cov_clip : float, optional, default 0.1
        clipping of the covariate for regions with low "overlap", to reduce variance

    opt_reweighted : bool, optional, default False
        Whether to reweight the samples to minimize variance. If True then
        model_final.fit must accept sample_weight as a kw argument. If True then
        assumes the model_final is flexible enough to fit the true CATE model. Otherwise,
        it method will return a biased projection to the model_final space, biased
        to give more weight on parts of the feature space where the instrument is strong.

    discrete_instrument: bool, optional, default False
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    discrete_treatment: bool, optional, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    n_splits: int, cross-validation generator or an iterable, optional, default 2
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`cv splitter`
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

    def __init__(self,
                 model_final,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=0.1,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 n_splits=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_final = clone(model_final, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.fit_cate_intercept = fit_cate_intercept
        self.cov_clip = cov_clip
        self.opt_reweighted = opt_reweighted
        super().__init__(discrete_instrument=discrete_instrument, discrete_treatment=discrete_treatment,
                         categories=categories, n_splits=n_splits,
                         mc_iters=mc_iters, mc_agg=mc_agg, random_state=random_state)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_ortho_learner_model_final(self):
        return _BaseDRIVModelFinal(self._gen_model_final(),
                                   clone(self.featurizer, safe=False),
                                   self.discrete_treatment,
                                   self.discrete_instrument,
                                   self.fit_cate_intercept,
                                   self.cov_clip,
                                   self.opt_reweighted)

    @_deprecate_positional("X, W, and Z should be passed by keyword only. In a future release "
                           "we will disallow passing X, W, and Z by position.", ['X', 'W', 'Z'])
    def fit(self, Y, T, Z, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
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
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the n_splits argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string,:class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of:class:`.BootstrapInference`).

        Returns
        -------
        self: _BaseDRIV instance
        """
        # Replacing fit from _OrthoLearner, to reorder arguments and improve the docstring
        return super().fit(Y, T, X=X, W=W, Z=Z,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    def score(self, Y, T, Z, X=None, W=None, sample_weight=None):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calls the score function of the model_final and returns the calculated score.
        The model_final model must have a score method.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix or None (Default=None)
            Instruments for each sample
        X: optional (n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples

        Returns
        -------
        score : float or (array of float)
            The score of the final CATE model on the new data. Same type as the return
            type of the model_final.score method.
        """
        # Replacing score from _OrthoLearner, to reorder arguments and improve the docstring
        return super().score(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight)

    @property
    def original_featurizer(self):
        return self.ortho_learner_model_final._original_featurizer

    @property
    def featurizer_(self):
        # NOTE This is used by the inference methods and has to be the overall featurizer. intended
        # for internal use by the library
        return self.ortho_learner_model_final._featurizer

    @property
    def model_final_(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        return self.ortho_learner_model_final._model_final

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


class _IntentToTreatDRIVModelNuisance:
    """
    Nuisance model fits the three models at fit time and at predict time
    returns :math:`Y-\\E[Y|X]` and :math:`\\E[T|X,Z]-\\E[T|X]` as residuals.
    """

    def __init__(self, model_Y_X, model_T_XZ, prel_model_effect):
        self._model_Y_X = clone(model_Y_X, safe=False)
        self._model_T_XZ = clone(model_T_XZ, safe=False)
        self._prel_model_effect = clone(prel_model_effect, safe=False)

    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        self._model_Y_X.fit(X=X, W=W, Target=Y, sample_weight=sample_weight, groups=groups)
        self._model_T_XZ.fit(X=X, W=W, Z=Z, Target=T, sample_weight=sample_weight, groups=groups)
        # we need to undo the one-hot encoding for calling effect,
        # since it expects raw values
        self._prel_model_effect.fit(Y, inverse_onehot(T), Z=inverse_onehot(Z), X=X, W=W,
                                    sample_weight=sample_weight, groups=groups)
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        if hasattr(self._model_Y_X, 'score'):
            Y_X_score = self._model_Y_X.score(X=X, W=W, Target=Y, sample_weight=sample_weight)
        else:
            Y_X_score = None
        if hasattr(self._model_T_XZ, 'score'):
            T_XZ_score = self._model_T_XZ.score(X=X, W=W, Z=Z, Target=T, sample_weight=sample_weight)
        else:
            T_XZ_score = None
        if hasattr(self._prel_model_effect, 'score'):
            # we need to undo the one-hot encoding for calling effect,
            # since it expects raw values
            effect_score = self._prel_model_effect.score(Y, inverse_onehot(T),
                                                         Z=inverse_onehot(Z), X=X, W=W, sample_weight=sample_weight)
        else:
            effect_score = None

        return Y_X_score, T_XZ_score, effect_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None):
        Y_pred = self._model_Y_X.predict(X, W)
        T_pred_zero = self._model_T_XZ.predict(X, W, np.zeros(Z.shape))
        T_pred_one = self._model_T_XZ.predict(X, W, np.ones(Z.shape))
        delta = (T_pred_one - T_pred_zero) / 2
        T_pred_mean = (T_pred_one + T_pred_zero) / 2
        prel_theta = self._prel_model_effect.effect(X)
        if X is None:  # In this case predict above returns a single row
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            prel_theta = np.tile(prel_theta.reshape(1, -1), (T.shape[0], 1))
        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = T - T_pred_mean.reshape(T.shape)
        return prel_theta, T_res, Y_res, 2 * Z - 1, delta


class _IntentToTreatDRIV(_BaseDRIV):
    """
    Helper class for the DRIV algorithm for the intent-to-treat A/B test setting
    """

    def __init__(self, *, model_Y_X,
                 model_T_XZ,
                 prel_model_effect,
                 model_final,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=.1,
                 n_splits=3,
                 mc_iters=None,
                 mc_agg='mean',
                 opt_reweighted=False,
                 categories='auto',
                 random_state=None):
        """
        """
        self.model_Y_X = clone(model_Y_X, safe=False)
        self.model_T_XZ = clone(model_T_XZ, safe=False)
        self.prel_model_effect = clone(prel_model_effect, safe=False)
        # TODO: check that Y, T, Z do not have multiple columns
        super().__init__(model_final,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         discrete_instrument=True,
                         discrete_treatment=True,
                         categories=categories,
                         opt_reweighted=opt_reweighted,
                         random_state=random_state)

    def _gen_prel_model_effect(self):
        return clone(self.prel_model_effect, safe=False)

    def _gen_ortho_learner_model_nuisance(self):
        return _IntentToTreatDRIVModelNuisance(_FirstStageWrapper(clone(self.model_Y_X, safe=False),
                                                                  discrete_target=False),
                                               _FirstStageWrapper(clone(self.model_T_XZ, safe=False),
                                                                  discrete_target=True),
                                               self._gen_prel_model_effect())


class _DummyCATE:
    """
    A dummy cate effect model that always returns zero effect
    """

    def __init__(self):
        return

    def fit(self, y, T, *, Z, X, W=None, sample_weight=None, groups=None, **kwargs):
        return self

    def effect(self, X):
        if X is None:
            return np.zeros(1)
        return np.zeros(X.shape[0])


class IntentToTreatDRIV(_IntentToTreatDRIV):
    """
    Implements the DRIV algorithm for the intent-to-treat A/B test setting

    Parameters
    ----------
    model_Y_X : estimator
        model to estimate :math:`\\E[Y | X]`.  Must support `fit` and `predict` methods.

    model_T_XZ : estimator
        model to estimate :math:`\\E[T | X, Z]`.  Must support `fit` and `predict_proba` methods.

    flexible_model_effect : estimator
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.

    final_model_effect : estimator, optional
        a final model for the CATE and projections. If None, then flexible_model_effect is also used as a final model

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    cov_clip : float, optional, default 0.1
        clipping of the covariate for regions with low "overlap", to reduce variance

    n_splits: int, cross-validation generator or an iterable, optional, default 3
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`cv splitter`
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

    opt_reweighted : bool, optional, default False
        Whether to reweight the samples to minimize variance. If True then
        final_model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
        utilities can be used for any linear model to enable sample_weights). If True then
        assumes the final_model_effect is flexible enough to fit the true CATE model. Otherwise,
        it method will return a biased projection to the model_effect space, biased
        to give more weight on parts of the feature space where the instrument is strong.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, *, model_Y_X, model_T_XZ,
                 flexible_model_effect,
                 model_final=None,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=.1,
                 n_splits=3,
                 mc_iters=None,
                 mc_agg='mean',
                 opt_reweighted=False,
                 categories='auto',
                 random_state=None):
        self.flexible_model_effect = clone(flexible_model_effect, safe=False)
        super().__init__(model_Y_X=model_Y_X,
                         model_T_XZ=model_T_XZ,
                         prel_model_effect=None,
                         model_final=model_final,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         opt_reweighted=opt_reweighted,
                         categories=categories,
                         random_state=random_state)

    def _gen_model_final(self):
        if self.model_final is None:
            return clone(self.flexible_model_effect, safe=False)
        return clone(self.model_final, safe=False)

    def _gen_prel_model_effect(self):
        return _IntentToTreatDRIV(model_Y_X=clone(self.model_Y_X, safe=False),
                                  model_T_XZ=clone(self.model_T_XZ, safe=False),
                                  prel_model_effect=_DummyCATE(),
                                  model_final=clone(self.flexible_model_effect, safe=False),
                                  cov_clip=1e-7,
                                  n_splits=1,
                                  opt_reweighted=True,
                                  random_state=self.random_state)

    @property
    def models_Y_X(self):
        return [mdl._model_Y_X._model for mdl in super().models_nuisance]

    @property
    def models_T_XZ(self):
        return [mdl._model_T_XZ._model for mdl in super().models_nuisance]

    @property
    def nuisance_scores_Y_X(self):
        return self.nuisance_scores_[0]

    @property
    def nuisance_scores_T_XZ(self):
        return self.nuisance_scores_[1]

    @property
    def nuisance_scores_effect(self):
        return self.nuisance_scores_[2]

    @property
    def prel_model_effect(self):
        return self._gen_prel_model_effect()

    @prel_model_effect.setter
    def prel_model_effect(self, value):
        if value is not None:
            raise ValueError("Parameter `prel_model_effect` cannot be altered for this estimator.")


class LinearIntentToTreatDRIV(StatsModelsCateEstimatorMixin, IntentToTreatDRIV):
    """
    Implements the DRIV algorithm for the intent-to-treat A/B test setting

    Parameters
    ----------
    model_Y_X : estimator
        model to estimate :math:`\\E[Y | X]`.  Must support `fit` and `predict` methods.

    model_T_XZ : estimator
        model to estimate :math:`\\E[T | X, Z]`.  Must support `fit` and `predict_proba` methods.

    flexible_model_effect : estimator
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    cov_clip : float, optional, default 0.1
        clipping of the covariate for regions with low "overlap", to reduce variance

    n_splits: int, cross-validation generator or an iterable, optional, default 3
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`cv splitter`
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

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, *, model_Y_X,
                 model_T_XZ,
                 flexible_model_effect,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=.1,
                 n_splits=3,
                 mc_iters=None,
                 mc_agg='mean',
                 categories='auto',
                 random_state=None):
        super().__init__(model_Y_X=model_Y_X,
                         model_T_XZ=model_T_XZ,
                         flexible_model_effect=flexible_model_effect,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         model_final=None,
                         cov_clip=cov_clip,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         opt_reweighted=False,
                         categories=categories, random_state=random_state)

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=False)

    # override only so that we can update the docstring to indicate support for `StatsModelsInference`
    @_deprecate_positional("X, W, and Z should be passed by keyword only. In a future release "
                           "we will disallow passing X, W, and Z by position.", ['X', 'W', 'Z'])
    def fit(self, Y, T, Z, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix or vector of length n
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the n_splits argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string,:class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of:class:`.BootstrapInference`) and 'statsmodels'
            (or an instance of :class:`.StatsModelsInference`).

        Returns
        -------
        self : instance
        """
        return super().fit(Y, T, Z=Z, X=X, W=W,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    @property
    def bias_part_of_coef(self):
        return self.ortho_learner_model_final._fit_cate_intercept

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final._fit_cate_intercept

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, value):
        if value is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator.")

    @property
    def opt_reweighted(self):
        return False

    @opt_reweighted.setter
    def opt_reweighted(self, value):
        if not (value is False):
            raise ValueError("Parameter `value` cannot be altered from `False` for this estimator.")
