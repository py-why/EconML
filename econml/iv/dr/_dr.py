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
from ...utilities import (_deprecate_positional, add_intercept, filter_none_kwargs,
                          inverse_onehot)
from .._nuisance_wrappers import _FirstStageWrapper, _FinalWrapper


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

    def __init__(self, *,
                 model_final,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=0.1,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 n_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_final = clone(model_final, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.fit_cate_intercept = fit_cate_intercept
        self.cov_clip = cov_clip
        self.opt_reweighted = opt_reweighted
        super().__init__(discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

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
            If groups is not None, the `cv` argument passed to this class's initializer
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
        return self.ortho_learner_model_final_._original_featurizer

    @property
    def featurizer_(self):
        # NOTE This is used by the inference methods and has to be the overall featurizer. intended
        # for internal use by the library
        return self.ortho_learner_model_final_._featurizer

    @property
    def model_final_(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        return self.ortho_learner_model_final_._model_final

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

    def __init__(self, *,
                 model_Y_X,
                 model_T_XZ,
                 prel_model_effect,
                 model_final,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=.1,
                 cv=3,
                 n_splits='raise',
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
        super().__init__(model_final=model_final,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         cv=cv,
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

    cv: int, cross-validation generator or an iterable, optional, default 3
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

    def __init__(self, *,
                 model_Y_X,
                 model_T_XZ,
                 flexible_model_effect,
                 model_final=None,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=.1,
                 cv=3,
                 n_splits='raise',
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
                         cv=cv,
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
                                  cv=1,
                                  opt_reweighted=True,
                                  random_state=self.random_state)

    @property
    def models_Y_X(self):
        return [mdl._model_Y_X._model for mdl in super().models_nuisance_]

    @property
    def models_T_XZ(self):
        return [mdl._model_T_XZ._model for mdl in super().models_nuisance_]

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

    cv: int, cross-validation generator or an iterable, optional, default 3
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
                 model_T_XZ,
                 flexible_model_effect,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=.1,
                 cv=3,
                 n_splits='raise',
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
                         cv=cv,
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
            If groups is not None, the `cv` argument passed to this class's initializer
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
        return self.ortho_learner_model_final_._fit_cate_intercept

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final_._fit_cate_intercept

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
