# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Doubly Robust IV for Heterogeneous Treatment Effects.

An Doubly Robust machine learning approach to estimation of heterogeneous
treatment effect with an endogenous treatment and an instrument.

Implements the DRIV algorithm for estimating CATE with IVs from the paper:

Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments
Vasilis Syrgkanis, Victor Lei, Miruna Oprescu, Maggie Hei, Keith Battocchi, Greg Lewis
https://arxiv.org/abs/1905.10176
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.dummy import DummyClassifier


from ..._ortho_learner import _OrthoLearner
from ..._cate_estimator import (StatsModelsCateEstimatorMixin, DebiasedLassoCateEstimatorMixin,
                                ForestModelFinalCateEstimatorMixin, GenericSingleTreatmentModelFinalInference,
                                LinearCateEstimator)
from ...inference import StatsModelsInference
from ...sklearn_extensions.linear_model import StatsModelsLinearRegression, DebiasedLasso, WeightedLassoCVWrapper
from ...sklearn_extensions.model_selection import WeightedStratifiedKFold
from ...utilities import (_deprecate_positional, add_intercept, filter_none_kwargs,
                          inverse_onehot, get_feature_names_or_default, check_high_dimensional, check_input_arrays)
from ...grf import RegressionForest
from ...dml.dml import _FirstStageWrapper, _FinalWrapper
from ...iv.dml import NonParamDMLIV
from ..._shap import _shap_explain_model_cate


class _BaseDRIVModelNuisance:
    def __init__(self, prel_model_effect, model_y_xw, model_t_xw, model_tz_xw, model_z, projection,
                 discrete_treatment, discrete_instrument):
        self._prel_model_effect = prel_model_effect
        self._model_y_xw = model_y_xw
        self._model_t_xw = model_t_xw
        self._model_tz_xw = model_tz_xw
        self._projection = projection
        self._discrete_treatment = discrete_treatment
        self._discrete_instrument = discrete_instrument
        if self._projection:
            self._model_t_xwz = model_z
        else:
            self._model_z_xw = model_z

    def _combine(self, W, Z, n_samples):
        if Z is not None:  # Z will not be None
            Z = Z.reshape(n_samples, -1)
            return Z if W is None else np.hstack([W, Z])
        return None if W is None else W

    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        # T and Z only allow single continuous or binary, keep the shape of (n,) for continuous and (n,1) for binary
        T = T.ravel() if not self._discrete_treatment else T
        Z = Z.ravel() if not self._discrete_instrument else Z

        self._model_y_xw.fit(X=X, W=W, Target=Y, sample_weight=sample_weight, groups=groups)
        self._model_t_xw.fit(X=X, W=W, Target=T, sample_weight=sample_weight, groups=groups)
        if self._projection:
            WZ = self._combine(W, Z, Y.shape[0])
            self._model_t_xwz.fit(X=X, W=WZ, Target=T, sample_weight=sample_weight, groups=groups)
            # fit on projected Z: E[T * E[T|X,Z]|X]
            T_proj = self._model_t_xwz.predict(X, WZ).reshape(T.shape)
            # if discrete, return shape (n,1); if continuous return shape (n,)
            target = (T * T_proj).reshape(T.shape[0],)
            self._model_tz_xw.fit(X=X, W=W, Target=target,
                                  sample_weight=sample_weight, groups=groups)
        else:
            self._model_z_xw.fit(X=X, W=W, Target=Z, sample_weight=sample_weight, groups=groups)
            if self._discrete_treatment:
                if self._discrete_instrument:
                    # target will be discrete and will be inversed from FirstStageWrapper, shape (n,1)
                    target = T * Z
                else:
                    # shape (n,)
                    target = inverse_onehot(T) * Z
            else:
                if self._discrete_instrument:
                    # shape (n,)
                    target = T * inverse_onehot(Z)
                else:
                    # shape(n,)
                    target = T * Z
            self._model_tz_xw.fit(X=X, W=W, Target=target,
                                  sample_weight=sample_weight, groups=groups)

        # TODO: prel_model_effect could allow sample_var and freq_weight?
        if self._discrete_instrument:
            Z = inverse_onehot(Z)
        if self._discrete_treatment:
            T = inverse_onehot(T)
        self._prel_model_effect.fit(Y, T, Z=Z, X=X,
                                    W=W, sample_weight=sample_weight, groups=groups)
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        # T and Z only allow single continuous or binary, keep the shape of (n,) for continuous and (n,1) for binary
        T = T.ravel() if not self._discrete_treatment else T
        Z = Z.ravel() if not self._discrete_instrument else Z

        if hasattr(self._model_y_xw, 'score'):
            y_xw_score = self._model_y_xw.score(X=X, W=W, Target=Y, sample_weight=sample_weight)
        else:
            y_xw_score = None
        if hasattr(self._model_t_xw, 'score'):
            t_xw_score = self._model_t_xw.score(X=X, W=W, Target=T, sample_weight=sample_weight)
        else:
            t_xw_score = None

        if hasattr(self._prel_model_effect, 'score'):
            # we need to undo the one-hot encoding for calling effect,
            # since it expects raw values
            raw_T = inverse_onehot(T) if self._discrete_treatment else T
            raw_Z = inverse_onehot(Z) if self._discrete_instrument else Z
            effect_score = self._prel_model_effect.score(Y, raw_T,
                                                         Z=raw_Z, X=X, W=W, sample_weight=sample_weight)
        else:
            effect_score = None

        if self._projection:
            if hasattr(self._model_t_xwz, 'score'):
                WZ = self._combine(W, Z, Y.shape[0])
                t_xwz_score = self._model_t_xwz.score(X=X, W=WZ, Target=T, sample_weight=sample_weight)
            else:
                t_xwz_score = None

            if hasattr(self._model_tz_xw, 'score'):
                T_proj = self._model_t_xwz.predict(X, WZ).reshape(T.shape)
                # if discrete, return shape (n,1); if continuous return shape (n,)
                target = (T * T_proj).reshape(T.shape[0],)
                tz_xw_score = self._model_tz_xw.score(X=X, W=W, Target=target, sample_weight=sample_weight)
            else:
                tz_xw_score = None
            return y_xw_score, t_xw_score, t_xwz_score, tz_xw_score, effect_score

        else:
            if hasattr(self._model_z_xw, 'score'):
                z_xw_score = self._model_z_xw.score(X=X, W=W, Target=Z, sample_weight=sample_weight)
            else:
                z_xw_score = None

            if hasattr(self._model_tz_xw, 'score'):
                if self._discrete_treatment:
                    if self._discrete_instrument:
                        # target will be discrete and will be inversed from FirstStageWrapper
                        target = T * Z
                    else:
                        target = inverse_onehot(T) * Z
                else:
                    if self._discrete_instrument:
                        target = T * inverse_onehot(Z)
                    else:
                        target = T * Z
                tz_xw_score = self._model_tz_xw.score(X=X, W=W, Target=target, sample_weight=sample_weight)
            else:
                tz_xw_score = None

            return y_xw_score, t_xw_score, z_xw_score, tz_xw_score, effect_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        Y_pred = self._model_y_xw.predict(X, W)
        T_pred = self._model_t_xw.predict(X, W)
        TZ_pred = self._model_tz_xw.predict(X, W)
        prel_theta = self._prel_model_effect.effect(X)

        if X is None:
            prel_theta = np.tile(prel_theta.reshape(1, -1), (Y.shape[0], 1))
            if W is None:
                Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
                T_pred = np.tile(T_pred.reshape(1, -1), (Y.shape[0], 1))
                TZ_pred = np.tile(TZ_pred.reshape(1, -1), (Y.shape[0], 1))

        # for convenience, reshape Z,T to a vector since they are either binary or single dimensional continuous
        T = T.reshape(T.shape[0],)
        Z = Z.reshape(Z.shape[0],)
        # reshape the predictions
        Y_pred = Y_pred.reshape(Y.shape)
        T_pred = T_pred.reshape(T.shape)
        TZ_pred = TZ_pred.reshape(T.shape)

        Y_res = Y - Y_pred
        T_res = T - T_pred

        if self._projection:
            # concat W and Z
            WZ = self._combine(W, Z, Y.shape[0])
            T_proj = self._model_t_xwz.predict(X, WZ).reshape(T.shape)
            Z_res = T_proj - T_pred
            cov = TZ_pred - T_pred**2
        else:
            Z_pred = self._model_z_xw.predict(X, W)
            if X is None and W is None:
                Z_pred = np.tile(Z_pred.reshape(1, -1), (Z.shape[0], 1))
            Z_pred = Z_pred.reshape(Z.shape)
            Z_res = Z - Z_pred
            cov = TZ_pred - T_pred * Z_pred

        # check nuisances outcome shape
        # Y_res could be a vector or 1-dimensional 2d-array
        assert T_res.ndim == 1, "Nuisance outcome should be vector!"
        assert Z_res.ndim == 1, "Nuisance outcome should be vector!"
        assert cov.ndim == 1, "Nuisance outcome should be vector!"

        return prel_theta, Y_res, T_res, Z_res, cov


class _BaseDRIVModelFinal:
    def __init__(self, model_final, featurizer, fit_cate_intercept, cov_clip, opt_reweighted):
        self._model_final = clone(model_final, safe=False)
        self._original_featurizer = clone(featurizer, safe=False)
        self._fit_cate_intercept = fit_cate_intercept
        self._cov_clip = cov_clip
        self._opt_reweighted = opt_reweighted

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

    def _effect_estimate(self, nuisances):
        # all could be reshaped to vector since Y, T, Z are all single dimensional.
        prel_theta, res_y, res_t, res_z, cov = [nuisance.reshape(nuisances[0].shape[0]) for nuisance in nuisances]

        # Estimate final model of theta(X) by minimizing the square loss:
        # (prel_theta(X) + (Y_res - prel_theta(X) * T_res) * Z_res / cov[T,Z | X] - theta(X))^2
        # We clip the covariance so that it is bounded away from zero, so as to reduce variance
        # at the expense of some small bias. For points with very small covariance we revert
        # to the model-based preliminary estimate and do not add the correction term.
        cov_sign = np.sign(cov)
        cov_sign[cov_sign == 0] = 1
        clipped_cov = cov_sign * np.clip(np.abs(cov),
                                         self._cov_clip, np.inf)
        return prel_theta + (res_y - prel_theta * res_t) * res_z / clipped_cov, clipped_cov, res_z

    def _transform_X(self, X, n=1, fitting=True):
        if X is not None:
            if self._featurizer is not None:
                F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
            else:
                F = X
        else:
            if not self._fit_cate_intercept:
                raise AttributeError("Cannot have X=None and also not allow for a CATE intercept!")
            F = np.ones((n, 1))
        return F

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None,
            sample_weight=None, freq_weight=None, sample_var=None, groups=None):
        self.d_y = Y.shape[1:]
        self.d_t = T.shape[1:]
        theta_dr, clipped_cov, res_z = self._effect_estimate(nuisances)

        X = self._transform_X(X, n=theta_dr.shape[0])
        if self._opt_reweighted and (sample_weight is not None):
            sample_weight = sample_weight * clipped_cov.ravel()**2
        elif self._opt_reweighted:
            sample_weight = clipped_cov.ravel()**2
        target_var = sample_var * (res_z**2 / clipped_cov**2) if sample_var is not None else None
        self._model_final.fit(X, theta_dr, **filter_none_kwargs(sample_weight=sample_weight,
                                                                freq_weight=freq_weight, sample_var=target_var))
        return self

    def predict(self, X=None):
        X = self._transform_X(X, fitting=False)
        return self._model_final.predict(X).reshape((-1,) + self.d_y + self.d_t)

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None):
        theta_dr, clipped_cov, _ = self._effect_estimate(nuisances)

        X = self._transform_X(X, fitting=False)

        if self._opt_reweighted and (sample_weight is not None):
            sample_weight = sample_weight * clipped_cov.ravel()**2
        elif self._opt_reweighted:
            sample_weight = clipped_cov.ravel()**2

        return np.average((theta_dr.ravel() - self._model_final.predict(X).ravel())**2,
                          weights=sample_weight, axis=0)


class _BaseDRIV(_OrthoLearner):
    # A helper class that access all the internal fitted objects of a DRIV Cate Estimator.
    # Used by both DRIV and IntentToTreatDRIV.
    def __init__(self, *,
                 model_final,
                 featurizer=None,
                 fit_cate_intercept=False,
                 cov_clip=1e-3,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
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
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    # Maggie: I think that would be the case?
    def _get_inference_options(self):
        options = super()._get_inference_options()
        options.update(auto=GenericSingleTreatmentModelFinalInference)
        return options

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_ortho_learner_model_final(self):
        return _BaseDRIVModelFinal(self._gen_model_final(), self._gen_featurizer(), self.fit_cate_intercept,
                                   self.cov_clip, self.opt_reweighted)

    def _check_inputs(self, Y, T, Z, X, W):
        Y1, T1, Z1, = check_input_arrays(Y, T, Z)
        if len(Y1.shape) > 1 and Y1.shape[1] > 1:
            raise AssertionError("DRIV only supports single dimensional outcome")
        if len(T1.shape) > 1 and T1.shape[1] > 1:
            if self.discrete_treatment:
                raise AttributeError("DRIV only supports binary treatments")
            else:
                raise AttributeError("DRIV only supports single-dimensional continuous treatments")
        if len(Z1.shape) > 1 and Z1.shape[1] > 1:
            if self.discrete_instrument:
                raise AttributeError("DRIV only supports binary instruments")
            else:
                raise AttributeError("DRIV only supports single-dimensional continuous instruments")
        return Y, T, Z, X, W

    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference="auto"):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight : (n,) array like, default None
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array like of integers, default None
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : (n,) nd array like, default None
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.GenericSingleTreatmentModelFinalInference`)

        Returns
        -------
        self
        """
        Y, T, Z, X, W = self._check_inputs(Y, T, Z, X, W)
        # Replacing fit from _OrthoLearner, to reorder arguments and improve the docstring
        return super().fit(Y, T, X=X, W=W, Z=Z,
                           sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    def score(self, Y, T, Z, X=None, W=None, sample_weight=None):
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
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples


        Returns
        -------
        score: float
            The MSE of the final CATE model on the new data.
        """
        # Replacing score from _OrthoLearner, to enforce Z to be required and improve the docstring
        return super().score(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight)

    @property
    def featurizer_(self):
        """
        Get the fitted featurizer.

        Returns
        -------
        featurizer: object of type(`featurizer`)
            An instance of the fitted featurizer that was used to preprocess X in the final CATE model training.
            Available only when featurizer is not None and X is not None.
        """
        return self.ortho_learner_model_final_._featurizer

    @property
    def original_featurizer(self):
        # NOTE: important to use the ortho_learner_model_final_ attribute instead of the
        #       attribute so that the trained featurizer will be passed through
        return self.ortho_learner_model_final_._original_featurizer

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
            final CATE model for each treatment is linear. It is the names of the features that are associated
            with each entry of the :meth:`coef_` parameter. Available only when the featurizer is not None and has
            a method: `get_feature_names(feature_names)`. Otherwise None is returned.
        """
        if self._d_x is None:
            # Handles the corner case when X=None but featurizer might be not None
            return None
        if feature_names is None:
            feature_names = self._input_names["feature_names"]
        if self.original_featurizer is None:
            return feature_names
        return get_feature_names_or_default(self.original_featurizer, feature_names)

    @property
    def model_final_(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        return self.ortho_learner_model_final_._model_final

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
        return self.ortho_learner_model_final_._model_final

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        return _shap_explain_model_cate(self.const_marginal_effect, self.model_cate, X, self._d_t, self._d_y,
                                        featurizer=self.featurizer_,
                                        feature_names=feature_names,
                                        treatment_names=treatment_names,
                                        output_names=output_names,
                                        input_names=self._input_names,
                                        background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__

    @property
    def residuals_(self):
        """
        A tuple (prel_theta, Y_res, T_res, Z_res, cov, X, W, Z), of the residuals from the first stage estimation
        along with the associated X, W and Z. Samples are not guaranteed to be in the same
        order as the input order.
        """
        if not hasattr(self, '_cached_values'):
            raise AttributeError("Estimator is not fitted yet!")
        if self._cached_values is None:
            raise AttributeError("`fit` was called with `cache_values=False`. "
                                 "Set to `True` to enable residual storage.")
        prel_theta, Y_res, T_res, Z_res, cov = self._cached_values.nuisances
        return (prel_theta, Y_res, T_res, Z_res, cov, self._cached_values.X, self._cached_values.W,
                self._cached_values.Z)


class _DRIV(_BaseDRIV):
    """
    Private Base class for the DRIV algorithm.
    """

    def __init__(self, *,
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_z_xw="auto",
                 model_t_xwz="auto",
                 model_tz_xw="auto",
                 prel_model_effect,
                 model_final,
                 projection=False,
                 featurizer=None,
                 fit_cate_intercept=False,
                 cov_clip=1e-3,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_y_xw = clone(model_y_xw, safe=False)
        self.model_t_xw = clone(model_t_xw, safe=False)
        self.model_t_xwz = clone(model_t_xwz, safe=False)
        self.model_z_xw = clone(model_z_xw, safe=False)
        self.model_tz_xw = clone(model_tz_xw, safe=False)
        self.prel_model_effect = clone(prel_model_effect, safe=False)
        self.projection = projection
        super().__init__(model_final=model_final,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         opt_reweighted=opt_reweighted,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_prel_model_effect(self):
        return clone(self.prel_model_effect, safe=False)

    def _gen_ortho_learner_model_nuisance(self):
        if self.model_y_xw == 'auto':
            model_y_xw = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_y_xw = clone(self.model_y_xw, safe=False)

        if self.model_t_xw == 'auto':
            if self.discrete_treatment:
                model_t_xw = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                                  random_state=self.random_state)
            else:
                model_t_xw = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_t_xw = clone(self.model_t_xw, safe=False)

        if self.projection:
            # this is a regression model since proj_t is probability
            if self.model_tz_xw == "auto":
                model_tz_xw = WeightedLassoCVWrapper(random_state=self.random_state)
            else:
                model_tz_xw = clone(self.model_tz_xw, safe=False)

            if self.model_t_xwz == 'auto':
                if self.discrete_treatment:
                    model_t_xwz = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                                       random_state=self.random_state)
                else:
                    model_t_xwz = WeightedLassoCVWrapper(random_state=self.random_state)
            else:
                model_t_xwz = clone(self.model_t_xwz, safe=False)

            return _BaseDRIVModelNuisance(self._gen_prel_model_effect(),
                                          _FirstStageWrapper(model_y_xw, True, self._gen_featurizer(), False, False),
                                          _FirstStageWrapper(model_t_xw, False, self._gen_featurizer(),
                                                             False, self.discrete_treatment),
                                          # outcome is continuous since proj_t is probability
                                          _FirstStageWrapper(model_tz_xw, False, self._gen_featurizer(), False,
                                                             False),
                                          _FirstStageWrapper(model_t_xwz, False, self._gen_featurizer(),
                                                             False, self.discrete_treatment),
                                          self.projection, self.discrete_treatment, self.discrete_instrument)

        else:
            if self.model_tz_xw == "auto":
                if self.discrete_treatment and self.discrete_instrument:
                    model_tz_xw = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                                       random_state=self.random_state)
                else:
                    model_tz_xw = WeightedLassoCVWrapper(random_state=self.random_state)
            else:
                model_tz_xw = clone(self.model_tz_xw, safe=False)

            if self.model_z_xw == 'auto':
                if self.discrete_instrument:
                    model_z_xw = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                                      random_state=self.random_state)
                else:
                    model_z_xw = WeightedLassoCVWrapper(random_state=self.random_state)
            else:
                model_z_xw = clone(self.model_z_xw, safe=False)

            return _BaseDRIVModelNuisance(self._gen_prel_model_effect(),
                                          _FirstStageWrapper(model_y_xw, True, self._gen_featurizer(), False, False),
                                          _FirstStageWrapper(model_t_xw, False, self._gen_featurizer(),
                                                             False, self.discrete_treatment),
                                          _FirstStageWrapper(model_tz_xw, False, self._gen_featurizer(), False,
                                                             self.discrete_treatment and self.discrete_instrument),
                                          _FirstStageWrapper(model_z_xw, False, self._gen_featurizer(),
                                                             False, self.discrete_instrument),
                                          self.projection, self.discrete_treatment, self.discrete_instrument)


class DRIV(_DRIV):
    """
    The DRIV algorithm for estimating CATE with IVs. It is the parent of the
    public classes {LinearDRIV, SparseLinearDRIV,ForestDRIV}

    Parameters
    ----------
    model_y_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Y | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_z_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument.

    model_t_xwz : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W, Z]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_tz_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T*Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument and discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument or continuous treatment.

    flexible_model_effect : estimator or 'auto' (default is 'auto')
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.
        If 'auto', :class:`.StatsModelsLinearRegression` will be applied.

    model_final : estimator, optional
        a final model for the CATE and projections. If None, then flexible_model_effect is also used as a final model

    prel_cate_approach : one of {'driv', 'dmliv'}, optional (default='driv')
        model that estimates a preliminary version of the CATE.
        If 'driv', :class:`._DRIV` will be used.
        If 'dmliv', :class:`.NonParamDMLIV` will be used

    prel_cv : int, cross-validation generator or an iterable, optional, default 1
        Determines the cross-validation splitting strategy for the preliminary effect model.

    prel_opt_reweighted : bool, optional, default True
        Whether to reweight the samples to minimize variance for the preliminary effect model.

    projection: bool, optional, default False
        If True, we fit a slight variant of DRIV where we use E[T|X, W, Z] as the instrument as opposed to Z,
        model_z_xw will be disabled; If False, model_t_xwz will be disabled.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default False
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
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_z_xw="auto",
                 model_t_xwz="auto",
                 model_tz_xw="auto",
                 flexible_model_effect="auto",
                 model_final=None,
                 prel_cate_approach="driv",
                 prel_cv=1,
                 prel_opt_reweighted=True,
                 projection=False,
                 featurizer=None,
                 fit_cate_intercept=False,
                 cov_clip=1e-3,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):

        if flexible_model_effect == "auto":
            self.flexible_model_effect = StatsModelsLinearRegression(fit_intercept=False)
        else:
            self.flexible_model_effect = clone(flexible_model_effect, safe=False)
        self.prel_cate_approach = prel_cate_approach
        self.prel_cv = prel_cv
        self.prel_opt_reweighted = prel_opt_reweighted
        super().__init__(model_y_xw=model_y_xw,
                         model_t_xw=model_t_xw,
                         model_z_xw=model_z_xw,
                         model_t_xwz=model_t_xwz,
                         model_tz_xw=model_tz_xw,
                         prel_model_effect=self.prel_cate_approach,
                         model_final=model_final,
                         projection=projection,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         opt_reweighted=opt_reweighted,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        if self.model_final is None:
            return clone(self.flexible_model_effect, safe=False)
        return clone(self.model_final, safe=False)

    def _gen_prel_model_effect(self):
        if self.prel_cate_approach == "driv":
            return _DRIV(model_y_xw=clone(self.model_y_xw, safe=False),
                         model_t_xw=clone(self.model_t_xw, safe=False),
                         model_z_xw=clone(self.model_z_xw, safe=False),
                         model_t_xwz=clone(self.model_t_xwz, safe=False),
                         model_tz_xw=clone(self.model_tz_xw, safe=False),
                         prel_model_effect=_DummyCATE(),
                         model_final=clone(self.flexible_model_effect, safe=False),
                         projection=self.projection,
                         featurizer=self._gen_featurizer(),
                         fit_cate_intercept=self.fit_cate_intercept,
                         cov_clip=self.cov_clip,
                         opt_reweighted=self.prel_opt_reweighted,
                         discrete_instrument=self.discrete_instrument,
                         discrete_treatment=self.discrete_treatment,
                         categories=self.categories,
                         cv=self.prel_cv,
                         mc_iters=self.mc_iters,
                         mc_agg=self.mc_agg,
                         random_state=self.random_state)
        elif self.prel_cate_approach == "dmliv":
            return NonParamDMLIV(model_y_xw=clone(self.model_y_xw, safe=False),
                                 model_t_xw=clone(self.model_t_xw, safe=False),
                                 model_t_xwz=clone(self.model_t_xwz, safe=False),
                                 model_final=clone(self.flexible_model_effect, safe=False),
                                 discrete_instrument=self.discrete_instrument,
                                 discrete_treatment=self.discrete_treatment,
                                 featurizer=self._gen_featurizer(),
                                 categories=self.categories,
                                 cv=self.prel_cv,
                                 mc_iters=self.mc_iters,
                                 mc_agg=self.mc_agg,
                                 random_state=self.random_state)
        else:
            raise ValueError(
                "We only support 'dmliv' or 'driv' preliminary model effect, "
                f"but received '{self.prel_cate_approach}'!")

    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference="auto"):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight : (n,) array like, default None
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array like of integers, default None
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : (n,) nd array like, default None
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.GenericSingleTreatmentModelFinalInference`)

        Returns
        -------
        self
        """
        if self.projection:
            assert self.model_z_xw == "auto", ("In the case of projection=True, model_z_xw will not be fitted, "
                                               "please keep it as default!")
        if self.prel_cate_approach == "driv" and not self.projection:
            assert self.model_t_xwz == "auto", ("In the case of projection=False and prel_cate_approach='driv', "
                                                "model_t_xwz will not be fitted, "
                                                "please keep it as default!")
        return super().fit(Y, T, X=X, W=W, Z=Z,
                           sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def models_y_xw(self):
        """
        Get the fitted models for :math:`\\E[Y | X]`.

        Returns
        -------
        models_y_xw: nested list of objects of type(`model_y_xw`)
            A nested list of instances of the `model_y_xw` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._model_y_xw._model for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_t_xw(self):
        """
        Get the fitted models for :math:`\\E[T | X]`.

        Returns
        -------
        models_t_xw: nested list of objects of type(`model_t_xw`)
            A nested list of instances of the `model_t_xw` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._model_t_xw._model for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_z_xw(self):
        """
        Get the fitted models for :math:`\\E[Z | X]`.

        Returns
        -------
        models_z_xw: nested list of objects of type(`model_z_xw`)
            A nested list of instances of the `model_z_xw` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        if self.projection:
            raise AttributeError("Projection model is fitted for instrument! Use models_t_xwz.")
        return [[mdl._model_z_xw._model for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_t_xwz(self):
        """
        Get the fitted models for :math:`\\E[Z | X]`.

        Returns
        -------
        models_z_xw: nested list of objects of type(`model_z_xw`)
            A nested list of instances of the `model_z_xw` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        if not self.projection:
            raise AttributeError("Direct model is fitted for instrument! Use models_z_xw.")
        return [[mdl._model_t_xwz._model for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_tz_xw(self):
        """
        Get the fitted models for :math:`\\E[T*Z | X]`.

        Returns
        -------
        models_tz_xw: nested list of objects of type(`model_tz_xw`)
            A nested list of instances of the `model_tz_xw` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._model_tz_xw._model for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_prel_model_effect(self):
        """
        Get the fitted preliminary CATE estimator.

        Returns
        -------
        prel_model_effect: nested list of objects of type(`prel_model_effect`)
            A nested list of instances of the `prel_model_effect` object. Number of sublist equals to number
            of monte carlo iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._prel_model_effect for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def nuisance_scores_y_xw(self):
        """
        Get the scores for y_xw model on the out-of-sample training data
        """
        return self.nuisance_scores_[0]

    @property
    def nuisance_scores_t_xw(self):
        """
        Get the scores for t_xw model on the out-of-sample training data
        """
        return self.nuisance_scores_[1]

    @property
    def nuisance_scores_z_xw(self):
        """
        Get the scores for z_xw model on the out-of-sample training data
        """
        if self.projection:
            raise AttributeError("Projection model is fitted for instrument! Use nuisance_scores_t_xwz.")
        return self.nuisance_scores_[2]

    @property
    def nuisance_scores_t_xwz(self):
        """
        Get the scores for z_xw model on the out-of-sample training data
        """
        if not self.projection:
            raise AttributeError("Direct model is fitted for instrument! Use nuisance_scores_z_xw.")
        return self.nuisance_scores_[2]

    @property
    def nuisance_scores_tz_xw(self):
        """
        Get the scores for tz_xw model on the out-of-sample training data
        """
        return self.nuisance_scores_[3]

    @property
    def nuisance_scores_prel_model_effect(self):
        """
        Get the scores for prel_model_effect model on the out-of-sample training data
        """
        return self.nuisance_scores_[4]


class LinearDRIV(StatsModelsCateEstimatorMixin, DRIV):
    """
    Special case of the :class:`.DRIV` where the final stage
    is a Linear Regression. In this case, inference can be performed via the StatsModels Inference approach
    and its asymptotic normal characterization of the estimated parameters. This is computationally
    faster than bootstrap inference. Leave the default ``inference='auto'`` unchanged, or explicitly set
    ``inference='statsmodels'`` at fit time to enable inference via asymptotic normality.

    Parameters
    ----------
    model_y_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Y | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_z_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument.

    model_t_xwz : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W, Z]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_tz_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T*Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument and discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument or continuous treatment.

    flexible_model_effect : estimator or 'auto' (default is 'auto')
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.
        If 'auto', :class:`.StatsModelsLinearRegression` will be applied.

    prel_cate_approach : one of {'driv', 'dmliv'}, optional (default='driv')
        model that estimates a preliminary version of the CATE.
        If 'driv', :class:`._DRIV` will be used.
        If 'dmliv', :class:`.NonParamDMLIV` will be used

    prel_cv : int, cross-validation generator or an iterable, optional, default 1
        Determines the cross-validation splitting strategy for the preliminary effect model.

    prel_opt_reweighted : bool, optional, default True
        Whether to reweight the samples to minimize variance for the preliminary effect model.

    projection: bool, optional, default False
        If True, we fit a slight variant of DRIV where we use E[T|X, W, Z] as the instrument as opposed to Z,
        model_z_xw will be disabled; If False, model_t_xwz will be disabled.

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
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_z_xw="auto",
                 model_t_xwz="auto",
                 model_tz_xw="auto",
                 flexible_model_effect="auto",
                 prel_cate_approach="driv",
                 prel_cv=1,
                 prel_opt_reweighted=True,
                 projection=False,
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=1e-3,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        super().__init__(model_y_xw=model_y_xw,
                         model_t_xw=model_t_xw,
                         model_z_xw=model_z_xw,
                         model_t_xwz=model_t_xwz,
                         model_tz_xw=model_tz_xw,
                         flexible_model_effect=flexible_model_effect,
                         model_final=None,
                         prel_cate_approach=prel_cate_approach,
                         prel_cv=prel_cv,
                         prel_opt_reweighted=prel_opt_reweighted,
                         projection=projection,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         opt_reweighted=opt_reweighted,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=False)

    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight : (n,) array like, default None
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array like of integers, default None
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : (n,) nd array like, default None
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports ``'bootstrap'``
            (or an instance of :class:`.BootstrapInference`) and ``'statsmodels'``
            (or an instance of :class:`.StatsModelsInferenceDiscrete`).

        Returns
        -------
        self
        """
        # Replacing fit from _OrthoLearner, to reorder arguments and improve the docstring
        return super().fit(Y, T, X=X, W=W, Z=Z,
                           sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final_._fit_cate_intercept

    @property
    def bias_part_of_coef(self):
        return self.ortho_learner_model_final_._fit_cate_intercept

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class SparseLinearDRIV(DebiasedLassoCateEstimatorMixin, DRIV):
    """
    Special case of the :class:`.DRIV` where the final stage
    is a Debiased Lasso Regression. In this case, inference can be performed via the debiased lasso approach
    and its asymptotic normal characterization of the estimated parameters. This is computationally
    faster than bootstrap inference. Leave the default ``inference='auto'`` unchanged, or explicitly set
    ``inference='debiasedlasso'`` at fit time to enable inference via asymptotic normality.

    Parameters
    ----------
    model_y_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Y | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_z_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument.

    model_t_xwz : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W, Z]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_tz_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T*Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument and discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument or continuous treatment.

    flexible_model_effect : estimator or 'auto' (default is 'auto')
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.
        If 'auto', :class:`.StatsModelsLinearRegression` will be applied.

    prel_cate_approach : one of {'driv', 'dmliv'}, optional (default='driv')
        model that estimates a preliminary version of the CATE.
        If 'driv', :class:`._DRIV` will be used.
        If 'dmliv', :class:`.NonParamDMLIV` will be used

    prel_cv : int, cross-validation generator or an iterable, optional, default 1
        Determines the cross-validation splitting strategy for the preliminary effect model.

    prel_opt_reweighted : bool, optional, default True
        Whether to reweight the samples to minimize variance for the preliminary effect model.

    projection: bool, optional, default False
        If True, we fit a slight variant of DRIV where we use E[T|X, W, Z] as the instrument as opposed to Z,
        model_z_xw will be disabled; If False, model_t_xwz will be disabled.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    alpha: string | float, optional., default 'auto'.
        CATE L1 regularization applied through the debiased lasso in the final model.
        'auto' corresponds to a CV form of the :class:`DebiasedLasso`.

    n_alphas : int, optional, default 100
        How many alphas to try if alpha='auto'

    alpha_cov : string | float, optional, default 'auto'
        The regularization alpha that is used when constructing the pseudo inverse of
        the covariance matrix Theta used to for correcting the final state lasso coefficient
        in the debiased lasso. Each such regression corresponds to the regression of one feature
        on the remainder of the features.

    n_alphas_cov : int, optional, default 10
        How many alpha_cov to try if alpha_cov='auto'.

    max_iter : int, optional, default 1000
        The maximum number of iterations in the Debiased Lasso

    tol : float, optional, default 1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors.

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
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_z_xw="auto",
                 model_t_xwz="auto",
                 model_tz_xw="auto",
                 flexible_model_effect="auto",
                 prel_cate_approach="driv",
                 prel_cv=1,
                 prel_opt_reweighted=True,
                 projection=False,
                 featurizer=None,
                 fit_cate_intercept=True,
                 alpha='auto',
                 n_alphas=100,
                 alpha_cov='auto',
                 n_alphas_cov=10,
                 max_iter=1000,
                 tol=1e-4,
                 n_jobs=None,
                 cov_clip=1e-3,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.alpha = alpha
        self.n_alphas = n_alphas
        self.alpha_cov = alpha_cov
        self.n_alphas_cov = n_alphas_cov
        self.max_iter = max_iter
        self.tol = tol
        self.n_jobs = n_jobs
        super().__init__(model_y_xw=model_y_xw,
                         model_t_xw=model_t_xw,
                         model_z_xw=model_z_xw,
                         model_t_xwz=model_t_xwz,
                         model_tz_xw=model_tz_xw,
                         flexible_model_effect=flexible_model_effect,
                         model_final=None,
                         prel_cate_approach=prel_cate_approach,
                         prel_cv=prel_cv,
                         prel_opt_reweighted=prel_opt_reweighted,
                         projection=projection,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         opt_reweighted=opt_reweighted,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return DebiasedLasso(alpha=self.alpha,
                             n_alphas=self.n_alphas,
                             alpha_cov=self.alpha_cov,
                             n_alphas_cov=self.n_alphas_cov,
                             fit_intercept=False,
                             max_iter=self.max_iter,
                             tol=self.tol,
                             n_jobs=self.n_jobs,
                             random_state=self.random_state)

    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight : (n,) array like, default None
            Individual weights for each sample. If None, it assumes equal weight.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports ``'bootstrap'``
            (or an instance of :class:`.BootstrapInference`) and ``'debiasedlasso'``
            (or an instance of :class:`.LinearModelInferenceDiscrete`).

        Returns
        -------
        self
        """
        # TODO: support freq_weight and sample_var in debiased lasso
        # Replacing fit from _OrthoLearner, to reorder arguments and improve the docstring
        check_high_dimensional(X, T, threshold=5, featurizer=self.featurizer,
                               discrete_treatment=self.discrete_treatment,
                               msg="The number of features in the final model (< 5) is too small for a sparse model. "
                               "We recommend using the LinearDRLearner for this low-dimensional setting.")
        return super().fit(Y, T, X=X, W=W, Z=Z,
                           sample_weight=sample_weight, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final_._fit_cate_intercept

    @property
    def bias_part_of_coef(self):
        return self.ortho_learner_model_final_._fit_cate_intercept

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class ForestDRIV(ForestModelFinalCateEstimatorMixin, DRIV):
    """ Instance of DRIV with a :class:`~econml.grf.RegressionForest`
    as a final model, so as to enable non-parametric inference.

    Parameters
    ----------
    model_y_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Y | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_z_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument.

    model_t_xwz : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W, Z]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous treatment.

    model_tz_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T*Z | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete instrument and discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
        will be applied for continuous instrument or continuous treatment.

    flexible_model_effect : estimator or 'auto' (default is 'auto')
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.
        If 'auto', :class:`.StatsModelsLinearRegression` will be applied.

    prel_cate_approach : one of {'driv', 'dmliv'}, optional (default='driv')
        model that estimates a preliminary version of the CATE.
        If 'driv', :class:`._DRIV` will be used.
        If 'dmliv', :class:`.NonParamDMLIV` will be used

    prel_cv : int, cross-validation generator or an iterable, optional, default 1
        Determines the cross-validation splitting strategy for the preliminary effect model.

    prel_opt_reweighted : bool, optional, default True
        Whether to reweight the samples to minimize variance for the preliminary effect model.

    projection: bool, optional, default False
        If True, we fit a slight variant of DRIV where we use E[T|X, W, Z] as the instrument as opposed to Z,
        model_z_xw will be disabled; If False, model_t_xwz will be disabled.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    n_estimators : integer, optional (default=100)
        The total number of trees in the forest. The forest consists of a
        forest of sqrt(n_estimators) sub-forests, where each sub-forest
        contains sqrt(n_estimators) trees.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of splitting samples required to split an internal node.

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` splitting samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression. After construction the tree is also pruned
        so that there are at least min_samples_leaf estimation samples on
        each leaf.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        splitting samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided. After construction
        the tree is pruned so that the fraction of the sum total weight
        of the estimation samples contained in each leaf node is at
        least min_weight_fraction_leaf

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of split samples, ``N_t`` is the number of
        split samples at the current node, ``N_t_L`` is the number of split samples in the
        left child, and ``N_t_R`` is the number of split samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    max_samples : int or float in (0, .5], default=.45,
        The number of samples to use for each subsample that is used to train each tree:

        - If int, then train each tree on `max_samples` samples, sampled without replacement from all the samples
        - If float, then train each tree on ceil(`max_samples` * `n_samples`), sampled without replacement
          from all the samples.

    min_balancedness_tol: float in [0, .5], default=.45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : boolean, optional (default=True)
        Whether to use honest trees, i.e. half of the samples are used for
        creating the tree structure and the other half for the estimation at
        the leafs. If False, then all samples are used for both parts.

    subforest_size : int, default=4,
        The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation.
        The parameter `n_estimators` must be divisible by `subforest_size`. Should typically be a small constant.

    n_jobs : int or None, optional (default=-1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

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
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_z_xw="auto",
                 model_t_xwz="auto",
                 model_tz_xw="auto",
                 flexible_model_effect="auto",
                 prel_cate_approach="driv",
                 prel_cv=1,
                 prel_opt_reweighted=True,
                 projection=False,
                 featurizer=None,
                 n_estimators=1000,
                 max_depth=None,
                 min_samples_split=5,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 subforest_size=4,
                 n_jobs=-1,
                 verbose=0,
                 cov_clip=1e-3,
                 opt_reweighted=False,
                 discrete_instrument=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.subforest_size = subforest_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__(model_y_xw=model_y_xw,
                         model_t_xw=model_t_xw,
                         model_z_xw=model_z_xw,
                         model_t_xwz=model_t_xwz,
                         model_tz_xw=model_tz_xw,
                         flexible_model_effect=flexible_model_effect,
                         model_final=None,
                         prel_cate_approach=prel_cate_approach,
                         prel_cv=prel_cv,
                         prel_opt_reweighted=prel_opt_reweighted,
                         projection=projection,
                         featurizer=featurizer,
                         fit_cate_intercept=False,
                         cov_clip=cov_clip,
                         opt_reweighted=opt_reweighted,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        return RegressionForest(n_estimators=self.n_estimators,
                                max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf,
                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                max_features=self.max_features,
                                min_impurity_decrease=self.min_impurity_decrease,
                                max_samples=self.max_samples,
                                min_balancedness_tol=self.min_balancedness_tol,
                                honest=self.honest,
                                inference=True,
                                subforest_size=self.subforest_size,
                                n_jobs=self.n_jobs,
                                random_state=self.random_state,
                                verbose=self.verbose,
                                warm_start=False)

    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        Z: (n, d_z) matrix
            Instruments for each sample
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample
        sample_weight : (n,) array like, default None
            Individual weights for each sample. If None, it assumes equal weight.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'blb'
            (for Bootstrap-of-Little-Bags based inference)

        Returns
        -------
        self
        """
        if X is None:
            raise ValueError("This estimator does not support X=None!")

        # Replacing fit from _OrthoLearner, to reorder arguments and improve the docstring
        return super().fit(Y, T, X=X, W=W, Z=Z,
                           sample_weight=sample_weight, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")


class _IntentToTreatDRIVModelNuisance:
    def __init__(self, model_y_xw, model_t_xwz, dummy_z, prel_model_effect):
        self._model_y_xw = clone(model_y_xw, safe=False)
        self._model_t_xwz = clone(model_t_xwz, safe=False)
        self._dummy_z = clone(dummy_z, safe=False)
        self._prel_model_effect = clone(prel_model_effect, safe=False)

    def _combine(self, W, Z, n_samples):
        if Z is not None:  # Z will not be None
            Z = Z.reshape(n_samples, -1)
            return Z if W is None else np.hstack([W, Z])
        return None if W is None else W

    def fit(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        self._model_y_xw.fit(X=X, W=W, Target=Y, sample_weight=sample_weight, groups=groups)
        # concat W and Z
        WZ = self._combine(W, Z, Y.shape[0])
        self._model_t_xwz.fit(X=X, W=WZ, Target=T, sample_weight=sample_weight, groups=groups)
        self._dummy_z.fit(X=X, W=W, Target=Z, sample_weight=sample_weight, groups=groups)
        # we need to undo the one-hot encoding for calling effect,
        # since it expects raw values
        self._prel_model_effect.fit(Y, inverse_onehot(T), Z=inverse_onehot(Z), X=X, W=W,
                                    sample_weight=sample_weight, groups=groups)
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        if hasattr(self._model_y_xw, 'score'):
            Y_X_score = self._model_y_xw.score(X=X, W=W, Target=Y, sample_weight=sample_weight)
        else:
            Y_X_score = None
        if hasattr(self._model_t_xwz, 'score'):
            # concat W and Z
            WZ = self._combine(W, Z, Y.shape[0])
            T_XZ_score = self._model_t_xwz.score(X=X, W=WZ, Target=T, sample_weight=sample_weight)
        else:
            T_XZ_score = None
        if hasattr(self._prel_model_effect, 'score'):
            # we need to undo the one-hot encoding for calling effect,
            # since it expects raw values
            effect_score = self._prel_model_effect.score(Y, inverse_onehot(T),
                                                         inverse_onehot(Z), X=X, W=W, sample_weight=sample_weight)
        else:
            effect_score = None

        return Y_X_score, T_XZ_score, effect_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        Y_pred = self._model_y_xw.predict(X, W)
        T_pred_zero = self._model_t_xwz.predict(X, self._combine(W, np.zeros(Z.shape), Y.shape[0]))
        T_pred_one = self._model_t_xwz.predict(X, self._combine(W, np.ones(Z.shape), Y.shape[0]))
        Z_pred = self._dummy_z.predict(X, W)
        prel_theta = self._prel_model_effect.effect(X)

        if X is None:  # In this case predict above returns a single row
            prel_theta = np.tile(prel_theta.reshape(1, -1), (T.shape[0], 1))
            if W is None:
                Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
                Z_pred = np.tile(Z_pred.reshape(1, -1), (Z.shape[0], 1))

        # reshape the predictions
        Y_pred = Y_pred.reshape(Y.shape)
        Z_pred = Z_pred.reshape(Z.shape)
        T_pred_one = T_pred_one.reshape(T.shape)
        T_pred_zero = T_pred_zero.reshape(T.shape)

        # T_res, Z_res, beta expect shape to be (n,1)
        beta = Z_pred * (1 - Z_pred) * (T_pred_one - T_pred_zero)
        T_pred = T_pred_one * Z_pred + T_pred_zero * (1 - Z_pred)
        Y_res = Y - Y_pred
        T_res = T - T_pred
        Z_res = Z - Z_pred

        return prel_theta, Y_res, T_res, Z_res, beta


class _DummyClassifier:
    """
    A dummy classifier that always returns the prior ratio

    Parameters
    ----------
    ratio: float
        The ratio of treatment samples
    """

    def __init__(self, *, ratio):
        self.ratio = ratio

    def fit(self, X, y, **kwargs):
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        return np.hstack((np.tile(1 - self.ratio, (n, 1)), np.tile(self.ratio, (n, 1))))


class _IntentToTreatDRIV(_BaseDRIV):
    """
    Base class for the DRIV algorithm for the intent-to-treat A/B test setting
    """

    def __init__(self, *,
                 model_y_xw="auto",
                 model_t_xwz="auto",
                 prel_model_effect,
                 model_final,
                 z_propensity="auto",
                 featurizer=None,
                 fit_cate_intercept=False,
                 cov_clip=1e-3,
                 opt_reweighted=False,
                 categories='auto',
                 cv=3,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.model_y_xw = clone(model_y_xw, safe=False)
        self.model_t_xwz = clone(model_t_xwz, safe=False)
        self.prel_model_effect = clone(prel_model_effect, safe=False)
        self.z_propensity = z_propensity

        super().__init__(model_final=model_final,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         cv=cv,
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
        if self.model_y_xw == 'auto':
            model_y_xw = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_y_xw = clone(self.model_y_xw, safe=False)

        if self.model_t_xwz == 'auto':
            model_t_xwz = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                               random_state=self.random_state)
        else:
            model_t_xwz = clone(self.model_t_xwz, safe=False)

        if self.z_propensity == "auto":
            dummy_z = DummyClassifier(strategy="prior")
        elif isinstance(self.z_propensity, float):
            dummy_z = _DummyClassifier(ratio=self.z_propensity)
        else:
            raise ValueError("Only 'auto' or float is allowed!")

        return _IntentToTreatDRIVModelNuisance(_FirstStageWrapper(model_y_xw, True, self._gen_featurizer(),
                                                                  False, False),
                                               _FirstStageWrapper(model_t_xwz, False,
                                                                  self._gen_featurizer(), False, True),
                                               _FirstStageWrapper(dummy_z, False,
                                                                  self._gen_featurizer(), False, True),
                                               self._gen_prel_model_effect()
                                               )


class _DummyCATE:
    """
    A dummy cate effect model that always returns zero effect
    """

    def __init__(self):
        return

    def fit(self, y, T, *, Z, X=None, W=None, sample_weight=None, groups=None, **kwargs):
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
    model_y_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Y | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t_xwz : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W, Z]`.  Must support `fit` and `predict_proba` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment.

    flexible_model_effect : estimator or 'auto' (default is 'auto')
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.
        If 'auto', :class:`.StatsModelsLinearRegression` will be applied.

    model_final : estimator, optional
        a final model for the CATE and projections. If None, then flexible_model_effect is also used as a final model

    prel_cate_approach : one of {'driv', 'dmliv'}, optional (default='driv')
        model that estimates a preliminary version of the CATE.
        If 'driv', :class:`._DRIV` will be used.
        If 'dmliv', :class:`.NonParamDMLIV` will be used

    prel_cv : int, cross-validation generator or an iterable, optional, default 1
        Determines the cross-validation splitting strategy for the preliminary effect model.

    prel_opt_reweighted : bool, optional, default True
        Whether to reweight the samples to minimize variance for the preliminary effect model.

    z_propensity: float or "auto", optional, default "auto"
        The ratio of the A/B test in treatment group. If "auto", we assume that the instrument is fully randomized
        and independent of any other variables. It's calculated as the proportion of Z=1 in the overall population;
        If input a ratio, it has to be a float between 0 to 1.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default False
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
                 model_y_xw="auto",
                 model_t_xwz="auto",
                 prel_cate_approach="driv",
                 flexible_model_effect="auto",
                 model_final=None,
                 prel_cv=1,
                 prel_opt_reweighted=True,
                 z_propensity="auto",
                 featurizer=None,
                 fit_cate_intercept=False,
                 cov_clip=1e-3,
                 cv=3,
                 mc_iters=None,
                 mc_agg='mean',
                 opt_reweighted=False,
                 categories='auto',
                 random_state=None):
        # maybe shouldn't expose fit_cate_intercept in this class?
        if flexible_model_effect == "auto":
            self.flexible_model_effect = StatsModelsLinearRegression(fit_intercept=False)
        else:
            self.flexible_model_effect = clone(flexible_model_effect, safe=False)
        self.prel_cate_approach = prel_cate_approach
        self.prel_cv = prel_cv
        self.prel_opt_reweighted = prel_opt_reweighted
        super().__init__(model_y_xw=model_y_xw,
                         model_t_xwz=model_t_xwz,
                         prel_model_effect=self.prel_cate_approach,
                         model_final=model_final,
                         z_propensity=z_propensity,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         opt_reweighted=opt_reweighted,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_model_final(self):
        if self.model_final is None:
            return clone(self.flexible_model_effect, safe=False)
        return clone(self.model_final, safe=False)

    def _gen_prel_model_effect(self):
        if self.prel_cate_approach == "driv":
            return _IntentToTreatDRIV(model_y_xw=clone(self.model_y_xw, safe=False),
                                      model_t_xwz=clone(self.model_t_xwz, safe=False),
                                      prel_model_effect=_DummyCATE(),
                                      model_final=clone(self.flexible_model_effect, safe=False),
                                      featurizer=self._gen_featurizer(),
                                      fit_cate_intercept=self.fit_cate_intercept,
                                      cov_clip=self.cov_clip,
                                      categories=self.categories,
                                      opt_reweighted=self.prel_opt_reweighted,
                                      cv=self.prel_cv,
                                      random_state=self.random_state)
        elif self.prel_cate_approach == "dmliv":
            return NonParamDMLIV(model_y_xw=clone(self.model_y_xw, safe=False),
                                 model_t_xw=clone(self.model_t_xwz, safe=False),
                                 model_t_xwz=clone(self.model_t_xwz, safe=False),
                                 model_final=clone(self.flexible_model_effect, safe=False),
                                 discrete_instrument=True,
                                 discrete_treatment=True,
                                 featurizer=self._gen_featurizer(),
                                 categories=self.categories,
                                 cv=self.prel_cv,
                                 mc_iters=self.mc_iters,
                                 mc_agg=self.mc_agg,
                                 random_state=self.random_state)
        else:
            raise ValueError(
                "We only support 'dmliv' or 'driv' preliminary model effect, "
                f"but received '{self.prel_cate_approach}'!")

    @property
    def models_y_xw(self):
        """
        Get the fitted models for :math:`\\E[Y | X]`.

        Returns
        -------
        models_y_xw: nested list of objects of type(`model_y_xw`)
            A nested list of instances of the `model_y_xw` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._model_y_xw._model for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_t_xwz(self):
        """
        Get the fitted models for :math:`\\E[T | X, Z]`.

        Returns
        -------
        models_t_xwz: nested list of objects of type(`model_t_xwz`)
            A nested list of instances of the `model_t_xwz` object. Number of sublist equals to number of monte carlo
            iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._model_t_xwz._model for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_prel_model_effect(self):
        """
        Get the fitted preliminary CATE estimator.

        Returns
        -------
        prel_model_effect: nested list of objects of type(`prel_model_effect`)
            A nested list of instances of the `prel_model_effect` object. Number of sublist equals to number
            of monte carlo iterations, each element in the sublist corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [[mdl._prel_model_effect for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def nuisance_scores_y_xw(self):
        """
        Get the scores for y_xw model on the out-of-sample training data
        """
        return self.nuisance_scores_[0]

    @property
    def nuisance_scores_t_xwz(self):
        """
        Get the scores for t_xw model on the out-of-sample training data
        """
        return self.nuisance_scores_[1]

    @property
    def nuisance_scores_prel_model_effect(self):
        """
        Get the scores for prel_model_effect model on the out-of-sample training data
        """
        return self.nuisance_scores_[2]


class LinearIntentToTreatDRIV(StatsModelsCateEstimatorMixin, IntentToTreatDRIV):
    """
    Implements the DRIV algorithm for the Linear Intent-to-Treat A/B test setting

    Parameters
    ----------
    model_y_xw : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[Y | X, W]`.  Must support `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.


    model_t_xwz : estimator or 'auto' (default is 'auto')
        model to estimate :math:`\\E[T | X, W, Z]`.  Must support `fit` and `predict_proba` methods.
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV`
        will be applied for discrete treatment.

    flexible_model_effect : estimator or 'auto' (default is 'auto')
        a flexible model for a preliminary version of the CATE, must accept sample_weight at fit time.
        If 'auto', :class:`.StatsModelsLinearRegression` will be applied.

    prel_cate_approach : one of {'driv', 'dmliv'}, optional (default='driv')
        model that estimates a preliminary version of the CATE.
        If 'driv', :class:`._DRIV` will be used.
        If 'dmliv', :class:`.NonParamDMLIV` will be used

    prel_cv : int, cross-validation generator or an iterable, optional, default 1
        Determines the cross-validation splitting strategy for the preliminary effect model.

    prel_opt_reweighted : bool, optional, default True
        Whether to reweight the samples to minimize variance for the preliminary effect model.

    z_propensity: float or "auto", optional, default "auto"
        The ratio of the A/B test in treatment group. If "auto", we assume that the instrument is fully randomized
        and independent of any other variables. It's calculated as the proportion of Z=1 in the overall population;
        If input a ratio, it has to be a float between 0 to 1.

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
                 model_y_xw="auto",
                 model_t_xwz="auto",
                 prel_cate_approach="driv",
                 flexible_model_effect="auto",
                 prel_cv=1,
                 prel_opt_reweighted=True,
                 z_propensity="auto",
                 featurizer=None,
                 fit_cate_intercept=True,
                 cov_clip=1e-3,
                 cv=3,
                 mc_iters=None,
                 mc_agg='mean',
                 opt_reweighted=False,
                 categories='auto',
                 random_state=None):
        super().__init__(model_y_xw=model_y_xw,
                         model_t_xwz=model_t_xwz,
                         flexible_model_effect=flexible_model_effect,
                         model_final=None,
                         prel_cate_approach=prel_cate_approach,
                         prel_cv=prel_cv,
                         prel_opt_reweighted=prel_opt_reweighted,
                         z_propensity=z_propensity,
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
                         cov_clip=cov_clip,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         opt_reweighted=opt_reweighted,
                         categories=categories,
                         random_state=random_state)

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=False)

    # override only so that we can update the docstring to indicate support for `StatsModelsInference`
    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
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
        sample_weight : (n,) array like or None
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array like of integers, default None
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : (n,) nd array like, default None
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
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
        # TODO: do correct adjustment for sample_var
        return super().fit(Y, T, Z=Z, X=X, W=W,
                           sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final_._fit_cate_intercept

    @property
    def bias_part_of_coef(self):
        return self.ortho_learner_model_final_._fit_cate_intercept

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, value):
        if value is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator.")
