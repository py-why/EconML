# Copyright (c) PyWhy contributors. All rights reserved.
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
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from itertools import product

from ..._ortho_learner import _OrthoLearner
from ..._cate_estimator import LinearModelFinalCateEstimatorMixin, StatsModelsCateEstimatorMixin, LinearCateEstimator
from ...inference import StatsModelsInference, GenericSingleTreatmentModelFinalInference
from ...sklearn_extensions.linear_model import StatsModels2SLS, StatsModelsLinearRegression, WeightedLassoCVWrapper
from ...sklearn_extensions.model_selection import (ModelSelector, SingleModelSelector,
                                                   WeightedStratifiedKFold, get_selector)
from ...utilities import (_deprecate_positional, get_feature_names_or_default, filter_none_kwargs, add_intercept,
                          cross_product, broadcast_unit_treatments, reshape_treatmentwise_effects, shape,
                          parse_final_model_params, deprecated, Summary)
from ...dml.dml import _make_first_stage_selector, _FinalWrapper
from ...dml._rlearner import _ModelFinal
from ..._shap import _shap_explain_joint_linear_model_cate, _shap_explain_model_cate


def _combine(W, Z, n_samples):
    if Z is not None:
        Z = Z.reshape(n_samples, -1)
        return Z if W is None else np.hstack([W, Z])
    return None if W is None else W


class _OrthoIVNuisanceSelector(ModelSelector):

    def __init__(self,
                 model_y_xw: SingleModelSelector,
                 model_t_xw: SingleModelSelector,
                 model_z: SingleModelSelector,
                 projection):
        self._model_y_xw = model_y_xw
        self._model_t_xw = model_t_xw
        self._projection = projection
        if self._projection:
            self._model_t_xwz = model_z
        else:
            self._model_z_xw = model_z

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        self._model_y_xw.train(is_selecting, folds, X=X, W=W, Target=Y, sample_weight=sample_weight, groups=groups)
        self._model_t_xw.train(is_selecting, folds, X=X, W=W, Target=T, sample_weight=sample_weight, groups=groups)
        if self._projection:
            # concat W and Z
            WZ = _combine(W, Z, Y.shape[0])
            self._model_t_xwz.train(is_selecting, folds, X=X, W=WZ, Target=T,
                                    sample_weight=sample_weight, groups=groups)
        else:
            self._model_z_xw.train(is_selecting, folds, X=X, W=W, Target=Z, sample_weight=sample_weight, groups=groups)
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        if hasattr(self._model_y_xw, 'score'):
            Y_X_score = self._model_y_xw.score(X=X, W=W, Target=Y, sample_weight=sample_weight)
        else:
            Y_X_score = None
        if hasattr(self._model_t_xw, 'score'):
            T_X_score = self._model_t_xw.score(X=X, W=W, Target=T, sample_weight=sample_weight)
        else:
            T_X_score = None
        if self._projection:
            # concat W and Z
            WZ = _combine(W, Z, Y.shape[0])
            if hasattr(self._model_t_xwz, 'score'):
                T_XZ_score = self._model_t_xwz.score(X=X, W=WZ, Target=T, sample_weight=sample_weight)
            else:
                T_XZ_score = None
            return Y_X_score, T_X_score, T_XZ_score

        else:
            if hasattr(self._model_z_xw, 'score'):
                Z_X_score = self._model_z_xw.score(X=X, W=W, Target=Z, sample_weight=sample_weight)
            else:
                Z_X_score = None
            return Y_X_score, T_X_score, Z_X_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        Y_pred = self._model_y_xw.predict(X=X, W=W)
        T_pred = self._model_t_xw.predict(X=X, W=W)

        if self._projection:
            # concat W and Z
            WZ = _combine(W, Z, Y.shape[0])
            T_proj = self._model_t_xwz.predict(X, WZ)
        else:
            Z_pred = self._model_z_xw.predict(X=X, W=W)

        if (X is None) and (W is None):  # In this case predict above returns a single row
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            T_pred = np.tile(T_pred.reshape(1, -1), (T.shape[0], 1))
            if not self._projection:
                Z_pred = np.tile(Z_pred.reshape(1, -1), (Z.shape[0], 1))

        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = T - T_pred.reshape(T.shape)

        if self._projection:
            Z_res = T_proj.reshape(T.shape) - T_pred.reshape(T.shape)
        else:
            Z_res = Z - Z_pred.reshape(Z.shape)
        return Y_res, T_res, Z_res


class _OrthoIVModelFinal:
    def __init__(self, model_final, featurizer, fit_cate_intercept):
        self._model_final = clone(model_final, safe=False)
        self._original_featurizer = clone(featurizer, safe=False)
        self._fit_cate_intercept = fit_cate_intercept

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

    def _combine(self, X, T, fitting=True):
        if X is not None:
            if self._featurizer is not None:
                F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
            else:
                F = X
        else:
            if not self._fit_cate_intercept:
                raise AttributeError("Cannot have X=None and also not allow for a CATE intercept!")
            F = np.ones((T.shape[0], 1))
        return cross_product(F, T)

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None,
            sample_weight=None, freq_weight=None, sample_var=None, groups=None):
        Y_res, T_res, Z_res = nuisances

        # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T_res)[1:]
        self._d_y = shape(Y_res)[1:]

        XT_res = self._combine(X, T_res)
        XZ_res = self._combine(X, Z_res)
        filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight,
                                             freq_weight=freq_weight, sample_var=sample_var)

        self._model_final.fit(XZ_res, XT_res, Y_res, **filtered_kwargs)

        return self

    def predict(self, X=None):
        X2, T = broadcast_unit_treatments(X if X is not None else np.empty((1, 0)),
                                          self._d_t[0] if self._d_t else 1)
        XT = self._combine(None if X is None else X2, T, fitting=False)
        prediction = self._model_final.predict(XT)
        return reshape_treatmentwise_effects(prediction,
                                             self._d_t, self._d_y)

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, groups=None):
        Y_res, T_res, Z_res = nuisances
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        effects = self.predict(X).reshape((-1, Y_res.shape[1], T_res.shape[1]))
        Y_res_pred = np.einsum('ijk,ik->ij', effects, T_res).reshape(Y_res.shape)
        if sample_weight is not None:
            return np.linalg.norm(np.average(cross_product(Z_res, Y_res - Y_res_pred), weights=sample_weight, axis=0),
                                  ord=2)
        else:
            return np.linalg.norm(np.mean(cross_product(Z_res, Y_res - Y_res_pred), axis=0), ord=2)


class OrthoIV(LinearModelFinalCateEstimatorMixin, _OrthoLearner):
    """
    Implementation of the orthogonal/double ml method for CATE estimation with
    IV as described in section 4.2:

    Double/Debiased Machine Learning for Treatment and Causal Parameters
    Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins
    https://arxiv.org/abs/1608.00060

    Solve the following moment equation:

    .. math::
        \\E[(Y-\\E[Y|X]-\\theta(X) * (T-\\E[T|X]))(Z-\\E[Z|X])] = 0

    Parameters
    ----------
    model_y_xw: estimator, default ``'auto'``
        Determines how to fit the outcome to the features and controls (:math:`\\E[Y | X, W]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t_xw: estimator, default ``'auto'``
        Determines how to fit the treatment to the features and controls (:math:`\\E[T | X, W]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_t_xwz: estimator, default ``'auto'``
        Determines how to fit the treatment to the features, controls, and instrument (:math:`\\E[T | X, W, Z]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_z_xw: estimator, default ``'auto'``
        Determines how to fit the instrument to the features and controls (:math:`\\E[Z | X, W]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_instrument` is True
          and a regressor otherwise

    projection: bool, default False
        If True, we fit a slight variant of OrthoIV where we use E[T|X, W, Z] as the instrument as opposed to Z,
        model_z_xw will be disabled; If False, model_t_xwz will be disabled.

    featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, default False
        Whether the linear CATE model should have a constant term.

    discrete_outcome: bool, default False
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    discrete_instrument: bool, default False
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, default 2
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

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply nuisance models
        that can handle missing values.

    Examples
    --------
    A simple example with the default models:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.iv.dml import OrthoIV

        # Define the data generation functions
        def dgp(n, p, true_fn):
            X = np.random.normal(0, 1, size=(n, p))
            Z = np.random.binomial(1, 0.5, size=(n,))
            nu = np.random.uniform(0, 10, size=(n,))
            coef_Z = 0.8
            C = np.random.binomial(
                1, coef_Z * scipy.special.expit(0.4 * X[:, 0] + nu)
            )  # Compliers when recomended
            C0 = np.random.binomial(
                1, 0.06 * np.ones(X.shape[0])
            )  # Non-compliers when not recommended
            T = C * Z + C0 * (1 - Z)
            y = true_fn(X) * T + 2 * nu + 5 * (X[:, 3] > 0) + 0.1 * np.random.uniform(0, 1, size=(n,))
            return y, T, Z, X

        def true_heterogeneity_function(X):
            return 5 * X[:, 0]

        np.random.seed(123)
        y, T, Z, X = dgp(1000, 5, true_heterogeneity_function)
        est = OrthoIV(discrete_treatment=True, discrete_instrument=True)
        est.fit(Y=y, T=T, Z=Z, X=X)

    >>> est.effect(X[:3])
    array([-4.28045... ,  6.02945..., -2.86851...])
    >>> est.effect_interval(X[:3])
    (array([-7.20729...,  1.75412..., -5.20897...]),
    array([-1.35361..., 10.30478..., -0.52805...]))
    >>> est.coef_
    array([ 4.51659...,  0.78512...,  0.23706...,  0.24126... , -0.47167...])
    >>> est.coef__interval()
    (array([ 3.15602..., -0.35785..., -0.89798..., -0.90530..., -1.62445...]),
    array([5.87715... , 1.92810... , 1.37211..., 1.38783..., 0.68110...]))
    >>> est.intercept_
    -0.13672...
    >>> est.intercept__interval()
    (-1.27036..., 0.99690...)
    """

    def __init__(self, *,
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_t_xwz="auto",
                 model_z_xw="auto",
                 projection=False,
                 featurizer=None,
                 fit_cate_intercept=True,
                 discrete_outcome=False,
                 discrete_treatment=False,
                 treatment_featurizer=None,
                 discrete_instrument=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False):
        self.model_y_xw = clone(model_y_xw, safe=False)
        self.model_t_xw = clone(model_t_xw, safe=False)
        self.model_t_xwz = clone(model_t_xwz, safe=False)
        self.model_z_xw = clone(model_z_xw, safe=False)
        self.projection = projection
        self.featurizer = clone(featurizer, safe=False)
        self.fit_cate_intercept = fit_cate_intercept

        super().__init__(discrete_outcome=discrete_outcome,
                         discrete_instrument=discrete_instrument,
                         discrete_treatment=discrete_treatment,
                         treatment_featurizer=treatment_featurizer,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing)

    def _gen_allowed_missing_vars(self):
        return ['W'] if self.allow_missing else []

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_final(self):
        return StatsModels2SLS(cov_type="HC0")

    def _gen_ortho_learner_model_final(self):
        return _OrthoIVModelFinal(self._gen_model_final(), self._gen_featurizer(), self.fit_cate_intercept)

    def _gen_ortho_learner_model_nuisance(self):
        model_y = _make_first_stage_selector(self.model_y_xw,
                                             is_discrete=self.discrete_outcome,
                                             random_state=self.random_state)

        model_t = _make_first_stage_selector(self.model_t_xw,
                                             is_discrete=self.discrete_treatment,
                                             random_state=self.random_state)

        if self.projection:
            # train E[T|X,W,Z]
            model_z = _make_first_stage_selector(self.model_t_xwz,
                                                 is_discrete=self.discrete_treatment,
                                                 random_state=self.random_state)

        else:
            # train E[Z|X,W]
            # note: discrete_instrument rather than discrete_treatment in call to _make_first_stage_selector
            model_z = _make_first_stage_selector(self.model_z_xw,
                                                 is_discrete=self.discrete_instrument,
                                                 random_state=self.random_state)

        return _OrthoIVNuisanceSelector(model_y, model_t, model_z,
                                        self.projection)

    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
            cache_values=False, inference="auto"):
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
        X:(n, d_x) matrix, optional
            Features for each sample
        W:(n, d_w) matrix, optional
            Controls for each sample
        sample_weight : (n,) array_like, optional
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array_like of int, optional
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(n,), (n, d_y)} nd array_like, optional
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of:class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.LinearModelFinalInference`)

        Returns
        -------
        self: OrthoIV instance
        """
        if self.projection:
            assert self.model_z_xw == "auto", ("In the case of projection=True, model_z_xw will not be fitted, "
                                               "please leave it when initializing the estimator!")
        else:
            assert self.model_t_xwz == "auto", ("In the case of projection=False, model_t_xwz will not be fitted, "
                                                "please leave it when initializing the estimator!")
        #  Replacing fit from _OrthoLearner, to reorder arguments and improve the docstring
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
        Z: (n, d_z) matrix, optional
            Instruments for each sample
        X:(n, d_x) matrix, optional
            Features for each sample
        W:(n, d_w) matrix, optional
            Controls for each sample
        sample_weight:(n,) vector, optional
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
        feature_names: list of str of length X.shape[1] or None
            The names of the input features. If None and X is a dataframe, it defaults to the column names
            from the dataframe.

        Returns
        -------
        out_feature_names: list of str or None
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
        return [[mdl._model_y_xw.best_model._model for mdl in mdls] for mdls in super().models_nuisance_]

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
        return [[mdl._model_t_xw.best_model._model for mdl in mdls] for mdls in super().models_nuisance_]

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
        return [[mdl._model_z_xw.best_model._model for mdl in mdls] for mdls in super().models_nuisance_]

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
        if not self.projection:
            raise AttributeError("Direct model is fitted for instrument! Use models_z_xw.")
        return [[mdl._model_t_xwz.best_model._model for mdl in mdls] for mdls in super().models_nuisance_]

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
        Get the scores for t_xwz model on the out-of-sample training data
        """
        if not self.projection:
            raise AttributeError("Direct model is fitted for instrument! Use nuisance_scores_z_xw.")
        return self.nuisance_scores_[2]

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

    @property
    def residuals_(self):
        """
        A tuple (y_res, T_res,Z_res, X, W, Z), of the residuals from the first stage estimation
        along with the associated X, W and Z. Samples are not guaranteed to be in the same
        order as the input order.
        """
        if not hasattr(self, '_cached_values'):
            raise AttributeError("Estimator is not fitted yet!")
        if self._cached_values is None:
            raise AttributeError("`fit` was called with `cache_values=False`. "
                                 "Set to `True` to enable residual storage.")
        Y_res, T_res, Z_res = self._cached_values.nuisances
        return Y_res, T_res, Z_res, self._cached_values.X, self._cached_values.W, self._cached_values.Z


class _BaseDMLIVNuisanceSelector(ModelSelector):
    """
    Nuisance model fits the three models at fit time and at predict time
    returns :math:`Y-\\E[Y|X]` and :math:`\\E[T|X,Z]-\\E[T|X]` as residuals.
    """

    def __init__(self, model_y_xw: ModelSelector, model_t_xw: ModelSelector, model_t_xwz: ModelSelector):
        self._model_y_xw = model_y_xw
        self._model_t_xw = model_t_xw
        self._model_t_xwz = model_t_xwz

    def train(self, is_selecting, folds, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        self._model_y_xw.train(is_selecting, folds, X, W, Y, **
                               filter_none_kwargs(sample_weight=sample_weight, groups=groups))
        self._model_t_xw.train(is_selecting, folds, X, W, T, **
                               filter_none_kwargs(sample_weight=sample_weight, groups=groups))
        # concat W and Z
        WZ = _combine(W, Z, Y.shape[0])
        self._model_t_xwz.train(is_selecting, folds, X, WZ, T,
                                **filter_none_kwargs(sample_weight=sample_weight, groups=groups))
        return self

    def score(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        # note that groups are not passed to score because they are only used for fitting
        if hasattr(self._model_y_xw, 'score'):
            Y_X_score = self._model_y_xw.score(X, W, Y, **filter_none_kwargs(sample_weight=sample_weight))
        else:
            Y_X_score = None
        if hasattr(self._model_t_xw, 'score'):
            T_X_score = self._model_t_xw.score(X, W, T, **filter_none_kwargs(sample_weight=sample_weight))
        else:
            T_X_score = None
        if hasattr(self._model_t_xwz, 'score'):
            # concat W and Z
            WZ = _combine(W, Z, Y.shape[0])
            T_XZ_score = self._model_t_xwz.score(X, WZ, T, **filter_none_kwargs(sample_weight=sample_weight))
        else:
            T_XZ_score = None
        return Y_X_score, T_X_score, T_XZ_score

    def predict(self, Y, T, X=None, W=None, Z=None, sample_weight=None, groups=None):
        # note that sample_weight and groups are not passed to predict because they are only used for fitting
        Y_pred = self._model_y_xw.predict(X, W)
        # concat W and Z
        WZ = _combine(W, Z, Y.shape[0])
        TXZ_pred = self._model_t_xwz.predict(X, WZ)
        TX_pred = self._model_t_xw.predict(X, W)
        if (X is None) and (W is None):  # In this case predict above returns a single row
            Y_pred = np.tile(Y_pred.reshape(1, -1), (Y.shape[0], 1))
            TX_pred = np.tile(TX_pred.reshape(1, -1), (T.shape[0], 1))
        Y_res = Y - Y_pred.reshape(Y.shape)
        T_res = TXZ_pred.reshape(T.shape) - TX_pred.reshape(T.shape)
        return Y_res, T_res


class _BaseDMLIVModelFinal(_ModelFinal):
    """
    Final model at fit time, fits a residual on residual regression with a heterogeneous coefficient
    that depends on X, i.e.

        .. math ::
            Y - \\E[Y | X] = \\theta(X) \\cdot (\\E[T | X, Z] - \\E[T | X]) + \\epsilon

    and at predict time returns :math:`\\theta(X)`. The score method returns the MSE of this final
    residual on residual regression.
    """
    pass


class _BaseDMLIV(_OrthoLearner):
    # A helper class that access all the internal fitted objects of a DMLIV Cate Estimator.
    # Used by both Parametric and Non Parametric DMLIV.
    # override only so that we can enforce Z to be required
    def fit(self, Y, T, *, Z, X=None, W=None, sample_weight=None, freq_weight=None, sample_var=None, groups=None,
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
        X:(n, d_x) matrix, optional
            Features for each sample
        W: (n, d_w) matrix, optional
            Controls for each sample
        sample_weight : (n,) array_like, optional
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (n,) array_like of int, optional
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(n,), (n, d_y)} nd array_like, optional
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W, Z=Z,
                           sample_weight=sample_weight, freq_weight=freq_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values, inference=inference)

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
        X:(n, d_x) matrix, optional
            Features for each sample
        W:(n, d_w) matrix, optional
            Controls for each sample
        sample_weight:(n,) vector, optional
            Weights for each samples

        Returns
        -------
        score: float
            The MSE of the final CATE model on the new data.
        """
        # Replacing score from _OrthoLearner, to enforce Z to be required and improve the docstring
        return super().score(Y, T, X=X, W=W, Z=Z, sample_weight=sample_weight)

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
        return [[mdl._model_y_xw.best_model._model for mdl in mdls] for mdls in super().models_nuisance_]

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
        return [[mdl._model_t_xw.best_model._model for mdl in mdls] for mdls in super().models_nuisance_]

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
        return [[mdl._model_t_xwz.best_model._model for mdl in mdls] for mdls in super().models_nuisance_]

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
    def nuisance_scores_t_xwz(self):
        """
        Get the scores for t_xwz model on the out-of-sample training data
        """
        return self.nuisance_scores_[2]

    @property
    def residuals_(self):
        """
        A tuple (y_res, T_res, X, W, Z), of the residuals from the first stage estimation
        along with the associated X, W and Z. Samples are not guaranteed to be in the same
        order as the input order.
        """
        if not hasattr(self, '_cached_values'):
            raise AttributeError("Estimator is not fitted yet!")
        if self._cached_values is None:
            raise AttributeError("`fit` was called with `cache_values=False`. "
                                 "Set to `True` to enable residual storage.")
        Y_res, T_res = self._cached_values.nuisances
        return Y_res, T_res, self._cached_values.X, self._cached_values.W, self._cached_values.Z

    def cate_feature_names(self, feature_names=None):
        """
        Get the output feature names.

        Parameters
        ----------
        feature_names: list of str of length X.shape[1] or None
            The names of the input features. If None and X is a dataframe, it defaults to the column names
            from the dataframe.

        Returns
        -------
        out_feature_names: list of str or None
            The names of the output features :math:`\\phi(X)`, i.e. the features with respect to which the
            final constant marginal CATE model is linear. It is the names of the features that are associated
            with each entry of the :meth:`coef_` parameter. Not available when the featurizer is not None and
            does not have a method: `get_feature_names(feature_names)`. Otherwise None is returned.
        """
        if self._d_x is None:
            # Handles the corner case when X=None but featurizer might be not None
            return None
        if feature_names is None:
            feature_names = self._input_names["feature_names"]
        if self.original_featurizer is None:
            return feature_names
        return get_feature_names_or_default(self.original_featurizer, feature_names)


class DMLIV(_BaseDMLIV):
    """
    The base class for parametric DMLIV estimators to estimate a CATE. It accepts three generic machine
    learning models as nuisance functions:
    1) model_y_xw that estimates :math:`\\E[Y | X]`
    2) model_t_xw that estimates :math:`\\E[T | X]`
    3) model_t_xwz that estimates :math:`\\E[T | X, Z]`
    These are estimated in a cross-fitting manner for each sample in the training set.
    Then it minimizes the square loss:

    .. math::
        \\sum_i (Y_i - \\E[Y|X_i] - \\theta(X) * (\\E[T|X_i, Z_i] - \\E[T|X_i]))^2

    This loss is minimized by the model_final class, which is passed as an input.

    Parameters
    ----------
    model_y_xw: estimator, default ``'auto'``
        Determines how to fit the outcome to the features and controls (:math:`\\E[Y | X, W]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t_xw: estimator, default ``'auto'``
        Determines how to fit the treatment to the features and controls (:math:`\\E[T | X, W]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_t_xwz: estimator, default ``'auto'``
        Determines how to fit the treatment to the features, controls, and instrument (:math:`\\E[T | X, W, Z]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_final : estimator (default is :class:`.StatsModelsLinearRegression`)
        final model that at fit time takes as input :math:`(Y-\\E[Y|X])`, :math:`(\\E[T|X,Z]-\\E[T|X])` and X
        and supports method predict(X) that produces the CATE at X

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    fit_cate_intercept : bool, default True
        Whether the linear CATE model should have a constant term.

    discrete_instrument: bool, default False
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    discrete_outcome: bool, default False
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, default 2
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

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    allow_missing: bool
        Whether to allow missing values in X, W. If True, will need to supply nuisance models and model_final
        that can handle missing values.

    Examples
    --------
    A simple example with the default models:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.iv.dml import DMLIV

        # Define the data generation functions
        def dgp(n, p, true_fn):
            X = np.random.normal(0, 1, size=(n, p))
            Z = np.random.binomial(1, 0.5, size=(n,))
            nu = np.random.uniform(0, 10, size=(n,))
            coef_Z = 0.8
            C = np.random.binomial(
                1, coef_Z * scipy.special.expit(0.4 * X[:, 0] + nu)
            )  # Compliers when recomended
            C0 = np.random.binomial(
                1, 0.06 * np.ones(X.shape[0])
            )  # Non-compliers when not recommended
            T = C * Z + C0 * (1 - Z)
            y = true_fn(X) * T + 2 * nu + 5 * (X[:, 3] > 0) + 0.1 * np.random.uniform(0, 1, size=(n,))
            return y, T, Z, X

        def true_heterogeneity_function(X):
            return 5 * X[:, 0]

        np.random.seed(123)
        y, T, Z, X = dgp(1000, 5, true_heterogeneity_function)
        est = DMLIV(discrete_treatment=True, discrete_instrument=True)
        est.fit(Y=y, T=T, Z=Z, X=X)

    >>> est.effect(X[:3])
    array([-3.83383...,  5.31902..., -2.78082...])
    >>> est.coef_
    array([ 4.03889...,  0.89335...,  0.12043...,  0.37958..., -0.66097...])
    >>> est.intercept_
    -0.18482...

    """

    def __init__(self, *,
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_t_xwz="auto",
                 model_final=StatsModelsLinearRegression(fit_intercept=False),
                 featurizer=None,
                 fit_cate_intercept=True,
                 discrete_outcome=False,
                 discrete_treatment=False,
                 treatment_featurizer=None,
                 discrete_instrument=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False):
        self.model_y_xw = clone(model_y_xw, safe=False)
        self.model_t_xw = clone(model_t_xw, safe=False)
        self.model_t_xwz = clone(model_t_xwz, safe=False)
        self.model_final = clone(model_final, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.fit_cate_intercept = fit_cate_intercept
        super().__init__(discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         treatment_featurizer=treatment_featurizer,
                         discrete_instrument=discrete_instrument,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing)

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y_xw(self):
        return _make_first_stage_selector(self.model_y_xw, self.discrete_outcome, self.random_state)

    def _gen_model_t_xw(self):
        return _make_first_stage_selector(self.model_t_xw, self.discrete_treatment, self.random_state)

    def _gen_model_t_xwz(self):
        return _make_first_stage_selector(self.model_t_xwz, self.discrete_treatment, self.random_state)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_ortho_learner_model_nuisance(self):
        return _BaseDMLIVNuisanceSelector(self._gen_model_y_xw(), self._gen_model_t_xw(), self._gen_model_t_xwz())

    def _gen_ortho_learner_model_final(self):
        return _BaseDMLIVModelFinal(_FinalWrapper(self._gen_model_final(),
                                                  self.fit_cate_intercept,
                                                  self._gen_featurizer(),
                                                  False))

    @property
    def bias_part_of_coef(self):
        return self.ortho_learner_model_final_._model_final._fit_cate_intercept

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final_._model_final._fit_cate_intercept

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        if hasattr(self, "featurizer_") and self.featurizer_ is not None:
            X = self.featurizer_.transform(X)
        feature_names = self.cate_feature_names(feature_names)
        return _shap_explain_joint_linear_model_cate(self.model_final_, X, self._d_t, self._d_y,
                                                     self.bias_part_of_coef,
                                                     feature_names=feature_names, treatment_names=treatment_names,
                                                     output_names=output_names,
                                                     input_names=self._input_names,
                                                     background_samples=background_samples)

    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__

    @property
    def coef_(self):
        """ The coefficients in the linear model of the constant marginal treatment
        effect.

        Returns
        -------
        coef: (n_x,) or (n_t, n_x) or (n_y, n_t, n_x) array_like
            Where n_x is the number of features that enter the final model (either the
            dimension of X or the dimension of featurizer.fit_transform(X) if the CATE
            estimator has a featurizer.), n_t is the number of treatments, n_y is
            the number of outcomes. Dimensions are omitted if the original input was
            a vector and not a 2D array. For binary treatment the n_t dimension is
            also omitted.
        """
        return parse_final_model_params(self.model_final_.coef_, self.model_final_.intercept_,
                                        self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                        self.fit_cate_intercept_)[0]

    @property
    def intercept_(self):
        """ The intercept in the linear model of the constant marginal treatment
        effect.

        Returns
        -------
        intercept: float or (n_y,) or (n_y, n_t) array_like
            Where n_t is the number of treatments, n_y is
            the number of outcomes. Dimensions are omitted if the original input was
            a vector and not a 2D array. For binary treatment the n_t dimension is
            also omitted.
        """
        if not self.fit_cate_intercept_:
            raise AttributeError("No intercept was fitted!")
        return parse_final_model_params(self.model_final_.coef_, self.model_final_.intercept_,
                                        self._d_y, self._d_t, self._d_t_in, self.bias_part_of_coef,
                                        self.fit_cate_intercept_)[1]

    def summary(self, decimals=3, feature_names=None, treatment_names=None, output_names=None):
        """ The summary of coefficient and intercept in the linear model of the constant marginal treatment
        effect.

        Parameters
        ----------
        decimals: int, default 3
            Number of decimal places to round each column to.
        feature_names: list of str, optional
            The input of the feature names
        treatment_names: list of str, optional
            The names of the treatments
        output_names: list of str, optional
            The names of the outputs

        Returns
        -------
        smry : Summary instance
            this holds the summary tables and text, which can be printed or
            converted to various output formats.
        """
        # Get input names
        treatment_names = self.cate_treatment_names(treatment_names)
        output_names = self.cate_output_names(output_names)
        feature_names = self.cate_feature_names(feature_names)
        # Summary
        smry = Summary()

        extra_txt = ["<sub>A linear parametric conditional average treatment effect (CATE) model was fitted:"]

        if self._original_treatment_featurizer:
            extra_txt.append("$Y = \\Theta(X)\\cdot \\psi(T) + g(X, W) + \\epsilon$")
            extra_txt.append("where $\\psi(T)$ is the output of the `treatment_featurizer")
            extra_txt.append(
                "and for every outcome $i$ and featurized treatment $j$ the CATE $\\Theta_{ij}(X)$ has the form:")
        else:
            extra_txt.append("$Y = \\Theta(X)\\cdot T + g(X, W) + \\epsilon$")
            extra_txt.append(
                "where for every outcome $i$ and treatment $j$ the CATE $\\Theta_{ij}(X)$ has the form:")

        if self.featurizer:
            extra_txt.append("$\\Theta_{ij}(X) = \\phi(X)' coef_{ij} + cate\\_intercept_{ij}$")
            extra_txt.append("where $\\phi(X)$ is the output of the `featurizer`")
        else:
            extra_txt.append("$\\Theta_{ij}(X) = X' coef_{ij} + cate\\_intercept_{ij}$")

        extra_txt.append("Coefficient Results table portrays the $coef_{ij}$ parameter vector for "
                         "each outcome $i$ and treatment $j$. "
                         "Intercept Results table portrays the $cate\\_intercept_{ij}$ parameter.</sub>")

        smry.add_extra_txt(extra_txt)
        d_t = self._d_t[0] if self._d_t else 1
        d_y = self._d_y[0] if self._d_y else 1

        def _reshape_array(arr, type):
            if np.isscalar(arr):
                arr = np.array([arr])
            if type == 'coefficient':
                arr = np.moveaxis(arr, -1, 0)
            arr = arr.reshape(-1, 1)
            return arr

        # coefficient
        try:
            if self.coef_.size == 0:  # X is None
                raise AttributeError("X is None, please call intercept_inference to learn the constant!")
            else:
                coef_array = np.round(_reshape_array(self.coef_, "coefficient"), decimals)
                coef_headers = ["point_estimate"]
                if d_t > 1 and d_y > 1:
                    index = list(product(feature_names, output_names, treatment_names))
                elif d_t > 1:
                    index = list(product(feature_names, treatment_names))
                elif d_y > 1:
                    index = list(product(feature_names, output_names))
                else:
                    index = list(product(feature_names))
                coef_stubs = ["|".join(ind_value) for ind_value in index]
                coef_title = 'Coefficient Results'
                smry.add_table(coef_array, coef_headers, coef_stubs, coef_title)
        except Exception as e:
            print("Coefficient Results: ", str(e))

        # intercept
        try:
            if not self.fit_cate_intercept:
                raise AttributeError("No intercept was fitted!")
            else:
                intercept_array = np.round(_reshape_array(self.intercept_, "intercept"), decimals)
                intercept_headers = ["point_estimate"]
                if d_t > 1 and d_y > 1:
                    index = list(product(["cate_intercept"], output_names, treatment_names))
                elif d_t > 1:
                    index = list(product(["cate_intercept"], treatment_names))
                elif d_y > 1:
                    index = list(product(["cate_intercept"], output_names))
                else:
                    index = list(product(["cate_intercept"]))
                intercept_stubs = ["|".join(ind_value) for ind_value in index]
                intercept_title = 'CATE Intercept Results'
                smry.add_table(intercept_array, intercept_headers, intercept_stubs, intercept_title)
        except Exception as e:
            print("CATE Intercept Results: ", str(e))
        if len(smry.tables) > 0:
            return smry


class NonParamDMLIV(_BaseDMLIV):
    """
    The base class for non-parametric DMLIV that allows for an arbitrary square loss based ML
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
    model_y_xw: estimator, default ``'auto'``
        Determines how to fit the outcome to the features and controls (:math:`\\E[Y | X, W]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t_xw: estimator, default ``'auto'``
        Determines how to fit the treatment to the features and controls (:math:`\\E[T | X, W]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_t_xwz: estimator, default ``'auto'``
        Determines how to fit the treatment to the features, controls, and instrument (:math:`\\E[T | X, W, Z]`).

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    model_final : estimator
        final model for predicting :math:`\\tilde{Y}` from X with sample weights V(X)

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    discrete_outcome: bool, default False
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    discrete_instrument: bool, default False
        Whether the instrument values should be treated as categorical, rather than continuous, quantities

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, default 2
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

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    allow_missing: bool
        Whether to allow missing values in X, W. If True, will need to supply nuisance models and model_final
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

        from econml.iv.dml import NonParamDMLIV
        from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression

        # Define the data generation functions
        def dgp(n, p, true_fn):
            X = np.random.normal(0, 1, size=(n, p))
            Z = np.random.binomial(1, 0.5, size=(n,))
            nu = np.random.uniform(0, 10, size=(n,))
            coef_Z = 0.8
            C = np.random.binomial(
                1, coef_Z * scipy.special.expit(0.4 * X[:, 0] + nu)
            )  # Compliers when recomended
            C0 = np.random.binomial(
                1, 0.06 * np.ones(X.shape[0])
            )  # Non-compliers when not recommended
            T = C * Z + C0 * (1 - Z)
            y = true_fn(X) * T + 2 * nu + 5 * (X[:, 3] > 0) + 0.1 * np.random.uniform(0, 1, size=(n,))
            return y, T, Z, X

        def true_heterogeneity_function(X):
            return 5 * X[:, 0]

        np.random.seed(123)
        y, T, Z, X = dgp(1000, 5, true_heterogeneity_function)
        est = NonParamDMLIV(
            model_final=StatsModelsLinearRegression(),
            discrete_treatment=True, discrete_instrument=True,
            cv=5
        )
        est.fit(Y=y, T=T, Z=Z, X=X)

    >>> est.effect(X[:3])
    array([-5.98517...,  9.03610..., -3.56684...])

    """

    def __init__(self, *,
                 model_y_xw="auto",
                 model_t_xw="auto",
                 model_t_xwz="auto",
                 model_final,
                 discrete_outcome=False,
                 discrete_treatment=False,
                 treatment_featurizer=None,
                 discrete_instrument=False,
                 featurizer=None,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False):
        self.model_y_xw = clone(model_y_xw, safe=False)
        self.model_t_xw = clone(model_t_xw, safe=False)
        self.model_t_xwz = clone(model_t_xwz, safe=False)
        self.model_final = clone(model_final, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        super().__init__(discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         discrete_instrument=discrete_instrument,
                         treatment_featurizer=treatment_featurizer,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing)

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y_xw(self):
        return _make_first_stage_selector(self.model_y_xw, self.discrete_outcome, self.random_state)

    def _gen_model_t_xw(self):
        return _make_first_stage_selector(self.model_t_xw, self.discrete_treatment, self.random_state)

    def _gen_model_t_xwz(self):
        return _make_first_stage_selector(self.model_t_xwz, self.discrete_treatment, self.random_state)

    def _gen_model_final(self):
        return clone(self.model_final, safe=False)

    def _gen_ortho_learner_model_nuisance(self):
        return _BaseDMLIVNuisanceSelector(self._gen_model_y_xw(), self._gen_model_t_xw(), self._gen_model_t_xwz())

    def _gen_ortho_learner_model_final(self):
        return _BaseDMLIVModelFinal(_FinalWrapper(self._gen_model_final(),
                                                  False,
                                                  self._gen_featurizer(),
                                                  True))

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        return _shap_explain_model_cate(self.const_marginal_effect, self.model_cate, X, self._d_t, self._d_y,
                                        featurizer=self.featurizer_,
                                        feature_names=feature_names,
                                        treatment_names=treatment_names,
                                        output_names=output_names,
                                        input_names=self._input_names,
                                        background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__
