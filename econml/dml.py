# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Double ML.

"Double Machine Learning" is an algorithm that applies arbitrary machine learning methods
to fit the treatment and response, then uses a linear model to predict the response residuals
from the treatment residuals.

"""

import numpy as np
import copy
from warnings import warn
from .utilities import (shape, reshape, ndim, hstack, cross_product, transpose, inverse_onehot,
                        broadcast_unit_treatments, reshape_treatmentwise_effects,
                        StatsModelsLinearRegression, LassoCVWrapper, check_high_dimensional)
from econml.sklearn_extensions.linear_model import MultiOutputDebiasedLasso
from econml.sklearn_extensions.ensemble import SubsampledHonestForest
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import (PolynomialFeatures, LabelEncoder, OneHotEncoder,
                                   FunctionTransformer)
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from .cate_estimator import (BaseCateEstimator, LinearCateEstimator,
                             TreatmentExpansionMixin, StatsModelsCateEstimatorMixin,
                             DebiasedLassoCateEstimatorMixin)
from .inference import StatsModelsInference, GenericModelFinalInference
from ._rlearner import _RLearner


class _FirstStageWrapper:
    def __init__(self, model, is_Y, featurizer, linear_first_stages, discrete_treatment):
        self._model = clone(model, safe=False)
        self._featurizer = clone(featurizer, safe=False)
        self._is_Y = is_Y
        self._linear_first_stages = linear_first_stages
        self._discrete_treatment = discrete_treatment

    def _combine(self, X, W, n_samples, fitting=True):
        if self._is_Y and self._linear_first_stages:
            if X is not None:
                F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
            else:
                X = np.ones((n_samples, 1))
                F = np.ones((n_samples, 1))
            if W is None:
                W = np.empty((n_samples, 0))
            XW = hstack([X, W])
            return cross_product(XW, hstack([np.ones((shape(XW)[0], 1)), F, W]))
        else:
            if X is None:
                X = np.ones((n_samples, 1))
            if W is None:
                W = np.empty((n_samples, 0))
            return hstack([X, W])

    def fit(self, X, W, Target, sample_weight=None):
        if (not self._is_Y) and self._discrete_treatment:
            # In this case, the Target is the one-hot-encoding of the treatment variable
            # We need to go back to the label representation of the one-hot so as to call
            # the classifier.
            if np.any(np.all(Target == 0, axis=0)) or (not np.any(np.all(Target == 0, axis=1))):
                raise AttributeError("Provided crossfit folds contain training splits that " +
                                     "don't contain all treatments")
            Target = inverse_onehot(Target)

        if sample_weight is not None:
            self._model.fit(self._combine(X, W, Target.shape[0]), Target, sample_weight=sample_weight)
        else:
            self._model.fit(self._combine(X, W, Target.shape[0]), Target)

    def predict(self, X, W):
        n_samples = X.shape[0] if X is not None else (W.shape[0] if W is not None else 1)
        if (not self._is_Y) and self._discrete_treatment:
            return self._model.predict_proba(self._combine(X, W, n_samples, fitting=False))[:, 1:]
        else:
            return self._model.predict(self._combine(X, W, n_samples, fitting=False))


class _FinalWrapper:
    def __init__(self, model_final, featurizer, use_weight_trick):
        self._model = clone(model_final, safe=False)
        self._featurizer = clone(featurizer, safe=False)
        self._use_weight_trick = use_weight_trick

    def fit(self, X, T_res, Y_res, sample_weight=None, sample_var=None):
        # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T_res)[1:]
        self._d_y = shape(Y_res)[1:]
        F = self._featurizer.fit_transform(X) if X is not None else np.ones((T_res.shape[0], 1))
        if not self._use_weight_trick:
            fts = cross_product(F, T_res)
            if sample_weight is not None:
                if sample_var is not None:
                    self._model.fit(fts,
                                    Y_res, sample_weight=sample_weight, sample_var=sample_var)
                else:
                    self._model.fit(fts,
                                    Y_res, sample_weight=sample_weight)
            else:
                self._model.fit(fts, Y_res)

            self._intercept = None
            intercept = self._model.predict(np.zeros_like(fts[0:1]))
            if (np.count_nonzero(intercept) > 0):
                warn("The final model has a nonzero intercept for at least one outcome; "
                     "it will be subtracted, but consider fitting a model without an intercept if possible.",
                     UserWarning)
                self._intercept = intercept
        else:
            if (np.ndim(T_res) > 1) and (self._d_t[0] > 1):
                raise AttributeError("This method can only be used with single-dimensional continuous treatment "
                                     "or binary categorical treatment.")
            self._intercept = None
            T_res = T_res.ravel()
            sign_T_res = np.sign(T_res)
            sign_T_res[sign_T_res == 0] = 1
            clipped_T_res = np.sign(T_res) * np.clip(np.abs(T_res), 1e-5, np.inf)
            if np.ndim(Y_res) > 1:
                target = Y_res / clipped_T_res.reshape(-1, 1)
                target_var = sample_var / (clipped_T_res**2).reshape(-1, 1) if sample_var is not None else None
            else:
                target = Y_res / clipped_T_res
                target_var = sample_var / clipped_T_res**2 if sample_var is not None else None

            if sample_weight is not None:
                if target_var is not None:
                    self._model.fit(F, target, sample_weight=sample_weight * T_res.flatten()**2,
                                    sample_var=target_var)
                else:
                    self._model.fit(F, target, sample_weight=sample_weight * T_res.flatten()**2)
            else:
                self._model.fit(F, target, sample_weight=T_res.flatten()**2)

    def predict(self, X):
        F = self._featurizer.transform(X) if X is not None else np.ones((1, 1))
        F, T = broadcast_unit_treatments(F, self._d_t[0] if self._d_t else 1)
        prediction = self._model.predict(cross_product(F, T))
        if self._intercept is not None:
            prediction -= self._intercept
        return reshape_treatmentwise_effects(prediction,
                                             self._d_t, self._d_y)


class DMLCateEstimator(_RLearner):
    """
    The base class for parametric Double ML estimators.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional (Default=2)
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

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self,
                 model_y, model_t, model_final,
                 featurizer,
                 linear_first_stages=False,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):

        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies

        super().__init__(model_y=_FirstStageWrapper(model_y, True,
                                                    featurizer, linear_first_stages, discrete_treatment),
                         model_t=_FirstStageWrapper(model_t, False,
                                                    featurizer, linear_first_stages, discrete_treatment),
                         model_final=_FinalWrapper(model_final, featurizer, False),
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)

    @property
    def featurizer(self):
        return super().model_final._featurizer

    @property
    def model_final(self):
        return super().model_final._model

    @property
    def models_y(self):
        return [mdl._model for mdl in super().models_y]

    @property
    def models_t(self):
        return [mdl._model for mdl in super().models_t]


class LinearDMLCateEstimator(StatsModelsCateEstimatorMixin, DMLCateEstimator):
    """
    The Double ML Estimator with a low-dimensional linear final stage implemented as a statsmodel regression.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.

    featurizer: transformer, optional (default is \
        :class:`PolynomialFeatures(degree=1, include_bias=True) <sklearn.preprocessing.PolynomialFeatures>`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional (Default=2)
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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.

    """

    def __init__(self,
                 model_y=LassoCV(), model_t=LassoCV(),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 linear_first_stages=True,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=StatsModelsLinearRegression(fit_intercept=False),
                         featurizer=featurizer,
                         linear_first_stages=linear_first_stages,
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)

    # override only so that we can update the docstring to indicate support for `StatsModelsInference`
    def fit(self, Y, T, X=None, W=None, sample_weight=None, sample_var=None, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: optional (n × dₓ) matrix
            Features for each sample
        W: optional (n × d_w) matrix
            Controls for each sample
        sample_weight: optional (n,) vector
            Weights for each row
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'statsmodels'
            (or an instance of :class:`.StatsModelsInference`)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, sample_var=sample_var, inference=inference)

    @property
    def statsmodels(self):
        return self.model_final


class SparseLinearDMLCateEstimator(DebiasedLassoCateEstimatorMixin, DMLCateEstimator):
    """
    A specialized version of the Double ML estimator for the sparse linear case.

    This estimator should be used when the features of heterogeneity are high-dimensional
    and the coefficients of the linear CATE function are sparse.

    The last stage is an instance of the
    :class:`MultiOutputDebiasedLasso <econml.sklearn_extensions.linear_model.MultiOutputDebiasedLasso>`

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    alpha: string | float, optional. Default='auto'.
        CATE L1 regularization applied through the debiased lasso in the final model.
        'auto' corresponds to a CV form of the :class:`MultiOutputDebiasedLasso`.

    max_iter : int, optional, default=1000
        The maximum number of iterations in the Debiased Lasso

    tol : float, optional, default=1e-4
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    featurizer: transformer, optional
    (default is :class:`PolynomialFeatures(degree=1, include_bias=True) <sklearn.preprocessing.PolynomialFeatures>`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional (Default=2)
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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self,
                 model_y=LassoCV(), model_t=LassoCV(),
                 alpha='auto',
                 max_iter=1000,
                 tol=1e-4,
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 linear_first_stages=True,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):
        model_final = MultiOutputDebiasedLasso(
            alpha=alpha,
            fit_intercept=False,
            max_iter=max_iter,
            tol=tol)
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=model_final,
                         featurizer=featurizer,
                         linear_first_stages=linear_first_stages,
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)

    def fit(self, Y, T, X=None, W=None, sample_weight=None, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: optional (n × dₓ) matrix
            Features for each sample
        W: optional (n × d_w) matrix
            Controls for each sample
        sample_weight: optional (n,) vector
            Weights for each row
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'debiasedlasso'
            (or an instance of :class:`.LinearCateInference`)

        Returns
        -------
        self
        """
        # TODO: support sample_var
        if sample_weight is not None and inference is not None:
            warn("This estimator does not yet support sample variances and inference does not take "
                 "sample variances into account. This feature will be supported in a future release.")
        check_high_dimensional(X, T, threshold=5, featurizer=self.featurizer,
                               discrete_treatment=self._discrete_treatment,
                               msg="The number of features in the final model (< 5) is too small for a sparse model. "
                               "We recommend using the LinearDMLCateEstimator for this low-dimensional setting.")
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, sample_var=None, inference=inference)


class KernelDMLCateEstimator(LinearDMLCateEstimator):
    """
    A specialized version of the linear Double ML Estimator that uses random fourier features.

    Parameters
    ----------
    model_y: estimator, optional (default is :class:`LassoCV() <sklearn.linear_model.LassoCV>`)
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator, optional (default is :class:`LassoCV() <sklearn.linear_model.LassoCV>`)
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.

    dim: int, optional (default is 20)
        The number of random Fourier features to generate

    bw: float, optional (default is 1.0)
        The bandwidth of the Gaussian used to generate features

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional (Default=2)
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

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, model_y=LassoCV(), model_t=LassoCV(),
                 dim=20, bw=1.0, discrete_treatment=False, n_splits=2, random_state=None):
        class RandomFeatures(TransformerMixin):
            def __init__(self, random_state):
                self._random_state = check_random_state(random_state)

            def fit(self, X):
                self.omegas = self._random_state.normal(0, 1 / bw, size=(shape(X)[1], dim))
                self.biases = self._random_state.uniform(0, 2 * np.pi, size=(1, dim))
                return self

            def transform(self, X):
                return np.sqrt(2 / dim) * np.cos(np.matmul(X, self.omegas) + self.biases)

        super().__init__(model_y=model_y, model_t=model_t,
                         featurizer=RandomFeatures(random_state),
                         discrete_treatment=discrete_treatment, n_splits=n_splits, random_state=random_state)


class NonParamDMLCateEstimator(_RLearner):
    """
    The base class for non-parametric Double ML estimators, that can have arbitrary final ML models of the CATE.
    Works only for single-dimensional continuous treatment or for binary categorical treatment and uses
    the re-weighting trick, reducing the final CATE estimation to a weighted square loss minimization.
    The model_final parameter must support the sample_weight keyword argument at fit time.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness. The `fit` method must
        accept `sample_weight` as a keyword argument.

    featurizer: transformer
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional (Default=2)
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

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self,
                 model_y, model_t, model_final,
                 featurizer,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):

        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies

        super().__init__(model_y=_FirstStageWrapper(model_y, True,
                                                    featurizer, False, discrete_treatment),
                         model_t=_FirstStageWrapper(model_t, False,
                                                    featurizer, False, discrete_treatment),
                         model_final=_FinalWrapper(model_final, featurizer, True),
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)

    @property
    def featurizer(self):
        return super().model_final._featurizer

    @property
    def model_final(self):
        return super().model_final._model

    @property
    def models_y(self):
        return [mdl._model for mdl in super().models_y]

    @property
    def models_t(self):
        return [mdl._model for mdl in super().models_t]


class ForestDMLCateEstimator(NonParamDMLCateEstimator):
    """ Instance of NonParamDMLCateEstimator with a subsampled honest forest
    as a final model, so as to enable non-parametric inference.
    See ``.sklearn_extensions.SubsampledHonestForest`` for a description of each of the input parameters.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_crossfit_splits: int, cross-validation generator or an iterable, optional (Default=2)
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

    n_estimators : integer, optional (default=100)
        The total number of trees in the forest. The forest consists of a
        forest of sqrt(n_estimators) sub-forests, where each sub-forest
        contains sqrt(n_estimators) trees.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.

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

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

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

    subsample_fr : float or 'auto', optional (default='auto')
        The fraction of the half-samples that are used on each tree. Each tree
        will be built on subsample_fr * n_samples/2.

        If 'auto', then the subsampling fraction is set to::

            (n_samples/2)**(1-1/(2*n_features+2))/(n_samples/2)

        which is sufficient to guarantee asympotitcally valid inference.

    honest : boolean, optional (default=True)
        Whether to use honest trees, i.e. half of the samples are used for
        creating the tree structure and the other half for the estimation at
        the leafs. If False, then all samples are used for both parts.

    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both `fit` and `predict`.
        `None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.

    random_state: int, :class:`~numpy.random.mtrand.RandomState` instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self,
                 model_y, model_t,
                 discrete_treatment=False,
                 n_crossfit_splits=2,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 subsample_fr='auto',
                 honest=True,
                 n_jobs=None,
                 verbose=0,
                 random_state=None):
        model_final = SubsampledHonestForest(n_estimators=n_estimators,
                                             criterion=criterion,
                                             max_depth=max_depth,
                                             min_samples_split=min_samples_split,
                                             min_samples_leaf=min_samples_leaf,
                                             min_weight_fraction_leaf=min_weight_fraction_leaf,
                                             max_features=max_features,
                                             max_leaf_nodes=max_leaf_nodes,
                                             min_impurity_decrease=min_impurity_decrease,
                                             subsample_fr=subsample_fr,
                                             honest=honest,
                                             n_jobs=n_jobs,
                                             random_state=random_state,
                                             verbose=verbose)
        super().__init__(model_y=model_y, model_t=model_t,
                         model_final=model_final, featurizer=FunctionTransformer(),
                         discrete_treatment=discrete_treatment,
                         n_splits=n_crossfit_splits, random_state=random_state)

    def _get_inference_options(self):
        # add statsmodels to parent's options
        options = super()._get_inference_options()
        options.update(blb=GenericModelFinalInference)
        return options

    def fit(self, Y, T, X=None, W=None, sample_weight=None, sample_var=None, inference=None):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: optional (n × dₓ) matrix
            Features for each sample
        W: optional (n × d_w) matrix
            Controls for each sample
        sample_weight: optional (n,) vector
            Weights for each row
        inference: string, `Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'blb'
            (for Bootstrap-of-Little-Bags based inference)

        Returns
        -------
        self
        """
        return super().fit(Y, T, X=X, W=W, sample_weight=sample_weight, sample_var=None, inference=inference)
