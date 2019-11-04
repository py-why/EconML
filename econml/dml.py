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
from .utilities import (shape, reshape, ndim, hstack, cross_product, transpose,
                        broadcast_unit_treatments, reshape_treatmentwise_effects,
                        StatsModelsLinearRegression, LassoCVWrapper)
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import (PolynomialFeatures, LabelEncoder, OneHotEncoder,
                                   FunctionTransformer)
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from .cate_estimator import (BaseCateEstimator, LinearCateEstimator,
                             TreatmentExpansionMixin, StatsModelsCateEstimatorMixin)
from .inference import StatsModelsInference
from ._rlearner import _RLearner


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

    n_splits: int, cross-validation generator or an iterable, optional
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
        by `np.random`.
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

        class FirstStageWrapper:
            def __init__(self, model, is_Y):
                self._model = clone(model, safe=False)
                self._featurizer = clone(featurizer, safe=False)
                self._is_Y = is_Y

            def _combine(self, X, W, n_samples, fitting=True):
                if self._is_Y and linear_first_stages:
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
                if (not self._is_Y) and discrete_treatment:
                    Target = np.matmul(Target, np.arange(1, Target.shape[1] + 1)).flatten()

                if sample_weight is not None:
                    self._model.fit(self._combine(X, W, Target.shape[0]), Target, sample_weight=sample_weight)
                else:
                    self._model.fit(self._combine(X, W, Target.shape[0]), Target)

            def predict(self, X, W):
                n_samples = X.shape[0] if X is not None else (W.shape[0] if W is not None else 1)
                if (not self._is_Y) and discrete_treatment:
                    return self._model.predict_proba(self._combine(X, W, n_samples, fitting=False))[:, 1:]
                else:
                    return self._model.predict(self._combine(X, W, n_samples, fitting=False))

        class FinalWrapper:
            def __init__(self):
                self._model = clone(model_final, safe=False)
                self._featurizer = clone(featurizer, safe=False)

            def fit(self, X, T_res, Y_res, sample_weight=None, sample_var=None):
                # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
                self._d_t = shape(T_res)[1:]
                self._d_y = shape(Y_res)[1:]
                F = self._featurizer.fit_transform(X) if X is not None else np.ones((T_res.shape[0], 1))
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

            def predict(self, X):
                F = self._featurizer.transform(X) if X is not None else np.ones((1, 1))
                F, T = broadcast_unit_treatments(F, self._d_t[0] if self._d_t else 1)
                prediction = self._model.predict(cross_product(F, T))
                prediction -= self._intercept if self._intercept is not None else 0
                return reshape_treatmentwise_effects(prediction,
                                                     self._d_t, self._d_y)

        super().__init__(model_y=FirstStageWrapper(model_y, is_Y=True),
                         model_t=FirstStageWrapper(model_t, is_Y=False),
                         model_final=FinalWrapper(),
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

    featurizer: transformer, optional
    (default is :class:`PolynomialFeatures(degree=1, include_bias=True) <sklearn.preprocessing.PolynomialFeatures>`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional
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
        by `np.random`.

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
        inference: string, `Inference` instance, or None
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


class SparseLinearDMLCateEstimator(DMLCateEstimator):
    """
    A specialized version of the Double ML estimator for the sparse linear case.

    Specifically, this estimator can be used when the controls are high-dimensional
    and the coefficients of the nuisance functions are sparse.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator
        The estimator for fitting the treatment to the features. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    model_final: estimator, optional (default is :class:`LassoCV(fit_intercept=False)  <sklearn.linear_model.LassoCV>`)
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model with no intercept for correctness.

    featurizer: transformer, optional
    (default is :class:`PolynomialFeatures(degree=1, include_bias=True) <sklearn.preprocessing.PolynomialFeatures>`)
        The transformer used to featurize the raw features when fitting the final model.  Must implement
        a `fit_transform` method.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional (default is ``False``)
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional
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
        by `np.random`.
    """

    def __init__(self,
                 model_y=LassoCV(), model_t=LassoCV(), model_final=LassoCVWrapper(fit_intercept=False),
                 featurizer=PolynomialFeatures(degree=1, include_bias=True),
                 linear_first_stages=True,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=model_final,
                         featurizer=featurizer,
                         linear_first_stages=linear_first_stages,
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)


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

    n_splits: int, cross-validation generator or an iterable, optional
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
        by `np.random`.
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
