# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Double Machine Learning. The method uses machine learning methods to identify the
part of the observed outcome and treatment that is not predictable by the controls X, W
(aka residual outcome and residual treatment).
Then estimates a CATE model by regressing the residual outcome on the residual treatment
in a manner that accounts for heterogeneity in the regression coefficient, with respect
to X.

References
----------

\\ V. Chernozhukov, D. Chetverikov, M. Demirer, E. Duflo, C. Hansen, and a. W. Newey.
    Double Machine Learning for Treatment and Causal Parameters.
    https://arxiv.org/abs/1608.00060, 2016.

\\ X. Nie and S. Wager.
    Quasi-Oracle Estimation of Heterogeneous Treatment Effects.
    arXiv preprint arXiv:1712.04912, 2017. URL http://arxiv.org/abs/1712.04912.

\\ V. Chernozhukov, M. Goldman, V. Semenova, and M. Taddy.
    Orthogonal Machine Learning for Demand Estimation: High Dimensional Causal Inference in Dynamic Panels.
    https://arxiv.org/abs/1712.09988, December 2017.

\\ V. Chernozhukov, D. Nekipelov, V. Semenova, and V. Syrgkanis.
    Two-Stage Estimation with a High-Dimensional Second Stage.
    https://arxiv.org/abs/1806.04823, 2018.

\\ Dylan Foster, Vasilis Syrgkanis (2019).
    Orthogonal Statistical Learning.
    ACM Conference on Learning Theory. https://arxiv.org/abs/1901.09036

"""

import numpy as np
import copy
from warnings import warn
from .utilities import (shape, reshape, ndim, hstack, cross_product, transpose, inverse_onehot,
                        broadcast_unit_treatments, reshape_treatmentwise_effects,
                        StatsModelsLinearRegression, LassoCVWrapper, check_high_dimensional)
from econml.sklearn_extensions.linear_model import MultiOutputDebiasedLasso, WeightedLassoCVWrapper
from sklearn.model_selection import KFold, StratifiedKFold, check_cv
from sklearn.linear_model import LinearRegression, LassoCV, LogisticRegressionCV, ElasticNetCV
from sklearn.preprocessing import (PolynomialFeatures, LabelEncoder, OneHotEncoder,
                                   FunctionTransformer)
from sklearn.base import clone, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state
from .cate_estimator import (BaseCateEstimator, LinearCateEstimator,
                             TreatmentExpansionMixin, StatsModelsCateEstimatorMixin,
                             DebiasedLassoCateEstimatorMixin)
from .inference import StatsModelsInference
from ._rlearner import _RLearner
from .sklearn_extensions.model_selection import WeightedStratifiedKFold


class DMLCateEstimator(_RLearner):
    """
    The base class for parametric Double ML estimators. The estimator is a special
    case of an :class:`~econml._rlearner._RLearner` estimator, which in turn is a special case
    of an :class:`~econml._ortho_learner._OrthoLearner` estimator, so it follows the two
    stage process, where a set of nuisance functions are estimated in the first stage in a crossfitting
    manner and a final stage estimates the CATE model. See the documentation of
    :class:`~econml._ortho_learner._OrthoLearner` for a description of this two stage process.

    In this estimator, the CATE is estimated by using the following estimating equations:

    .. math ::
        Y - \\E[Y | X, W] = \\Theta(X) \\cdot (T - \\E[T | X, W]) + \\epsilon

    Thus if we estimate the nuisance functions :math:`q(X, W) = \\E[Y | X, W]` and
    :math:`f(X, W)=\\E[T | X, W]` in the first stage, we can estimate the final stage cate for each
    treatment t, by running a regression, minimizing the residual on residual square loss:

    .. math ::
        \\hat{\\theta} = \\arg\\min_{\\Theta}\
        \\E_n\\left[ (\\tilde{Y} - \\Theta(X) \\cdot \\tilde{T})^2 \\right]

    Where :math:`\\tilde{Y}=Y - \\E[Y | X, W]` and :math:`\\tilde{T}=T-\\E[T | X, W]` denotes the
    residual outcome and residual treatment.

    The DMLCateEstimator further assumes a linear parametric form for the cate, i.e. for each outcome
    :math:`i` and treatment :math:`j`:

    .. math ::
        \\Theta_{i, j}(X) =  \\phi(X)' \\cdot \\Theta_{ij}

    For some given feature mapping :math:`\\phi(X)` (the user can provide this featurizer via the `featurizer`
    parameter at init time and could be any arbitrary class that adheres to the scikit-learn transformer
    interface :py:class:`~sklearn.base.TransformerMixin`).

    The second nuisance function :math:`q` is a simple regression problem and the
    :class:`~econml.dml.DMLCateEstimator`
    class takes as input the parameter `model_y`, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem.

    The problem of estimating the nuisance function :math:`f` is also a regression problem and
    the :class:`~econml.dml.DMLCateEstimator`
    class takes as input the parameter `model_t`, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem. If the init flag `discrete_treatment` is set
    to `True`, then the parameter `model_t` is treated as a scikit-learn classifier. The input categorical
    treatment is one-hot encoded (excluding the lexicographically smallest treatment which is used as the
    baseline) and the `predict_proba` method of the `model_t` classifier is used to
    residualize the one-hot encoded treatment.

    The final stage is (potentially multi-task) linear regression problem with outcomes the labels
    :math:`\\tilde{Y}` and regressors the composite features
    :math:`\\tilde{T}\\otimes \\phi(X) = \\mathtt{vec}(\\tilde{T}\\cdot \\phi(X)^T)`.
    The :class:`~econml.dml.DMLCateEstimator` takes as input parameter
    ``model_final``, which is any linear scikit-learn regressor that is internally used to solve this
    (multi-task) linear regresion problem.

    Parameters
    ----------
    model_y: estimator
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.  Must be a linear model for correctness when linear_first_stages is ``True``.

    model_t: estimator or 'auto' (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods.  Must be a linear model for correctness
        when linear_first_stages is ``True``;
        If 'auto', :class:`LogisticRegressionCV() <sklearn.linear_model.LogisticRegressionCV>`
        will be applied for discrete treatment,
        and :class:`WeightedLassoCV() <econml.sklearn_extensions.linear_model.WeightedLassoCV>`/
        :class:`WeightedMultitaskLassoCV() <econml.sklearn_extensions.linear_model.WeightedMultitaskLassoCV>`
        will be applied for continuous treatment.

    model_final: estimator
        The estimator for fitting the response residuals to the treatment residuals. Must implement
        `fit` and `predict` methods, and must be a linear model for correctness.

    featurizer: :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

    linear_first_stages: bool
        Whether the first stage models are linear (in which case we will expand the features passed to
        `model_y` accordingly)

    discrete_treatment: bool, optional, default False
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

    n_splits: int, cross-validation generator or an iterable, optional, default 2
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
                 featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages=False,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):

        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies

        if model_t == 'auto':
            if discrete_treatment:
                model_t = LogisticRegressionCV(cv=WeightedStratifiedKFold())
            else:
                model_t = WeightedLassoCVWrapper()

        class FirstStageWrapper:
            def __init__(self, model, is_Y):
                self._model = clone(model, safe=False)
                self._featurizer = clone(featurizer, safe=False)
                self._is_Y = is_Y

            def _combine(self, X, W, n_samples, fitting=True):
                if X is None:
                    # if both X and W are None, just return a column of ones
                    return (W if W is not None else np.ones((n_samples, 1)))
                XW = hstack([X, W]) if W is not None else X
                if self._is_Y and linear_first_stages:
                    if self._featurizer is None:
                        F = X
                    else:
                        F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
                    return cross_product(XW, hstack([np.ones((shape(XW)[0], 1)), F]))
                else:
                    return XW

            def fit(self, X, W, Target, sample_weight=None):
                if (not self._is_Y) and discrete_treatment:
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
                if (not self._is_Y) and discrete_treatment:
                    return self._model.predict_proba(self._combine(X, W, n_samples, fitting=False))[:, 1:]
                else:
                    return self._model.predict(self._combine(X, W, n_samples, fitting=False))

        class FinalWrapper:
            def __init__(self):
                self._model = clone(model_final, safe=False)
                self._original_featurizer = clone(featurizer, safe=False)
                self._fit_cate_intercept = fit_cate_intercept
                if self._fit_cate_intercept:
                    add_intercept = FunctionTransformer(lambda F:
                                                        hstack([np.ones((F.shape[0], 1)), F]))
                    if featurizer:
                        self._featurizer = Pipeline([('featurize', self._original_featurizer),
                                                     ('add_intercept', add_intercept)])
                    else:
                        self._featurizer = add_intercept
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

            def fit(self, X, T_res, Y_res, sample_weight=None, sample_var=None):
                # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
                self._d_t = shape(T_res)[1:]
                self._d_y = shape(Y_res)[1:]
                fts = self._combine(X, T_res)
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
                X2, T = broadcast_unit_treatments(X if X is not None else np.empty((1, 0)),
                                                  self._d_t[0] if self._d_t else 1)
                prediction = self._model.predict(self._combine(None if X is None else X2, T, fitting=False))
                if self._intercept is not None:
                    prediction -= self._intercept
                return reshape_treatmentwise_effects(prediction,
                                                     self._d_t, self._d_y)
        self.bias_part_of_coef = fit_cate_intercept
        self.fit_cate_intercept = fit_cate_intercept
        super().__init__(model_y=FirstStageWrapper(model_y, is_Y=True),
                         model_t=FirstStageWrapper(model_t, is_Y=False),
                         model_final=FinalWrapper(),
                         discrete_treatment=discrete_treatment,
                         n_splits=n_splits,
                         random_state=random_state)

    @property
    def original_featurizer(self):
        return super().model_final._original_featurizer

    @property
    def featurizer(self):
        # NOTE This is used by the inference methods and has to be the overall featurizer. intended
        # for internal use by the library
        return super().model_final._featurizer

    @property
    def model_final(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        return super().model_final._model

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
        return super().model_final._model

    @property
    def models_y(self):
        """
        Get the fitted models for E[Y | X, W].

        Returns
        -------
        models_y: list of objects of type(`model_y`)
            A list of instances of the `model_y` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model for mdl in super().models_y]

    @property
    def models_t(self):
        """
        Get the fitted models for E[T | X, W].

        Returns
        -------
        models_y: list of objects of type(`model_t`)
            A list of instances of the `model_y` object. Each element corresponds to a crossfitting
            fold and is the model instance that was fitted for that training fold.
        """
        return [mdl._model for mdl in super().models_t]

    def cate_feature_names(self, input_feature_names=None):
        """
        Get the output feature names.

        Parameters
        ----------
        input_feature_names: list of strings of length X.shape[1] or None
            The names of the input features

        Returns
        -------
        out_feature_names: list of strings or None
            The names of the output features :math:`\\phi(X)`, i.e. the features with respect to which the
            final constant marginal CATE model is linear. It is the names of the features that are associated
            with each entry of the :meth:`coef_` parameter. Not available when the featurizer is not None and
            does not have a method: `get_feature_names(input_feature_names)`. Otherwise None is returned.
        """
        if self.original_featurizer is None:
            return input_feature_names
        elif hasattr(self.original_featurizer, 'get_feature_names'):
            return self.original_featurizer.get_feature_names(input_feature_names)
        else:
            raise AttributeError("Featurizer does not have a method: get_feature_names!")


class LinearDMLCateEstimator(StatsModelsCateEstimatorMixin, DMLCateEstimator):
    """
    The Double ML Estimator with a low-dimensional linear final stage implemented as a statsmodel regression.

    Parameters
    ----------
    model_y: estimator, optional (default is :class:`WeightedLassoCVWrapper()
        <econml.sklearn_extensions.linear_model.WeightedLassoCVWrapper>`)
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods;
        If 'auto', :class:`LogisticRegressionCV() <sklearn.linear_model.LogisticRegressionCV>`
        will be applied for discrete treatment,
        and :class:`WeightedLassoCV() <econml.sklearn_extensions.linear_model.WeightedLassoCV>`/
        :class:`WeightedMultitaskLassoCV() <econml.sklearn_extensions.linear_model.WeightedMultitaskLassoCV>`
        will be applied for continuous treatment.

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

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
                 model_y=WeightedLassoCVWrapper(), model_t='auto',
                 featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages=True,
                 discrete_treatment=False,
                 n_splits=2,
                 random_state=None):
        super().__init__(model_y=model_y,
                         model_t=model_t,
                         model_final=StatsModelsLinearRegression(fit_intercept=False),
                         featurizer=featurizer,
                         fit_cate_intercept=fit_cate_intercept,
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
    model_y: estimator, optional (default is :class:`WeightedLassoCVWrapper()
        <econml.sklearn_extensions.linear_model.WeightedLassoCVWrapper>`)
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods, and must be a
        linear model for correctness;
        If 'auto', :class:`LogisticRegressionCV() <sklearn.linear_model.LogisticRegressionCV>`
        will be applied for discrete treatment,
        and :class:`WeightedLassoCV() <econml.sklearn_extensions.linear_model.WeightedLassoCV>`/
        :class:`WeightedMultitaskLassoCV() <econml.sklearn_extensions.linear_model.WeightedMultitaskLassoCV>`
        will be applied for continuous treatment.

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

    featurizer : :term:`transformer`, optional, default None
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

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
                 model_y=WeightedLassoCVWrapper(), model_t='auto',
                 alpha='auto',
                 max_iter=1000,
                 tol=1e-4,
                 featurizer=None,
                 fit_cate_intercept=True,
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
                         fit_cate_intercept=fit_cate_intercept,
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


class KernelDMLCateEstimator(DMLCateEstimator):
    """
    A specialized version of the linear Double ML Estimator that uses random fourier features.

    Parameters
    ----------
    model_y: estimator, optional (default is :class:`<econml.sklearn_extensions.linear_model.WeightedLassoCVWrapper>`)
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods;
        If 'auto', :class:`LogisticRegressionCV() <sklearn.linear_model.LogisticRegressionCV>`
        will be applied for discrete treatment,
        and :class:`WeightedLassoCV() <econml.sklearn_extensions.linear_model.WeightedLassoCV>`/
        :class:`WeightedMultitaskLassoCV() <econml.sklearn_extensions.linear_model.WeightedMultitaskLassoCV>`
        will be applied for continuous treatment.

    fit_cate_intercept : bool, optional, default True
        Whether the linear CATE model should have a constant term.

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

    def __init__(self, model_y=WeightedLassoCVWrapper(), model_t='auto', fit_cate_intercept=True,
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
                         model_final=ElasticNetCV(),
                         featurizer=RandomFeatures(random_state),
                         fit_cate_intercept=fit_cate_intercept,
                         discrete_treatment=discrete_treatment, n_splits=n_splits, random_state=random_state)
