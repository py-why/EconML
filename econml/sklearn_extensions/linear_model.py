# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Collection of scikit-learn extensions for linear models.

.. testsetup::

    # Our classes that derive from sklearn ones sometimes include
    # inherited docstrings that have embedded doctests; we need the following imports
    # so that they don't break.

    import numpy as np
    from sklearn.linear_model import lasso_path
"""

from __future__ import annotations  # needed to allow type signature to refer to containing type

import numbers
import numpy as np
import warnings
from collections.abc import Iterable
from scipy.stats import norm
from ..utilities import ndim, shape, reshape, _safe_norm_ppf, check_input_arrays
from sklearn import clone
from sklearn.linear_model import LinearRegression, LassoCV, MultiTaskLassoCV, Lasso, MultiTaskLasso
from sklearn.linear_model._base import _preprocess_data
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold
# TODO: consider working around relying on sklearn implementation details
from sklearn.model_selection._split import _CVIterableWrapper
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from statsmodels.tools.tools import add_constant
from statsmodels.api import RLM
import statsmodels
from joblib import Parallel, delayed
from typing import List


class _WeightedCVIterableWrapper(_CVIterableWrapper):
    def __init__(self, cv):
        super().__init__(cv)

    def get_n_splits(self, X=None, y=None, groups=None, sample_weight=None):
        if groups is not None and sample_weight is not None:
            raise ValueError("Cannot simultaneously use grouping and weighting")
        return super().get_n_splits(X, y, groups)

    def split(self, X=None, y=None, groups=None, sample_weight=None):
        if groups is not None and sample_weight is not None:
            raise ValueError("Cannot simultaneously use grouping and weighting")
        return super().split(X, y, groups)


def _weighted_check_cv(cv=5, y=None, classifier=False, random_state=None):
    # local import to avoid circular imports
    from .model_selection import WeightedKFold, WeightedStratifiedKFold
    cv = 5 if cv is None else cv
    if isinstance(cv, numbers.Integral):
        if (classifier and (y is not None) and
                (type_of_target(y) in ('binary', 'multiclass'))):
            return WeightedStratifiedKFold(cv, random_state=random_state)
        else:
            return WeightedKFold(cv, random_state=random_state)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             "or an iterable. Got %s." % cv)
        return _WeightedCVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


class WeightedModelMixin:
    """Mixin class for weighted models.

    For linear models, weights are applied as reweighting of the data matrix X and targets y.
    """

    def _fit_weighted_linear_model(self, X, y, sample_weight, check_input=None):
        # Convert X, y into numpy arrays
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        # Define fit parameters
        fit_params = {'X': X, 'y': y}
        # Some algorithms don't have a check_input option
        if check_input is not None:
            fit_params['check_input'] = check_input

        if sample_weight is not None:
            # Check weights array
            if np.atleast_1d(sample_weight).ndim > 1:
                # Check that weights are size-compatible
                raise ValueError("Sample weights must be 1D array or scalar")
            if np.ndim(sample_weight) == 0:
                sample_weight = np.repeat(sample_weight, X.shape[0])
            else:
                sample_weight = check_array(sample_weight, ensure_2d=False, allow_nd=False)
                if sample_weight.shape[0] != X.shape[0]:
                    raise ValueError(
                        "Found array with {0} sample(s) while {1} samples were expected.".format(
                            sample_weight.shape[0], X.shape[0])
                    )

            # Normalize inputs
            X, y, X_offset, y_offset, X_scale = _preprocess_data(
                X, y, fit_intercept=self.fit_intercept,
                copy=self.copy_X, check_input=check_input if check_input is not None else True,
                sample_weight=sample_weight)
            # Weight inputs
            normalized_weights = X.shape[0] * sample_weight / np.sum(sample_weight)
            sqrt_weights = np.sqrt(normalized_weights)
            X_weighted = sqrt_weights.reshape(-1, 1) * X
            y_weighted = sqrt_weights.reshape(-1, 1) * y if y.ndim > 1 else sqrt_weights * y
            fit_params['X'] = X_weighted
            fit_params['y'] = y_weighted
            if self.fit_intercept:
                # Fit base class without intercept
                self.fit_intercept = False
                # Fit Lasso
                super().fit(**fit_params)
                # Reset intercept
                self.fit_intercept = True
                # The intercept is not calculated properly due the sqrt(weights) factor
                # so it must be recomputed
                self._set_intercept(X_offset, y_offset, X_scale)
            else:
                super().fit(**fit_params)
        else:
            # Fit lasso without weights
            super().fit(**fit_params)


class WeightedLasso(WeightedModelMixin, Lasso):
    """Version of sklearn Lasso that accepts weights.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to ordinary least squares, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with Lasso is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    fit_intercept : bool, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    precompute : True | False | array_like, default False
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.

    copy_X : bool, default True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance, or None, default None
            The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random
        number generator; If None, the random number generator is the
        :class:`~numpy.random.mtrand.RandomState` instance used by :mod:`np.random<numpy.random>`. Used when
        ``selection='random'``.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : int | array_like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    """

    def __init__(self, alpha=1.0, fit_intercept=True,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super().__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.

        check_input : bool, default True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        """
        self._fit_weighted_linear_model(X, y, sample_weight, check_input)
        return self


class WeightedMultiTaskLasso(WeightedModelMixin, MultiTaskLasso):
    """Version of sklearn MultiTaskLasso that accepts weights.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    fit_intercept : bool, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_X : bool, default True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance, or None, default None
            The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random
        number generator; If None, the random number generator is the
        :class:`~numpy.random.mtrand.RandomState` instance used by :mod:`np.random<numpy.random>`. Used when
        ``selection='random'``.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : int | array_like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    """

    def __init__(self, alpha=1.0, fit_intercept=True,
                 copy_X=True, max_iter=1000, tol=1e-4, warm_start=False,
                 random_state=None, selection='cyclic'):
        super().__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            copy_X=copy_X, max_iter=max_iter, tol=tol, warm_start=warm_start,
            random_state=random_state, selection=selection)

    def fit(self, X, y, sample_weight=None):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.
        """
        self._fit_weighted_linear_model(X, y, sample_weight)
        return self


class WeightedLassoCV(WeightedModelMixin, LassoCV):
    """Version of sklearn LassoCV that accepts weights.

    .. testsetup::

        import numpy as np
        from sklearn.linear_model import lasso_path

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : bool, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    precompute : True | False | 'auto' | array_like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    copy_X : bool, default True
        If ``True``, X will be copied; else, it may be overwritten.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold weighted cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, :class:`WeightedKFold` is used.

        If None then 5 folds are used.

    verbose : bool or int
        Amount of verbosity.

    n_jobs : int or None, optional
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive : bool, optional
        If positive, restrict regression coefficients to be positive

    random_state : int, RandomState instance, or None, default None
            The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random
        number generator; If None, the random number generator is the
        :class:`~numpy.random.mtrand.RandomState` instance used by :mod:`np.random<numpy.random>`. Used when
        ``selection='random'``.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 precompute='auto', max_iter=1000, tol=1e-4,
                 copy_X=True, cv=None, verbose=False, n_jobs=None,
                 positive=False, random_state=None, selection='cyclic'):

        super().__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas,
            fit_intercept=fit_intercept,
            precompute=precompute, max_iter=max_iter, tol=tol, copy_X=copy_X,
            cv=cv, verbose=verbose, n_jobs=n_jobs, positive=positive,
            random_state=random_state, selection=selection)

    def fit(self, X, y, sample_weight=None):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.
        """
        # Make weighted splitter
        cv_temp = self.cv
        self.cv = _weighted_check_cv(self.cv, random_state=self.random_state).split(X, y, sample_weight=sample_weight)
        # Fit weighted model
        self._fit_weighted_linear_model(X, y, sample_weight)
        self.cv = cv_temp
        return self


class WeightedMultiTaskLassoCV(WeightedModelMixin, MultiTaskLassoCV):
    """Version of sklearn MultiTaskLassoCV that accepts weights.

    .. testsetup::

        import numpy as np
        from sklearn.linear_model import lasso_path

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : array_like, optional
        List of alphas where to compute the models.
        If not provided, set automatically.

    fit_intercept : bool
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    max_iter : int, optional
        The maximum number of iterations.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    copy_X : bool, default True
        If ``True``, X will be copied; else, it may be overwritten.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold weighted cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, :class:`WeightedKFold` is used.

        If None then 5-folds are used.

    verbose : bool or int
        Amount of verbosity.

    n_jobs : int or None, optional
        Number of CPUs to use during the cross validation. Note that this is
        used only if multiple values for l1_ratio are given.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, or None, default None
            The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random
        number generator; If None, the random number generator is the
        :class:`~numpy.random.mtrand.RandomState` instance used by :mod:`np.random<numpy.random>`. Used when
        ``selection='random'``.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 max_iter=1000, tol=1e-4,
                 copy_X=True, cv=None, verbose=False, n_jobs=None,
                 random_state=None, selection='cyclic'):

        super().__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas,
            fit_intercept=fit_intercept,
            max_iter=max_iter, tol=tol, copy_X=copy_X,
            cv=cv, verbose=verbose, n_jobs=n_jobs,
            random_state=random_state, selection=selection)

    def fit(self, X, y, sample_weight=None):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.
        """
        # Make weighted splitter
        cv_temp = self.cv
        self.cv = _weighted_check_cv(self.cv, random_state=self.random_state).split(X, y, sample_weight=sample_weight)
        # Fit weighted model
        self._fit_weighted_linear_model(X, y, sample_weight)
        self.cv = cv_temp
        return self


def _get_theta_coefs_and_tau_sq(i, X, sample_weight, alpha_cov, n_alphas_cov, max_iter, tol, random_state):
    n_samples, n_features = X.shape
    y = X[:, i]
    X_reduced = X[:, list(range(i)) + list(range(i + 1, n_features))]
    # Call weighted lasso on reduced design matrix
    if alpha_cov == 'auto':
        local_wlasso = WeightedLassoCV(cv=3, n_alphas=n_alphas_cov,
                                       fit_intercept=False,
                                       max_iter=max_iter,
                                       tol=tol, n_jobs=1,
                                       random_state=random_state)
    else:
        local_wlasso = WeightedLasso(alpha=alpha_cov,
                                     fit_intercept=False,
                                     max_iter=max_iter,
                                     tol=tol,
                                     random_state=random_state)
    local_wlasso.fit(X_reduced, y, sample_weight=sample_weight)
    coefs = local_wlasso.coef_
    # Weighted tau
    if sample_weight is not None:
        y_weighted = y * sample_weight / np.sum(sample_weight)
    else:
        y_weighted = y / n_samples
    tausq = np.dot(y - local_wlasso.predict(X_reduced), y_weighted)
    return coefs, tausq


class DebiasedLasso(WeightedLasso):
    """Debiased Lasso model.

    Implementation was derived from <https://arxiv.org/abs/1303.0518>.

    Only implemented for single-dimensional output.

    .. testsetup::

        import numpy as np
        from sklearn.linear_model import lasso_path

    Parameters
    ----------
    alpha : str | float, default 'auto'.
        Constant that multiplies the L1 term. Defaults to 'auto'.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`.LinearRegression` object.

    n_alphas : int, default 100
        How many alphas to try if alpha='auto'

    alpha_cov : str | float, default 'auto'
        The regularization alpha that is used when constructing the pseudo inverse of
        the covariance matrix Theta used to for correcting the lasso coefficient. Each
        such regression corresponds to the regression of one feature on the remainder
        of the features.

    n_alphas_cov : int, default 10
        How many alpha_cov to try if alpha_cov='auto'.

    fit_intercept : bool, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    precompute : True | False | array_like, default False
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.

    copy_X : bool, default True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance, or None, default None
            The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random
        number generator; If None, the random number generator is the
        :class:`~numpy.random.mtrand.RandomState` instance used by :mod:`np.random<numpy.random>`. Used when
        ``selection='random'``.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    n_jobs : int, optional
        How many jobs to use whenever parallelism is invoked

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        Parameter vector (w in the cost function formula).

    intercept_ : float
        Independent term in decision function.

    n_iter_ : int | array_like, shape (n_targets,)
        Number of iterations run by the coordinate descent solver to reach
        the specified tolerance.

    selected_alpha_ : float
        Penalty chosen through cross-validation, if alpha='auto'.

    coef_stderr_ : array, shape (n_features,)
        Estimated standard errors for coefficients (see ``coef_`` attribute).

    intercept_stderr_ : float
        Estimated standard error intercept (see ``intercept_`` attribute).

    """

    def __init__(self, alpha='auto', n_alphas=100, alpha_cov='auto', n_alphas_cov=10,
                 fit_intercept=True, precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False,
                 random_state=None, selection='cyclic', n_jobs=None):
        self.n_jobs = n_jobs
        self.n_alphas = n_alphas
        self.alpha_cov = alpha_cov
        self.n_alphas_cov = n_alphas_cov
        super().__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=False, random_state=random_state,
            selection=selection)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit debiased lasso model.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Input data.

        y : array, shape (n_samples,)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.

        check_input : bool, default True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        """
        self.selected_alpha_ = None
        if self.alpha == 'auto':
            # Select optimal penalty
            self.alpha = self._get_optimal_alpha(X, y, sample_weight)
            self.selected_alpha_ = self.alpha
        else:
            # Warn about consistency
            warnings.warn("Setting a suboptimal alpha can lead to miscalibrated confidence intervals. "
                          "We recommend setting alpha='auto' for optimality.")

        # Convert X, y into numpy arrays
        X, y = check_X_y(X, y, y_numeric=True, multi_output=False)
        # Fit weighted lasso with user input
        super().fit(X, y, sample_weight, check_input)
        # Center X, y
        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X, y, fit_intercept=self.fit_intercept,
            copy=self.copy_X, check_input=check_input, sample_weight=sample_weight)

        # Calculate quantities that will be used later on. Account for centered data
        y_pred = self.predict(X) - self.intercept_
        self._theta_hat = self._get_theta_hat(X, sample_weight)
        self._X_offset = X_offset

        # Calculate coefficient and error variance
        num_nonzero_coefs = np.count_nonzero(self.coef_)
        self._error_variance = np.average((y - y_pred)**2, weights=sample_weight) / \
            (1 - num_nonzero_coefs / X.shape[0])
        self._mean_error_variance = self._error_variance / X.shape[0]
        self._coef_variance = self._get_unscaled_coef_var(
            X, self._theta_hat, sample_weight) * self._error_variance

        # Add coefficient correction
        coef_correction = self._get_coef_correction(
            X, y, y_pred, sample_weight, self._theta_hat)
        self.coef_ += coef_correction

        # Set coefficients and intercept standard errors
        self.coef_stderr_ = np.sqrt(np.diag(self._coef_variance))
        if self.fit_intercept:
            self.intercept_stderr_ = np.sqrt(
                self._X_offset @ self._coef_variance @ self._X_offset +
                self._mean_error_variance
            )
        else:
            self.intercept_stderr_ = 0

        # Set intercept
        self._set_intercept(X_offset, y_offset, X_scale)
        # Return alpha to 'auto' state
        if self.selected_alpha_ is not None:
            self.alpha = 'auto'
        return self

    def prediction_stderr(self, X):
        """Get the standard error of the predictions using the debiased lasso.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Samples.

        Returns
        -------
        prediction_stderr : array_like, shape (n_samples, )
            The standard error of each coordinate of the output at each point we predict
        """
        # Note that in the case of no intercept, X_offset is 0
        if self.fit_intercept:
            X = X - self._X_offset
        # Calculate the variance of the predictions
        var_pred = np.sum(np.matmul(X, self._coef_variance) * X, axis=1)
        if self.fit_intercept:
            var_pred += self._mean_error_variance
        pred_stderr = np.sqrt(var_pred)
        return pred_stderr

    def predict_interval(self, X, alpha=0.05):
        """Build prediction confidence intervals using the debiased lasso.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Samples.

        alpha:  float in [0, 1], default 0.05
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        (y_lower, y_upper) : tuple of array, shape (n_samples, )
            Returns lower and upper interval endpoints.
        """
        lower = alpha / 2
        upper = 1 - alpha / 2
        y_pred = self.predict(X)
        # Calculate prediction confidence intervals
        sd_pred = self.prediction_stderr(X)
        y_lower = y_pred + \
            np.apply_along_axis(lambda s: norm.ppf(
                lower, scale=s), 0, sd_pred)
        y_upper = y_pred + \
            np.apply_along_axis(lambda s: norm.ppf(
                upper, scale=s), 0, sd_pred)
        return y_lower, y_upper

    def coef__interval(self, alpha=0.05):
        """Get a confidence interval bounding the fitted coefficients.

        Parameters
        ----------
        alpha : float, default 0.05
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        (coef_lower, coef_upper) : tuple of array, shape (n_coefs, )
            Returns lower and upper interval endpoints for the coefficients.
        """
        lower = alpha / 2
        upper = 1 - alpha / 2
        return self.coef_ + np.apply_along_axis(lambda s: norm.ppf(lower, scale=s), 0, self.coef_stderr_), \
            self.coef_ + np.apply_along_axis(lambda s: norm.ppf(upper, scale=s), 0, self.coef_stderr_)

    def intercept__interval(self, alpha=0.05):
        """Get a confidence interval bounding the fitted intercept.

        Parameters
        ----------
        alpha : float, default 0.05
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        (intercept_lower, intercept_upper) : tuple floats
            Returns lower and upper interval endpoints for the intercept.
        """
        lower = alpha / 2
        upper = 1 - alpha / 2
        if self.fit_intercept:
            return self.intercept_ + norm.ppf(lower, scale=self.intercept_stderr_), self.intercept_ + \
                norm.ppf(upper, scale=self.intercept_stderr_),
        else:
            return 0.0, 0.0

    def _get_coef_correction(self, X, y, y_pred, sample_weight, theta_hat):
        # Assumes flattened y
        n_samples, _ = X.shape
        y_res = np.ndarray.flatten(y) - y_pred
        # Compute weighted residuals
        if sample_weight is not None:
            y_res_scaled = y_res * sample_weight / np.sum(sample_weight)
        else:
            y_res_scaled = y_res / n_samples
        delta_coef = np.matmul(
            theta_hat, np.matmul(X.T, y_res_scaled))
        return delta_coef

    def _get_optimal_alpha(self, X, y, sample_weight):
        # To be done once per target. Assumes y can be flattened.
        cv_estimator = WeightedLassoCV(cv=5, n_alphas=self.n_alphas, fit_intercept=self.fit_intercept,
                                       precompute=self.precompute, copy_X=True,
                                       max_iter=self.max_iter, tol=self.tol,
                                       random_state=self.random_state,
                                       selection=self.selection,
                                       n_jobs=self.n_jobs)
        cv_estimator.fit(X, y.flatten(), sample_weight=sample_weight)
        return cv_estimator.alpha_

    def _get_theta_hat(self, X, sample_weight):
        # Assumes that X has already been offset
        n_samples, n_features = X.shape
        # Special case: n_features=1
        if n_features == 1:
            C_hat = np.ones((1, 1))
            tausq = (X.T @ X / n_samples).flatten()
            return np.diag(1 / tausq) @ C_hat
        # Compute Lasso coefficients for the columns of the design matrix
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_get_theta_coefs_and_tau_sq)(i, X, sample_weight,
                                                 self.alpha_cov, self.n_alphas_cov,
                                                 self.max_iter, self.tol, self.random_state)
            for i in range(n_features))
        coefs, tausq = zip(*results)
        coefs = np.array(coefs)
        tausq = np.array(tausq)
        # Compute C_hat
        C_hat = np.diag(np.ones(n_features))
        C_hat[0][1:] = -coefs[0]
        for i in range(1, n_features):
            C_hat[i][:i] = -coefs[i][:i]
            C_hat[i][i + 1:] = -coefs[i][i:]
        # Compute theta_hat
        theta_hat = np.diag(1 / tausq) @ C_hat
        return theta_hat

    def _get_unscaled_coef_var(self, X, theta_hat, sample_weight):
        if sample_weight is not None:
            norm_weights = sample_weight / np.sum(sample_weight)
            sigma = X.T @ (norm_weights.reshape(-1, 1) * X)
        else:
            sigma = np.matmul(X.T, X) / X.shape[0]
        _unscaled_coef_var = np.matmul(
            np.matmul(theta_hat, sigma), theta_hat.T) / X.shape[0]
        return _unscaled_coef_var


class MultiOutputDebiasedLasso(MultiOutputRegressor):
    """Debiased MultiOutputLasso model.

    Implementation was derived from <https://arxiv.org/abs/1303.0518>.
    Applies debiased lasso once per target. If only a flat target is passed in,
    it reverts to the DebiasedLasso algorithm.

    Parameters
    ----------
    alpha : str | float, optional. Default 'auto'.
        Constant that multiplies the L1 term. Defaults to 'auto'.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    n_alphas : int, default 100
        How many alphas to try if alpha='auto'

    alpha_cov : str | float, default 'auto'
        The regularization alpha that is used when constructing the pseudo inverse of
        the covariance matrix Theta used to for correcting the lasso coefficient. Each
        such regression corresponds to the regression of one feature on the remainder
        of the features.

    n_alphas_cov : int, default 10
        How many alpha_cov to try if alpha_cov='auto'.

    fit_intercept : bool, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    precompute : True | False | array_like, default False
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.

    copy_X : bool, default True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance, or None, default None
            The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random
        number generator; If None, the random number generator is the
        :class:`~numpy.random.mtrand.RandomState` instance used by :mod:`np.random<numpy.random>`. Used when
        ``selection='random'``.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    n_jobs : int, optional
        How many jobs to use whenever parallelism is invoked

    Attributes
    ----------
    coef_ : array, shape (n_targets, n_features) or (n_features,)
        Parameter vector (w in the cost function formula).

    intercept_ : array, shape (n_targets, ) or float
        Independent term in decision function.

    selected_alpha_ : array, shape (n_targets, ) or float
        Penalty chosen through cross-validation, if alpha='auto'.

    coef_stderr_ : array, shape (n_targets, n_features) or (n_features, )
        Estimated standard errors for coefficients (see ``coef_`` attribute).

    intercept_stderr_ : array, shape (n_targets, ) or float
        Estimated standard error intercept (see ``intercept_`` attribute).

    """

    def __init__(self, alpha='auto', n_alphas=100, alpha_cov='auto', n_alphas_cov=10,
                 fit_intercept=True,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False,
                 random_state=None, selection='cyclic', n_jobs=None):
        self.estimator = DebiasedLasso(alpha=alpha, n_alphas=n_alphas, alpha_cov=alpha_cov, n_alphas_cov=n_alphas_cov,
                                       fit_intercept=fit_intercept,
                                       precompute=precompute, copy_X=copy_X, max_iter=max_iter,
                                       tol=tol, warm_start=warm_start,
                                       random_state=random_state, selection=selection,
                                       n_jobs=n_jobs)
        super().__init__(estimator=self.estimator, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        """Fit the multi-output debiased lasso model.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Input data.

        y : array, shape (n_samples, n_targets) or (n_samples, )
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.
        """
        # Allow for single output as well
        # When only one output is passed in, the MultiOutputDebiasedLasso behaves like the DebiasedLasso
        self.flat_target = False
        if np.ndim(y) == 1:
            self.flat_target = True
            y = np.asarray(y).reshape(-1, 1)
        super().fit(X, y, sample_weight)
        # Set coef_ attribute
        self._set_attribute("coef_")
        # Set intercept_ attribute
        self._set_attribute("intercept_",
                            condition=self.estimators_[0].fit_intercept,
                            default=0.0)
        # Set selected_alpha_ attribute
        self._set_attribute("selected_alpha_",
                            condition=(self.estimators_[0].alpha == 'auto'))
        # Set coef_stderr_
        self._set_attribute("coef_stderr_")
        # intercept_stderr_
        self._set_attribute("intercept_stderr_")
        return self

    def predict(self, X):
        """Get the prediction using the debiased lasso.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Samples.

        Returns
        -------
        prediction : array_like, shape (n_samples, ) or (n_samples, n_targets)
            The prediction at each point.

        """
        pred = super().predict(X)
        if self.flat_target:
            pred = pred.flatten()
        return pred

    def prediction_stderr(self, X):
        """Get the standard error of the predictions using the debiased lasso.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Samples.

        Returns
        -------
        prediction_stderr : array_like, shape (n_samples, ) or (n_samples, n_targets)
            The standard error of each coordinate of the output at each point we predict
        """
        n_estimators = len(self.estimators_)
        X = check_array(X)
        pred_stderr = np.empty((X.shape[0], n_estimators))
        for i, estimator in enumerate(self.estimators_):
            pred_stderr[:, i] = estimator.prediction_stderr(X)
        if self.flat_target:
            pred_stderr = pred_stderr.flatten()
        return pred_stderr

    def predict_interval(self, X, alpha=0.05):
        """Build prediction confidence intervals using the debiased lasso.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Samples.

        alpha: float in [0, 1], default 0.05
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.

        Returns
        -------
        (y_lower, y_upper) : tuple of array, shape (n_samples, n_targets) or (n_samples, )
            Returns lower and upper interval endpoints.
        """
        n_estimators = len(self.estimators_)
        X = check_array(X)
        y_lower = np.empty((X.shape[0], n_estimators))
        y_upper = np.empty((X.shape[0], n_estimators))
        for i, estimator in enumerate(self.estimators_):
            y_lower[:, i], y_upper[:, i] = estimator.predict_interval(X, alpha=alpha)
        if self.flat_target:
            y_lower = y_lower.flatten()
            y_upper = y_upper.flatten()
        return y_lower, y_upper

    def coef__interval(self, alpha=0.05):
        """Get a confidence interval bounding the fitted coefficients.

        Parameters
        ----------
        alpha : float, default 0.05
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        (coef_lower, coef_upper) : tuple of array, shape (n_targets, n_coefs) or (n_coefs, )
            Returns lower and upper interval endpoints for the coefficients.
        """
        n_estimators = len(self.estimators_)
        coef_lower = np.empty((n_estimators, self.estimators_[0].coef_.shape[0]))
        coef_upper = np.empty((n_estimators, self.estimators_[0].coef_.shape[0]))
        for i, estimator in enumerate(self.estimators_):
            coef_lower[i], coef_upper[i] = estimator.coef__interval(alpha=alpha)
        if self.flat_target == 1:
            coef_lower = coef_lower.flatten()
            coef_upper = coef_upper.flatten()
        return coef_lower, coef_upper

    def intercept__interval(self, alpha=0.05):
        """Get a confidence interval bounding the fitted intercept.

        Parameters
        ----------
        alpha : float, default 0.05
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        (intercept_lower, intercept_upper) : tuple of array of size (n_targets, ) or tuple of floats
            Returns lower and upper interval endpoints for the intercept.
        """
        if len(self.estimators_) == 1:
            return self.estimators_[0].intercept__interval(alpha=alpha)
        else:
            intercepts = np.array([estimator.intercept__interval(alpha=alpha) for estimator in self.estimators_])
            return intercepts[:, 0], intercepts[:, 1]

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return self.estimator.get_params(deep=deep)

    def set_params(self, **params):
        """Set parameters for this estimator."""
        self.estimator.set_params(**params)

    def _set_attribute(self, attribute_name, condition=True, default=None):
        if condition:
            if not self.flat_target:
                attribute_value = np.array([getattr(estimator, attribute_name) for estimator in self.estimators_])
            else:
                attribute_value = getattr(self.estimators_[0], attribute_name)
        else:
            attribute_value = default
        setattr(self, attribute_name, attribute_value)


class _PairedEstimatorWrapper:
    """Helper class to wrap two different estimators, one of which can be used only with single targets and the other
    which can be used on multiple targets.  Not intended to be used directly by users."""

    _SingleEst = None
    _MultiEst = None
    _known_params = []
    _post_fit_attrs = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        # set model to the single-target estimator by default so there's always a model to get and set attributes on
        self.model = self._SingleEst(*args, **kwargs)

    def fit(self, X, y, sample_weight=None):
        self._needs_unravel = False
        params = {key: value
                  for (key, value) in self.get_params().items()
                  if key in self._known_params}
        if ndim(y) == 2 and shape(y)[1] > 1:
            self.model = self._MultiEst(**params)
        else:
            if ndim(y) == 2 and shape(y)[1] == 1:
                y = np.ravel(y)
                self._needs_unravel = True
            self.model = self._SingleEst(**params)
        self.model.fit(X, y, sample_weight)
        for param in self._post_fit_attrs:
            setattr(self, param, getattr(self.model, param))
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        return reshape(predictions, (-1, 1)) if self._needs_unravel else predictions

    def score(self, X, y, sample_weight=None):
        return self.model.score(X, y, sample_weight)

    def __getattr__(self, key):
        if key in self._known_params:
            return getattr(self.model, key)
        else:
            raise AttributeError("No attribute " + key)

    def __setattr__(self, key, value):
        if key in self._known_params:
            setattr(self.model, key, value)
        else:
            super().__setattr__(key, value)

    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {k: v for k, v in self.model.get_params(deep=deep).items() if k in self._known_params}

    def set_params(self, **params):
        """Set parameters for this estimator."""
        self.model.set_params(**params)


class WeightedLassoCVWrapper(_PairedEstimatorWrapper):
    """Helper class to wrap either WeightedLassoCV or WeightedMultiTaskLassoCV depending on the shape of the target."""

    _SingleEst = WeightedLassoCV
    _MultiEst = WeightedMultiTaskLassoCV

    # whitelist known params because full set is not necessarily identical between LassoCV and MultiTaskLassoCV
    # (e.g. former has 'positive' and 'precompute' while latter does not)
    _known_params = set(['eps', 'n_alphas', 'alphas', 'fit_intercept', 'normalize', 'max_iter', 'tol', 'copy_X',
                         'cv', 'verbose', 'n_jobs', 'random_state', 'selection'])

    _post_fit_attrs = set(['alpha_', 'alphas_', 'coef_', 'dual_gap_',
                           'intercept_', 'n_iter_', 'n_features_in_', 'mse_path_'])


class WeightedLassoWrapper(_PairedEstimatorWrapper):
    """Helper class to wrap either WeightedLasso or WeightedMultiTaskLasso depending on the shape of the target."""

    _SingleEst = WeightedLasso
    _MultiEst = WeightedMultiTaskLasso
    _known_params = set(['alpha', 'fit_intercept', 'copy_X', 'max_iter', 'tol',
                         'random_state', 'selection'])
    _post_fit_attrs = set(['coef_', 'dual_gap_', 'intercept_', 'n_iter_', 'n_features_in_'])


class SelectiveRegularization:
    """
    Estimator of a linear model where regularization is applied to only a subset of the coefficients.

    Assume that our loss is

    .. math::
        \\ell(\\beta_1, \\beta_2) = \\lVert y - X_1 \\beta_1 - X_2 \\beta_2 \\rVert^2 + f(\\beta_2)

    so that we're regularizing only the coefficients in :math:`\\beta_2`.

    Then, since :math:`\\beta_1` doesn't appear in the penalty, the problem of finding :math:`\\beta_1` to minimize the
    loss once :math:`\\beta_2` is known reduces to just a normal OLS regression, so that:

    .. math::
        \\beta_1 = (X_1^\\top X_1)^{-1}X_1^\\top(y - X_2 \\beta_2)

    Plugging this into the loss, we obtain

    .. math::

        ~& \\lVert y - X_1 (X_1^\\top X_1)^{-1}X_1^\\top(y - X_2 \\beta_2) - X_2 \\beta_2 \\rVert^2 + f(\\beta_2) \\\\
        =~& \\lVert (I - X_1 (X_1^\\top  X_1)^{-1}X_1^\\top)(y - X_2 \\beta_2) \\rVert^2 + f(\\beta_2)

    But, letting :math:`M_{X_1} = I - X_1 (X_1^\\top  X_1)^{-1}X_1^\\top`, we see that this is

    .. math::
        \\lVert (M_{X_1} y) - (M_{X_1} X_2) \\beta_2 \\rVert^2 + f(\\beta_2)

    so finding the minimizing :math:`\\beta_2` can be done by regressing :math:`M_{X_1} y` on :math:`M_{X_1} X_2` using
    the penalized regression method incorporating :math:`f`.  Note that these are just the residual values of :math:`y`
    and :math:`X_2` when regressed on :math:`X_1` using OLS.

    Parameters
    ----------
    unpenalized_inds : list of int, other 1-dimensional indexing expression, or callable
        The indices that should not be penalized when the model is fit; all other indices will be penalized.
        If this is a callable, it will be called with the arguments to `fit` and should return a corresponding
        indexing expression.  For example, ``lambda X, y: unpenalized_inds=slice(1,-1)`` will result in only the first
        and last indices being penalized.
    penalized_model : :term:`regressor`
        A penalized linear regression model
    fit_intercept : bool, default True
        Whether to fit an intercept; the intercept will not be penalized if it is fit

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.
    intercept_ : float or array of shape (n_targets)
        Independent term in the linear model.
    penalized_model : :term:`regressor`
        The penalized linear regression model, cloned from the one passed into the initializer
    """

    def __init__(self, unpenalized_inds, penalized_model, fit_intercept=True):
        self._unpenalized_inds_expr = unpenalized_inds
        self.penalized_model = clone(penalized_model)
        self._fit_intercept = fit_intercept

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model.

        Parameters
        ----------
        X : array_like, shape (n, d_x)
            The features to regress against
        y : array_like, shape (n,) or (n, d_y)
            The regression target
        sample_weight : array_like, shape (n,), optional
            Relative weights for each sample
        """
        X, y = check_X_y(X, y, multi_output=True, estimator=self)
        if callable(self._unpenalized_inds_expr):
            if sample_weight is None:
                self._unpenalized_inds = self._unpenalized_inds_expr(X, y)
            else:
                self._unpenalized_inds = self._unpenalized_inds_expr(X, y, sample_weight=sample_weight)
        else:
            self._unpenalized_inds = self._unpenalized_inds_expr
        mask = np.ones(X.shape[1], dtype=bool)
        mask[self._unpenalized_inds] = False
        self._penalized_inds = np.arange(X.shape[1])[mask]
        X1 = X[:, self._unpenalized_inds]
        X2 = X[:, self._penalized_inds]

        X2_res = X2 - LinearRegression(fit_intercept=self._fit_intercept).fit(X1, X2,
                                                                              sample_weight=sample_weight).predict(X1)
        y_res = y - LinearRegression(fit_intercept=self._fit_intercept).fit(X1, y,
                                                                            sample_weight=sample_weight).predict(X1)

        if sample_weight is not None:
            self.penalized_model.fit(X2_res, y_res, sample_weight=sample_weight)
        else:
            self.penalized_model.fit(X2_res, y_res)

        # The unpenalized model can't contain an intercept, because in the analysis above
        # we rely on the fact that M(X beta) = (M X) beta, but M(X beta + c) is not the same
        # as (M X) beta + c, so the learned coef and intercept will be wrong
        intercept = self.penalized_model.predict(np.zeros_like(X2[0:1]))
        if not np.allclose(intercept, 0):
            raise AttributeError("The penalized model has a non-zero intercept; to fit an intercept "
                                 "you should instead either set fit_intercept to True when initializing the "
                                 "SelectiveRegression instance (for an unpenalized intercept) or "
                                 "explicitly add a column of ones to the data being fit and include that "
                                 "column in the penalized indices.")

        # now regress X1 on y - X2 * beta2 to learn beta1
        self._model_X1 = LinearRegression(fit_intercept=self._fit_intercept)
        self._model_X1.fit(X1, y - self.penalized_model.predict(X2), sample_weight=sample_weight)

        # set coef_ and intercept_ attributes
        self.coef_ = np.empty(shape(y)[1:] + shape(X)[1:])
        self.coef_[..., self._penalized_inds] = self.penalized_model.coef_
        self.coef_[..., self._unpenalized_inds] = self._model_X1.coef_

        # Note that the penalized model should *not* have an intercept
        self.intercept_ = self._model_X1.intercept_

        return self

    def predict(self, X):
        """
        Make a prediction for each sample.

        Parameters
        ----------
        X : array_like, shape (m, d_x)
            The samples whose targets to predict

        Output
        ------
        arr : array_like, shape (m,) or (m, d_y)
            The predicted targets
        """
        check_is_fitted(self, "coef_")
        X1 = X[:, self._unpenalized_inds]
        X2 = X[:, self._penalized_inds]
        return self._model_X1.predict(X1) + self.penalized_model.predict(X2)

    def score(self, X, y):
        """
        Score the predictions for a set of features to ground truth.

        Parameters
        ----------
        X : array_like, shape (m, d_x)
            The samples to predict
        y : array_like, shape (m,) or (m, d_y)
            The ground truth targets

        Output
        ------
        score : float
            The model's score
        """
        check_is_fitted(self, "coef_")
        X, y = check_X_y(X, y, multi_output=True, estimator=self)
        return r2_score(y, self.predict(X))

    known_params = {'known_params', 'coef_', 'intercept_', 'penalized_model',
                    '_unpenalized_inds_expr', '_fit_intercept', '_unpenalized_inds', '_penalized_inds', '_model_X1'}

    def __getattr__(self, key):
        # don't proxy special methods
        if key.startswith('__'):
            raise AttributeError(key)

        # don't pass get_params through to model, because that will cause sklearn to clone this
        # regressor incorrectly
        if key != "get_params" and key not in self.known_params:
            return getattr(self.penalized_model, key)
        else:
            # Note: for known attributes that have been set this method will not be called,
            # so we should just throw here because this is an attribute belonging to this class
            # but which hasn't yet been set on this instance
            raise AttributeError("No attribute " + key)

    def __setattr__(self, key, value):
        if key not in self.known_params:
            setattr(self.penalized_model, key, value)
        else:
            super().__setattr__(key, value)


class _StatsModelsWrapper(BaseEstimator):
    """ Parent class for statsmodels linear models. At init time each children class should set the
    boolean flag property fit_intercept. At fit time, each children class must calculate and set the
    following properties:

    _param: (m,) or (m, p) array
        Where m is number of features and p is number of outcomes, which corresponds to the
        coefficients of the linear model (including the intercept in the first column if fit_intercept=True).
    _param_var: (m, m) or (p, m, m) array
        Where m is number of features and p is number of outcomes, where each (m, m) matrix corresponds
        to the scaled covariance matrix of the parameters of the linear model.
    _n_out: the second dimension of the training y, or 0 if y is a vector
    """

    def predict(self, X):
        """
        Predicts the output given an array of instances.

        Parameters
        ----------
        X : (n, d) array_like
            The covariates on which to predict

        Returns
        -------
        predictions : {(n,) array, (n,p) array}
            The predicted mean outcomes
        """
        if X is None:
            X = np.empty((1, 0))
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        return np.matmul(X, self._param)

    @property
    def coef_(self):
        """
        Get the model's coefficients on the covariates.

        Returns
        -------
        coef_ : {(d,), (p, d)} nd array_like
            The coefficients of the variables in the linear regression. If label y
            was p-dimensional, then the result is a matrix of coefficents, whose p-th
            row containts the coefficients corresponding to the p-th coordinate of the label.
        """
        if self.fit_intercept:
            if self._n_out == 0:
                return self._param[1:]
            else:
                return self._param[1:].T
        else:
            if self._n_out == 0:
                return self._param
            else:
                return self._param.T

    @property
    def intercept_(self):
        """
        Get the intercept(s) (or 0 if no intercept was fit).

        Returns
        -------
        intercept_ : float or (p,) nd array_like
            The intercept of the linear regresion. If label y was p-dimensional, then the result is a vector
            whose p-th entry containts the intercept corresponding to the p-th coordinate of the label.
        """
        return self._param[0] if self.fit_intercept else (0 if self._n_out == 0 else np.zeros(self._n_out))

    @property
    def _param_stderr(self):
        """
        The standard error of each parameter that was estimated.

        Returns
        -------
        _param_stderr : {(d (+1),) (d (+1), p)} nd array_like
            The standard error of each parameter that was estimated.
        """
        if self._n_out == 0:
            return np.sqrt(np.clip(np.diag(self._param_var), 0, np.inf))
        else:
            return np.array([np.sqrt(np.clip(np.diag(v), 0, np.inf)) for v in self._param_var]).T

    @property
    def coef_stderr_(self):
        """
        Gets the standard error of the fitted coefficients.

        Returns
        -------
        coef_stderr_ : {(d,), (p, d)} nd array_like
            The standard error of the coefficients
        """
        return self._param_stderr[1:].T if self.fit_intercept else self._param_stderr.T

    @property
    def intercept_stderr_(self):
        """
        Gets the standard error of the intercept(s) (or 0 if no intercept was fit).

        Returns
        -------
        intercept_stderr_ : float or (p,) nd array_like
            The standard error of the intercept(s)
        """
        return self._param_stderr[0] if self.fit_intercept else (0 if self._n_out == 0 else np.zeros(self._n_out))

    def prediction_stderr(self, X):
        """
        Gets the standard error of the predictions.

        Parameters
        ----------
        X : (n, d) array_like
            The covariates at which to predict

        Returns
        -------
        prediction_stderr : (n, p) array_like
            The standard error of each coordinate of the output at each point we predict
        """
        if X is None:
            X = np.empty((1, 0))
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')
        if self._n_out == 0:
            return np.sqrt(np.clip(np.sum(np.matmul(X, self._param_var) * X, axis=1), 0, np.inf))
        else:
            return np.array([np.sqrt(np.clip(np.sum(np.matmul(X, v) * X, axis=1), 0, np.inf))
                             for v in self._param_var]).T

    def coef__interval(self, alpha=0.05):
        """
        Gets a confidence interval bounding the fitted coefficients.

        Parameters
        ----------
        alpha : float, default 0.05
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        coef__interval : {tuple ((p, d) array, (p,d) array), tuple ((d,) array, (d,) array)}
            The lower and upper bounds of the confidence interval of the coefficients
        """
        return (_safe_norm_ppf(alpha / 2, loc=self.coef_, scale=self.coef_stderr_),
                _safe_norm_ppf(1 - alpha / 2, loc=self.coef_, scale=self.coef_stderr_))

    def intercept__interval(self, alpha=0.05):
        """
        Gets a confidence interval bounding the intercept(s) (or 0 if no intercept was fit).

        Parameters
        ----------
        alpha : float, default 0.05
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        intercept__interval : {tuple ((p,) array, (p,) array), tuple (float, float)}
            The lower and upper bounds of the confidence interval of the intercept(s)
        """
        if not self.fit_intercept:
            return (0 if self._n_out == 0 else np.zeros(self._n_out)), \
                (0 if self._n_out == 0 else np.zeros(self._n_out))

        return (_safe_norm_ppf(alpha / 2, loc=self.intercept_, scale=self.intercept_stderr_),
                _safe_norm_ppf(1 - alpha / 2, loc=self.intercept_, scale=self.intercept_stderr_))

    def predict_interval(self, X, alpha=0.05):
        """
        Gets a confidence interval bounding the prediction.

        Parameters
        ----------
        X : (n, d) array_like
            The covariates on which to predict
        alpha : float, default 0.05
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        prediction_intervals : {tuple ((n,) array, (n,) array), tuple ((n,p) array, (n,p) array)}
            The lower and upper bounds of the confidence intervals of the predicted mean outcomes
        """

        pred = self.predict(X)
        pred_stderr = self.prediction_stderr(X)

        return (_safe_norm_ppf(alpha / 2, loc=pred, scale=pred_stderr),
                _safe_norm_ppf(1 - alpha / 2, loc=pred, scale=pred_stderr))


class StatsModelsLinearRegression(_StatsModelsWrapper):
    """
    Class which mimics weighted linear regression from the statsmodels package.

    However, unlike statsmodels WLS, this class also supports sample variances in addition to sample weights,
    which enables more accurate inference when working with summarized data.

    Parameters
    ----------
    fit_intercept : bool, default True
        Whether to fit an intercept in this model
    cov_type : string, default "HC0"
        The covariance approach to use.  Supported values are "HCO", "HC1", and "nonrobust".
    enable_federation : bool, default False
        Whether to enable federation (aggregating this model's results with other models in a distributed setting).
        This requires additional memory proportional to the number of columns in X to the fourth power.
    """

    def __init__(self, fit_intercept=True, cov_type="HC0", *, enable_federation=False):
        self.cov_type = cov_type
        self.fit_intercept = fit_intercept
        self.enable_federation = enable_federation

    def _check_input(self, X, y, sample_weight, freq_weight, sample_var):
        """Check dimensions and other assertions."""

        X, y, sample_weight, freq_weight, sample_var = check_input_arrays(
            X, y, sample_weight, freq_weight, sample_var, dtype='numeric')
        if X is None:
            X = np.empty((y.shape[0], 0))
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')

        # set default values for None
        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])
        if freq_weight is None:
            freq_weight = np.ones(y.shape[0])
        if sample_var is None:
            sample_var = np.zeros(y.shape)

        # check freq_weight should be integer and should be accompanied by sample_var
        if np.any(np.not_equal(np.mod(freq_weight, 1), 0)):
            raise AttributeError("Frequency weights must all be integers for inference to be valid!")
        if sample_var.ndim < 2:
            if np.any(np.equal(freq_weight, 1) & np.not_equal(sample_var, 0)):
                warnings.warn(
                    "Variance was set to non-zero for an observation with freq_weight=1! "
                    "sample_var represents the variance of the original observations that are "
                    "summarized in this sample. Hence, cannot have a non-zero variance if only "
                    "one observations was summarized. Inference will be invalid!")
            elif np.any(np.not_equal(freq_weight, 1) & np.equal(sample_var, 0)):
                warnings.warn(
                    "Variance was set to zero for an observation with freq_weight>1! "
                    "sample_var represents the variance of the original observations that are "
                    "summarized in this sample. If it's zero, please use sample_wegiht instead "
                    "to reflect the weight for each individual sample!")
        else:
            if np.any(np.equal(freq_weight, 1) & np.not_equal(np.sum(sample_var, axis=1), 0)):
                warnings.warn(
                    "Variance was set to non-zero for an observation with freq_weight=1! "
                    "sample_var represents the variance of the original observations that are "
                    "summarized in this sample. Hence, cannot have a non-zero variance if only "
                    "one observations was summarized. Inference will be invalid!")
            elif np.any(np.not_equal(freq_weight, 1) & np.equal(np.sum(sample_var, axis=1), 0)):
                warnings.warn(
                    "Variance was set to zero for an observation with freq_weight>1! "
                    "sample_var represents the variance of the original observations that are "
                    "summarized in this sample. If it's zero, please use sample_wegiht instead "
                    "to reflect the weight for each individual sample!")

        # check array shape
        assert (X.shape[0] == y.shape[0] == sample_weight.shape[0] ==
                freq_weight.shape[0] == sample_var.shape[0]), "Input lengths not compatible!"
        if y.ndim >= 2:
            assert (y.ndim == sample_var.ndim and
                    y.shape[1] == sample_var.shape[1]), "Input shapes not compatible: {}, {}!".format(
                y.shape, sample_var.shape)

        # weight X and y and sample_var
        weighted_X = X * np.sqrt(sample_weight).reshape(-1, 1)
        if y.ndim < 2:
            weighted_y = y * np.sqrt(sample_weight)
            sample_var = sample_var * sample_weight
        else:
            weighted_y = y * np.sqrt(sample_weight).reshape(-1, 1)
            sample_var = sample_var * (sample_weight.reshape(-1, 1))
        return weighted_X, weighted_y, freq_weight, sample_var

    def fit(self, X, y, sample_weight=None, freq_weight=None, sample_var=None):
        """
        Fits the model.

        Parameters
        ----------
        X : (N, d) nd array_like
            co-variates
        y : {(N,), (N, p)} nd array_like
            output variable(s)
        sample_weight : (N,) array_like or None
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (N, ) array_like of int or None
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(N,), (N, p)} nd array_like or None
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.

        Returns
        -------
        self : StatsModelsLinearRegression
        """
        # TODO: Add other types of covariance estimation (e.g. Newey-West (HAC), HC2, HC3)
        X, y, freq_weight, sample_var = self._check_input(X, y, sample_weight, freq_weight, sample_var)

        WX = X * np.sqrt(freq_weight).reshape(-1, 1)

        if y.ndim < 2:
            self._n_out = 0
            wy = y * np.sqrt(freq_weight)
        else:
            self._n_out = y.shape[1]
            wy = y * np.sqrt(freq_weight).reshape(-1, 1)

        param, _, rank, _ = np.linalg.lstsq(WX, wy, rcond=None)
        n_obs = np.sum(freq_weight)
        self._n_obs = n_obs

        df = param.shape[0]

        if rank < df:
            warnings.warn("Co-variance matrix is underdetermined. Inference will be invalid!")

        if n_obs <= df:
            warnings.warn("Number of observations <= than number of parameters. Using biased variance calculation!")
            correction = 1
        else:
            correction = (n_obs / (n_obs - df))

        # For aggregation calculations, always treat wy as an array so that einsum expressions don't need to change
        # We'll collapse results back down afterwards if necessary
        wy = wy.reshape(-1, 1) if y.ndim < 2 else wy
        sv = sample_var.reshape(-1, 1) if y.ndim < 2 else sample_var
        self.XX = np.matmul(WX.T, WX)
        self.Xy = np.matmul(WX.T, wy)

        if self.enable_federation:
            # for federation, we need to store these 5 arrays when using heteroskedasticity-robust inference
            if (self.cov_type in ['HC0', 'HC1']):
                # y dimension is always first in the output when present so that broadcasting works correctly
                self.XXyy = np.einsum('nw,nx,ny,ny->ywx', X, X, wy, wy)
                self.XXXy = np.einsum('nv,nw,nx,ny->yvwx', X, X, WX, wy)
                self.XXXX = np.einsum('nu,nv,nw,nx->uvwx', X, X, WX, WX)
                self.sample_var = np.einsum('nw,nx,ny->ywx', WX, WX, sv)
            elif (self.cov_type is None) or (self.cov_type == 'nonrobust'):
                self.XXyy = np.einsum('ny,ny->y', wy, wy)
                self.XXXy = np.einsum('nx,ny->yx', WX, wy)
                self.XXXX = np.einsum('nw,nx->wx', WX, WX)
                self.sample_var = np.average(sv, weights=freq_weight, axis=0) * n_obs

        sigma_inv = np.linalg.pinv(self.XX)

        var_i = sample_var + (y - np.matmul(X, param))**2

        self._param = param

        if (self.cov_type is None) or (self.cov_type == 'nonrobust'):
            if y.ndim < 2:
                self._var = correction * np.average(var_i, weights=freq_weight) * sigma_inv
            else:
                vars = correction * np.average(var_i, weights=freq_weight, axis=0)
                self._var = [v * sigma_inv for v in vars]
        elif (self.cov_type == 'HC0'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(WX.T, WX * var_i.reshape(-1, 1))
                self._var = np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(WX.T, WX * var_i[:, [j]])
                    self._var.append(np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv)))
        elif (self.cov_type == 'HC1'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(WX.T, WX * var_i.reshape(-1, 1))
                self._var = correction * np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(WX.T, WX * var_i[:, [j]])
                    self._var.append(correction * np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv)))
        else:
            raise AttributeError("Unsupported cov_type. Must be one of nonrobust, HC0, HC1.")

        self._param_var = np.array(self._var)

        return self

    @staticmethod
    def aggregate(models: List[StatsModelsLinearRegression]):
        """
        Aggregate multiple models into one.

        Parameters
        ----------
        models : list of StatsModelsLinearRegression
            The models to aggregate

        Returns
        -------
        agg_model : StatsModelsLinearRegression
            The aggregated model
        """
        if len(models) == 0:
            raise ValueError("Must aggregate at least one model!")
        cov_types = set([model.cov_type for model in models])
        fit_intercepts = set([model.fit_intercept for model in models])
        enable_federation = set([model.enable_federation for model in models])
        _n_outs = set([model._n_out for model in models])
        assert enable_federation == {True}, "All models must have enable_federation=True!"
        assert len(cov_types) == 1, "All models must have the same cov_type!"
        assert len(fit_intercepts) == 1, "All models must have the same fit_intercept!"
        assert len(_n_outs) == 1, "All models must have the same number of outcomes!"
        agg_model = StatsModelsLinearRegression(cov_type=models[0].cov_type, fit_intercept=models[0].fit_intercept)

        agg_model._n_out = models[0]._n_out

        XX = np.sum([model.XX for model in models], axis=0)
        Xy = np.sum([model.Xy for model in models], axis=0)
        XXyy = np.sum([model.XXyy for model in models], axis=0)
        XXXy = np.sum([model.XXXy for model in models], axis=0)
        XXXX = np.sum([model.XXXX for model in models], axis=0)

        sample_var = np.sum([model.sample_var for model in models], axis=0)
        n_obs = np.sum([model._n_obs for model in models], axis=0)

        sigma_inv = np.linalg.pinv(XX)
        param = sigma_inv @ Xy
        df = np.shape(param)[0]

        agg_model._param = param if agg_model._n_out > 0 else param.squeeze(1)

        if n_obs <= df:
            warnings.warn("Number of observations <= than number of parameters. Using biased variance calculation!")
            correction = 1
        elif agg_model.cov_type == 'HC0':
            correction = 1
        else:  # both HC1 and nonrobust use the same correction factor
            correction = (n_obs / (n_obs - df))

        if agg_model.cov_type in ['HC0', 'HC1']:
            weighted_sigma = XXyy - 2 * np.einsum('yvwx,vy->ywx', XXXy, param) + \
                np.einsum('uvwx,uy,vy->ywx', XXXX, param, param) + sample_var
            if agg_model._n_out == 0:
                agg_model._var = correction * np.matmul(sigma_inv, np.matmul(weighted_sigma.squeeze(0), sigma_inv))
            else:
                agg_model._var = [correction * np.matmul(sigma_inv, np.matmul(ws, sigma_inv)) for ws in weighted_sigma]

        else:
            assert agg_model.cov_type == 'nonrobust' or agg_model.cov_type is None
            sigma = XXyy - 2 * np.einsum('yx,xy->y', XXXy, param) + np.einsum('wx,wy,xy->y', XXXX, param, param)
            var_i = (sample_var + sigma) / n_obs
            if agg_model._n_out == 0:
                agg_model._var = correction * var_i * sigma_inv
            else:
                agg_model._var = [correction * var * sigma_inv for var in var_i]

        agg_model._param_var = np.array(agg_model._var)

        (agg_model.XX, agg_model.Xy, agg_model.XXyy, agg_model.XXXy, agg_model.XXXX,
         agg_model.sample_var, agg_model._n_obs) = XX, Xy, XXyy, XXXy, XXXX, sample_var, n_obs

        return agg_model


class StatsModelsRLM(_StatsModelsWrapper):
    """
    Class which mimics robust linear regression from the statsmodels package.

    Parameters
    ----------
    t : float, default 1.345
        The tuning constant for Huber’s t function
    maxiter : int, default 50
        The maximum number of iterations to try
    tol : float, default 1e-08
        The convergence tolerance of the estimate
    fit_intercept : bool, default True
        Whether to fit an intercept in this model
    cov_type : {'H1', 'H2', or 'H3'}, default 'H1'
        Indicates how the covariance matrix is estimated. See statsmodels.robust.robust_linear_model.RLMResults
        for more information.
    """

    def __init__(self, t=1.345,
                 maxiter=50,
                 tol=1e-08,
                 fit_intercept=True,
                 cov_type='H1'):
        self.t = t
        self.maxiter = maxiter
        self.tol = tol
        self.cov_type = cov_type
        self.fit_intercept = fit_intercept
        return

    def _check_input(self, X, y):
        """Check dimensions and other assertions."""
        if X is None:
            X = np.empty((y.shape[0], 0))

        assert (X.shape[0] == y.shape[0]), "Input lengths not compatible!"

        return X, y

    def fit(self, X, y):
        """
        Fits the model.

        Parameters
        ----------
        X : (N, d) nd array_like
            co-variates
        y : (N,) nd array_like or (N, p) array_like
            output variable

        Returns
        -------
        self : StatsModelsRLM
        """
        X, y = self._check_input(X, y)
        if self.fit_intercept:
            X = add_constant(X, has_constant='add')

        self._n_out = 0 if len(y.shape) == 1 else (y.shape[1],)

        def model_gen(y):
            return RLM(endog=y,
                       exog=X,
                       M=statsmodels.robust.norms.HuberT(t=self.t)).fit(cov=self.cov_type,
                                                                        maxiter=self.maxiter,
                                                                        tol=self.tol)
        if y.ndim < 2:
            self.model = model_gen(y)
            self._param = self.model.params
            self._param_var = self.model.cov_params()
        else:
            self.models = [model_gen(y[:, i]) for i in range(y.shape[1])]
            self._param = np.array([mdl.params for mdl in self.models]).T
            self._param_var = np.array([mdl.cov_params() for mdl in self.models])

        return self


class StatsModels2SLS(_StatsModelsWrapper):
    """
    Class that solves the moment equation E[(y-theta*T)*Z]=0

    Parameters
    ----------
    cov_type : {'HC0', 'HC1', 'nonrobust', or None}, default 'HC0'
        Indicates how the covariance matrix is estimated.
    """

    def __init__(self, cov_type="HC0"):
        self.fit_intercept = False
        self.cov_type = cov_type
        return

    def _check_input(self, Z, T, y, sample_weight):
        """Check dimensions and other assertions."""

        # set default values for None
        if sample_weight is None:
            sample_weight = np.ones(y.shape[0])

        # check array shape
        assert (T.shape[0] == Z.shape[0] == y.shape[0] == sample_weight.shape[0]), "Input lengths not compatible!"

        # check dimension of instruments is more than dimension of treatments
        if Z.shape[1] < T.shape[1]:
            raise AssertionError("The number of treatments couldn't be larger than the number of instruments!" +
                                 " If you are using a treatment featurizer, make sure the number of featurized" +
                                 " treatments is less than or equal to the number of instruments. You can either" +
                                 " featurize the instrument yourself, or consider using projection = True" +
                                 " along with a flexible model_t_xwz.")

        # weight X and y
        weighted_Z = Z * np.sqrt(sample_weight).reshape(-1, 1)
        weighted_T = T * np.sqrt(sample_weight).reshape(-1, 1)
        if y.ndim < 2:
            weighted_y = y * np.sqrt(sample_weight)
        else:
            weighted_y = y * np.sqrt(sample_weight).reshape(-1, 1)
        return weighted_Z, weighted_T, weighted_y

    def fit(self, Z, T, y, sample_weight=None, freq_weight=None, sample_var=None):
        """
        Fits the model.

        Parameters
        ----------
        Z :  {(N, p)} nd array_like
            instrumental variables
        T :  {(N, p)} nd array_like
            treatment variables
        y :  {(N,), (N, p)} nd array_like
            output variables
        sample_weight : (N,) array_like or None
            Individual weights for each sample. If None, it assumes equal weight.
        freq_weight: (N, ) array_like of int or None
            Weight for the observation. Observation i is treated as the mean
            outcome of freq_weight[i] independent observations.
            When ``sample_var`` is not None, this should be provided.
        sample_var : {(N,), (N, p)} nd array_like or None
            Variance of the outcome(s) of the original freq_weight[i] observations that were used to
            compute the mean outcome represented by observation i.


        Returns
        -------
        self : StatsModels2SLS
        """
        assert freq_weight is None, "freq_weight is not supported yet for this class!"
        assert sample_var is None, "sample_var is not supported yet for this class!"

        Z, T, y = self._check_input(Z, T, y, sample_weight)

        self._n_out = 0 if y.ndim < 2 else y.shape[1]

        # learn point estimate
        # solve first stage linear regression E[T|Z]
        zT_z = np.dot(Z.T, Z)
        zT_t = np.dot(Z.T, T)
        # "that" means T̂
        self._thatparams = np.linalg.solve(zT_z, zT_t)
        that = np.dot(Z, self._thatparams)
        # solve second stage linear regression E[Y|that]
        # (T̂.T*T̂)^{-1}
        thatT_that = np.dot(that.T, that)
        thatT_y = np.dot(that.T, y)
        param = np.linalg.solve(thatT_that, thatT_y)
        self._param = param

        n_obs = y.shape[0]
        df = len(param) if self._n_out == 0 else param.shape[0]

        if n_obs <= df:
            warnings.warn("Number of observations <= than number of parameters. Using biased variance calculation!")
            correction = 1
        else:
            correction = (n_obs / (n_obs - df))

        # learn cov(theta)
        # (T̂.T*T̂)^{-1}
        thatT_that_inv = np.linalg.inv(thatT_that)
        # sigma^2
        var_i = (y - np.dot(T, param))**2

        # reference: http://www.hec.unil.ch/documents/seminars/deep/361.pdf
        if (self.cov_type is None) or (self.cov_type == 'nonrobust'):
            if y.ndim < 2:
                self._var = correction * np.average(var_i) * thatT_that_inv
            else:
                sigma2 = correction * np.average(var_i, axis=0)
                self._var = [s * thatT_that_inv for s in sigma2]
        elif (self.cov_type == 'HC0'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(that.T, that * var_i.reshape(-1, 1))
                self._var = np.matmul(thatT_that_inv, np.matmul(weighted_sigma, thatT_that_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(that.T, that * var_i[:, [j]])
                    self._var.append(np.matmul(thatT_that_inv, np.matmul(weighted_sigma, thatT_that_inv)))
        elif (self.cov_type == 'HC1'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(that.T, that * var_i.reshape(-1, 1))
                self._var = correction * np.matmul(thatT_that_inv, np.matmul(weighted_sigma, thatT_that_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(that.T, that * var_i[:, [j]])
                    self._var.append(correction * np.matmul(thatT_that_inv,
                                                            np.matmul(weighted_sigma, thatT_that_inv)))
        else:
            raise AttributeError("Unsupported cov_type. Must be one of nonrobust, HC0, HC1.")

        self._param_var = np.array(self._var)
        return self
