# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import abc
import numpy as np
from warnings import warn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from scipy.stats import norm
from sklearn.linear_model import (ElasticNetCV, LassoCV, LogisticRegressionCV)
from ...sklearn_extensions.linear_model import (StatsModelsLinearRegression, WeightedLassoCVWrapper)
from ...sklearn_extensions.model_selection import ModelSelector, WeightedStratifiedKFold
from ...dml.dml import _make_first_stage_selector, _FinalWrapper
from ..._cate_estimator import TreatmentExpansionMixin, LinearModelFinalCateEstimatorMixin
from ..._ortho_learner import _OrthoLearner
from ...utilities import (_deprecate_positional, add_intercept,
                          broadcast_unit_treatments, check_high_dimensional,
                          cross_product, deprecated,
                          hstack, inverse_onehot, ndim, reshape,
                          reshape_treatmentwise_effects, shape, transpose,
                          get_feature_names_or_default, check_input_arrays,
                          filter_none_kwargs)


def _get_groups_period_filter(groups, n_periods):
    group_counts = {}
    group_period_filter = {i: [] for i in range(n_periods)}
    for i, g in enumerate(groups):
        if g not in group_counts:
            group_counts[g] = 0
        group_period_filter[group_counts[g]].append(i)
        group_counts[g] += 1
    return group_period_filter


class _DynamicModelNuisanceSelector(ModelSelector):
    """
    Nuisance model fits the model_y and model_t at fit time and at predict time
    calculates the residual Y and residual T based on the fitted models and returns
    the residuals as two nuisance parameters.
    """

    def __init__(self, model_y, model_t, n_periods):
        self._model_y = model_y
        self._model_t = model_t
        self.n_periods = n_periods

    def train(self, is_selecting, folds, Y, T, X=None, W=None, sample_weight=None, groups=None):
        """Fit a series of nuisance models for each period or period pairs."""
        assert Y.shape[0] % self.n_periods == 0, \
            "Length of training data should be an integer multiple of time periods."
        period_filters = _get_groups_period_filter(groups, self.n_periods)
        if not hasattr(self, '_model_y_trained'):  # create the per-period y and t models
            self._model_y_trained = {t: clone(self._model_y, safe=False)
                                     for t in np.arange(self.n_periods)}
            self._model_t_trained = {j: {t: clone(self._model_t, safe=False)
                                         for t in np.arange(j + 1)}
                                     for j in np.arange(self.n_periods)}

        # we have to filter the folds because they contain the indices in the original data not
        # the indices in the period-filtered data

        def _translate_inds(t, inds):
            # translate the indices in a fold to the indices in the period-filtered data
            # if groups was [3,3,4,4,5,5,6,6,1,1,2,2,0,0] (the group ids can be in any order, but the
            # time periods for each group should be contguous), and we had [10,11,0,1] as the indices in a fold
            # (so the fold is taking the entries corresponding to groups 2 and 3)
            # then group_period_filter(0) is [0,2,4,6,8,10,12] and gpf(1) is [1,3,5,7,9,11,13]
            # so for period 1, the fold should be [10,0] => [5,0] (the indices that return 10 and 0 in the t=0 data)
            # and for period 2, the fold should be [11,1] => [5,0] again (the indices that return 11,1 in the t=1 data)

            # filter to the indices for the time period
            inds = inds[np.isin(inds, period_filters[t])]

            # now find their index in the period-filtered data, which is always sorted
            return np.searchsorted(period_filters[t], inds)

        if folds is not None:
            translated_folds = []
            for (train, test) in folds:
                translated_folds.append((_translate_inds(0, train), _translate_inds(0, test)))
                # sanity check that the folds are the same no matter the time period
                for t in range(1, self.n_periods):
                    assert np.array_equal(_translate_inds(t, train), _translate_inds(0, train))
                    assert np.array_equal(_translate_inds(t, test), _translate_inds(0, test))
        else:
            translated_folds = None

        for t in np.arange(self.n_periods):
            self._model_y_trained[t].train(
                is_selecting, translated_folds,
                self._index_or_None(X, period_filters[t]),
                self._index_or_None(
                    W, period_filters[t]),
                Y[period_filters[self.n_periods - 1]])
            for j in np.arange(t, self.n_periods):
                self._model_t_trained[j][t].train(
                    is_selecting, translated_folds,
                    self._index_or_None(X, period_filters[t]),
                    self._index_or_None(W, period_filters[t]),
                    T[period_filters[j]])
        return self

    def predict(self, Y, T, X=None, W=None, sample_weight=None, groups=None):
        """Calculate nuisances for each period or period pairs.

        Returns
        -------
        Y_res : (n, d_y) matrix or vector of length n
            Y residuals for each period in panel format.
            This shape is required for _OrthoLearner's crossfitting.
        T_res : (n, d_t, n_periods) matrix
            T residuals for pairs of periods (t, j), where the data is in panel format for t
            and in index form for j. For example, the residuals for (t, j) can be retrieved via
            T_res[np.arange(n) % n_periods == t, ..., j]. For t < j, the entries of this
            matrix are np.nan.
            This shape is required for _OrthoLearner's crossfitting.
        """
        assert Y.shape[0] % self.n_periods == 0, \
            "Length of training data should be an integer multiple of time periods."
        period_filters = _get_groups_period_filter(groups, self.n_periods)
        Y_res = np.full(Y.shape, np.nan)
        T_res = np.full(T.shape + (self.n_periods, ), np.nan)
        shape_formatter = self._get_shape_formatter(X, W)
        for t in np.arange(self.n_periods):
            Y_slice = Y[period_filters[self.n_periods - 1]]
            Y_pred = self._model_y_trained[t].predict(
                self._index_or_None(X, period_filters[t]),
                self._index_or_None(W, period_filters[t]))
            Y_res[period_filters[t]] = Y_slice\
                - shape_formatter(Y_slice, Y_pred)
            for j in np.arange(t, self.n_periods):
                T_slice = T[period_filters[j]]
                T_pred = self._model_t_trained[j][t].predict(
                    self._index_or_None(X, period_filters[t]),
                    self._index_or_None(W, period_filters[t]))
                T_res[period_filters[j], ..., t] = T_slice\
                    - shape_formatter(T_slice, T_pred)
        return Y_res, T_res

    def score(self, Y, T, X=None, W=None, sample_weight=None, groups=None):
        assert Y.shape[0] % self.n_periods == 0, \
            "Length of training data should be an integer multiple of time periods."
        period_filters = _get_groups_period_filter(groups, self.n_periods)
        if hasattr(self._model_y, 'score'):
            Y_score = np.full((self.n_periods, ), np.nan)
            for t in np.arange(self.n_periods):
                Y_score[t] = self._model_y_trained[t].score(
                    self._index_or_None(X, period_filters[t]),
                    self._index_or_None(W, period_filters[t]),
                    Y[period_filters[self.n_periods - 1]])
        else:
            Y_score = None
        if hasattr(self._model_t, 'score'):
            T_score = np.full((self.n_periods, self.n_periods), np.nan)
            for t in np.arange(self.n_periods):
                for j in np.arange(t, self.n_periods):
                    T_score[j][t] = self._model_t_trained[j][t].score(
                        self._index_or_None(X, period_filters[t]),
                        self._index_or_None(W, period_filters[t]),
                        T[period_filters[j]])
        else:
            T_score = None
        return Y_score, T_score

    def _get_shape_formatter(self, X, W):
        if (X is None) and (W is None):
            return lambda x, x_pred: np.tile(x_pred.reshape(1, -1), (x.shape[0], 1)).reshape(x.shape)
        return lambda x, x_pred: x_pred.reshape(x.shape)

    def _index_or_None(self, X, filter_idx):
        return None if X is None else X[filter_idx]


class _DynamicModelFinal:
    """
    Final model at fit time, fits a residual on residual regression with a heterogeneous coefficient
    that depends on X, i.e.

        .. math ::
            Y - E[Y | X, W] = \\theta(X) \\cdot (T - E[T | X, W]) + \\epsilon

    and at predict time returns :math:`\\theta(X)`. The score method returns the MSE of this final
    residual on residual regression.
    Assumes model final is parametric with no intercept.
    """
    # TODO: update docs

    def __init__(self, model_final, n_periods):
        self._model_final = model_final
        self.n_periods = n_periods
        self._model_final_trained = {k: clone(self._model_final, safe=False) for k in np.arange(n_periods)}

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None, groups=None):
        # NOTE: sample weight, sample var are not passed in
        period_filters = _get_groups_period_filter(groups, self.n_periods)
        Y_res, T_res = nuisances
        self._d_y = Y.shape[1:]
        for t in np.arange(self.n_periods - 1, -1, -1):
            Y_adj = Y_res[period_filters[t]].copy()
            if t < self.n_periods - 1:
                Y_adj -= np.sum(
                    [self._model_final_trained[j].predict_with_res(
                        X[period_filters[0]] if X is not None else None,
                        T_res[period_filters[j], ..., t]
                    ) for j in np.arange(t + 1, self.n_periods)], axis=0)
            self._model_final_trained[t].fit(
                X[period_filters[0]] if X is not None else None, T[period_filters[t]],
                T_res[period_filters[t], ..., t], Y_adj)

        return self

    def predict(self, X=None):
        """
        Return shape: m x dy x (p*dt)
        """
        d_t_tuple = self._model_final_trained[0]._d_t
        d_t = d_t_tuple[0] if d_t_tuple else 1
        x_dy_shape = (X.shape[0] if X is not None else 1, ) + \
            self._model_final_trained[0]._d_y
        preds = np.zeros(
            x_dy_shape +
            (self.n_periods * d_t, )
        )
        for t in range(self.n_periods):
            preds[..., t * d_t: (t + 1) * d_t] = \
                self._model_final_trained[t].predict(X).reshape(
                x_dy_shape + (d_t, )
            )
        return preds

    def score(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None, groups=None):
        assert Y.shape[0] % self.n_periods == 0, \
            "Length of training data should be an integer multiple of time periods."
        Y_res, T_res = nuisances
        scores = np.full((self.n_periods, ), np.nan)
        period_filters = _get_groups_period_filter(groups, self.n_periods)
        for t in np.arange(self.n_periods - 1, -1, -1):
            Y_adj = Y_res[period_filters[t]].copy()
            if t < self.n_periods - 1:
                Y_adj -= np.sum(
                    [self._model_final_trained[j].predict_with_res(
                        X[period_filters[0]] if X is not None else None,
                        T_res[period_filters[j], ..., t]
                    ) for j in np.arange(t + 1, self.n_periods)], axis=0)
            Y_adj_pred = self._model_final_trained[t].predict_with_res(
                X[period_filters[0]] if X is not None else None,
                T_res[period_filters[t], ..., t])
            if sample_weight is not None:
                scores[t] = np.mean(np.average((Y_adj - Y_adj_pred)**2, weights=sample_weight, axis=0))
            else:
                scores[t] = np.mean((Y_adj - Y_adj_pred) ** 2)
        return scores


class _LinearDynamicModelFinal(_DynamicModelFinal):
    """Wrapper for the DynamicModelFinal with StatsModelsLinearRegression final model.

    The final model is a linear model with (d_t*n_periods) coefficients.
    This model is defined after the coefficients and covariance are calculated.
    """

    def __init__(self, model_final, n_periods):
        super().__init__(model_final, n_periods)
        self.model_final_ = StatsModelsLinearRegression(fit_intercept=False)

    def fit(self, Y, T, X=None, W=None, Z=None, nuisances=None, sample_weight=None, sample_var=None, groups=None):
        super().fit(Y, T, X=X, W=W, Z=Z, nuisances=nuisances,
                    sample_weight=sample_weight, sample_var=sample_var, groups=groups)
        # Compose final model
        cov = self._get_cov(nuisances, X, groups)
        coef = self._get_coef_()
        self.model_final_._n_out = self._d_y[0] if self._d_y else 0
        self.model_final_._param_var = cov / (Y.shape[0] / self.n_periods)
        self.model_final_._param = coef.T if self.model_final_._n_out else coef

    def _get_coef_(self):
        period_coefs = np.array([self._model_final_trained[t]._model.coef_ for t in range(self.n_periods)])
        if self._d_y:
            return np.array([
                np.array([period_coefs[k, i, :] for k in range(self.n_periods)]).flatten()
                for i in range(self._d_y[0])
            ])
        return period_coefs.flatten()

    def _get_cov(self, nuisances, X, groups):
        if self._d_y:
            return np.array(
                [self._fit_single_output_cov((nuisances[0][:, i], nuisances[1]), X, i, groups)
                 for i in range(self._d_y[0])]
            )
        return self._fit_single_output_cov(nuisances, X, -1, groups)

    def _fit_single_output_cov(self, nuisances, X, y_index, groups):
        """ Calculates the covariance (n_periods*n_treatments)
            x (n_periods*n_treatments) matrix for a single outcome.
        """
        Y_res, T_res = nuisances
        # Calculate auxiliary quantities
        period_filters = _get_groups_period_filter(groups, self.n_periods)
        # X ⨂ T_res
        XT_res = np.array([
            [
                self._model_final_trained[0]._combine(
                    X[period_filters[0]] if X is not None else None,
                    T_res[period_filters[t], ..., j],
                    fitting=False
                )
                for j in range(self.n_periods)
            ]
            for t in range(self.n_periods)
        ])
        d_xt = XT_res.shape[-1]
        # sum(model_final.predict(X, T_res))
        Y_diff = np.array([
            np.sum([
                self._model_final_trained[j].predict_with_res(
                    X[period_filters[0]] if X is not None else None,
                    T_res[period_filters[j], ..., t]
                ) for j in np.arange(t, self.n_periods)],
                axis=0
            )
            for t in np.arange(self.n_periods)
        ])
        J = np.zeros((self.n_periods * d_xt,
                      self.n_periods * d_xt))
        Sigma = np.zeros((self.n_periods * d_xt,
                          self.n_periods * d_xt))
        for t in np.arange(self.n_periods):
            res_epsilon_t = (Y_res[period_filters[t]] -
                             (Y_diff[t][:, y_index] if y_index >= 0 else Y_diff[t])
                             ).reshape(-1, 1, 1)
            resT_t = XT_res[t][t]
            for j in np.arange(self.n_periods):
                # Calculating the (t, j) block entry (of size n_treatments x n_treatments) of matrix Sigma
                res_epsilon_j = (Y_res[period_filters[j]] -
                                 (Y_diff[j][:, y_index] if y_index >= 0 else Y_diff[j])
                                 ).reshape(-1, 1, 1)
                resT_j = XT_res[j][j]
                cov_resT_tj = resT_t.reshape(-1, d_xt, 1) @ resT_j.reshape(-1, 1, d_xt)
                sigma_tj = np.mean((res_epsilon_t * res_epsilon_j) * cov_resT_tj, axis=0)
                Sigma[t * d_xt:(t + 1) * d_xt,
                      j * d_xt:(j + 1) * d_xt] = sigma_tj
                if j >= t:
                    # Calculating the (t, j) block entry (of size n_treatments x n_treatments) of matrix J
                    m_tj = np.mean(
                        XT_res[j][t].reshape(-1, d_xt, 1) @ resT_t.reshape(-1, 1, d_xt),
                        axis=0)
                    J[t * d_xt:(t + 1) * d_xt,
                      j * d_xt:(j + 1) * d_xt] = m_tj
        return np.linalg.inv(J) @ Sigma @ np.linalg.inv(J).T


class _DynamicFinalWrapper(_FinalWrapper):

    def predict_with_res(self, X, T_res):
        fts = self._combine(X, T_res, fitting=False)
        prediction = self._model.predict(fts)
        if self._intercept is not None:
            prediction -= self._intercept
        return reshape(prediction, (prediction.shape[0],) + self._d_y)


class DynamicDML(LinearModelFinalCateEstimatorMixin, _OrthoLearner):
    """CATE estimator for dynamic treatment effect estimation.

    This estimator is an extension of the Double ML approach for treatments assigned sequentially
    over time periods.

    The estimator is a special case of an :class:`_OrthoLearner` estimator, so it follows the two
    stage process, where a set of nuisance functions are estimated in the first stage in a crossfitting
    manner and a final stage estimates the CATE model. See the documentation of
    :class:`._OrthoLearner` for a description of this two stage process.

    Parameters
    ----------
    model_y: estimator, default ``'auto'``
        Determines how to fit the outcome to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    model_t: estimator, default ``'auto'``
        Determines how to fit the treatment to the features.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_treatment` is True
          and a regressor otherwise

    featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    fit_cate_intercept : bool, default True
        Whether the linear CATE model should have a constant term.

    discrete_outcome: bool, default False
        Whether the outcome should be treated as binary

    discrete_treatment: bool, default ``False``
        Whether the treatment values should be treated as categorical, rather than continuous, quantities

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
          Iterables should make sure a group belongs to a single split.

        For integer/None inputs, :class:`~sklearn.model_selection.GroupKFold` is used

        Unless an iterable is used, we call `split(X, T, groups)` to generate the splits.

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
        Whether to allow missing values in W. If True, will need to supply nuisance_models
        that can handle missing values.

    Examples
    --------
    A simple example with default models:

    .. testcode::
        :hide:

        import numpy as np
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.panel.dml import DynamicDML

        np.random.seed(123)

        n_panels = 100  # number of panels
        n_periods = 3  # number of time periods per panel
        n = n_panels * n_periods
        groups = np.repeat(a=np.arange(n_panels), repeats=n_periods, axis=0)
        X = np.random.normal(size=(n, 1))
        T = np.random.normal(size=(n, 2))
        y = np.random.normal(size=(n, ))
        est = DynamicDML()
        est.fit(y, T, X=X, W=None, groups=groups, inference="auto")

    >>> est.const_marginal_effect(X[:2])
    array([[-0.363..., -0.049..., -0.044...,  0.042..., -0.202...,
             0.023...],
           [-0.128...,  0.424...,  0.050... , -0.203..., -0.115...,
            -0.135...]])
    >>> est.effect(X[:2], T0=0, T1=1)
    array([-0.594..., -0.107...])
    >>> est.effect(X[:2], T0=np.zeros((2, n_periods*T.shape[1])), T1=np.ones((2, n_periods*T.shape[1])))
    array([-0.594..., -0.107...])
    >>> est.coef_
    array([[ 0.112... ],
           [ 0.227...],
           [ 0.045...],
           [-0.118...],
           [ 0.041...],
           [-0.076...]])
    >>> est.coef__interval()
    (array([[-0.060...],
           [-0.008...],
           [-0.120...],
           [-0.392...],
           [-0.120...],
           [-0.257...]]), array([[0.286...],
           [0.463...],
           [0.212... ],
           [0.156...],
           [0.204...],
           [0.104...]]))
    """

    def __init__(self, *,
                 model_y='auto', model_t='auto',
                 featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages="deprecated",
                 discrete_outcome=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None,
                 allow_missing=False):
        self.fit_cate_intercept = fit_cate_intercept
        if linear_first_stages != "deprecated":
            warn("The linear_first_stages parameter is deprecated and will be removed in a future version of EconML",
                 DeprecationWarning)
        self.featurizer = clone(featurizer, safe=False)
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        super().__init__(discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         treatment_featurizer=None,
                         discrete_instrument=False,
                         categories=categories,
                         cv=GroupKFold(cv) if isinstance(cv, int) else cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing)

    def _gen_allowed_missing_vars(self):
        return ['W'] if self.allow_missing else []

    # override only so that we can exclude treatment featurization verbiage in docstring
    def const_marginal_effect(self, X=None):
        """
        Calculate the constant marginal CATE :math:`\\theta(·)`.

        The marginal effect is conditional on a vector of
        features on a set of m test samples X[i].

        Parameters
        ----------
        X: (m, d_x) matrix, optional
            Features for each sample.

        Returns
        -------
        theta: (m, d_y, d_t) matrix or (d_y, d_t) matrix if X is None
            Constant marginal CATE of each treatment on each outcome for each sample X[i].
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will also be a vector)
        """
        return super().const_marginal_effect(X=X)

    # override only so that we can exclude treatment featurization verbiage in docstring
    def const_marginal_ate(self, X=None):
        """
        Calculate the average constant marginal CATE :math:`E_X[\\theta(X)]`.

        Parameters
        ----------
        X: (m, d_x) matrix, optional
            Features for each sample.

        Returns
        -------
        theta: (d_y, d_t) matrix
            Average constant marginal CATE of each treatment on each outcome.
            Note that when Y or T is a vector rather than a 2-dimensional array,
            the corresponding singleton dimensions in the output will be collapsed
            (e.g. if both are vectors, then the output of this method will be a scalar)
        """
        return super().const_marginal_ate(X=X)

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        return _make_first_stage_selector(self.model_y,
                                          is_discrete=self.discrete_outcome,
                                          random_state=self.random_state)

    def _gen_model_t(self):
        return _make_first_stage_selector(self.model_t,
                                          is_discrete=self.discrete_treatment,
                                          random_state=self.random_state)

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=False)

    def _gen_ortho_learner_model_nuisance(self):
        return _DynamicModelNuisanceSelector(
            model_t=self._gen_model_t(),
            model_y=self._gen_model_y(),
            n_periods=self._n_periods)

    def _gen_ortho_learner_model_final(self):
        wrapped_final_model = _DynamicFinalWrapper(
            StatsModelsLinearRegression(fit_intercept=False),
            fit_cate_intercept=self.fit_cate_intercept,
            featurizer=self.featurizer,
            use_weight_trick=False)
        return _LinearDynamicModelFinal(wrapped_final_model, n_periods=self._n_periods)

    def _prefit(self, Y, T, *args, groups=None, only_final=False, **kwargs):
        # we need to set the number of periods before calling super()._prefit, since that will generate the
        # final and nuisance models, which need to have self._n_periods set
        u_periods = np.unique(np.unique(groups, return_counts=True)[1])
        if len(u_periods) > 1:
            raise AttributeError(
                "Imbalanced panel. Method currently expects only panels with equal number of periods. Pad your data")
        self._n_periods = u_periods[0]
        super()._prefit(Y, T, *args, **kwargs)

    def _postfit(self, Y, T, *args, **kwargs):
        super()._postfit(Y, T, *args, **kwargs)
        # Set _d_t to effective number of treatments
        self._d_t = (self._n_periods * self._d_t[0], ) if self._d_t else (self._n_periods, )

    def _strata(self, Y, T, X=None, W=None, Z=None,
                sample_weight=None, sample_var=None, groups=None,
                cache_values=False, only_final=False, check_input=True):
        # Required for bootstrap inference
        return groups

    def fit(self, Y, T, *, X=None, W=None, sample_weight=None, sample_var=None, groups,
            cache_values=False, inference='auto'):
        """Estimate the counterfactual model from data, i.e. estimates function :math:`\\theta(\\cdot)`.

        The input data must contain groups with the same size corresponding to the number
        of time periods the treatments were assigned over.

        The data should be preferably in panel format, with groups clustered together.
        If group members do not appear together, the following is assumed:

        * the first instance of a group in the dataset is assumed to correspond to the first period of that group
        * the second instance of a group in the dataset is assumed to correspond to the
          second period of that group

        ...etc.

        Only the value of the features X at the first period of each unit are used for
        heterogeneity. The value of X in subseuqnet periods is used as a time-varying control
        but not for heterogeneity.

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample (required: n = n_groups * n_periods)
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample (required: n = n_groups * n_periods)
        X:(n, d_x) matrix, optional
            Features for each sample (Required: n = n_groups * n_periods). Only first
            period features from each unit are used for heterogeneity, the rest are
            used as time-varying controls together with W
        W:(n, d_w) matrix, optional
            Controls for each sample (Required: n = n_groups * n_periods)
        sample_weight:(n,) vector, optional
            Weights for each samples
        sample_var:(n,) vector, optional
            Sample variance for each sample
        groups: (n,) vector, required
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str,:class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`) and 'auto'
            (or an instance of :class:`.LinearModelFinalInference`).

        Returns
        -------
        self: DynamicDML instance
        """
        if sample_weight is not None or sample_var is not None:
            warn("This CATE estimator does not yet support sample weights and sample variance. "
                 "These inputs will be ignored during fitting.",
                 UserWarning)
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=None, sample_var=None, groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    def score(self, Y, T, X=None, W=None, sample_weight=None, *, groups):
        """
        Score the fitted CATE model on a new data set. Generates nuisance parameters
        for the new data set based on the fitted residual nuisance models created at fit time.
        It uses the mean prediction of the models fitted by the different crossfit folds.
        Then calculates the MSE of the final residual Y on residual T regression.

        If model_final does not have a score method, then it raises an :exc:`.AttributeError`

        Parameters
        ----------
        Y: (n, d_y) matrix or vector of length n
            Outcomes for each sample (required: n = n_groups * n_periods)
        T: (n, d_t) matrix or vector of length n
            Treatments for each sample (required: n = n_groups * n_periods)
        X:(n, d_x) matrix, optional
            Features for each sample (Required: n = n_groups * n_periods)
        W:(n, d_w) matrix, optional
            Controls for each sample (Required: n = n_groups * n_periods)
        groups: (n,) vector, required
            All rows corresponding to the same group will be kept together during splitting.

        Returns
        -------
        score: float
            The MSE of the final CATE model on the new data.
        """
        if not hasattr(self._ortho_learner_model_final, 'score'):
            raise AttributeError("Final model does not have a score method!")
        Y, T, X, groups = check_input_arrays(Y, T, X, groups)
        W, = check_input_arrays(W, force_all_finite='allow-nan' if 'W' in self._gen_allowed_missing_vars() else True)
        self._check_fitted_dims(X)
        X, T = super()._expand_treatments(X, T)
        n_iters = len(self._models_nuisance)
        n_splits = len(self._models_nuisance[0])

        # for each mc iteration
        for i, models_nuisances in enumerate(self._models_nuisance):
            # for each model under cross fit setting
            for j, mdl in enumerate(models_nuisances):
                nuisance_temp = mdl.predict(Y, T, **filter_none_kwargs(X=X, W=W, groups=groups))
                if not isinstance(nuisance_temp, tuple):
                    nuisance_temp = (nuisance_temp,)

                if i == 0 and j == 0:
                    nuisances = [np.zeros((n_iters * n_splits,) + nuis.shape) for nuis in nuisance_temp]

                for it, nuis in enumerate(nuisance_temp):
                    nuisances[it][i * n_iters + j] = nuis

        for it in range(len(nuisances)):
            nuisances[it] = np.mean(nuisances[it], axis=0)
        return self._ortho_learner_model_final.score(Y, T, nuisances=nuisances,
                                                     **filter_none_kwargs(X=X, W=W,
                                                                          sample_weight=sample_weight, groups=groups))

    def cate_treatment_names(self, treatment_names=None):
        """
        Get treatment names for each time period.

        If the treatment is discrete, it will return expanded treatment names.

        Parameters
        ----------
        treatment_names: list of str of length T.shape[1] or None
            The names of the treatments. If None and the T passed to fit was a dataframe,
            it defaults to the column names from the dataframe.

        Returns
        -------
        out_treatment_names: list of str
            Returns (possibly expanded) treatment names.
        """
        slice_treatment_names = super().cate_treatment_names(treatment_names)
        treatment_names_out = []
        for k in range(self._n_periods):
            treatment_names_out += [f"({t})$_{k}$" for t in slice_treatment_names]
        return treatment_names_out

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

    def _expand_treatments(self, X, *Ts, transform=True):
        # Expand treatments for each time period
        outTs = []
        base_expand_treatments = super()._expand_treatments
        for T in Ts:
            if ndim(T) == 0:
                one_T = base_expand_treatments(X, T, transform=transform)[1]
                one_T = one_T.reshape(-1, 1) if ndim(one_T) == 1 else one_T
                T = np.tile(one_T, (1, self._n_periods, ))
            else:
                assert (T.shape[1] == self._n_periods if self.transformer else T.shape[1] == self._d_t[0]), \
                    f"Expected a list of time period * d_t, instead got a treatment array of shape {T.shape}."
                if self.transformer:
                    T = np.hstack([
                        base_expand_treatments(
                            X, T[:, [t]], transform=transform)[1] for t in range(self._n_periods)
                    ])
            outTs.append(T)
        return (X,) + tuple(outTs)

    @property
    def bias_part_of_coef(self):
        return self.ortho_learner_model_final_._model_final._fit_cate_intercept

    @property
    def fit_cate_intercept_(self):
        return self.ortho_learner_model_final_._model_final._fit_cate_intercept

    @property
    def original_featurizer(self):
        # NOTE: important to use the _ortho_learner_model_final_ attribute instead of the
        #       attribute so that the trained featurizer will be passed through
        return self.ortho_learner_model_final_._model_final_trained[0]._original_featurizer

    @property
    def featurizer_(self):
        # NOTE This is used by the inference methods and has to be the overall featurizer. intended
        # for internal use by the library
        return self.ortho_learner_model_final_._model_final_trained[0]._featurizer

    @property
    def model_final_(self):
        # NOTE This is used by the inference methods and is more for internal use to the library
        #      We need to use the _ortho_learner's copy to retain the information from fitting
        return self.ortho_learner_model_final_.model_final_

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")

    @property
    def models_y(self):
        return [[mdl._model_y for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def models_t(self):
        return [[mdl._model_t for mdl in mdls] for mdls in super().models_nuisance_]

    @property
    def nuisance_scores_y(self):
        return self.nuisance_scores_[0]

    @property
    def nuisance_scores_t(self):
        return self.nuisance_scores_[1]

    @property
    def residuals_(self):
        """
        A tuple (y_res, T_res, X, W), of the residuals from the first stage estimation
        along with the associated X and W. Samples are not guaranteed to be in the same
        order as the input order.
        """
        if not hasattr(self, '_cached_values'):
            raise AttributeError("Estimator is not fitted yet!")
        if self._cached_values is None:
            raise AttributeError("`fit` was called with `cache_values=False`. "
                                 "Set to `True` to enable residual storage.")
        Y_res, T_res = self._cached_values.nuisances
        return Y_res, T_res, self._cached_values.X, self._cached_values.W
