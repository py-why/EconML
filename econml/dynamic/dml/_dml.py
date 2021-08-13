# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import numpy as np
from warnings import warn
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from scipy.stats import norm
from sklearn.linear_model import (ElasticNetCV, LassoCV, LogisticRegressionCV)
from ...sklearn_extensions.linear_model import (StatsModelsLinearRegression, WeightedLassoCVWrapper)
from ...sklearn_extensions.model_selection import WeightedStratifiedKFold
from ...dml.dml import _FirstStageWrapper, _FinalWrapper
from ..._cate_estimator import TreatmentExpansionMixin, LinearModelFinalCateEstimatorMixin
from ..._ortho_learner import _OrthoLearner
from ...utilities import (_deprecate_positional, add_intercept,
                          broadcast_unit_treatments, check_high_dimensional,
                          cross_product, deprecated, fit_with_groups,
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


class _DynamicModelNuisance:
    """
    Nuisance model fits the model_y and model_t at fit time and at predict time
    calculates the residual Y and residual T based on the fitted models and returns
    the residuals as two nuisance parameters.
    """

    def __init__(self, model_y, model_t, n_periods):
        self._model_y = model_y
        self._model_t = model_t
        self.n_periods = n_periods

    def fit(self, Y, T, X=None, W=None, sample_weight=None, groups=None):
        """Fit a series of nuisance models for each period or period pairs."""
        assert Y.shape[0] % self.n_periods == 0, \
            "Length of training data should be an integer multiple of time periods."
        period_filters = _get_groups_period_filter(groups, self.n_periods)
        self._model_y_trained = {}
        self._model_t_trained = {j: {} for j in np.arange(self.n_periods)}
        for t in np.arange(self.n_periods):
            self._model_y_trained[t] = clone(self._model_y, safe=False).fit(
                self._index_or_None(X, period_filters[t]),
                self._index_or_None(
                    W, period_filters[t]),
                Y[period_filters[self.n_periods - 1]])
            for j in np.arange(t, self.n_periods):
                self._model_t_trained[j][t] = clone(self._model_t, safe=False).fit(
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
    model_y: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the response to the features. Must implement
        `fit` and `predict` methods.
        If 'auto' :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV` will be chosen.

    model_t: estimator or 'auto', optional (default is 'auto')
        The estimator for fitting the treatment to the features.
        If estimator, it must implement `fit` and `predict` methods;
        If 'auto', :class:`~sklearn.linear_model.LogisticRegressionCV` will be applied for discrete treatment,
        and :class:`.WeightedLassoCV`/:class:`.WeightedMultiTaskLassoCV`
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

    categories: 'auto' or list, default 'auto'
        The categories to use when encoding discrete treatments (or 'auto' to use the unique sorted values).
        The first category will be treated as the control treatment.

    cv: int, cross-validation generator or an iterable, optional (Default=2)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`
        - An iterable yielding (train, test) splits as arrays of indices.
          Iterables should make sure a group belongs to a single split.

        For integer/None inputs, :class:`~sklearn.model_selection.GroupKFold` is used

        Unless an iterable is used, we call `split(X, T, groups)` to generate the splits.

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

    Examples
    --------
    A simple example with default models:

    .. testcode::
        :hide:

        import numpy as np
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dynamic.dml import DynamicDML

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
    array([[-0.336..., -0.048..., -0.061...,  0.042..., -0.204...,
         0.00667271],
        [-0.101...,  0.433...,  0.054..., -0.217..., -0.101...,
         -0.159...]])
    >>> est.effect(X[:2], T0=0, T1=1)
    array([-0.601..., -0.091...])
    >>> est.effect(X[:2], T0=np.zeros((2, n_periods*T.shape[1])), T1=np.ones((2, n_periods*T.shape[1])))
    array([-0.601..., -0.091...])
    >>> est.coef_
    array([[ 0.112...],
       [ 0.231...],
       [ 0.055...],
       [-0.125...],
       [ 0.049...],
       [-0.079...]])
    >>> est.coef__interval()
    (array([[-0.063...],
           [-0.009...],
           [-0.114...],
           [-0.413...],
           [-0.117...],
           [-0.262...]]), array([[0.289...],
           [0.471...],
           [0.225...],
           [0.163...],
           [0.216...],
           [0.103...]]))
    """

    def __init__(self, *,
                 model_y='auto', model_t='auto',
                 featurizer=None,
                 fit_cate_intercept=True,
                 linear_first_stages=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 random_state=None):
        self.fit_cate_intercept = fit_cate_intercept
        self.linear_first_stages = linear_first_stages
        self.featurizer = clone(featurizer, safe=False)
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        super().__init__(discrete_treatment=discrete_treatment,
                         discrete_instrument=False,
                         categories=categories,
                         cv=GroupKFold(cv) if isinstance(cv, int) else cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        if self.model_y == 'auto':
            model_y = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_y = clone(self.model_y, safe=False)
        return _FirstStageWrapper(model_y, True, self._gen_featurizer(),
                                  self.linear_first_stages, self.discrete_treatment)

    def _gen_model_t(self):
        if self.model_t == 'auto':
            if self.discrete_treatment:
                model_t = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                               random_state=self.random_state)
            else:
                model_t = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_t = clone(self.model_t, safe=False)
        return _FirstStageWrapper(model_t, False, self._gen_featurizer(),
                                  self.linear_first_stages, self.discrete_treatment)

    def _gen_model_final(self):
        return StatsModelsLinearRegression(fit_intercept=False)

    def _gen_ortho_learner_model_nuisance(self, n_periods):
        return _DynamicModelNuisance(
            model_t=self._gen_model_t(),
            model_y=self._gen_model_y(),
            n_periods=n_periods)

    def _gen_ortho_learner_model_final(self, n_periods):
        wrapped_final_model = _DynamicFinalWrapper(
            StatsModelsLinearRegression(fit_intercept=False),
            fit_cate_intercept=self.fit_cate_intercept,
            featurizer=self.featurizer,
            use_weight_trick=False)
        return _LinearDynamicModelFinal(wrapped_final_model, n_periods=n_periods)

    def _prefit(self, Y, T, *args, groups=None, only_final=False, **kwargs):
        u_periods = np.unique(np.unique(groups, return_counts=True)[1])
        if len(u_periods) > 1:
            raise AttributeError(
                "Imbalanced panel. Method currently expects only panels with equal number of periods. Pad your data")
        self._n_periods = u_periods[0]
        # generate an instance of the final model
        self._ortho_learner_model_final = self._gen_ortho_learner_model_final(self._n_periods)
        if not only_final:
            # generate an instance of the nuisance model
            self._ortho_learner_model_nuisance = self._gen_ortho_learner_model_nuisance(self._n_periods)
        TreatmentExpansionMixin._prefit(self, Y, T, *args, **kwargs)

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
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample (Required: n = n_groups * n_periods). Only first
            period features from each unit are used for heterogeneity, the rest are
            used as time-varying controls together with W
        W: optional(n, d_w) matrix or None (Default=None)
            Controls for each sample (Required: n = n_groups * n_periods)
        sample_weight: optional(n,) vector or None (Default=None)
            Weights for each samples
        sample_var: optional(n,) vector or None (Default=None)
            Sample variance for each sample
        groups: (n,) vector, required
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string,:class:`.Inference` instance, or None
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
        X: optional(n, d_x) matrix or None (Default=None)
            Features for each sample (Required: n = n_groups * n_periods)
        W: optional(n, d_w) matrix or None (Default=None)
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
        Y, T, X, W, groups = check_input_arrays(Y, T, X, W, groups)
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
        treatment_names: list of strings of length T.shape[1] or None
            The names of the treatments. If None and the T passed to fit was a dataframe,
            it defaults to the column names from the dataframe.

        Returns
        -------
        out_treatment_names: list of strings
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
        if self._d_x is None:
            # Handles the corner case when X=None but featurizer might be not None
            return None
        if feature_names is None:
            feature_names = self._input_names["feature_names"]
        if self.original_featurizer is None:
            return feature_names
        return get_feature_names_or_default(self.original_featurizer, feature_names)

    def _expand_treatments(self, X, *Ts):
        # Expand treatments for each time period
        outTs = []
        base_expand_treatments = super()._expand_treatments
        for T in Ts:
            if ndim(T) == 0:
                one_T = base_expand_treatments(X, T)[1]
                one_T = one_T.reshape(-1, 1) if ndim(one_T) == 1 else one_T
                T = np.tile(one_T, (1, self._n_periods, ))
            else:
                assert (T.shape[1] == self._n_periods if self.transformer else T.shape[1] == self._d_t[0]), \
                    f"Expected a list of time period * d_t, instead got a treatment array of shape {T.shape}."
                if self.transformer:
                    T = np.hstack([
                        base_expand_treatments(
                            X, T[:, [t]])[1] for t in range(self._n_periods)
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
