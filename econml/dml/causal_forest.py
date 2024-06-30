# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

from warnings import warn

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from itertools import product
from .dml import _BaseDML
from .dml import _make_first_stage_selector
from ..sklearn_extensions.linear_model import WeightedLassoCVWrapper
from ..sklearn_extensions.model_selection import WeightedStratifiedKFold
from ..inference import NormalInferenceResults
from ..inference._inference import Inference
from ..utilities import (add_intercept, shape, check_inputs, check_input_arrays,
                         _deprecate_positional, cross_product, Summary)
from ..grf import CausalForest, MultiOutputGRF
from .._cate_estimator import LinearCateEstimator
from .._shap import _shap_explain_multitask_model_cate
from .._ortho_learner import _OrthoLearner


class _CausalForestFinalWrapper:

    def __init__(self, model_final, featurizer, discrete_treatment, drate):
        self._model = clone(model_final, safe=False)
        self._original_featurizer = clone(featurizer, safe=False)
        self._featurizer = self._original_featurizer
        self._discrete_treatment = discrete_treatment
        self._drate = drate

    def _combine(self, X, fitting=True):
        if X is not None:
            if self._featurizer is not None:
                F = self._featurizer.fit_transform(X) if fitting else self._featurizer.transform(X)
            else:
                F = X
        else:
            raise AttributeError("Cannot use this method with X=None. Consider "
                                 "using the LinearDML estimator.")
        return F

    def _ate_and_stderr(self, drpreds, mask=None):
        if mask is not None:
            drpreds = drpreds[mask]
        point = np.nanmean(drpreds, axis=0).reshape(self._d_y + self._d_t)
        nonnan = np.sum(~np.isnan(drpreds))
        stderr = (np.nanstd(drpreds, axis=0) / np.sqrt(nonnan)).reshape(self._d_y + self._d_t)
        return point, stderr

    def fit(self, X, T, T_res, Y_res, sample_weight=None, freq_weight=None, sample_var=None, groups=None):
        # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T_res)[1:]
        self._d_y = shape(Y_res)[1:]
        fts = self._combine(X)
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        self._model.fit(fts, T_res, Y_res, sample_weight=sample_weight)
        # Fit a doubly robust average effect
        if self._discrete_treatment and self._drate:
            oob_preds = self._model.oob_predict(fts)
            self._oob_preds = oob_preds
            if np.any(np.isnan(oob_preds)):
                warn("Could not generate out-of-bag predictions on some training data. "
                     "Consider increasing the number of trees. `ate_` results will take the "
                     "average of the subset of training data for which out-of-bag predictions "
                     "where available.")
            residuals = Y_res - np.einsum('ijk,ik->ij', oob_preds, T_res)
            propensities = T - T_res
            VarT = np.clip(propensities * (1 - propensities), 1e-2, np.inf)
            drpreds = oob_preds
            drpreds += cross_product(residuals, T_res / VarT).reshape((-1, Y_res.shape[1], T_res.shape[1]))
            drpreds[np.isnan(oob_preds)] = np.nan
            self.ate_, self.ate_stderr_ = self._ate_and_stderr(drpreds)
            self.att_ = []
            self.att_stderr_ = []
            att, stderr = self._ate_and_stderr(drpreds, np.all(T == 0, axis=1))
            self.att_.append(att)
            self.att_stderr_.append(stderr)
            for t in range(self._d_t[0]):
                att, stderr = self._ate_and_stderr(drpreds, (T[:, t] == 1))
                self.att_.append(att)
                self.att_stderr_.append(stderr)

        return self

    def predict(self, X):
        return self._model.predict(self._combine(X, fitting=False)).reshape((-1,) + self._d_y + self._d_t)

    @property
    def ate_(self):
        if not self._discrete_treatment:
            raise AttributeError("Doubly Robust ATE calculation on training data "
                                 "is available only on discrete treatments!")
        if not self._drate:
            raise AttributeError("Doubly Robust ATE calculation on training data "
                                 "is available only when `drate=True`!")
        return self._ate

    @ate_.setter
    def ate_(self, value):
        self._ate = value

    @property
    def ate_stderr_(self):
        if not self._discrete_treatment:
            raise AttributeError("Doubly Robust ATE calculation on training data "
                                 "is available only on discrete treatments!")
        if not self._drate:
            raise AttributeError("Doubly Robust ATE calculation on training data "
                                 "is available only when `drate=True`!")
        return self._ate_stderr

    @ate_stderr_.setter
    def ate_stderr_(self, value):
        self._ate_stderr = value

    @property
    def att_(self):
        if not self._discrete_treatment:
            raise AttributeError("Doubly Robust ATT calculation on training data "
                                 "is available only on discrete treatments!")
        if not self._drate:
            raise AttributeError("Doubly Robust ATT calculation on training data "
                                 "is available only when `drate=True`!")
        return self._att

    @att_.setter
    def att_(self, value):
        self._att = value

    @property
    def att_stderr_(self):
        if not self._discrete_treatment:
            raise AttributeError("Doubly Robust ATT calculation on training data "
                                 "is available only on discrete treatments!")
        if not self._drate:
            raise AttributeError("Doubly Robust ATT calculation on training data "
                                 "is available only when `drate=True`!")
        return self._att_stderr

    @att_stderr_.setter
    def att_stderr_(self, value):
        self._att_stderr = value


class _GenericSingleOutcomeModelFinalWithCovInference(Inference):

    def prefit(self, estimator, *args, **kwargs):
        self.model_final = estimator.model_final_
        self.featurizer = estimator.featurizer_ if hasattr(estimator, 'featurizer_') else None

    def fit(self, estimator, *args, **kwargs):
        # once the estimator has been fit, it's kosher to store d_t here
        # (which needs to have been expanded if there's a discrete treatment)
        self._est = estimator
        self._d_t = estimator._d_t
        self._d_y = estimator._d_y
        self.d_t = self._d_t[0] if self._d_t else 1
        self.d_y = self._d_y[0] if self._d_y else 1

    def const_marginal_effect_interval(self, X, *, alpha=0.05):
        return self.const_marginal_effect_inference(X).conf_int(alpha=alpha)

    def const_marginal_effect_inference(self, X):
        if X is None:
            raise ValueError("This inference method currently does not support X=None!")
        if self.featurizer is not None:
            X = self.featurizer.transform(X)
        pred, pred_var = self.model_final.predict_and_var(X)
        pred = pred.reshape((-1,) + self._d_y + self._d_t)
        pred_stderr = np.sqrt(np.diagonal(pred_var, axis1=2, axis2=3).reshape((-1,) + self._d_y + self._d_t))
        return NormalInferenceResults(d_t=self.d_t, d_y=self.d_y, pred=pred,
                                      pred_stderr=pred_stderr, mean_pred_stderr=None, inf_type='effect')

    def effect_interval(self, X, *, T0, T1, alpha=0.05):
        return self.effect_inference(X, T0=T0, T1=T1).conf_int(alpha=alpha)

    def effect_inference(self, X, *, T0, T1):
        if X is None:
            raise ValueError("This inference method currently does not support X=None!")
        X, T0, T1 = self._est._expand_treatments(X, T0, T1)
        if self.featurizer is not None:
            X = self.featurizer.transform(X)
        dT = T1 - T0
        if dT.ndim == 1:
            dT = dT.reshape((-1, 1))
        pred, pred_var = self.model_final.predict_projection_and_var(X, dT)
        pred = pred.reshape((-1,) + self._d_y)
        pred_stderr = np.sqrt(pred_var.reshape((-1,) + self._d_y))
        return NormalInferenceResults(d_t=None, d_y=self.d_y, pred=pred,
                                      pred_stderr=pred_stderr, mean_pred_stderr=None, inf_type='effect')

    def marginal_effect_interval(self, T, X, alpha=0.05):
        return self.marginal_effect_inference(T, X).conf_int(alpha=alpha)

    def marginal_effect_inference(self, T, X):
        if X is None:
            raise ValueError("This inference method currently does not support X=None!")
        if not self._est._original_treatment_featurizer:
            return self.const_marginal_effect_inference(X)
        X, T = self._est._expand_treatments(X, T, transform=False)
        if self.featurizer is not None:
            X = self.featurizer.transform(X)

        feat_T = self._est.transformer.transform(T)
        jac_T = self._est.transformer.jac(T)

        d_t_orig = T.shape[1:]
        d_t_orig = d_t_orig[0] if d_t_orig else 1

        d_y = self._d_y[0] if self._d_y else 1
        d_t = self._d_t[0] if self._d_t else 1

        output_shape = [X.shape[0]]
        if self._d_y:
            output_shape.append(self._d_y[0])
        if T.shape[1:]:
            output_shape.append(T.shape[1])
        me_pred = np.zeros(shape=output_shape)
        me_stderr = np.zeros(shape=output_shape)

        for i in range(d_t_orig):
            # conditionally index multiple dimensions depending on shapes of T, Y and feat_T
            jac_index = [slice(None)]
            me_index = [slice(None)]
            if self._d_y:
                me_index.append(slice(None))
            if T.shape[1:]:
                jac_index.append(i)
                me_index.append(i)
            if feat_T.shape[1:]:  # if featurized T is not a vector
                jac_index.append(slice(None))

            jac_slice = jac_T[tuple(jac_index)]
            if jac_slice.ndim == 1:
                jac_slice.reshape((-1, 1))

            e_pred, e_var = self.model_final.predict_projection_and_var(X, jac_slice)
            e_stderr = np.sqrt(e_var)

            if not self._d_y:
                e_pred = e_pred.squeeze(axis=1)
                e_stderr = e_stderr.squeeze(axis=1)

            me_pred[tuple(me_index)] = e_pred
            me_stderr[tuple(me_index)] = e_stderr

        return NormalInferenceResults(d_t=d_t_orig, d_y=self.d_y, pred=me_pred,
                                      pred_stderr=me_stderr, mean_pred_stderr=None, inf_type='effect')


class CausalForestDML(_BaseDML):
    """A Causal Forest [cfdml1]_ combined with double machine learning based residualization of the treatment
    and outcome variables. It fits a forest that solves the local moment equation problem:

    .. code-block::

        E[ (Y - E[Y|X, W] - <theta(x), T - E[T|X, W]> - beta(x)) (T;1) | X=x] = 0

    where E[Y|X, W] and E[T|X, W] are fitted in a first stage in a cross-fitting manner.

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

    treatment_featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite treatment in the final CATE regression.
        The final CATE will be trained on the outcome of featurizer.fit_transform(T).
        If featurizer=None, then CATE is trained on T.

    discrete_outcome: bool, default ``False``
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

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    mc_iters: int, optional
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, default 'mean'
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    drate : bool, default True
        Whether to calculate doubly robust average treatment effect estimate on training data at fit time.
        This happens only if `discrete_treatment=True`. Doubly robust ATE estimation on the training data
        is not available for continuous treatments.

    n_estimators : int, default 100
        Number of trees

    criterion : {``"mse"``, ``"het"``}, default "mse"
        The function to measure the quality of a split. Supported criteria
        are ``"mse"`` for the mean squared error in a linear moment estimation tree and ``"het"`` for
        heterogeneity score.

        - The ``"mse"`` criterion finds splits that minimize the score

          .. code-block::

            sum_{child} E[(Y - <theta(child), T> - beta(child))^2 | X=child] weight(child)

          Internally, for the case of more than two treatments or for the case of two treatments with
          ``fit_intercept=True`` then this criterion is approximated by computationally simpler variants for
          computational purposes. In particular, it is replaced by:

          .. code-block::

            sum_{child} weight(child) * rho(child).T @ E[(T;1) @ (T;1).T | X in child] @ rho(child)

          where:

          .. code-block::

                rho(child) := E[(T;1) @ (T;1).T | X in parent]^{-1}
                                * E[(Y - <theta(x), T> - beta(x)) (T;1) | X in child]

          This can be thought as a heterogeneity inducing score, but putting more weight on scores
          with a large minimum eigenvalue of the child jacobian ``E[(T;1) @ (T;1).T | X in child]``,
          which leads to smaller variance of the estimate and stronger identification of the parameters.

        - The "het" criterion finds splits that maximize the pure parameter heterogeneity score

          .. code-block::

            sum_{child} weight(child) * rho(child)[:n_T].T @ rho(child)[:n_T]

          This can be thought as an approximation to the ideal heterogeneity score:

          .. code-block::

            weight(left) * weight(right) || theta(left) - theta(right)||_2^2 / weight(parent)^2

          as outlined in [cfdml1]_

    max_depth : int, default None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default 10
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default 5
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default 0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    min_var_fraction_leaf : None or float in (0, 1], default None
        A constraint on some proxy of the variation of the treatment vector that should be contained within each
        leaf as a percentage of the total variance of the treatment vector on the whole sample. This avoids
        performing splits where either the variance of the treatment is small and hence the local parameter
        is not well identified and has high variance. The proxy of variance is different for different criterion,
        primarily for computational efficiency reasons. If ``criterion='het'``, then this constraint translates to::

            for all i in {1, ..., T.shape[1]}:
                Var(T[i] | X in leaf) > `min_var_fraction_leaf` * Var(T[i])

        If ``criterion='mse'``, because the criterion stores more information about the leaf for
        every candidate split, then this constraint imposes further constraints on the pairwise correlations
        of different coordinates of each treatment, i.e.::

            for all i neq j:
                sqrt( Var(T[i]|X in leaf) * Var(T[j]|X in leaf)
                    * ( 1 - rho(T[i], T[j]| in leaf)^2 ) )
                    > `min_var_fraction_leaf` sqrt( Var(T[i]) * Var(T[j]) * (1 - rho(T[i], T[j])^2 ) )

        where rho(X, Y) is the Pearson correlation coefficient of two random variables X, Y. Thus this
        constraint also enforces that no two pairs of treatments be very co-linear within a leaf. This
        extra constraint primarily has bite in the case of more than two input treatments and also avoids
        leafs where the parameter estimate has large variance due to local co-linearities of the treatments.

    min_var_leaf_on_val : bool, default False
        Whether the `min_var_fraction_leaf` constraint should also be enforced to hold on the validation set of the
        honest split too. If ``min_var_leaf=None`` then this flag does nothing. Setting this to True should
        be done with caution, as this partially violates the honesty structure, since the treatment variable
        of the validation set is used to inform the split structure of the tree. However, this is a benign
        dependence as it only uses local correlation structure of the treatment T to decide whether
        a split is feasible.

    max_features : int, float, {"auto", "sqrt", "log2"}, or None, default None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and `int(max_features * n_features)` features
          are considered at each split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    min_impurity_decrease : float, default 0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

    max_samples : int or float in (0, 1], default .45,
        The number of samples to use for each subsample that is used to train each tree:

        - If int, then train each tree on `max_samples` samples, sampled without replacement from all the samples
        - If float, then train each tree on `ceil(`max_samples` * `n_samples`)`, sampled without replacement
          from all the samples.

        If ``inference=True``, then `max_samples` must either be an integer smaller than `n_samples//2` or a float
        less than or equal to .5.

    min_balancedness_tol: float in [0, .5], default .45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : bool, default True
        Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal
        sized subsets, the train and the val set. All samples in train are used to create the split structure
        and all samples in val are used to calculate the value of each node in the tree.

    inference : bool, default True
        Whether inference (i.e. confidence interval construction and uncertainty quantification of the estimates)
        should be enabled. If ``inference=True``, then the estimator uses a bootstrap-of-little-bags approach
        to calculate the covariance of the parameter vector, with am objective Bayesian debiasing correction
        to ensure that variance quantities are positive.

    fit_intercept : bool, default True
        Whether we should fit an intercept nuisance parameter beta(x).

    subforest_size : int, default 4,
        The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation.
        The parameter `n_estimators` must be divisible by `subforest_size`. Should typically be a small constant.

    n_jobs : int or None, default -1
        The number of parallel jobs to be used for parallelism; follows joblib semantics.
        `n_jobs=-1` means all available cpu cores. `n_jobs=None` means no parallelism.

    random_state : int, RandomState instance, or None, default None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    verbose : int, default 0
        Controls the verbosity when fitting and predicting.

    allow_missing: bool
        Whether to allow missing values in W. If True, will need to supply model_y, model_y that can handle
        missing values.

    use_ray: bool, default False
        Whether to use Ray to parallelize the cross-validation step. If True, Ray must be installed.

    ray_remote_func_options : dict, default None
        Options to pass to the remote function when using Ray.
        See https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote.html

    Examples
    --------
    A simple example with the default models and discrete treatment:

    .. testcode::
        :hide:

        import numpy as np
        import scipy.special
        np.set_printoptions(suppress=True)

    .. testcode::

        from econml.dml import CausalForestDML

        np.random.seed(123)
        X = np.random.normal(size=(1000, 5))
        T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
        y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
        est = CausalForestDML(discrete_treatment=True)
        est.fit(y, T, X=X, W=None)

    >>> est.effect(X[:3])
    array([0.62947..., 1.64576..., 0.68496... ])
    >>> est.effect_interval(X[:3])
    (array([0.19136...  , 1.17143..., 0.10789...]),
    array([1.06758..., 2.12009..., 1.26203...]))

    Attributes
    ----------
    ate_ : ndarray of shape (n_outcomes, n_treatments)
        The average constant marginal treatment effect of each treatment for each outcome,
        averaged over the training data and with a doubly robust correction. Available only
        when `discrete_treatment=True` and `drate=True`.
    ate_stderr_ : ndarray of shape (n_outcomes, n_treatments)
        The standard error of the `ate_` attribute.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances based on the amount of parameter heterogeneity they create.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized) total heterogeneity that the feature
        creates. Each split that the feature was chosen adds::

            parent_weight * (left_weight * right_weight)
                * mean((value_left[k] - value_right[k])**2) / parent_weight**2

        to the importance of the feature. Each such quantity is also weighted by the depth of the split.
        By default splits below `max_depth=4` are not used in this calculation and also each split
        at depth `depth`, is re-weighted by 1 / (1 + `depth`)**2.0. See the method ``feature_importances``
        for a method that allows one to change these defaults.

    References
    ----------
    .. [cfdml1] Athey, Susan, Julie Tibshirani, and Stefan Wager. "Generalized random forests."
        The Annals of Statistics 47.2 (2019): 1148-1178
        https://arxiv.org/pdf/1610.01271.pdf

    """

    def __init__(self, *,
                 model_y='auto',
                 model_t='auto',
                 featurizer=None,
                 treatment_featurizer=None,
                 discrete_outcome=False,
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 drate=True,
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=False,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
                 min_balancedness_tol=.45,
                 honest=True,
                 inference=True,
                 fit_intercept=True,
                 subforest_size=4,
                 n_jobs=-1,
                 random_state=None,
                 verbose=0,
                 allow_missing=False,
                 use_ray=False,
                 ray_remote_func_options=None):

        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies
        self.drate = drate
        self.model_y = clone(model_y, safe=False)
        self.model_t = clone(model_t, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.discrete_instrument = discrete_treatment
        self.categories = categories
        self.cv = cv
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_var_fraction_leaf = min_var_fraction_leaf
        self.min_var_leaf_on_val = min_var_leaf_on_val
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.inference = inference
        self.fit_intercept = fit_intercept
        self.subforest_size = subforest_size
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__(discrete_outcome=discrete_outcome,
                         discrete_treatment=discrete_treatment,
                         treatment_featurizer=treatment_featurizer,
                         categories=categories,
                         cv=cv,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state,
                         allow_missing=allow_missing,
                         use_ray=use_ray,
                         ray_remote_func_options=ray_remote_func_options)

    def _gen_allowed_missing_vars(self):
        return ['W'] if self.allow_missing else []

    def _get_inference_options(self):
        options = super()._get_inference_options()
        options.update(blb=_GenericSingleOutcomeModelFinalWithCovInference)
        options.update(auto=_GenericSingleOutcomeModelFinalWithCovInference)
        return options

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        return _make_first_stage_selector(self.model_y, self.discrete_outcome, self.random_state)

    def _gen_model_t(self):
        return _make_first_stage_selector(self.model_t, self.discrete_treatment, self.random_state)

    def _gen_model_final(self):
        return MultiOutputGRF(CausalForest(n_estimators=self.n_estimators,
                                           criterion=self.criterion,
                                           max_depth=self.max_depth,
                                           min_samples_split=self.min_samples_split,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                           min_var_fraction_leaf=self.min_var_fraction_leaf,
                                           min_var_leaf_on_val=self.min_var_leaf_on_val,
                                           max_features=self.max_features,
                                           min_impurity_decrease=self.min_impurity_decrease,
                                           max_samples=self.max_samples,
                                           min_balancedness_tol=self.min_balancedness_tol,
                                           honest=self.honest,
                                           inference=self.inference,
                                           fit_intercept=self.fit_intercept,
                                           subforest_size=self.subforest_size,
                                           n_jobs=self.n_jobs,
                                           random_state=self.random_state,
                                           verbose=self.verbose,
                                           warm_start=False))

    def _gen_rlearner_model_final(self):
        return _CausalForestFinalWrapper(self._gen_model_final(), self._gen_featurizer(),
                                         self.discrete_treatment, self.drate)

    @property
    def tunable_params(self):
        return ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf',
                'min_weight_fraction_leaf', 'min_var_fraction_leaf', 'min_var_leaf_on_val',
                'max_features', 'min_impurity_decrease', 'max_samples', 'min_balancedness_tol',
                'honest', 'inference', 'fit_intercept', 'subforest_size']

    def tune(self, Y, T, *, X=None, W=None,
             sample_weight=None, groups=None,
             params='auto'):
        """
        Tunes the major hyperparameters of the final stage causal forest based on out-of-sample R-score
        performance. It trains small forests of size 100 trees on a grid of parameters and tests the
        out of sample R-score. After the function is called, then all parameters of `self` have been
        set to the optimal hyperparameters found. The estimator however remains un-fitted, so you need to
        call fit afterwards to fit the estimator with the chosen hyperparameters. The list of tunable parameters
        can be accessed via the property `tunable_params`.

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: (n × dₓ) matrix
            Features for each sample
        W:  (n × d_w) matrix, optional
            Controls for each sample
        sample_weight:  (n,) vector, optional
            Weights for each row
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        params: dict or 'auto', default 'auto'
            A dictionary that contains the grid of hyperparameters to try, i.e.
            {'param1': [value1, value2, ...], 'param2': [value1, value2, ...], ...}
            If `params='auto'`, then a default grid is used.

        Returns
        -------
        self : CausalForestDML object
            The tuned causal forest object. This is the same object (not a copy) as the original one, but where
            all parameters of the object have been set to the best performing parameters from the tuning grid.
        """
        from ..score import RScorer  # import here to avoid circular import issue
        Y, T, X, sample_weight, groups = check_input_arrays(Y, T, X, sample_weight, groups)
        W, = check_input_arrays(W, force_all_finite='allow-nan' if 'W' in self._gen_allowed_missing_vars() else True)

        if params == 'auto':
            params = {
                'min_weight_fraction_leaf': [0.0001, .01],
                'max_depth': [3, 5, None],
                'min_var_fraction_leaf': [0.001, .01]
            }
        else:
            # If custom param grid, check that only estimator parameters are being altered
            estimator_param_names = self.tunable_params
            for key in params.keys():
                if key not in estimator_param_names:
                    raise ValueError(f"Parameter `{key}` is not an tunable causal forest parameter.")

        strata = None
        if self.discrete_treatment:
            strata = self._strata(Y, T, X=X, W=W, sample_weight=sample_weight, groups=groups)
        # use 0.699 instead of 0.7 as train size so that if there are 5 examples in a stratum, we get 2 in test
        train, test = train_test_split(np.arange(Y.shape[0]), train_size=0.699,
                                       random_state=self.random_state, stratify=strata)
        ytrain, yval, Ttrain, Tval = Y[train], Y[test], T[train], T[test]
        Xtrain, Xval = (X[train], X[test]) if X is not None else (None, None)
        Wtrain, Wval = (W[train], W[test]) if W is not None else (None, None)
        groups_train, groups_val = (groups[train], groups[test]) if groups is not None else (None, None)
        if sample_weight is not None:
            sample_weight_train, sample_weight_val = sample_weight[train], sample_weight[test]
        else:
            sample_weight_train, sample_weight_val = None, None

        est = clone(self, safe=False)
        est.n_estimators = 100
        est.inference = False

        scorer = RScorer(model_y=est.model_y, model_t=est.model_t,
                         discrete_treatment=est.discrete_treatment, categories=est.categories,
                         cv=est.cv, mc_iters=est.mc_iters, mc_agg=est.mc_agg,
                         random_state=est.random_state)
        scorer.fit(yval, Tval, X=Xval, W=Wval, sample_weight=sample_weight_val, groups=groups_val)

        names = params.keys()
        scores = []
        for it, values in enumerate(product(*params.values())):
            for key, value in zip(names, values):
                setattr(est, key, value)
            if it == 0:
                est.fit(ytrain, Ttrain, X=Xtrain, W=Wtrain, sample_weight=sample_weight_train,
                        groups=groups_train, cache_values=True)
            else:
                est.refit_final()
            scores.append((scorer.score(est), tuple(zip(names, values))))

        bestind = np.argmax([s[0] for s in scores])
        _, best_params = scores[bestind]
        for key, value in best_params:
            setattr(self, key, value)

        return self

    # override only so that we can update the docstring to indicate support for `blb`
    def fit(self, Y, T, *, X=None, W=None, sample_weight=None, groups=None,
            cache_values=False, inference='auto'):
        """
        Estimate the counterfactual model from data, i.e. estimates functions τ(·,·,·), ∂τ(·,·).

        Parameters
        ----------
        Y: (n × d_y) matrix or vector of length n
            Outcomes for each sample
        T: (n × dₜ) matrix or vector of length n
            Treatments for each sample
        X: (n × dₓ) matrix
            Features for each sample
        W:  (n × d_w) matrix, optional
            Controls for each sample
        sample_weight : (n,) array_like or None
            Individual weights for each sample. If None, it assumes equal weight.
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: str, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`), 'blb' or 'auto'
            (for Bootstrap-of-Little-Bags based inference)

        Returns
        -------
        self
        """
        if X is None:
            raise ValueError("This estimator does not support X=None!")
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        imps = self.model_final_.feature_importances(max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
        return imps.reshape(self._d_y + (-1,))

    def summary(self, alpha=0.05, value=0, decimals=3, feature_names=None, treatment_names=None, output_names=None):
        """ The summary of coefficient and intercept in the linear model of the constant marginal treatment
        effect.

        Parameters
        ----------
        alpha:  float in [0, 1], default 0.05
            The overall level of confidence of the reported interval.
            The alpha/2, 1-alpha/2 confidence interval is reported.
        value: float, default 0
            The mean value of the metric you'd like to test under null hypothesis.
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
        # Summary
        if self._cached_values is not None:
            print("Population summary of CATE predictions on Training Data")
            smry = self.const_marginal_ate_inference(self._cached_values.X).summary(alpha=alpha, value=value,
                                                                                    decimals=decimals,
                                                                                    output_names=output_names,
                                                                                    treatment_names=treatment_names)
        else:
            print("Population summary results are available only if `cache_values=True` at fit time!")
            smry = Summary()
        d_t = self._d_t[0] if self._d_t else 1
        d_y = self._d_y[0] if self._d_y else 1

        try:
            intercept_table = self.ate__inference().summary_frame(alpha=alpha,
                                                                  value=value, decimals=decimals,
                                                                  feature_names=None,
                                                                  treatment_names=treatment_names,
                                                                  output_names=output_names)
            intercept_array = intercept_table.values
            intercept_headers = intercept_table.columns.tolist()
            n_level = intercept_table.index.nlevels
            if n_level > 1:
                intercept_stubs = ["|".join(ind_value) for ind_value in intercept_table.index.values]
            else:
                intercept_stubs = intercept_table.index.tolist()
            intercept_title = 'Doubly Robust ATE on Training Data Results'
            smry.add_table(intercept_array, intercept_headers, intercept_stubs, intercept_title)
        except Exception as e:
            print("Doubly Robust ATE on Training Data Results: ", str(e))

        for t in range(0, d_t + 1):
            try:
                intercept_table = self.att__inference(T=t).summary_frame(alpha=alpha,
                                                                         value=value, decimals=decimals,
                                                                         feature_names=None,
                                                                         output_names=output_names)
                intercept_array = intercept_table.values
                intercept_headers = intercept_table.columns.tolist()
                n_level = intercept_table.index.nlevels
                if n_level > 1:
                    intercept_stubs = ["|".join(ind_value) for ind_value in intercept_table.index.values]
                else:
                    intercept_stubs = intercept_table.index.tolist()
                intercept_title = "Doubly Robust ATT(T={}) on Training Data Results".format(t)
                smry.add_table(intercept_array, intercept_headers, intercept_stubs, intercept_title)
            except Exception as e:
                print("Doubly Robust ATT on Training Data Results: ", str(e))
                break
        if len(smry.tables) > 0:
            return smry

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        return _shap_explain_multitask_model_cate(self.const_marginal_effect, self.model_cate.estimators_, X,
                                                  self._d_t, self._d_y, featurizer=self.featurizer_,
                                                  feature_names=feature_names,
                                                  treatment_names=treatment_names,
                                                  output_names=output_names,
                                                  input_names=self._input_names,
                                                  background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__

    def ate__inference(self):
        """
        Returns
        -------
        ate__inference : NormalInferenceResults
            Inference results information for the `ate_` attribute, which is the average
            constant marginal treatment effect of each treatment for each outcome, averaged
            over the training data and with a doubly robust correction.
            Available only when `discrete_treatment=True` and `drate=True`.
        """
        return NormalInferenceResults(d_t=self._d_t[0] if self._d_t else 1,
                                      d_y=self._d_y[0] if self._d_y else 1,
                                      pred=self.ate_,
                                      pred_stderr=self.ate_stderr_,
                                      mean_pred_stderr=None,
                                      inf_type='ate',
                                      feature_names=self.cate_feature_names(),
                                      output_names=self.cate_output_names(),
                                      treatment_names=self.cate_treatment_names())

    @property
    def ate_(self):
        return self.rlearner_model_final_.ate_

    @property
    def ate_stderr_(self):
        return self.rlearner_model_final_.ate_stderr_

    def att__inference(self, *, T):
        """
        Parameters
        ----------
        T : int
            The index of the treatment for which to get the ATT. It corresponds to the
            lexicographic rank of the discrete input treatments.

        Returns
        -------
        att__inference : NormalInferenceResults
            Inference results information for the `att_` attribute, which is the average
            constant marginal treatment effect of each treatment for each outcome, averaged
            over the training data treated with treatment T and with a doubly robust correction.
            Available only when `discrete_treatment=True` and `drate=True`.
        """
        return NormalInferenceResults(d_t=self._d_t[0] if self._d_t else 1,
                                      d_y=self._d_y[0] if self._d_y else 1,
                                      pred=self.att_(T=T),
                                      pred_stderr=self.att_stderr_(T=T),
                                      mean_pred_stderr=None,
                                      inf_type='att',
                                      feature_names=self.cate_feature_names(),
                                      output_names=self.cate_output_names(),
                                      treatment_names=self.cate_treatment_names())

    def att_(self, *, T):
        """
        Parameters
        ----------
        T : int
            The index of the treatment for which to get the ATT. It corresponds to the
            lexicographic rank of the discrete input treatments.

        Returns
        -------
        att_ : ndarray (n_y, n_t)
            The average constant marginal treatment effect of each treatment for each outcome, averaged
            over the training data treated with treatment T and with a doubly robust correction.
            Singleton dimensions are dropped if input variable was a vector.
        """
        return self.rlearner_model_final_.att_[T]

    def att_stderr_(self, *, T):
        """
        Parameters
        ----------
        T : int
            The index of the treatment for which to get the ATT. It corresponds to the
            lexicographic rank of the discrete input treatments.

        Returns
        -------
        att_stderr_ : ndarray (n_y, n_t)
            The standard error of the corresponding `att_`
        """
        return self.rlearner_model_final_.att_stderr_[T]

    @property
    def feature_importances_(self):
        return self.feature_importances()

    @property
    def model_final(self):
        return self._gen_model_final()

    @model_final.setter
    def model_final(self, model):
        if model is not None:
            raise ValueError("Parameter `model_final` cannot be altered for this estimator!")

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return self.model_cate.__len__()

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.model_cate.__getitem__(index)

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return self.model_cate.__iter__()
