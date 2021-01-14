# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from warnings import warn

import numpy as np
from .dml import _BaseDML
from .dml import _FirstStageWrapper, _FinalWrapper
from ..sklearn_extensions.linear_model import WeightedLassoCVWrapper
from ..sklearn_extensions.model_selection import WeightedStratifiedKFold
from ..inference import NormalInferenceResults
from ..inference._inference import Inference
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import clone, BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from ..utilities import add_intercept, shape, check_inputs, _deprecate_positional
from ..grf import CausalForest, MultiOutputGRF
from .._cate_estimator import LinearCateEstimator
from .._shap import _shap_explain_multitask_model_cate
from .._ortho_learner import _OrthoLearner


class _CausalForestFinalWrapper(_FinalWrapper):

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

    def fit(self, X, T_res, Y_res, sample_weight=None, sample_var=None):
        # Track training dimensions to see if Y or T is a vector instead of a 2-dimensional array
        self._d_t = shape(T_res)[1:]
        self._d_y = shape(Y_res)[1:]
        fts = self._combine(X)
        if sample_var is not None:
            raise ValueError("This estimator does not support sample_var!")
        if T_res.ndim == 1:
            T_res = T_res.reshape((-1, 1))
        if Y_res.ndim == 1:
            Y_res = Y_res.reshape((-1, 1))
        self._model.fit(fts, T_res, Y_res, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._model.predict(self._combine(X, fitting=False)).reshape((-1,) + self._d_y + self._d_t)


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

    def const_marginal_effect_interval(self, X, *, alpha=0.1):
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
                                      pred_stderr=pred_stderr, inf_type='effect')

    def effect_interval(self, X, *, T0, T1, alpha=0.1):
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
        return NormalInferenceResults(d_t=1, d_y=self.d_y, pred=pred,
                                      pred_stderr=pred_stderr, inf_type='effect')


class CausalForestDML(_BaseDML):
    """A Causal Forest [cfdml1]_ combined with double machine learning based residualization of the treatment
    and outcome variables. It fits a forest that solves the local moment equation problem:

    .. code-block::

        E[ (Y - E[Y|X, W] - <theta(x), T - E[T|X, W]> - beta(x)) (T;1) | X=x] = 0

    where E[Y|X, W] and E[T|X, W] are fitted in a first stage in a cross-fitting manner.

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

        For integer/None inputs, if the treatment is discrete
        :class:`~sklearn.model_selection.StratifiedKFold` is used, else,
        :class:`~sklearn.model_selection.KFold` is used
        (with a random shuffle in either case).

        Unless an iterable is used, we call `split(X,T)` to generate the splits.

    mc_iters: int, optional (default=None)
        The number of times to rerun the first stage models to reduce the variance of the nuisances.

    mc_agg: {'mean', 'median'}, optional (default='mean')
        How to aggregate the nuisance value for each sample across the `mc_iters` monte carlo iterations of
        cross-fitting.

    n_estimators : int, default=100
        Number of trees

    criterion : {``"mse"``, ``"het"``}, default="mse"
        The function to measure the quality of a split. Supported criteria
        are ``"mse"`` for the mean squared error in a linear moment estimation tree and ``"het"`` for
        heterogeneity score.

        - The ``"mse"`` criterion finds splits that minimize the score

          .. code-block::

            sum_{child} E[(Y - <theta(child), T> - beta(child))^2 | X=child] weight(child)

          Internally, for the case of more than two treatments or for the case of one treatment with
          ``fit_intercept=True`` then this criterion is approximated by computationally simpler variants for
          computationaly purposes. In particular, it is replaced by:

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

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=10
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float, default=5
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    min_var_fraction_leaf : None or float in (0, 1], default=None
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

    min_var_leaf_on_val : bool, default=False
        Whether the `min_var_fraction_leaf` constraint should also be enforced to hold on the validation set of the
        honest split too. If ``min_var_leaf=None`` then this flag does nothing. Setting this to True should
        be done with caution, as this partially violates the honesty structure, since the treatment variable
        of the validation set is used to inform the split structure of the tree. However, this is a benign
        dependence as it only uses local correlation structure of the treatment T to decide whether
        a split is feasible.

    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
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

    min_impurity_decrease : float, default=0.0
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

    max_samples : int or float in (0, 1], default=.45,
        The number of samples to use for each subsample that is used to train each tree:

        - If int, then train each tree on `max_samples` samples, sampled without replacement from all the samples
        - If float, then train each tree on `ceil(`max_samples` * `n_samples`)`, sampled without replacement
          from all the samples.

        If ``inference=True``, then `max_samples` must either be an integer smaller than `n_samples//2` or a float
        less than or equal to .5.

    min_balancedness_tol: float in [0, .5], default=.45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : bool, default=True
        Whether each tree should be trained in an honest manner, i.e. the training set is split into two equal
        sized subsets, the train and the val set. All samples in train are used to create the split structure
        and all samples in val are used to calculate the value of each node in the tree.

    inference : bool, default=True
        Whether inference (i.e. confidence interval construction and uncertainty quantification of the estimates)
        should be enabled. If ``inference=True``, then the estimator uses a bootstrap-of-little-bags approach
        to calculate the covariance of the parameter vector, with am objective Bayesian debiasing correction
        to ensure that variance quantities are positive.

    fit_intercept : bool, default=True
        Whether we should fit an intercept nuisance parameter beta(x).

    subforest_size : int, default=4,
        The number of trees in each sub-forest that is used in the bootstrap-of-little-bags calculation.
        The parameter `n_estimators` must be divisible by `subforest_size`. Should typically be a small constant.

    n_jobs : int or None, default=-1
        The number of parallel jobs to be used for parallelism; follows joblib semantics.
        `n_jobs=-1` means all available cpu cores. `n_jobs=None` means no parallelism.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.

    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    warm_start : bool, default=False
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
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
                 discrete_treatment=False,
                 categories='auto',
                 cv=2,
                 n_crossfit_splits='raise',
                 mc_iters=None,
                 mc_agg='mean',
                 n_estimators=100,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 min_var_fraction_leaf=None,
                 min_var_leaf_on_val=True,
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
                 warm_start=False):

        # TODO: consider whether we need more care around stateful featurizers,
        #       since we clone it and fit separate copies
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
        self.warm_start = warm_start
        self.n_crossfit_splits = n_crossfit_splits
        if self.n_crossfit_splits != 'raise':
            cv = self.n_crossfit_splits
        super().__init__(discrete_treatment=discrete_treatment,
                         categories=categories,
                         cv=cv,
                         n_splits=n_crossfit_splits,
                         mc_iters=mc_iters,
                         mc_agg=mc_agg,
                         random_state=random_state)

    def _get_inference_options(self):
        options = super()._get_inference_options()
        options.update(blb=_GenericSingleOutcomeModelFinalWithCovInference)
        options.update(auto=_GenericSingleOutcomeModelFinalWithCovInference)
        return options

    def _gen_featurizer(self):
        return clone(self.featurizer, safe=False)

    def _gen_model_y(self):
        if self.model_y == 'auto':
            model_y = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_y = clone(self.model_y, safe=False)
        return _FirstStageWrapper(model_y, True, self._gen_featurizer(), False, self.discrete_treatment)

    def _gen_model_t(self):
        if self.model_t == 'auto':
            if self.discrete_treatment:
                model_t = LogisticRegressionCV(cv=WeightedStratifiedKFold(random_state=self.random_state),
                                               random_state=self.random_state)
            else:
                model_t = WeightedLassoCVWrapper(random_state=self.random_state)
        else:
            model_t = clone(self.model_t, safe=False)
        return _FirstStageWrapper(model_t, False, self._gen_featurizer(), False, self.discrete_treatment)

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
                                           warm_start=self.warm_start))

    def _gen_rlearner_model_final(self):
        return _CausalForestFinalWrapper(self._gen_model_final(), False, self._gen_featurizer(), False)

    # override only so that we can update the docstring to indicate support for `blb`
    @_deprecate_positional("X and W should be passed by keyword only. In a future release "
                           "we will disallow passing X and W by position.", ['X', 'W'])
    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, sample_var=None, groups=None,
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
        W: optional (n × d_w) matrix
            Controls for each sample
        sample_weight: optional (n,) vector
            Weights for each row
        sample_var: optional (n, n_y) vector
            Variance of sample, in case it corresponds to summary of many samples. Currently
            not in use by this method (as inference method does not require sample variance info).
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.
        cache_values: bool, default False
            Whether to cache inputs and first stage results, which will allow refitting a different final model
        inference: string, :class:`.Inference` instance, or None
            Method for performing inference.  This estimator supports 'bootstrap'
            (or an instance of :class:`.BootstrapInference`), 'blb' or 'auto'
            (for Bootstrap-of-Little-Bags based inference)

        Returns
        -------
        self
        """
        if sample_var is not None:
            raise ValueError("This estimator does not support sample_var!")
        if X is None:
            raise ValueError("This estimator does not support X=None!")
        Y, T, X, W = check_inputs(Y, T, X, W=W, multi_output_T=True, multi_output_Y=True)
        return super().fit(Y, T, X=X, W=W,
                           sample_weight=sample_weight, sample_var=sample_var, groups=groups,
                           cache_values=cache_values,
                           inference=inference)

    def refit_final(self, *, inference='auto'):
        return super().refit_final(inference=inference)
    refit_final.__doc__ = _OrthoLearner.refit_final.__doc__

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        imps = self.model_final_.feature_importances(max_depth=max_depth, depth_decay_exponent=depth_decay_exponent)
        return imps.reshape(self._d_y + (-1,))

    def shap_values(self, X, *, feature_names=None, treatment_names=None, output_names=None, background_samples=100):
        if self.featurizer_ is not None:
            F = self.featurizer_.transform(X)
        else:
            F = X
        feature_names = self.cate_feature_names(feature_names)
        return _shap_explain_multitask_model_cate(self.const_marginal_effect, self.model_cate.estimators_, F,
                                                  self._d_t, self._d_y, feature_names=feature_names,
                                                  treatment_names=treatment_names,
                                                  output_names=output_names,
                                                  input_names=self._input_names,
                                                  background_samples=background_samples)
    shap_values.__doc__ = LinearCateEstimator.shap_values.__doc__

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

    #######################################################
    # These should be removed once `n_splits` is deprecated
    #######################################################

    @property
    def n_crossfit_splits(self):
        return self.cv

    @n_crossfit_splits.setter
    def n_crossfit_splits(self, value):
        if value != 'raise':
            warn("Deprecated by parameter `n_crossfit_splits` and will be removed in next version.")
        self.cv = value
