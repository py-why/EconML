# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

from warnings import warn
import numpy as np
from sklearn.base import clone
from ..utilities import check_inputs, filter_none_kwargs, check_input_arrays
from ..dr import DRLearner
from ..dr._drlearner import _ModelFinal
from .._tree_exporter import _SingleTreeExporterMixin
from ._base import PolicyLearner
from . import PolicyTree, PolicyForest


class _PolicyModelFinal(_ModelFinal):

    def fit(self, Y, T, X=None, W=None, *, nuisances,
            sample_weight=None, freq_weight=None, sample_var=None, groups=None):
        if sample_var is not None:
            warn('Parameter `sample_var` is ignored by the final estimator')
            sample_var = None
        Y_pred, _ = nuisances
        self.d_y = Y_pred.shape[1:-1]  # track whether there's a Y dimension (must be a singleton)
        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.fit_transform(X)
        filtered_kwargs = filter_none_kwargs(sample_weight=sample_weight, sample_var=sample_var)
        ys = Y_pred[..., 1:] - Y_pred[..., [0]]  # subtract control results from each other arm
        if self.d_y:  # need to squeeze out singleton so that we fit on 2D array
            ys = ys.squeeze(1)
        ys = np.hstack([np.zeros((ys.shape[0], 1)), ys])
        self.model_cate = self._model_final.fit(X, ys, **filtered_kwargs)
        return self

    def predict(self, X=None):
        if (X is not None) and (self._featurizer is not None):
            X = self._featurizer.transform(X)
        pred = self.model_cate.predict_value(X)[:, 1:]
        if self.d_y:  # need to reintroduce singleton Y dimension
            return pred[:, np.newaxis, :]
        return pred

    def score(self, Y, T, X=None, W=None, *, nuisances, sample_weight=None, groups=None):
        return 0


class _DRLearnerWrapper(DRLearner):

    def _gen_ortho_learner_model_final(self):
        return _PolicyModelFinal(self._gen_model_final(), self._gen_featurizer(), self.multitask_model_final)


class _BaseDRPolicyLearner(PolicyLearner):

    def _gen_drpolicy_learner(self):
        pass

    def fit(self, Y, T, *, X=None, W=None, sample_weight=None, groups=None):
        """
        Estimate a policy model from data.

        Parameters
        ----------
        Y: (n,) vector of length n
            Outcomes for each sample
        T: (n,) vector of length n
            Treatments for each sample
        X:(n, d_x) matrix, optional
            Features for each sample
        W:(n, d_w) matrix, optional
            Controls for each sample
        sample_weight:(n,) vector, optional
            Weights for each samples
        groups: (n,) vector, optional
            All rows corresponding to the same group will be kept together during splitting.
            If groups is not None, the `cv` argument passed to this class's initializer
            must support a 'groups' argument to its split method.

        Returns
        -------
        self: object instance
        """
        self.drlearner_ = self._gen_drpolicy_learner()
        self.drlearner_.fit(Y, T, X=X, W=W, sample_weight=sample_weight, groups=groups)
        return self

    def predict_value(self, X):
        """ Get effect values for each non-baseline treatment and for each sample.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        values : array_like of shape (n_samples, n_treatments - 1)
            The predicted average value for each sample and for each non-baseline treatment, as compared
            to the baseline treatment value and based on the feature neighborhoods defined by the trees.
        """
        return self.drlearner_.const_marginal_effect(X)

    def predict_proba(self, X):
        """ Predict the probability of recommending each treatment

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        treatment_proba : array_like of shape (n_samples, n_treatments)
            The probability of each treatment recommendation
        """
        X, = check_input_arrays(X)
        if self.drlearner_.featurizer_ is not None:
            X = self.drlearner_.featurizer_.fit_transform(X)
        return self.policy_model_.predict_proba(X)

    def predict(self, X):
        """ Get recommended treatment for each sample.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        treatment : array_like of shape (n_samples,)
            The index of the recommended treatment in the same order as in categories, or in
            lexicographic order if `categories='auto'`. 0 corresponds to the baseline/control treatment.
            For ensemble policy models, recommended treatments are aggregated from each model in the ensemble
            and the treatment that receives the most votes is returned. Use `predict_proba` to get the fraction
            of models in the ensemble that recommend each treatment for each sample.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def policy_feature_names(self, *, feature_names=None):
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
            The names of the output features on which the policy model is fitted.
        """
        return self.drlearner_.cate_feature_names(feature_names=feature_names)

    def policy_treatment_names(self, *, treatment_names=None):
        """
        Get the names of the treatments.

        Parameters
        ----------
        treatment_names: list of str of length n_categories
            The names of the treatments (including the baseling). If None then values are auto-generated
            based on input metadata.

        Returns
        -------
        out_treatment_names: list of str
            The names of the treatments including the baseline/control treatment.
        """
        if treatment_names is not None:
            if len(treatment_names) != len(self.drlearner_.cate_treatment_names()) + 1:
                raise ValueError('The variable `treatment_names` should have length equal to '
                                 'n_treatments + 1, containing the value of the control/none/baseline treatment as '
                                 'the first element and the names of all the treatments as subsequent elements.')
            return treatment_names
        return ['None'] + self.drlearner_.cate_treatment_names()

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        """

        Parameters
        ----------
        max_depth : int, default 4
            Splits of depth larger than `max_depth` are not used in this calculation
        depth_decay_exponent: double, default 2.0
            The contribution of each split to the total score is re-weighted by ``1 / (1 + `depth`)**2.0``.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total parameter heterogeneity inducing importance of each feature
        """
        return self.policy_model_.feature_importances(max_depth=max_depth,
                                                      depth_decay_exponent=depth_decay_exponent)

    @property
    def feature_importances_(self):
        return self.feature_importances()

    @property
    def policy_model_(self):
        """ The trained final stage policy model
        """
        return self.drlearner_.multitask_model_cate


class DRPolicyTree(_BaseDRPolicyLearner):
    """
    Policy learner that uses doubly-robust correction techniques to account for
    covariate shift (selection bias) between the treatment arms.

    In this estimator, the policy is estimated by first constructing doubly robust estimates of the counterfactual
    outcomes

    .. math ::
        Y_{i, t}^{DR} = E[Y | X_i, W_i, T_i=t]\
            + \\frac{Y_i - E[Y | X_i, W_i, T_i=t]}{Pr[T_i=t | X_i, W_i]} \\cdot 1\\{T_i=t\\}

    Then optimizing the objective

    .. math ::
        V(\\pi) = \\sum_i \\sum_t \\pi_t(X_i) * (Y_{i, t} - Y_{i, 0})

    with the constraint that only one of :math:`\\pi_t(X_i)` is 1 and the rest are 0, for each :math:`X_i`.

    Thus if we estimate the nuisance functions :math:`h(X, W, T) = E[Y | X, W, T]` and
    :math:`p_t(X, W)=Pr[T=t | X, W]` in the first stage, we can estimate the final stage cate for each
    treatment t, by running a constructing a decision tree that maximizes the objective :math:`V(\\pi)`

    The problem of estimating the nuisance function :math:`p` is a simple multi-class classification
    problem of predicting the label :math:`T` from :math:`X, W`. The :class:`.DRLearner`
    class takes as input the parameter ``model_propensity``, which is an arbitrary scikit-learn
    classifier, that is internally used to solve this classification problem.

    The second nuisance function :math:`h` is a simple regression problem and the :class:`.DRLearner`
    class takes as input the parameter ``model_regressor``, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem.

    Parameters
    ----------
    model_propensity: estimator, default ``'auto'``
        Classifier for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options

    model_regression: estimator, default ``'auto'``
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    min_propensity : float, default ``1e-6``
        The minimum propensity at which to clip propensity estimates to avoid dividing by zero.

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

    max_depth : int or None, optional
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, default 10
        The minimum number of splitting samples required to split an internal node.

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, default 5
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

    min_weight_fraction_leaf : float, default 0.
        The minimum weighted fraction of the sum total of weights (of all
        splitting samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided. After construction
        the tree is pruned so that the fraction of the sum total weight
        of the estimation samples contained in each leaf node is at
        least min_weight_fraction_leaf

    max_features : int, float, str, or None, default "auto"
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

    min_impurity_decrease : float, default 0.
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

    min_balancedness_tol: float in [0, .5], default .45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : bool, default True
        Whether to use honest trees, i.e. half of the samples are used for
        creating the tree structure and the other half for the estimation at
        the leafs. If False, then all samples are used for both parts.

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, *,
                 model_regression="auto",
                 model_propensity="auto",
                 featurizer=None,
                 min_propensity=1e-6,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 min_balancedness_tol=.45,
                 honest=True,
                 random_state=None):
        self.model_regression = clone(model_regression, safe=False)
        self.model_propensity = clone(model_propensity, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.min_propensity = min_propensity
        self.categories = categories
        self.cv = cv
        self.mc_iters = mc_iters
        self.mc_agg = mc_agg
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.random_state = random_state

    def _gen_drpolicy_learner(self):
        return _DRLearnerWrapper(model_regression=self.model_regression,
                                 model_propensity=self.model_propensity,
                                 featurizer=self.featurizer,
                                 min_propensity=self.min_propensity,
                                 categories=self.categories,
                                 cv=self.cv,
                                 mc_iters=self.mc_iters,
                                 mc_agg=self.mc_agg,
                                 model_final=PolicyTree(max_depth=self.max_depth,
                                                        min_samples_split=self.min_samples_split,
                                                        min_samples_leaf=self.min_samples_leaf,
                                                        min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                        max_features=self.max_features,
                                                        min_impurity_decrease=self.min_impurity_decrease,
                                                        min_balancedness_tol=self.min_balancedness_tol,
                                                        honest=self.honest,
                                                        random_state=self.random_state),
                                 multitask_model_final=True,
                                 random_state=self.random_state)

    def plot(self, *, feature_names=None, treatment_names=None, ax=None, title=None,
             max_depth=None, filled=True, rounded=True, precision=3, fontsize=None):
        """
        Exports policy trees to matplotlib

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            The axes on which to plot

        title : str, optional
            A title for the final figure to be printed at the top of the page.

        feature_names : list of str, optional
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments including the baseline/control

        max_depth: int or None, optional
            The maximum tree depth to plot

        filled : bool, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        rounded : bool, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        precision : int, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.

        fontsize : int, optional
            Font size for text
        """
        return self.policy_model_.plot(feature_names=self.policy_feature_names(feature_names=feature_names),
                                       treatment_names=self.policy_treatment_names(treatment_names=treatment_names),
                                       ax=ax,
                                       title=title,
                                       max_depth=max_depth,
                                       filled=filled,
                                       rounded=rounded,
                                       precision=precision,
                                       fontsize=fontsize)

    def export_graphviz(self, *, out_file=None,
                        feature_names=None, treatment_names=None,
                        max_depth=None, filled=True, leaves_parallel=True,
                        rotate=False, rounded=True, special_characters=False, precision=3):
        """
        Export a graphviz dot file representing the learned tree model

        Parameters
        ----------
        out_file : file object or str, optional
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        feature_names : list of str, optional
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments, including the baseline treatment

        max_depth: int or None, optional
            The maximum tree depth to plot

        filled : bool, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, default True
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, default False
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, default False
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        return self.policy_model_.export_graphviz(out_file=out_file,
                                                  feature_names=self.policy_feature_names(feature_names=feature_names),
                                                  treatment_names=self.policy_treatment_names(
                                                      treatment_names=treatment_names),
                                                  max_depth=max_depth,
                                                  filled=filled,
                                                  leaves_parallel=leaves_parallel,
                                                  rotate=rotate,
                                                  rounded=rounded,
                                                  special_characters=special_characters,
                                                  precision=precision)

    def render(self, out_file, *, format='pdf', view=True, feature_names=None,
               treatment_names=None, max_depth=None,
               filled=True, leaves_parallel=True, rotate=False, rounded=True,
               special_characters=False, precision=3):
        """
        Render the tree to a flie

        Parameters
        ----------
        out_file : file name to save to

        format : str, default 'pdf'
            The file format to render to; must be supported by graphviz

        view : bool, default True
            Whether to open the rendered result with the default application.

        feature_names : list of str, optional
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments, including the baseline/control

        max_depth: int or None, optional
            The maximum tree depth to plot

        filled : bool, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, default True
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, default False
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, default False
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        return self.policy_model_.render(out_file,
                                         format=format,
                                         view=view,
                                         feature_names=self.policy_feature_names(feature_names=feature_names),
                                         treatment_names=self.policy_treatment_names(treatment_names=treatment_names),
                                         max_depth=max_depth,
                                         filled=filled,
                                         leaves_parallel=leaves_parallel,
                                         rotate=rotate,
                                         rounded=rounded,
                                         special_characters=special_characters,
                                         precision=precision)


class DRPolicyForest(_BaseDRPolicyLearner):
    """
    Policy learner that uses doubly-robust correction techniques to account for
    covariate shift (selection bias) between the treatment arms.

    In this estimator, the policy is estimated by first constructing doubly robust estimates of the counterfactual
    outcomes

    .. math ::
        Y_{i, t}^{DR} = E[Y | X_i, W_i, T_i=t]\
            + \\frac{Y_i - E[Y | X_i, W_i, T_i=t]}{Pr[T_i=t | X_i, W_i]} \\cdot 1\\{T_i=t\\}

    Then optimizing the objective

    .. math ::
        V(\\pi) = \\sum_i \\sum_t \\pi_t(X_i) * (Y_{i, t} - Y_{i, 0})

    with the constraint that only one of :math:`\\pi_t(X_i)` is 1 and the rest are 0, for each :math:`X_i`.

    Thus if we estimate the nuisance functions :math:`h(X, W, T) = E[Y | X, W, T]` and
    :math:`p_t(X, W)=Pr[T=t | X, W]` in the first stage, we can estimate the final stage cate for each
    treatment t, by running a constructing a decision tree that maximizes the objective :math:`V(\\pi)`

    The problem of estimating the nuisance function :math:`p` is a simple multi-class classification
    problem of predicting the label :math:`T` from :math:`X, W`. The :class:`.DRLearner`
    class takes as input the parameter ``model_propensity``, which is an arbitrary scikit-learn
    classifier, that is internally used to solve this classification problem.

    The second nuisance function :math:`h` is a simple regression problem and the :class:`.DRLearner`
    class takes as input the parameter ``model_regressor``, which is an arbitrary scikit-learn regressor that
    is internally used to solve this regression problem.

    Parameters
    ----------
    model_propensity: estimator, default ``'auto'``
        Classifier for Pr[T=t | X, W]. Trained by regressing treatments on (features, controls) concatenated.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options

    model_regression: estimator, default ``'auto'``
        Estimator for E[Y | X, W, T]. Trained by regressing Y on (features, controls, one-hot-encoded treatments)
        concatenated. The one-hot-encoding excludes the baseline treatment.

        - If ``'auto'``, the model will be the best-fitting of a set of linear and forest models

        - Otherwise, see :ref:`model_selection` for the range of supported options;
          if a single model is specified it should be a classifier if `discrete_outcome` is True
          and a regressor otherwise

    featurizer : :term:`transformer`, optional
        Must support fit_transform and transform. Used to create composite features in the final CATE regression.
        It is ignored if X is None. The final CATE will be trained on the outcome of featurizer.fit_transform(X).
        If featurizer=None, then CATE is trained on X.

    min_propensity : float, default ``1e-6``
        The minimum propensity at which to clip propensity estimates to avoid dividing by zero.

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

    n_estimators : int, default 100
        The total number of trees in the forest. The forest consists of a
        forest of sqrt(n_estimators) sub-forests, where each sub-forest
        contains sqrt(n_estimators) trees.

    max_depth : int or None, optional
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, default 10
        The minimum number of splitting samples required to split an internal node.

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, default 5
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

    min_weight_fraction_leaf : float, default 0.
        The minimum weighted fraction of the sum total of weights (of all
        splitting samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided. After construction
        the tree is pruned so that the fraction of the sum total weight
        of the estimation samples contained in each leaf node is at
        least min_weight_fraction_leaf

    max_features : int, float, str, or None, default "auto"
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

    min_impurity_decrease : float, default 0.
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

    max_samples : int or float in (0, 1], default .5,
        The number of samples to use for each subsample that is used to train each tree:

        - If int, then train each tree on `max_samples` samples, sampled without replacement from all the samples
        - If float, then train each tree on ceil(`max_samples` * `n_samples`), sampled without replacement
          from all the samples.

    min_balancedness_tol: float in [0, .5], default .45
        How imbalanced a split we can tolerate. This enforces that each split leaves at least
        (.5 - min_balancedness_tol) fraction of samples on each side of the split; or fraction
        of the total weight of samples, when sample_weight is not None. Default value, ensures
        that at least 5% of the parent node weight falls in each side of the split. Set it to 0.0 for no
        balancedness and to .5 for perfectly balanced splits. For the formal inference theory
        to be valid, this has to be any positive constant bounded away from zero.

    honest : bool, default True
        Whether to use honest trees, i.e. half of the samples are used for
        creating the tree structure and the other half for the estimation at
        the leafs. If False, then all samples are used for both parts.

    n_jobs : int or None, default -1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``None`` means 1 unless in a :func:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default 0
        Controls the verbosity when fitting and predicting.

    random_state : int, RandomState instance, or None, default None

        If int, random_state is the seed used by the random number generator;
        If :class:`~numpy.random.mtrand.RandomState` instance, random_state is the random number generator;
        If None, the random number generator is the :class:`~numpy.random.mtrand.RandomState` instance used
        by :mod:`np.random<numpy.random>`.
    """

    def __init__(self, *,
                 model_regression="auto",
                 model_propensity="auto",
                 featurizer=None,
                 min_propensity=1e-6,
                 categories='auto',
                 cv=2,
                 mc_iters=None,
                 mc_agg='mean',
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=10,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.5,
                 min_balancedness_tol=.45,
                 honest=True,
                 n_jobs=-1,
                 verbose=0,
                 random_state=None):
        self.model_regression = clone(model_regression, safe=False)
        self.model_propensity = clone(model_propensity, safe=False)
        self.featurizer = clone(featurizer, safe=False)
        self.min_propensity = min_propensity
        self.categories = categories
        self.cv = cv
        self.mc_iters = mc_iters
        self.mc_agg = mc_agg
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
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    def _gen_drpolicy_learner(self):
        return _DRLearnerWrapper(model_regression=self.model_regression,
                                 model_propensity=self.model_propensity,
                                 featurizer=self.featurizer,
                                 min_propensity=self.min_propensity,
                                 categories=self.categories,
                                 cv=self.cv,
                                 mc_iters=self.mc_iters,
                                 mc_agg=self.mc_agg,
                                 model_final=PolicyForest(n_estimators=self.n_estimators,
                                                          max_depth=self.max_depth,
                                                          min_samples_split=self.min_samples_split,
                                                          min_samples_leaf=self.min_samples_leaf,
                                                          min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                          max_features=self.max_features,
                                                          min_impurity_decrease=self.min_impurity_decrease,
                                                          max_samples=self.max_samples,
                                                          min_balancedness_tol=self.min_balancedness_tol,
                                                          honest=self.honest,
                                                          n_jobs=self.n_jobs,
                                                          verbose=self.verbose,
                                                          random_state=self.random_state),
                                 multitask_model_final=True,
                                 random_state=self.random_state)

    def plot(self, tree_id, *, feature_names=None, treatment_names=None,
             ax=None, title=None,
             max_depth=None, filled=True, rounded=True, precision=3, fontsize=None):
        """
        Exports policy trees to matplotlib

        Parameters
        ----------
        tree_id : int
            The id of the tree of the forest to plot

        ax : :class:`matplotlib.axes.Axes`, optional
            The axes on which to plot

        title : str, optional
            A title for the final figure to be printed at the top of the page.

        feature_names : list of str, optional
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments, starting with a name for the baseline/control treatment
            (alphanumerically smallest)

        max_depth: int or None, optional
            The maximum tree depth to plot

        filled : bool, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        rounded : bool, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        precision : int, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.

        fontsize : int, optional
            Font size for text
        """
        return self.policy_model_[tree_id].plot(feature_names=self.policy_feature_names(feature_names=feature_names),
                                                treatment_names=self.policy_treatment_names(
                                                    treatment_names=treatment_names),
                                                ax=ax,
                                                title=title,
                                                max_depth=max_depth,
                                                filled=filled,
                                                rounded=rounded,
                                                precision=precision,
                                                fontsize=fontsize)

    def export_graphviz(self, tree_id, *, out_file=None, feature_names=None, treatment_names=None,
                        max_depth=None,
                        filled=True, leaves_parallel=True,
                        rotate=False, rounded=True, special_characters=False, precision=3):
        """
        Export a graphviz dot file representing the learned tree model

        Parameters
        ----------
        tree_id : int
            The id of the tree of the forest to plot

        out_file : file object or str, optional
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        feature_names : list of str, optional
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments, starting with a name for the baseline/control/None treatment
            (alphanumerically smallest in case of discrete treatment)

        max_depth: int or None, optional
            The maximum tree depth to plot

        filled : bool, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, default True
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, default False
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, default False
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        feature_names = self.policy_feature_names(feature_names=feature_names)
        return self.policy_model_[tree_id].export_graphviz(out_file=out_file,
                                                           feature_names=feature_names,
                                                           treatment_names=self.policy_treatment_names(
                                                               treatment_names=treatment_names),
                                                           max_depth=max_depth,
                                                           filled=filled,
                                                           leaves_parallel=leaves_parallel,
                                                           rotate=rotate,
                                                           rounded=rounded,
                                                           special_characters=special_characters,
                                                           precision=precision)

    def render(self, tree_id, out_file, *, format='pdf', view=True,
               feature_names=None,
               treatment_names=None,
               max_depth=None,
               filled=True, leaves_parallel=True, rotate=False, rounded=True,
               special_characters=False, precision=3):
        """
        Render the tree to a flie

        Parameters
        ----------
        tree_id : int
            The id of the tree of the forest to plot

        out_file : file name to save to

        format : str, default 'pdf'
            The file format to render to; must be supported by graphviz

        view : bool, default True
            Whether to open the rendered result with the default application.

        feature_names : list of str, optional
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments, starting with a name for the baseline/control treatment
            (alphanumerically smallest in case of discrete treatment)

        max_depth: int or None, optional
            The maximum tree depth to plot

        filled : bool, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, default True
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, default False
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, default False
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        feature_names = self.policy_feature_names(feature_names=feature_names)
        return self.policy_model_[tree_id].render(out_file,
                                                  feature_names=feature_names,
                                                  treatment_names=self.policy_treatment_names(
                                                      treatment_names=treatment_names),
                                                  format=format,
                                                  view=view,
                                                  max_depth=max_depth,
                                                  filled=filled,
                                                  leaves_parallel=leaves_parallel,
                                                  rotate=rotate,
                                                  rounded=rounded,
                                                  special_characters=special_characters,
                                                  precision=precision)
