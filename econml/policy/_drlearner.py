# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from warnings import warn
import numpy as np
from sklearn.base import clone
from ..utilities import check_inputs, filter_none_kwargs
from ..drlearner import _ModelFinal, DRLearner
from .._tree_exporter import _SingleTreeExporterMixin
from . import PolicyTree, PolicyForest


class _PolicyModelFinal(_ModelFinal):

    def fit(self, Y, T, X=None, W=None, *, nuisances, sample_weight=None, sample_var=None):
        if sample_var is not None:
            warn('Parameter `sample_var` is ignored by the final estimator')
            sample_var = None
        Y_pred, = nuisances
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

    def score(self, Y, T, X=None, W=None, *, nuisances, sample_weight=None, sample_var=None):
        return 0


class _DRLearnerWrapper(DRLearner):

    def _gen_ortho_learner_model_final(self):
        return _PolicyModelFinal(self._gen_model_final(), self._gen_featurizer(), self.multitask_model_final)


class _BaseDRPolicyLearner():

    def fit(self, Y, T, X=None, W=None, *, sample_weight=None, groups=None):
        Y, T, X, W = check_inputs(Y, T, X, W=W, multi_output_T=True, multi_output_Y=False)
        self.drlearner_ = self._gen_drpolicy_learner()
        self.drlearner_.fit(Y, T, X=X, W=W, sample_weight=sample_weight, groups=groups)
        return self

    def predict_value(self, X):
        return self.drlearner_.const_marginal_effect(X)

    def predict(self, X):
        values = self.predict_value(X)
        return np.argmax(np.hstack([np.zeros((values.shape[0], 1)), values]), axis=1)

    def policy_feature_names(self, *, feature_names=None):
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
            The names of the output features on which the policy model is fitted.
        """
        return self.drlearner_.cate_feature_names(feature_names=feature_names)

    def feature_importances(self, max_depth=4, depth_decay_exponent=2.0):
        """

        Parameters
        ----------
        max_depth : int, default=4
            Splits of depth larger than `max_depth` are not used in this calculation
        depth_decay_exponent: double, default=2.0
            The contribution of each split to the total score is re-weighted by ``1 / (1 + `depth`)**2.0``.

        Returns
        -------
        feature_importances_ : ndarray of shape (n_features,)
            Normalized total parameter heterogeneity inducing importance of each feature
        """
        return self.policy_model.feature_importances(max_depth=max_depth,
                                                     depth_decay_exponent=depth_decay_exponent)

    @property
    def feature_importances_(self):
        return self.feature_importances()

    @property
    def policy_model(self):
        return self.drlearner_.multitask_model_cate


class DRPolicyTree(_BaseDRPolicyLearner):
    """ TODO Enable inference on `predict_value` with leaf-wise normality
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
                 min_samples_split=5,
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

    def plot(self, *, feature_names=None, treatment_names=None, **kwargs):
        return self.policy_model.plot(feature_names=self.policy_feature_names(feature_names=feature_names),
                                      treatment_names=treatment_names,
                                      **kwargs)
    plot.__doc__ = _SingleTreeExporterMixin.plot.__doc__

    def export_graphviz(self, *, feature_names=None, treatment_names=None, **kwargs):
        return self.policy_model.export_graphviz(feature_names=self.policy_feature_names(feature_names=feature_names),
                                                 treatment_names=treatment_names,
                                                 **kwargs)
    export_graphviz.__doc__ = _SingleTreeExporterMixin.export_graphviz.__doc__

    def render(self, out_file, *, feature_names=None, treatment_names=None, **kwargs):
        return self.policy_model.render(out_file,
                                        feature_names=self.policy_feature_names(feature_names=feature_names),
                                        treatment_names=treatment_names,
                                        **kwargs)
    render.__doc__ = _SingleTreeExporterMixin.render.__doc__


class DRPolicyForest(_BaseDRPolicyLearner):
    """ TODO Enable inference on `predict_value` with BLB
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
                 n_estimators=1000,
                 max_depth=None,
                 min_samples_split=5,
                 min_samples_leaf=5,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 min_impurity_decrease=0.,
                 max_samples=.45,
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
                                 model_final=PolicyForest(max_depth=self.max_depth,
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

    def plot(self, tree_id, *, feature_names=None, treatment_names=None, **kwargs):
        """
        Exports policy trees to matplotlib

        Parameters
        ----------
        tree_id : int
            The id of the tree of the forest to plot

        ax : :class:`matplotlib.axes.Axes`, optional, default None
            The axes on which to plot

        title : string, optional, default None
            A title for the final figure to be printed at the top of the page.

        feature_names : list of strings, optional, default None
            Names of each of the features.

        treatment_names : list of strings, optional, default None
            Names of each of the treatments, starting with a name for the baseline/control treatment
            (alphanumerically smallest in case of discrete treatment or the all-zero treatment
            in the case of continuous)

        filled : bool, optional, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        rounded : bool, optional, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        precision : int, optional, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.

        fontsize : int, optional, default None
            Font size for text
        """
        return self.policy_model[tree_id].plot(feature_names=self.policy_feature_names(feature_names=feature_names),
                                               treatment_names=treatment_names,
                                               **kwargs)

    def export_graphviz(self, tree_id, *, feature_names=None, treatment_names=None, **kwargs):
        """
        Export a graphviz dot file representing the learned tree model

        Parameters
        ----------
        tree_id : int
            The id of the tree of the forest to plot

        out_file : file object or string, optional, default None
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        feature_names : list of strings, optional, default None
            Names of each of the features.

        treatment_names : list of strings, optional, default None
            Names of each of the treatments, starting with a name for the baseline/control treatment
            (alphanumerically smallest in case of discrete treatment or the all-zero treatment
            in the case of continuous)

        max_depth: int or None, optional, default None
            The maximum tree depth to plot

        filled : bool, optional, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, optional, default True
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, optional, default False
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, optional, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, optional, default False
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, optional, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        feature_names = self.policy_feature_names(feature_names=feature_names)
        return self.policy_model[tree_id].export_graphviz(feature_names=feature_names,
                                                          treatment_names=treatment_names,
                                                          **kwargs)

    def render(self, tree_id, out_file, *, feature_names=None, treatment_names=None, **kwargs):
        """
        Render the tree to a flie

        Parameters
        ----------
        tree_id : int
            The id of the tree of the forest to plot

        out_file : file name to save to

        format : string, optional, default 'pdf'
            The file format to render to; must be supported by graphviz

        view : bool, optional, default True
            Whether to open the rendered result with the default application.

        feature_names : list of strings, optional, default None
            Names of each of the features.

        treatment_names : list of strings, optional, default None
            Names of each of the treatments, starting with a name for the baseline/control treatment
            (alphanumerically smallest in case of discrete treatment or the all-zero treatment
            in the case of continuous)

        max_depth: int or None, optional, default None
            The maximum tree depth to plot

        filled : bool, optional, default False
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, optional, default True
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, optional, default False
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, optional, default True
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, optional, default False
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, optional, default 3
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        feature_names = self.policy_feature_names(feature_names=feature_names)
        return self.policy_model[tree_id].render(out_file,
                                                 feature_names=feature_names,
                                                 treatment_names=treatment_names,
                                                 **kwargs)
