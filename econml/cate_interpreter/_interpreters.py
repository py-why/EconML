# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import abc
import numpy as np
from io import StringIO
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
import graphviz
from ._tree_exporter import _CateTreeDOTExporter, _CateTreeMPLExporter, _PolicyTreeDOTExporter, _PolicyTreeMPLExporter


class _SingleTreeInterpreter(metaclass=abc.ABCMeta):

    tree_model = None
    node_dict = None

    @abc.abstractmethod
    def interpret(self, cate_estimator, X):
        """
        Interpret a linear CATE estimator when applied to a set of features

        Parameters
        ----------
        cate_estimator : :class:`.LinearCateEstimator`
            The fitted estimator to interpret

        X : array-like
            The features against which to interpret the estimator;
            must be compatible shape-wise with the features used to fit
            the estimator
        """
        pass

    @abc.abstractmethod
    def _make_dot_exporter(self, *, out_file, feature_names, filled,
                           leaves_parallel, rotate, rounded,
                           special_characters, precision):
        """
        Make a dot file exporter

        Parameters
        ----------
        out_file : file object
            Handle to write to.

        feature_names : list of strings
            Names of each of the features.

        filled : bool
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        pass

    @abc.abstractmethod
    def _make_mpl_exporter(self, *, title=None, feature_names=None,
                           filled=True, rounded=True, precision=3, fontsize=None):
        """
        Make a matplotlib exporter

        Parameters
        ----------
        title : string
            A title for the final figure to be printed at the top of the page.

        feature_names : list of strings
            Names of each of the features.

        filled : bool
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        rounded : bool
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        precision : int
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.

        fontsize : int
            Fontsize for text
        """
        pass

    def export_graphviz(self, out_file=None, feature_names=None,
                        filled=True, leaves_parallel=True,
                        rotate=False, rounded=True, special_characters=False, precision=3):
        """
        Export a graphviz dot file representing the learned tree model

        Parameters
        ----------
        out_file : file object or string, optional, default None
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        feature_names : list of strings, optional, default None
            Names of each of the features.

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

        check_is_fitted(self.tree_model, 'tree_')
        own_file = False
        try:
            if isinstance(out_file, str):
                out_file = open(out_file, "w", encoding="utf-8")
                own_file = True

            return_string = out_file is None
            if return_string:
                out_file = StringIO()

            exporter = self._make_dot_exporter(out_file=out_file, feature_names=feature_names, filled=filled,
                                               leaves_parallel=leaves_parallel, rotate=rotate, rounded=rounded,
                                               special_characters=special_characters, precision=precision)
            exporter.export(self.tree_model, node_dict=self.node_dict)

            if return_string:
                return out_file.getvalue()

        finally:
            if own_file:
                out_file.close()

    def render(self, out_file, format='pdf', view=True, feature_names=None,
               filled=True, leaves_parallel=True, rotate=False, rounded=True,
               special_characters=False, precision=3):
        """
        Render the tree to a flie

        Parameters
        ----------
        out_file : file name to save to

        format : string, optional, default 'pdf'
            The file format to render to; must be supported by graphviz

        view : bool, optional, default True
            Whether to open the rendered result with the default application.

        feature_names : list of strings, optional, default None
            Names of each of the features.

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
        dot_source = self.export_graphviz(out_file=None,  # want the output as a string, only write the final file
                                          feature_names=feature_names, filled=filled,
                                          leaves_parallel=leaves_parallel, rotate=rotate,
                                          rounded=rounded, special_characters=special_characters,
                                          precision=precision)
        graphviz.Source(dot_source).render(out_file, format=format, view=view)

    def plot(self, ax=None, title=None, feature_names=None,
             filled=True, rounded=True, precision=3, fontsize=None):
        """
        Exports policy trees to matplotlib

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional, default None
            The axes on which to plot

        title : string, optional, default None
            A title for the final figure to be printed at the top of the page.

        feature_names : list of strings, optional, default None
            Names of each of the features.

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
        check_is_fitted(self.tree_model, 'tree_')
        exporter = self._make_mpl_exporter(title=title, feature_names=feature_names, filled=filled,
                                           rounded=rounded, precision=precision, fontsize=fontsize)
        exporter.export(self.tree_model, node_dict=self.node_dict, ax=ax)


class SingleTreeCateInterpreter(_SingleTreeInterpreter):
    """
    An interpreter for the effect estimated by a CATE estimator

    Parameters
    ----------
    include_uncertainty : bool, optional, default False
        Whether to include confidence interval information when building a
        simplified model of the cate model. If set to True, then
        cate estimator needs to support the `const_marginal_ate_inference` method.

    uncertainty_level : double, optional, default .05
        The uncertainty level for the confidence intervals to be constructed
        and used in the simplified model creation. If value=alpha
        then a multitask decision tree will be built such that all samples
        in a leaf have similar target prediction but also similar alpha
        confidence intervals.

    uncertainty_only_on_leaves : bool, optional, default True
        Whether uncertainty information should be displayed only on leaf nodes.
        If False, then interpretation can be slightly slower, especially for cate
        models that have a computationally expensive inference method.

    splitter : string, optional, default "best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional, default None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional, default 2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional, default 1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional, default 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional, default None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional, default 0.
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
    """

    def __init__(self,
                 include_model_uncertainty=False,
                 uncertainty_level=.1,
                 uncertainty_only_on_leaves=True,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.):
        self.include_uncertainty = include_model_uncertainty
        self.uncertainty_level = uncertainty_level
        self.uncertainty_only_on_leaves = uncertainty_only_on_leaves
        self.criterion = "mse"
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    def interpret(self, cate_estimator, X):
        self.tree_model = DecisionTreeRegressor(criterion=self.criterion,
                                                splitter=self.splitter,
                                                max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                min_samples_leaf=self.min_samples_leaf,
                                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                max_features=self.max_features,
                                                random_state=self.random_state,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                min_impurity_decrease=self.min_impurity_decrease)
        y_pred = cate_estimator.const_marginal_effect(X)

        self.tree_model.fit(X, y_pred.reshape((y_pred.shape[0], -1)))
        paths = self.tree_model.decision_path(X)
        node_dict = {}
        for node_id in range(paths.shape[1]):
            mask = paths.getcol(node_id).toarray().flatten().astype(bool)
            Xsub = X[mask]
            if (self.include_uncertainty and
                    ((not self.uncertainty_only_on_leaves) or (self.tree_model.tree_.children_left[node_id] < 0))):
                res = cate_estimator.const_marginal_ate_inference(Xsub)
                node_dict[node_id] = {'mean': res.mean_point,
                                      'std': res.std_point,
                                      'ci': res.conf_int_mean(alpha=self.uncertainty_level)}
            else:
                cate_node = y_pred[mask]
                node_dict[node_id] = {'mean': np.mean(cate_node, axis=0),
                                      'std': np.std(cate_node, axis=0)}
        self.node_dict = node_dict
        return self

    def _make_dot_exporter(self, *, out_file, feature_names, filled,
                           leaves_parallel, rotate, rounded,
                           special_characters, precision):
        return _CateTreeDOTExporter(self.include_uncertainty, self.uncertainty_level,
                                    out_file=out_file, feature_names=feature_names, filled=filled,
                                    leaves_parallel=leaves_parallel, rotate=rotate, rounded=rounded,
                                    special_characters=special_characters, precision=precision)

    def _make_mpl_exporter(self, *, title, feature_names,
                           filled,
                           rounded, precision, fontsize):
        return _CateTreeMPLExporter(self.include_uncertainty, self.uncertainty_level,
                                    title=title, feature_names=feature_names, filled=filled,
                                    rounded=rounded,
                                    precision=precision, fontsize=fontsize)


class SingleTreePolicyInterpreter(_SingleTreeInterpreter):
    """
    An interpreter for a policy estimated based on a CATE estimation

    Parameters
    ----------
    risk_level : float or None,
        If None then the point estimate of the CATE of every point will be used as the
        effect of treatment. If any float alpha and risk_seeking=False (default), then the
        lower end point of an alpha confidence interval of the CATE will be used.
        Otherwise if risk_seeking=True, then the upper end of an alpha confidence interval
        will be used.

    risk_seeking : bool, optional, default False,
        Whether to use an optimistic or pessimistic value for the effect estimate at a
        sample point. Used only when risk_level is not None.

    splitter : string, optional, default "best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional, default None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional, default 2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional, default 1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional, default 0.
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    random_state : int, RandomState instance or None, optional, default None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional, default None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional, default 0.
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

    Attributes
    ----------
    tree_model : :class:`~sklearn.tree.DecisionTreeClassifier`
        The classifier that determines whether units should be treated; available only after
        :meth:`interpret` has been called.
    policy_value : float
        The value of applying the learned policy, applied to the sample used with :meth:`interpret`
    always_treat_value : float
        The value of the policy that always treats all units, applied to the sample used with :meth:`interpret`
    treatment_names : list
        The list of treatment names that were passed to :meth:`interpret`
    """

    def __init__(self,
                 risk_level=None,
                 risk_seeking=False,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.):
        self.risk_level = risk_level
        self.risk_seeking = risk_seeking
        self.criterion = "gini"
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

    def interpret(self, cate_estimator, X, sample_treatment_costs=None, treatment_names=None):
        """
        Interpret a policy based on a linear CATE estimator when applied to a set of features

        Parameters
        ----------
        cate_estimator : :class:`.LinearCateEstimator`
            The fitted estimator to interpret

        X : array-like
            The features against which to interpret the estimator;
            must be compatible shape-wise with the features used to fit
            the estimator

        sample_treatment_costs : array-like, optional
            The cost of treatment.  Can be a scalar or a variable cost with the same number of rows as ``X``

        treatment_names : list of string, optional
            The names of the two treatments
        """
        self.tree_model = DecisionTreeClassifier(criterion=self.criterion,
                                                 splitter=self.splitter,
                                                 max_depth=self.max_depth,
                                                 min_samples_split=self.min_samples_split,
                                                 min_samples_leaf=self.min_samples_leaf,
                                                 min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                 max_features=self.max_features,
                                                 random_state=self.random_state,
                                                 max_leaf_nodes=self.max_leaf_nodes,
                                                 min_impurity_decrease=self.min_impurity_decrease)
        if self.risk_level is None:
            y_pred = cate_estimator.const_marginal_effect(X)
        elif not self.risk_seeking:
            y_pred, _ = cate_estimator.const_marginal_effect_interval(X, alpha=self.risk_level)
        else:
            _, y_pred = cate_estimator.const_marginal_effect_interval(X, alpha=self.risk_level)

        # TODO: generalize to multiple treatment case?
        assert all(d == 1 for d in y_pred.shape[1:]), ("Interpretation is only available for "
                                                       "single-dimensional treatments and outcomes")

        y_pred = y_pred.ravel()

        if sample_treatment_costs is not None:
            assert np.ndim(sample_treatment_costs) < 2, "Sample treatment costs should be a vector or scalar"
            y_pred -= sample_treatment_costs

        # get index of best treatment
        all_y = np.hstack([np.zeros((y_pred.shape[0], 1)), y_pred.reshape(-1, 1)])
        best_y = np.argmax(all_y, axis=-1)

        used_t = np.unique(best_y)
        if len(used_t) == 1:
            best_y, = used_t
            if best_y > 0:
                raise AttributeError("All samples should be treated with the given treatment costs. " +
                                     "Consider increasing the cost!")
            else:
                raise AttributeError("No samples should be treated with the given treatment costs. " +
                                     "Consider decreasing the cost!")

        self.tree_model.fit(X, best_y, sample_weight=np.abs(y_pred))
        self.policy_value = np.mean(all_y[:, self.tree_model.predict(X)])
        self.always_treat_value = np.mean(y_pred)
        self.treatment_names = treatment_names
        return self

    def treat(self, X):
        """
        Using the policy model learned by a call to :meth:`interpret`, assign treatment to a set of units

        Parameters
        ----------
        X : array-like
            The features for the units to treat;
            must be compatible shape-wise with the features used during interpretation

        Returns
        -------
        T : array-like
            The treatments implied by the policy learned by the interpreter
        """
        assert self.tree_model is not None, "Interpret must be called prior to trying to assign treatment."
        return self.tree_model.predict(X)

    def _make_dot_exporter(self, *, out_file, feature_names, filled,
                           leaves_parallel, rotate, rounded,
                           special_characters, precision):
        title = "Average policy gains over no treatment: {} \n".format(np.around(self.policy_value, precision))
        title += "Average policy gains over always treating: {}".format(
            np.around(self.policy_value - self.always_treat_value, precision))
        return _PolicyTreeDOTExporter(out_file=out_file, title=title,
                                      treatment_names=self.treatment_names, feature_names=feature_names,
                                      filled=filled, leaves_parallel=leaves_parallel, rotate=rotate,
                                      rounded=rounded, special_characters=special_characters,
                                      precision=precision)

    def _make_mpl_exporter(self, *, title, feature_names, filled,
                           rounded, precision, fontsize):
        title = "" if title is None else title
        title += "Average policy gains over no treatment: {} \n".format(np.around(self.policy_value, precision))
        title += "Average policy gains over always treating: {}".format(
            np.around(self.policy_value - self.always_treat_value, precision))
        return _PolicyTreeMPLExporter(treatment_names=self.treatment_names, title=title,
                                      feature_names=feature_names, filled=filled,
                                      rounded=rounded,
                                      precision=precision, fontsize=fontsize)
