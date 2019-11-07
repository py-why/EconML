import abc
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from .dot_exporter import _CATETreeDOTExporter, _CATETreeMPLExporter, _PolicyTreeDOTExporter, _PolicyTreeMPLExporter

class CateInterpreter(metaclass=abc.ABCMeta):
    
    @abc.abstractmethod
    def interpret(self, cate_estimator, X):
        pass
    
    @abc.abstractmethod
    def summary(self):
        pass
    
    @abc.abstractmethod
    def export(self):
        pass

class SingleTreeCateInterpreter:
    """
    include_uncertainty : bool, optional (default=False)
        Whether to include confidence interval information when building a
        simplified model of the cate model. If set to True, then
        cate estimator needs to support the `effect_interval` method.

    uncertainty_level : double, optional (default=.05)
        The uncertainty level for the confidence intervals to be constructed
        and used in the simplified model creation. If value=alpha
        then a multitask decision tree will be built such that all samples
        in a leaf have similar target prediction but also similar alpha
        confidence intervals.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
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
                 include_uncertainty=False,
                 uncertainty_level=.05,
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.):
        self.include_uncertainty = include_uncertainty
        self.uncertainty_level = uncertainty_level
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
        y_pred = cate_estimator.effect(X)
        
        assert (y_pred.ndim==1) or (y_pred.shape[1]==1), "Only single-dimensional treatment interpretation available!"

        if y_pred.ndim==1:
            y_pred = y_pred.reshape(-1, 1)

        if self.include_uncertainty:
            y_lower, y_upper = cate_estimator.effect_interval(X, alpha=self.uncertainty_level)
            if y_lower.ndim==1:
                y_lower = y_lower.reshape(-1, 1)
                y_upper = y_upper.reshape(-1, 1)
            y_pred = np.hstack([y_pred, y_lower, y_upper])
        self.tree_model.fit(X, y_pred)

        return self
    
    def export_graphviz(self, out_file=None, feature_names=None,
                        filled=True, leaves_parallel=False,
                        rotate=False, rounded=False, special_characters=False, precision=3):
        """
        Parameters
        ----------
        decision_tree : decision tree classifier
            The decision tree to be exported to GraphViz.

        out_file : file object or string, optional (default=None)
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        feature_names : list of strings, optional (default=None)
            Names of each of the features.

        filled : bool, optional (default=False)
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, optional (default=False)
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, optional (default=False)
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, optional (default=False)
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, optional (default=False)
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, optional (default=3)
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        check_is_fitted(self.tree_model, 'tree_')
        own_file = False
        return_string = False
        try:
            if isinstance(out_file, six.string_types):
                if six.PY3:
                    out_file = open(out_file, "w", encoding="utf-8")
                else:
                    out_file = open(out_file, "wb")
                own_file = True

            if out_file is None:
                return_string = True
                out_file = six.StringIO()

            exporter = _CATETreeDOTExporter(self.include_uncertainty, self.uncertainty_level,
                                            out_file=out_file, feature_names=feature_names, filled=filled,
                                            leaves_parallel=leaves_parallel, rotate=rotate, rounded=rounded,
                                            special_characters=special_characters, precision=precision)
            exporter.export(self.tree_model)

            if return_string:
                return exporter.out_file.getvalue()

        finally:
            if own_file:
                out_file.close()
    
    def render(self, out_file, format='pdf', view=True, feature_names=None,
                    filled=True, leaves_parallel=False,
                    rotate=False, rounded=False, special_characters=False, precision=3):
        import graphviz
        graphviz.Source(self.export_graphviz(feature_names=feature_names, filled=filled, rotate=rotate,
                                             leaves_parallel=leaves_parallel, rounded=rounded,
                                             special_characters=special_characters,
                                             precision=precision)).render(out_file, format=format, view=view)

    def plot(self, ax=None, title=None, feature_names=None,
                    filled=True,
                    rounded=False, precision=3, fontsize=None):
        exporter = _CATETreeMPLExporter(self.include_uncertainty, self.uncertainty_level,
                                            title=title, feature_names=feature_names, filled=filled,
                                            rounded=rounded,
                                            precision=precision, fontsize=fontsize)
        exporter.export(self.tree_model, ax=ax)

class SingleTreePolicyInterpreter:
    """
    risk_level : float or None,
        If None then the point estimate of the CATE of every point will be used as the
        effect of treatment. If any float alpha and risk_seeking=False (default), then the
        lower end point of an alpha confidence interval of the CATE will be used.
        Otherwise if risk_seeking=True, then the upper end of an alpha confidence interval
        will be used.

    risk_seeking : bool, optional (default=False),
        Whether to use an optimistic or pessimistic value for the effect estimate at a
        sample point. Used only when risk_level is not None.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, optional (default=0.)
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
    
    def interpret(self, cate_estimator, X, sample_treatment_costs=None):
        self.tree_model = DecisionTreeClassifier(criterion=self.criterion,
                                                splitter=self.splitter,
                                                max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                max_features=self.max_features,
                                                random_state=self.random_state,
                                                max_leaf_nodes=self.max_leaf_nodes,
                                                min_impurity_decrease=self.min_impurity_decrease)
        if self.risk_level is None:
            y_pred = cate_estimator.effect(X) 
        elif not self.risk_seeking:
            y_pred, _ = cate_estimator.effect_interval(X, alpha=self.risk_level)
        else:
            _, y_pred = cate_estimator.effect_interval(X, alpha=self.risk_level)

        if sample_treatment_costs is not None:
            y_pred -= sample_treatment_costs
        
        assert (y_pred.ndim==1) or (y_pred.shape[1]==1), "Only binary treatment interpretation available!"

        self.tree_model.fit(X, np.sign(y_pred).flatten(), sample_weight=np.abs(y_pred))
        self.policy_value = np.mean(y_pred * (self.tree_model.predict(X) == 1))
        self.always_treat_value = np.mean(y_pred)
        return self

    def export_graphviz(self, out_file=None, treatment_names=None, feature_names=None,
                        filled=True, leaves_parallel=False,
                        rotate=False, rounded=False, special_characters=False, precision=3):
        """
        Parameters
        ----------
        out_file : file object or string, optional (default=None)
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        treatment_names : list of strings, optional (default=None)
            Names of each of the treatments.

        feature_names : list of strings, optional (default=None)
            Names of each of the features.

        filled : bool, optional (default=False)
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        leaves_parallel : bool, optional (default=False)
            When set to ``True``, draw all leaf nodes at the bottom of the tree.

        rotate : bool, optional (default=False)
            When set to ``True``, orient tree left to right rather than top-down.

        rounded : bool, optional (default=False)
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        special_characters : bool, optional (default=False)
            When set to ``False``, ignore special characters for PostScript
            compatibility.

        precision : int, optional (default=3)
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.
        """
        check_is_fitted(self.tree_model, 'tree_')
        own_file = False
        return_string = False
        try:
            if isinstance(out_file, six.string_types):
                if six.PY3:
                    out_file = open(out_file, "w", encoding="utf-8")
                else:
                    out_file = open(out_file, "wb")
                own_file = True

            if out_file is None:
                return_string = True
                out_file = six.StringIO()

            title = "Average policy gains over no treatment: {} \n".format(np.around(self.policy_value, precision))
            title += "Average policy gains over always treating: {}".format(np.around(self.policy_value - self.always_treat_value, precision))
            exporter = _PolicyTreeDOTExporter(out_file=out_file, title=title,
                                              treatment_names=treatment_names, feature_names=feature_names, filled=filled,
                                              leaves_parallel=leaves_parallel, rotate=rotate, rounded=rounded,
                                              special_characters=special_characters, precision=precision)
            exporter.export(self.tree_model)

            if return_string:
                return exporter.out_file.getvalue()

        finally:
            if own_file:
                out_file.close()
    
    def render(self, out_file, format='pdf', view=True, feature_names=None, treatment_names=None,
                    filled=True, leaves_parallel=False,
                    rotate=False, rounded=False, special_characters=False, precision=3):
        import graphviz
        graphviz.Source(self.export_graphviz(treatment_names=treatment_names, feature_names=feature_names, filled=filled, rotate=rotate,
                                             leaves_parallel=leaves_parallel, rounded=rounded,
                                             special_characters=special_characters,
                                             precision=precision)).render(out_file, format=format, view=view)

    def plot(self, ax=None, title=None, treatment_names=None, feature_names=None,
                    filled=True,
                    rounded=False, precision=3, fontsize=None):
        title = "" if title is None else title
        title += "Average policy gains over no treatment: {} \n".format(np.around(self.policy_value, precision))
        title += "Average policy gains over always treating: {}".format(np.around(self.policy_value - self.always_treat_value, precision))
        exporter = _PolicyTreeMPLExporter(treatment_names=treatment_names, title=title, feature_names=feature_names, filled=filled,
                                            rounded=rounded,
                                            precision=precision, fontsize=fontsize)
        exporter.export(self.tree_model, ax=ax)

