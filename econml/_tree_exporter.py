# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# HACK: We're relying on some of sklearn's non-public classes which are not completely stable.
#       However, the alternative is reimplementing a bunch of intricate stuff by hand
from sklearn.tree import _tree
try:
    from sklearn.tree._export import _BaseTreeExporter, _MPLTreeExporter, _DOTTreeExporter
except ImportError:  # prior to sklearn 0.22.0, the ``export`` submodule was public
    from sklearn.tree.export import _BaseTreeExporter, _MPLTreeExporter, _DOTTreeExporter


class _TreeExporter(_BaseTreeExporter):
    """
    Tree exporter that supports replacing the "value" part of each node's text with something customized
    """

    def node_replacement_text(self, tree, node_id, criterion):
        return None

    def node_to_str(self, tree, node_id, criterion):
        text = super().node_to_str(tree, node_id, criterion)
        replacement = self.node_replacement_text(tree, node_id, criterion)
        if replacement is not None:
            # HACK: it's not optimal to use a regex like this, but the base class's node_to_str doesn't expose any
            #       clean way of achieving this
            text = re.sub("value = .*(?=" + re.escape(self.characters[5]) + ")",
                          # make sure we don't accidentally escape anything in the substitution
                          replacement.replace('\\', '\\\\'),
                          text,
                          flags=re.S)
        return text


class _MPLExporter(_MPLTreeExporter):
    """
    Base class that supports adding a title to an MPL tree exporter
    """

    def __init__(self, *args, title=None, **kwargs):
        self.title = title
        super().__init__(*args, **kwargs)

    def export(self, decision_tree, ax=None):
        if ax is None:
            ax = plt.gca()
        anns = super().export(decision_tree, ax=ax)
        if self.title is not None:
            ax.set_title(self.title)
        return anns


class _DOTExporter(_DOTTreeExporter):
    """
    Base class that supports adding a title to a DOT tree exporter
    """

    def __init__(self, *args, title=None, **kwargs):
        self.title = title
        super().__init__(*args, **kwargs)

    def tail(self):
        if self.title is not None:
            self.out_file.write("labelloc=\"t\"; \n")
            self.out_file.write("label=\"{}\"; \n".format(self.title))
        super().tail()


class _CateTreeMixin(_TreeExporter):
    """
    Mixin that supports writing out the nodes of a CATE tree
    """

    def __init__(self, include_uncertainty=False, uncertainty_level=0.1, *args, **kwargs):
        self.include_uncertainty = include_uncertainty
        self.uncertainty_level = uncertainty_level
        super().__init__(*args, **kwargs)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            # red for negative, green for positive
            self.colors['rgb'] = [(179, 108, 96), (81, 157, 96)]

        # in multi-target use first target
        tree_min = np.min(tree.value, axis=0, keepdims=True)[(0,) * tree.value.ndim]
        tree_max = np.max(tree.value, axis=0, keepdims=True)[(0,) * tree.value.ndim]

        node_val = tree.value[(node_id,) + (0,) * (tree.value.ndim - 1)]

        if node_val > 0:
            value = [max(0, tree_min) / tree_max, node_val / tree_max]
        else:
            value = [node_val / tree_min, min(0, tree_max) / tree_min]

        return self.get_color(value)

    def node_replacement_text(self, tree, node_id, criterion):
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        # Write node mean CATE
        node_string = 'CATE mean = '
        value_text = np.array2string(value[0, 0] if self.include_uncertainty else value[0], precision=self.precision)
        node_string += value_text + self.characters[4]

        # Write node std of CATE
        node_string += "CATE std = "
        value_text = np.array2string(np.sqrt(np.clip(tree.impurity[node_id], 0, np.inf)), precision=self.precision)
        node_string += value_text + self.characters[4]

        # Write confidence interval information if at leaf node
        if (tree.children_left[node_id] == _tree.TREE_LEAF) and self.include_uncertainty:
            ci_text = "Mean Endpoints of {}% CI: ({}, {})".format(int((1 - self.uncertainty_level) * 100),
                                                                  np.around(value[1, 0], self.precision),
                                                                  np.around(value[2, 0], self.precision))
            node_string += ci_text + self.characters[4]

        return node_string


class _PolicyTreeMixin(_TreeExporter):
    """
    Mixin that supports writing out the nodes of a policy tree

    Parameters
    ----------
    treatment_names : list of strings, optional, default None
        The names of the two treatments
    """

    def __init__(self, treatment_names=None, *args, **kwargs):
        self.treatment_names = treatment_names
        super().__init__(*args, **kwargs)

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            # red for negative, green for positive
            self.colors['rgb'] = [(179, 108, 96), (81, 157, 96)]

        node_val = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
        return self.get_color(node_val)

    def node_replacement_text(self, tree, node_id, criterion):
        value = tree.value[node_id][0, :]
        node_string = '(effect - cost) mean = %s' % np.round((value[1] -
                                                              value[0]) / tree.n_node_samples[node_id],
                                                             self.precision)
        node_string += self.characters[4]

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # Write node mean CATE
            node_string += 'Recommended Treatment = '
            if self.treatment_names:
                class_name = self.treatment_names[np.argmax(value)]
            else:
                class_name = "T%s%s%s" % (self.characters[1],
                                          np.argmax(value),
                                          self.characters[2])
            node_string += class_name + self.characters[4]

        return node_string


class _PolicyTreeMPLExporter(_PolicyTreeMixin, _MPLExporter):
    """
    Exports policy trees to matplotlib

    Parameters
    ----------
    treatment_names : list of strings, optional, default None
        The names of the two treatments

    title : string, optional, default None
        A title for the final figure to be printed at the top of the page.

    feature_names : list of strings, optional, default None
        Names of each of the features.

    filled : bool, optional, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    rounded : bool, optional, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, optional, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    fontsize : int, optional, default None
        Fontsize for text
    """

    def __init__(self, treatment_names=None, title=None, feature_names=None,
                 filled=True,
                 rounded=False, precision=3, fontsize=None):
        super().__init__(treatment_names=treatment_names, title=title,
                         feature_names=feature_names, filled=filled, rounded=rounded, precision=precision,
                         fontsize=fontsize,
                         impurity=False)


class _CateTreeMPLExporter(_CateTreeMixin, _MPLExporter):
    """
    Exports CATE trees into matplotlib

    Parameters
    ----------
    include_uncertainty: bool
        whether the tree includes uncertainty information

    uncertainty_level: float
        the confidence level of the confidence interval included in the tree

    title : string, optional, default None
        A title for the final figure to be printed at the top of the page.

    feature_names : list of strings, optional, default None
        Names of each of the features.

    filled : bool, optional, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    rounded : bool, optional, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, optional, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    fontsize : int, optional, default None
        Fontsize for text
    """

    def __init__(self, include_uncertainty, uncertainty_level, title=None,
                 feature_names=None, filled=True, rounded=False, precision=3, fontsize=None):
        super().__init__(include_uncertainty, uncertainty_level, title=None,
                         feature_names=feature_names, filled=filled,
                         rounded=rounded, precision=precision, fontsize=fontsize,
                         impurity=False)


class _PolicyTreeDOTExporter(_PolicyTreeMixin, _DOTExporter):
    """
    Exports policy trees to dot files

    Parameters
    ----------
    out_file : file object or string, optional, default None
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

    title : string, optional, default None
        A title for the final figure to be printed at the top of the page.

    treatment_names : list of strings, optional, default None
        The names of the two treatments

    feature_names : list of strings, optional, default None
        Names of each of the features.

    filled : bool, optional, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, optional, default False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    rotate : bool, optional, default False
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, optional, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    special_characters : bool, optional, default False
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    precision : int, optional, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.
    """

    def __init__(self, out_file=None, title=None, treatment_names=None, feature_names=None,
                 filled=True, leaves_parallel=False,
                 rotate=False, rounded=False, special_characters=False, precision=3):
        super().__init__(title=title, out_file=out_file, feature_names=feature_names,
                         filled=filled, leaves_parallel=leaves_parallel,
                         rotate=rotate, rounded=rounded, special_characters=special_characters,
                         precision=precision, treatment_names=treatment_names,
                         impurity=False)


class _CateTreeDOTExporter(_CateTreeMixin, _DOTExporter):
    """
    Exports CATE trees to dot files

    Parameters
    ----------
    include_uncertainty: bool
        whether the tree includes uncertainty information

    uncertainty_level: float
        the confidence level of the confidence interval included in the tree

    out_file : file object or string, optional, default None
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

    title : string, optional, default None
        A title for the final figure to be printed at the top of the page.

    feature_names : list of strings, optional, default None
        Names of each of the features.

    filled : bool, optional, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, optional, default False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    rotate : bool, optional, default False
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, optional, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    special_characters : bool, optional, default False
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    precision : int, optional, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.
    """

    def __init__(self, include_uncertainty, uncertainty_level, out_file=None, title=None, feature_names=None,
                 filled=True, leaves_parallel=False,
                 rotate=False, rounded=False, special_characters=False, precision=3):

        super().__init__(include_uncertainty, uncertainty_level,
                         out_file=out_file, title=title, feature_names=feature_names,
                         filled=filled, leaves_parallel=leaves_parallel,
                         rotate=rotate, rounded=rounded, special_characters=special_characters,
                         precision=precision,
                         impurity=False)
