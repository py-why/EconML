# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
#
# This code contains some snippets of code from:
# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_export.py
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

import abc
import numpy as np
import re
from io import StringIO
from sklearn.utils.validation import check_is_fitted

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as exn:
    from .utilities import MissingModule

    # make any access to matplotlib or plt throw an exception
    matplotlib = plt = MissingModule("matplotlib is no longer a dependency of the main econml package; "
                                     "install econml[plt] or econml[all] to require it, or install matplotlib "
                                     "separately, to use the tree interpreters", exn)

try:
    import graphviz
except ImportError as exn:
    from .utilities import MissingModule

    # make any access to graphviz or plt throw an exception
    graphviz = MissingModule("graphviz is no longer a dependency of the main econml package; "
                             "install econml[plt] or econml[all] to require it, or install graphviz "
                             "separately, to use the tree interpreters", exn)

# HACK: We're relying on some of sklearn's non-public classes which are not completely stable.
#       However, the alternative is reimplementing a bunch of intricate stuff by hand
from sklearn.tree import _tree
try:
    from sklearn.tree._export import _BaseTreeExporter, _MPLTreeExporter, _DOTTreeExporter
except ImportError:  # prior to sklearn 0.22.0, the ``export`` submodule was public
    from sklearn.tree.export import _BaseTreeExporter, _MPLTreeExporter, _DOTTreeExporter


def _color_brew(n):
    """Generate n colors with equally spaced hues.
    Parameters
    ----------
    n : int
        The number of colors required.
    Returns
    -------
    color_list : list, length n
        List of n tuples of form (R, G, B) being the components of each color.
    """
    color_list = []

    # Initialize saturation & value; calculate chroma & value shift
    s, v = 0.75, 0.9
    c = s * v
    m = v - c

    for h in np.arange(25, 385, 360. / n).astype(int):
        # Calculate some intermediate values
        h_bar = h / 60.
        x = c * (1 - abs((h_bar % 2) - 1))
        # Initialize RGB with same hue & chroma as our color
        rgb = [(c, x, 0),
               (x, c, 0),
               (0, c, x),
               (0, x, c),
               (x, 0, c),
               (c, 0, x),
               (c, x, 0)]
        r, g, b = rgb[int(h_bar)]
        # Shift the initial RGB values to match value and store
        rgb = [(int(255 * (r + m))),
               (int(255 * (g + m))),
               (int(255 * (b + m)))]
        color_list.append(rgb)

    return color_list


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

    def export(self, decision_tree, node_dict=None, ax=None):
        if ax is None:
            ax = plt.gca()
        self.node_dict = node_dict
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

    def export(self, decision_tree, node_dict=None):
        self.node_dict = node_dict
        return super().export(decision_tree)

    def tail(self):
        if self.title is not None:
            self.out_file.write("labelloc=\"t\"; \n")
            self.out_file.write("label=\"{}\"; \n".format(self.title))
        super().tail()


class _CateTreeMixin(_TreeExporter):
    """
    Mixin that supports writing out the nodes of a CATE tree
    """

    def __init__(self, include_uncertainty=False, uncertainty_level=0.1,
                 *args, treatment_names=None, **kwargs):
        self.include_uncertainty = include_uncertainty
        self.uncertainty_level = uncertainty_level
        self.treatment_names = treatment_names
        super().__init__(*args, **kwargs)

    def get_fill_color(self, tree, node_id):

        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            # red for negative, green for positive
            self.colors['rgb'] = [(179, 108, 96), (81, 157, 96)]

        # in multi-target use mean of targets
        tree_min = np.min(np.mean(tree.value, axis=1)) - 1e-12
        tree_max = np.max(np.mean(tree.value, axis=1)) + 1e-12

        node_val = np.mean(tree.value[node_id])

        if node_val > 0:
            value = [max(0, tree_min) / tree_max, node_val / tree_max]
        elif node_val < 0:
            value = [node_val / tree_min, min(0, tree_max) / tree_min]
        else:
            value = [0, 0]

        return self.get_color(value)

    def node_replacement_text(self, tree, node_id, criterion):

        # Write node mean CATE
        node_info = self.node_dict[node_id]
        node_string = 'CATE mean' + self.characters[4]
        value_text = ""
        mean = node_info['mean']
        if hasattr(mean, 'shape') and (len(mean.shape) > 0):
            if len(mean.shape) == 1:
                for i in range(mean.shape[0]):
                    value_text += "{}".format(np.around(mean[i], self.precision))
                    if 'ci' in node_info:
                        value_text += " ({}, {})".format(np.around(node_info['ci'][0][i], self.precision),
                                                         np.around(node_info['ci'][1][i], self.precision))
                    if i != mean.shape[0] - 1:
                        value_text += ", "
                value_text += self.characters[4]
            elif len(mean.shape) == 2:
                for i in range(mean.shape[0]):
                    for j in range(mean.shape[1]):
                        value_text += "{}".format(np.around(mean[i, j], self.precision))
                        if 'ci' in node_info:
                            value_text += " ({}, {})".format(np.around(node_info['ci'][0][i, j], self.precision),
                                                             np.around(node_info['ci'][1][i, j], self.precision))
                        if j != mean.shape[1] - 1:
                            value_text += ", "
                    value_text += self.characters[4]
            else:
                raise ValueError("can only handle up to 2d values")
        else:
            value_text += "{}".format(np.around(mean, self.precision))
            if 'ci' in node_info:
                value_text += " ({}, {})".format(np.around(node_info['ci'][0], self.precision),
                                                 np.around(node_info['ci'][1], self.precision))
            value_text += self.characters[4]
        node_string += value_text

        # Write node std of CATE
        node_string += "CATE std" + self.characters[4]
        std = node_info['std']
        value_text = ""
        if hasattr(std, 'shape') and (len(std.shape) > 0):
            if len(std.shape) == 1:
                for i in range(std.shape[0]):
                    value_text += "{}".format(np.around(std[i], self.precision))
                    if i != std.shape[0] - 1:
                        value_text += ", "
            elif len(std.shape) == 2:
                for i in range(std.shape[0]):
                    for j in range(std.shape[1]):
                        value_text += "{}".format(np.around(std[i, j], self.precision))
                        if j != std.shape[1] - 1:
                            value_text += ", "
                    if i != std.shape[0] - 1:
                        value_text += self.characters[4]
            else:
                raise ValueError("can only handle up to 2d values")
        else:
            value_text += "{}".format(np.around(std, self.precision))
        node_string += value_text

        return node_string


class _PolicyTreeMixin(_TreeExporter):
    """
    Mixin that supports writing out the nodes of a policy tree

    Parameters
    ----------
    treatment_names : list of str, optional
        The names of the two treatments
    """

    def __init__(self, *args, treatment_names=None, **kwargs):
        self.treatment_names = treatment_names
        super().__init__(*args, **kwargs)

    def get_fill_color(self, tree, node_id):
        # TODO. Create our own color pallete for multiple treatments. The one below is for binary treatments.
        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            self.colors['rgb'] = _color_brew(tree.n_outputs)  # [(179, 108, 96), (81, 157, 96)]

        node_val = tree.value[node_id][:, 0]
        node_val = node_val - np.min(node_val)
        if np.max(node_val) > 0:
            node_val = node_val / np.max(node_val)
        return self.get_color(node_val)

    def node_replacement_text(self, tree, node_id, criterion):
        if self.node_dict is not None:
            return self._node_replacement_text_with_dict(tree, node_id, criterion)
        value = tree.value[node_id][:, 0]
        node_string = 'value = %s' % np.round(value[1:] - value[0], self.precision)

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            node_string += self.characters[4]
            # Write node mean CATE
            node_string += 'Treatment = '
            if self.treatment_names:
                class_name = self.treatment_names[np.argmax(value)]
            else:
                class_name = "T%s%s%s" % (self.characters[1],
                                          np.argmax(value),
                                          self.characters[2])
            node_string += class_name

        return node_string

    def _node_replacement_text_with_dict(self, tree, node_id, criterion):

        # Write node mean CATE
        node_info = self.node_dict[node_id]
        node_string = 'CATE' + self.characters[4]
        value_text = ""
        mean = node_info['mean']
        if hasattr(mean, 'shape') and (len(mean.shape) > 0):
            if len(mean.shape) == 1:
                for i in range(mean.shape[0]):
                    value_text += "{}".format(np.around(mean[i], self.precision))
                    if 'ci' in node_info:
                        value_text += " ({}, {})".format(np.around(node_info['ci'][0][i], self.precision),
                                                         np.around(node_info['ci'][1][i], self.precision))
                    if i != mean.shape[0] - 1:
                        value_text += ", "
                value_text += self.characters[4]
            else:
                raise ValueError("can only handle up to 1d values")
        else:
            value_text += "{}".format(np.around(mean, self.precision))
            if 'ci' in node_info:
                value_text += " ({}, {})".format(np.around(node_info['ci'][0], self.precision),
                                                 np.around(node_info['ci'][1], self.precision))
            value_text += self.characters[4]
        node_string += value_text

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # Write recommended treatment and value - cost
            value = tree.value[node_id][:, 0]
            node_string += 'value - cost = %s' % np.round(value[1:], self.precision) + self.characters[4]

            value = tree.value[node_id][:, 0]
            node_string += "Treatment: "
            if self.treatment_names:
                class_name = self.treatment_names[np.argmax(value)]
            else:
                class_name = "T%s%s%s" % (self.characters[1],
                                          np.argmax(value),
                                          self.characters[2])
            node_string += "{}".format(class_name)
            node_string += self.characters[4]

        return node_string


class _PolicyTreeMPLExporter(_PolicyTreeMixin, _MPLExporter):
    """
    Exports policy trees to matplotlib

    Parameters
    ----------
    treatment_names : list of str, optional
        The names of the treatments

    title : str, optional
        A title for the final figure to be printed at the top of the page.

    feature_names : list of str, optional
        Names of each of the features.

    max_depth: int, optional
        The maximum tree depth to plot

    filled : bool, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    rounded : bool, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    fontsize : int, optional
        Fontsize for text
    """

    def __init__(self, treatment_names=None, title=None, feature_names=None,
                 max_depth=None,
                 filled=True,
                 rounded=False, precision=3, fontsize=None):
        super().__init__(treatment_names=treatment_names, title=title,
                         feature_names=feature_names,
                         max_depth=max_depth,
                         filled=filled, rounded=rounded, precision=precision,
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

    title : str, optional
        A title for the final figure to be printed at the top of the page.

    feature_names : list of str, optional
        Names of each of the features.

    treatment_names : list of str, optional
        The names of the treatments

    max_depth: int, optional
        The maximum tree depth to plot

    filled : bool, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    rounded : bool, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    precision : int, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.

    fontsize : int, optional
        Fontsize for text
    """

    def __init__(self, include_uncertainty, uncertainty_level, title=None,
                 feature_names=None,
                 treatment_names=None,
                 max_depth=None,
                 filled=True, rounded=False, precision=3, fontsize=None):
        super().__init__(include_uncertainty, uncertainty_level, title=None,
                         feature_names=feature_names,
                         treatment_names=treatment_names,
                         max_depth=max_depth,
                         filled=filled,
                         rounded=rounded, precision=precision, fontsize=fontsize,
                         impurity=False)


class _PolicyTreeDOTExporter(_PolicyTreeMixin, _DOTExporter):
    """
    Exports policy trees to dot files

    Parameters
    ----------
    out_file : file object or str, optional
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

    title : str, optional
        A title for the final figure to be printed at the top of the page.

    feature_names : list of str, optional
        Names of each of the features.

    treatment_names : list of str, optional
        The names of the treatments

    max_depth: int, optional
        The maximum tree depth to plot

    filled : bool, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, default False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    rotate : bool, default False
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    special_characters : bool, default False
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    precision : int, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.
    """

    def __init__(self, out_file=None, title=None, treatment_names=None, feature_names=None,
                 max_depth=None,
                 filled=True, leaves_parallel=False,
                 rotate=False, rounded=False, special_characters=False, precision=3):
        super().__init__(title=title, out_file=out_file, feature_names=feature_names,
                         max_depth=max_depth, filled=filled, leaves_parallel=leaves_parallel,
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

    out_file : file object or str, optional
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

    title : str, optional
        A title for the final figure to be printed at the top of the page.

    feature_names : list of str, optional
        Names of each of the features.

    treatment_names : list of str, optional
        The names of the treatments

    max_depth: int, optional
        The maximum tree depth to plot

    filled : bool, default False
        When set to ``True``, paint nodes to indicate majority class for
        classification, extremity of values for regression, or purity of node
        for multi-output.

    leaves_parallel : bool, default False
        When set to ``True``, draw all leaf nodes at the bottom of the tree.

    rotate : bool, default False
        When set to ``True``, orient tree left to right rather than top-down.

    rounded : bool, default False
        When set to ``True``, draw node boxes with rounded corners and use
        Helvetica fonts instead of Times-Roman.

    special_characters : bool, default False
        When set to ``False``, ignore special characters for PostScript
        compatibility.

    precision : int, default 3
        Number of digits of precision for floating point in the values of
        impurity, threshold and value attributes of each node.
    """

    def __init__(self, include_uncertainty, uncertainty_level, out_file=None, title=None, feature_names=None,
                 treatment_names=None,
                 max_depth=None, filled=True, leaves_parallel=False,
                 rotate=False, rounded=False, special_characters=False, precision=3):
        super().__init__(include_uncertainty, uncertainty_level,
                         out_file=out_file, title=title, feature_names=feature_names,
                         treatment_names=treatment_names,
                         max_depth=max_depth, filled=filled, leaves_parallel=leaves_parallel,
                         rotate=rotate, rounded=rounded, special_characters=special_characters,
                         precision=precision,
                         impurity=False)


class _SingleTreeExporterMixin(metaclass=abc.ABCMeta):

    tree_model_ = None
    node_dict_ = None

    @abc.abstractmethod
    def _make_dot_exporter(self, *, out_file, feature_names, treatment_names, max_depth, filled,
                           leaves_parallel, rotate, rounded,
                           special_characters, precision):
        """
        Make a dot file exporter

        Parameters
        ----------
        out_file : file object
            Handle to write to.

        feature_names : list of str
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments, starting with a name for the baseline/control treatment
            (alphanumerically smallest in case of discrete treatment or the all-zero treatment
            in the case of continuous)

        max_depth: int, optional
            The maximum tree depth to plot

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
        raise NotImplementedError("Abstract method")

    @abc.abstractmethod
    def _make_mpl_exporter(self, *, title=None, feature_names=None, treatment_names=None, max_depth=None,
                           filled=True, rounded=True, precision=3, fontsize=None):
        """
        Make a matplotlib exporter

        Parameters
        ----------
        title : str
            A title for the final figure to be printed at the top of the page.

        feature_names : list of str
            Names of each of the features.

        treatment_names : list of str, optional
            Names of each of the treatments

        max_depth: int, optional
            The maximum tree depth to plot

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
        raise NotImplementedError("Abstract method")

    def export_graphviz(self, out_file=None, feature_names=None, treatment_names=None,
                        max_depth=None,
                        filled=True, leaves_parallel=True,
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
            Names of each of the treatments

        max_depth: int, optional
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

        check_is_fitted(self.tree_model_, 'tree_')
        own_file = False
        try:
            if isinstance(out_file, str):
                out_file = open(out_file, "w", encoding="utf-8")
                own_file = True

            return_string = out_file is None
            if return_string:
                out_file = StringIO()

            exporter = self._make_dot_exporter(out_file=out_file, feature_names=feature_names,
                                               treatment_names=treatment_names,
                                               max_depth=max_depth, filled=filled,
                                               leaves_parallel=leaves_parallel, rotate=rotate, rounded=rounded,
                                               special_characters=special_characters, precision=precision)
            exporter.export(self.tree_model_, node_dict=self.node_dict_)

            if return_string:
                return out_file.getvalue()

        finally:
            if own_file:
                out_file.close()

    def render(self, out_file, format='pdf', view=True, feature_names=None,
               treatment_names=None,
               max_depth=None,
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
            Names of each of the treatments

        max_depth: int, optional
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
        dot_source = self.export_graphviz(out_file=None,  # want the output as a string, only write the final file
                                          feature_names=feature_names, treatment_names=treatment_names,
                                          max_depth=max_depth,
                                          filled=filled,
                                          leaves_parallel=leaves_parallel, rotate=rotate,
                                          rounded=rounded, special_characters=special_characters,
                                          precision=precision)
        graphviz.Source(dot_source).render(out_file, format=format, view=view)

    def plot(self, ax=None, title=None, feature_names=None, treatment_names=None,
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
            Names of each of the treatments

        max_depth: int, optional
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
        check_is_fitted(self.tree_model_, 'tree_')
        exporter = self._make_mpl_exporter(title=title, feature_names=feature_names, treatment_names=treatment_names,
                                           max_depth=max_depth,
                                           filled=filled,
                                           rounded=rounded, precision=precision, fontsize=fontsize)
        exporter.export(self.tree_model_, node_dict=self.node_dict_, ax=ax)
