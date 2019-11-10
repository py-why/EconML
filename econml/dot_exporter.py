import abc
from numbers import Integral
import numpy as np
import sklearn.tree
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import _criterion
from sklearn.tree import _tree
from sklearn.tree._reingold_tilford import buchheim, Tree
import warnings
import seaborn as sns


class _BaseExporter(metaclass=abc.ABCMeta):
    """
    Parameters
    ----------
    out_file : file object or string, optional (default=None)
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

    title : string, optional (default=None)
        A title for the final figure to be printed at the top of the page.

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

    def __init__(self, out_file=None, title=None, feature_names=None,
                 filled=True, leaves_parallel=False,
                 rotate=False, rounded=False, special_characters=False, precision=3):
        self.feature_names = feature_names
        self.filled = filled
        self.leaves_parallel = leaves_parallel
        self.rotate = rotate
        self.rounded = rounded
        self.special_characters = special_characters
        self.out_file = out_file
        self.title = title

        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError("'precision' should be greater or equal to 0."
                                 " Got {} instead.".format(precision))
        else:
            raise ValueError("'precision' should be an integer. Got {}"
                             " instead.".format(type(precision)))
        self.precision = precision

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {'leaves': []}
        # The colors to render each node with
        self.colors = {'bounds': None}

        # PostScript compatibility for special characters
        if special_characters:
            self.characters = ['&#35;', '<SUB>', '</SUB>', '&le;', '<br/>', '>', '<']
        else:
            self.characters = ['#', '[', ']', '<=', '\\n', '"', '"']

    def get_color(self, value):
        # Find the appropriate color & intensity for a node
        if self.colors['bounds'] is None:
            # Classification tree
            colors = np.array([sns.cubehelix_palette(100, start=1, rot=0, dark=.5, light=1),
                               sns.cubehelix_palette(100, start=2, rot=0, dark=.5, light=1)])
            color = colors[np.argmax(value)]
            sorted_values = sorted(value, reverse=True)
            if len(sorted_values) == 1:
                alpha = 0
            else:
                alpha = (sorted_values[0] - sorted_values[1]) / (1 - sorted_values[1])
            v_grid = np.linspace(0, 1, 100)
            rgb_color = np.round(color[np.searchsorted(v_grid, alpha)] * 255)
            return '#{:02x}{:02x}{:02x}'.format(*rgb_color.astype(int))
        else:
            # Regression tree or multi-output
            if value > 0:
                v_grid = np.linspace(max(self.colors['bounds'][0], 0), self.colors['bounds'][1], 100)
                rgb_color = np.round(np.array(sns.cubehelix_palette(100, start=2,
                                                                    rot=0,
                                                                    dark=.5,
                                                                    light=1))[np.searchsorted(v_grid, value)] * 255)
                return '#{:02x}{:02x}{:02x}'.format(*rgb_color.astype(int))
            else:
                v_grid = np.linspace(self.colors['bounds'][0], min(self.colors['bounds'][1], 0), 100)
                rgb_color = np.round(np.array(sns.cubehelix_palette(100, start=1,
                                                                    rot=0, dark=.5, light=1,
                                                                    reverse=True))[np.searchsorted(v_grid,
                                                                                                   value)] * 255)
                return '#{:02x}{:02x}{:02x}'.format(*rgb_color.astype(int))

    @abc.abstractmethod
    def get_fill_color(self, tree, node_id):
        pass

    @abc.abstractmethod
    def node_to_str(self, tree, node_id, criterion):
        pass

    @abc.abstractmethod
    def export(self, decision_tree):
        pass


class _DOTExporterMixin(_BaseExporter):

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Collect ranks for 'leaf' option in plot_options
        if left_child == _tree.TREE_LEAF:
            self.ranks['leaves'].append(str(node_id))
        elif str(depth) not in self.ranks:
            self.ranks[str(depth)] = [str(node_id)]
        else:
            self.ranks[str(depth)].append(str(node_id))

        self.out_file.write('%d [label=%s'
                            % (node_id,
                               self.node_to_str(tree, node_id, criterion)))

        if self.filled:
            self.out_file.write(', fillcolor="%s"' % self.get_fill_color(tree, node_id))
        self.out_file.write('] ;\n')

        if parent is not None:
            # Add edge to parent
            self.out_file.write('%d -> %d' % (parent, node_id))
            if parent == 0:
                # Draw True/False labels if parent is root node
                angles = np.array([45, -45]) * ((self.rotate - .5) * -2)
                self.out_file.write(' [labeldistance=2.5, labelangle=')
                if node_id == 1:
                    self.out_file.write('%d, headlabel="True"]' % angles[0])
                else:
                    self.out_file.write('%d, headlabel="False"]' % angles[1])
            self.out_file.write(' ;\n')

        if left_child != _tree.TREE_LEAF:
            self.recurse(tree, left_child, criterion=criterion, parent=node_id,
                         depth=depth + 1)
            self.recurse(tree, right_child, criterion=criterion, parent=node_id,
                         depth=depth + 1)

    def export(self, decision_tree):
        # Check length of feature_names before getting into the tree node
        # Raise error if length of feature_names does not match
        # n_features_ in the decision_tree
        if self.feature_names is not None:
            if len(self.feature_names) != decision_tree.n_features_:
                raise ValueError("Length of feature_names, %d "
                                 "does not match number of features, %d"
                                 % (len(self.feature_names),
                                    decision_tree.n_features_))

        self.out_file.write('digraph Tree {\n')

        # Specify node aesthetics
        self.out_file.write('node [shape=box')
        rounded_filled = []
        if self.filled:
            rounded_filled.append('filled')
        if self.rounded:
            rounded_filled.append('rounded')
        if len(rounded_filled) > 0:
            self.out_file.write(', style="%s", color="black"'
                                % ", ".join(rounded_filled))
        if self.rounded:
            self.out_file.write(', fontname=helvetica')
        self.out_file.write('] ;\n')

        # Specify graph & edge aesthetics
        if self.leaves_parallel:
            self.out_file.write('graph [ranksep=equally, splines=polyline] ;\n')
        if self.rounded:
            self.out_file.write('edge [fontname=helvetica] ;\n')
        if self.rotate:
            self.out_file.write('rankdir=LR ;\n')

        # Now recurse the tree and add node & edge attributes
        self.recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)

        # If required, draw leaf nodes at same depth as each other
        if self.leaves_parallel:
            for rank in sorted(self.ranks):
                self.out_file.write("{rank=same ; " +
                                    "; ".join(r for r in self.ranks[rank]) + "} ;\n")

        if self.title is not None:
            self.out_file.write("labelloc=\"t\"; \n")
            self.out_file.write("label=\"{}\"; \n".format(self.title))

        self.out_file.write("}")


class _MPLExporterMixin(_BaseExporter):

    def __init__(self, title=None, feature_names=None,
                 filled=True, rounded=False, precision=3, fontsize=None):

        _BaseExporter.__init__(self,
                               out_file=None, title=title, feature_names=feature_names,
                               filled=filled, leaves_parallel=False,
                               rotate=False, rounded=rounded, precision=precision)

        self.fontsize = fontsize

        # validate
        if isinstance(precision, Integral):
            if precision < 0:
                raise ValueError("'precision' should be greater or equal to 0."
                                 " Got {} instead.".format(precision))
        else:
            raise ValueError("'precision' should be an integer. Got {}"
                             " instead.".format(type(precision)))

        # The depth of each node for plotting with 'leaf' option
        self.ranks = {'leaves': []}
        # The colors to render each node with
        self.colors = {'bounds': None}

        self.characters = ['#', '[', ']', '<=', '\n', '', '']

        self.bbox_args = dict(fc='w')
        if self.rounded:
            self.bbox_args['boxstyle'] = "round"
        else:
            # matplotlib <1.5 requires explicit boxstyle
            self.bbox_args['boxstyle'] = "square"

        self.arrow_args = dict(arrowstyle="<-")

    def _make_tree(self, node_id, et, criterion, depth=0):
        # traverses _tree.Tree recursively, builds intermediate
        # "_reingold_tilford.Tree" object
        name = self.node_to_str(et, node_id, criterion=criterion)
        if (et.children_left[node_id] != _tree.TREE_LEAF):
            children = [self._make_tree(et.children_left[node_id], et,
                                        criterion, depth=depth + 1),
                        self._make_tree(et.children_right[node_id], et,
                                        criterion, depth=depth + 1)]
        else:
            return Tree(name, node_id)
        return Tree(name, node_id, *children)

    def export(self, decision_tree, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.text import Annotation
        if ax is None:
            ax = plt.gca()
        ax.clear()
        ax.set_axis_off()
        my_tree = self._make_tree(0, decision_tree.tree_,
                                  decision_tree.criterion)
        draw_tree = buchheim(my_tree)

        # important to make sure we're still
        # inside the axis after drawing the box
        # this makes sense because the width of a box
        # is about the same as the distance between boxes
        max_x, max_y = draw_tree.max_extents() + 1
        ax_width = ax.get_window_extent().width
        ax_height = ax.get_window_extent().height

        scale_x = ax_width / max_x
        scale_y = ax_height / max_y

        self.recurse(draw_tree, decision_tree.tree_, ax,
                     scale_x, scale_y, ax_height)

        anns = [ann for ann in ax.get_children()
                if isinstance(ann, Annotation)]

        # update sizes of all bboxes
        renderer = ax.figure.canvas.get_renderer()

        for ann in anns:
            ann.update_bbox_position_size(renderer)

        if self.fontsize is None:
            # get figure to data transform
            # adjust fontsize to avoid overlap
            # get max box width and height
            try:
                extents = [ann.get_bbox_patch().get_window_extent()
                           for ann in anns]
                max_width = max([extent.width for extent in extents])
                max_height = max([extent.height for extent in extents])
                # width should be around scale_x in axis coordinates
                size = anns[0].get_fontsize() * min(scale_x / max_width,
                                                    scale_y / max_height)
                for ann in anns:
                    ann.set_fontsize(size)
            except AttributeError:
                # matplotlib < 1.5
                warnings.warn("Automatic scaling of tree plots requires "
                              "matplotlib 1.5 or higher. Please specify "
                              "fontsize.")

        if self.title is not None:
            ax.set_title(self.title)
        return anns

    def recurse(self, node, tree, ax, scale_x, scale_y, height, depth=0):
        # need to copy bbox args because matplotib <1.5 modifies them
        kwargs = dict(bbox=self.bbox_args.copy(), ha='center', va='center',
                      zorder=100 - 10 * depth, xycoords='axes pixels')

        if self.fontsize is not None:
            kwargs['fontsize'] = self.fontsize

        # offset things by .5 to center them in plot
        xy = ((node.x + .5) * scale_x, height - (node.y + .5) * scale_y)

        if self.filled:
            kwargs['bbox']['fc'] = self.get_fill_color(tree,
                                                       node.tree.node_id)
        if node.parent is None:
            # root
            ax.annotate(node.tree.label, xy, **kwargs)
        else:
            xy_parent = ((node.parent.x + .5) * scale_x,
                         height - (node.parent.y + .5) * scale_y)
            kwargs["arrowprops"] = self.arrow_args
            ax.annotate(node.tree.label, xy_parent, xy, **kwargs)
        for child in node.children:
            self.recurse(child, tree, ax, scale_x, scale_y, height,
                         depth=depth + 1)


class _CATETreeExporterMixin(_BaseExporter):
    """
    Parameters
    ----------
    include_uncertainty: bool
        whether the tree includes uncertainty information

    uncertainty_level: float
        the confidence level of the confidence interval included in the tree

    out_file : file object or string, optional (default=None)
        Handle or name of the output file. If ``None``, the result is
        returned as a string.

    title : string, optional (default=None)
        A title for the final figure to be printed at the top of the page.

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

    def __init__(self, include_uncertainty, uncertainty_level):
        self.include_uncertainty = include_uncertainty
        self.uncertainty_level = uncertainty_level

    def get_fill_color(self, tree, node_id):
        # Fetch appropriate color for node
        if 'rgb' not in self.colors:
            # Initialize colors and bounds if required
            # Find max and min values in leaf nodes for regression
            if tree.value.ndim > 1:  # in multi-target use first target
                self.colors['bounds'] = (np.min([v[0, 0] for v in tree.value]),
                                         np.max([v[0, 0] for v in tree.value]))
            else:
                self.colors['bounds'] = (np.min(tree.value),
                                         np.max(tree.value))

        # Use first target label of the regression
        if tree.value.ndim > 2:
            node_val = tree.value[node_id][0, 0]
        else:
            node_val = tree.value[node_id][0]
        return self.get_color(node_val)

    def node_to_str(self, tree, node_id, criterion):
        # Generate the node content string
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]

        node_string = self.characters[-1]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (self.characters[1],
                                       tree.feature[node_id],
                                       self.characters[2])
            node_string += '%s %s %s%s' % (feature,
                                           self.characters[3],
                                           round(tree.threshold[node_id],
                                                 self.precision),
                                           self.characters[4])

        # Write node sample count
        node_string += 'samples = '
        node_string += (str(tree.n_node_samples[node_id]) +
                        self.characters[4])

        # Write node mean CATE
        node_string += 'CATE mean = '
        if self.include_uncertainty:
            value_text = np.around(value[0, 0], self.precision)
        else:
            value_text = np.around(value[0], self.precision)
        # Strip whitespace
        value_text = str(value_text.astype('S32')).replace("b'", "'")
        value_text = value_text.replace("' '", ", ").replace("'", "")
        value_text = value_text.replace("\n ", self.characters[4])
        node_string += value_text + self.characters[4]

        # Write node std of CATE
        node_string += "CATE std = "
        node_string += (str(round(np.sqrt(np.clip(tree.impurity[node_id], 0, np.inf)), self.precision)) +
                        self.characters[4])

        # Write confidence interval information if at leaf node
        if (tree.children_left[node_id] == _tree.TREE_LEAF) and self.include_uncertainty:
            ci_text = "Mean Endpoints of {}% CI: ({}, {})".format(int((1 - self.uncertainty_level) * 100),
                                                                  np.around(value[1, 0], self.precision),
                                                                  np.around(value[2, 0], self.precision))
            node_string += ci_text + self.characters[4]

        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + self.characters[5]


class _CATETreeDOTExporter(_CATETreeExporterMixin, _DOTExporterMixin):

    def __init__(self, include_uncertainty, uncertainty_level, out_file=None, title=None, feature_names=None,
                 filled=True, leaves_parallel=False,
                 rotate=False, rounded=False, special_characters=False, precision=3):
        """
        Parameters
        ----------
        include_uncertainty: bool
            whether the tree includes uncertainty information

        uncertainty_level: float
            the confidence level of the confidence interval included in the tree

        out_file : file object or string, optional (default=None)
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        title : string, optional (default=None)
            A title for the final figure to be printed at the top of the page.

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
        _CATETreeExporterMixin.__init__(self, include_uncertainty, uncertainty_level)
        _DOTExporterMixin.__init__(self, out_file=out_file, title=title, feature_names=feature_names,
                                   filled=filled, leaves_parallel=leaves_parallel,
                                   rotate=rotate, rounded=rounded, special_characters=special_characters,
                                   precision=precision)


class _CATETreeMPLExporter(_CATETreeExporterMixin, _MPLExporterMixin):

    def __init__(self, include_uncertainty, uncertainty_level, title=None, feature_names=None,
                 filled=True,
                 rounded=False, precision=3, fontsize=None):
        """
        Parameters
        ----------
        include_uncertainty: bool
            whether the tree includes uncertainty information

        uncertainty_level: float
            the confidence level of the confidence interval included in the tree

        out_file : file object or string, optional (default=None)
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        title : string, optional (default=None)
            A title for the final figure to be printed at the top of the page.

        feature_names : list of strings, optional (default=None)
            Names of each of the features.

        filled : bool, optional (default=False)
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        rounded : bool, optional (default=False)
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        precision : int, optional (default=3)
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.

        fontsize : int, optional (default=None)
            Fontsize for text
        """
        _CATETreeExporterMixin.__init__(self, include_uncertainty, uncertainty_level)
        _MPLExporterMixin.__init__(self, title=title, feature_names=feature_names,
                                   filled=filled,
                                   rounded=rounded, precision=precision,
                                   fontsize=fontsize)


class _PolicyTreeExporterMixin(_BaseExporter):

    def __init__(self, treatment_names=None):
        """
        Parameters
        ----------
        treatment_names : list of strings, optional (default=None)
            The names of the two treatments
        """
        self.treatment_names = treatment_names

    def get_fill_color(self, tree, node_id):
        node_value = tree.value[node_id][0, :] / tree.weighted_n_node_samples[node_id]
        return self.get_color(node_value)

    def node_to_str(self, tree, node_id, criterion):
        # Generate the node content string
        value = tree.value[node_id][0, :]

        node_string = self.characters[-1]

        # Write decision criteria
        if tree.children_left[node_id] != _tree.TREE_LEAF:
            # Always write node decision criteria, except for leaves
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
            else:
                feature = "X%s%s%s" % (self.characters[1],
                                       tree.feature[node_id],
                                       self.characters[2])
            node_string += '%s %s %s%s' % (feature,
                                           self.characters[3],
                                           round(tree.threshold[node_id],
                                                 self.precision),
                                           self.characters[4])

        # Write node sample count
        node_string += 'samples = '
        node_string += (str(tree.n_node_samples[node_id]) +
                        self.characters[4])

        node_string += '(effect - cost) mean = %s' % np.round((value[1] -
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

        # Clean up any trailing newlines
        if node_string[-2:] == '\\n':
            node_string = node_string[:-2]
        if node_string[-5:] == '<br/>':
            node_string = node_string[:-5]

        return node_string + self.characters[5]


class _PolicyTreeDOTExporter(_PolicyTreeExporterMixin, _DOTExporterMixin):

    def __init__(self, out_file=None, title=None, treatment_names=None, feature_names=None,
                 filled=True, leaves_parallel=False,
                 rotate=False, rounded=False, special_characters=False, precision=3):
        """
        Parameters
        ----------
        out_file : file object or string, optional (default=None)
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        title : string, optional (default=None)
            A title for the final figure to be printed at the top of the page.

        treatment_names : list of strings, optional (default=None)
            The names of the two treatments

        out_file : file object or string, optional (default=None)
            Handle or name of the output file. If ``None``, the result is
            returned as a string.

        title : string, optional (default=None)
            A title for the final figure to be printed at the top of the page.

        treatment_names : list of strings, optional (default=None)
            The names of the two treatments

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
        _DOTExporterMixin.__init__(self, title=title, out_file=out_file, feature_names=feature_names,
                                   filled=filled, leaves_parallel=leaves_parallel,
                                   rotate=rotate, rounded=rounded, special_characters=special_characters,
                                   precision=precision)
        _PolicyTreeExporterMixin.__init__(self, treatment_names=treatment_names)


class _PolicyTreeMPLExporter(_PolicyTreeExporterMixin, _MPLExporterMixin):

    def __init__(self, treatment_names=None, title=None, feature_names=None,
                 filled=True,
                 rounded=False, precision=3, fontsize=None):
        """
        Parameters
        ----------
        treatment_names : list of strings, optional (default=None)
            The names of the two treatments

        title : string, optional (default=None)
            A title for the final figure to be printed at the top of the page.

        feature_names : list of strings, optional (default=None)
            Names of each of the features.

        filled : bool, optional (default=False)
            When set to ``True``, paint nodes to indicate majority class for
            classification, extremity of values for regression, or purity of node
            for multi-output.

        rounded : bool, optional (default=False)
            When set to ``True``, draw node boxes with rounded corners and use
            Helvetica fonts instead of Times-Roman.

        precision : int, optional (default=3)
            Number of digits of precision for floating point in the values of
            impurity, threshold and value attributes of each node.

        fontsize : int, optional (default=None)
            Fontsize for text
        """
        _MPLExporterMixin.__init__(self, title=title, feature_names=feature_names,
                                   filled=filled,
                                   rounded=rounded, precision=precision,
                                   fontsize=fontsize)
        _PolicyTreeExporterMixin.__init__(self, treatment_names=treatment_names)
