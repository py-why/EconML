# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code is a fork from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_tree.pyx
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.


from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX
from libc.math cimport pow

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csr_matrix

from ._utils cimport Stack
from ._utils cimport StackRecord
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float64 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps

TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

# The definition of a numpy type to be used for converting the malloc'ed memory space that
# contains an array of Node struct's into a structured numpy parallel array that can be as
# array[key][index].
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'depth', 'feature', 'threshold',
              'impurity', 'n_node_samples', 'weighted_n_node_samples',
              'impurity_train', 'n_node_samples_train', 'weighted_n_node_samples_train'],
    'formats': [np.intp, np.intp, np.intp, np.intp, np.float64,
                np.float64, np.intp, np.float64,
                np.float64, np.intp, np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).depth,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).impurity_train,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples_train,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples_train,
    ]
})

cdef class TreeBuilder:
    """Interface for different tree building strategies."""

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray samples_train,
                np.ndarray samples_val,
                np.ndarray sample_weight=None,
                bint store_jac=False):
        """Build tree from the training set (X, y) using samples_train for the
        constructing the splits and samples_val for estimating the node values. store_jac
        controls whether jacobian information is stored in the tree nodes in the case of
        generalized random forests.
        """
        pass

    cdef inline _check_input(self, object X, np.ndarray y, np.ndarray sample_weight):
        """Check input dtype, layout and format"""

        # since we have to copy and perform linear algebra we will make it fortran for efficiency
        if X.dtype != DTYPE:
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease):
        """ Initialize parameters.
        
        Parameters
        ----------
        splitter : cython extension class of type Splitter
            The splitter to be used for deciding the best split of each node.
        min_samples_split : SIZE_t
            The minimum number of samples required for a node to be considered for splitting
        min_samples_leaf : SIZE_t
            The minimum number of samples that each node must contain
        min_weight_leaf : double
            The minimum total weight of samples that each node must contain
        max_depth : SIZE_t
            The maximum depth of the tree
        min_impurity_decrease : SIZE_t
            The minimum improvement in impurity that a split must provide to be executed
        """
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray samples_train,
                np.ndarray samples_val,
                np.ndarray sample_weight=None,
                bint store_jac=False):
        """Build an honest tree from data (X, y).

        Parameters
        ----------
        X : (n, d) np.array
            The features to use for splitting
        y : (n, p) np.array
            Any information used by the criterion to calculate the node values for a given node defined by
            the X's
        samples_train : (n,) np.array of type np.intp
            The indices of the samples in X to be used for creating the splits of the tree (training set).
        samples_val : (n,) np.array of type np.intp
            The indices of the samples in X to be used for calculating the node values of the tree (val set).
        sample_weight : (n,) np.array of type np.float64
            The weight of each sample
        store_jac : bool, optional (default=False)
            Whether jacobian information should be stored in the tree nodes by calling the node_jacobian_val
            and node_precond_val of the splitter. This is related to trees that solve linear moment equations
            of the form: J(x) * theta(x) - precond(x) = 0. If store_jac=True, then J(x) and precond(x) are also
            stored at the tree nodes at the end of the build for easy access.
        """

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr, samples_train, samples_val)

        # The indices of all samples are stored in two arrays, `samples` and `samples_val` by the splitter,
        # such that each node contains the samples from `start` to `end` in the samples array and from
        # `start_val` to `end_val` in the `samples_val` array. Thus these four numbers are sufficient to
        # represent a "node" of the tree during building.
        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t start_val
        cdef SIZE_t end_val
        cdef SIZE_t depth       # The depth of the current node considered for splitting
        cdef SIZE_t parent      # The parent of the current node considered for splitting
        cdef bint is_left       # Whether the current node considered for splitting is a left or right child
        cdef SIZE_t n_node_samples = splitter.n_samples                        # Total number of training samples
        cdef double weighted_n_samples = splitter.weighted_n_samples           # Total weight of training samples
        cdef double weighted_n_node_samples     # Will be storing the total training weight of the current node
        cdef SIZE_t n_node_samples_val = splitter.n_samples_val                # Total number of val samples
        cdef double weighted_n_samples_val = splitter.weighted_n_samples_val   # Total weight of val samples
        cdef double weighted_n_node_samples_val # Will be storing the total val weight of the current node
        cdef SplitRecord split  # A split record is a struct produced by the splitter that contains all the split info
        cdef SIZE_t node_id     # Will be storing the id that the tree assigns to a node when added to the tree

        cdef double impurity = INFINITY         # Will be storing the impurity of the node considered for splitting
        cdef double proxy_impurity = INFINITY   # An approximate version of the impurity used for min impurity decrease
        cdef SIZE_t n_constant_features         # number of features identified as taking a constant value in the node
        cdef bint is_leaf                       # Whether the node we are about to add to the tree is a leaf
        cdef bint first = 1                     # If this is the root node we are splitting
        cdef SIZE_t max_depth_seen = -1         # Max depth we've seen so far
        cdef int rc = 0                         # To be used as a success flag for memory resizing calls

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)    # A stack contains the entries of all nodes to be considered
        cdef StackRecord stack_record   # A stack record contains all the information required to split a node

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, n_node_samples_val,
                            0, _TREE_UNDEFINED, 0, INFINITY, INFINITY, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)        # Let's pop a node from the stack to split
                # Let's store the stack record in local values for easy access and manipulation
                start = stack_record.start
                end = stack_record.end
                start_val = stack_record.start_val
                end_val = stack_record.end_val
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                impurity_val = stack_record.impurity_val
                n_constant_features = stack_record.n_constant_features

                # Some easy calculations
                n_node_samples = end - start
                n_node_samples_val = end_val - start_val
                # Let's reset the splitter to the initial state of considering the current node to split
                # This will also return the total weight of the node in the training and validation set
                # in the two variables passed by reference.
                splitter.node_reset(start, end, &weighted_n_node_samples,
                                    start_val, end_val, &weighted_n_node_samples_val)

                # Determine if the node is a leaf based on simple constraints on the training and val set
                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf or
                           n_node_samples_val < min_samples_split or
                           n_node_samples_val < 2 * min_samples_leaf or
                           weighted_n_node_samples_val < 2 * min_weight_leaf)

                # If either the splitter only returns approximate children impurities at the end of each split
                # or if we are in the root node, then we need to calculate node impurity for both evaluating
                # the min_impurity_decrease constraint and for storing the impurity at the tree. This is done
                # because sometimes node impurity might be computationally intensive to calculate and can be easily
                # done once the splitter has calculated all the quantities at `node_reset`, but would add too much
                # computational burden if done twice (once when considering the node for splitting and once when
                # calculating the node's impurity when it is the children of a node that has just been split). Thus
                # sometimes we just want to calculate an approximate child node impurity, solely for the purpose
                # of evaluating whether the min_impurity_decrease constraint is satisfied by the returned split.
                if (splitter.is_children_impurity_proxy()) or first:
                    # This is the baseline of what we should use for impurity improvement
                    proxy_impurity = splitter.proxy_node_impurity()
                    # This is the true node impurity we want to store in the tree. The two should coincide if
                    # `splitter.is_children_impurity_proxy()==False`.
                    impurity = splitter.node_impurity()             # The node impurity on the training set
                    impurity_val = splitter.node_impurity_val()     # The node impurity on the val set
                    first = 0
                else:
                    # We use the impurity value stored in the stack, which was returned by children_impurity()
                    # when the parent node of this node was split.
                    proxy_impurity = impurity

                if not is_leaf:
                    # Find the best split of the node and return it in the `split` variable
                    # Also use the fact that so far we have deemed `n_constant_features` to be constant in the
                    # parent node and at the end return the number of features that have been deemed as taking
                    # constant value, by updating the `n_constant_features` which is passed by reference.
                    # This is used for speeding up computation as these features are not considered further for
                    # splitting. This speed up is also enabled by the fact that we are doing depth-first-search
                    # build.
                    splitter.node_split(proxy_impurity, &split, &n_constant_features)
                    # Note from original sklearn comments: If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are dissimilar to v0.18
                    is_leaf = (is_leaf or
                               split.pos >= end or # no split of the training set was valid
                               split.pos_val >= end_val or # no split of the validation set was valid
                               (split.improvement + EPSILON < min_impurity_decrease)) # min impurity is violated

                # Add the node that was just split to the tree, with all the auxiliary information and
                # get the `node_id` assigned to it.
                node_id = tree._add_node(parent, is_left, is_leaf, 
                                         split.feature, split.threshold,
                                         impurity, n_node_samples, weighted_n_node_samples,
                                         impurity_val, n_node_samples_val, weighted_n_node_samples_val)
                # Memory error
                if node_id == SIZE_MAX:
                    rc = -1
                    break

                # Store value for all nodes, to facilitate tree/model inspection and interpretation
                splitter.node_value_val(tree.value + node_id * tree.value_stride)
                # If we are in a linear moment case and we want to store the node jacobian and node precond,
                # i.e. value = Jacobian^{-1} @ precond
                if store_jac:
                    splitter.node_jacobian_val(tree.jac + node_id * tree.jac_stride)
                    splitter.node_precond_val(tree.precond + node_id * tree.precond_stride)

                if not is_leaf:
                    # Push right child on stack
                    rc = stack.push(split.pos, end, split.pos_val, end_val, depth + 1, node_id, 0,
                                    split.impurity_right, split.impurity_right_val, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, start_val, split.pos_val, depth + 1, node_id, 1,
                                    split.impurity_left, split.impurity_left_val, n_constant_features)
                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            # Resize the tree to use the minimal required memory
            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            # Update trees max_depth variable for the maximum seen depth
            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    # This is only used for compatibility with sklearn trees. In sklearn this represents the number of classes
    # in a classification tree for each target output. Here it is always an array of 1's of size `self.n_outputs`.
    property n_classes:
        def __get__(self):
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property depth:
        def __get__(self):
            return self._get_node_ndarray()['depth'][:self.node_count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]
    
    property impurity_train:
        def __get__(self):
            return self._get_node_ndarray()['impurity_train'][:self.node_count]

    property n_node_samples_train:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples_train'][:self.node_count]

    property weighted_n_node_samples_train:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples_train'][:self.node_count]

    # Value returns the relevant parameters estimated at each node (the ones we care about)
    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count, :self.n_relevant_outputs]

    # Value returns all the parameters estimated at each node (even the nuisance ones we don't care about)
    property full_value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    # The jacobian J(x) of the node, for the case of linear moment trees with moment: J(x) * theta(x) - precond(x) = 0
    property jac:
        def __get__(self):
            if not self.store_jac:
                raise AttributeError("Jacobian computation was not enabled. Set store_jac=True")
            return self._get_jac_ndarray()[:self.node_count]
    
    # The precond(x) of the node, for the case of linear moment trees with moment: J(x) * theta(x) - precond(x) = 0
    property precond:
        def __get__(self):
            if not self.store_jac:
                raise AttributeError("Preconditioned quantity computation was not enabled. Set store_jac=True")
            return self._get_precond_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, int n_outputs, int n_relevant_outputs=-1, bint store_jac=False):
        """ Initialize parameters

        Parameters
        ----------
        n_features : int
            Number of features X at train time
        n_outputs : int
            How many parameters/outputs are stored/estimated at each node
        n_relevant_outputs : int, optional (default=-1)
            Which prefix of the parameters do we care about. The remainder are nuisance parameters.
            If `n_relevant_outputs=-1`, then all parameters are relevant.
        store_jac : bool, optional (default=False)
            Whether we will be storing jacobian and precond of linear moments information at each node.
        """
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_relevant_outputs = n_relevant_outputs if n_relevant_outputs > 0 else n_outputs
        self.value_stride = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)
        self.max_n_classes = 1
        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = 1

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL
        self.store_jac = store_jac
        self.jac = NULL
        self.jac_stride = n_outputs * n_outputs
        self.precond = NULL
        self.precond_stride = n_outputs

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.value)
        free(self.nodes)
        if self.store_jac:
            free(self.jac)
            free(self.precond)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features, self.n_outputs,
                       self.n_relevant_outputs, self.store_jac), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        # capacity is inferred during the __setstate__ using nodes
        d['max_depth'] = self.max_depth
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        if self.store_jac:
            d['jac'] = self._get_jac_ndarray()
            d['precond'] = self._get_precond_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.max_depth = d['max_depth']
        self.node_count = d['node_count']

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous):
            raise ValueError('Did not recognise loaded array layout for `node_ndarray`')

        value_shape = (node_ndarray.shape[0], self.n_outputs, self.max_n_classes)
        if (value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout for `value_ndarray`')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))
        
        if self.store_jac:
            jac_ndarray = d['jac']
            jac_shape = (node_ndarray.shape[0], self.n_outputs * self.n_outputs)
            if (jac_ndarray.shape != jac_shape or
                    not jac_ndarray.flags.c_contiguous or
                    jac_ndarray.dtype != np.float64):
                raise ValueError('Did not recognise loaded array layout for `jac_ndarray`')
            jac = memcpy(self.jac, (<np.ndarray> jac_ndarray).data,
                         self.capacity * self.jac_stride * sizeof(double))
            precond_ndarray = d['precond']
            precond_shape = (node_ndarray.shape[0], self.n_outputs)
            if (precond_ndarray.shape != precond_shape or
                    not precond_ndarray.flags.c_contiguous or
                    precond_ndarray.dtype != np.float64):
                raise ValueError('Did not recognise loaded array layout for `precond_ndarray`')
            precond = memcpy(self.precond, (<np.ndarray> precond_ndarray).data,
                             self.capacity * self.precond_stride * sizeof(double))

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if self._resize_c(capacity) != 0:
            # Acquire gil only if we need to raise
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        """Guts of _resize
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))
        
        if self.store_jac:
            safe_realloc(&self.jac, capacity * self.jac_stride)
            safe_realloc(&self.precond, capacity * self.precond_stride)
            if capacity > self.capacity:
                memset(<void*>(self.jac + self.capacity * self.jac_stride), 0,
                       (capacity - self.capacity) * self.jac_stride * sizeof(double))
                memset(<void*>(self.precond + self.capacity * self.precond_stride), 0,
                       (capacity - self.capacity) * self.precond_stride * sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, 
                          double impurity_train, SIZE_t n_node_samples_train,
                          double weighted_n_node_samples_train,
                          double impurity_val, SIZE_t n_node_samples_val,
                          double weighted_n_node_samples_val) nogil except -1:
        """Add a node to the tree.
        The new node registers itself as the child of its parent.
        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity_val
        node.n_node_samples = n_node_samples_val
        node.weighted_n_node_samples = weighted_n_node_samples_val
        node.impurity_train = impurity_train
        node.n_node_samples_train = n_node_samples_train
        node.weighted_n_node_samples_train = weighted_n_node_samples_train

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id
            node.depth = self.nodes[parent].depth + 1
        else:
            node.depth = 0

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')[:, :self.n_relevant_outputs, 0]
        return out
    
    cpdef np.ndarray predict_full(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')[:, :, 0]
        return out
    
    cpdef np.ndarray predict_jac(self, object X):
        """Predict target for X."""
        if not self.store_jac:
            raise AttributeError("Jacobian computation was not enalbed. Set store_jac=True")
        out = self._get_jac_ndarray().take(self.apply(X), axis=0,
                                           mode='clip')
        return out
    
    cpdef np.ndarray predict_precond(self, object X):
        """Predict target for X."""
        if not self.store_jac:
            raise AttributeError("Preconditioned quantity computation was not enalbed. Set store_jac=True")
        out = self._get_precond_ndarray().take(self.apply(X), axis=0,
                                               mode='clip')
        return out
    
    cpdef predict_precond_and_jac(self, object X):
        if not self.store_jac:
            raise AttributeError("Preconditioned quantity computation was not enalbed. Set store_jac=True")
        leafs = self.apply(X)
        precond = self._get_precond_ndarray().take(leafs, axis=0,
                                                   mode='clip')
        jac = self._get_jac_ndarray().take(leafs, axis=0,
                                           mode='clip')
        return precond, jac

    cpdef np.ndarray apply(self, object X):
        return self._apply(X)

    cdef inline np.ndarray _apply(self, object X):
        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float64, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out
    
    cpdef object decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""
        return self._decision_path(X)

    cdef inline object _decision_path(self, object X):
        """Finds the decision path (=node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float64, got %s" % X.dtype)

        # Extract input
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] indptr = np.zeros(n_samples + 1, dtype=np.intp)
        cdef SIZE_t* indptr_ptr = <SIZE_t*> indptr.data

        cdef np.ndarray[SIZE_t] indices = np.zeros(n_samples *
                                                   (1 + self.max_depth),
                                                   dtype=np.intp)
        cdef SIZE_t* indices_ptr = <SIZE_t*> indices.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                indptr_ptr[i + 1] = indptr_ptr[i]

                # Add all external nodes
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                    indptr_ptr[i + 1] += 1

                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                # Add the leave node
                indices_ptr[indptr_ptr[i + 1]] = <SIZE_t>(node - self.nodes)
                indptr_ptr[i + 1] += 1

        indices = indices[:indptr[n_samples]]
        cdef np.ndarray[SIZE_t] data = np.ones(shape=len(indices),
                                               dtype=np.intp)
        out = csr_matrix((data, indices, indptr),
                         shape=(n_samples, self.node_count))

        return out

    cpdef compute_feature_importances(self, normalize=True, max_depth=None, depth_decay=.0):
        """Computes the importance of each feature (aka variable) based on impurity decrease.
        
        Parameters
        ----------
        normalize : bool, optional (default=True)
            Whether to normalize importances to sum to 1
        max_depth : int or None, optional (default=None)
            The max depth of a split to consider when calculating importance
        depth_decay : float, optional (default=.0)
            The decay of the importance of a split as a function of depth. The split importance is
            re-weighted by 1 / (1 + depth)**depth_decay.
        """
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count
        cdef double c_depth_decay = depth_decay
        cdef SIZE_t c_max_depth

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        if max_depth is None:
            c_max_depth = self.max_depth
        else:
            c_max_depth = max_depth

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if (max_depth is None) or node.depth <= c_max_depth:
                        left = &nodes[node.left_child]
                        right = &nodes[node.right_child]

                        importance_data[node.feature] += pow(1 + node.depth, -c_depth_decay) * (
                            node.weighted_n_node_samples * node.impurity -
                            left.weighted_n_node_samples * left.impurity -
                            right.weighted_n_node_samples * right.impurity)

                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances
    
    cpdef compute_feature_heterogeneity_importances(self, normalize=True, max_depth=None, depth_decay=.0):
        """Computes the importance of each feature (aka variable) based on amount of
        parameter heterogeneity it creates. Each split adds:
        parent_weight * (left_weight * right_weight) * mean((value_left[k] - value_right[k])**2) / parent_weight**2
        
        Parameters
        ----------
        normalize : bool, optional (default=True)
            Whether to normalize importances to sum to 1
        max_depth : int or None, optional (default=None)
            The max depth of a split to consider when calculating importance
        depth_decay : float, optional (default=.0)
            The decay of the importance of a split as a function of depth. The split importance is
            re-weighted by 1 / (1 + depth)**depth_decay.
        """
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count
        cdef double c_depth_decay = depth_decay
        cdef SIZE_t c_max_depth
        cdef SIZE_t i

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        if max_depth is None:
            c_max_depth = self.max_depth
        else:
            c_max_depth = max_depth

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    if (max_depth is None) or node.depth <= c_max_depth:
                        # ... and node.right_child != _TREE_LEAF:
                        left = &nodes[node.left_child]
                        right = &nodes[node.right_child]
                        # node_value = &self.value[(node - nodes) * self.value_stride]
                        left_value = &self.value[(left - nodes) * self.value_stride]
                        right_value = &self.value[(right - nodes) * self.value_stride]
                        for i in range(self.n_relevant_outputs):
                            importance_data[node.feature] += pow(1 + node.depth, -c_depth_decay) * (
                                left.weighted_n_node_samples * right.weighted_n_node_samples *
                                (left_value[i] - right_value[i])**2 / node.weighted_n_node_samples)
                node += 1

        importances /= (nodes[0].weighted_n_node_samples * self.n_relevant_outputs)

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        # we make it a 3d array even though we only need 2d, for compatibility with sklearn
        # plotting of trees.
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
    
    cdef np.ndarray _get_jac_ndarray(self):
        """Wraps jacobian as a 2-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> (self.n_outputs * self.n_outputs)
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, self.jac)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_precond_ndarray(self):
        """Wraps precond as a 2-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, self.precond)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array.
        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
