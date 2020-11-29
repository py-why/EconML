# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from cpython cimport Py_INCREF, PyObject, PyTypeObject

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdint cimport SIZE_MAX

import numpy as np
cimport numpy as np
np.import_array()

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

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold',
              'impurity', 'n_node_samples', 'weighted_n_node_samples',
              'impurity_train', 'n_node_samples_train', 'weighted_n_node_samples_train'],
    'formats': [np.intp, np.intp, np.intp, np.float64,
                np.float64, np.intp, np.float64,
                np.float64, np.intp, np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
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

    cpdef build(self, Tree tree, object Data, np.ndarray y,
                object Data_val, np.ndarray y_val,
                SIZE_t n_features,
                np.ndarray sample_weight=None,
                np.ndarray sample_weight_val=None):
        """Build a causal tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object Data, np.ndarray y, np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        
        # since we have to copy and perform linear algebra we will make it fortran for efficiency
        if Data.dtype != DTYPE or not Data.flags.f_contiguous:
            Data = np.asfortranarray(Data, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return Data, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a causal tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease

    cpdef build(self, Tree tree, object Data, np.ndarray y,
                object Data_val, np.ndarray y_val,
                SIZE_t n_features,
                np.ndarray sample_weight=None,
                np.ndarray sample_weight_val=None,
                bint store_jac=False):
        """Build a causal tree from the training set (Data, y, Data_val, y_val). The first n_features
        columns of Data are the training X, the remainder are auxiliary variables that are
        used by the Criterion to calculate the splitting criterion.
        """

        # check input
        Data, y, sample_weight = self._check_input(Data, y, sample_weight)
        Data_val, y_val, sample_weight_val = self._check_input(Data_val, y_val, sample_weight_val)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data
        
        cdef DOUBLE_t* sample_weight_val_ptr = NULL
        if sample_weight_val is not None:
            sample_weight_val_ptr = <DOUBLE_t*> sample_weight_val.data

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
        splitter.init(Data, y, sample_weight_ptr, Data_val, y_val, sample_weight_val_ptr, n_features)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t start_val
        cdef SIZE_t end_val
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SIZE_t n_node_samples_val = splitter.n_samples_val
        cdef double weighted_n_samples_val = splitter.weighted_n_samples_val
        cdef double weighted_n_node_samples_val
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        with nogil:
            # push root node onto stack
            rc = stack.push(0, n_node_samples, 0, n_node_samples_val,
                            0, _TREE_UNDEFINED, 0, INFINITY, INFINITY, 0)
            if rc == -1:
                # got return code -1 - out-of-memory
                with gil:
                    raise MemoryError()

            while not stack.is_empty():
                stack.pop(&stack_record)

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

                n_node_samples = end - start
                n_node_samples_val = end_val - start_val
                splitter.node_reset(start, end, &weighted_n_node_samples,
                                    start_val, end_val, &weighted_n_node_samples_val)

                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf or
                           n_node_samples_val < min_samples_split or
                           n_node_samples_val < 2 * min_samples_leaf or
                           weighted_n_node_samples_val < 2 * min_weight_leaf)

                if first:
                    impurity = splitter.node_impurity()
                    impurity_val = splitter.node_impurity_val()
                    first = 0

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    # If EPSILON=0 in the below comparison, float precision
                    # issues stop splitting, producing trees that are
                    # dissimilar to v0.18
                    is_leaf = (is_leaf or split.pos >= end or
                               split.pos_val >= end_val or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))

                node_id = tree._add_node(parent, is_left, is_leaf, 
                                         split.feature, split.threshold,
                                         impurity, n_node_samples, weighted_n_node_samples,
                                         impurity_val, n_node_samples_val, weighted_n_node_samples_val)

                if node_id == SIZE_MAX:
                    rc = -1
                    break
                
                # Store value for all nodes, to facilitate tree/model
                # inspection and interpretation
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

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

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

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]
    
    property jac:
        def __get__(self):
            if not self.store_jac:
                raise AttributeError("Jacobian computation was not enalbed. Set store_jac=True")
            return self._get_jac_ndarray()[:self.node_count]
    
    property precond:
        def __get__(self):
            if not self.store_jac:
                raise AttributeError("Preconditioned quantity computation was not enalbed. Set store_jac=True")
            return self._get_precond_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, int n_outputs, bint store_jac=False):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.value_stride = n_outputs

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
        return (Tree, (self.n_features, self.n_outputs, self.store_jac), self.__getstate__())

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
        
        value_shape = (node_ndarray.shape[0], self.n_outputs)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

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
                raise ValueError('Did not recognise loaded array layout')
            jac = memcpy(self.jac, (<np.ndarray> jac_ndarray).data,
                         self.capacity * self.jac_stride * sizeof(double))
            precond_ndarray = d['precond']
            precond_shape = (node_ndarray.shape[0], self.n_outputs)
            if (precond_ndarray.shape != precond_shape or
                    not precond_ndarray.flags.c_contiguous or
                    precond_ndarray.dtype != np.float64):
                raise ValueError('Did not recognise loaded array layout')
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
                                             mode='clip')
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

    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
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

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 2-d NumPy array.
        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[2]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr
    
    cdef np.ndarray _get_jac_ndarray(self):
        """Wraps value as a 2-d NumPy array.
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
        """Wraps value as a 2-d NumPy array.
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
