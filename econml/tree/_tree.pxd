# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
#
# This code is a fork from: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_tree.pxd
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                   # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t depth                         # the depth level of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node on the val set
    SIZE_t n_node_samples                # Number of samples at the node on the val set
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node on the val set
    DOUBLE_t impurity_train                    # Impurity of the node on the training set
    SIZE_t n_node_samples_train                # Number of samples at the node on the training set
    DOUBLE_t weighted_n_node_samples_train     # Weighted number of samples at the node on the training set

cdef class Tree:
    # The Tree object is a binary tree structure constructed by the
    # TreeBuilder. The tree structure is used for predictions and
    # feature importances.

    # Input/Output layout
    cdef public SIZE_t n_features        # Number of features in X
    cdef public SIZE_t n_outputs         # Number of parameters estimated at each node
    cdef public SIZE_t n_relevant_outputs # Prefix of the parameters that we care about
    cdef SIZE_t* n_classes                # Legacy from sklearn for compatibility. Number of classes in classification
    cdef public SIZE_t max_n_classes      # Number of classes for each output coordinate

    # Inner structures: values are stored separately from node structure,
    # since size is determined at runtime.
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes) array of values
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes
    cdef bint store_jac                  # wether to store jacobian and precond information
    cdef double* jac                     # node jacobian in linear moment: J(x) * theta - precond(x) = 0
    cdef SIZE_t jac_stride               # = n_outputs * n_outputs
    cdef double* precond                 # node preconditioned value
    cdef SIZE_t precond_stride           # = n_outputs

    # Methods
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, 
                          double impurity_train, SIZE_t n_node_samples_train,
                          double weighted_n_samples_train,
                          double impurity_val, SIZE_t n_node_samples_val,
                          double weighted_n_samples_val) except -1 nogil
    cdef int _resize(self, SIZE_t capacity) except -1 nogil
    cdef int _resize_c(self, SIZE_t capacity=*) except -1 nogil

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_jac_ndarray(self)
    cdef np.ndarray _get_precond_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef np.ndarray predict(self, object X)
    cpdef np.ndarray predict_jac(self, object X)
    cpdef np.ndarray predict_precond(self, object X)
    cpdef predict_precond_and_jac(self, object X)
    cpdef np.ndarray predict_full(self, object X)

    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply(self, object X)
    cpdef object decision_path(self, object X)
    cdef object _decision_path(self, object X)

    cpdef compute_feature_importances(self, normalize=*, max_depth=*, depth_decay=*)
    cpdef compute_feature_heterogeneity_importances(self, normalize=*, max_depth=*, depth_decay=*)

# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    # The TreeBuilder recursively builds a Tree object from training samples,
    # using a Splitter object for splitting internal nodes and assigning
    # values to leaves.
    #
    # This class controls the various stopping criteria and the node splitting
    # evaluation order, e.g. depth-first or best-first.

    cdef Splitter splitter              # Splitting algorithm

    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef double min_weight_leaf         # Minimum weight in a leaf
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_decrease   # Impurity threshold for early stopping

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray samples_train,
                np.ndarray samples_val,
                np.ndarray sample_weight=*,
                bint store_jac=*)
    cdef _check_input(self, object X, np.ndarray y, np.ndarray sample_weight)
