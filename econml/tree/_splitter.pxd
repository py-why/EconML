# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code is a fork from: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_splitter.pxd

# See _splitter.pyx for details.

import numpy as np
cimport numpy as np

from ._criterion cimport Criterion

from ._tree cimport DTYPE_t          # Type of X
from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport INT32_t          # Signed 32 bit integer
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    SIZE_t pos_val         # Split samples_val array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos_val is >= end_val if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split on train set.
    double impurity_right  # Impurity of the right split on train set.
    double impurity_left_val   # Impurity of the left split on validation set.
    double impurity_right_val  # Impurity of the right split on validation set.

cdef class Splitter:
    # The splitter searches in the input space for a feature and a threshold
    # to split the samples samples[start:end] and samples_val[start_val:end_val].
    #
    # The impurity and value computations are delegated to a criterion and criterion_val object.

    # Internal structures
    cdef public Criterion criterion      # Impurity criterion for train set calculations
    cdef public Criterion criterion_val      # Impurity criterion for val set calculations
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf (on both train and val set)
    cdef public double min_weight_leaf   # Minimum weight in a leaf (on both train and val set)
    cdef public double min_eig_leaf      # Minimum value of proxy for the min eigenvalue of the jacobian (on train)
    cdef public bint min_eig_leaf_on_val # Whether minimum eigenvalue constraint should also be enforced on val
    cdef public double min_balancedness_tol # Tolerance level of how balanced a split can be (in [0, .5])
    cdef public bint honest              # Are we doing train/val honest splitting

    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    cdef SIZE_t* samples_val             # Sample indices in Xval
    cdef SIZE_t n_samples_val            # Xval.shape[0]
    cdef double weighted_n_samples_val   # Weighted number of samples
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t* constant_features       # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t* feature_values         # temp. array holding feature values
    cdef DTYPE_t* feature_values_val     # temp. array holding feature values from validation set

    cdef SIZE_t start                    # Start position for the current node for samples
    cdef SIZE_t end                      # End position for the current node for samples
    cdef SIZE_t start_val                # Start position for the current node for samples_val
    cdef SIZE_t end_val                  # End position for the current node for samples_val

    cdef const DTYPE_t[:, :] X
    cdef const DOUBLE_t[:, ::1] y
    cdef DOUBLE_t* sample_weight

    # The samples vector `samples` is maintained by the Splitter object such
    # that the train samples contained in a node are contiguous. With this setting,
    # `node_split` reorganizes the node samples `samples[start:end]` in two
    # subsets `samples[start:pos]` and `samples[pos:end]`.

    # When `honest=True` then we also store a samples vector `samples_val` that
    # is also maintained by the Splitter that keeps indices of the val set and
    # such that the val samples contained in a node are contiguous in `samples_val`.
    # `node_split` also reoganizes the node samples `samples_val[start_val:end_val]` in two
    # subsets `samples_val[start_val:pos_val]` and `samples_val[pos_val:end_val]`.

    # The 1-d  `features` array of size n_features contains the features
    # indices and allows fast sampling without replacement of features.

    # The 1-d `constant_features` array of size n_features holds in
    # `constant_features[:n_constant_features]` the feature ids with
    # constant values for all the samples in the train set that reached a specific node.
    # The value `n_constant_features` is given by the parent node to its
    # child nodes.  The content of the range `[n_constant_features:]` is left
    # undefined, but preallocated for performance reasons
    # This allows optimization with depth-based tree building.

    # Methods
    cdef int init_sample_inds(self, SIZE_t* samples,
                              const SIZE_t[::1] np_samples,
                              DOUBLE_t* sample_weight,
                              SIZE_t* n_samples, double* weighted_n_samples) nogil except -1

    cdef int init(self, const DTYPE_t[:, :] X, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  const SIZE_t[::1] np_samples_train,
                  const SIZE_t[::1] np_samples_val) nogil except -1

    cdef int node_reset(self, SIZE_t start, SIZE_t end, double* weighted_n_node_samples,
                        SIZE_t start_val, SIZE_t end_val, double* weighted_n_node_samples_val) nogil except -1

    cdef int node_split(self,
                        double impurity,   # Impurity of the node
                        SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1

    cdef void node_value_val(self, double* dest) nogil
    cdef void node_jacobian_val(self, double* dest) nogil
    cdef void node_precond_val(self, double* dest) nogil
    cdef double node_impurity(self) nogil
    cdef double node_impurity_val(self) nogil
    cdef double proxy_node_impurity(self) nogil
    cdef double proxy_node_impurity_val(self) nogil
    cdef bint is_children_impurity_proxy(self) nogil
    