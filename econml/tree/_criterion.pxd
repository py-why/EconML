# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
#
# This code is a fork from: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pxd
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

# See _criterion.pyx for implementation details.

import numpy as np
cimport numpy as np

from ._tree cimport DOUBLE_t         # Type of y, sample_weight
from ._tree cimport SIZE_t           # Type for indices and counters
from ._tree cimport UINT32_t         # Unsigned 32 bit integer

cdef class Criterion:
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification
    # and parameter estimates in a tree that solves a moment equation.

    # Internal structures
    cdef bint proxy_children_impurity    # Whether the value returned by children_impurity is only an approximation
    cdef const DOUBLE_t[:, ::1] y        # Values of y (y contains all the variables for node parameter estimation)
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_relevant_outputs       # The first n_relevant_outputs are the ones we care about
    cdef SIZE_t n_features               # Number of features
    cdef SIZE_t n_y                      # The first n_y columns of the y matrix correspond to raw labels.
                                         # The remainder are auxiliary variables required for parameter estimation
    cdef UINT32_t random_state           # A random seed for any internal randomness

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end

    cdef SIZE_t n_samples                # Number of all samples in y (i.e. rows of y)
    cdef SIZE_t max_node_samples         # The maximum number of samples that can ever be contained in a node
                                         # Used for memory space saving, as we need to allocate memory space for
                                         # internal quantities that will store as many values as the number of samples
                                         # in the current node under consideration. Providing this can save space
                                         # allocation time.
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_samples       # Weighted number of samples (in total)
    cdef double weighted_n_node_samples  # Weighted number of samples in the node
    cdef double weighted_n_left          # Weighted number of samples in the left node
    cdef double weighted_n_right         # Weighted number of samples in the right node

    cdef double* sum_total          # For classification criteria, the sum of the
                                    # weighted count of each label. For regression,
                                    # the sum of w*y. sum_total[k] is equal to
                                    # sum_{i=start}^{end-1} w[samples[i]]*y[samples[i], k],
                                    # where k is output index.
    cdef double* sum_left           # Same as above, but for the left side of the split
    cdef double* sum_right          # same as above, but for the right side of the split

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef int init(self, const DOUBLE_t[:, ::1] y, 
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples) nogil except -1
    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1
    cdef int reset(self) nogil except -1
    cdef int reverse_reset(self) nogil except -1
    cdef int update(self, SIZE_t new_pos) nogil except -1
    cdef double node_impurity(self) nogil
    cdef double proxy_node_impurity(self) nogil
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil
    cdef void node_jacobian(self, double* dest) nogil
    cdef void node_precond(self, double* dest) nogil
    cdef double impurity_improvement(self, double impurity) nogil
    cdef double proxy_impurity_improvement(self) nogil
    cdef double min_eig_left(self) nogil
    cdef double min_eig_right(self) nogil


cdef class RegressionCriterion(Criterion):
    """Abstract regression criterion."""

    cdef double sq_sum_total    # Stores sum_i sum_k y_{ik}^2, used for MSE calculation
