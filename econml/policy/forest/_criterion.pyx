# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code contains some snippets of code from:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pyx
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset

import numpy as np
cimport numpy as np
np.import_array()

cdef double INFINITY = np.inf

###################################################################################
# GRF Criteria
###################################################################################

cdef class LinearPolicyCriterion(RegressionCriterion):
    r""" 
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_relevant_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t max_node_samples, UINT32_t random_state):
        """Initialize parameters for this criterion. Parent `__cinit__` is always called before children.
        So we only perform extra initializations that were not perfomed by the parent classes.

        Parameters
        ----------
        n_outputs : SIZE_t
            The number of parameters/values to be estimated
        n_relevant_outputs : SIZE_t
            We only care about the first n_relevant_outputs of these parameters/values
        n_features : SIZE_t
            The number of features
        n_y : SIZE_t
            The first n_y columns of the 2d matrix y, contain the raw labels y_{ik}, the rest are auxiliary variables
        n_samples : SIZE_t
            The total number of rows in the 2d matrix y
        max_node_samples : SIZE_t
            The maximum number of samples that can ever be contained in a node
        random_state : UINT32_t
            A random seed for any internal randomness
        """

        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.parameter = NULL

        # Allocate memory for the proxy for y, which rho in the generalized random forest
        # Since rho is node dependent it needs to be re-calculated and stored for each sample
        # in the node for every node we are investigating
        self.parameter = <double *> calloc(n_outputs, sizeof(double))

        if (self.parameter == NULL):
            raise MemoryError()

    def __dealloc__(self):
        # __dealloc__ of parents is also called. Deallocating the extras
        free(self.parameter)

    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""

        cdef SIZE_t i, p, k, argumax
        cdef DOUBLE_t y_ik, w_y_ik, w = 1.0
        cdef DOUBLE_t umax
        cdef SIZE_t n_outputs = self.n_outputs
        
        # Initialize fields
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_node_samples = 0.

        memset(self.sum_total, 0, n_outputs * sizeof(double))

        for p in range(start, end):
            i = self.samples[p]

            if self.sample_weight != NULL:
                w = self.sample_weight[i]
            
            self.weighted_n_node_samples += w

            for k in range(n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik

        memset(self.parameter, 0, n_outputs * sizeof(DOUBLE_t))
        umax = self.sum_total[0]
        argumax = 0
        for k in range(1, n_outputs):
            if self.sum_total[k] > umax:
                umax = self.sum_total[k]
                argumax = k
        self.parameter[argumax] = 1.0

        # Reset to pos=start
        self.reset()
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Return the estimated node parameter of samples[start:end] into dest."""
        memcpy(dest, self.parameter, self.n_outputs * sizeof(double))

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end].
        """
        return - _max(self.sum_total, self.n_outputs) / self.weighted_n_node_samples

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        """
        return (_max(self.sum_left, self.n_outputs) + _max(self.sum_right, self.n_outputs))

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
        left child (samples[start:pos]) and the impurity the right child
        (samples[pos:end]).
        """

        impurity_left[0] = - _max(self.sum_left, self.n_outputs) / self.weighted_n_left
        impurity_right[0] = - _max(self.sum_right, self.n_outputs) / self.weighted_n_right


cdef inline double _max(double* array, SIZE_t n) nogil:
    cdef SIZE_t k
    cdef double max_val
    max_val = - INFINITY 
    for k in range(n):
        if array[k] > max_val:
            max_val = array[k]
    return max_val
