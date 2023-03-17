# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np

cdef double INFINITY = np.inf

###################################################################################
# Policy Criteria
###################################################################################

cdef class LinearPolicyCriterion(RegressionCriterion):
    r""" 
    """

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
