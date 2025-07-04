# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

# See _criterion.pyx for implementation details.

import numpy as np
cimport numpy as np

from ..tree._tree cimport DTYPE_t          # Type of X
from ..tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from ..tree._tree cimport SIZE_t           # Type for indices and counters
from ..tree._tree cimport INT32_t          # Signed 32 bit integer
from ..tree._tree cimport UINT32_t         # Unsigned 32 bit integer

from ..tree._criterion cimport Criterion, RegressionCriterion

cdef class LinearMomentGRFCriterion(RegressionCriterion):
    """ A criterion class that estimates local parameters defined via linear moment equations
    of the form:

    E[ m(J, A; theta(x)) | X=x] = E[ J * theta(x) - A | X=x] = 0

    Calculates impurity based on heterogeneity induced on the estimated parameters, based on the proxy score
    defined in the Generalized Random Forest paper:
        Athey, Susan, Julie Tibshirani, and Stefan Wager. "Generalized random forests."
        The Annals of Statistics 47.2 (2019): 1148-1178
        https://arxiv.org/pdf/1610.01271.pdf
    """
    cdef const DOUBLE_t[:, ::1] alpha   # The A random vector of the linear moment equation for each sample
    cdef const DOUBLE_t[:, ::1] pointJ  # The J random vector of the linear moment equation for each sample

    cdef DOUBLE_t* rho                  # Proxy heterogeneity label: rho = E[J | X in Node]^{-1} m(J, A; theta(Node))
    cdef DOUBLE_t* moment               # Moment for each sample: m(J, A; theta(Node))
    cdef DOUBLE_t* parameter            # Estimated node parameter: theta(Node) = E[J|X in Node]^{-1} E[A|X in Node]
    cdef DOUBLE_t* parameter_pre        # Preconditioned node parameter: theta_pre(Node) = E[A | X in Node]
    cdef DOUBLE_t* J                    # Node average jacobian: J(Node) = E[J | X in Node]
    cdef DOUBLE_t* invJ                 # Inverse of node average jacobian: J(Node)^{-1}
    cdef DOUBLE_t* var_total            # The diagonal elements of J(Node) (used for proxy of min eigenvalue)
    cdef DOUBLE_t* var_left             # The diagonal elements of J(Left) = E[J | X in Left-Child]
    cdef DOUBLE_t* var_right            # The diagonal elements of J(Right) = E[J | X in Right-Child]
    cdef SIZE_t* node_index_mapping     # Used internally to map between sample index in y, with sample index in
                                        # internal memory space that stores rho and moment for each sample
    cdef DOUBLE_t y_sq_sum_total        # The sum of the raw labels y: \sum_i sum_k w_i y_{ik}^2

    cdef int node_reset_jacobian(self, DOUBLE_t* J, DOUBLE_t* invJ, double* weighted_n_node_samples,
                                  const DOUBLE_t[:, ::1] pointJ,
                                  DOUBLE_t* sample_weight,
                                  SIZE_t* samples, SIZE_t start, SIZE_t end) except -1 nogil
    cdef int node_reset_parameter(self, DOUBLE_t* parameter, DOUBLE_t* parameter_pre,
                                   DOUBLE_t* invJ,
                                   const DOUBLE_t[:, ::1] alpha,
                                   DOUBLE_t* sample_weight, double weighted_n_node_samples,
                                   SIZE_t* samples, SIZE_t start, SIZE_t end) except -1 nogil
    cdef int node_reset_rho(self, DOUBLE_t* rho, DOUBLE_t* moment, SIZE_t* node_index_mapping,
                       DOUBLE_t* parameter, DOUBLE_t* invJ, double weighted_n_node_samples,
                       const DOUBLE_t[:, ::1] pointJ, const DOUBLE_t[:, ::1] alpha,
                       DOUBLE_t* sample_weight, SIZE_t* samples, 
                       SIZE_t start, SIZE_t end) except -1 nogil
    cdef int node_reset_sums(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* rho,
                             DOUBLE_t* J,
                             DOUBLE_t* sample_weight, SIZE_t* samples,
                             DOUBLE_t* sum_total, DOUBLE_t* var_total,
                             DOUBLE_t* sq_sum_total, DOUBLE_t* y_sq_sum_total,
                             SIZE_t start, SIZE_t end) except -1 nogil

cdef class LinearMomentGRFCriterionMSE(LinearMomentGRFCriterion):
    cdef DOUBLE_t* J_left           # The jacobian of the left child: J(Left) = E[J | X in Left-Child]
    cdef DOUBLE_t* J_right          # The jacobian of the right child: J(Right) = E[J | X in Right-Child]
    cdef DOUBLE_t* invJ_left           # The jacobian of the left child: J(Left) = E[J | X in Left-Child]
    cdef DOUBLE_t* invJ_right          # The jacobian of the right child: J(Right) = E[J | X in Right-Child]
    cdef DOUBLE_t* parameter_pre_left 
    cdef DOUBLE_t* parameter_pre_right
    cdef DOUBLE_t* parameter_left
    cdef DOUBLE_t* parameter_right

    cdef double _get_min_eigv(self, DOUBLE_t* J_child, DOUBLE_t* var_child,
                              double weighted_n_child) except -1 nogil
