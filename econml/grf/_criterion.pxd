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
    cdef const DOUBLE_t[:, ::1] alpha
    cdef const DOUBLE_t[:, ::1] pointJ

    cdef DOUBLE_t* rho
    cdef DOUBLE_t* moment
    cdef DOUBLE_t* parameter
    cdef DOUBLE_t* parameter_pre
    cdef DOUBLE_t* J
    cdef DOUBLE_t* invJ
    cdef SIZE_t* node_index_mapping
    cdef DOUBLE_t y_sq_sum_total

    cdef int node_reset_jacobian(self, DOUBLE_t* J, DOUBLE_t* invJ, double* weighted_n_node_samples,
                                  const DOUBLE_t[:, ::1] pointJ,
                                  DOUBLE_t* sample_weight,
                                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1
    cdef int node_reset_parameter(self, DOUBLE_t* parameter, DOUBLE_t* parameter_pre,
                                   DOUBLE_t* invJ,
                                   const DOUBLE_t[:, ::1] alpha,
                                   DOUBLE_t* sample_weight, double weighted_n_node_samples,
                                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1
    cdef int node_reset_rho(self, DOUBLE_t* rho, DOUBLE_t* moment, SIZE_t* node_index_mapping,
                       DOUBLE_t* parameter, DOUBLE_t* invJ,
                       const DOUBLE_t[:, ::1] pointJ, const DOUBLE_t[:, ::1] alpha,
                       DOUBLE_t* sample_weight, SIZE_t* samples, 
                       SIZE_t start, SIZE_t end) nogil except -1
    cdef int node_reset_sums(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* rho,
                             DOUBLE_t* J,
                             DOUBLE_t* sample_weight, SIZE_t* samples,
                             DOUBLE_t* sum_total, DOUBLE_t* sq_sum_total, DOUBLE_t* y_sq_sum_total,
                             SIZE_t start, SIZE_t end) nogil except -1

cdef class LinearMomentGRFCriterionMSE(LinearMomentGRFCriterion):
    cdef DOUBLE_t* J_left
    cdef DOUBLE_t* J_right
