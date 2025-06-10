# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cpdef bint matinv(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] inv_a) noexcept nogil

cdef bint matinv_(DOUBLE_t* a, DOUBLE_t* inv_a, int m) noexcept nogil

cpdef void lstsq(DOUBLE_t[::1,:] a, DOUBLE_t[::1,:] b, DOUBLE_t[::1, :] sol, bint copy_b=*) noexcept nogil

cdef void lstsq_(DOUBLE_t* a, DOUBLE_t* b, DOUBLE_t* sol, int m, int n, int ldb, int nrhs, bint copy_b=*) noexcept nogil

cpdef void pinv(DOUBLE_t[::1,:] a, DOUBLE_t[::1, :] sol) noexcept nogil

cdef void pinv_(DOUBLE_t* a, DOUBLE_t* sol, int m, int n) noexcept nogil

cpdef double fast_max_eigv(DOUBLE_t[::1, :] A, int reps, UINT32_t random_state) noexcept nogil

cdef double fast_max_eigv_(DOUBLE_t* A, int n, int reps, UINT32_t* random_state) noexcept nogil

cpdef double fast_min_eigv(DOUBLE_t[::1, :] A, int reps, UINT32_t random_state) noexcept nogil

cdef double fast_min_eigv_(DOUBLE_t* A, int n, int reps, UINT32_t* random_state) noexcept nogil