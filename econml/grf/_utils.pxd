import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cpdef bint matinv(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] inv_a) nogil

cdef bint matinv_(DOUBLE_t* a, DOUBLE_t* inv_a, int m) nogil

cpdef void lstsq(DOUBLE_t[::1,:] a, DOUBLE_t[::1,:] b, DOUBLE_t[::1, :] sol, bint copy_b=*) nogil

cdef void lstsq_(DOUBLE_t* a, DOUBLE_t* b, DOUBLE_t* sol, int m, int n, int nrhs, bint copy_b=*) nogil

cpdef void pinv(DOUBLE_t[::1,:] a, DOUBLE_t[::1, :] sol) nogil

cdef void pinv_(DOUBLE_t* a, DOUBLE_t* sol, int m, int n) nogil
