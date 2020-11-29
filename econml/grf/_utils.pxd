import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cpdef void matmul(DOUBLE_t[::1,:] a, DOUBLE_t[::1,:] b, 
                  DOUBLE_t[::1,:] out, char* TransA, char* TransB) nogil

cdef void matmul_(DOUBLE_t* a, int lda, int col_a, DOUBLE_t* b, int ldb, int col_b,
                  DOUBLE_t* out, char* TransA, char* TransB) nogil

cpdef void matinv(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] inv_a) nogil

cdef void matinv_(DOUBLE_t* a, DOUBLE_t* inv_a, int m, int n) nogil

cpdef void lstsq(DOUBLE_t[::1,:] a, DOUBLE_t[::1,:] b, DOUBLE_t[::1, :] sol) nogil

cdef void lstsq_(DOUBLE_t* a, DOUBLE_t* b, DOUBLE_t* sol, int m, int n, int nrhs) nogil

cpdef void pinv(DOUBLE_t[::1,:] a, DOUBLE_t[::1, :] sol) nogil

cdef void pinv_(DOUBLE_t* a, DOUBLE_t* sol, int m, int n) nogil
