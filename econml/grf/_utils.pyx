# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport calloc
from libc.stdlib cimport realloc
from libc.string cimport memcpy
from libc.math cimport log as ln
from libc.stdlib cimport abort

from scipy.linalg.cython_lapack cimport dgelsy, dgetrf, dgetri, dgecon, dlacpy, dlange


import numpy as np
cimport numpy as np
np.import_array()


rcond_ = np.finfo(np.float64).eps
cdef inline double RCOND = rcond_



# =============================================================================
# Linear Algebra Functions
# =============================================================================


cpdef bint matinv(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] inv_a) nogil:
    cdef int m, n
    m = a.shape[0]
    if not (m == a.shape[1]):
        raise ValueError("Can only invert square matrices!")
    return matinv_(&a[0, 0], &inv_a[0, 0], m)

cdef bint matinv_(DOUBLE_t* a, DOUBLE_t* inv_a, int m) nogil:
    cdef:
        int* pivot
        DOUBLE_t* work
        int lda, INFO, Lwork
        bint failed

    lda = m
    Lwork = m**2
    pivot = <int*> malloc(m * sizeof(int))
    work = <DOUBLE_t*> malloc(Lwork * sizeof(DOUBLE_t))
    failed = False
    if (pivot==NULL or work==NULL):
        with gil:
            raise MemoryError()
    
    try:
        memcpy(inv_a, a, m * m * sizeof(DOUBLE_t))
        
        #Conduct the LU factorization of the array a
        dgetrf(&m, &m, inv_a, &lda, pivot, &INFO)
        if not (INFO == 0):
            failed = True
        else:
            #Now use the LU factorization and the pivot information to invert
            dgetri(&m, inv_a, &lda, pivot, work, &Lwork, &INFO)
            if not (INFO == 0):
                failed = True
    finally:
        free(pivot)
        free(work)

    return (not failed)
    


cpdef void lstsq(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] b, DOUBLE_t[::1, :] sol, bint copy_b=True) nogil:
    cdef int m, n, nrhs
    m = a.shape[0]
    n = a.shape[1]
    nrhs = b.shape[1]
    lstsq_(&a[0, 0], &b[0, 0], &sol[0, 0], m, n, nrhs, copy_b)
    

cdef void lstsq_(DOUBLE_t* a, DOUBLE_t* b, DOUBLE_t* sol, int m, int n, int nrhs, bint copy_b=True) nogil:
    cdef:
        int lda, ldb, rank, info, lwork, n_out
        double rcond
        Py_ssize_t i, j
        #array pointers
        int* jpvt
        double* work
        double* b_copy    
        char* UPLO = 'O' #Any letter other then 'U' or 'L' will copy entire array
    lda = m
    ldb = m
    rcond = max(m, n) * RCOND
    jpvt = <int*> calloc(n, sizeof(int))
    work = <DOUBLE_t*> malloc(sizeof(DOUBLE_t))
    n_out = max(ldb, n)
    # TODO. can we avoid all this malloc and copying in our context?
    a_copy = <DOUBLE_t*> calloc(lda * n, sizeof(DOUBLE_t))
    b_copy = b
    if copy_b:
        b_copy = <DOUBLE_t*> calloc(n_out * nrhs, sizeof(DOUBLE_t))
    try:
        dlacpy(UPLO, &lda, &n, a, &lda, a_copy, &n)
        if copy_b:
            dlacpy(UPLO, &ldb, &nrhs, b, &ldb, b_copy, &n_out)

        # preliminary call to calculate the optimal size of the work array
        lwork = -1
        dgelsy(&m, &n, &nrhs, a_copy, &lda, b_copy, &n_out,
               &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
        if info < 0:
            with gil:
                raise ValueError('illegal value in %d-th argument of internal dgelsy'
                                 % (-info,))
    
        lwork = int(work[0])
        work = <DOUBLE_t*> realloc(work, lwork * sizeof(DOUBLE_t))
        dgelsy(&m, &n, &nrhs, a_copy, &lda, b_copy, &n_out,
               &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)
        
        for i in xrange(nrhs):
            for j in xrange(n):
                sol[j + i*n] = b_copy[j + i*n_out]

    finally:
        free(jpvt)
        free(work)
        free(a_copy)
        if copy_b:
            free(b_copy)

cpdef void pinv(DOUBLE_t[::1,:] a, DOUBLE_t[::1, :] sol) nogil:
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    pinv_(&a[0, 0], &sol[0, 0], m, n)

cdef void pinv_(DOUBLE_t* a, DOUBLE_t* sol, int m, int n) nogil:
    # TODO. can we avoid this mallon in our context. Maybe create some fixed memory allocations?
    cdef double* b = <DOUBLE_t*> calloc(m * m, sizeof(double))
    cdef Py_ssize_t i
    for i in range(m):
        b[i + i*m] = 1.0
    try:
        lstsq_(a, b, sol, m, n, m, copy_b=False)

    finally:
        free(b)

