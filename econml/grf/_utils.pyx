# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

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

from ..tree._utils cimport rand_int


rcond_ = np.finfo(np.float64).eps
cdef inline double RCOND = rcond_



# =============================================================================
# Linear Algebra Functions
# =============================================================================


cpdef bint matinv(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] inv_a) nogil:
    """ Compute matrix inverse and store it in inv_a.
    """
    cdef int m, n
    m = a.shape[0]
    if not (m == a.shape[1]):
        raise ValueError("Can only invert square matrices!")
    return matinv_(&a[0, 0], &inv_a[0, 0], m)

cdef bint matinv_(DOUBLE_t* a, DOUBLE_t* inv_a, int m) nogil:
    """ Compute matrix inverse of matrix a of size (m, m) and store it in inv_a.
    """
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
    """ Compute solution to least squares problem min ||b - a sol||_2^2,
    where a is a matrix of size (m, n), b is (m, nrhs). Store (n, nrhs) solution in sol.
    The memory view b, must have at least max(m, n) rows. If m < n, then pad remainder with zeros.
    If copy_b=True, then b is left unaltered on output. Otherwise b is altered by this call.
    """
    cdef int m, n, nrhs
    m = a.shape[0]
    n = a.shape[1]
    nrhs = b.shape[1]
    ldb = b.shape[0]
    if ldb < max(m, n):
        with gil:
            raise ValueError("Matrix b must have first dimension at least max(a.shape[0], a.shape[1]). "
                             "Please pad with zeros.")
    if (sol.shape[0] != n) or (sol.shape[1] != nrhs):
        with gil:
            raise ValueError("Matrix sol must have dimensions (a.shape[1], b.shape[1]).")
    lstsq_(&a[0, 0], &b[0, 0], &sol[0, 0], m, n, ldb, nrhs, copy_b)


cdef void lstsq_(DOUBLE_t* a, DOUBLE_t* b, DOUBLE_t* sol, int m, int n, int ldb, int nrhs, bint copy_b=True) nogil:
    """ Compute solution to least squares problem min ||b - a sol||_2^2,
    where a is a matrix of size (m, n), b is (m, nrhs). Store (n, nrhs) solution in sol.
    The leading (row) dimension b, must be at least max(m, n). If m < n, then pad remainder with zeros.
    If copy_b=True, then b is left unaltered on output. Otherwise b is altered by this call.
    """
    cdef:
        int lda, rank, info, lwork, n_out
        double rcond
        Py_ssize_t i, j
        #array pointers
        int* jpvt
        double* work
        double* b_copy    
        char* UPLO = 'O' #Any letter other then 'U' or 'L' will copy entire array
    lda = m
    if ldb < max(m, n):
        with gil:
            raise ValueError("Matrix b must have dimension at least max(a.shape[0], a.shape[1]). "
                             "Please pad with zeros.")
    rcond = max(m, n) * RCOND
    jpvt = <int*> calloc(n, sizeof(int))
    lwork = max(min(n, m) + 3 * n + 1, 2 * min(n, m) + nrhs)
    work = <DOUBLE_t*> malloc(lwork * sizeof(DOUBLE_t))

    # TODO. can we avoid all this malloc and copying in our context?
    a_copy = <DOUBLE_t*> calloc(lda * n, sizeof(DOUBLE_t))
    if copy_b:
        b_copy = <DOUBLE_t*> calloc(ldb * nrhs, sizeof(DOUBLE_t))
    else:
        b_copy = b
    try:
        dlacpy(UPLO, &lda, &n, a, &lda, a_copy, &lda)
        if copy_b:
            dlacpy(UPLO, &ldb, &nrhs, b, &ldb, b_copy, &ldb)

        dgelsy(&m, &n, &nrhs, a_copy, &lda, b_copy, &ldb,
               &jpvt[0], &rcond, &rank, &work[0], &lwork, &info)

        for i in range(n):
            for j in range(nrhs):
                sol[i + j * n] = b_copy[i + j * ldb]

    finally:
        free(jpvt)
        free(work)
        free(a_copy)
        if copy_b:
            free(b_copy)

cpdef void pinv(DOUBLE_t[::1,:] a, DOUBLE_t[::1, :] sol) nogil:
    """ Compute pseudo-inverse of (m, n) matrix a and store it in (n, m) matrix sol.
    Matrix a is left un-altered by this call.
    """
    cdef int m = a.shape[0]
    cdef int n = a.shape[1]
    pinv_(&a[0, 0], &sol[0, 0], m, n)

cdef void pinv_(DOUBLE_t* a, DOUBLE_t* sol, int m, int n) nogil:
    """ Compute pseudo-inverse of (m, n) matrix a and store it in (n, m) matrix sol.
    Matrix a is left un-altered by this call.
    """
    # TODO. can we avoid this mallon in our context. Maybe create some fixed memory allocations?
    cdef int ldb = max(m, n)
    cdef double* b = <DOUBLE_t*> calloc(ldb * m, sizeof(double))
    cdef Py_ssize_t i
    for i in range(m):
        b[i + i * ldb] = 1.0
    try:
        lstsq_(a, b, sol, m, n, ldb, m, copy_b=False)

    finally:
        free(b)


cpdef double fast_max_eigv(DOUBLE_t[::1, :] A, int reps, UINT32_t random_state) nogil:
    """ Calculate approximation of maximum eigenvalue via randomized power iteration algorithm.
    See e.g.: http://theory.stanford.edu/~trevisan/expander-online/lecture03.pdf
    Use reps repetition and random seed based on random_state
    """
    return fast_max_eigv_(&A[0, 0], A.shape[0], reps, &random_state)

cdef double fast_max_eigv_(DOUBLE_t* A, int n, int reps, UINT32_t* random_state) nogil:
    """ Calculate approximation of maximum eigenvalue via randomized power iteration algorithm.
    See e.g.: http://theory.stanford.edu/~trevisan/expander-online/lecture03.pdf
    Use reps repetition and random seed based on random_state
    """
    cdef int t, i, j
    cdef double normx, Anormx
    cdef double* xnew
    cdef double* xold
    cdef double* temp
    xnew = NULL
    xold = NULL

    try:
        xnew = <double*> calloc(n, sizeof(double))
        xold = <double*> calloc(n, sizeof(double))

        if xnew == NULL or xold == NULL:
            with gil:
                raise MemoryError()
        for i in range(n):
            xold[i] = (1 - 2*rand_int(0, 2, random_state))
        for t in range(reps):
            for i in range(n):
                xnew[i] = 0
                for j in range(n):
                    xnew[i] += A[i + j * n] * xold[j]
            temp = xold
            xold = xnew
            xnew = temp
        normx = 0
        Anormx = 0
        for i in range(n):
            normx += xnew[i] * xnew[i]
            for j in range(n):
                Anormx += xnew[i] * A[i + j * n] * xnew[j]

        return Anormx / normx
    finally:
        free(xnew)
        free(xold)


cpdef double fast_min_eigv(DOUBLE_t[::1, :] A, int reps, UINT32_t random_state) nogil:
    """ Calculate approximation of minimum eigenvalue via randomized power iteration algorithm.
    See e.g.: http://theory.stanford.edu/~trevisan/expander-online/lecture03.pdf
    Use reps repetition and random seed based on random_state
    """
    return fast_min_eigv_(&A[0, 0], A.shape[0], reps, &random_state)

cdef double fast_min_eigv_(DOUBLE_t* A, int n, int reps, UINT32_t* random_state) nogil:
    """ Calculate approximation of minimum eigenvalue via randomized power iteration algorithm.
    See e.g.: http://theory.stanford.edu/~trevisan/expander-online/lecture03.pdf
    Use reps repetition and random seed based on random_state.
    """
    cdef int t, i, j
    cdef double normx, Anormx
    cdef double* xnew
    cdef double* xold
    cdef double* temp
    cdef double* update
    xnew = NULL
    xold = NULL

    try:
        xnew = <double*> calloc(n, sizeof(double))
        xold = <double*> calloc(n, sizeof(double))
        update = <double*> calloc(n, sizeof(double))

        if xnew == NULL or xold == NULL or update == NULL:
            with gil:
                raise MemoryError()
        for i in range(n):
            xold[i] = (1 - 2*rand_int(0, 2, random_state))
        for t in range(reps):
            lstsq_(A, xold, update, n, n, n, 1, copy_b=False)
            for i in range(n):
                xnew[i] = 0
                for j in range(n):
                    xnew[i] += update[i]
            temp = xold
            xold = xnew
            xnew = temp
        normx = 0
        Anormx = 0
        for i in range(n):
            normx += xnew[i] * xnew[i]
            for j in range(n):
                Anormx += xnew[i] * A[i + j * n] * xnew[j]

        return Anormx / normx
    finally:
        free(xnew)
        free(xold)
        free(update)
