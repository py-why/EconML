# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport calloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln
from libc.stdlib cimport abort

from scipy.linalg.cython_blas cimport dgemm
from scipy.linalg.cython_lapack cimport dgelsy, dgetrf, dgetri, dgecon, dlacpy, dlange


import numpy as np
cimport numpy as np
np.import_array()


rcond_ = np.finfo(np.float64).eps
cdef inline double RCOND = rcond_



# =============================================================================
# Linear Algebra Functions
# =============================================================================


cpdef void matmul(DOUBLE_t[::1,:] a, DOUBLE_t[::1,:] b, 
              DOUBLE_t[::1,:] out, char* TransA, char* TransB) nogil:
    cdef int lda, col_a, ldb, col_b
    lda = a.shape[0]
    col_a = a.shape[1]
    ldb = b.shape[0]
    col_b = b.shape[1]
    matmul_(&a[0, 0], lda, col_a, &b[0, 0], ldb, col_b, &out[0, 0], TransA, TransB)

cdef void matmul_(DOUBLE_t* a, int lda, int col_a, DOUBLE_t* b, int ldb, int col_b,
              DOUBLE_t* out, char* TransA, char* TransB) nogil:
    
    cdef:
        char* Trans='T'
        char* No_Trans='N'
        int m, n, k, ldc
        double alpha, beta
    
    #dimensions of arrays post operation (after transposing, or not)
    if TransA[0]==Trans[0] and TransB[0]==No_Trans[0]:
        m = col_a; n = col_b ; k = lda
    elif TransB[0]==Trans[0] and TransA[0]==No_Trans[0]:
        m = lda; n = ldb ; k = col_a
    elif TransA[0]==Trans[0] and TransB[0]==Trans[0]:
        m = col_a; n = ldb ; k = lda
    else: 
        m = lda; n = col_b ; k = ldb
    
    #leading dimension of c from above
    ldc = m
    
    #scalars associated with C = beta*op(A)*op(B) + alpha*C
    alpha = 1.0
    beta = 0.0
    
    #Fortran BLAS function for calculating the multiplication of arrays
    dgemm(TransA, TransB, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, out, &ldc)


cpdef void matinv(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] inv_a) nogil:
    cdef int m, n
    m = a.shape[0]
    n = a.shape[1]
    matinv_(&a[0, 0], &inv_a[0, 0], m, n)

cdef void matinv_(DOUBLE_t* a, DOUBLE_t* inv_a, int m, int n) nogil:
    cdef:
        #array pointers
        int* pivot
        int* IWORK
        DOUBLE_t* work
        DOUBLE_t* work_dgecon
        
        #variables characterizing the arrays
        int lda, INFO, Lwork
        double ANORM, RCOND, sing_tol = 1e-12
        
        #setting for the functions
        char* NORM = 'I' #The infinity norm (consistent use between dlange & dgecon)
        char* UPLO = 'O' #Any letter other then 'U' or 'L' will copy entire array
    
    #Dimensions of arrays
    lda = m
    # TODO. maybe we should be doing a prelim call to get the optimal work size
    Lwork = m**2

    #manually allocate memory
    #Note: 'work' can be used by both dlange and dgetri as its construction is the same
    pivot = <int*> malloc(m * sizeof(int))
    IWORK = <int*> malloc(n * sizeof(int))
    work = <DOUBLE_t*> malloc(Lwork * sizeof(DOUBLE_t))
    work_dgecon = <DOUBLE_t*> malloc(4*n * sizeof(DOUBLE_t))
    
    if (pivot==NULL or IWORK==NULL
        or work==NULL or work_dgecon==NULL): 
            abort()
    
    try:
        #First, create a copy of the array to invert
        dlacpy(UPLO, &m, &n, a, &lda, inv_a, &lda)
        
        #Next, compute the NORM(a) on the a_copy to preserve array a
        ANORM = dlange(NORM, &m, &n, inv_a, &lda, work)
        
        #Conduct the LU factorization of the array a
        dgetrf(&m, &n, inv_a, &lda, pivot, &INFO)
        
        #Check that LU factorization was successful:
        if INFO==0:
        
            #Now use dgecon to check that the array is invertible (non-singular)
            dgecon(NORM, &n, inv_a, &lda, &ANORM, &RCOND, work_dgecon, IWORK, &INFO)
            
            if RCOND > sing_tol:
       
                #Now use the LU factorization and the pivot information to invert
                dgetri(&n, inv_a, &lda, pivot, work, &Lwork, &INFO)
            
            else: 
                with gil:
                    raise ValueError("Array is singular and will not be inverted. Condition number: %lf" % RCOND)
            
        else: 
            with gil:
                raise ValueError("The factor U is singular")
        
    finally:
        free(pivot)
        free(work)
        free(work_dgecon)
        free(IWORK)


cpdef void lstsq(DOUBLE_t[::1, :] a, DOUBLE_t[::1, :] b, DOUBLE_t[::1, :] sol) nogil:
    cdef int m, n, nrhs
    m = a.shape[0]
    n = a.shape[1]
    nrhs = b.shape[1]
    lstsq_(&a[0, 0], &b[0, 0], &sol[0, 0], m, n, nrhs)
    

cdef void lstsq_(DOUBLE_t* a, DOUBLE_t* b, DOUBLE_t* sol, int m, int n, int nrhs) nogil:
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
    b_copy = <DOUBLE_t*> calloc(n_out * nrhs, sizeof(DOUBLE_t))
    try:
        dlacpy(UPLO, &lda, &n, a, &lda, a_copy, &n)
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
        lstsq_(a, b, sol, m, n, m)

    finally:
        free(b)

