# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln

import numpy as np
cimport numpy as np
np.import_array()


cdef inline UINT32_t DEFAULT_SEED = 1

# =============================================================================
# Helper functions
# =============================================================================

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    """Generate a pseudo-random np.uint32 from a np.uint32 seed"""
    # seed shouldn't ever be 0.
    if (seed[0] == 0): seed[0] = DEFAULT_SEED

    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    # Note: we must be careful with the final line cast to np.uint32 so that
    # the function behaves consistently across platforms.
    #
    # The following cast might yield different results on different platforms:
    # wrong_cast = <UINT32_t> RAND_R_MAX + 1
    #
    # We can use:
    # good_cast = <UINT32_t>(RAND_R_MAX + 1)
    # or:
    # cdef np.uint32_t another_good_cast = <UINT32_t>RAND_R_MAX + 1
    return seed[0] % <UINT32_t>(RAND_R_MAX + 1)


cdef realloc_ptr safe_realloc(realloc_ptr* p, SIZE_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef SIZE_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience


cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Return copied data as 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data).copy()


cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)


cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """Generate a random double in [low; high)."""
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)


# =============================================================================
# Stack data structure
# =============================================================================

cdef class Stack:
    """A LIFO data structure.
    Attributes
    ----------
    capacity : SIZE_t
        The elements the stack can hold; if more added then ``self.stack_``
        needs to be resized.
    top : SIZE_t
        The number of elements currently on the stack.
    stack : StackRecord pointer
        The stack of records (upward in the stack corresponds to the right).
    """

    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t start_val, SIZE_t end_val,
                  SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity, double impurity_val,
                  SIZE_t n_constant_features) nogil except -1:
        """Push a new element onto the stack.
        Return -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = NULL

        # Resize if capacity not sufficient
        if top >= self.capacity:
            self.capacity *= 2
            # Since safe_realloc can raise MemoryError, use `except -1`
            safe_realloc(&self.stack_, self.capacity)

        stack = self.stack_
        stack[top].start = start
        stack[top].end = end
        stack[top].start_val = start_val
        stack[top].end_val = end_val
        stack[top].depth = depth
        stack[top].parent = parent
        stack[top].is_left = is_left
        stack[top].impurity = impurity
        stack[top].impurity_val = impurity_val
        stack[top].n_constant_features = n_constant_features

        # Increment stack pointer
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        """Remove the top element from the stack and copy to ``res``.
        Returns 0 if pop was successful (and ``res`` is set); -1
        otherwise.
        """
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = self.stack_

        if top <= 0:
            return -1

        res[0] = stack[top - 1]
        self.top = top - 1

        return 0