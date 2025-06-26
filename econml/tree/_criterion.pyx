# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
#
# This code is a fork from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/tree/_criterion.pyx
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport safe_realloc
from ._utils cimport sizet_ptr_to_ndarray

cdef class Criterion:
    """Interface for impurity criteria.
    This object stores methods on how to calculate how good a split is using
    different metrics.
    """

    def __dealloc__(self):
        """Destructor."""

        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples,
                  SIZE_t* samples) nogil except -1:
        """Placeholder for a method which will initialize the criterion.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : array_like, dtype=DOUBLE_t
            y is a buffer that can store values for variables required for parameter/value estimation
        sample_weight : array_like, dtype=DOUBLE_t
            The weight of each sample in y
        weighted_n_samples : double
            The total weight of all the samples whose indices are contained in the samples array
        samples : array_like, dtype=SIZE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        """

        pass
    
    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1:
        """ Initialize a node calculation

        Parameters
        ----------
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """
 
        pass

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start.
        This method must be implemented by the subclass.
        """

        pass

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end.
        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left child.
        This updates the collected statistics by moving samples[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.
        Parameters
        ----------
        new_pos : SIZE_t
            New starting index position of the samples in the right child
        """

        pass

    cdef double node_impurity(self) nogil:
        """Placeholder for calculating the impurity of the node.
        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

    cdef double proxy_node_impurity(self) nogil:
        """ A proxy for the node impurity to be used for min_impurity_decrease.
        By default it is equivalent to node_impurity, unless overwritten by child class.
        """
        return self.node_impurity()

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Placeholder for calculating the impurity of children.
        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of samples[start:pos] + the impurity
        of samples[pos:end].
        Parameters
        ----------
        impurity_left : double pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : double pointer
            The memory address where the impurity of the right child should be
            stored
        """

        pass

    cdef void node_value(self, double* dest) nogil:
        """Placeholder for storing the node value.
        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.
        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """

        pass
    
    cdef void node_jacobian(self, double* dest) nogil:
        """Placeholder for storing the node jacobian value.
        Placeholder for a method which will compute the node jacobian value in a linear
        moment J(x) * theta(x) - precond(x) = 0 of samples[start:end] and save the value
        into dest. If not implemented by child, raises an AttributeError if called.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node jacobian should be stored.
        """
        with gil:
            raise AttributeError("Criterion does not support jacobian calculation")
    
    cdef void node_precond(self, double* dest) nogil:
        """Placeholder for storing the node precond value.
        Placeholder for a method which will compute the node precond value in a linear
        moment J(x) * theta(x) - precond(x) = 0 of samples[start:end] and save the value
        into dest. If not implemented by child, raises an AttributeError if called.

        Parameters
        ----------
        dest : double pointer
            The memory address where the node precond should be stored.
        """
        with gil:
            raise AttributeError("Criterion does not support preconditioned value calculation")
    
    cdef double min_eig_left(self) nogil:
        """Placeholder for calculating proxy for minimum eigenvalue of the jacobian
        of the left child of the current split. If not implemented by child, raises an AttributeError if called.
        """
        with gil:
            raise AttributeError("Criterion does not support jacobian and eigenvalue calculation!")
    
    cdef double min_eig_right(self) nogil:
        """Placeholder for calculating proxy for minimum eigenvalue of the jacobian
        of the right child of the current split. If not implemented by child, raises an AttributeError if called.
        """
        with gil:
            raise AttributeError("Criterion does not support jacobian and eigenvalue calculation!")

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        """Compute the improvement in impurity
        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : float64_t
            The initial impurity of the parent node before the split

        impurity_left : float64_t
            The impurity of the left child

        impurity_right : float64_t
            The impurity of the right child

        Return
        ------
        double : improvement in impurity after the split occurs
        """
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right / 
                             self.weighted_n_node_samples * impurity_right)
                          - (self.weighted_n_left / 
                             self.weighted_n_node_samples * impurity_left)))


# =============================================================================
# Regression Criterion
# =============================================================================

cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.
    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::
        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_relevant_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t max_node_samples, UINT32_t random_state):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of parameters/values to be estimated
        n_relevant_outputs : SIZE_t
            We only care about the first n_relevant_outputs of these parameters/values
        n_features : SIZE_t
            The number of features
        n_y : SIZE_t
            The first n_y columns of the 2d matrix y, contain the raw labels y_{ik}, the rest are auxiliary variables
        n_samples : SIZE_t
            The total number of rows in the 2d matrix y
        max_node_samples : SIZE_t
            The maximum number of samples that can ever be contained in a node
        random_state : UINT32_t
            A random seed for any internal randomness
        """

        # Default values
        self.n_outputs = n_outputs
        self.n_relevant_outputs = n_relevant_outputs
        self.n_features = n_features
        self.n_y = n_y
        self.random_state = random_state
        self.proxy_children_impurity = False

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples
        self.max_node_samples = max_node_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or 
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_relevant_outputs, self.n_features, self.n_y,
                             self.n_samples, self.max_node_samples, self.random_state), self.__getstate__())

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples,
                  SIZE_t* samples) nogil except -1:
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.weighted_n_samples = weighted_n_samples

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_node_samples = 0.

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t y_ik
        cdef DOUBLE_t w_y_ik
        cdef DOUBLE_t w = 1.0

        self.sq_sum_total = 0.0
        memset(self.sum_total, 0, self.n_outputs * sizeof(double))

        for p in range(start, end):
            i = self.samples[p]

            if self.sample_weight != NULL:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos

        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.
        MSE = var_left + var_right
    """

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction
        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.
        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef SIZE_t k
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += sum_left[k] * sum_left[k]
            proxy_impurity_right += sum_right[k] * sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs
