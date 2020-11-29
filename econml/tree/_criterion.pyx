# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

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
        free(self.sum_total_val)
        free(self.sum_left_val)
        free(self.sum_right_val)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self, const DTYPE_t[::1, :] Data, const DOUBLE_t[:, ::1] y, 
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples,
                  const DTYPE_t[::1, :] Data_val, const DOUBLE_t[:, ::1] y_val, 
                  DOUBLE_t* sample_weight_val, double weighted_n_samples_val,
                  SIZE_t* samples_val) nogil except -1:
        """Placeholder for a method which will initialize the criterion.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        Data : 2d array-like, dtype=DTYPE_t
            This contains the input variables.
        y : array-like, dtype=DOUBLE_t
            y is a buffer that can store values for n_outputs target variables
        samples : array-like, dtype=SIZE_t
            Indices of the samples in X and y, where samples[start:end]
            correspond to the samples in this node
        start : SIZE_t
            The first sample to be used on this node
        end : SIZE_t
            The last sample used on this node
        """

        pass
    
    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        SIZE_t start_val, SIZE_t end_val) nogil except -1:
        
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
    
    cdef int reverse_reset_train(self) nogil except -1:
        """Reset the criterion at pos=end.
        This method must be implemented by the subclass.
        """
        pass
    
    cdef int reverse_reset_val(self) nogil except -1:
        """Reset the criterion at pos=end.
        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, SIZE_t new_pos, SIZE_t new_pos_val) nogil except -1:
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
    
    cdef double node_impurity_val(self) nogil:
        """Placeholder for calculating the impurity of the node.
        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of samples[start:end]. This is the
        primary function of the criterion class.
        """

        pass

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
    
    cdef void children_impurity_val(self, double* impurity_left,
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

    cdef void node_value_val(self, double* dest) nogil:
        """Placeholder for storing the node value.
        Placeholder for a method which will compute the node value
        of samples[start:end] and save the value into dest.
        Parameters
        ----------
        dest : double pointer
            The memory address where the node value should be stored.
        """

        pass
    
    cdef void node_jacobian_val(self, double* dest) nogil:
        with gil:
            raise AttributeError("Criterion does not support jacobian calculation")
    
    cdef void node_precond_val(self, double* dest) nogil:
        with gil:
            raise AttributeError("Criterion does not support preconditioned value calculation")

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

    cdef double impurity_improvement(self, double impurity) nogil:
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
        impurity : double
            The initial impurity of the node before the split
        Return
        ------
        double : improvement in impurity after the split occurs
        """

        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - (self.weighted_n_right / 
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

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_features, SIZE_t n_samples, SIZE_t n_samples_val):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        n_samples_val : SIZE_t
            The total number of validation samples
        """

        # Default values
        self.n_outputs = n_outputs
        self.n_features = n_features

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.samples_val = NULL
        self.start_val = 0
        self.pos_val = 0
        self.end_val = 0

        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.n_samples_val = n_samples_val
        self.n_node_samples_val = 0
        self.weighted_n_node_samples_val = 0.0
        self.weighted_n_left_val = 0.0
        self.weighted_n_right_val = 0.0

        self.sq_sum_total = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL

        self.sum_total_val = NULL
        self.sum_left_val = NULL
        self.sum_right_val = NULL

        # Allocate memory for the accumulators
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))
        
        self.sum_total_val = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left_val = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right_val = <double*> calloc(n_outputs, sizeof(double))

        if (self.sum_total == NULL or 
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()
        
        if (self.sum_total_val == NULL or 
                self.sum_left_val == NULL or
                self.sum_right_val == NULL):
            raise MemoryError()

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_features, 
                             self.n_samples, self.n_samples_val), self.__getstate__())

    cdef int init(self, const DTYPE_t[::1, :] Data, const DOUBLE_t[:, ::1] y, 
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples,
                  const DTYPE_t[::1, :] Data_val, const DOUBLE_t[:, ::1] y_val, 
                  DOUBLE_t* sample_weight_val, double weighted_n_samples_val,
                  SIZE_t* samples_val) nogil except -1:
        # Initialize fields
        self.Data = Data
        self.y = y
        self.sample_weight = sample_weight
        self.samples = samples
        self.weighted_n_samples = weighted_n_samples

        self.Data_val = Data_val
        self.y_val = y_val
        self.sample_weight_val = sample_weight_val
        self.samples_val = samples_val
        self.weighted_n_samples_val = weighted_n_samples_val

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        SIZE_t start_val, SIZE_t end_val) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_node_samples = 0.

        self.start_val = start_val
        self.end_val = end_val
        self.n_node_samples_val = end_val - start_val
        self.weighted_n_node_samples_val = 0.

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
        
        w = 1.0
        self.sq_sum_total_val = 0.0
        memset(self.sum_total_val, 0, self.n_outputs * sizeof(double))

        for p in range(start_val, end_val):
            i = self.samples_val[p]

            if self.sample_weight_val != NULL:
                w = self.sample_weight_val[i]

            for k in range(self.n_outputs):
                y_ik = self.y_val[i, k]
                w_y_ik = w * y_ik
                self.sum_total_val[k] += w_y_ik
                self.sq_sum_total_val += w_y_ik * y_ik
            
            self.weighted_n_node_samples_val += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        memset(self.sum_left_val, 0, n_bytes)
        memcpy(self.sum_right_val, self.sum_total_val, n_bytes)

        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        self.weighted_n_left_val = 0
        self.weighted_n_right_val = self.weighted_n_node_samples_val
        self.pos_val = self.start_val

        return 0

    cdef int reverse_reset(self) nogil except -1:
        if self.reverse_reset_train() == -1:
            return -1
        if self.reverse_reset_val() == -1:
            return -1
        return 0

    cdef int reverse_reset_train(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)

        self.weighted_n_right = 0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        return 0
    
    cdef int reverse_reset_val(self) nogil except -1:
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right_val, 0, n_bytes)
        memcpy(self.sum_left_val, self.sum_total_val, n_bytes)

        self.weighted_n_right_val = 0
        self.weighted_n_left_val = self.weighted_n_node_samples_val
        self.pos_val = self.end_val

        return 0

    cdef int update(self, SIZE_t new_pos, SIZE_t new_pos_val) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef double* sum_left_val = self.sum_left_val
        cdef double* sum_right_val = self.sum_right_val
        cdef double* sum_total_val = self.sum_total_val

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples_val = self.samples_val
        cdef DOUBLE_t* sample_weight_val = self.sample_weight_val

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t pos_val = self.pos_val
        cdef SIZE_t end_val = self.end_val
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
            self.reverse_reset_train()

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

        # Update val
        w = 1.0
        if (new_pos_val - pos_val) <= (end_val - new_pos_val):
            for p in range(pos_val, new_pos_val):
                i = samples_val[p]

                if sample_weight_val != NULL:
                    w = sample_weight_val[i]

                for k in range(self.n_outputs):
                    sum_left_val[k] += w * self.y_val[i, k]

                self.weighted_n_left_val += w
        else:
            self.reverse_reset_val()

            for p in range(end_val - 1, new_pos_val - 1, -1):
                i = samples_val[p]

                if sample_weight_val != NULL:
                    w = sample_weight_val[i]

                for k in range(self.n_outputs):
                    sum_left_val[k] -= w * self.y_val[i, k]

                self.weighted_n_left_val -= w

        self.weighted_n_right_val = (self.weighted_n_node_samples_val -
                                     self.weighted_n_left_val)
        for k in range(self.n_outputs):
            sum_right_val[k] = sum_total_val[k] - sum_left_val[k]

        self.pos_val = new_pos_val
        return 0

    cdef double node_impurity(self) nogil:
        pass
    
    cdef double node_impurity_val(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass
    
    cdef void children_impurity_val(self, double* impurity_left,
                                    double* impurity_right) nogil:
        pass

    cdef void node_value_val(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""

        cdef SIZE_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total_val[k] / self.weighted_n_node_samples_val


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
    
    cdef double node_impurity_val(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total_val = self.sum_total_val
        cdef double impurity_val
        cdef SIZE_t k

        impurity_val = self.sq_sum_total_val / self.weighted_n_node_samples_val
        for k in range(self.n_outputs):
            impurity_val -= (sum_total_val[k] / self.weighted_n_node_samples_val)**2.0

        return impurity_val / self.n_outputs

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
    
    cdef void children_impurity_val(self, double* impurity_left,
                                    double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""

        cdef DOUBLE_t* sample_weight_val = self.sample_weight_val
        cdef SIZE_t* samples_val = self.samples_val
        cdef SIZE_t pos_val = self.pos_val
        cdef SIZE_t start_val = self.start_val

        cdef double* sum_left_val = self.sum_left_val
        cdef double* sum_right_val = self.sum_right_val
        cdef DOUBLE_t y_ik

        cdef double sq_sum_left_val = 0.0
        cdef double sq_sum_right_val

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0

        for p in range(start_val, pos_val):
            i = samples_val[p]

            if sample_weight_val != NULL:
                w = sample_weight_val[i]

            for k in range(self.n_outputs):
                y_ik = self.y_val[i, k]
                sq_sum_left_val += w * y_ik * y_ik

        sq_sum_right_val = self.sq_sum_total_val - sq_sum_left_val

        impurity_left[0] = sq_sum_left_val / self.weighted_n_left_val
        impurity_right[0] = sq_sum_right_val / self.weighted_n_right_val

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left_val[k] / self.weighted_n_left_val) ** 2.0
            impurity_right[0] -= (sum_right_val[k] / self.weighted_n_right_val) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs
