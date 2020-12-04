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

from ._utils cimport matmul_, pinv_

from sklearn.linear_model import Lasso

###################################################################################
# GRF Criteria: Still unfinished
###################################################################################

cdef class LinearMomentGRFCriterion(RegressionCriterion):
    r"""Abstract regression criterion.
    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::
        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t n_samples_val):
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

        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.
        if n_y > 1:
            raise AttributeError("LinearMomentGRFCriterion currently only supports a scalar y")

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.rho = NULL
        self.rho_val = NULL
        self.moment = NULL
        self.moment_val = NULL
        self.parameter = NULL
        self.parameter_val = NULL
        self.parameter_pre = NULL
        self.parameter_pre_val = NULL
        self.J = NULL
        self.J_val = NULL
        self.invJ = NULL
        self.invJ_val = NULL

        # Allocate memory for the proxy for y, which rho in the generalized random forest
        # Since rho is node dependent it needs to be re-calculated and stored for each sample
        # in the node for every node we are investigating
        self.rho = <double*> calloc(n_samples * n_outputs, sizeof(double))
        self.rho_val = <double*> calloc(n_samples_val * n_outputs, sizeof(double))
        self.moment = <double*> calloc(n_samples * n_outputs, sizeof(double))
        self.moment_val = <double*> calloc(n_samples_val * n_outputs, sizeof(double))
        self.parameter = <double *> calloc(n_outputs, sizeof(double))
        self.parameter_val = <double *> calloc(n_outputs, sizeof(double))
        self.parameter_pre = <double *> calloc(n_outputs, sizeof(double))
        self.parameter_pre_val = <double *> calloc(n_outputs, sizeof(double))
        self.J = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.J_val = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.invJ = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.invJ_val = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        
        if (self.rho == NULL or
                self.rho_val == NULL or
                self.moment == NULL or
                self.moment_val == NULL or
                self.parameter == NULL or
                self.parameter_pre == NULL or
                self.parameter_val == NULL or
                self.parameter_pre_val == NULL or
                self.J == NULL or
                self.J_val == NULL or
                self.invJ == NULL or
                self.invJ_val == NULL):
            raise MemoryError()

    def __dealloc__(self):
        # __dealloc__ of parents is also called. Deallocating the extras
        free(self.rho)
        free(self.rho_val)
        free(self.moment)
        free(self.moment_val)
        free(self.parameter)
        free(self.parameter_pre)
        free(self.parameter_val)
        free(self.parameter_pre_val)
        free(self.J)
        free(self.J_val)
        free(self.invJ)
        free(self.invJ_val)

    cdef int init(self, const DOUBLE_t[:, ::1] y, 
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples,
                  const DOUBLE_t[:, ::1] y_val, 
                  DOUBLE_t* sample_weight_val, double weighted_n_samples_val,
                  SIZE_t* samples_val) nogil except -1:
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_y = self.n_y

        self.y = y[:, :n_y]
        self.sample_weight = sample_weight
        self.samples = samples
        self.weighted_n_samples = weighted_n_samples
        self.alpha = y[:, n_y:(n_y + n_outputs)]
        self.pointJ = y[:, (n_y + n_outputs):(n_y + n_outputs + n_outputs * n_outputs)]

        self.y_val = y_val[:, :n_y]
        self.sample_weight_val = sample_weight_val
        self.samples_val = samples_val
        self.weighted_n_samples_val = weighted_n_samples_val
        self.alpha_val = y_val[:, n_y:(n_y + n_outputs)]
        self.pointJ_val = y_val[:, (n_y + n_outputs):(n_y + n_outputs + n_outputs * n_outputs)]

        return 0

    cdef int node_reset_jacobian(self, DOUBLE_t* J, DOUBLE_t* invJ,
                                  const DOUBLE_t[:, ::1] pointJ,
                                  DOUBLE_t* sample_weight,
                                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, j, k, p
        cdef double w
        cdef double weighted_n_node_samples = 0.0
        cdef SIZE_t n_outputs = self.n_outputs

        # Init jacobian matrix to zero
        memset(J, 0, n_outputs * n_outputs * sizeof(DOUBLE_t))

        # Calculate un-normalized empirical jacobian
        for p in range(start, end):
            i = samples[p]
            w = 1.0
            if sample_weight != NULL:
                w = sample_weight[i]
            for j in range(n_outputs):
                for k in range(n_outputs):
                    J[j + k * n_outputs] += w * pointJ[i, j + k * n_outputs]
            weighted_n_node_samples += w

        # Normalize
        for j in range(n_outputs):
            for k in range(n_outputs):
                J[j + k * n_outputs] /= weighted_n_node_samples
        
        # Calcualte inverse and store it in invJ
        pinv_(J, invJ, n_outputs, n_outputs)

        return 0
    
    cdef int node_reset_parameter(self, DOUBLE_t* parameter, DOUBLE_t* parameter_pre,
                                   DOUBLE_t* invJ,
                                   const DOUBLE_t[:, ::1] alpha,
                                   DOUBLE_t* sample_weight,
                                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, j, k, p
        cdef double w
        cdef double weighted_n_node_samples = 0.0
        cdef SIZE_t n_outputs = self.n_outputs

        # init pre-conditioned parameter to zero
        memset(parameter_pre, 0, n_outputs * sizeof(DOUBLE_t))

        for p in range(start, end):
            i = samples[p]
            w = 1.0
            if sample_weight != NULL:
                w = sample_weight[i]
            for j in range(n_outputs):
                parameter_pre[j] += w * alpha[i, j]
            weighted_n_node_samples += w

        for j in range(n_outputs):
            parameter_pre[j] /= weighted_n_node_samples
        
        matmul_(invJ, n_outputs, n_outputs,
                parameter_pre, n_outputs, 1,
                parameter, b'N', b'N')
        
        return 0

    cdef int node_reset_rho(self, DOUBLE_t* rho, DOUBLE_t* moment,
                       DOUBLE_t* parameter, DOUBLE_t* invJ,
                       const DOUBLE_t[:, ::1] pointJ, const DOUBLE_t[:, ::1] alpha,
                       DOUBLE_t* sample_weight, SIZE_t* samples, 
                       SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, j, k, p
        cdef SIZE_t n_outputs = self.n_outputs

        for p in range(start, end):
            i = samples[p]
            for j in range(n_outputs):
                moment[j + i * n_outputs] = - alpha[i, j]
                for k in range(n_outputs):
                    moment[j + i * n_outputs] += pointJ[i, j + k * n_outputs] * parameter[k]
            for j in range(n_outputs):
                rho[j + i * n_outputs] = 0.0
                for k in range(n_outputs):
                    rho[j + i * n_outputs] -= invJ[j + k * n_outputs] * moment[k + i * n_outputs]
        return 0

    cdef int node_reset_sums(self, DOUBLE_t* rho,
                             DOUBLE_t* J,
                             DOUBLE_t* sample_weight, SIZE_t* samples,
                             DOUBLE_t* weighted_n_node_samples,
                             DOUBLE_t* sum_total, DOUBLE_t* sq_sum_total,
                             SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, p, k
        cdef DOUBLE_t y_ik, w_y_ik, w = 1.0
        cdef SIZE_t n_outputs = self.n_outputs

        weighted_n_node_samples[0] = 0.0
        sq_sum_total[0] = 0.0
        memset(sum_total, 0, n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                y_ik = rho[i * n_outputs + k]
                w_y_ik = w * y_ik
                sum_total[k] += w_y_ik
                sq_sum_total[0] += w_y_ik * y_ik

            weighted_n_node_samples[0] += w

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        SIZE_t start_val, SIZE_t end_val) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        
        # Initialize fields
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_node_samples = 0.
        self.node_reset_jacobian(self.J, self.invJ,
                                 self.pointJ,
                                 self.sample_weight, self.samples,
                                 self.start, self.end)
        self.node_reset_parameter(self.parameter, self.parameter_pre,
                                  self.invJ, self.alpha,
                                  self.sample_weight, self.samples,
                                  self.start, self.end)
        self.node_reset_rho(self.rho, self.moment,
                            self.parameter, self.invJ,
                            self.pointJ, self.alpha,
                            self.sample_weight, self.samples,
                            self.start, self.end)
        self.node_reset_sums(self.rho,
                             self.J,
                             self.sample_weight, self.samples,
                             &self.weighted_n_node_samples,
                             self.sum_total, &self.sq_sum_total,
                             self.start, self.end)

        self.start_val = start_val
        self.end_val = end_val
        self.n_node_samples_val = end_val - start_val
        self.weighted_n_node_samples_val = 0.
        self.node_reset_jacobian(self.J_val, self.invJ_val,
                                 self.pointJ_val,
                                 self.sample_weight_val, self.samples_val,
                                 self.start_val, self.end_val)
        self.node_reset_parameter(self.parameter_val, self.parameter_pre_val,
                                  self.invJ_val, self.alpha_val,
                                  self.sample_weight_val, self.samples_val,
                                  self.start_val, self.end_val)
        self.node_reset_rho(self.rho_val, self.moment_val,
                            self.parameter_val, self.invJ_val,
                            self.pointJ_val, self.alpha_val,
                            self.sample_weight_val, self.samples_val,
                            self.start_val, self.end_val)
        self.node_reset_sums(self.rho_val,
                             self.J_val,
                             self.sample_weight_val, self.samples_val,
                             &self.weighted_n_node_samples_val,
                             self.sum_total_val, &self.sq_sum_total_val,
                             self.start_val, self.end_val)

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)

        memset(self.sum_left_val, 0, n_bytes)
        memcpy(self.sum_right_val, self.sum_total_val, n_bytes)

        self.weighted_n_left_val = 0
        self.weighted_n_right_val = self.weighted_n_node_samples_val
        self.pos_val = self.start_val


        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)

        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

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
        cdef SIZE_t n_outputs = self.n_outputs
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

                for k in range(n_outputs):
                    sum_left[k] += w * self.rho[i * n_outputs + k]

                self.weighted_n_left += w
        else:
            self.reverse_reset_train()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] -= w * self.rho[i * n_outputs + k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos

        # Update val
        w = 1.0
        if (new_pos_val - pos_val) <= (end_val - new_pos_val):
            for p in range(pos_val, new_pos_val):
                i = samples_val[p]

                if sample_weight_val != NULL:
                    w = sample_weight_val[i]

                for k in range(n_outputs):
                    sum_left_val[k] += w * self.rho_val[k + i * n_outputs]

                self.weighted_n_left_val += w
        else:
            self.reverse_reset_val()

            for p in range(end_val - 1, new_pos_val - 1, -1):
                i = samples_val[p]

                if sample_weight_val != NULL:
                    w = sample_weight_val[i]

                for k in range(self.n_outputs):
                    sum_left_val[k] -= w * self.rho_val[k + i * n_outputs]

                self.weighted_n_left_val -= w

        self.weighted_n_right_val = (self.weighted_n_node_samples_val -
                                     self.weighted_n_left_val)
        for k in range(self.n_outputs):
            sum_right_val[k] = sum_total_val[k] - sum_left_val[k]

        self.pos_val = new_pos_val
        return 0

    cdef void node_value_val(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        memcpy(dest, self.parameter_val, self.n_outputs * sizeof(double))

    cdef void node_jacobian_val(self, double* dest) nogil:
        """Compute the node Jacobian of samples[start:end] into dest."""
        cdef SIZE_t i, j 
        # Jacobian is stored in f-contiguous format for fortran. We translate it to c-contiguous for
        # user interfacing.
        cdef SIZE_t n_outputs = self.n_outputs
        for i in range(n_outputs):
            for j in range(n_outputs):
                dest[i * n_outputs + j] = self.J_val[i + j * n_outputs]
        
    cdef void node_precond_val(self, double* dest) nogil:
        """Compute the node preconditioned value of samples[start:end] into dest."""
        memcpy(dest, self.parameter_pre_val, self.n_outputs * sizeof(double))

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
                y_ik = self.rho[i * self.n_outputs + k]
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
                y_ik = self.rho_val[i * self.n_outputs + k]
                sq_sum_left_val += w * y_ik * y_ik

        sq_sum_right_val = self.sq_sum_total_val - sq_sum_left_val

        impurity_left[0] = sq_sum_left_val / self.weighted_n_left_val
        impurity_right[0] = sq_sum_right_val / self.weighted_n_right_val

        for k in range(self.n_outputs):
            impurity_left[0] -= (sum_left_val[k] / self.weighted_n_left_val) ** 2.0
            impurity_right[0] -= (sum_right_val[k] / self.weighted_n_right_val) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


cdef class LinearMomentGRFCriterionMSE(LinearMomentGRFCriterion):

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t n_samples_val):

        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.
        self.proxy_children_impurity = True
    
        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.J_left = NULL
        self.J_right = NULL
        self.J_val_left = NULL
        self.J_val_right = NULL

        # Allocate memory for the proxy for y, which rho in the generalized random forest
        # Since rho is node dependent it needs to be re-calculated and stored for each sample
        # in the node for every node we are investigating
        self.J_left = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.J_right = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.J_val_left = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.J_val_right = <double *> calloc(n_outputs * n_outputs, sizeof(double))

        if (self.J_left == NULL or
            self.J_right == NULL or
            self.J_val_left == NULL or
            self.J_val_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        # __dealloc__ of parents is also called. Deallocating the extras
        free(self.J_left)
        free(self.J_right)
        free(self.J_val_left)
        free(self.J_val_right)

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * self.n_outputs * sizeof(double)
        memset(self.J_left, 0, n_bytes)
        memcpy(self.J_right, self.J, n_bytes)
        memset(self.J_val_left, 0, n_bytes)
        memcpy(self.J_val_right, self.J_val, n_bytes)
        return LinearMomentGRFCriterion.reset(self)

    cdef int reverse_reset_train(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * self.n_outputs * sizeof(double)
        memset(self.J_right, 0, n_bytes)
        memcpy(self.J_left, self.J, n_bytes)
        return LinearMomentGRFCriterion.reverse_reset_train(self)
    
    cdef int reverse_reset_val(self) nogil except -1:
        cdef SIZE_t n_bytes = self.n_outputs * self.n_outputs * sizeof(double)
        memset(self.J_val_right, 0, n_bytes)
        memcpy(self.J_val_left, self.J_val, n_bytes)
        return LinearMomentGRFCriterion.reverse_reset_val(self)

    cdef int update(self, SIZE_t new_pos, SIZE_t new_pos_val) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total
        cdef double* J_left = self.J_left
        cdef double* J_right = self.J_right
        cdef double* J = self.J

        cdef double* sum_left_val = self.sum_left_val
        cdef double* sum_right_val = self.sum_right_val
        cdef double* sum_total_val = self.sum_total_val
        cdef double* J_val_left = self.J_val_left
        cdef double* J_val_right = self.J_val_right
        cdef double* J_val = self.J_val

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples_val = self.samples_val
        cdef DOUBLE_t* sample_weight_val = self.sample_weight_val
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double weighted_n_node_samples_val = self.weighted_n_node_samples_val

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t pos_val = self.pos_val
        cdef SIZE_t end_val = self.end_val
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i, p, k, m
        cdef DOUBLE_t w = 1.0

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] += w * self.rho[k + i * n_outputs]
                    for m in range(n_outputs):
                        J_left[k + m * n_outputs] += w * self.pointJ[i, k + m * n_outputs] / weighted_n_node_samples

                self.weighted_n_left += w
        else:
            self.reverse_reset_train()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] -= w * self.rho[k + i * n_outputs]
                    for m in range(n_outputs):
                        J_left[k + m * n_outputs] -= w * self.pointJ[i, k + m * n_outputs] / weighted_n_node_samples

                self.weighted_n_left -= w

        self.weighted_n_right = (weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]
            for m in range(n_outputs):
                J_right[k + m * n_outputs] = J[k + m * n_outputs] - J_left[k + m * n_outputs]

        self.pos = new_pos

        # Update val
        w = 1.0
        if (new_pos_val - pos_val) <= (end_val - new_pos_val):
            for p in range(pos_val, new_pos_val):
                i = samples_val[p]

                if sample_weight_val != NULL:
                    w = sample_weight_val[i]

                for k in range(n_outputs):
                    sum_left_val[k] += w * self.rho_val[k + i * n_outputs]
                    for m in range(n_outputs):
                        J_val_left[k + m * n_outputs] += w * self.pointJ_val[i, k + m * n_outputs] / weighted_n_node_samples_val

                self.weighted_n_left_val += w
        else:
            self.reverse_reset_val()

            for p in range(end_val - 1, new_pos_val - 1, -1):
                i = samples_val[p]

                if sample_weight_val != NULL:
                    w = sample_weight_val[i]

                for k in range(self.n_outputs):
                    sum_left_val[k] -= w * self.rho_val[k + i * n_outputs]
                    for m in range(n_outputs):
                        J_val_left[k + m * n_outputs] -= w * self.pointJ_val[i, k + m * n_outputs] / weighted_n_node_samples_val

                self.weighted_n_left_val -= w

        self.weighted_n_right_val = (weighted_n_node_samples_val -
                                     self.weighted_n_left_val)
        for k in range(self.n_outputs):
            sum_right_val[k] = sum_total_val[k] - sum_left_val[k]
            for m in range(n_outputs):
                J_val_right[k + m * n_outputs] = J_val[k + m * n_outputs] - J_val_left[k + m * n_outputs]

        self.pos_val = new_pos_val

        return 0

    cdef double proxy_node_impurity(self) nogil:
        return LinearMomentGRFCriterion.node_impurity(self)

    cdef double proxy_node_impurity_val(self) nogil:
        return LinearMomentGRFCriterion.node_impurity_val(self)

    cdef double mse_impurity(self, SIZE_t start, SIZE_t end,
                             DOUBLE_t* parameter, DOUBLE_t* J, const DOUBLE_t[:, ::1] y,
                             DOUBLE_t* sample_weight, SIZE_t* samples,
                             double weighted_n_node_samples) nogil except -1:
        # E[y^2] - theta' J theta
        cdef SIZE_t n_y = self.n_y
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i, p, k, m
        cdef DOUBLE_t y_ik, w_y_ik, w = 1.0
        cdef double y_sq_sum_total = 0.0
        cdef double impurity

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_y):
                y_ik = y[i, k]
                w_y_ik = w * y_ik
                y_sq_sum_total += w_y_ik * y_ik

        impurity = y_sq_sum_total / weighted_n_node_samples
        for k in range(n_outputs):
            for m in range(n_outputs):
                impurity -= parameter[k] * parameter[m] * J[k + m * n_outputs]

        return impurity

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        return self.mse_impurity(self.start, self.end, self.parameter, self.J, self.y, 
                                 self.sample_weight, self.samples, self.weighted_n_node_samples)
    
    cdef double node_impurity_val(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        return self.mse_impurity(self.start_val, self.end_val, self.parameter_val, self.J_val, self.y_val, 
                                 self.sample_weight_val, self.samples_val, self.weighted_n_node_samples_val)

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
        cdef DOUBLE_t* J_left = self.J_left
        cdef DOUBLE_t* J_right = self.J_right

        cdef SIZE_t k, m
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0
        for k in range(self.n_outputs):
            for m in range(self.n_outputs):
                proxy_impurity_left += ((sum_left[k] / self.weighted_n_left) * 
                                        (sum_left[m] / self.weighted_n_left) * 
                                        J_left[k + m * self.n_outputs])
                proxy_impurity_right += ((sum_right[k] / self.weighted_n_right) * 
                                         (sum_right[m] / self.weighted_n_right) *
                                         J_right[k + m * self.n_outputs])

        return (proxy_impurity_left * self.weighted_n_node_samples +
                proxy_impurity_right * self.weighted_n_node_samples)
