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

from ._utils cimport matinv_, pinv_

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

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_relevant_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t max_node_samples):
        """Initialize parameters for this criterion.
        Parameters
        ----------
        n_outputs : SIZE_t
            The number of targets to be predicted
        n_samples : SIZE_t
            The total number of samples to fit on
        """

        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.
        if n_y > 1:
            raise AttributeError("LinearMomentGRFCriterion currently only supports a scalar y")

        self.proxy_children_impurity = True

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.rho = NULL
        self.moment = NULL
        self.parameter = NULL
        self.parameter_pre = NULL
        self.J = NULL
        self.invJ = NULL
        self.node_index_mapping = NULL

        # Allocate memory for the proxy for y, which rho in the generalized random forest
        # Since rho is node dependent it needs to be re-calculated and stored for each sample
        # in the node for every node we are investigating
        self.rho = <double*> calloc(max_node_samples * n_outputs, sizeof(double))
        self.moment = <double*> calloc(max_node_samples * n_outputs, sizeof(double))
        self.parameter = <double *> calloc(n_outputs, sizeof(double))
        self.parameter_pre = <double *> calloc(n_outputs, sizeof(double))
        self.J = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.invJ = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.node_index_mapping = <SIZE_t *> calloc(n_samples * n_outputs, sizeof(SIZE_t))

        if (self.rho == NULL or
                self.moment == NULL or
                self.parameter == NULL or
                self.parameter_pre == NULL or
                self.J == NULL or
                self.invJ == NULL or
                self.node_index_mapping == NULL):
            raise MemoryError()

    def __dealloc__(self):
        # __dealloc__ of parents is also called. Deallocating the extras
        free(self.rho)
        free(self.moment)
        free(self.parameter)
        free(self.parameter_pre)
        free(self.J)
        free(self.invJ)
        free(self.node_index_mapping)

    cdef int init(self, const DOUBLE_t[:, ::1] y, 
                  DOUBLE_t* sample_weight, double weighted_n_samples,
                  SIZE_t* samples) nogil except -1:
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_y = self.n_y

        self.y = y[:, :n_y]
        self.sample_weight = sample_weight
        self.samples = samples
        self.weighted_n_samples = weighted_n_samples
        self.alpha = y[:, n_y:(n_y + n_outputs)]
        self.pointJ = y[:, (n_y + n_outputs):(n_y + n_outputs + n_outputs * n_outputs)]

        return 0

    cdef int node_reset_jacobian(self, DOUBLE_t* J, DOUBLE_t* invJ, double* weighted_n_node_samples,
                                  const DOUBLE_t[:, ::1] pointJ,
                                  DOUBLE_t* sample_weight,
                                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, j, k, p
        cdef double w, local_weighted_n_node_samples, det
        cdef SIZE_t n_outputs = self.n_outputs
        local_weighted_n_node_samples = 0.0

        # Init jacobian matrix to zero
        memset(J, 0, n_outputs * n_outputs * sizeof(DOUBLE_t))

        # Calculate un-normalized empirical jacobian
        for p in range(start, end):
            i = samples[p]
            w = 1.0
            if sample_weight != NULL:
                w = sample_weight[i]
            for k in range(n_outputs):
                J[k + k * n_outputs] += 1e-6
                for j in range(n_outputs):
                    J[j + k * n_outputs] += w * pointJ[i, j + k * n_outputs]
            local_weighted_n_node_samples += w

        # Normalize
        for k in range(n_outputs):
            for j in range(n_outputs):
                J[j + k * n_outputs] /= local_weighted_n_node_samples
        
        # Calcualte inverse and store it in invJ
        if n_outputs == 1:
            invJ[0] = 1.0 / J[0] if fabs(J[0]) > 0 else 0.0
        elif n_outputs == 2:
            det = J[0] * J[3] - J[1] * J[2]
            if fabs(det) < 1e-6:
                det = 1e-6
            invJ[0] = J[3] / det
            invJ[1] = - J[1] / det
            invJ[2] = - J[2] / det
            invJ[3] = J[1] / det
        else:
            if not matinv_(J, invJ, n_outputs):
                pinv_(J, invJ, n_outputs, n_outputs)
        weighted_n_node_samples[0] = local_weighted_n_node_samples

        return 0

    cdef int node_reset_parameter(self, DOUBLE_t* parameter, DOUBLE_t* parameter_pre,
                                   DOUBLE_t* invJ,
                                   const DOUBLE_t[:, ::1] alpha,
                                   DOUBLE_t* sample_weight, double weighted_n_node_samples,
                                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, j, k, p
        cdef double w
        cdef SIZE_t n_outputs = self.n_outputs

        # init pre-conditioned parameter to zero
        memset(parameter_pre, 0, n_outputs * sizeof(DOUBLE_t))
        memset(parameter, 0, n_outputs * sizeof(DOUBLE_t))

        for p in range(start, end):
            i = samples[p]
            w = 1.0
            if sample_weight != NULL:
                w = sample_weight[i]
            for j in range(n_outputs):
                parameter_pre[j] += w * alpha[i, j] / weighted_n_node_samples

        for j in range(n_outputs):
            for i in range(n_outputs):
                parameter[i] += invJ[i + j * n_outputs] * parameter_pre[j]

        return 0

    cdef int node_reset_rho(self, DOUBLE_t* rho, DOUBLE_t* moment, SIZE_t* node_index_mapping,
                       DOUBLE_t* parameter, DOUBLE_t* invJ,
                       const DOUBLE_t[:, ::1] pointJ, const DOUBLE_t[:, ::1] alpha,
                       DOUBLE_t* sample_weight, SIZE_t* samples, 
                       SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, j, k, p, offset
        cdef SIZE_t n_outputs = self.n_outputs

        for p in range(start, end):
            i = samples[p]
            offset = p - start
            node_index_mapping[i] = offset
            for j in range(n_outputs):
                moment[j + offset * n_outputs] = - alpha[i, j]
                for k in range(n_outputs):
                    moment[j + offset * n_outputs] += pointJ[i, j + k * n_outputs] * parameter[k]
            for j in range(n_outputs):
                rho[j + offset * n_outputs] = 0.0
                for k in range(n_outputs):
                    rho[j + offset * n_outputs] -= invJ[j + k * n_outputs] * moment[k + offset * n_outputs]
        return 0

    cdef int node_reset_sums(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* rho,
                             DOUBLE_t* J,
                             DOUBLE_t* sample_weight, SIZE_t* samples,
                             DOUBLE_t* sum_total, DOUBLE_t* sq_sum_total,
                             DOUBLE_t* y_sq_sum_total,
                             SIZE_t start, SIZE_t end) nogil except -1:
        cdef SIZE_t i, p, k, offset
        cdef DOUBLE_t y_ik, w_y_ik, w = 1.0
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_relevant_outputs = self.n_relevant_outputs
        cdef SIZE_t n_y = self.n_y

        sq_sum_total[0] = 0.0
        y_sq_sum_total[0] = 0.0
        memset(sum_total, 0, n_outputs * sizeof(double))

        for p in range(start, end):
            i = samples[p]
            offset = p - start

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                y_ik = rho[offset * n_outputs + k]
                w_y_ik = w * y_ik
                sum_total[k] += w_y_ik
                if k < n_relevant_outputs:
                    sq_sum_total[0] += w_y_ik * y_ik
            for k in range(n_y):
                y_sq_sum_total[0] += w * (y[i, k]**2)

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        
        # Initialize fields
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_node_samples = 0.
        self.node_reset_jacobian(self.J, self.invJ, &self.weighted_n_node_samples,
                                 self.pointJ,
                                 self.sample_weight, self.samples,
                                 self.start, self.end)
        self.node_reset_parameter(self.parameter, self.parameter_pre,
                                  self.invJ, self.alpha,
                                  self.sample_weight, self.weighted_n_node_samples, self.samples,
                                  self.start, self.end)
        self.node_reset_rho(self.rho, self.moment, self.node_index_mapping,
                            self.parameter, self.invJ,
                            self.pointJ, self.alpha,
                            self.sample_weight, self.samples,
                            self.start, self.end)
        self.node_reset_sums(self.y, self.rho, self.J,
                             self.sample_weight, self.samples,
                             self.sum_total, &self.sq_sum_total, &self.y_sq_sum_total,
                             self.start, self.end)

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
        cdef SIZE_t* node_index_mapping = self.node_index_mapping

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i, p, k, offset
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
                offset = node_index_mapping[i]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] += w * self.rho[offset * n_outputs + k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                offset = node_index_mapping[i]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] -= w * self.rho[offset * n_outputs + k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]

        self.pos = new_pos

        return 0

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        memcpy(dest, self.parameter, self.n_outputs * sizeof(double))

    cdef void node_jacobian(self, double* dest) nogil:
        """Compute the node Jacobian of samples[start:end] into dest."""
        cdef SIZE_t i, j 
        # Jacobian is stored in f-contiguous format for fortran. We translate it to c-contiguous for
        # user interfacing.
        cdef SIZE_t n_outputs = self.n_outputs
        for i in range(n_outputs):
            for j in range(n_outputs):
                dest[i * n_outputs + j] = self.J[i + j * n_outputs]
        
    cdef void node_precond(self, double* dest) nogil:
        """Compute the node preconditioned value of samples[start:end] into dest."""
        memcpy(dest, self.parameter_pre, self.n_outputs * sizeof(double))

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_relevant_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_relevant_outputs

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

        for k in range(self.n_relevant_outputs):
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
        cdef SIZE_t* node_index_mapping = self.node_index_mapping
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef DOUBLE_t y_ik

        cdef double sq_sum_left = 0.0
        cdef double sq_sum_right

        cdef SIZE_t i, p, k, offset
        cdef DOUBLE_t w = 1.0

        for p in range(start, pos):
            i = samples[p]
            offset = node_index_mapping[i]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_relevant_outputs):
                y_ik = self.rho[offset * self.n_outputs + k]
                sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_relevant_outputs):
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_relevant_outputs
        impurity_right[0] /= self.n_relevant_outputs


cdef class LinearMomentGRFCriterionMSE(LinearMomentGRFCriterion):

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_relevant_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t max_node_samples):

        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.
        self.proxy_children_impurity = True
    
        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.J_left = NULL
        self.J_right = NULL

        # Allocate memory for the proxy for y, which rho in the generalized random forest
        # Since rho is node dependent it needs to be re-calculated and stored for each sample
        # in the node for every node we are investigating
        self.J_left = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.J_right = <double *> calloc(n_outputs * n_outputs, sizeof(double))

        if (self.J_left == NULL or
            self.J_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        # __dealloc__ of parents is also called. Deallocating the extras
        free(self.J_left)
        free(self.J_right)

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * self.n_outputs * sizeof(double)
        memset(self.J_left, 0, n_bytes)
        memcpy(self.J_right, self.J, n_bytes)
        return LinearMomentGRFCriterion.reset(self)

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * self.n_outputs * sizeof(double)
        memset(self.J_right, 0, n_bytes)
        memcpy(self.J_left, self.J, n_bytes)
        return LinearMomentGRFCriterion.reverse_reset(self)

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total
        cdef double* J_left = self.J_left
        cdef double* J_right = self.J_right
        cdef double* J = self.J

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* node_index_mapping = self.node_index_mapping
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i, p, k, m, offset
        cdef DOUBLE_t w = 1.0

        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]
                offset = node_index_mapping[i]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] += w * self.rho[k + offset * n_outputs]

                for m in range(n_outputs):
                    for k in range(n_outputs):             
                        J_left[k + m * n_outputs] += w * self.pointJ[i, k + m * n_outputs] / weighted_n_node_samples

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                offset = node_index_mapping[i]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] -= w * self.rho[k + offset * n_outputs]
                
                for m in range(n_outputs):
                    for k in range(n_outputs):
                        J_left[k + m * n_outputs] -= w * self.pointJ[i, k + m * n_outputs] / weighted_n_node_samples

                self.weighted_n_left -= w

        self.weighted_n_right = (weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]
        for m in range(n_outputs):
            for k in range(n_outputs):
                J_right[k + m * n_outputs] = J[k + m * n_outputs] - J_left[k + m * n_outputs]

        self.pos = new_pos

        return 0

    cdef double proxy_node_impurity(self) nogil:
        return LinearMomentGRFCriterion.node_impurity(self)

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef SIZE_t k, m
        cdef DOUBLE_t pk
        cdef double impurity
        cdef SIZE_t n_outputs = self.n_outputs
        impurity = self.y_sq_sum_total / self.weighted_n_node_samples
        for k in range(n_outputs):
            pk = self.parameter[k]
            for m in range(n_outputs):
                impurity -= pk * self.parameter[m] * self.J[k + m * n_outputs]
        return impurity

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
        cdef double slm, srm
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0
        for m in range(self.n_outputs):
            slm = sum_left[m] / self.weighted_n_left
            srm = sum_right[m] / self.weighted_n_right
            for k in range(self.n_outputs):
                proxy_impurity_left += ((sum_left[k] / self.weighted_n_left) * slm * 
                                        J_left[k + m * self.n_outputs])
                proxy_impurity_right += ((sum_right[k] / self.weighted_n_right) * srm *
                                         J_right[k + m * self.n_outputs])

        return (proxy_impurity_left * self.weighted_n_node_samples +
                proxy_impurity_right * self.weighted_n_node_samples)
