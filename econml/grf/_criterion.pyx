# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code contains some snippets of code from:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_criterion.pyx
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs, sqrt

import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport matinv_, pinv_

cdef double INFINITY = np.inf

###################################################################################
# GRF Criteria
###################################################################################

cdef class LinearMomentGRFCriterion(RegressionCriterion):
    r""" A criterion class that estimates local parameters defined via linear moment equations
    of the form::

        E[ m(J, A; theta(x)) | X=x] = E[ J * theta(x) - A | X=x] = 0

    Calculates impurity based on heterogeneity induced on the estimated parameters, based on the proxy score
    defined in the Generalized Random Forest paper::

        Athey, Susan, Julie Tibshirani, and Stefan Wager.
        "Generalized random forests." The Annals of Statistics
        47.2 (2019): 1148-1178 https://arxiv.org/pdf/1610.01271.pdf.

    Calculates proxy labels for each sample::

        rho[i] := - J(Node)^{-1} (J[i] * theta(Node) - A[i])
        J(Node) := E[J[i] | X[i] in Node]
        theta(Node) := J(Node)^{-1} E[A[i] | X[i] in Node]

    Then uses as proxy_impurity_improvement for a split (Left, Right) the quantity::

        sum_{k=1}^{n_relevant_outputs} E[rho[i, k] | X[i] in Left]^2 + E[rho[i, k] | X[i] in Right]^2

    Stores as node impurity the quantity::

        sum_{k=1}^{n_relevant_outputs} Var(rho[i, k] | X[i] in Node)
         = sum_{k=1}^{n_relevant_outputs} E[rho[i, k]^2 | X[i] in Node] - E[rho[i, k] | X[i] in Node]^2

    """

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_relevant_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t max_node_samples, UINT32_t random_state):
        """Initialize parameters for this criterion. Parent `__cinit__` is always called before children.
        So we only perform extra initializations that were not perfomed by the parent classes.

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

        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.
        if n_y > 1:
            raise AttributeError("LinearMomentGRFCriterion currently only supports a scalar y")

        self.proxy_children_impurity = True     # The children_impurity() only returns an approximate proxy

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.rho = NULL
        self.moment = NULL
        self.parameter = NULL
        self.parameter_pre = NULL
        self.J = NULL
        self.invJ = NULL
        self.var_total = NULL
        self.var_left = NULL
        self.var_right = NULL
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
        self.var_total = <double *> calloc(n_outputs, sizeof(double))
        self.var_left = <double *> calloc(n_outputs, sizeof(double))
        self.var_right = <double *> calloc(n_outputs, sizeof(double))
        self.node_index_mapping = <SIZE_t *> calloc(n_samples * n_outputs, sizeof(SIZE_t))

        if (self.rho == NULL or
                self.moment == NULL or
                self.parameter == NULL or
                self.parameter_pre == NULL or
                self.J == NULL or
                self.invJ == NULL or
                self.var_total == NULL or
                self.var_left == NULL or
                self.var_right == NULL or
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
        free(self.var_total)
        free(self.var_left)
        free(self.var_right)
        free(self.node_index_mapping)

    cdef int init(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* sample_weight,
                  double weighted_n_samples,
                  SIZE_t* samples) nogil except -1:
        cdef SIZE_t n_features = self.n_features
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_y = self.n_y

        self.y = y[:, :n_y]                     # The first n_y columns of y are the original raw outcome
        self.alpha = y[:, n_y:(n_y + n_outputs)]        # A[i] part of the moment is the next n_outputs columns
        # J[i] part of the moment is the next n_outputs * n_outputs columns, stored in Fortran contiguous format
        self.pointJ = y[:, (n_y + n_outputs):(n_y + n_outputs + n_outputs * n_outputs)]
        self.sample_weight = sample_weight      # Store the sample_weight locally
        self.samples = samples                  # Store the sample index structure used and updated by the splitter
        self.weighted_n_samples = weighted_n_samples    # Store total weight of all samples computed by splitter

        return 0

    cdef int node_reset_jacobian(self, DOUBLE_t* J, DOUBLE_t* invJ, double* weighted_n_node_samples,
                                  const DOUBLE_t[:, ::1] pointJ,
                                  DOUBLE_t* sample_weight,
                                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        """ Calculate the node un-normalized jacobian::

            J(node) := E[J[i] | X[i] in Node] weight(node) = sum_{i in Node} w[i] J[i]

        and its inverse J(node)^{-1} (or revert to pseudo-inverse if the matrix is not invertible). For
        dimensions n_outputs={1, 2}, we also add a small constant of 1e-6 on the diagonals of J, to ensure
        invertibility. For larger dimensions we revert to pseudo-inverse if the matrix is not invertible.

        Parameters
        ----------
        J : DOUBLE_t* of size n_outputs * n_outputs
            On output, contains the calculated un-normalized jacobian J(node) in Fortran contiguous format
        invJ : DOUBLE_t* of size n_outputs * n_outputs
            On output, contains the inverse (or pseudo-inverse) jacobian J(node)^{-1} in Fortran contiguous format
        weighted_n_node_samples : DOUBLE_t* of size 1
            On output, contains the total weight of the samples in the node
        pointJ : DOUBLE_t[:, ::1] of size (n_samples, n_outputs * n_outputs)
            The memory view that contains the J[i] for each sample i
        sample_weight : DOUBLE_t* of size (n_samples,)
            An array taht holds the sample weight w[i] of each sample i
        samples: SIZE_t*
            An array that contains all the indices of samples from pointJ that are handled by the criterion
        start, end: SIZE_t
            The node is defined as the samples corresponding to the set of indices from [`start`:`end`] in
            the samples array.
        """
        cdef SIZE_t i, j, k, p
        cdef double w, local_weighted_n_node_samples, det
        cdef SIZE_t n_outputs = self.n_outputs
        local_weighted_n_node_samples = 0.0

        # Init jacobian matrix to zero
        memset(J, 0, n_outputs * n_outputs * sizeof(DOUBLE_t))

        # Calculate un-normalized empirical jacobian
        w = 1.0
        for p in range(start, end):
            i = samples[p]
            if sample_weight != NULL:
                w = sample_weight[i]
            for k in range(n_outputs):
                for j in range(n_outputs):
                    J[j + k * n_outputs] += w * pointJ[i, j + k * n_outputs]
            local_weighted_n_node_samples += w

        # Calcualte inverse and store it in invJ
        if n_outputs <=2:
            # Fast closed form inverse calculation
            _fast_invJ(J, invJ, n_outputs, clip=1e-6)
        else:
            for k in range(n_outputs):
                J[k + k * n_outputs] += 1e-6    # add small diagonal constant for invertibility

            # Slower matrix inverse via lapack package
            if not matinv_(J, invJ, n_outputs):     # if matrix is invertible use the inverse
                pinv_(J, invJ, n_outputs, n_outputs)    # else calculate pseudo-inverse

            for k in range(n_outputs):
                J[k + k * n_outputs] -= 1e-6    # remove the invertibility constant

        # Update weighted_n_node_samples to the amount calculated by for loop
        weighted_n_node_samples[0] = local_weighted_n_node_samples

        return 0

    cdef int node_reset_parameter(self, DOUBLE_t* parameter, DOUBLE_t* parameter_pre,
                                   DOUBLE_t* invJ,
                                   const DOUBLE_t[:, ::1] alpha,
                                   DOUBLE_t* sample_weight, double weighted_n_node_samples,
                                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        """ Calculate the node parameter and the un-normalized pre-conditioned parameter

            theta_pre(node) := E[A[i] | X[i] in Node] weight(node) = sum_{i in Node} w[i] A[i]
            theta(node) := J(node)^{-1} theta_pre(node)

        Parameters
        ----------
        parameter : DOUBLE_t* of size n_outputs
            On output, contains the calculated node parameter theta(node)
        parameter_pre : DOUBLE_t* of size n_outputs
            On output, contains the calculated node un-normalized pre-conditioned parameter theta_pre(node)
        invJ : DOUBLE_t* of size n_outputs * n_outputs
            On input, contains the calculated node un-normalized jacobian inverse J(node)^{-1} in F-contiguous format
        alpha : DOUBLE_t[:, ::1] of size (n_samples, n_outputs)
            The memory view that contains the A[i] for each sample i
        sample_weight : DOUBLE_t* of size (n_samples,)
            An array taht holds the sample weight w[i] of each sample i
        weighted_n_node_samples : double
            The total weight of the samples in the node
        samples: SIZE_t*
            An array that contains all the indices of samples from pointJ that are handled by the criterion
        start, end: SIZE_t
            The node is defined as the samples corresponding to the set of indices from [`start`:`end`] in
            the samples array.
        """
        cdef SIZE_t i, j, k, p
        cdef double w
        cdef SIZE_t n_outputs = self.n_outputs

        # init parameter and pre-conditioned parameter to zero
        memset(parameter_pre, 0, n_outputs * sizeof(DOUBLE_t))
        memset(parameter, 0, n_outputs * sizeof(DOUBLE_t))
        w = 1.0
        for p in range(start, end):
            i = samples[p]
            if sample_weight != NULL:
                w = sample_weight[i]
            for j in range(n_outputs):
                parameter_pre[j] += w * alpha[i, j]

        for j in range(n_outputs):
            for i in range(n_outputs):
                parameter[i] += invJ[i + j * n_outputs] * parameter_pre[j]

        return 0

    cdef int node_reset_rho(self, DOUBLE_t* rho, DOUBLE_t* moment, SIZE_t* node_index_mapping,
                       DOUBLE_t* parameter, DOUBLE_t* invJ, double weighted_n_node_samples,
                       const DOUBLE_t[:, ::1] pointJ, const DOUBLE_t[:, ::1] alpha, DOUBLE_t* sample_weight,
                       SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        """ Calculate the node proxy labels and moment for each sample i in the node

            moment[i] := J[i] * theta(Node) - A[i]
            rho[i] := - (J(Node) / weight(node))^{-1} * moment[i]

        Parameters
        ----------
        rho : DOUBLE_t* of size `max_node_samples` * `n_outputs`
            On output, the first `n_node_samples * n_outputs` entries contain the rho for each sample in the node
            in a C-continguous manner. The rho for sample with index i is stored in the interval
            [`node_index_mapping[i] * n_outputs` : `(node_index_mapping[i] + 1) * n_outputs`]
            location of the array
        moment : DOUBLE_t* of size `max_node_samples` * `n_outputs`
            On output, the first `n_node_samples * n_outputs` entries contain the moment for each sample in the node
            in a C-continguous manner. The moment for sample with index i is stored in the interval
            [`node_index_mapping[i] * n_outputs` : `(node_index_mapping[i] + 1) * n_outputs`]
            location of the array
        node_index_mapping : SIZE_t* of size `n_samples`
            On output, contains the mapping between the original sample index (i.e. the row in pointJ and alpha)
            with the location in the internal arrays rho and moment
        parameter : DOUBLE_t* of size n_outputs
            On input, contains the calculated node parameter theta(node)
        invJ : DOUBLE_t* of size n_outputs * n_outputs
            On input, contains the calculated unnormalized node jacobian inverse J(node)^{-1} in F-contiguous format
        pointJ : DOUBLE_t[:, ::1] of size (n_samples, n_outputs * n_outputs)
            The memory view that contains the J[i] for each sample i
        alpha : DOUBLE_t[:, ::1] of size (n_samples, n_outputs)
            The memory view that contains the A[i] for each sample i
        sample_weight : DOUBLE_t* of size (n_samples,)
            An array taht holds the sample weight w[i] of each sample i
        samples: SIZE_t*
            An array that contains all the indices of samples from pointJ that are handled by the criterion
        start, end: SIZE_t
            The node is defined as the samples corresponding to the set of indices from [`start`:`end`] in
            the samples array.
        """
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
                    rho[j + offset * n_outputs] -= (invJ[j + k * n_outputs] * moment[k + offset * n_outputs]
                                                     * weighted_n_node_samples)
        return 0

    cdef int node_reset_sums(self, const DOUBLE_t[:, ::1] y, DOUBLE_t* rho,
                             DOUBLE_t* J,
                             DOUBLE_t* sample_weight, SIZE_t* samples,
                             DOUBLE_t* sum_total, DOUBLE_t* var_total, DOUBLE_t* sq_sum_total,
                             DOUBLE_t* y_sq_sum_total,
                             SIZE_t start, SIZE_t end) nogil except -1:
        """ Initialize several sums of quantities that will be useful to speed up the `update()` method:

            for k in {1..n_outputs}: sum_total[k] := sum_{i in Node} w[i] rho[i, k]
            for k in {1..n_outputs}: var_total[k] := J(node)[k, k] (i.e. the k-th diagonal element of J(node))
            sq_sum_total[0] := sum_{i in Node} sum_{j in n_relevant_outputs} w[i] * rho[i, k]^2
            y_sq_sum_total[0] := sum_{i in Node} sum_{j in n_y} w[i] * y[i, k]^2

        Parameters
        ----------
        y : DOUBLE_t[:, ::1] of size `n_samples` * `n_y`
            The original outcome. Used by child class LinearMomentGRFCriterionMSE, but calculated here
            too for code convenience, so as not to re-implement the node_reset_sums method.
        rho : DOUBLE_t* of size `max_node_samples` * `n_outputs`
            On input, the first `n_node_samples * n_outputs` entries contain the rho for each sample in the node
            in a C-continguous manner. The rho for sample with index i is stored in the interval
            [`node_index_mapping[i] * n_outputs` : `(node_index_mapping[i] + 1) * n_outputs`]
            location of the array
        J : DOUBLE_t* of size n_outputs * n_outputs
            On input, contains the calculate jacobian J(node) in Fortran contiguous format
        sample_weight : DOUBLE_t* of size (n_samples,)
            An array taht holds the sample weight w[i] of each sample i
        samples: SIZE_t*
            An array that contains all the indices of samples from pointJ that are handled by the criterion
        sum_total : DOUBLE_t* of size n_outputs
            On output, contains the calculated sum_total[:] vector
        var_total : DOUBLE_t* of size n_outputs
            On output, contains the calculated var_total[:] vector
        sq_sum_total : DOUBLE_t* of size 1
            On output, contains the calculated sq_sum_total
        y_sq_sum_total : DOUBLE_t* of size 1
            On output, contains the calculated y_sq_sum_total
        start, end: SIZE_t
            The node is defined as the samples corresponding to the set of indices from [`start`:`end`] in
            the samples array.
        """
        cdef SIZE_t i, p, k, offset
        cdef DOUBLE_t y_ik, w_y_ik, w = 1.0
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t n_relevant_outputs = self.n_relevant_outputs
        cdef SIZE_t n_y = self.n_y

        sq_sum_total[0] = 0.0
        y_sq_sum_total[0] = 0.0
        memset(sum_total, 0, n_outputs * sizeof(double))
        memset(var_total, 0, n_outputs * sizeof(double))

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

        for k in range(n_outputs):
            var_total[k] = J[k + k * n_outputs]

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end) nogil except -1:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        
        # Initialize fields
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_node_samples = 0.
        # calculate jacobian, inverse jacobian and total node weights
        self.node_reset_jacobian(self.J, self.invJ, &self.weighted_n_node_samples,
                                 self.pointJ,
                                 self.sample_weight, self.samples,
                                 self.start, self.end)
        # calculate node parameter and pre-conditioned parameter
        self.node_reset_parameter(self.parameter, self.parameter_pre,
                                  self.invJ, self.alpha,
                                  self.sample_weight, self.weighted_n_node_samples, self.samples,
                                  self.start, self.end)
        # calculate proxy labels rho and moment for each sample in the node and an index map for associating
        # internal array locations with original sample indices
        self.node_reset_rho(self.rho, self.moment, self.node_index_mapping,
                            self.parameter, self.invJ, self.weighted_n_node_samples,
                            self.pointJ, self.alpha,
                            self.sample_weight, self.samples,
                            self.start, self.end)
        # calculate helper sums that are useful for speeding up `update()`
        self.node_reset_sums(self.y, self.rho, self.J,
                             self.sample_weight, self.samples,
                             self.sum_total, self.var_total, &self.sq_sum_total, &self.y_sq_sum_total,
                             self.start, self.end)

        # Reset to pos=start
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)

        memset(self.sum_left, 0, n_bytes)
        memcpy(self.sum_right, self.sum_total, n_bytes)
        memset(self.var_left, 0, n_bytes)
        memcpy(self.var_right, self.var_total, n_bytes)

        self.weighted_n_left = 0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        return 0

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * sizeof(double)
        memset(self.sum_right, 0, n_bytes)
        memcpy(self.sum_left, self.sum_total, n_bytes)
        memset(self.var_right, 0, n_bytes)
        memcpy(self.var_left, self.var_total, n_bytes)

        self.weighted_n_right = 0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        return 0

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left."""

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* var_total = self.var_total

        cdef SIZE_t* samples = self.samples
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* node_index_mapping = self.node_index_mapping

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef SIZE_t i, p, k, offset
        cdef DOUBLE_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] + sum_right[x] = sum_total[x]
        #           var_left[x] + var_right[x] = var_total[x]
        # and that sum_total and var_total are known, we are going to update
        # sum_left and var_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        
        # The invariance of the update is that:
        #   sum_left[k] = sum_{i in Left} w[i] rho[i, k]
        #   var_left[k] = sum_{i in Left} w[i] pointJ[i, k, k]
        # and similarly for the right child. Notably, the second is un-normalized,
        # so to be used for further calculations it needs to be normalized by the child weight.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]
                offset = node_index_mapping[i]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    # we add w[i] * rho[i, k] to sum_left[k]
                    sum_left[k] += w * self.rho[offset * n_outputs + k]
                    # we add w[i] * J[i, k, k] to var_left[k]
                    var_left[k] += w * self.pointJ[i, k + k * n_outputs]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                offset = node_index_mapping[i]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    # we subtract w[i] * rho[i, k] from sum_left[k]
                    sum_left[k] -= w * self.rho[offset * n_outputs + k]
                    # we subtract w[i] * J[i, k, k] from var_left[k]
                    var_left[k] -= w * self.pointJ[i, k + k * n_outputs]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]
            var_right[k] = var_total[k] - var_left[k]

        self.pos = new_pos

        return 0

    cdef void node_value(self, double* dest) nogil:
        """Return the estimated node parameter of samples[start:end] into dest."""
        memcpy(dest, self.parameter, self.n_outputs * sizeof(double))

    cdef void node_jacobian(self, double* dest) nogil:
        """Return the node normalized Jacobian of samples[start:end] into dest in a C contiguous format."""
        cdef SIZE_t i, j 
        # Jacobian is stored in f-contiguous format for fortran. We translate it to c-contiguous for
        # user interfacing. Moreover, we normalize by weight(node).
        cdef SIZE_t n_outputs = self.n_outputs
        for i in range(n_outputs):
            for j in range(n_outputs):
                dest[i * n_outputs + j] = self.J[i + j * n_outputs] / self.weighted_n_node_samples
        
    cdef void node_precond(self, double* dest) nogil:
        """Return the normalized node preconditioned value of samples[start:end] into dest."""
        cdef SIZE_t i
        for i in range(self.n_outputs):
            dest[i] = self.parameter_pre[i] / self.weighted_n_node_samples

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
        samples[start:end]. We use as node_impurity the proxy quantity:
        sum_{k=1}^{n_relevant_outputs} Var(rho[i, k] | i in Node) / n_relevant_outputs
        = sum_{k=1}^{n_relevant_outputs} (E[rho[i, k]^2 | i in Node] - E[rho[i, k] | i in Node]^2) / n_relevant_outputs
        """

        cdef double* sum_total = self.sum_total
        cdef double impurity
        cdef SIZE_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_relevant_outputs):
            impurity -= (sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_relevant_outputs

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction.
        This method is used to speed up the search for the best split. It is a proxy quantity such that the
        split that maximizes this value also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split. The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        Here we use the quantity:
            sum_{k=1}^{n_relevant_outputs} sum_{child in {Left, Right}} weight(child) * E[rho[i, k] | i in child]^2
        Since:
            E[rho[i, k] | i in child] = sum_{i in child} w[i] rhp[i, k] / weight(child) = sum_child[k] / weight(child)
        This simplifies to:
            sum_{k=1}^{n_relevant_outputs} sum_{child in {Left, Right}} sum_child[k]^2 / weight(child)
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
        (samples[pos:end]). Here we use the proxy child impurity:
            impurity_child[k] = sum_{i in child} w[i] rho[i, k]^2 / weight(child)
                                - (sum_{i in child} w[i] * rho[i, k] / weight(child))^2
            impurity_child = sum_{k in n_relevant_outputs} impurity_child[k] / n_relevant_outputs
        """

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

        # We calculate: sq_sum_left = sum_{i in child} w[i] rho[i, k]^2
        for p in range(start, pos):
            i = samples[p]
            offset = node_index_mapping[i]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_relevant_outputs):
                y_ik = self.rho[offset * self.n_outputs + k]
                sq_sum_left += w * y_ik * y_ik
        # We calculate sq_sum_right = sq_sum_total - sq_sum_left
        sq_sum_right = self.sq_sum_total - sq_sum_left

        # We normalize each sq_sum_child by the weight of that child
        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        # We subtract from the impurity_child, the quantity:
        # sum_{k in n_relevant_outputs} (sum_{i in child} w[i] * rho[i, k] / weight(child))^2
        #   = sum_{k in n_relevant_outputs} (sum_child[k] / weight(child))^2
        for k in range(self.n_relevant_outputs):
            impurity_left[0] -= (sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_relevant_outputs
        impurity_right[0] /= self.n_relevant_outputs

    cdef double min_eig_left(self) nogil:
        """ Calculate proxy for minimum eigenvalue of jacobian of left child. Here we simply
        use the minimum absolute value of the diagonals of the jacobian. This proxy needs to be
        super fast as this calculation happens for every candidate split. So we cannot afford
        anything that is not very simple calculation. We tried a power iteration approximation
        algorithm implemented in `_utils.fast_min_eigv_()`. But even such power iteration algorithm
        is slow as it requires calculating a pseudo-inverse first.
        """
        cdef int i
        cdef double min, abs
        min = fabs(self.var_left[0])
        for i in range(self.n_outputs):
            abs = fabs(self.var_left[i])
            if abs < min:
                min = abs
        # The `update()` method maintains that var_left[k] = J_left[k, k], where J_left is the
        # un-normalized jacobian of the left child. Thus we normalize by weight(left)
        return min / self.weighted_n_left

    cdef double min_eig_right(self) nogil:
        """ Calculate proxy for minimum eigenvalue of jacobian of right child
        (see min_eig_left for more details).
        """
        cdef int i
        cdef double min, abs
        min = fabs(self.var_right[0])
        for i in range(self.n_outputs):
            abs = fabs(self.var_right[i])
            if abs < min:
                min = abs
        return min / self.weighted_n_right


cdef class LinearMomentGRFCriterionMSE(LinearMomentGRFCriterion):

    def __cinit__(self, SIZE_t n_outputs, SIZE_t n_relevant_outputs, SIZE_t n_features, SIZE_t n_y,
                  SIZE_t n_samples, SIZE_t max_node_samples, UINT32_t random_state):
        """ Initialize parameters. See `LinearMomentGRFCriterion.__cinit__`.
        """

        # Most initializations are handled by __cinit__ of RegressionCriterion
        # which is always called in cython. We initialize the extras.
        
        # If n_outputs is small, then we can invert the child jacobian matrix quickly and calculate
        # exact children impurities and exact impurity improvements. Otherwise, we use the heterogeneity
        # based calculations for min impurity decrease so as to avoid matrix inversion when evaluating
        # a split, and we only use a jacobian re-weighted heterogeneity score, as a proxy for the impurity
        # improvement to find the best split.
        if self.n_outputs > 2:
            self.proxy_children_impurity = True
        else:
            self.proxy_children_impurity = False

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.J_left = NULL
        self.J_right = NULL
        if self.n_outputs <= 2:
            self.invJ_left = NULL
            self.invJ_right = NULL
            self.parameter_pre_left = NULL
            self.parameter_pre_right = NULL
            self.parameter_left = NULL
            self.parameter_right = NULL

        # Allocate memory for the proxy for y, which rho in the generalized random forest
        # Since rho is node dependent it needs to be re-calculated and stored for each sample
        # in the node for every node we are investigating
        self.J_left = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        self.J_right = <double *> calloc(n_outputs * n_outputs, sizeof(double))
        if self.n_outputs <= 2:
            self.invJ_left = <double *> calloc(n_outputs * n_outputs, sizeof(double))
            self.invJ_right = <double *> calloc(n_outputs * n_outputs, sizeof(double))
            self.parameter_pre_left = <double *> calloc(n_outputs, sizeof(double))
            self.parameter_pre_right = <double *> calloc(n_outputs, sizeof(double))
            self.parameter_left = <double *> calloc(n_outputs, sizeof(double))
            self.parameter_right = <double *> calloc(n_outputs, sizeof(double))

        if (self.J_left == NULL or
            self.J_right == NULL):
            raise MemoryError()
        if self.n_outputs <= 2 and (self.invJ_left == NULL or
                                    self.invJ_right == NULL or 
                                    self.parameter_pre_left == NULL or
                                    self.parameter_pre_right == NULL or
                                    self.parameter_left == NULL or
                                    self.parameter_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        # __dealloc__ of parents is also called. Deallocating the extras
        free(self.J_left)
        free(self.J_right)
        if self.n_outputs <= 2:
            free(self.invJ_left)
            free(self.invJ_right)
            free(self.parameter_pre_left)
            free(self.parameter_pre_right)
            free(self.parameter_left)
            free(self.parameter_right)

    cdef int reset(self) nogil except -1:
        """Reset the criterion at pos=start."""
        cdef SIZE_t n_bytes = self.n_outputs * self.n_outputs * sizeof(double)
        memset(self.J_left, 0, n_bytes)
        memcpy(self.J_right, self.J, n_bytes)
        if self.n_outputs <= 2:
            n_bytes = self.n_outputs * sizeof(double)
            memset(self.parameter_pre_left, 0, n_bytes)
            memcpy(self.parameter_pre_right, self.parameter_pre, n_bytes)
        return LinearMomentGRFCriterion.reset(self)

    cdef int reverse_reset(self) nogil except -1:
        """Reset the criterion at pos=end."""
        cdef SIZE_t n_bytes = self.n_outputs * self.n_outputs * sizeof(double)
        memset(self.J_right, 0, n_bytes)
        memcpy(self.J_left, self.J, n_bytes)
        if self.n_outputs <= 2:
            n_bytes = self.n_outputs * sizeof(double)
            memset(self.parameter_pre_right, 0, n_bytes)
            memcpy(self.parameter_pre_left, self.parameter_pre, n_bytes)
        return LinearMomentGRFCriterion.reverse_reset(self)

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        """Updated statistics by moving samples[pos:new_pos] to the left.
        We need to re-implement this method, as now we need to keep track of the whole left
        and right jacobian matrices, not just the diagonals.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total
        cdef double* J_left = self.J_left
        cdef double* J_right = self.J_right
        cdef double* J = self.J
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* parameter_pre = self.parameter_pre
        cdef double* parameter_pre_left = self.parameter_pre_left
        cdef double* parameter_pre_right = self.parameter_pre_right

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* node_index_mapping = self.node_index_mapping
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t i, p, k, m, offset
        cdef double slm, srm
        cdef DOUBLE_t w = 1.0

        # The invariance of the update is that:
        #   sum_left[k] = sum_{i in Left} w[i] rho[i, k]
        #   J_left[k, m] = sum_{i in Left} w[i] pointJ[i, k, m]
        #   var_left[k] = sum_{i in Left} w[i] pointJ[i, k, k]
        # and similarly for the right child. Notably, the second and third are un-normalized by the total weight
        # of the samples in the child. Also if n_outputs <= 2, then a parameter_pre_left and right is maintained:
        #   parameter_pre_left[k] = sum_{i in Left} w[i] alpha[i, k]
        # to be used for left and right child parameter estimation.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]
                offset = node_index_mapping[i]

                if sample_weight != NULL:
                    w = sample_weight[i]

                for k in range(n_outputs):
                    sum_left[k] += w * self.rho[offset * n_outputs + k]

                # update the whole jacobian matrix of left and right child not just diagonals
                for m in range(n_outputs):
                    for k in range(n_outputs):             
                        J_left[k + m * n_outputs] += w * self.pointJ[i, k + m * n_outputs]
                    var_left[m] = J_left[m + m * n_outputs]
                    if self.n_outputs <= 2:
                        parameter_pre_left[m] += w * self.alpha[i, m]

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

                # update the whole jacobian matrix of left and right child not just diagonals
                for m in range(n_outputs):
                    for k in range(n_outputs):
                        J_left[k + m * n_outputs] -= w * self.pointJ[i, k + m * n_outputs]
                    var_left[m] = J_left[m + m * n_outputs]
                    if self.n_outputs <= 2:
                        parameter_pre_left[m] -= w * self.alpha[i, m]

                self.weighted_n_left -= w

        self.weighted_n_right = (weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(n_outputs):
            sum_right[k] = sum_total[k] - sum_left[k]
        for m in range(n_outputs):
            for k in range(n_outputs):
                J_right[k + m * n_outputs] = J[k + m * n_outputs] - J_left[k + m * n_outputs]
            var_right[m] = J_right[m + m * n_outputs]
            if self.n_outputs <= 2:
                parameter_pre_right[m] = parameter_pre[m] - parameter_pre_left[m]

        self.pos = new_pos

        return 0

    cdef double proxy_node_impurity(self) nogil:
        """ For n_outputs > 2, we use parent class children impurity, which is slightly different than
        the MSE criterion. Thus for `min_impurity_decrease` purposes we also use the `node_impurity` of
        the parent class and not the MSE impurity.
        """
        if self.n_outputs > 2:
            return LinearMomentGRFCriterion.node_impurity(self)
        else:
            return self.node_impurity()

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of samples[start:end].
        Here we use the MSE criterion:

            impurity(Node) = E[y^2 | X in Node] - theta(node).T (J(node)/weight(node)) theta(node)

        where J(node) / weight(node) is the normalized jacobian of the node: E[J | X in Node].
        - In the case of a causal tree, this is equivalent to the MSE of predicting y:
            impurity(node) = E[( y - <theta(node), T> )^2 | X in Node]
        - In the case of an IV tree, assuming that Z = E[T | Z] (i.e. the optimal instruments are used), then this
          is approximately equivalent to the MSE of the projected treatment:
            impurity(node) ~ E[( y - <theta(node), E[T|Z]> )^2 | X in Node]
          There is a small discrepancy in that the latter has a negative part of
            theta(node)' E[ E[T|Z] E[T|Z].T ] theta(node),
          while we use theta(node).T E[ T E[T|Z].T ] theta(node). In fairly reasonably large samples the two should be
          approximately the same.
        """
        cdef SIZE_t k, m
        cdef DOUBLE_t pk
        cdef double impurity
        cdef SIZE_t n_outputs = self.n_outputs
        impurity = self.y_sq_sum_total / self.weighted_n_node_samples
        for k in range(n_outputs):
            pk = self.parameter[k]
            for m in range(n_outputs):
                impurity -= pk * self.parameter[m] * self.J[k + m * n_outputs] / self.weighted_n_node_samples
        return impurity

    cdef double proxy_impurity_improvement(self) nogil:
        """Compute a proxy of the impurity reduction
        This method is used to speed up the search for the best split. It is a proxy quantity such that the
        split that maximizes this value also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split. The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        If n_outputs <= 2, we use an exact proxy for impurity improvement:
            sum_{child in {Left, Right}} theta(child).T J(child) theta(child).
        where we remind that J(child) is the un-normalized jacobian of the child node.
        Since, computing theta(child) requires inversion of J(child), and since this method is called for
        every candidate split, for n_outputs > 2, we avoid matrix inversion. Instead, we use an
        approximate proxy_impurity_improvement:
            sum_{child in {Left, Right}} E[rho[i] | i in child].T J(child) E[rho[i] | i in child]
        So this is roughly a heterogeneity criterion but penalizes splits with small minimum eigenvalue
        in J(child). The above criterion uses the approximation fact that:
            theta(child) ~ theta(node) + E[rho[i] | i in child]
        and we only use the rho part in the impurity improvement.
        """

        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef DOUBLE_t* J_left = self.J_left
        cdef DOUBLE_t* J_right = self.J_right

        cdef SIZE_t k, m
        cdef double slm, srm
        cdef double proxy_impurity_left = 0.0
        cdef double proxy_impurity_right = 0.0

        if self.n_outputs <= 2:
            # We estimate child params
            _fast_invJ(J_left, self.invJ_left, self.n_outputs, clip=1e-6)
            _fast_invJ(J_right, self.invJ_right, self.n_outputs, clip=1e-6)
            for k in range(self.n_outputs):
                self.parameter_left[k] = 0.0
                self.parameter_right[k] = 0.0
                for m in range(self.n_outputs):
                    self.parameter_left[k] += self.invJ_left[k + m * self.n_outputs] * self.parameter_pre_left[m]
                    self.parameter_right[k] += self.invJ_right[k + m * self.n_outputs] * self.parameter_pre_right[m]
            # We calculate exact improvement of criterion
            proxy_impurity_left = 0.0
            proxy_impurity_right = 0.0
            for m in range(self.n_outputs):
                slm = self.parameter_left[m]
                srm = self.parameter_right[m]
                for k in range(self.n_outputs):
                    proxy_impurity_left += ((self.parameter_left[k]) * slm *
                                            J_left[k + m * self.n_outputs])
                    proxy_impurity_right += ((self.parameter_right[k]) * srm *
                                            J_right[k + m * self.n_outputs])
            return (proxy_impurity_left + proxy_impurity_right)
        else:
            for m in range(self.n_outputs):
                slm = sum_left[m] / self.weighted_n_left
                srm = sum_right[m] / self.weighted_n_right
                for k in range(self.n_outputs):
                    proxy_impurity_left += ((sum_left[k] / self.weighted_n_left) * slm * 
                                            J_left[k + m * self.n_outputs])
                    proxy_impurity_right += ((sum_right[k] / self.weighted_n_right) * srm *
                                            J_right[k + m * self.n_outputs])

            return (proxy_impurity_left + proxy_impurity_right)

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
        left child (samples[start:pos]) and the impurity the right child
        (samples[pos:end]). For n_outputs > 2, we use the parent class children impurity.
        Fro n_outputs <=2, we calculate child parameters and the exact MSE criterion on
        each child node, i.e.:
            impurity(child) = E[y^2 | X in child] - theta(child).T (J(child)/weight(child)) theta(child)
        """

        if self.n_outputs > 2:
            LinearMomentGRFCriterion.children_impurity(self, impurity_left, impurity_right)
            return

        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* node_index_mapping = self.node_index_mapping
        cdef SIZE_t pos = self.pos
        cdef SIZE_t start = self.start
        cdef DOUBLE_t y_ik

        cdef double y_sq_sum_left = 0.0
        cdef double y_sq_sum_right
        cdef double proxy_impurity_left, proxy_impurity_right, slm, srm

        cdef SIZE_t i, p, k, m
        cdef DOUBLE_t w = 1.0

        # We calculate: sq_sum_left = sum_{i in child} w[i] rho[i, k]^2
        for p in range(start, pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(self.n_y):
                y_ik = self.y[i, k]
                y_sq_sum_left += w * y_ik * y_ik
        # We calculate sq_sum_right = sq_sum_total - sq_sum_left
        y_sq_sum_right = self.y_sq_sum_total - y_sq_sum_left

        # We normalize each sq_sum_child by the weight of that child
        impurity_left[0] = y_sq_sum_left / self.weighted_n_left
        impurity_right[0] = y_sq_sum_right / self.weighted_n_right

        _fast_invJ(self.J_left, self.invJ_left, self.n_outputs, clip=1e-6)
        _fast_invJ(self.J_right, self.invJ_right, self.n_outputs, clip=1e-6)
        for k in range(self.n_outputs):
            self.parameter_left[k] = 0.0
            self.parameter_right[k] = 0.0
            for m in range(self.n_outputs):
                self.parameter_left[k] += self.invJ_left[k + m * self.n_outputs] * self.parameter_pre_left[m]
                self.parameter_right[k] += self.invJ_right[k + m * self.n_outputs] * self.parameter_pre_right[m]
        proxy_impurity_left = 0.0
        proxy_impurity_right = 0.0
        for m in range(self.n_outputs):
            slm = self.parameter_left[m]
            srm = self.parameter_right[m]
            for k in range(self.n_outputs):
                proxy_impurity_left += ((self.parameter_left[k]) * slm *
                                        self.J_left[k + m * self.n_outputs])
                proxy_impurity_right += ((self.parameter_right[k]) * srm *
                                        self.J_right[k + m * self.n_outputs])
        impurity_left[0] -= proxy_impurity_left / self.weighted_n_left
        impurity_right[0] -= proxy_impurity_right / self.weighted_n_right


    cdef double _get_min_eigv(self, DOUBLE_t* J_child, DOUBLE_t* var_child,
                              double weighted_n_child) nogil except -1:
        """ Since in this class we keep track of the whole jacobian of each child, we can
        afford a better proxy for the minimum eigenvalue. Here we propose that on top of the minimum of
        diagonal entries, we also take the minimum of that and the sqrt of every pairwise determinant.
        This roughly corresponds to pairwise Pearson correlation coefficient for each pair of variables
        if the jacobian corresponds to a covariance matrix. If for instance, the jacobian is the co-variance
        of residualized treatment variables, i.e. J[i, j] = E[ Tres[i] * Tres[j] | X in node], then
        the determinant is:

            E[Tres[i]^2] E[Tres[j]^2] - E[Tres[i] * Tres[j]]^2
                = E[Tres[i]^2] E[Tres[j]^2] (1 -  E[Tres[i] * Tres[j]]^2 / E[Tres[i]^2] E[Tres[j]^2])
                = sigma(Tres[i])^2 * sigma(Tres[j])^2 (1 - rho(Tres[i], Tres[j])^2)
            where rho(X, Y) is the pearson correlation coefficient of two random variables X, Y.

        This is close to zero if any two treatment variables are very co-linear for samples in the node,
        which implies that there we wont be able to separate the effect of each individual variable
        locally within the node. This quantity needs to be larger or equal to the parameter `min_eig_leaf`,
        for the split to be considered valid.

        Example. In the causal tree setting, if we have a single treatment variable that *is not residualized*,
        but we append a constant treatment of 1 (i.e. by `fit_intercept=True`), then the pairwise determinant with
        that constant becomes:
            sqrt(E[T[i]^2] - E[T[i]]^2) = sqrt(Var(T[i])) < `min_eig_leaf`
        Thus this constraint implies that we must have a standard deviation of T within the leaf of at least
        `min_eig_leaf`.

        If we have a single treatment variable that *is residualized*, and we append a constant
        treatment of 1 (i.e. by `fit_intercept=True`), then the pairwise determinant with that constant becomes:
            sqrt(E[Tres[i]^2] - E[Tres[i]]^2) = sqrt(Var[Tres[i]^2]) ~ sqrt(Var(T[i])) > `min_eig_leaf`
        Thus this constraint is for `min_eig_leaf` <= 1 redundant as compared to the diagonal constraint:
            E[Tres[i]^2] ~ Var(T[i]) > `min_eig_leaf`
        in many samples when E[Tres[i]]=0 and the T residual model is relatively unbiased.
        """
        cdef SIZE_t i, j, n_outputs
        cdef double min, abs, rescale, Jii, Jij, Jji, Jjj
        n_outputs = self.n_outputs
        # take the min of the square of the diagonals and all pairwise determinants.
        # then at the end we take the root of the min.
        min = var_child[0]
        min *= min
        for i in range(n_outputs):
            abs = fabs(var_child[i])
            abs *= abs
            if abs < min:
                min = abs
            Jii = J_child[i + i * n_outputs]
            for j in range(n_outputs):
                if j != i:
                    Jij = J_child[i + j * n_outputs]
                    Jji = J_child[j + i * n_outputs]
                    Jjj = J_child[j + j * n_outputs]
                    # Since this contains products of un-scaled J_child quantities, we need to multiply by the
                    # square of the rescale factor
                    abs = fabs(Jii * Jjj - Jij * Jji)
                    if abs < min:
                        min = abs
        # Since J_child and var_child that are maintained by `update()` contain variables that are
        # un-normalized, we normalize by child weight.
        return sqrt(min) / weighted_n_child

    cdef double min_eig_left(self) nogil:
        return self._get_min_eigv(self.J_left, self.var_left, self.weighted_n_left)
    
    cdef double min_eig_right(self) nogil:
        return self._get_min_eigv(self.J_right, self.var_right, self.weighted_n_right)


cdef void _fast_invJ(DOUBLE_t* J, DOUBLE_t* invJ, SIZE_t n, double clip) nogil:
    cdef double det
    if n == 1:
        invJ[0] = 1.0 / J[0] if fabs(J[0]) >= clip else 1/clip     # Explicit inverse calculation
    elif n == 2:
        # Explicit inverse calculation
        det = J[0] * J[3] - J[1] * J[2]
        if fabs(det) < clip:
            det = clip
        invJ[0] = J[3] / det
        invJ[1] = - J[1] / det
        invJ[2] = - J[2] / det
        invJ[3] = J[0] / det
