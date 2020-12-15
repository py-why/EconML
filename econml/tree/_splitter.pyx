# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# This code is a fork from: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/tree/_splitter.pyx

from ._criterion cimport Criterion

from libc.stdlib cimport free
from libc.string cimport memcpy
from libc.math cimport floor

import copy
import numpy as np
cimport numpy as np
np.import_array()

from ._utils cimport log
from ._utils cimport rand_int
from ._utils cimport rand_uniform
from ._utils cimport RAND_R_MAX
from ._utils cimport safe_realloc

cdef double INFINITY = np.inf

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos, SIZE_t start_pos_val) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.impurity_left_val = INFINITY
    self.impurity_right_val = INFINITY
    self.pos = start_pos
    self.pos_val = start_pos_val
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


cdef class Splitter:
    """Abstract splitter class.
    Splitters are called by tree builders to find the best splits, one split at a time.
    """

    def __cinit__(self, Criterion criterion, Criterion criterion_val,
                  SIZE_t max_features, SIZE_t min_samples_leaf, double min_weight_leaf,
                  DTYPE_t min_balancedness_tol, bint honest, double min_eig_leaf,
                  UINT32_t random_state):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split on the train set.
        criterion_val : Criterion
            The criterion to be used to calculate quantities related to split quality on the val set.
        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.
        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not considered.
            Constraint is enforced on both the train and val set separately.
        min_weight_leaf : SIZE_t
            The minimal number of total weight of samples each leaf can have, where splits
            which would result in having less weight in a leaf are not considered.
            Constraint is enforced on both train and val set separately.
        min_balancedness_tol : DTYPE_t
            Tolerance level of how balanced a split can be (in [0, .5]) with
            0 meaning split has to be fully balanced and .5 meaning no balancedness
            constraint. Constraint is enforced on both train and val set separately.
        honest : bint
            Whether we should do honest splitting, i.e. train and val set are different.
        min_eig_leaf : double
            The minimum value of computationally fast proxies for the minimum eigenvalue
            of the jacobian J(x) of a node in the case of linear moment equation trees:
            J(x) * theta(x) - precond(x) = 0. The proxy used is defered and must be implemented
            by the criterion objects, if min_eig_leaf >= 0.0, via the methods `min_eig_left()`
            and `min_eig_right()` for the minimum eigenvalue proxy of the left and right child
            correspondingly.
        random_state : UINT32_t
            The user inputed random seed to be used for pseudo-randomness
        """

        self.criterion = criterion
        if honest:
            self.criterion_val = criterion_val
        else:
            self.criterion_val = criterion

        self.features = NULL
        self.n_features = 0

        self.samples = NULL
        self.n_samples = 0
        self.samples_val = NULL
        self.n_samples_val = 0
        self.feature_values = NULL
        self.feature_values_val = NULL

        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.min_balancedness_tol = min_balancedness_tol
        self.honest = honest
        self.min_eig_leaf = min_eig_leaf
        self.rand_r_state = random_state

    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)
        if self.honest:
            free(self.samples_val)
            free(self.feature_values_val)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init_sample_inds(self, SIZE_t* samples,
                               const SIZE_t[::1] np_samples,
                               DOUBLE_t* sample_weight,
                               SIZE_t* n_samples, double* weighted_n_samples) nogil except -1:
        cdef SIZE_t i, j, ind
        weighted_n_samples[0] = 0.0
        j = 0
        for i in range(np_samples.shape[0]):
            ind = np_samples[i]
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[ind] != 0.0:
                samples[j] = ind
                j += 1

            if sample_weight != NULL:
                weighted_n_samples[0] += sample_weight[ind]
            else:
                weighted_n_samples[0] += 1.0

        # Number of samples is number of positively weighted samples
        n_samples[0] = j

    cdef int init(self, const DTYPE_t[:, :] X, const DOUBLE_t[:, ::1] y,
                  DOUBLE_t* sample_weight,
                  const SIZE_t[::1] np_samples_train,
                  const SIZE_t[::1] np_samples_val) nogil except -1:
        """Initialize the splitter.
        Take in the input data X, T, W, Z, the target Y.
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.
        y : ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples
        """

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t n_samples = np_samples_train.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)
        
        self.init_sample_inds(self.samples, np_samples_train, sample_weight,
                         &self.n_samples, &self.weighted_n_samples)

        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, self.n_samples)
        safe_realloc(&self.constant_features, self.n_features)

        self.X = X
        self.y = y
        self.sample_weight = sample_weight

        self.criterion.init(self.y, self.sample_weight, self.weighted_n_samples,
                            self.samples)
        
        cdef SIZE_t n_samples_val
        cdef SIZE_t* samples_val
        if self.honest:
            n_samples_val = np_samples_val.shape[0]
            samples_val = safe_realloc(&self.samples_val, n_samples_val)
            self.init_sample_inds(self.samples_val, np_samples_val, sample_weight,
                            &self.n_samples_val, &self.weighted_n_samples_val)
            safe_realloc(&self.feature_values_val, self.n_samples_val)
            self.criterion_val.init(self.y, self.sample_weight, self.weighted_n_samples_val,
                                    self.samples_val)
        else:
            self.n_samples_val = self.n_samples
            self.samples_val = self.samples
            self.weighted_n_samples_val = self.weighted_n_samples
            self.feature_values_val = self.feature_values

        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,  double* weighted_n_node_samples,
                        SIZE_t start_val, SIZE_t end_val, double* weighted_n_node_samples_val) nogil except -1:
        """Reset splitter on node samples[start:end].
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        """

        self.start = start
        self.end = end
        self.start_val = start_val
        self.end_val = end_val

        self.criterion.node_reset(start, end)
        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        
        if self.honest:
            self.criterion_val.node_reset(start_val, end_val)
            weighted_n_node_samples_val[0] = self.criterion_val.weighted_n_node_samples
        else:
            weighted_n_node_samples_val[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end].
        This is a placeholder method. The majority of computation will be done
        here.
        It should return -1 upon errors.
        """

        pass

    cdef void node_value_val(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion_val.node_value(dest)
    
    cdef void node_jacobian_val(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion_val.node_jacobian(dest)
    
    cdef void node_precond_val(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion_val.node_precond(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()
    
    cdef double node_impurity_val(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion_val.node_impurity()
    
    cdef double proxy_node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.proxy_node_impurity()
    
    cdef double proxy_node_impurity_val(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion_val.proxy_node_impurity()

    cdef bint is_children_impurity_proxy(self) nogil:
        return (self.criterion.proxy_children_impurity or
                self.criterion_val.proxy_children_impurity)


cdef class BestSplitter(Splitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitter, (self.criterion,
                               self.criterion_val,
                               self.max_features,
                               self.min_samples_leaf,
                               self.min_weight_leaf,
                               self.min_balancedness_tol,
                               self.honest,
                               self.min_eig_leaf,
                               self.random_state), self.__getstate__())

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end]
        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t* samples_val = self.samples_val
        cdef SIZE_t start_val = self.start_val
        cdef SIZE_t end_val = self.end_val

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef DTYPE_t* Xf_val = self.feature_values_val
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef double min_eig_leaf = self.min_eig_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY
        cdef double current_threshold = 0.0
        cdef double weighted_n_node_samples, weighted_n_samples, weighted_n_left, weighted_n_right

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t p_val
        cdef SIZE_t i

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t partition_end

        _init_split(&best, end, end_val)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            
            # TODO. Add some randomness that rejects some splits so that each the threshold on
            # each feature is sufficiently random and so that the number of features that
            # are inspected is also random. This is required by the theory so that every feature
            # will have the possibility to be the one that is split upon.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                features[n_drawn_constants], features[f_j] = features[f_j], features[n_drawn_constants]

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]

                # Sort samples along that feature; by
                # copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                for i in range(start, end):
                    Xf[i] = self.X[samples[i], current.feature]
                
                sort(Xf + start, samples + start, end - start)

                if self.honest:
                    for i in range(start_val, end_val):
                        Xf_val[i] = self.X[samples_val[i], current.feature]
                    
                    sort(Xf_val + start_val, samples_val + start_val, end_val - start_val)
                
                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j], features[n_total_constants] = features[n_total_constants], features[f_j]

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    if self.honest:
                        self.criterion_val.reset()
                    p = start + <int>floor((.5 - self.min_balancedness_tol) * (end - start)) - 1
                    p_val = start_val

                    while p < end and p_val < end_val:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        current_threshold = Xf[p] / 2.0 + Xf[p - 1] / 2.0
                        if ((current_threshold == Xf[p]) or
                            (current_threshold == INFINITY) or
                            (current_threshold == -INFINITY)):
                            current_threshold = Xf[p - 1]

                        # We need to advance p_val such that if we partition samples_val[start_val:end_val]
                        # into samples_val[start_val:best.pos_val] and samples_val[best:pos_val:end_val], then
                        # the first part contains all samples in Xval that are below the threshold. Thus we need
                        # to advance p_val, until Xf_val[p_val] is the first p such that Xf_val[p] > threshold.
                        if self.honest:
                            while (p_val < end_val and
                                Xf_val[p_val] <= current_threshold):
                                p_val += 1
                        else:
                            p_val = p

                        if p < end and p_val < end_val:
                            current.pos = p
                            current.pos_val = p_val

                            # Reject if imbalanced
                            if (end - current.pos) < (.5 - self.min_balancedness_tol) * (end - start):
                                break
                            if (current.pos_val - start_val) < (.5 - self.min_balancedness_tol) * (end_val - start_val):
                                continue
                            if (end_val - current.pos_val) < (.5 - self.min_balancedness_tol) * (end_val - start_val):
                                break

                            # Reject if min_samples_leaf is not guaranteed
                            if (current.pos - start) < min_samples_leaf:
                                continue
                            if (end - current.pos) < min_samples_leaf:
                                break
                            # Reject if min_samples_leaf is not guaranteed on val
                            if (current.pos_val - start_val) < min_samples_leaf:
                                continue
                            if (end_val - current.pos_val) < min_samples_leaf:
                                break

                            self.criterion.update(current.pos)
                            if self.honest:
                                self.criterion_val.update(current.pos_val)

                            # Reject if min_weight_leaf is not satisfied
                            if self.criterion.weighted_n_left < min_weight_leaf:
                                continue
                            if self.criterion.weighted_n_right < min_weight_leaf:
                                break
                            # Reject if minimum eigenvalue proxy requirement is not satisfied
                            if min_eig_leaf >= 0.0:
                                if self.criterion.min_eig_left() < min_eig_leaf:
                                    continue
                                if self.criterion.min_eig_right() < min_eig_leaf:
                                    continue

                            if self.honest:
                                if self.criterion_val.weighted_n_left < min_weight_leaf:
                                    continue
                                if self.criterion_val.weighted_n_right < min_weight_leaf:
                                    break

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                # sum of halves is used to avoid infinite value
                                current.threshold = current_threshold
                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end and best.pos_val < end_val:
            partition_end = end
            p = start

            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1
                else:
                    partition_end -= 1

                    samples[p], samples[partition_end] = samples[partition_end], samples[p]
            
            if self.honest:
                partition_end = end_val
                p = start_val

                while p < partition_end:
                    if self.X[samples_val[p], best.feature] <= best.threshold:
                        p += 1
                    else:
                        partition_end -= 1

                        samples_val[p], samples_val[partition_end] = samples_val[partition_end], samples_val[p]

            self.criterion.reset()
            self.criterion.update(best.pos)
            if self.honest:
                self.criterion_val.reset()
                self.criterion_val.update(best.pos_val)
            best.improvement = self.criterion.impurity_improvement(impurity)
            if not self.is_children_impurity_proxy():
                self.criterion.children_impurity(&best.impurity_left, &best.impurity_right)
                if self.honest:
                    self.criterion_val.children_impurity(&best.impurity_left_val,
                                                         &best.impurity_right_val)
                else:
                    best.impurity_left_val = best.impurity_left
                    best.impurity_right_val = best.impurity_right

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1
