import numpy as np
cimport numpy as np

def rebuild(feature, threshold, split_sample_inds,
            est_sample_inds, estimate, left, right):
    p = Node(split_sample_inds, est_sample_inds)
    p.feature=feature
    p.threshold=threshold
    p.estimate=estimate
    p.left=left
    p.right=right
    return p

cdef class Node:
    """Building block of `CausalTree` class.

    Parameters
    ----------
    sample_inds : array-like, shape (n, )
        Indices defining the sample that the split criterion will be computed on.

    estimate_inds : array-like, shape (n, )
        Indices defining the sample used for calculating balance criteria.

    """

    cdef public int feature
    cdef public double threshold, estimate
    cdef public Node left, right
    cdef readonly np.ndarray split_sample_inds, est_sample_inds

    def __cinit__(self, np.ndarray[np.int_t, ndim=1] sample_inds, np.ndarray[np.int_t, ndim=1] estimate_inds):
        self.feature = -1
        self.threshold = np.inf
        self.split_sample_inds = sample_inds
        self.est_sample_inds = estimate_inds
        self.estimate = 0
        self.left = None
        self.right = None
    
    cdef _find_tree_node(self, np.ndarray[np.float64_t, ndim=1] value):
        if self.feature == -1:
            return self
        elif value[self.feature] < self.threshold:
            return self.left._find_tree_node(value)
        else:
            return self.right._find_tree_node(value)

    cpdef find_tree_node(self, np.ndarray[np.float64_t, ndim=1] value):
        return self._find_tree_node(value)
    
    def __reduce__(self):
        return (rebuild, (self.feature,
                            self.threshold,
                            self.split_sample_inds,
                            self.est_sample_inds,
                            self.estimate,
                            self.left,
                            self.right))



        