# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Utility methods."""

import numpy as np
import scipy.sparse
import sparse as sp
import itertools
from operator import getitem
from collections import defaultdict, Counter
from sklearn.base import TransformerMixin
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, Lasso, MultiTaskLasso
from functools import reduce
from sklearn.utils import check_array, check_X_y
from statsmodels.tools.tools import add_constant
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from collections.abc import Iterable
from sklearn.model_selection._split import _CVIterableWrapper, CV_WARNING
from sklearn.utils.multiclass import type_of_target
import numbers

MAX_RAND_SEED = np.iinfo(np.int32).max


class IdentityFeatures(TransformerMixin):
    """Featurizer that just returns the input data."""

    def fit(self, X):
        """Fit method (does nothing, just returns self)."""
        return self

    def transform(self, X):
        """Perform the identity transform, which returns the input unmodified."""
        return X


def issparse(X):
    """Determine whether an input is sparse.

    For the purposes of this function, both `scipy.sparse` matrices and `sparse.SparseArray`
    types are considered sparse.

    Parameters
    ----------
    X : array-like
        The input to check

    Returns
    -------
    bool
        Whether the input is sparse

    """
    return scipy.sparse.issparse(X) or isinstance(X, sp.SparseArray)


def iscoo(X):
    """Determine whether an input is a `sparse.COO` array.

    Parameters
    ----------
    X : array-like
        The input to check

    Returns
    -------
    bool
        Whether the input is a `COO` array

    """
    return isinstance(X, sp.COO)


def tocoo(X):
    """
    Convert an array to a sparse COO array.

    If the input is already an `sparse.COO` object, this returns the object directly; otherwise it is converted.
    """
    if isinstance(X, sp.COO):
        return X
    elif isinstance(X, sp.DOK):
        return sp.COO(X)
    elif scipy.sparse.issparse(X):
        return sp.COO.from_scipy_sparse(X)
    else:
        return sp.COO.from_numpy(X)


def todense(X):
    """
    Convert an array to a dense numpy array.

    If the input is already a numpy array, this may create a new copy.
    """
    if scipy.sparse.issparse(X):
        return X.toarray()
    elif isinstance(X, sp.SparseArray):
        return X.todense()
    else:
        # TODO: any way to avoid creating a copy if the array was already dense?
        #       the call is necessary if the input was something like a list, though
        return np.array(X)


def size(X):
    """Return the number of elements in the array.

    Parameters
    ----------
    a : array_like
        Input data

    Returns
    -------
    int
        The number of elements of the array
    """
    return X.size if issparse(X) else np.size(X)


def shape(X):
    """Return a tuple of array dimensions."""
    return X.shape if issparse(X) else np.shape(X)


def ndim(X):
    """Return the number of array dimensions."""
    return X.ndim if issparse(X) else np.ndim(X)


def reshape(X, shape):
    """Return a new array that is a reshaped version of an input array.

    The output will be sparse iff the input is.

    Parameters
    ----------
    X : array_like
        The array to reshape

    shape : tuple of ints
        The desired shape of the output array

    Returns
    -------
    ndarray or SparseArray
        The reshaped output array
    """
    if scipy.sparse.issparse(X):
        # scipy sparse arrays don't support reshaping (even for 2D they throw not implemented errors),
        # so convert to pydata sparse first
        X = sp.COO.from_scipy_sparse(X)
        if len(shape) == 2:
            # in the 2D case, we can convert back to scipy sparse; in other cases we can't
            return X.reshape(shape).to_scipy_sparse()
    return X.reshape(shape)


def _apply(op, *XS):
    """
    Apply a function to a sequence of sparse or dense array arguments.

    If any array is sparse then all arrays are converted to COO before the function is applied;
    if all of the arrays are scipy sparse arrays, and if the result is 2D,
    the returned value will be a scipy sparse array as well
    """
    all_scipy_sparse = all(scipy.sparse.issparse(X) for X in XS)
    if any(issparse(X) for X in XS):
        XS = tuple(tocoo(X) for X in XS)
    result = op(*XS)
    if all_scipy_sparse and len(shape(result)) == 2:
        # both inputs were scipy and we can safely convert back to scipy because it's 2D
        return result.to_scipy_sparse()
    return result


def tensordot(X1, X2, axes):
    """
    Compute tensor dot product along specified axes for arrays >= 1-D.

    Parameters
    ----------
    X1, X2 : array_like, len(shape) >= 1
        Tensors to "dot"

    axes : int or (2,) array_like
        integer_like
            If an int N, sum over the last N axes of `X1` and the first N axes
            of `X2` in order. The sizes of the corresponding axes must match
        (2,) array_like
            Or, a list of axes to be summed over, first sequence applying to `X1`,
            second to `X2`. Both elements array_like must be of the same length.
    """
    def td(X1, X2):
        return sp.tensordot(X1, X2, axes) if iscoo(X1) else np.tensordot(X1, X2, axes)
    return _apply(td, X1, X2)


def cross_product(*XS):
    """
    Compute the cross product of features.

    Parameters
    ----------
    X1 : n x d1 matrix
        First matrix of n samples of d1 features
        (or an n-element vector, which will be treated as an n x 1 matrix)
    X2 : n x d2 matrix
        Second matrix of n samples of d2 features
        (or an n-element vector, which will be treated as an n x 1 matrix)
    …

    Returns
    -------
    n x (d1*d2*...) matrix
        Matrix of n samples of d1*d2*... cross product features,
        arranged in form such that each row t of X12 contains:
        [X1[t,0]*X2[t,0]*..., ..., X1[t,d1-1]*X2[t,0]*..., X1[t,0]*X2[t,1]*..., ..., X1[t,d1-1]*X2[t,1]*..., ...]

    """
    for X in XS:
        assert 2 >= ndim(X) >= 1
    n = shape(XS[0])[0]
    for X in XS:
        assert n == shape(X)[0]

    # TODO: wouldn't making X1 vary more slowly than X2 be more intuitive?
    #       (but note that changing this would necessitate changes to callers
    #       to switch the order to preserve behavior where order is important)
    def cross(XS):
        k = len(XS)
        XS = [reshape(XS[i], (n,) + (1,) * (k - i - 1) + (-1,) + (1,) * i) for i in range(k)]
        return reshape(reduce(np.multiply, XS), (n, -1))
    return _apply(cross, XS)


def stack(XS, axis=0):
    """
    Join a sequence of arrays along a new axis.

    The axis parameter specifies the index of the new axis in the dimensions of the result.
    For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.

    Parameters
    ----------
    arrays : sequence of array_like
        Each array must have the same shape

    axis : int, optional
        The axis in the result array along which the input arrays are stacked

    Returns
    -------
    ndarray or SparseArray
        The stacked array, which has one more dimension than the input arrays.
        It will be sparse if the inputs are.
    """
    def st(*XS):
        return sp.stack(XS, axis=axis) if iscoo(XS[0]) else np.stack(XS, axis=axis)
    return _apply(st, *XS)


def concatenate(XS, axis=0):
    """
    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    X1, X2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).

    axis : int, optional
        The axis along which the arrays will be joined.  Default is 0.

    Returns
    -------
    ndarray or SparseArray
        The concatenated array. It will be sparse if the inputs are.
    """
    def conc(*XS):
        return sp.concatenate(XS, axis=axis) if iscoo(XS[0]) else np.concatenate(XS, axis=axis)
    return _apply(conc, *XS)


# note: in contrast to np.hstack this only works with arrays of dimension at least 2
def hstack(XS):
    """
    Stack arrays in sequence horizontally (column wise).

    This is equivalent to concatenation along the second axis


    Parameters
    ----------
    XS : sequence of ndarrays
        The arrays must have the same shape along all but the second axis.

    Returns
    -------
    ndarray or SparseArray
        The array formed by stacking the given arrays. It will be sparse if the inputs are.
    """
    # Confusingly, this needs to concatenate, not stack (stack returns an array with an extra dimension)
    return concatenate(XS, 1)


def vstack(XS):
    """
    Stack arrays in sequence vertically (row wise).

    This is equivalent to concatenation along the first axis after
    1-D arrays of shape (N,) have been reshaped to (1,N).

    Parameters
    ----------
    XS : sequence of ndarrays
        The arrays must have the same shape along all but the first axis.
        1-D arrays must have the same length.

    Returns
    -------
    ndarray or SparseArray
        The array formed by stacking the given arrays, will be at least 2-D.   It will be sparse if the inputs are.
    """
    # Confusingly, this needs to concatenate, not stack (stack returns an array with an extra dimension)
    return concatenate(XS, 0)


def transpose(X, axes=None):
    """
    Permute the dimensions of an array.

    Parameters
    ----------
    X : array_like
        Input array.
    axes :  list of ints, optional
            By default, reverse the dimensions, otherwise permute the axes according to the values given

    Returns
    -------
    p : ndarray or SparseArray
        `X` with its axes permuted. This will be sparse if `X` is.

    """
    def t(X):
        if iscoo(X):
            return X.transpose(axes)
        else:
            return np.transpose(X, axes)
    return _apply(t, X)


def reshape_Y_T(Y, T):
    """
    Reshapes Y and T when Y.ndim = 2 and/or T.ndim = 1.

    Parameters
    ----------
    Y : array_like, shape (n, ) or (n, 1)
        Outcome for the treatment policy. Must be a vector or single-column matrix.

    T : array_like, shape (n, ) or (n, d_t)
        Treatment policy.

    Returns
    -------
    Y : array_like, shape (n, )
        Flattened outcome for the treatment policy.

    T : array_like, shape (n, 1) or (n, d_t)
        Reshaped treatment policy.

    """
    assert(len(Y) == len(T))
    assert(Y.ndim <= 2)
    if Y.ndim == 2:
        assert(Y.shape[1] == 1)
        Y = Y.flatten()
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    return Y, T


def check_inputs(Y, T, X, W=None, multi_output_T=True, multi_output_Y=True):
    """
    Input validation for CATE estimators.

    Checks Y, T, X, W for consistent length, enforces X, W 2d.
    Standard input checks are only applied to all inputs,
    such as checking that an input does not have np.nan or np.inf targets.
    Converts regular Python lists to numpy arrays.

    Parameters
    ----------
    Y : array_like, shape (n, ) or (n, d_y)
        Outcome for the treatment policy.

    T : array_like, shape (n, ) or (n, d_t)
        Treatment policy.

    X : array-like, shape (n, d_x)
        Feature vector that captures heterogeneity.

    W : array-like, shape (n, d_w) or None (default=None)
        High-dimensional controls.

    multi_output_T : bool
        Whether to allow more than one treatment.

    multi_output_Y: bool
        Whether to allow more than one outcome.

    Returns
    -------
    Y : array_like, shape (n, ) or (n, d_y)
        Converted and validated Y.

    T : array_like, shape (n, ) or (n, d_t)
        Converted and validated T.

    X : array-like, shape (n, d_x)
        Converted and validated X.

    W : array-like, shape (n, d_w) or None (default=None)
        Converted and validated W.

    """
    X, T = check_X_y(X, T, multi_output=multi_output_T, y_numeric=True)
    _, Y = check_X_y(X, Y, multi_output=multi_output_Y, y_numeric=True)
    if W is not None:
        W, _ = check_X_y(W, Y)
    return Y, T, X, W


def broadcast_unit_treatments(X, d_t):
    """
    Generate `d_t` unit treatments for each row of `X`.

    Parameters
    ----------
    d_t: int
        Number of treatments
    X : array
        Features

    Returns
    -------
    X, T : (array, array)
        The updated `X` array (with each row repeated `d_t` times),
        and the generated `T` array
    """
    d_x = shape(X)[0]
    eye = np.eye(d_t)
    # tile T and repeat X along axis 0 (so that the duplicated rows of X remain consecutive)
    T = np.tile(eye, (d_x, 1))
    Xs = np.repeat(X, d_t, axis=0)
    return Xs, T


def reshape_treatmentwise_effects(A, d_t, d_y):
    """
    Given an effects matrix ordered first by treatment, transform it to be ordered by outcome.

    Parameters
    ----------
    A : array
        The array of effects, of size n*d_y*d_t
    d_t : tuple of int
        Either () if T was a vector, or a 1-tuple of the number of columns of T if it was an array
    d_y : tuple of int
        Either () if Y was a vector, or a 1-tuple of the number of columns of Y if it was an array

    Returns
    -------
    A : array (shape (m, d_y, d_t))
        The transformed array.  Note that singleton dimensions will be dropped for any inputs which
        were vectors, as in the specification of `BaseCateEstimator.marginal_effect`.
    """
    A = reshape(A, (-1,) + d_t + d_y)
    if d_t and d_y:
        return transpose(A, (0, 2, 1))  # need to return as m by d_y by d_t matrix
    else:
        return A


def check_treatments(T0, T1, n, dt):
    """
    Extend scalar treatments to correct dimensionality (returning non-scalar treatments unmodified)

    Parameters
    ----------
    T0 : scalar or array
    T1 : scalar or array
    n : number of rows to output
    dt : tuple
        Trailing dimensions of T (that is, ``()`` for a vector treatment, or ``(d_t,)`` for an array treatment)

    Returns
    -------
    T0 : array of shape ``(n,) + dt``
    T1 : array of shape ``(n,) + dt``
    """
    if (ndim(T0) == 0 or ndim(T1) == 0) and dt and shape(dt)[1] > 1:
        warnings.warn("A scalar was specified but there are multiple treatments; "
                      "the same value will be used for each treatment.  Consider specifying"
                      "all treatments, or using the const_marginal_effect method.")
    if ndim(T0) == 0:
        T0 = np.repeat(T0, (n,) + dt)
    if ndim(T1) == 0:
        T1 = np.repeat(T1, (n,) + dt)
    return T0, T1


def einsum_sparse(subscripts, *arrs):
    """
    Evaluate the Einstein summation convention on the operands.

    Using the Einstein summation convention, many common multi-dimensional array operations can be represented
    in a simple fashion. This function provides a way to compute such summations.

    Parameters
    ----------
    subscripts : str
        Specifies the subscripts for summation.
        Unlike `np.eisnum` elipses are not supported and the output must be explicitly included

    arrs : list of COO arrays
        These are the sparse arrays for the operation.

    Returns
    -------
    SparseArray
        The sparse array calculated based on the Einstein summation convention.
    """
    inputs, outputs = subscripts.split('->')
    inputs = inputs.split(',')
    outputInds = set(outputs)
    allInds = set.union(*[set(i) for i in inputs])

    # same number of input definitions as arrays
    assert len(inputs) == len(arrs)

    # input definitions have same number of dimensions as each array
    assert all(arr.ndim == len(input) for (arr, input) in zip(arrs, inputs))

    # all result indices are unique
    assert len(outputInds) == len(outputs)

    # all result indices must match at least one input index
    assert outputInds <= allInds

    # map indices to all array, axis pairs for that index
    indMap = {c: [(n, i) for n in range(len(inputs)) for (i, x) in enumerate(inputs[n]) if x == c] for c in allInds}

    for c in indMap:
        # each index has the same cardinality wherever it appears
        assert len({arrs[n].shape[i] for (n, i) in indMap[c]}) == 1

    # State: list of (set of letters, list of (corresponding indices, value))
    # Algo: while list contains more than one entry
    #         take two entries
    #         sort both lists by intersection of their indices
    #         merge compatible entries (where intersection of indices is equal - in the resulting list,
    #         take the union of indices and the product of values), stepping through each list linearly

    # TODO: might be faster to break into connected components first
    #       e.g. for "ab,d,bc->ad", the two components "ab,bc" and "d" are independent,
    #       so compute their content separately, then take cartesian product
    #       this would save a few pointless sorts by empty tuples

    # TODO: Consider investigating other performance ideas for these cases
    #       where the dense method beat the sparse method (usually sparse is faster)
    #       e,facd,c->cfed
    #        sparse: 0.0335489
    #        dense:  0.011465999999999997
    #       gbd,da,egb->da
    #        sparse: 0.0791625
    #        dense:  0.007319099999999995
    #       dcc,d,faedb,c->abe
    #        sparse: 1.2868097
    #        dense:  0.44605229999999985

    def merge(x1, x2):
        (s1, l1), (s2, l2) = x1, x2
        keys = {c for c in s1 if c in s2}  # intersection of strings
        outS = ''.join(set(s1 + s2))  # union of strings
        outMap = [(True, s1.index(c)) if c in s1 else (False, s2.index(c)) for c in outS]

        def keyGetter(s):
            inds = [s.index(c) for c in keys]
            return lambda p: tuple(p[0][ind] for ind in inds)
        kg1 = keyGetter(s1)
        kg2 = keyGetter(s2)
        l1.sort(key=kg1)
        l2.sort(key=kg2)
        i1 = i2 = 0
        outL = []
        while i1 < len(l1) and i2 < len(l2):
            k1, k2 = kg1(l1[i1]), kg2(l2[i2])
            if k1 < k2:
                i1 += 1
            elif k2 < k1:
                i2 += 1
            else:
                j1, j2 = i1, i2
                while j1 < len(l1) and kg1(l1[j1]) == k1:
                    j1 += 1
                while j2 < len(l2) and kg2(l2[j2]) == k2:
                    j2 += 1
                for c1, d1 in l1[i1:j1]:
                    for c2, d2 in l2[i2:j2]:
                        outL.append((tuple(c1[charIdx] if inFirst else c2[charIdx] for inFirst, charIdx in outMap),
                                     d1 * d2))
                i1 = j1
                i2 = j2
        return outS, outL

    # when indices are repeated within an array, pre-filter the coordinates and data
    def filter_inds(coords, data, n):
        counts = Counter(inputs[n])
        repeated = [(c, counts[c]) for c in counts if counts[c] > 1]
        if len(repeated) > 0:
            mask = np.full(len(data), True)
            for (k, v) in repeated:
                inds = [i for i in range(len(inputs[n])) if inputs[n][i] == k]
                for i in range(1, v):
                    mask &= (coords[:, inds[0]] == coords[:, inds[i]])
            if not all(mask):
                return coords[mask, :], data[mask]
        return coords, data

    xs = [(s, list(zip(c, d)))
          for n, (s, arr) in enumerate(zip(inputs, arrs))
          for c, d in [filter_inds(arr.coords.T, arr.data, n)]]

    # TODO: would using einsum's paths to optimize the order of merging help?
    while len(xs) > 1:
        xs.append(merge(xs.pop(), xs.pop()))

    results = defaultdict(int)

    for (s, l) in xs:
        coordMap = [s.index(c) for c in outputs]
        for (c, d) in l:
            results[tuple(c[i] for i in coordMap)] += d

    return sp.COO(np.array([k for k in results.keys()]).T,
                  np.array([v for v in results.values()]),
                  [arrs[indMap[c][0][0]].shape[indMap[c][0][1]] for c in outputs])


class WeightedModelWrapper(object):
    """Helper class for assiging weights to models without this option.

    Parameters
    ----------
    model_instance : estimator
        Model that requires weights.

    sample_type : string, optional (default=`weighted`)
        Method for adding weights to the model. `weighted` for linear regression models
        where the weights can be incorporated in the matrix multiplication,
        `sampled` for other models. `sampled` samples the training set according
        to the normalized weights and creates a dataset larger than the original.

    """

    def __init__(self, model_instance, sample_type="weighted"):
        self.model_instance = model_instance
        if sample_type == "weighted":
            self.data_transform = self._weighted_inputs
        else:
            warnings.warn("The model provided does not support sample weights. " +
                          "Manual weighted sampling may icrease the variance in the results.", UserWarning)
            self.data_transform = self._sampled_inputs

    def fit(self, X, y, sample_weight=None):
        """Fit underlying model instance with weighted inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, n_outcomes)
            Target values.

        Returns
        -------
        self: an instance of the underlying estimator.
        """
        if sample_weight is not None:
            X, y = self.data_transform(X, y, sample_weight)
        return self.model_instance.fit(X, y)

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples, n_outcomes)
            Returns predicted values.
        """
        return self.model_instance.predict(X)

    def _weighted_inputs(self, X, y, sample_weight):
        normalized_weights = sample_weight * X.shape[0] / np.sum(sample_weight)
        sqrt_weights = np.sqrt(normalized_weights)
        weight_mat = np.diag(sqrt_weights)
        return np.matmul(weight_mat, X), np.matmul(weight_mat, y)

    def _sampled_inputs(self, X, y, sample_weight):
        # Normalize weights
        normalized_weights = sample_weight / np.sum(sample_weight)
        data_length = int(min(1 / np.min(normalized_weights[normalized_weights > 0]), 10) * X.shape[0])
        data_indices = np.random.choice(X.shape[0], size=data_length, p=normalized_weights)
        return X[data_indices], y[data_indices]


def _fit_weighted_linear_model(self, class_name, X, y, sample_weight, check_input=None):
    # Convert X, y into numpy arrays
    X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
    # Define fit parameters
    fit_params = {'X': X, 'y': y}
    # Some algorithms doen't have a check_input option
    if check_input is not None:
        fit_params['check_input'] = check_input

    if sample_weight is not None:
        # Check weights array
        if np.atleast_1d(sample_weight).ndim > 1:
            # Check that weights are size-compatible
            raise ValueError("Sample weights must be 1D array or scalar")
        if np.ndim(sample_weight) == 0:
            sample_weight = np.repeat(sample_weight, X.shape[0])
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False, allow_nd=False)
            if sample_weight.shape[0] != X.shape[0]:
                raise ValueError(
                    "Found array with {0} sample(s) while {1} samples were expected.".format(
                        sample_weight.shape[0], X.shape[0])
                )

        # Normalize inputs
        X, y, X_offset, y_offset, X_scale = self._preprocess_data(
            X, y, fit_intercept=self.fit_intercept, normalize=False,
            copy=self.copy_X, check_input=check_input if check_input is not None else True,
            sample_weight=sample_weight, return_mean=True)
        # Weight inputs
        normalized_weights = X.shape[0] * sample_weight / np.sum(sample_weight)
        sqrt_weights = np.sqrt(normalized_weights)
        weight_mat = np.diag(sqrt_weights)
        X_weighted = np.matmul(weight_mat, X)
        y_weighted = np.matmul(weight_mat, y)
        fit_params['X'] = X_weighted
        fit_params['y'] = y_weighted
        if self.fit_intercept:
            # Fit base class without intercept
            self.fit_intercept = False
            # Fit Lasso
            super(class_name, self).fit(**fit_params)
            # Reset intercept
            self.fit_intercept = True
            # The intercept is not calculated properly due the sqrt(weights) factor
            # so it must be recomputed
            self._set_intercept(X_offset, y_offset, X_scale)
        else:
            super(class_name, self).fit(**fit_params)
    else:
        # Fit lasso without weights
        super(class_name, self).fit(**fit_params)


def _split_weighted_sample(self, X, y, sample_weight, is_stratified=False):
    if is_stratified:
        kfold_model = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle,
                                      random_state=self.random_state)
    else:
        kfold_model = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                            random_state=self.random_state)
    if sample_weight is None:
        return kfold_model.split(X, y)
    weights_sum = np.sum(sample_weight)
    max_deviations = []
    all_splits = []
    for i in range(self.n_trials + 1):
        splits = [test for (train, test) in list(kfold_model.split(X, y))]
        weight_fracs = np.array([np.sum(sample_weight[split]) / weights_sum for split in splits])
        if np.all(weight_fracs > .95 / self.n_splits):
            # Found a good split, return.
            return self._get_folds_from_splits(splits, X.shape[0])
        # Record all splits in case the stratification by weight yeilds a worse partition
        all_splits.append(splits)
        max_deviation = np.abs(weight_fracs - 1 / self.n_splits)
        max_deviations.append(max_deviation)
        # Reseed random generator and try again
        kfold_model.shuffle = True
        kfold_model.random_state = None

    # If KFold fails after n_trials, we try the next best thing: stratifying by weight groups
    warnings.warn("The KFold algorithm failed to find a weight-balanced partition after {n_trials} trials." +
                  " Falling back on a weight stratification algorithm.".format(n_trials=self.n_trials), UserWarning)
    if is_stratified:
        stratified_weight_splits = [[]] * self.n_splits
        for y_unique in np.unique(y.flatten()):
            class_inds = np.argwhere(y == y_unique).flatten()
            class_splits = self._get_splits_from_weight_stratification(sample_weight[class_inds])
            stratified_weight_splits = [split + list(class_inds[class_split]) for split, class_split in zip(
                stratified_weight_splits, class_splits)]
    else:
        stratified_weight_splits = self._get_splits_from_weight_stratification(sample_weight)
    weight_fracs = np.array([np.sum(sample_weight[split]) / weights_sum for split in stratified_weight_splits])
    if np.all(weight_fracs > .95 / self.n_splits):
        # Found a good split, return.
        return self._get_folds_from_splits(stratified_weight_splits, X.shape[0])
    else:
        # Did not find a good split
        # Record the devaiation for the weight-stratified split to compare with KFold splits
        all_splits.append(stratified_weight_splits)
        max_deviation = np.abs(weight_fracs - 1 / self.n_splits)
        max_deviations.append(max_deviation)
    # Return most weight-balanced partition
    min_deviation_index = np.argmin(max_deviations)
    return self._get_folds_from_splits(all_splits[min_deviation_index], X.shape[0])


def _weighted_check_cv(cv='warn', y=None, classifier=False):
    if cv is None or cv == 'warn':
        warnings.warn(CV_WARNING, FutureWarning)
        cv = 3

    if isinstance(cv, numbers.Integral):
        if (classifier and (y is not None) and
                (type_of_target(y) in ('binary', 'multiclass'))):
            return WeightedStratifiedKFold(cv)
        else:
            return WeightedKFold(cv)

    if not hasattr(cv, 'split') or isinstance(cv, str):
        if not isinstance(cv, Iterable) or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             "or an iterable. Got %s." % cv)
        return _WeightedCVIterableWrapper(cv)

    return cv  # New style cv objects are passed without any modification


class _WeightedCVIterableWrapper(_CVIterableWrapper):
    def __init__(self, cv):
        super().__init__(cv)

    def get_n_splits(self, X=None, y=None, groups=None, sample_weight=None):
        return super().get_n_splits(self, X, y, groups)

    def split(self, X=None, y=None, groups=None, sample_weight=None):
        return super().split(X, y, groups)


class WeightedLasso(Lasso):
    """Version of sklearn Lasso that accepts weights.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    fit_intercept : boolean, optional, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    precompute : True | False | array-like, default=False
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    sparse_coef_ : scipy.sparse matrix, shape (n_features, 1) | (n_targets, n_features)
        ``sparse_coef_`` is a readonly property derived from ``coef_``

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : int | array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """

    def __init__(self, alpha=1.0, fit_intercept=True,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super(WeightedLasso, self).__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            normalize=False, precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        """
        _fit_weighted_linear_model(self, WeightedLasso, X, y, sample_weight, check_input)
        return self


class WeightedMultiTaskLasso(MultiTaskLasso):
    """Version of sklearn MultiTaskLasso that accepts weights.

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the ``Lasso`` object is not advised.
        Given this, you should use the :class:`LinearRegression` object.

    fit_intercept : boolean, optional, default True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        See :term:`the Glossary <warm_start>`.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.

    Attributes
    ----------
    coef_ : array, shape (n_features,) | (n_targets, n_features)
        parameter vector (w in the cost function formula)

    intercept_ : float | array, shape (n_targets,)
        independent term in decision function.

    n_iter_ : int | array-like, shape (n_targets,)
        number of iterations run by the coordinate descent solver to reach
        the specified tolerance.
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 copy_X=True, max_iter=1000, tol=1e-4, warm_start=False,
                 random_state=None, selection='cyclic'):
        super(WeightedMultiTaskLasso, self).__init__(
            alpha=alpha, fit_intercept=fit_intercept, normalize=False,
            copy_X=copy_X, max_iter=max_iter, tol=tol, warm_start=warm_start,
            random_state=random_state, selection=selection)

    def fit(self, X, y, sample_weight=None):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.
        """
        _fit_weighted_linear_model(self, WeightedMultiTaskLasso, X, y, sample_weight)
        return self


class WeightedKFold(object):
    """K-Folds cross-validator for weighted data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k folds of roughly equal size and equal total weight.

    The default is to try sklearn.model_selection.KFold a number of trials to find
    a weight-balanced k-way split. If it cannot find such a split, it will fall back
    onto a more rigorous weight stratification algorithm.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    n_trials : int, default=10
        Number of times to try sklearn.model_selection.KFold before falling back to another
        weight stratification algorithm.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    """

    def __init__(self, n_splits=3, n_trials=10, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.n_trials = n_trials
        self.random_state = random_state
        return

    def split(self, X, y, sample_weight=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array-like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=False)

    def _get_folds_from_splits(self, splits, sample_size):
        folds = []
        sample_indices = np.arange(sample_size)
        for it in range(self.n_splits):
            folds.append([np.setdiff1d(sample_indices, splits[it], assume_unique=True), splits[it]])
        return folds

    def _get_splits_from_weight_stratification(self, sample_weight):
        # Weight stratification algorithm
        # Sort weights for weight strata search
        sorted_inds = np.argsort(sample_weight)
        sorted_weights = sample_weight[sorted_inds]
        max_split_size = sorted_weights.shape[0] // self.n_splits
        max_divisible_length = max_split_size * self.n_splits
        sorted_inds_subset = np.reshape(sorted_inds[:max_divisible_length], (max_split_size, self.n_splits))
        shuffled_sorted_inds_subset = np.apply_along_axis(np.random.permutation, axis=1, arr=sorted_inds_subset)
        splits = [list(shuffled_sorted_inds_subset[:, i]) for i in range(self.n_splits)]
        if max_divisible_length != sorted_weights.shape[0]:
            # There are some leftover indices that have yet to be assigned
            subsample = sorted_inds[max_divisible_length:]
            if self.shuffle:
                np.random.shuffle(subsample)
            new_splits = np.array_split(subsample, self.n_splits)
            np.random.shuffle(new_splits)
            # Append stratum splits to overall splits
            splits = [split + list(new_split) for split, new_split in zip(splits, new_splits)]
        return splits


class WeightedStratifiedKFold(WeightedKFold):
    """Stratified K-Folds cross-validator for weighted data.

    Provides train/test indices to split data in train/test sets.
    Split dataset into k folds of roughly equal size and equal total weight.

    The default is to try sklearn.model_selection.StratifiedKFold a number of trials to find
    a weight-balanced k-way split. If it cannot find such a split, it will fall back
    onto a more rigorous weight stratification algorithm.

    Parameters
    ----------
    n_splits : int, default=3
        Number of folds. Must be at least 2.

    n_trials : int, default=10
        Number of times to try sklearn.model_selection.StratifiedKFold before falling back to another
        weight stratification algorithm.

    shuffle : boolean, optional
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`. Used when ``shuffle`` == True.
    """

    def split(self, X, y, sample_weight=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        sample_weight : array-like, shape (n_samples,)
            Weights associated with the training data.
        """
        return _split_weighted_sample(self, X, y, sample_weight, is_stratified=True)


class WeightedLassoCV(LassoCV):
    """Version of sklearn LassoCV that accepts weights.

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : numpy array, optional
        List of alphas where to compute the models.
        If ``None`` alphas are set automatically

    fit_intercept : boolean, default True
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument.

    max_iter : int, optional
        The maximum number of iterations

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold weighted cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, :class:`WeightedKFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive : bool, optional
        If positive, restrict regression coefficients to be positive

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'.

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 precompute='auto', max_iter=1000, tol=1e-4, normalize=False,
                 copy_X=True, cv='warn', verbose=False, n_jobs=None,
                 positive=False, random_state=None, selection='cyclic'):

        super().__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas,
            fit_intercept=fit_intercept, normalize=False,
            precompute=precompute, max_iter=max_iter, tol=tol, copy_X=copy_X,
            cv=cv, verbose=verbose, n_jobs=n_jobs, positive=positive,
            random_state=random_state, selection=selection)

    def fit(self, X, y, sample_weight=None):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.
        """
        # Make weighted splitter
        cv_temp = self.cv
        self.cv = _weighted_check_cv(self.cv).split(X, y, sample_weight=sample_weight)
        # Fit weighted model
        _fit_weighted_linear_model(self, WeightedLassoCV, X, y, sample_weight)
        self.cv = cv_temp
        return self


class WeightedMultiTaskLassoCV(MultiTaskLassoCV):
    """Version of sklearn MultiTaskLassoCV that accepts weights.

    Parameters
    ----------
    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``.

    n_alphas : int, optional
        Number of alphas along the regularization path

    alphas : array-like, optional
        List of alphas where to compute the models.
        If not provided, set automatically.

    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    max_iter : int, optional
        The maximum number of iterations.

    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold weighted cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, :class:`WeightedKFold` is used.

    verbose : bool or integer
        Amount of verbosity.

    n_jobs : int or None, optional (default=None)
        Number of CPUs to use during the cross validation. Note that this is
        used only if multiple values for l1_ratio are given.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, optional, default None
        The seed of the pseudo random number generator that selects a random
        feature to update.  If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`. Used when ``selection`` ==
        'random'

    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
    """

    def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                 normalize=False, max_iter=1000, tol=1e-4,
                 copy_X=True, cv='warn', verbose=False, n_jobs=None,
                 random_state=None, selection='cyclic'):

        super().__init__(
            eps=eps, n_alphas=n_alphas, alphas=alphas,
            fit_intercept=fit_intercept, normalize=False,
            max_iter=max_iter, tol=tol, copy_X=copy_X,
            cv=cv, verbose=verbose, n_jobs=n_jobs,
            random_state=random_state, selection=selection)

    def fit(self, X, y, sample_weight=None):
        """Fit model with coordinate descent.

        Parameters
        ----------
        X : ndarray or scipy.sparse matrix, (n_samples, n_features)
            Data

        y : ndarray, shape (n_samples,) or (n_samples, n_targets)
            Target. Will be cast to X's dtype if necessary

        sample_weight : numpy array of shape [n_samples]
                        Individual weights for each sample.
                        The weights will be normalized internally.
        """
        # Make weighted splitter
        cv_temp = self.cv
        self.cv = _weighted_check_cv(self.cv).split(X, y, sample_weight=sample_weight)
        # Fit weighted model
        _fit_weighted_linear_model(self, WeightedMultiTaskLassoCV, X, y, sample_weight)
        self.cv = cv_temp
        return self


class MultiModelWrapper(object):
    """Helper class for assiging weights to models without this option.

    Parameters
    ----------
    model_list : array-like, shape (n_T, )
        List of models to be trained separately for each treatment group.
    """

    def __init__(self, model_list=[]):
        self.model_list = model_list
        self.n_T = len(model_list)

    def fit(self, Xt, y, sample_weight=None):
        """Fit underlying list of models with weighted inputs.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + n_treatments)
            Training data. The last n_T columns should be a one-hot encoding of the treatment assignment.

        y : array-like, shape (n_samples, )
            Target values.

        Returns
        -------
        self: an instance of the class
        """
        X = Xt[:, :-self.n_T]
        t = Xt[:, -self.n_T:]
        if sample_weight is None:
            for i in range(self.n_T):
                mask = (t[:, i] == 1)
                self.model_list[i].fit(X[mask], y[mask])
        else:
            for i in range(self.n_T):
                mask = (t[:, i] == 1)
                self.model_list[i].fit(X[mask], y[mask], sample_weight[mask])
        return self

    def predict(self, Xt):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features + n_treatments)
            Samples. The last n_T columns should be a one-hot encoding of the treatment assignment.

        Returns
        -------
        C : array, shape (n_samples, )
            Returns predicted values.
        """
        X = Xt[:, :-self.n_T]
        t = Xt[:, -self.n_T:]
        predictions = [self.model_list[np.nonzero(t[i])[0][0]].predict(X[[i]]) for i in range(len(X))]
        return np.concatenate(predictions)


def _safe_norm_ppf(q, loc=0, scale=1):
    prelim = loc.copy()
    if hasattr(prelim, "__len__"):
        if np.any(scale > 0):
            prelim[scale > 0] = scipy.stats.norm.ppf(q, loc=loc[scale > 0], scale=scale[scale > 0])
    elif scale > 0:
        prelim = scipy.stats.norm.ppf(q, loc=loc, scale=scale)
    return prelim


class StatsModelsLinearRegression:
    """
    Class which mimics weighted linear regression from the statsmodels package.

    However, unlike statsmodels WLS, this class also supports sample variances in addition to sample weights,
    which enables more accurate inference when working with summarized data.

    Parameters
    ----------
    fit_intercept : bool (optional, default=True)
        Whether to fit an intercept in this model
    fit_args : dict (optional, default=`{}`)
        The statsmodels-style fit arguments; keys can include 'cov_type', 'cov_kwds', and 'use_t'.
    """

    def __init__(self, fit_intercept=True, fit_args={}):
        self.fit_args = fit_args
        self._fit_intercept = fit_intercept
        return

    def _check_input(self, X, y, sample_weight, sample_var):
        """Check dimensions and other assertions."""
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        elif np.any(np.not_equal(np.mod(sample_weight, 1), 0)):
            raise AttributeError("Sample weights must all be integers for inference to be valid!")

        if sample_var is None:
            if np.any(np.not_equal(sample_weight, 1)):
                warnings.warn(
                    """No variance information was given for samples with sample_weight not equal to 1,
                       that represent summaries of multiple original samples. Inference will be invalid!""")
            sample_var = np.zeros(y.shape)

        if sample_var.ndim < 2:
            if np.any(np.equal(sample_weight, 1) & np.not_equal(sample_var, 0)):
                warnings.warn(
                    """Variance was set to non-zero for an observation with sample_weight=1!
                       sample_var represents the variance of the original observations that are
                       summarized in this sample. Hence, cannot have a non-zero variance if only
                       one observations was summarized. Inference will be invalid!""")
        else:
            if np.any(np.equal(sample_weight, 1) & np.not_equal(np.sum(sample_var, axis=1), 0)):
                warnings.warn(
                    """Variance was set to non-zero for an observation with sample_weight=1!
                       sample_var represents the variance of the original observations that are
                       summarized in this sample. Hence, cannot have a non-zero variance if only
                       one observations was summarized. Inference will be invalid!""")

        assert (X.shape[0] == y.shape[0] ==
                sample_weight.shape[0] == sample_var.shape[0]), "Input lengths not compatible!"
        if y.ndim >= 2:
            assert (y.ndim == sample_var.ndim and
                    y.shape[1] == sample_var.shape[1]), "Input shapes not compatible: {}, {}!".format(
                y.shape, sample_var.shape)

        return X, y, sample_weight, sample_var

    def fit(self, X, y, sample_weight=None, sample_var=None):
        """Fits the model.

        Parameters
        ----------
        X : (N, d) nd array like
            co-variates
        y : {(N,), (N, p)} nd array like
            output variable(s)
        sample_weight : (N,) nd array like of integers
            Weight for the observation. Observation i is treated as the mean
            outcome of sample_weight[i] independent observations
        sample_var : {(N,), (N, p)} nd array like
            Variance of the outcome(s) of the original sample_weight[i] observations
            that were used to compute the mean outcome represented by observation i.

        Returns
        -------
        self : StatsModelsLinearRegression
        """

        # TODO: Add other types of covariance estimation (e.g. Newey-West (HAC), HC2, HC3)
        X, y, sample_weight, sample_var = self._check_input(X, y, sample_weight, sample_var)

        if self._fit_intercept:
            X = add_constant(X, has_constant='add')
        WX = X * np.sqrt(sample_weight).reshape(-1, 1)

        if y.ndim < 2:
            self._n_out = 0
            wy = y * np.sqrt(sample_weight)
        else:
            self._n_out = y.shape[1]
            wy = y * np.sqrt(sample_weight).reshape(-1, 1)

        param, _, rank, _ = np.linalg.lstsq(WX, wy, rcond=None)

        if rank < param.shape[0]:
            warnings.warn("Co-variance matrix is undertermined. Inference will be invalid!")

        sigma_inv = np.linalg.pinv(np.matmul(WX.T, WX))
        self._param = param
        var_i = sample_var + (y - np.matmul(X, param))**2
        n_obs = np.sum(sample_weight)
        df = len(param) if self._n_out == 0 else param.shape[0]

        if n_obs <= df:
            warnings.warn("Number of observations <= than number of parameters. Using biased variance calculation!")
            correction = 1
        else:
            correction = (n_obs / (n_obs - df))

        if ('cov_type' not in self.fit_args) or (self.fit_args['cov_type'] == 'nonrobust'):
            if y.ndim < 2:
                self._var = correction * np.average(var_i, weights=sample_weight) * sigma_inv
            else:
                vars = correction * np.average(var_i, weights=sample_weight, axis=0)
                self._var = [v * sigma_inv for v in vars]
        elif (self.fit_args['cov_type'] == 'HC0'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(WX.T, WX * var_i.reshape(-1, 1))
                self._var = np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(WX.T, WX * var_i[:, [j]])
                    self._var.append(np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv)))
        elif (self.fit_args['cov_type'] == 'HC1'):
            if y.ndim < 2:
                weighted_sigma = np.matmul(WX.T, WX * var_i.reshape(-1, 1))
                self._var = correction * np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv))
            else:
                self._var = []
                for j in range(self._n_out):
                    weighted_sigma = np.matmul(WX.T, WX * var_i[:, [j]])
                    self._var.append(correction * np.matmul(sigma_inv, np.matmul(weighted_sigma, sigma_inv)))
        return self

    def predict(self, X):
        """
        Predicts the output given an array of instances.

        Parameters
        ----------
        X : (n, d) array like
            The covariates on which to predict

        Returns
        -------
        predictions : {(n,) array, (n,p) array}
            The predicted mean outcomes
        """
        if self._fit_intercept:
            X = add_constant(X, has_constant='add')
        return np.matmul(X, self._param)

    @property
    def coef_(self):
        """Get the model's coefficients on the covariates.

        Returns
        -------
        coef_ : {(d,), (p, d)} nd array like
            The coefficients of the variables in the linear regression. If label y
            was p-dimensional, then the result is a matrix of coefficents, whose p-th
            row containts the coefficients corresponding to the p-th coordinate of the label.
        """
        if self._fit_intercept:
            if self._n_out == 0:
                return self._param[1:]
            else:
                return self._param[1:].T
        else:
            if self._n_out == 0:
                return self._param
            else:
                return self._param.T

    @property
    def intercept_(self):
        """Get the intercept(s) (or 0 if no intercept was fit).

        Returns
        -------
        intercept_ : float or (p,) nd array like
            The intercept of the linear regresion. If label y was p-dimensional, then the result is a vector
            whose p-th entry containts the intercept corresponding to the p-th coordinate of the label.
        """
        return self._param[0] if self._fit_intercept else (0 if self._n_out == 0 else np.zeros(self._n_out))

    @property
    def _param_var(self):
        """
        Returns
        -------
        var : {(d (+1), d (+1)), (p, d (+1), d (+1))} nd array like
            The covariance matrix of all the parameters in the regression (including the intercept
            as the first parameter).  If intercept was set to False then this is the covariance matrix
            of the coefficients; otherwise, the intercept is treated as the first parameter of the regression
            and the coefficients as the remaining. If outcome y is p-dimensional, then this is a tensor whose
            p-th entry contains the co-variance matrix for the parameters corresponding to the regression of
            the p-th coordinate of the outcome.
        """
        return np.array(self._var)

    @property
    def _param_stderr(self):
        """
        Returns
        -------
        _param_stderr : {(d (+1),) (d (+1), p)} nd array like
            The standard error of each parameter that was estimated.
        """
        if self._n_out == 0:
            return np.sqrt(np.clip(np.diag(self._param_var), 0, np.inf))
        else:
            return np.array([np.sqrt(np.clip(np.diag(v), 0, np.inf)) for v in self._param_var]).T

    @property
    def coef_stderr_(self):
        """
        Gets the standard error of the fitted coefficients.

        Returns
        -------
        coef_stderr_ : {(d,), (p, d)} nd array like
            The standard error of the coefficients
        """
        return self._param_stderr[1:].T if self._fit_intercept else self._param_stderr.T

    @property
    def intercept_stderr_(self):
        """
        Gets the standard error of the intercept(s) (or 0 if no intercept was fit).

        Returns
        -------
        intercept_stderr_ : float or (p,) nd array like
            The standard error of the intercept(s)
        """
        return self._param_stderr[0] if self._fit_intercept else (0 if self._n_out == 0 else np.zeros(self._n_out))

    def prediction_stderr(self, X):
        """
        Gets the standard error of the predictions.

        Parameters
        ----------
        X : (n, d) array like
            The covariates at which to predict

        Returns
        -------
        prediction_stderr : (n, p) array like
            The standard error of each coordinate of the output at each point we predict
        """
        if self._fit_intercept:
            X = add_constant(X, has_constant='add')
        if self._n_out == 0:
            return np.sqrt(np.clip(np.sum(np.matmul(X, self._param_var) * X, axis=1), 0, np.inf))
        else:
            return np.array([np.sqrt(np.clip(np.sum(np.matmul(X, v) * X, axis=1), 0, np.inf)) for v in self._var]).T

    def coef__interval(self, alpha=.05):
        """
        Gets a confidence interval bounding the fitted coefficients.

        Parameters
        ----------
        alpha : float
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        coef__interval : {tuple ((p, d) array, (p,d) array), tuple ((d,) array, (d,) array)}
            The lower and upper bounds of the confidence interval of the coefficients
        """
        return np.array([_safe_norm_ppf(alpha / 2, loc=p, scale=err)
                         for p, err in zip(self.coef_, self.coef_stderr_)]),\
            np.array([_safe_norm_ppf(1 - alpha / 2, loc=p, scale=err)
                      for p, err in zip(self.coef_, self.coef_stderr_)])

    def intercept__interval(self, alpha=.05):
        """
        Gets a confidence interval bounding the intercept(s) (or 0 if no intercept was fit).

        Parameters
        ----------
        alpha : float
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        intercept__interval : {tuple ((p,) array, (p,) array), tuple (float, float)}
            The lower and upper bounds of the confidence interval of the intercept(s)
        """
        if not self._fit_intercept:
            return (0 if self._n_out == 0 else np.zeros(self._n_out)),\
                (0 if self._n_out == 0 else np.zeros(self._n_out))

        if self._n_out == 0:
            return _safe_norm_ppf(alpha / 2, loc=self.intercept_, scale=self.intercept_stderr_),\
                _safe_norm_ppf(1 - alpha / 2, loc=self.intercept_, scale=self.intercept_stderr_)
        else:
            return np.array([_safe_norm_ppf(alpha / 2, loc=p, scale=err)
                             for p, err in zip(self.intercept_, self.intercept_stderr_)]),\
                np.array([_safe_norm_ppf(1 - alpha / 2, loc=p, scale=err)
                          for p, err in zip(self.intercept_, self.intercept_stderr_)])

    def predict_interval(self, X, alpha=.05):
        """
        Gets a confidence interval bounding the prediction.

        Parameters
        ----------
        X : (n, d) array like
            The covariates on which to predict
        alpha : float
            The confidence level. Will calculate the alpha/2-quantile and the (1-alpha/2)-quantile
            of the parameter distribution as confidence interval

        Returns
        -------
        prediction_intervals : {tuple ((n,) array, (n,) array), tuple ((n,p) array, (n,p) array)}
            The lower and upper bounds of the confidence intervals of the predicted mean outcomes
        """
        return np.array([_safe_norm_ppf(alpha / 2, loc=p, scale=err)
                         for p, err in zip(self.predict(X), self.prediction_stderr(X))]),\
            np.array([_safe_norm_ppf(1 - alpha / 2, loc=p, scale=err)
                      for p, err in zip(self.predict(X), self.prediction_stderr(X))])


class LassoCVWrapper:
    """Helper class to wrap either LassoCV or MultiTaskLassoCV depending on the shape of the target."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, X, Y):
        assert shape(X)[0] == shape(Y)[0]
        assert ndim(Y) <= 2
        self.needs_unravel = False
        if ndim(Y) == 2 and shape(Y)[1] > 1:
            self.model = MultiTaskLassoCV(*self.args, **self.kwargs)
        else:
            if ndim(Y) == 2 and shape(Y)[1] == 1:
                Y = np.ravel(Y)
                self.needs_unravel = True
            self.model = LassoCV(*self.args, **self.kwargs)
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        predictions = self.model.predict(X)
        return reshape(predictions, (-1, 1)) if self.needs_unravel else predictions
