# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Utility methods."""

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
import sparse as sp
import itertools
import inspect
import types
from operator import getitem
from collections import defaultdict, Counter
from sklearn import clone
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LassoCV, MultiTaskLassoCV, Lasso, MultiTaskLasso
from functools import reduce, wraps
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import assert_all_finite
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, LabelEncoder
import warnings
from warnings import warn
from collections.abc import Iterable
from sklearn.utils.multiclass import type_of_target
import numbers
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.summary import summary_return
from statsmodels.compat.python import lmap
import copy
from inspect import signature

MAX_RAND_SEED = np.iinfo(np.int32).max


class IdentityFeatures(TransformerMixin):
    """Featurizer that just returns the input data."""

    def fit(self, X):
        """Fit method (does nothing, just returns self)."""
        return self

    def transform(self, X):
        """Perform the identity transform, which returns the input unmodified."""
        return X


def parse_final_model_params(coef, intercept, d_y, d_t, d_t_in, bias_part_of_coef, fit_cate_intercept):
    dt = d_t
    if (d_t_in != d_t) and (d_t and d_t[0] == 1):  # binary treatment or single dim featurized treatment
        dt = ()
    cate_intercept = None
    if bias_part_of_coef:
        cate_coef = coef.reshape(d_y + dt + (-1,))[..., 1:]
        if fit_cate_intercept:
            cate_intercept = coef.reshape(d_y + dt + (-1,))[..., 0]
    else:
        cate_coef = coef.reshape(d_y + dt + (-1,))
        if fit_cate_intercept:
            cate_intercept = np.reshape(intercept, d_y + dt)
    if (cate_intercept is not None) and (np.ndim(cate_intercept) == 0):
        cate_intercept = cate_intercept.item()
    return cate_coef, cate_intercept


def check_high_dimensional(X, T, *, threshold, featurizer=None, discrete_treatment=False, msg=""):
    # Check if model is sparse enough for this model
    X, T = check_input_arrays(X, T)
    if X is None:
        d_x = 1
    elif featurizer is None:
        d_x = X.shape[1]
    else:
        d_x = clone(featurizer, safe=False).fit_transform(X[[0], :]).shape[1]
    if discrete_treatment:
        d_t = len(set(T.flatten())) - 1
    else:
        d_t = 1 if np.ndim(T) < 2 else T.shape[1]
    if d_x * d_t < threshold:
        warn(msg, UserWarning)


def inverse_onehot(T):
    """
    Given a one-hot encoding of a value, return a vector reversing the encoding to get numeric treatment indices.

    Note that we assume that the first column has been removed from the input.

    Parameters
    ----------
    T : array (shape (n, d_t-1))
        The one-hot-encoded array

    Returns
    -------
    A : vector of int (shape (n,))
        The un-encoded 0-based category indices
    """
    assert ndim(T) == 2
    # note that by default OneHotEncoder returns float64s, so need to convert to int
    return (T @ np.arange(1, T.shape[1] + 1)).astype(int)


def issparse(X):
    """Determine whether an input is sparse.

    For the purposes of this function, both `scipy.sparse` matrices and `sparse.SparseArray`
    types are considered sparse.

    Parameters
    ----------
    X : array_like
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
    X : array_like
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

    shape : tuple of int
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

    Returns
    -------
    A : n x (d1*d2*...) matrix
        Matrix of n samples of d1*d2*... cross product features,
        arranged in form such that each row t of X12 contains:
        [X1[t,0]*X2[t,0]*..., ..., X1[t,d1-1]*X2[t,0]*..., X1[t,0]*X2[t,1]*..., ..., X1[t,d1-1]*X2[t,1]*..., ...]

    """
    for X in XS:
        assert 2 >= ndim(X) >= 1
    n = shape(XS[0])[0]
    for X in XS:
        assert n == shape(X)[0]

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
    XS : sequence of ndarray
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
    XS : sequence of ndarray
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
    axes :  list of int, optional
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


def add_intercept(X):
    """
    Adds an intercept feature to an array by prepending a column of ones.

    Parameters
    ----------
    X : array_like
        Input array.  Must be 2D.

    Returns
    -------
    arr : ndarray
        `X` with a column of ones prepended
    """
    return hstack([np.ones((X.shape[0], 1)), X])


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
    assert (len(Y) == len(T))
    assert (Y.ndim <= 2)
    if Y.ndim == 2:
        assert (Y.shape[1] == 1)
        Y = Y.flatten()
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    return Y, T


def check_inputs(Y, T, X, W=None, multi_output_T=True, multi_output_Y=True,
                 force_all_finite_X=True, force_all_finite_W=True):
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

    X : array_like, shape (n, d_x)
        Feature vector that captures heterogeneity.

    W : array_like, shape (n, d_w), optional
        High-dimensional controls.

    multi_output_T : bool
        Whether to allow more than one treatment.

    multi_output_Y: bool
        Whether to allow more than one outcome.

    force_all_finite_X : bool or 'allow-nan', default True
        Whether to allow inf and nan in input arrays in X.
        'allow-nan': accepts only np.nan and pd.NA values in array. Values
        cannot be infinite.

    force_all_finite_W : bool or 'allow-nan', default True
        Whether to allow inf and nan in input arrays in W.
        'allow-nan': accepts only np.nan and pd.NA values in array. Values
        cannot be infinite.

    Returns
    -------
    Y : array_like, shape (n, ) or (n, d_y)
        Converted and validated Y.

    T : array_like, shape (n, ) or (n, d_t)
        Converted and validated T.

    X : array_like, shape (n, d_x)
        Converted and validated X.

    W : array_like, shape (n, d_w), optional
        Converted and validated W.

    """
    X, T = check_X_y(X, T, multi_output=multi_output_T, y_numeric=True, force_all_finite=force_all_finite_X)
    if force_all_finite_X == 'allow-nan':
        try:
            assert_all_finite(X)
        except ValueError:
            warnings.warn("X contains NaN. Causal identification strategy can be erroneous"
                          " in the presence of missing values.")
    _, Y = check_X_y(X, Y, multi_output=multi_output_Y, y_numeric=True, force_all_finite=force_all_finite_X)
    if W is not None:
        W, _ = check_X_y(W, Y, multi_output=multi_output_Y, y_numeric=True, force_all_finite=force_all_finite_W)
        if force_all_finite_W == 'allow-nan':
            try:
                assert_all_finite(W)
            except ValueError:
                warnings.warn("W contains NaN. Causal identification strategy can be erroneous"
                              " in the presence of missing values.")
    return Y, T, X, W


def check_input_arrays(*args, validate_len=True, force_all_finite=True, dtype=None):
    """Cast input sequences into numpy arrays.

    Only inputs that are sequence-like will be converted, all other inputs will be left as is.
    When `validate_len` is True, the sequences will be checked for equal length.

    Parameters
    ----------
    args : scalar or array_like
        Inputs to be checked.

    validate_len : bool, default True
        Whether to check if the input arrays have the same length.

    force_all_finite : bool or 'allow-nan', default True
        Whether to allow inf and nan in input arrays.
        'allow-nan': accepts only np.nan and pd.NA values in array. Values
        cannot be infinite.

    dtype : 'numeric', type, list of type, optional
        Argument passed to sklearn.utils.check_array.
        Specifies data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    Returns
    -------
    args: array_like
        List of inputs where sequence-like objects have been cast to numpy arrays.

    """
    n = None
    args = list(args)
    for i, arg in enumerate(args):
        if np.ndim(arg) > 0:
            new_arg = check_array(arg, dtype=dtype, ensure_2d=False, accept_sparse=True,
                                  force_all_finite=force_all_finite)
            if not force_all_finite:
                # For when checking input values is disabled
                try:
                    assert_all_finite(new_arg)
                except ValueError:
                    warnings.warn("Input contains NaN, infinity or a value too large for dtype('float64') "
                                  "but input check is disabled. Check the inputs before proceeding.")
            elif force_all_finite == 'allow-nan':
                try:
                    assert_all_finite(new_arg)
                except ValueError:
                    warnings.warn("Input contains NaN. Causal identification strategy can be erroneous"
                                  " in the presence of missing values.")

            if validate_len:
                m = new_arg.shape[0]
                if n is None:
                    n = m
                else:
                    assert (m == n), "Input arrays have incompatible lengths: {} and {}".format(n, m)
            args[i] = new_arg
    return args


def get_input_columns(X, prefix="X"):
    """Extracts column names from dataframe-like input object.

    Currently supports column name extraction from pandas DataFrame and Series objects.

    Parameters
    ----------
    X : array_like or None
        Input array with column names to be extracted.

    prefix: str, default "X"
        If input array doesn't have column names, a default using the naming scheme
        "{prefix}{column number}" will be returned.

    Returns
    -------
    cols: array_like or None
        List of columns corresponding to the dataframe-like object.
        None if the input array is not in the supported types.
    """
    if X is None:
        return None
    if np.ndim(X) == 0:
        raise ValueError(
            f"Expected array_like object for imput with prefix {prefix} but got '{X}' object instead.")
    # Type to column extraction function
    type_to_func = {
        pd.DataFrame: lambda x: x.columns.tolist(),
        pd.Series: lambda x: [x.name]
    }
    if type(X) in type_to_func:
        column_names = type_to_func[type(X)](X)

        # if not all column names are strings
        if not all(isinstance(item, str) for item in column_names):
            warnings.warn("Not all column names are strings. Coercing to strings for now.", UserWarning)

        return [str(item) for item in column_names]

    len_X = 1 if np.ndim(X) == 1 else np.asarray(X).shape[1]
    return [f"{prefix}{i}" for i in range(len_X)]


def get_feature_names_or_default(featurizer, feature_names, prefix="feat(X)"):
    """
    Extract feature names from sklearn transformers. Otherwise attempts to assign default feature names.

    Designed to be compatible with old and new sklearn versions.

    Parameters
    ----------
    featurizer ： featurizer to extract feature names from
    feature_names : sequence of str
        input features
    prefix : str, default "feat(X)"
        output prefix in the event where we assign default feature names

    Returns
    ----------
    feature_names_out : list of str
        The feature names
    """

    # coerce feature names to be strings
    if not all(isinstance(item, str) for item in feature_names):
        warnings.warn("Not all feature names are strings. Coercing to strings for now.", UserWarning)
    feature_names = [str(item) for item in feature_names]

    # Prefer sklearn 1.0's get_feature_names_out method to deprecated get_feature_names method
    if hasattr(featurizer, "get_feature_names_out"):
        try:
            return featurizer.get_feature_names_out(feature_names)
        except Exception:
            # Some featurizers will throw, such as a pipeline with a transformer that doesn't itself support names
            pass
    if hasattr(featurizer, 'get_feature_names'):
        try:
            # Get number of arguments, some sklearn featurizer don't accept feature_names
            arg_no = len(inspect.getfullargspec(featurizer.get_feature_names).args)
            if arg_no == 1:
                return featurizer.get_feature_names()
            elif arg_no == 2:
                return featurizer.get_feature_names(feature_names)
        except Exception:
            # Handles cases where the passed feature names create issues
            pass
    # Featurizer doesn't have 'get_feature_names' or has atypical 'get_feature_names'
    try:
        # Get feature names using featurizer
        dummy_X = np.ones((1, len(feature_names)))
        return get_input_columns(featurizer.transform(dummy_X), prefix=prefix)
    except Exception:
        # All attempts at retrieving transformed feature names have failed
        # Delegate handling to downstream logic
        return None


def check_models(models, n):
    """
    Input validation for metalearner models.

    Check whether the input models satisfy the criteria below.

    Parameters
    ----------
    models ： estimator or list or tuple of estimators
    n : int
        Number of models needed

    Returns
    ----------
    models : list or tuple of estimator

    """
    if isinstance(models, (tuple, list)):
        if n != len(models):
            raise ValueError("The number of estimators doesn't equal to the number of treatments. "
                             "Please provide either a tuple/list of estimators "
                             "with same number of treatments or an unified estimator.")
    elif hasattr(models, 'fit'):
        models = [clone(models, safe=False) for i in range(n)]
    else:
        raise ValueError(
            "models must be either a tuple/list of estimators with same number of treatments "
            "or an unified estimator.")
    return models


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

    arrs : list of `sparse.COO`
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

    return sp.COO(np.array(list(results.keys())).T if results else
                  np.empty((len(outputs), 0)),
                  np.array(list(results.values())),
                  [arrs[indMap[c][0][0]].shape[indMap[c][0][1]] for c in outputs])


def filter_none_kwargs(**kwargs):
    """
    Filters out any keyword arguments that are None.

    This is useful when specific optional keyword arguments might not be universally supported,
    so that stripping them out when they are not set enables more uses to succeed.

    Parameters
    ----------
    kwargs: dict
        The keyword arguments to filter

    Returns
    -------
    filtered_kwargs: dict
        The input dictionary, but with all entries having value None removed
    """
    return {key: value for key, value in kwargs.items() if value is not None}


class WeightedModelWrapper:
    """Helper class for assiging weights to models without this option.

    Parameters
    ----------
    model_instance : estimator
        Model that requires weights.

    sample_type : str, default `weighted`
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
            warnings.warn("The model provided does not support sample weights. "
                          "Manual weighted sampling may icrease the variance in the results.", UserWarning)
            self.data_transform = self._sampled_inputs

    def fit(self, X, y, sample_weight=None):
        """Fit underlying model instance with weighted inputs.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            Training data.

        y : array_like, shape (n_samples, n_outcomes)
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
        X : array_like or sparse matrix, shape (n_samples, n_features)
            Samples.

        Returns
        -------
        C : array, shape (n_samples, n_outcomes)
            Returns predicted values.
        """
        return self.model_instance.predict(X)

    def _weighted_inputs(self, X, y, sample_weight):
        X, y = check_X_y(X, y, y_numeric=True, multi_output=True)
        normalized_weights = sample_weight * X.shape[0] / np.sum(sample_weight)
        sqrt_weights = np.sqrt(normalized_weights)
        weighted_X = sqrt_weights.reshape(-1, 1) * X
        weighted_y = sqrt_weights.reshape(-1, 1) * y if y.ndim > 1 else sqrt_weights * y
        return weighted_X, weighted_y

    def _sampled_inputs(self, X, y, sample_weight):
        # Normalize weights
        normalized_weights = sample_weight / np.sum(sample_weight)
        data_length = int(min(1 / np.min(normalized_weights[normalized_weights > 0]), 10) * X.shape[0])
        data_indices = np.random.choice(X.shape[0], size=data_length, p=normalized_weights)
        return X[data_indices], y[data_indices]


class MultiModelWrapper:
    """Helper class for training different models for each treatment.

    Parameters
    ----------
    model_list : array_like, shape (n_T, )
        List of models to be trained separately for each treatment group.
    """

    def __init__(self, model_list=[]):
        self.model_list = model_list
        self.n_T = len(model_list)

    def fit(self, Xt, y, sample_weight=None):
        """Fit underlying list of models with weighted inputs.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features + n_treatments)
            Training data. The last n_T columns should be a one-hot encoding of the treatment assignment.

        y : array_like, shape (n_samples, )
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
        X : array_like, shape (n_samples, n_features + n_treatments)
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
    if hasattr(loc, "__len__"):
        prelim = loc.copy()
        if np.any(scale > 0):
            prelim[scale > 0] = scipy.stats.norm.ppf(q, loc=loc[scale > 0], scale=scale[scale > 0])
    elif scale > 0:
        prelim = scipy.stats.norm.ppf(q, loc=loc, scale=scale)
    else:
        prelim = loc
    return prelim


class Summary:
    # This class is mainly derived from statsmodels.iolib.summary.Summary
    """
    Result summary

    Construction does not take any parameters. Tables and text can be added
    with the `add_` methods.

    Attributes
    ----------
    tables : list of table
        Contains the list of SimpleTable instances, horizontally concatenated
        tables are not saved separately.
    extra_txt : str
        extra lines that are added to the text output, used for warnings
        and explanations.
    """

    def __init__(self):
        self.tables = []
        self.extra_txt = None

    def __str__(self):
        return self.as_text()

    def __repr__(self):
        return str(type(self)) + '\n"""\n' + self.__str__() + '\n"""'

    def _repr_html_(self):
        '''Display as HTML in IPython notebook.'''
        return self.as_html()

    def add_table(self, res, header, index, title):
        table = SimpleTable(res, header, index, title)
        self.tables.append(table)

    def add_extra_txt(self, etext):
        '''add additional text that will be added at the end in text format

        Parameters
        ----------
        etext : list[str]
            string with lines that are added to the text output.

        '''
        self.extra_txt = '\n'.join(etext)

    def as_text(self):
        '''return tables as string

        Returns
        -------
        txt : str
            summary tables and extra text as one string

        '''
        txt = summary_return(self.tables, return_fmt='text')
        if self.extra_txt is not None:
            txt = txt + '\n\n' + self.extra_txt
        return txt

    def as_latex(self):
        '''return tables as string

        Returns
        -------
        latex : str
            summary tables and extra text as string of Latex

        Notes
        -----
        This currently merges tables with different number of columns.
        It is recommended to use `as_latex_tabular` directly on the individual
        tables.

        '''
        latex = summary_return(self.tables, return_fmt='latex')
        if self.extra_txt is not None:
            latex = latex + '\n\n' + self.extra_txt.replace('\n', ' \\newline\n ')
        return latex

    def as_csv(self):
        '''return tables as string

        Returns
        -------
        csv : str
            concatenated summary tables in comma delimited format

        '''
        csv = summary_return(self.tables, return_fmt='csv')
        if self.extra_txt is not None:
            csv = csv + '\n\n' + self.extra_txt
        return csv

    def as_html(self):
        '''return tables as string

        Returns
        -------
        html : str
            concatenated summary tables in HTML format

        '''
        html = summary_return(self.tables, return_fmt='html')
        if self.extra_txt is not None:
            html = html + '<br/><br/>' + self.extra_txt.replace('\n', '<br/>')
        return html


class SeparateModel:
    """
    Splits the data based on the last feature and trains
    a separate model for each subsample. At predict time, it
    uses the last feature to choose which model to use
    to predict.
    """

    def __init__(self, *models):
        self.models = [clone(model) for model in models]

    def fit(self, XZ, T):
        for (i, m) in enumerate(self.models):
            inds = (XZ[:, -1] == i)
            m.fit(XZ[inds, :-1], T[inds])
        return self

    def predict(self, XZ):
        t_pred = np.zeros(XZ.shape[0])
        for (i, m) in enumerate(self.models):
            inds = (XZ[:, -1] == i)
            if np.any(inds):
                t_pred[inds] = m.predict(XZ[inds, :-1])
        return t_pred

    @property
    def coef_(self):
        return np.concatenate((model.coef_ for model in self.models))


def deprecated(message, category=FutureWarning):
    """
    Enables decorating a method or class to providing a warning when it is used.

    Parameters
    ----------
    message: str
        The deprecation message to use
    category:  :class:`type`, default :class:`FutureWarning`
        The warning category to use
    """
    def decorator(to_wrap):

        # if we're decorating a class, just update the __init__ method,
        # so that the result is still a class instead of a wrapper method
        if isinstance(to_wrap, type):
            old_init = to_wrap.__init__

            @wraps(to_wrap.__init__)
            def new_init(*args, **kwargs):
                warn(message, category, stacklevel=2)
                old_init(*args, **kwargs)

            to_wrap.__init__ = new_init

            return to_wrap
        else:
            @wraps(to_wrap)
            def m(*args, **kwargs):
                warn(message, category, stacklevel=2)
                return to_wrap(*args, **kwargs)
            return m
    return decorator


def _deprecate_positional(message, bad_args, category=FutureWarning):
    """
    Enables decorating a method to provide a warning when certain arguments are used positionally.

    Parameters
    ----------
    message: str
        The deprecation message to use
    bad_args : list of str
        The positional arguments that will be keyword-only in the future
    category:  :class:`type`, default :class:`FutureWarning`
        The warning category to use
    """
    def decorator(to_wrap):
        @wraps(to_wrap)
        def m(*args, **kwargs):
            # want to enforce that each bad_arg was either in kwargs,
            # or else it was in neither and is just taking its default value
            bound = signature(m).bind(*args, **kwargs)

            wrong_args = False
            for arg in bad_args:
                if arg not in kwargs and arg in bound.arguments:
                    wrong_args = True
            if wrong_args:
                warn(message, category, stacklevel=2)
            return to_wrap(*args, **kwargs)
        return m
    return decorator


class MissingModule:
    """
    Placeholder to stand in for a module that couldn't be imported, delaying ImportErrors until use.

    Parameters
    ----------
    msg:str
        The message to display when an attempt to access a module memeber is made
    exn:ImportError
        The original ImportError to pass as the source of the exception
    """

    def __init__(self, msg, exn):
        self.msg = msg
        self.exn = exn

    # Any access should throw
    def __getattr__(self, _):
        raise ImportError(self.msg) from self.exn

    # As a convenience, also throw on calls to allow MissingModule to be used in lieu of specific imports
    def __call__(self, *args, **kwargs):
        raise ImportError(self.msg) from self.exn


def transpose_dictionary(d):
    """
    Transpose a dictionary of dictionaries, bringing the keys from the second level
    to the top and vice versa

    Parameters
    ----------
    d: dict
        The dictionary to transpose; the values of this dictionary should all themselves
        be dictionaries

    Returns
    -------
    output: dict
        The output dictionary with first- and second-level keys swapped
    """
    output = defaultdict(dict)
    for key1, value in d.items():
        for key2, val in value.items():
            output[key2][key1] = val

    # return plain dictionary so that erroneous accesses don't half work (see e.g. #708)
    return dict(output)


def reshape_arrays_2dim(length, *args):
    """
    Reshape the input arrays as two dimensional.
    If None, will be reshaped as (n, 0).

    Parameters
    ----------
    length: scalar
        Number of samples
    args: tuple of array_like
        Inputs to be reshaped

    Returns
    -------
    new_args: list of array
        Output of reshaped arrays
    """
    new_args = []
    for arg in args:
        if arg is None:
            new_args.append(np.array([]).reshape(length, 0))
        elif arg.ndim == 1:
            new_args.append(arg.reshape((-1, 1)))
        else:
            new_args.append(arg)
    return new_args


class _RegressionWrapper:
    """
    A simple wrapper that makes a binary classifier behave like a regressor.
    Essentially .fit, calls the fit method of the classifier and
    .predict calls the .predict_proba method of the classifier
    and returns the probability of label 1.
    """

    def __init__(self, clf):
        """
        Parameters
        ----------
        clf : the classifier model
        """
        self._clf = clf

    def fit(self, X, y, **kwargs):
        """
        Parameters
        ----------
        X : features
        y : one-hot-encoding of binary label, with drop='first'
        """
        if len(y.shape) > 1 and y.shape[1] > 1:
            y = y @ np.arange(1, y.shape[1] + 1)
        self._clf.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : features
        """
        return self._clf.predict_proba(X)[:, 1:]


class _TransformerWrapper:
    """Wrapper that takes a featurizer as input and adds jacobian calculation functionality"""

    def __init__(self, featurizer):
        self.featurizer = featurizer

    def fit(self, X):
        return self.featurizer.fit(X)

    def transform(self, X):
        return self.featurizer.transform(X)

    def fit_transform(self, X):
        return self.featurizer.fit_transform(X)

    def get_feature_names_out(self, feature_names):
        return get_feature_names_or_default(self.featurizer, feature_names, prefix="feat(T)")

    def jac(self, X, epsilon=0.001):
        if hasattr(self.featurizer, 'jac'):
            return self.featurizer.jac(X)
        elif (isinstance(self.featurizer, PolynomialFeatures)):
            powers = self.featurizer.powers_
            result = np.zeros(X.shape + (self.featurizer.n_output_features_,))
            for i in range(X.shape[1]):
                p = powers.copy()
                c = powers[:, i]
                p[:, i] -= 1
                M = np.float_power(X[:, np.newaxis, :], p[np.newaxis, :, :])
                result[:, i, :] = c[np.newaxis, :] * np.prod(M, axis=-1)
            return result

        else:
            squeeze = []

            n = X.shape[0]
            d_t = X.shape[-1] if ndim(X) > 1 else 1
            X_out = self.transform(X)
            d_f_t = X_out.shape[-1] if ndim(X_out) > 1 else 1

            jacob = np.zeros((n, d_t, d_f_t))

            if ndim(X) == 1:
                squeeze.append(1)
                X = X[:, np.newaxis]
            if ndim(X_out) == 1:
                squeeze.append(2)

            # for every dimension of the treatment add some epsilon and observe change in featurized treatment
            for k in range(d_t):
                eps_matrix = np.zeros(shape=X.shape)
                eps_matrix[:, k] = epsilon

                X_in_plus = X + eps_matrix
                X_in_plus = X_in_plus.squeeze(axis=1) if 1 in squeeze else X_in_plus
                X_out_plus = self.transform(X_in_plus)
                X_out_plus = X_out_plus[:, np.newaxis] if 2 in squeeze else X_out_plus

                X_in_minus = X - eps_matrix
                X_in_minus = X_in_minus.squeeze(axis=1) if 1 in squeeze else X_in_minus
                X_out_minus = self.transform(X_in_minus)
                X_out_minus = X_out_minus[:, np.newaxis] if 2 in squeeze else X_out_minus

                diff = X_out_plus - X_out_minus
                deriv = diff / (2 * epsilon)

                jacob[:, k, :] = deriv

            return jacob.squeeze(axis=tuple(squeeze))


def jacify_featurizer(featurizer):
    """
       Function that takes a featurizer as input and returns a wrapper class that includes
       a function for calculating the jacobian
    """
    return _TransformerWrapper(featurizer)


def strata_from_discrete_arrays(arrs):
    """
    Combine multiple discrete arrays into a single array for stratification purposes:

    e.g. if arrs are
    [0 1 2 0 1 2 0 1 2 0 1 2],
    [0 1 0 1 0 1 0 1 0 1 0 1],
    [0 0 0 0 0 0 1 1 1 1 1 1]
    then output will be
    [0 8 4 6 2 10 1 9 5 7 3 11]

    Every distinct combination of these discrete arrays will have it's own label.
    """
    if not arrs:
        return None

    curr_array = np.zeros(shape=np.ravel(arrs[0]).shape, dtype='int')

    for arr in arrs:
        enc = LabelEncoder()
        temp = enc.fit_transform(np.ravel(arr))
        curr_array = temp + curr_array * len(enc.classes_)

    return curr_array


def one_hot_encoder(sparse=False, **kwargs):
    """
    Wrapper for sklearn's OneHotEncoder that handles the name change from `sparse` to `sparse_output`
    between sklearn versions 1.1 and 1.2.
    """
    from packaging.version import parse
    if parse(sklearn.__version__) < parse("1.2"):
        return OneHotEncoder(sparse=sparse, **kwargs)
    else:
        return OneHotEncoder(sparse_output=sparse, **kwargs)
