# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Bootstrap sampling."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from scipy.stats import norm
from collections import OrderedDict
import pandas as pd


class BootstrapEstimator:
    """Estimator that uses bootstrap sampling to wrap an existing estimator.

    This estimator provides a `fit` method with the same signature as the wrapped estimator.

    The bootstrap estimator will also wrap all other methods and attributes of the wrapped estimator,
    but return the average of the sampled calculations (this will fail for non-numeric outputs).

    It will also provide a wrapper method suffixed with `_interval` for each method or attribute of
    the wrapped estimator that takes two additional optional keyword arguments `lower` and `upper` specifiying
    the percentiles of the interval, and which uses `np.percentile` to return the corresponding lower
    and upper bounds based on the sampled calculations.  For example, if the underlying estimator supports
    an `effect` method with signature `(X,T) -> Y`, this class will provide a method `effect_interval`
    with pseudo-signature `(lower=5, upper=95, X, T) -> (Y, Y)` (where `lower` and `upper` cannot be
    supplied as positional arguments).

    Parameters
    ----------
    wrapped : object
        The basis for the clones used for estimation.
        This object must support a `fit` method which takes numpy arrays with consistent first dimensions
        as arguments.

    n_bootstrap_samples : int, default: 100
        How many draws to perform.

    n_jobs: int, default: None
        The maximum number of concurrently running jobs, as in joblib.Parallel.

    verbose: int, default: 0
        Verbosity level

    compute_means : bool, default: True
        Whether to pass calls through to the underlying collection and return the mean.  Setting this
        to ``False`` can avoid ambiguities if the wrapped object itself has method names with an `_interval` suffix.

    bootstrap_type: 'percentile', 'pivot', or 'normal', default 'pivot'
        Bootstrap method used to compute results.  'percentile' will result in using the empiracal CDF of
        the replicated computations of the statistics.   'pivot' will also use the replicates but create a pivot
        interval that also relies on the estimate over the entire dataset.  'normal' will instead compute an interval
        assuming the replicates are normally distributed.
    """

    def __init__(self, wrapped,
                 n_bootstrap_samples=100,
                 n_jobs=None,
                 verbose=0,
                 compute_means=True,
                 bootstrap_type='pivot'):
        self._instances = [clone(wrapped, safe=False) for _ in range(n_bootstrap_samples)]
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs
        self._verbose = verbose
        self._compute_means = compute_means
        self._bootstrap_type = bootstrap_type
        self._wrapped = wrapped

    # TODO: Add a __dir__ implementation?

    @staticmethod
    def __stratified_indices(arr):
        assert 1 <= np.ndim(arr) <= 2
        unique = np.unique(arr, axis=0)
        indices = []
        for el in unique:
            ind, = np.where(np.all(arr == el, axis=1) if np.ndim(arr) == 2 else arr == el)
            indices.append(ind)
        return indices

    def fit(self, *args, **named_args):
        """
        Fit the model.

        The full signature of this method is the same as that of the wrapped object's `fit` method.
        """
        from .._cate_estimator import BaseCateEstimator  # need to nest this here to avoid circular import

        index_chunks = None
        if isinstance(self._instances[0], BaseCateEstimator):
            index_chunks = self._instances[0]._strata(*args, **named_args)
            if index_chunks is not None:
                index_chunks = self.__stratified_indices(index_chunks)
        if index_chunks is None:
            n_samples = np.shape(args[0] if args else named_args[(*named_args,)[0]])[0]
            index_chunks = [np.arange(n_samples)]  # one chunk with all indices

        indices = []
        for chunk in index_chunks:
            n_samples = len(chunk)
            indices.append(chunk[np.random.choice(n_samples,
                                                  size=(self._n_bootstrap_samples, n_samples),
                                                  replace=True)])

        indices = np.hstack(indices)

        def fit(x, *args, **kwargs):
            x.fit(*args, **kwargs)
            return x  # Explicitly return x in case fit fails to return its target

        def convertArg(arg, inds):
            if arg is None:
                return None
            arr = np.asarray(arg)
            if arr.ndim > 0:
                return arr[inds]
            else:  # arg was a scalar, so we shouldn't have converted it
                return arg

        self._instances = Parallel(n_jobs=self._n_jobs, prefer='threads', verbose=self._verbose)(
            delayed(fit)(obj,
                         *[convertArg(arg, inds) for arg in args],
                         **{arg: convertArg(named_args[arg], inds) for arg in named_args})
            for obj, inds in zip(self._instances, indices)
        )
        return self

    def __getattr__(self, name):
        """
        Get proxy attribute that wraps the corresponding attribute with the same name from the wrapped object.

        Additionally, the suffix "_interval" is supported for getting an interval instead of a point estimate.
        """

        # don't proxy special methods
        if name.startswith('__'):
            raise AttributeError(name)

        def proxy(make_call, name, summary):
            def summarize_with(f):
                results = np.array(Parallel(n_jobs=self._n_jobs, prefer='threads', verbose=self._verbose)(
                    (f, (obj, name), {}) for obj in self._instances)), f(self._wrapped, name)
                return summary(*results)
            if make_call:
                def call(*args, **kwargs):
                    return summarize_with(lambda obj, name: getattr(obj, name)(*args, **kwargs))
                return call
            else:
                return summarize_with(lambda obj, name: getattr(obj, name))

        def get_mean():
            # for attributes that exist on the wrapped object, just compute the mean of the wrapped calls
            return proxy(callable(getattr(self._instances[0], name)), name, lambda arr, _: np.mean(arr, axis=0))

        def get_std():
            prefix = name[: - len('_std')]
            return proxy(callable(getattr(self._instances[0], prefix)), prefix,
                         lambda arr, _: np.std(arr, axis=0))

        def get_interval():
            # if the attribute exists on the wrapped object once we remove the suffix,
            # then we should be computing a confidence interval for the wrapped calls
            prefix = name[: - len("_interval")]

            def call_with_bounds(can_call, lower, upper):
                def percentile_bootstrap(arr, _):
                    return np.percentile(arr, lower, axis=0), np.percentile(arr, upper, axis=0)

                def pivot_bootstrap(arr, est):
                    return 2 * est - np.percentile(arr, upper, axis=0), 2 * est - np.percentile(arr, lower, axis=0)

                def normal_bootstrap(arr, est):
                    std = np.std(arr, axis=0)
                    return est - norm.ppf(upper / 100) * std, est - norm.ppf(lower / 100) * std

                # TODO: studentized bootstrap? this would be more accurate in most cases but can we avoid
                #       second level bootstrap which would be prohibitive computationally?

                fn = {'percentile': percentile_bootstrap,
                      'normal': normal_bootstrap,
                      'pivot': pivot_bootstrap}[self._bootstrap_type]
                return proxy(can_call, prefix, fn)

            can_call = callable(getattr(self._instances[0], prefix))
            if can_call:
                # collect extra arguments and pass them through, if the wrapped attribute was callable
                def call(*args, lower=5, upper=95, **kwargs):
                    return call_with_bounds(can_call, lower, upper)(*args, **kwargs)
                return call
            else:
                # don't pass extra arguments if the wrapped attribute wasn't callable to begin with
                def call(lower=5, upper=95):
                    return call_with_bounds(can_call, lower, upper)
                return call

        def get_inference():
            # can't import from econml.inference at top level without creating cyclical dependencies
            from ._inference import EmpiricalInferenceResults, NormalInferenceResults
            from .._cate_estimator import LinearModelFinalCateEstimatorDiscreteMixin

            prefix = name[: - len("_inference")]

            def fname_transformer(x):
                return x

            if prefix in ['const_marginal_effect', 'marginal_effect', 'effect']:
                inf_type = 'effect'
            elif prefix == 'coef_':
                inf_type = 'coefficient'
                if (hasattr(self._instances[0], 'cate_feature_names') and
                        callable(self._instances[0].cate_feature_names)):
                    def fname_transformer(x):
                        return self._instances[0].cate_feature_names(x)
            elif prefix == 'intercept_':
                inf_type = 'intercept'
            else:
                raise AttributeError("Unsupported inference: " + name)

            d_t = self._wrapped._d_t[0] if self._wrapped._d_t else 1
            if prefix == 'effect' or (isinstance(self._wrapped, LinearModelFinalCateEstimatorDiscreteMixin) and
                                      (inf_type == 'coefficient' or inf_type == 'intercept')):
                d_t = None
            d_y = self._wrapped._d_y[0] if self._wrapped._d_y else 1

            can_call = callable(getattr(self._instances[0], prefix))

            kind = self._bootstrap_type
            if kind == 'percentile' or kind == 'pivot':
                def get_dist(est, arr):
                    if kind == 'percentile':
                        return arr
                    elif kind == 'pivot':
                        return 2 * est - arr
                    else:
                        raise ValueError("Invalid kind, must be either 'percentile' or 'pivot'")

                def get_result():
                    return proxy(can_call, prefix,
                                 lambda arr, est: EmpiricalInferenceResults(
                                     d_t=d_t, d_y=d_y,
                                     pred=est, pred_dist=get_dist(est, arr),
                                     inf_type=inf_type,
                                     fname_transformer=fname_transformer,
                                     feature_names=self._wrapped.cate_feature_names(),
                                     output_names=self._wrapped.cate_output_names(),
                                     treatment_names=self._wrapped.cate_treatment_names()
                                 ))

                # Note that inference results are always methods even if the inference is for a property
                # (e.g. coef__inference() is a method but coef_ is a property)
                # Therefore we must insert a lambda if getting inference for a non-callable
                return get_result() if can_call else get_result

            else:
                assert kind == 'normal'

                def normal_inference(*args, **kwargs):
                    pred = getattr(self._wrapped, prefix)
                    if can_call:
                        pred = pred(*args, **kwargs)
                    stderr = getattr(self, prefix + '_std')
                    if can_call:
                        stderr = stderr(*args, **kwargs)
                    return NormalInferenceResults(
                        d_t=d_t, d_y=d_y, pred=pred,
                        pred_stderr=stderr, mean_pred_stderr=None, inf_type=inf_type,
                        fname_transformer=fname_transformer,
                        feature_names=self._wrapped.cate_feature_names(),
                        output_names=self._wrapped.cate_output_names(),
                        treatment_names=self._wrapped.cate_treatment_names())

                # If inference is for a property, create a fresh lambda to avoid passing args through
                return normal_inference if can_call else lambda: normal_inference()

        caught = None
        m = None
        if name.endswith("_interval"):
            m = get_interval
        elif name.endswith("_std"):
            m = get_std
        elif name.endswith("_inference"):
            m = get_inference

        # try to get interval/std first if appropriate,
        # since we don't prefer a wrapped method with this name
        if m is not None:
            try:
                return m()
            except AttributeError as err:
                caught = err
        if self._compute_means:
            return get_mean()

        raise (caught if caught else AttributeError(name))
