# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bootstrap sampling."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone


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

    n_bootstrap_samples : int
        How many draws to perform.

    n_jobs: int, default: None
        The maximum number of concurrently running jobs, as in joblib.Parallel.

    compute_means : bool, default: True
        Whether to pass calls through to the underlying collection and return the mean.  Setting this
        to ``False`` can avoid ambiguities if the wrapped object itself has method names with an `_interval` suffix.

    prefer_wrapped: bool, default: False
        In case a method ending in '_interval' exists on the wrapped object, whether
        that should be preferred (meaning this wrapper will compute the mean of it).
        This option only affects behavior if `compute_means` is set to ``True``.
    """

    def __init__(self, wrapped, n_bootstrap_samples=1000, n_jobs=None, compute_means=True, prefer_wrapped=False):
        self._instances = [clone(wrapped, safe=False) for _ in range(n_bootstrap_samples)]
        self._n_bootstrap_samples = n_bootstrap_samples
        self._n_jobs = n_jobs
        self._compute_means = compute_means
        self._prefer_wrapped = prefer_wrapped

    # TODO: Add a __dir__ implementation?

    def fit(self, *args, **named_args):
        """
        Fit the model.

        The full signature of this method is the same as that of the wrapped object's `fit` method.
        """
        n_samples = np.shape(args[0] if args else named_args[(*named_args,)[0]])[0]
        indices = np.random.choice(n_samples, size=(self._n_bootstrap_samples, n_samples), replace=True)

        def fit(x, *args, **kwargs):
            x.fit(*args, **kwargs)
            return x  # Explicitly return x in case fit fails to return its target

        def convertArg(arg, inds):
            return arg[inds] if arg is not None else None
        self._instances = Parallel(n_jobs=self._n_jobs, prefer='threads', verbose=3)(
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
        def proxy(make_call, name, summary):
            def summarize_with(f):
                return summary(np.array(Parallel(n_jobs=self._n_jobs, prefer='threads', verbose=3)(
                    (f, (obj, name), {}) for obj in self._instances)))
            if make_call:
                def call(*args, **kwargs):
                    return summarize_with(lambda obj, name: getattr(obj, name)(*args, **kwargs))
                return call
            else:
                return summarize_with(lambda obj, name: getattr(obj, name))

        def get_mean():
            # for attributes that exist on the wrapped object, just compute the mean of the wrapped calls
            return proxy(callable(getattr(self._instances[0], name)), name, lambda arr: np.mean(arr, axis=0))

        def get_interval():
            # if the attribute exists on the wrapped object once we remove the suffix,
            # then we should be computing a confidence interval for the wrapped calls
            prefix = name[: - len("_interval")]

            def call_with_bounds(can_call, lower, upper):
                return proxy(can_call, prefix,
                             lambda arr: (np.percentile(arr, lower, axis=0), np.percentile(arr, upper, axis=0)))

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

        caught = None
        if self._compute_means and self._prefer_wrapped:
            try:
                return get_mean()
            except AttributeError as err:
                caught = err
            if name.endswith("_interval"):
                return get_interval()
        else:
            # try to get interval first if appropriate, since we don't prefer a wrapped method with this name
            if name.endswith("_interval"):
                try:
                    return get_interval()
                except AttributeError as err:
                    caught = err
            if self._compute_means:
                return get_mean()

        raise (caught if caught else AttributeError(name))
