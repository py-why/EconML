# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Test helpers for verifying scikit-learn-wrapper invariants.

These exist to give wrappers around sklearn estimators (e.g. those in
:mod:`econml.sklearn_extensions`) a small, consistent vocabulary for the two
regression checks that PR-#1031-class bugs need:

1. :func:`assert_sklearn_roundtrip` — verifies that constructing an estimator
   with explicit kwargs and then calling :func:`sklearn.base.clone` preserves
   those kwargs end-to-end. The common failure mode this catches: a wrapper
   omits an arg from its ``super().__init__()`` call, and on newer sklearn
   versions the parent ``__init__`` writes a sentinel value (e.g.
   ``"deprecated"``) onto ``self`` instead. ``get_params`` then reports the
   sentinel rather than the user's value, and the clone constructed from
   those params silently diverges from what the user asked for (often
   triggering an obscure constraint-violation error at fit time).

2. :func:`no_sklearn_future_warnings` — a context manager that promotes
   sklearn-originated ``FutureWarning``/``DeprecationWarning`` to errors,
   so a wrapper's happy-path tests fail loudly when an upstream sklearn
   deprecation starts firing instead of silently emitting tens of thousands
   of warnings until someone notices.

Both helpers intentionally have a narrow, opinionated surface; they're meant
to be one-liners in tests, not a general framework.
"""

from __future__ import annotations

import contextlib
import inspect
import warnings
from typing import Any, Callable, Iterable, Optional

import numpy as np
from sklearn.base import clone


def _params_equal(a: Any, b: Any) -> bool:
    """Compare two ``get_params`` values for equality, handling arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(a, b)
        except (TypeError, ValueError):
            return False
    # Sklearn estimators must store constructor parameters without coercing
    # them. Treat scalar type changes (e.g. int -> np.int64) as real drift,
    # even when the values compare equal.
    if type(a) is not type(b):
        return False
    return a == b


def assert_sklearn_roundtrip(
    estimator_or_factory,
    /,
    **init_kwargs: Any,
):
    """Assert that constructor kwargs survive ``get_params`` and ``clone``.

    Two failure modes are checked:

    1. **Constructor-parameter drift.** After constructing the estimator
       with ``init_kwargs``, ``get_params`` must return those same values for
       the keys passed in. This catches the PR-#1031 class of bug where a
       wrapper's cross-version translation causes the parent and wrapper to
       disagree about the effective public parameter value. The correct fix
       depends on the upstream API: it may be a targeted reassignment, a
       renamed-parameter translation, or intentionally filtering a deprecated
       alias and testing only the stable replacement parameter.

    2. **Clone equivalence.** ``clone(estimator).get_params()`` must equal
       ``estimator.get_params()``. This catches cases where ``get_params`` is
       lossy with respect to the constructor.

    Parameters
    ----------
    estimator_or_factory : class, callable, or sklearn-compatible instance
        Disambiguation rule: if ``init_kwargs`` are passed, the first
        argument is treated as a class/factory and invoked as
        ``estimator_or_factory(**init_kwargs)``. If no ``init_kwargs`` are
        passed, it is treated as an already-built instance, and only check
        (2) runs because the user's intended kwargs are not recoverable.

        Any sklearn-compatible object (i.e. exposes ``get_params`` and
        cooperates with :func:`sklearn.base.clone`) is accepted; subclassing
        :class:`sklearn.base.BaseEstimator` is not required. Cooperation with
        ``clone`` includes accepting its standard ``get_params(deep=False)``
        call.
    **init_kwargs
        Kwargs to pass when constructing the estimator. These are what
        ``get_params`` is asserted to faithfully report.

    Returns
    -------
    object
        The constructed (or input) estimator, handy for chaining further
        assertions.

    Raises
    ------
    AssertionError
        With a per-parameter diff identifying which kwargs were not
        preserved.

    Examples
    --------
    >>> from econml.sklearn_extensions.linear_model import WeightedLassoCV
    >>> from econml.tests._sklearn_compat_helpers import assert_sklearn_roundtrip
    >>> est = assert_sklearn_roundtrip(WeightedLassoCV, cv=3, fit_intercept=False)
    """
    if init_kwargs:
        # factory form
        if not (inspect.isclass(estimator_or_factory) or callable(estimator_or_factory)):
            raise TypeError(
                "assert_sklearn_roundtrip with init_kwargs requires a class or "
                f"callable as the first argument, got {type(estimator_or_factory).__name__}."
            )
        estimator = estimator_or_factory(**init_kwargs)
        expected_kwargs: dict[str, Any] = dict(init_kwargs)
    else:
        # Instance form: do not auto-construct classes so callers are explicit
        # about which form they want.
        if inspect.isclass(estimator_or_factory):
            raise TypeError(
                "assert_sklearn_roundtrip without init_kwargs requires an "
                "already-built instance; instantiate the class first or pass "
                "the class with explicit constructor kwargs."
            )
        estimator = estimator_or_factory
        expected_kwargs = {}

    if not hasattr(estimator, "get_params"):
        raise TypeError(
            f"{type(estimator).__name__} has no get_params() method; it is not "
            f"sklearn-compatible."
        )
    original_params = estimator.get_params(deep=False)

    # 1. constructor-kwargs preservation (catches PR-#1031 sentinel overwrite)
    if expected_kwargs:
        missing_keys = sorted(set(expected_kwargs) - set(original_params))
        if missing_keys:
            raise AssertionError(
                f"{type(estimator).__name__}: get_params does not expose the "
                f"following constructor kwargs: {missing_keys}. Either the kwarg "
                f"is not accepted, or the wrapper is dropping it."
            )
        drifted = [
            (k, expected_kwargs[k], original_params[k])
            for k in sorted(expected_kwargs)
            if not _params_equal(expected_kwargs[k], original_params[k])
        ]
        if drifted:
            lines = [
                f"{type(estimator).__name__}: get_params does not reflect the "
                f"constructor kwargs (likely a sklearn-sentinel overwrite — see "
                f"econml._sklearn_compat for the standard fix-up pattern):",
            ]
            for name, passed, reported in drifted:
                lines.append(f"  - {name}: passed {passed!r}, get_params reports {reported!r}")
            raise AssertionError("\n".join(lines))

    # 2. clone equivalence
    cloned = clone(estimator)
    cloned_params = cloned.get_params(deep=False)

    if set(original_params) != set(cloned_params):
        added = sorted(set(cloned_params) - set(original_params))
        removed = sorted(set(original_params) - set(cloned_params))
        raise AssertionError(
            f"{type(estimator).__name__}: clone changed the parameter set "
            f"(added: {added}, removed: {removed})."
        )
    clone_drifted = [
        (k, original_params[k], cloned_params[k])
        for k in sorted(original_params)
        if not _params_equal(original_params[k], cloned_params[k])
    ]
    if clone_drifted:
        lines = [f"{type(estimator).__name__}: clone did not preserve params:"]
        for name, before, after in clone_drifted:
            lines.append(f"  - {name}: {before!r} -> {after!r}")
        raise AssertionError("\n".join(lines))

    return estimator


@contextlib.contextmanager
def no_sklearn_future_warnings(
    *,
    categories: Iterable[type[Warning]] = (FutureWarning, DeprecationWarning),
    extra_modules: Iterable[str] = (),
):
    """Promote sklearn-originated deprecation warnings to errors.

    Use this around the *happy path* of a wrapper test (i.e. the call that
    should not be emitting any sklearn deprecation messages with the current
    pinned sklearn). If an upstream deprecation starts firing, the test fails
    loudly with the exact warning message and source instead of silently
    contributing to a warning storm.

    Parameters
    ----------
    categories : iterable of warning classes, default ``(FutureWarning, DeprecationWarning)``
        Warning categories to promote to errors when originating from sklearn.
    extra_modules : iterable of str, default ``()``
        Additional regex fragments (matched against the warning's source
        module) to treat alongside ``sklearn``. Useful when an indirect
        sklearn deprecation surfaces through a closely related library
        (e.g. ``"imblearn"``).

    Examples
    --------
    >>> from econml.tests._sklearn_compat_helpers import no_sklearn_future_warnings
    >>> from econml.sklearn_extensions.linear_model import WeightedLassoCV
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(50, 3))
    >>> y = rng.normal(size=50)
    >>> with no_sklearn_future_warnings():
    ...     WeightedLassoCV(cv=3).fit(X, y)  # doctest: +SKIP
    """
    module_patterns = ("sklearn", *extra_modules)
    module_regex = r"^(?:" + "|".join(module_patterns) + r")(?:\..*)?$"

    with warnings.catch_warnings():
        # Start from a clean slate so prior filters don't mask what we promote.
        warnings.resetwarnings()
        for category in categories:
            warnings.filterwarnings("error", category=category, module=module_regex)
        yield


def assert_no_sklearn_future_warnings(
    callable_: Callable[..., Any],
    /,
    *args: Any,
    categories: Optional[Iterable[type[Warning]]] = None,
    extra_modules: Iterable[str] = (),
    **kwargs: Any,
) -> Any:
    """Call ``callable_(*args, **kwargs)`` under :func:`no_sklearn_future_warnings`.

    Convenience wrapper for one-shot assertions; equivalent to::

        with no_sklearn_future_warnings(...):
            result = callable_(*args, **kwargs)

    Returns the callable's return value so it can be chained into further
    assertions.
    """
    cm_kwargs: dict[str, Any] = {"extra_modules": extra_modules}
    if categories is not None:
        cm_kwargs["categories"] = categories
    with no_sklearn_future_warnings(**cm_kwargs):
        return callable_(*args, **kwargs)
