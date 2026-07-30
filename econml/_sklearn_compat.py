# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Centralized scikit-learn cross-version compatibility shims.

This module is the single place where EconML adapts to scikit-learn API
differences. When a new sklearn version introduces a breaking change that
affects EconML, the version gate and any associated helper belong here, not
scattered across individual modules.

EconML deliberately supports a wide range of sklearn versions (the exact
bounds live in ``pyproject.toml`` as the ``scikit-learn`` dependency
constraint; kept there as the single source of truth). Many of EconML's
users have their own non-trivial environments where pinning a recent
sklearn would conflict with unrelated dependencies, and EconML's CI matrix
relies on different sklearn releases naturally falling out of the
Python-version matrix to get coverage of multiple sklearn versions for
free. The cost of this is that wrappers under ``econml.sklearn_extensions``
and a handful of utilities need to branch on sklearn version; the benefit
is that bumping the floor is rare and the library stays installable in a
broad set of environments. The shims in this module are how we pay that
cost in one place.

Two kinds of things live here:

1. **Version constants** (``SKLEARN_GE_*``) — booleans computed once at
   import time from ``sklearn.__version__``. Use these instead of repeating
   ``parse(sklearn.__version__) >= parse("X.Y")`` at call sites.

2. **Compatibility helpers** — small wrappers that hide a sklearn API
   difference behind a stable EconML-facing interface (e.g.
   :func:`one_hot_encoder`, :func:`ensure_finite_kwargs`). Each helper
   documents the upstream change it papers over so contributors know when
   the shim can eventually be deleted.

When adding a new shim, prefer extending this module over adding another
local ``parse(sklearn.__version__)`` block. Each version gate should appear
exactly once in the codebase. See the project README's "Working with
scikit-learn version differences" section for a survey of the kinds of
sklearn changes EconML has had to absorb (with example call sites).

Recipe: wrapping a sklearn estimator across versions
----------------------------------------------------

EconML's ``sklearn_extensions`` package contains many wrappers around
sklearn estimators (``WeightedLassoCV``, ``DebiasedLasso``, ...). When
sklearn renames or removes a constructor argument across versions, the
recommended pattern is:

1. **Add a version flag here** for the boundary you care about (e.g.
   ``SKLEARN_GE_17``). Don't repeat the ``parse(sklearn.__version__)``
   check at the call site.

2. **In the wrapper's ``__init__``, branch the call to
   ``super().__init__`` on that flag** so each branch only passes the
   kwargs sklearn accepts on that version.

3. **In the newer-sklearn branch, reassign the deprecated arg onto
   ``self`` ONLY if the parent's ``fit`` code path does not check that
   attribute against the sentinel.** On newer sklearn versions, the parent
   ``__init__`` may write its own sentinel (e.g. ``"deprecated"`` or
   ``"warn"``) onto ``self`` for a deprecated arg you omitted. It is
   tempting to reassign ``self.<deprecated_arg> = <deprecated_arg>`` to
   preserve the user's value for ``get_params`` / ``clone``, but this
   often backfires: sklearn's ``fit`` implementation may check that same
   attribute against the sentinel to determine whether the user
   explicitly opted into the deprecated API and emit a ``FutureWarning``
   if not. Overwriting the sentinel with the user's value tricks
   sklearn's own check into firing on every fit. If the parent's fit
   does this, LEAVE THE SENTINEL and instead emit your OWN wrapper-level
   ``FutureWarning`` (see step 3.5). Users will see ``est.<deprecated_arg>``
   return the sentinel on affected sklearn versions, matching native
   sklearn behavior.

   Once a later sklearn removes the attribute entirely, however,
   ``BaseEstimator.get_params()`` still inspects the wrapper's backwards-
   compatible ``__init__`` signature and expects that attribute to exist.
   Detect that transition after ``super().__init__`` with ``hasattr``. If
   the parent no longer created the attribute, restore the same sentinel
   used by earlier sklearn versions. This keeps one simple invariant on
   all newer versions: the legacy attribute exists for BaseEstimator
   introspection but is filtered from the public params dict.

   **Do NOT reassign every constructor argument unconditionally.** In
   particular, if you translated ``deprecated_arg=X`` into
   ``renamed_arg=X`` on the ``super().__init__`` call, do NOT reassign
   ``self.renamed_arg = renamed_arg`` afterwards — the parent already
   stored the correctly-translated value there, and blindly reassigning
   the caller's default (usually ``None``) will clobber it. See PR #1031
   for the specific bug this rule is meant to prevent, and PR #1042 for
   the fix.

3.5. **When you accept a legacy alias for a deprecated parent arg, emit
   your OWN wrapper-level ``FutureWarning`` nudging users toward the
   modern name.** The wrapper still supports the legacy name for
   backwards compat, but users are silently opting into a code path that
   will eventually stop working (and, if step 3 applies, are also losing
   introspection because ``est.<deprecated_arg>`` returns the sentinel).
   Firing our own warning at the wrapper level surfaces the migration to
   users on their timeline rather than forcing them to discover it when
   the parent's next release removes the arg. Gate the warning on the
   same version flag you use for the ``super().__init__`` dispatch, and
   only fire it when the user explicitly passed a non-default value (so
   ``MyWrapper()`` with all defaults stays quiet). See
   ``_warn_n_alphas_deprecated`` and its call sites in
   ``econml/sklearn_extensions/linear_model.py`` for a concrete example.

4. **If the parent removed the arg entirely on a newer sklearn (not just
   deprecated it), add BOTH a ``get_params`` and a paired ``set_params``
   override.**

   - ``get_params``: sklearn's ``_get_param_names`` inspects the wrapper's
     ``__init__`` signature (which still lists the arg for backwards
     compat), so ``get_params()`` still returns it. That value leaks into
     sklearn's internal fit-time calls (e.g. ``lasso_path(**path_params)``
     inside ``LinearModelCV.fit``) and either warns or errors depending on
     the sklearn version. Override ``get_params`` to drop the arg on the
     affected sklearn versions.

   - ``set_params``: because sklearn's default ``set_params`` validates
     unknown keys against ``get_params()``, dropping the arg from
     ``get_params`` would also break ``set_params(<removed_arg>=...)`` --
     and by extension ``GridSearchCV`` / ``Pipeline`` parameter grids that
     name it -- even though ``__init__`` still accepts it. Add a paired
     ``set_params`` override that translates the legacy name to the
     current one (matching your ``__init__`` dispatch semantics).

   Pattern originally suggested in PR #1046.

5. **Test it** with
   ``econml.tests._sklearn_compat_helpers.assert_sklearn_roundtrip(cls,
   **kwargs)`` to assert user-passed kwargs survive ``get_params`` and
   ``clone``, and wrap the happy-path fit/predict in
   ``no_sklearn_future_warnings()`` to fail loudly if an upstream
   deprecation starts firing.

Skeleton (deprecated arg translated to a renamed target)::

    from econml._sklearn_compat import SKLEARN_GE_17

    _DEFAULT_N_ALPHAS = 100

    def _warn_n_alphas_deprecated(cls_name):
        warnings.warn(
            f"The 'n_alphas' parameter of {cls_name} is deprecated on "
            "scikit-learn >= 1.7; use 'alphas=<int>' instead.",
            FutureWarning, stacklevel=3,
        )

    class MyWrapper(SomeSklearnEstimator):
        def __init__(self, n_alphas=100, alphas=None, cv=None, ...):
            if SKLEARN_GE_17:
                # 'alphas' now accepts an int; translate 'n_alphas' into it.
                super().__init__(
                    alphas=alphas if alphas is not None else n_alphas,
                    cv=cv, ...)
                # Preserve sklearn 1.7-1.8's sentinel, and restore that same
                # sentinel after sklearn removes the attribute entirely.
                if not hasattr(self, 'n_alphas'):
                    self.n_alphas = 'deprecated'
                # Do NOT reassign self.alphas either — the parent already
                # stored the correctly-translated value.
                if n_alphas != _DEFAULT_N_ALPHAS:
                    _warn_n_alphas_deprecated(type(self).__name__)
            else:
                super().__init__(n_alphas=n_alphas, alphas=alphas, cv=cv, ...)

        def get_params(self, deep=True):
            # On sklearn 1.7+, drop 'n_alphas' from the params dict so it
            # doesn't leak into sklearn's internal path_params / lasso_path
            # calls and trigger a FutureWarning (1.7-1.10) or TypeError (1.11+).
            params = super().get_params(deep=deep)
            if SKLEARN_GE_17:
                params.pop('n_alphas', None)
            return params

        def set_params(self, **params):
            # Because get_params drops 'n_alphas' on sklearn 1.7+, sklearn's
            # default set_params (which validates unknown keys against
            # get_params) would reject set_params(n_alphas=...) and any
            # GridSearchCV / Pipeline parameter grid that names it. Preserve
            # the legacy contract by mapping 'n_alphas' back to the
            # equivalent 'alphas=<int>' on sklearn 1.7+. Keep the legacy
            # attribute as the sentinel on all affected versions.
            if SKLEARN_GE_17 and 'n_alphas' in params:
                _warn_n_alphas_deprecated(type(self).__name__)
                n_alphas_value = params.pop('n_alphas')
                if 'alphas' not in params:
                    params['alphas'] = n_alphas_value
            return super().set_params(**params)

And the matching test::

    from econml.tests._sklearn_compat_helpers import (
        assert_sklearn_roundtrip, no_sklearn_future_warnings,
    )

    def test_my_wrapper_roundtrips():
        # Don't probe the DEPRECATED kwarg (n_alphas) here — it's
        # intentionally filtered out of get_params on newer sklearns and
        # would fail the round-trip. Probe the other user-facing kwargs.
        assert_sklearn_roundtrip(MyWrapper, cv=3, ...)

    def test_my_wrapper_fit_no_deprecations():
        with no_sklearn_future_warnings():
            MyWrapper(cv=3).fit(X, y)

Forward-compatibility practices
-------------------------------

When you see a new sklearn ``FutureWarning`` or ``DeprecationWarning``
surface in CI, **treat it as a removal timer**, not just noise to silence.
Sklearn deprecation warnings include the removal version (e.g. *"deprecated
in 1.7 and will be removed in 1.11"*); open a follow-up issue with that
version stamp so the migration is scheduled rather than perpetually
deferred until it becomes a hard break.

Two habits that make deprecations show up early rather than at removal
time:

- Add ``no_sklearn_future_warnings()`` around the happy-path fit/predict
  in every new wrapper's test file. This surfaces a fresh sklearn
  deprecation as a red test on the sklearn version where it *begins*
  firing (e.g. 1.7 for ``n_alphas``) rather than only becoming visible
  on the sklearn version where the arg is actually removed (e.g. 1.9).

- When you migrate a wrapper for a deprecated arg, consider whether the
  arg has also disappeared from the parent's ``__init__`` signature on
  the newest supported sklearn (i.e. whether step 4 above applies now
  even if step 3 alone is enough today). Adding the ``get_params``
  override pre-emptively — before the parent actually removes the arg —
  shields us from the hard-break moment.
"""

from typing import Union

import sklearn
from packaging.version import parse
from sklearn.preprocessing import OneHotEncoder
# ``_get_column_indices`` and ``_print_elapsed_time`` moved to these private
# submodules in sklearn 1.5, which is below our supported floor, so the new
# locations are always correct. They are re-exported here under stable names.
from sklearn.utils._indexing import _get_column_indices as get_column_indices  # noqa: F401
from sklearn.utils._user_interface import _print_elapsed_time as print_elapsed_time  # noqa: F401

_SKLEARN_VERSION = parse(sklearn.__version__)

# Version flags. Add a new constant when (and only when) a sklearn release
# introduces a behavior or API change that EconML needs to branch on.
SKLEARN_GE_17 = _SKLEARN_VERSION >= parse("1.7")
SKLEARN_GE_18 = _SKLEARN_VERSION >= parse("1.8")


# ---------------------------------------------------------------------------
# Constructor / kwarg shims
# ---------------------------------------------------------------------------

def one_hot_encoder(sparse: bool = False, **kwargs) -> OneHotEncoder:
    """Construct a :class:`~sklearn.preprocessing.OneHotEncoder`.

    Handles the breaking rename of the ``sparse`` constructor argument to
    ``sparse_output`` between sklearn 1.1 and 1.2.
    """
    return OneHotEncoder(sparse_output=sparse, **kwargs)


def ensure_finite_kwargs(ensure_all_finite: Union[str, bool]) -> dict:
    """Return the kwargs dict that requests finite-value checking.

    ``force_all_finite`` was renamed to ``ensure_all_finite`` in sklearn 1.6
    and is scheduled to be removed in 1.8+. Splat the returned dict into
    sklearn ``check_array``/``check_X_y`` calls to stay version-agnostic.
    """
    return {"ensure_all_finite": ensure_all_finite}
