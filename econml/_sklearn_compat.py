# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Centralized scikit-learn cross-version compatibility shims.

This module is the single place where EconML adapts to scikit-learn API
differences. When a new sklearn version introduces a breaking change that
affects EconML, the version gate and any associated helper belong here, not
scattered across individual modules.

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
exactly once in the codebase.
"""

from typing import Union

import sklearn
from packaging.version import parse
from sklearn.preprocessing import OneHotEncoder

_SKLEARN_VERSION = parse(sklearn.__version__)

# Version flags. Add a new constant when (and only when) a sklearn release
# introduces a behavior or API change that EconML needs to branch on.
SKLEARN_GE_12 = _SKLEARN_VERSION >= parse("1.2")
SKLEARN_GE_14 = _SKLEARN_VERSION >= parse("1.4")
SKLEARN_GE_15 = _SKLEARN_VERSION >= parse("1.5")
SKLEARN_GE_16 = _SKLEARN_VERSION >= parse("1.6")
SKLEARN_GE_17 = _SKLEARN_VERSION >= parse("1.7")
SKLEARN_GE_18 = _SKLEARN_VERSION >= parse("1.8")


# ---------------------------------------------------------------------------
# Symbol relocations
# ---------------------------------------------------------------------------

# ``_get_column_indices`` moved from ``sklearn.utils`` to
# ``sklearn.utils._indexing`` in sklearn 1.5.
if SKLEARN_GE_15:
    from sklearn.utils._indexing import _get_column_indices as get_column_indices
else:
    from sklearn.utils import _get_column_indices as get_column_indices  # noqa: F401

# ``_print_elapsed_time`` moved from ``sklearn.utils`` to
# ``sklearn.utils._user_interface`` in sklearn 1.5.
if SKLEARN_GE_15:
    from sklearn.utils._user_interface import _print_elapsed_time as print_elapsed_time
else:
    from sklearn.utils import _print_elapsed_time as print_elapsed_time  # noqa: F401


# ---------------------------------------------------------------------------
# Constructor / kwarg shims
# ---------------------------------------------------------------------------

def one_hot_encoder(sparse: bool = False, **kwargs) -> OneHotEncoder:
    """Construct a :class:`~sklearn.preprocessing.OneHotEncoder`.

    Handles the breaking rename of the ``sparse`` constructor argument to
    ``sparse_output`` between sklearn 1.1 and 1.2.
    """
    if SKLEARN_GE_12:
        return OneHotEncoder(sparse_output=sparse, **kwargs)
    return OneHotEncoder(sparse=sparse, **kwargs)


def ensure_finite_kwargs(ensure_all_finite: Union[str, bool]) -> dict:
    """Return the kwargs dict that requests finite-value checking.

    ``force_all_finite`` was renamed to ``ensure_all_finite`` in sklearn 1.6
    and is scheduled to be removed in 1.8+. Splat the returned dict into
    sklearn ``check_array``/``check_X_y`` calls to stay version-agnostic.
    """
    if SKLEARN_GE_16:
        return {"ensure_all_finite": ensure_all_finite}
    return {"force_all_finite": ensure_all_finite}
