# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Lazy module loading to avoid expensive imports at package load time."""

import importlib


class _LazyModule:
    """Proxy that delays importing a module until an attribute is accessed.

    Use at module level as a drop-in replacement for ``import heavy_lib``::

        heavy_lib = _LazyModule("heavy_lib")

    The real module is imported on first attribute access, so the cost is
    deferred until the functionality is actually needed.
    """

    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self._module_name)
        return self._module

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __repr__(self):
        if self._module is not None:
            return repr(self._module)
        return f"<_LazyModule '{self._module_name}' (not yet loaded)>"
