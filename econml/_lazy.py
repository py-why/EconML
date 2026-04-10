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
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_module", None)

    def _load(self):
        module = object.__getattribute__(self, "_module")
        if module is None:
            name = object.__getattribute__(self, "_module_name")
            module = importlib.import_module(name)
            object.__setattr__(self, "_module", module)
        return module

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __repr__(self):
        module = object.__getattribute__(self, "_module")
        if module is not None:
            return repr(module)
        name = object.__getattribute__(self, "_module_name")
        return f"<_LazyModule '{name}' (not yet loaded)>"
