# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import sys
import warnings

# HACK: work around bug in assertWarns (https://bugs.python.org/issue29620)
# this can be removed if the corresponding pull request (https://github.com/python/cpython/pull/4800) is ever merged


def _enter(self):
    # The __warningregistry__'s need to be in a pristine state for tests
    # to work properly.
    for v in list(sys.modules.values()):
        if getattr(v, '__warningregistry__', None):
            v.__warningregistry__ = {}
    self.warnings_manager = warnings.catch_warnings(record=True)
    self.warnings = self.warnings_manager.__enter__()
    warnings.simplefilter("always", self.expected)
    return self


unittest.case._AssertWarnsContext.__enter__ = _enter
