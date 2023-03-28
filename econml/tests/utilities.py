# Copyright (c) PyWhy contributors. All rights reserved.
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

import numpy as np
from sklearn.calibration import check_cv


class GroupingModel:
    """
    Class for validating that grouping has been done correctly.
    Checks that the number of distinct y values is within a given range, and that each y value
    occurs some known number of times.

    For example, if the groups each have the same size, and the target is identical to the group, then the number
    of distinct y values should be in the range [n_groups-ceil(n_groups/cv), n_groups-floor(n_groups/cv)],
    and the number of copies of each y value should be equal to the group size
    """

    def __init__(self, model, limits, n_copies):
        self.model = model
        self.limits = limits
        self.n_copies = n_copies

    def validate(self, y):
        (yvals, cts) = np.unique(y, return_counts=True)
        (llim, ulim) = self.limits
        if not (llim <= len(yvals) <= ulim):
            raise Exception(f"Grouping failed: received {len(yvals)} groups instead of {llim}-{ulim}")

        # ensure that the grouping has worked correctly and we get exactly the number of copies
        # of the items in whichever groups we see
        for (yval, ct) in zip(yvals, cts):
            if ct != self.n_copies[yval]:
                raise Exception(
                    f"Grouping failed; received {ct} copies of {yval} instead of {self.n_copies[yval]}")

    def fit(self, X, y):
        self.validate(y)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class NestedModel(GroupingModel):
    """
    Class for testing nested grouping. The wrapped model must have a 'cv' attribute;
    this class exposes an identical 'cv' attribute, which is how nested CV is implemented in fit_with_groups
    """

    def __init__(self, model, limits, n_copies):
        super().__init__(model, limits, n_copies)

    # DML nested CV works via a 'cv' attribute
    @property
    def cv(self):
        return self.model.cv

    @cv.setter
    def cv(self, value):
        self.model.cv = value

    def fit(self, X, y):
        for (train, test) in check_cv(self.cv, y).split(X, y):
            # want to validate the nested grouping, not the outer grouping in the nesting tests
            self.validate(y[train])
        self.model.fit(X, y)
        return self
