# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

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

    def __init__(self, model, total, limits, n_copies):
        self.model = model
        self.total = total
        self.limits = limits
        self.n_copies = n_copies

    def validate(self, y, skip_group_counts=False):
        (yvals, cts) = np.unique(y, return_counts=True)
        (llim, ulim) = self.limits
        # if we aren't fitting on the whole dataset, ensure that the limits are respected
        if (not skip_group_counts) and (not (llim <= len(yvals) <= ulim)):
            raise Exception(f"Grouping failed: received {len(yvals)} groups instead of {llim}-{ulim}")

        # ensure that the grouping has worked correctly and we get exactly the number of copies
        # of the items in whichever groups we see
        for (yval, ct) in zip(yvals, cts):
            if ct != self.n_copies[yval]:
                raise Exception(
                    f"Grouping failed; received {ct} copies of {yval} instead of {self.n_copies[yval]}")

    def fit(self, X, y):
        self.validate(y, len(y) == self.total)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class NestedModel(GroupingModel):
    """
    Class for testing nested grouping.

    The wrapped model must have a 'cv' attribute;
    this class exposes an identical 'cv' attribute, which is how nested CV is implemented in _fit_with_groups
    """

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
            self.validate(y[train], len(y) == self.total)
        self.model.fit(X, y)
        return self
