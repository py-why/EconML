# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all Policy estimators."""

import abc
import numpy as np


class PolicyLearner(metaclass=abc.ABCMeta):

    def fit(self, Y, T, *, X=None, **kwargs):
        pass

    def predict_value(self, X):
        pass

    def predict(self, X):
        pass
