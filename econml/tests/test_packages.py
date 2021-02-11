# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests package installation"""

import unittest
import numpy as np
import warnings
import pytest


class TestPackages(unittest.TestCase):
    @pytest.mark.shap
    def test_shap(self):
        import shap
        from sklearn.ensemble import RandomForestRegressor
        shap.TreeExplainer(RandomForestRegressor(max_depth=4, n_estimators=10).fit(
            np.random.normal(size=(30, 6)), np.random.normal(size=(30,))))
        pass

    @pytest.mark.econml
    def test_econml(self):
        import econml
        pass

    @pytest.mark.dowhy
    def test_dowhy(self):
        import dowhy
        pass
