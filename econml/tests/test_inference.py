# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
from econml.dml import LinearDMLCateEstimator


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        # DGP constants
        cls.n = 1000
        cls.d_w = 30
        cls.d_x = 5
        # Generate data
        cls.X = np.random.uniform(0, 1, size=(cls.n, cls.d_x))
        cls.W = np.random.normal(0, 1, size=(cls.n, cls.d_w))
        cls.T = np.random.normal(0, 1, size=(cls.n, ))
        cls.Y = np.random.normal(0, 1, size=(cls.n, ))

    def test_inference_results(self):
        """Tests the inference results summary."""
        # Test inference results when `cate_feature_names` doesn not exist
        cate_est = LinearDMLCateEstimator(
            featurizer=PolynomialFeatures(degree=1,
                                          include_bias=False)
        )
        wrapped_est = self._NoFeatNamesEst(cate_est)
        wrapped_est.fit(
            TestInference.Y,
            TestInference.T,
            TestInference.X,
            TestInference.W,
            inference='statsmodels'
        )
        summary_results = wrapped_est.summary()
        coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
        np.testing.assert_array_equal(coef_rows, ['X{}'.format(i) for i in range(TestInference.d_x)])

    class _NoFeatNamesEst:
        def __init__(self, cate_est):
            self.cate_est = clone(cate_est, safe=False)

        def __getattr__(self, name):
            if name != 'cate_feature_names':
                return getattr(self.cate_est, name)
            else:
                return self.__getattribute__(name)
