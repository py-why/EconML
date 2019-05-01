# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from econml.bootstrap import BootstrapEstimator
from econml.dml import DMLCateEstimator
from sklearn.linear_model import LinearRegression
import numpy as np
import unittest


class TestBootstrap(unittest.TestCase):

    def test_with_sklearn(self):
        """
        Test that we can bootstrap sklearn estimators
        """
        for n_jobs in [None, -1]:  # test parallelism
            x = np.random.normal(size=(1000, 1))
            y = x * 0.5 + np.random.normal(size=(1000, 1))
            y = y.flatten()

            est = LinearRegression()
            est.fit(x, y)

            bs = BootstrapEstimator(est, 50, n_jobs=n_jobs)
            # test that we can fit with the same arguments as the base estimator
            bs.fit(x, y)

            # test that we can get the same attribute for the bootstrap as the original, with the same shape
            self.assertEqual(np.shape(est.coef_), np.shape(bs.coef_))

            # test that we can get an interval for the same attribute for the bootstrap as the original,
            # with the same shape for the lower and upper bounds
            lower, upper = bs.coef__interval()
            for bound in [lower, upper]:
                self.assertEqual(np.shape(est.coef_), np.shape(bound))

            # test that we can do the same thing once we provide percentile bounds
            lower, upper = bs.coef__interval(lower=10, upper=90)
            for bound in [lower, upper]:
                self.assertEqual(np.shape(est.coef_), np.shape(bound))

            # test that we can do the same thing with the results of a method, rather than an attribute
            self.assertEqual(np.shape(est.predict(x)), np.shape(bs.predict(x)))

            # test that we can get an interval for the same attribute for the bootstrap as the original,
            # with the same shape for the lower and upper bounds
            lower, upper = bs.predict_interval(x)
            for bound in [lower, upper]:
                self.assertEqual(np.shape(est.predict(x)), np.shape(bound))

            # test that we can do the same thing once we provide percentile bounds
            lower, upper = bs.predict_interval(x, lower=10, upper=90)
            for bound in [lower, upper]:
                self.assertEqual(np.shape(est.predict(x)), np.shape(bound))

    def test_with_econml(self):
        """
        Test that we can bootstrap econml estimators
        """
        x = np.random.normal(size=(1000, 2))
        t = np.random.normal(size=(1000, 1))
        t2 = np.random.normal(size=(1000, 1))
        y = x[:, 0] * 0.5 + t + np.random.normal(size=(1000, 1))

        est = DMLCateEstimator(LinearRegression(), LinearRegression())
        est.fit(y, t, x)

        bs = BootstrapEstimator(est, 50)
        # test that we can fit with the same arguments as the base estimator
        bs.fit(y, t, x)

        # test that we can get the same attribute for the bootstrap as the original, with the same shape
        self.assertEqual(np.shape(est.coef_), np.shape(bs.coef_))

        # test that we can get an interval for the same attribute for the bootstrap as the original,
        # with the same shape for the lower and upper bounds
        lower, upper = bs.coef__interval()
        for bound in [lower, upper]:
            self.assertEqual(np.shape(est.coef_), np.shape(bound))

        # test that we can do the same thing once we provide percentile bounds
        lower, upper = bs.coef__interval(lower=10, upper=90)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(est.coef_), np.shape(bound))

        # test that we can do the same thing with the results of a method, rather than an attribute
        self.assertEqual(np.shape(est.effect(x, t, t2)), np.shape(bs.effect(x, t, t2)))

        # test that we can get an interval for the same attribute for the bootstrap as the original,
        # with the same shape for the lower and upper bounds
        lower, upper = bs.effect_interval(x, t, t2)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(est.effect(x, t, t2)), np.shape(bound))

        # test that we can do the same thing once we provide percentile bounds
        lower, upper = bs.effect_interval(x, t, t2, lower=10, upper=90)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(est.effect(x, t, t2)), np.shape(bound))
