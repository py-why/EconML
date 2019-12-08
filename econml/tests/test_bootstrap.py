# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from econml.bootstrap import BootstrapEstimator
from econml.inference import BootstrapInference
from econml.dml import LinearDMLCateEstimator
from econml.two_stage_least_squares import NonparametricTwoStageLeastSquares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import unittest
import joblib


class TestBootstrap(unittest.TestCase):

    def test_with_sklearn(self):
        """Test that we can bootstrap sklearn estimators."""
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

            # test that the lower and upper bounds differ
            assert (lower <= upper).all()
            assert (lower < upper).any()

            # test that we can do the same thing once we provide percentile bounds
            lower, upper = bs.coef__interval(lower=10, upper=90)
            for bound in [lower, upper]:
                self.assertEqual(np.shape(est.coef_), np.shape(bound))

            # test that the lower and upper bounds differ
            assert (lower <= upper).all()
            assert (lower < upper).any()

            # test that we can do the same thing with the results of a method, rather than an attribute
            self.assertEqual(np.shape(est.predict(x)), np.shape(bs.predict(x)))

            # test that we can get an interval for the same attribute for the bootstrap as the original,
            # with the same shape for the lower and upper bounds
            lower, upper = bs.predict_interval(x)
            for bound in [lower, upper]:
                self.assertEqual(np.shape(est.predict(x)), np.shape(bound))

            # test that the lower and upper bounds differ
            assert (lower <= upper).all()
            assert (lower < upper).any()

            # test that we can do the same thing once we provide percentile bounds
            lower, upper = bs.predict_interval(x, lower=10, upper=90)
            for bound in [lower, upper]:
                self.assertEqual(np.shape(est.predict(x)), np.shape(bound))

            # test that the lower and upper bounds differ
            assert (lower <= upper).all()
            assert (lower < upper).any()

    def test_with_econml(self):
        """Test that we can bootstrap econml estimators."""
        x = np.random.normal(size=(1000, 2))
        t = np.random.normal(size=(1000, 1))
        t2 = np.random.normal(size=(1000, 1))
        y = x[:, 0:1] * 0.5 + t + np.random.normal(size=(1000, 1))

        est = LinearDMLCateEstimator(LinearRegression(), LinearRegression())
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

        # test that the lower and upper bounds differ
        assert (lower <= upper).all()
        assert (lower < upper).any()

        # test that we can do the same thing once we provide percentile bounds
        lower, upper = bs.coef__interval(lower=10, upper=90)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(est.coef_), np.shape(bound))

        # test that we can do the same thing with the results of a method, rather than an attribute
        self.assertEqual(np.shape(est.effect(x, T0=t, T1=t2)), np.shape(bs.effect(x, T0=t, T1=t2)))

        # test that we can get an interval for the same attribute for the bootstrap as the original,
        # with the same shape for the lower and upper bounds
        lower, upper = bs.effect_interval(x, T0=t, T1=t2)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(est.effect(x, T0=t, T1=t2)), np.shape(bound))

        # test that the lower and upper bounds differ
        assert (lower <= upper).all()
        assert (lower < upper).any()

        # test that we can do the same thing once we provide percentile bounds
        lower, upper = bs.effect_interval(x, T0=t, T1=t2, lower=10, upper=90)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(est.effect(x, T0=t, T1=t2)), np.shape(bound))

        # test that the lower and upper bounds differ
        assert (lower <= upper).all()
        assert (lower < upper).any()

    def test_backends(self):
        """Test that we can use threading or multiprocess backends."""
        for backend in ['threading', 'loky']:
            with joblib.parallel_backend(backend):
                x = np.random.normal(size=(1000, 1))
                y = x * 0.5 + np.random.normal(size=(1000, 1))
                y = y.flatten()

                est = LinearRegression()
                est.fit(x, y)

                bs = BootstrapEstimator(est, 50, n_jobs=2)
                # test that we can fit with the same arguments as the base estimator

                bs.fit(x, y)

                # test that we can get the same attribute for the bootstrap as the original, with the same shape
                self.assertEqual(np.shape(est.coef_), np.shape(bs.coef_))

                # test that we can get an interval for the same attribute for the bootstrap as the original,
                # with the same shape for the lower and upper bounds
                lower, upper = bs.coef__interval()
                for bound in [lower, upper]:
                    self.assertEqual(np.shape(est.coef_), np.shape(bound))

                # test that the lower and upper bounds differ
                assert (lower <= upper).all()
                assert (lower < upper).any()

                # test that we can do the same thing once we provide percentile bounds
                lower, upper = bs.coef__interval(lower=10, upper=90)
                for bound in [lower, upper]:
                    self.assertEqual(np.shape(est.coef_), np.shape(bound))

                # test that the lower and upper bounds differ
                assert (lower <= upper).all()
                assert (lower < upper).any()

                # test that we can do the same thing with the results of a method, rather than an attribute
                self.assertEqual(np.shape(est.predict(x)), np.shape(bs.predict(x)))

                # test that we can get an interval for the same attribute for the bootstrap as the original,
                # with the same shape for the lower and upper bounds
                lower, upper = bs.predict_interval(x)
                for bound in [lower, upper]:
                    self.assertEqual(np.shape(est.predict(x)), np.shape(bound))

                # test that the lower and upper bounds differ
                assert (lower <= upper).all()
                assert (lower < upper).any()

                # test that we can do the same thing once we provide percentile bounds
                lower, upper = bs.predict_interval(x, lower=10, upper=90)
                for bound in [lower, upper]:
                    self.assertEqual(np.shape(est.predict(x)), np.shape(bound))

                # test that the lower and upper bounds differ
                assert (lower <= upper).all()
                assert (lower < upper).any()

    def test_internal(self):
        """Test that the internal use of bootstrap within an estimator works."""
        x = np.random.normal(size=(1000, 2))
        t = np.random.normal(size=(1000, 1))
        t2 = np.random.normal(size=(1000, 1))
        y = x[:, 0:1] * 0.5 + t + np.random.normal(size=(1000, 1))

        est = LinearDMLCateEstimator(LinearRegression(), LinearRegression())
        est.fit(y, t, x, inference='bootstrap')

        # test that we can get an interval for the same attribute for the bootstrap as the original,
        # with the same shape for the lower and upper bounds
        eff = est.effect(x, T0=t, T1=t2)

        lower, upper = est.effect_interval(x, T0=t, T1=t2)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(eff), np.shape(bound))

        # test that the lower and upper bounds differ
        assert (lower <= upper).all()
        assert (lower < upper).any()

        # test that the estimated effect is usually within the bounds
        assert np.mean(np.logical_and(lower <= eff, eff <= upper)) >= 0.9

        # test that we can do the same thing once we provide alpha explicitly
        lower, upper = est.effect_interval(x, T0=t, T1=t2, alpha=0.2)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(eff), np.shape(bound))

        # test that the lower and upper bounds differ
        assert (lower <= upper).all()
        assert (lower < upper).any()

        # test that the estimated effect is usually within the bounds
        assert np.mean(np.logical_and(lower <= eff, eff <= upper)) >= 0.8

    def test_internal_options(self):
        """Test that the internal use of bootstrap within an estimator using custom options works."""
        x = np.random.normal(size=(1000, 2))
        z = np.random.normal(size=(1000, 1))
        t = np.random.normal(size=(1000, 1))
        t2 = np.random.normal(size=(1000, 1))
        y = x[:, 0:1] * 0.5 + t + np.random.normal(size=(1000, 1))

        opts = BootstrapInference(50, 2)

        est = NonparametricTwoStageLeastSquares(PolynomialFeatures(2),
                                                PolynomialFeatures(2),
                                                PolynomialFeatures(2),
                                                None)
        est.fit(y, t, x, None, z, inference=opts)

        # test that we can get an interval for the same attribute for the bootstrap as the original,
        # with the same shape for the lower and upper bounds
        eff = est.effect(x, T0=t, T1=t2)

        lower, upper = est.effect_interval(x, T0=t, T1=t2)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(eff), np.shape(bound))

        # test that the lower and upper bounds differ
        assert (lower <= upper).all()
        assert (lower < upper).any()

        # TODO: test that the estimated effect is usually within the bounds
        #       and that the true effect is also usually within the bounds

        # test that we can do the same thing once we provide percentile bounds
        lower, upper = est.effect_interval(x, T0=t, T1=t2, alpha=0.2)
        for bound in [lower, upper]:
            self.assertEqual(np.shape(eff), np.shape(bound))

        # test that the lower and upper bounds differ
        assert (lower <= upper).all()
        assert (lower < upper).any()

        # TODO: test that the estimated effect is usually within the bounds
        #       and that the true effect is also usually within the bounds
