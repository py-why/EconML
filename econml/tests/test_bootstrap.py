# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from econml.inference._bootstrap import BootstrapEstimator
from econml.inference import BootstrapInference
from econml.dml import LinearDML
from econml.iv.dr import LinearIntentToTreatDRIV
from econml.iv.sieve import SieveTSLS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import unittest
import joblib


class TestBootstrap(unittest.TestCase):

    def test_with_sklearn(self):
        """Test that we can bootstrap sklearn estimators."""
        for n_jobs in [None, -1]:  # test parallelism
            for kind in ['percentile', 'pivot', 'normal']:  # test both percentile and pivot intervals
                x = np.random.normal(size=(1000, 1))
                y = x * 0.5 + np.random.normal(size=(1000, 1))
                y = y.flatten()

                est = LinearRegression()
                est.fit(x, y)

                bs = BootstrapEstimator(est, 50, n_jobs=n_jobs, bootstrap_type=kind)
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

        est = LinearDML(model_y=LinearRegression(), model_t=LinearRegression())
        est.fit(y, t, X=x)

        bs = BootstrapEstimator(est, 50)
        # test that we can fit with the same arguments as the base estimator
        bs.fit(y, t, X=x)

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

        est = LinearDML(model_y=LinearRegression(), model_t=LinearRegression())
        est.fit(y, t, X=x, inference='bootstrap')

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

        est = SieveTSLS(t_featurizer=PolynomialFeatures(2),
                        x_featurizer=PolynomialFeatures(2),
                        z_featurizer=PolynomialFeatures(2),
                        dt_featurizer=None)
        est.fit(y, t, X=x, W=None, Z=z, inference=opts)

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

    def test_stratify(self):
        """Test that we can properly stratify by treatment"""
        T = [1, 0, 1, 2, 0, 2]
        Y = [1, 2, 3, 4, 5, 6]
        X = np.array([1, 1, 2, 2, 1, 2]).reshape(-1, 1)
        est = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(), discrete_treatment=True)
        inference = BootstrapInference(n_bootstrap_samples=5, n_jobs=-1, verbose=0)
        est.fit(Y, T, inference=inference)
        est.const_marginal_effect_interval()

        est.fit(Y, T, X=X, inference=inference)
        est.const_marginal_effect_interval(X)

        est.fit(Y, np.asarray(T).reshape(-1, 1), inference=inference)  # test stratifying 2D treatment
        est.const_marginal_effect_interval()

    def test_stratify_orthoiv(self):
        """Test that we can properly stratify by treatment/instrument pair"""
        T = [1, 0, 1, 1, 0, 0, 1, 0]
        Z = [1, 0, 0, 1, 0, 1, 0, 1]
        Y = [1, 2, 3, 4, 5, 6, 7, 8]
        X = np.array([1, 1, 2, 2, 1, 2, 1, 2]).reshape(-1, 1)
        est = LinearIntentToTreatDRIV(model_Y_X=LinearRegression(), model_T_XZ=LogisticRegression(),
                                      flexible_model_effect=LinearRegression(), cv=2)
        inference = BootstrapInference(n_bootstrap_samples=20, n_jobs=-1, verbose=3)
        est.fit(Y, T, Z=Z, X=X, inference=inference)
        est.const_marginal_effect_interval(X)

    def test_all_kinds(self):
        T = [1, 0, 1, 2, 0, 2] * 5
        Y = [1, 2, 3, 4, 5, 6] * 5
        X = np.array([1, 1, 2, 2, 1, 2] * 5).reshape(-1, 1)
        est = LinearDML(cv=2)
        for kind in ['percentile', 'pivot', 'normal']:
            with self.subTest(kind=kind):
                inference = BootstrapInference(n_bootstrap_samples=5, n_jobs=-1, verbose=0, bootstrap_type=kind)
                est.fit(Y, T, inference=inference)
                i = est.const_marginal_effect_interval()
                inf = est.const_marginal_effect_inference()
                assert i[0].shape == i[1].shape == inf.point_estimate.shape
                assert np.allclose(i[0], inf.conf_int()[0])
                assert np.allclose(i[1], inf.conf_int()[1])

                est.fit(Y, T, X=X, inference=inference)
                i = est.const_marginal_effect_interval(X)
                inf = est.const_marginal_effect_inference(X)
                assert i[0].shape == i[1].shape == inf.point_estimate.shape
                assert np.allclose(i[0], inf.conf_int()[0])
                assert np.allclose(i[1], inf.conf_int()[1])

                i = est.coef__interval()
                inf = est.coef__inference()
                assert i[0].shape == i[1].shape == inf.point_estimate.shape
                assert np.allclose(i[0], inf.conf_int()[0])
                assert np.allclose(i[1], inf.conf_int()[1])

                i = est.effect_interval(X)
                inf = est.effect_inference(X)
                assert i[0].shape == i[1].shape == inf.point_estimate.shape
                assert np.allclose(i[0], inf.conf_int()[0])
                assert np.allclose(i[1], inf.conf_int()[1])
