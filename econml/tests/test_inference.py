# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
from sklearn.base import clone
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.dml import LinearDML
from econml.drlearner import LinearDRLearner
from econml.inference import (BootstrapInference, NormalInferenceResults,
                              EmpiricalInferenceResults, PopulationSummaryResults)


class TestInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        np.random.seed(123)
        # DGP constants
        cls.n = 1000
        cls.d_w = 3
        cls.d_x = 3
        # Generate data
        cls.X = np.random.uniform(0, 1, size=(cls.n, cls.d_x))
        cls.W = np.random.normal(0, 1, size=(cls.n, cls.d_w))
        cls.T = np.random.binomial(1, .5, size=(cls.n,))
        cls.Y = np.random.normal(0, 1, size=(cls.n, ))

    def test_summary(self):
        """Tests the inference results summary for continuous treatment estimators."""
        # Test inference results when `cate_feature_names` doesn not exist

        for inference in [BootstrapInference(n_bootstrap_samples=5), 'auto']:
            cate_est = LinearDML(model_t=LinearRegression(), model_y=LinearRegression(),
                                 featurizer=PolynomialFeatures(degree=2,
                                                               include_bias=False)
                                 )
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            summary_results = cate_est.summary()
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            fnames = PolynomialFeatures(degree=2, include_bias=False).fit(TestInference.X).get_feature_names()
            np.testing.assert_array_equal(coef_rows, fnames)
            intercept_rows = np.asarray(summary_results.tables[1].data)[1:, 0]
            np.testing.assert_array_equal(intercept_rows, ['cate_intercept'])

            cate_est = LinearDML(model_t=LinearRegression(), model_y=LinearRegression(),
                                 featurizer=PolynomialFeatures(degree=2,
                                                               include_bias=False)
                                 )
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            fnames = ['Q' + str(i) for i in range(TestInference.d_x)]
            summary_results = cate_est.summary(feat_name=fnames)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            fnames = PolynomialFeatures(degree=2, include_bias=False).fit(
                TestInference.X).get_feature_names(input_features=fnames)
            np.testing.assert_array_equal(coef_rows, fnames)
            cate_est = LinearDML(model_t=LinearRegression(), model_y=LinearRegression(), featurizer=None)
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            summary_results = cate_est.summary()
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, ['X' + str(i) for i in range(TestInference.d_x)])

            cate_est = LinearDML(model_t=LinearRegression(), model_y=LinearRegression(), featurizer=None)
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            fnames = ['Q' + str(i) for i in range(TestInference.d_x)]
            summary_results = cate_est.summary(feat_name=fnames)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, fnames)

            cate_est = LinearDML(model_t=LinearRegression(), model_y=LinearRegression(), featurizer=None)
            wrapped_est = self._NoFeatNamesEst(cate_est)
            wrapped_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            summary_results = wrapped_est.summary()
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, ['X' + str(i) for i in range(TestInference.d_x)])

            cate_est = LinearDML(model_t=LinearRegression(), model_y=LinearRegression(), featurizer=None)
            wrapped_est = self._NoFeatNamesEst(cate_est)
            wrapped_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            fnames = ['Q' + str(i) for i in range(TestInference.d_x)]
            summary_results = wrapped_est.summary(feat_name=fnames)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, fnames)

    def test_summary_discrete(self):
        """Tests the inference results summary for discrete treatment estimators."""
        # Test inference results when `cate_feature_names` doesn not exist

        for inference in [BootstrapInference(n_bootstrap_samples=5), 'auto']:
            cate_est = LinearDRLearner(model_regression=LinearRegression(), model_propensity=LogisticRegression(),
                                       featurizer=PolynomialFeatures(degree=2,
                                                                     include_bias=False)
                                       )
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            summary_results = cate_est.summary(T=1)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            fnames = PolynomialFeatures(degree=2, include_bias=False).fit(TestInference.X).get_feature_names()
            np.testing.assert_array_equal(coef_rows, fnames)
            intercept_rows = np.asarray(summary_results.tables[1].data)[1:, 0]
            np.testing.assert_array_equal(intercept_rows, ['cate_intercept'])

            cate_est = LinearDRLearner(model_regression=LinearRegression(),
                                       model_propensity=LogisticRegression(),
                                       featurizer=PolynomialFeatures(degree=2,
                                                                     include_bias=False)
                                       )
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            fnames = ['Q' + str(i) for i in range(TestInference.d_x)]
            summary_results = cate_est.summary(T=1, feat_name=fnames)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            fnames = PolynomialFeatures(degree=2, include_bias=False).fit(
                TestInference.X).get_feature_names(input_features=fnames)
            np.testing.assert_array_equal(coef_rows, fnames)
            cate_est = LinearDRLearner(model_regression=LinearRegression(),
                                       model_propensity=LogisticRegression(), featurizer=None)
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            summary_results = cate_est.summary(T=1)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, ['X' + str(i) for i in range(TestInference.d_x)])

            cate_est = LinearDRLearner(model_regression=LinearRegression(),
                                       model_propensity=LogisticRegression(), featurizer=None)
            cate_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            fnames = ['Q' + str(i) for i in range(TestInference.d_x)]
            summary_results = cate_est.summary(T=1, feat_name=fnames)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, fnames)

            cate_est = LinearDRLearner(model_regression=LinearRegression(),
                                       model_propensity=LogisticRegression(), featurizer=None)
            wrapped_est = self._NoFeatNamesEst(cate_est)
            wrapped_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            summary_results = wrapped_est.summary(T=1)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, ['X' + str(i) for i in range(TestInference.d_x)])

            cate_est = LinearDRLearner(model_regression=LinearRegression(),
                                       model_propensity=LogisticRegression(), featurizer=None)
            wrapped_est = self._NoFeatNamesEst(cate_est)
            wrapped_est.fit(
                TestInference.Y,
                TestInference.T,
                TestInference.X,
                TestInference.W,
                inference=inference
            )
            fnames = ['Q' + str(i) for i in range(TestInference.d_x)]
            summary_results = wrapped_est.summary(T=1, feat_name=fnames)
            coef_rows = np.asarray(summary_results.tables[0].data)[1:, 0]
            np.testing.assert_array_equal(coef_rows, fnames)

    def test_degenerate_cases(self):
        """Test that we return the correct values when our distribution doesn't vary"""
        predictions = np.array([[1, 0], [1, 1]])  # first component is always 1
        for inf in [EmpiricalInferenceResults(d_t=1, d_y=2,
                                              pred=np.mean(predictions, axis=0), pred_dist=predictions,
                                              inf_type='coefficient'),
                    NormalInferenceResults(d_t=1, d_y=2,
                                           pred=np.mean(predictions, axis=0), pred_stderr=np.std(predictions, axis=0),
                                           inf_type='coefficient')]:
            zs = inf.zstat()
            pv = inf.pvalue()
            # test value 0 is less than estimate of 1 and variance is 0, so z score should be inf
            assert np.isposinf(zs[0])
            # predictions in column 1 have nonzero variance, so the zstat should always be some finite value
            assert np.isfinite(zs[1])
            assert pv[0] == 0  # pvalue should be zero when test value is greater or less than all samples

            test_point = np.array([1, 0.5])
            zs = inf.zstat(test_point)
            pv = inf.pvalue(test_point)
            # test value 1 is equal to the estimate of 1 and variance is 0, so z score should be nan
            assert np.isnan(zs[0])
            # predictions in column 1 have nonzero variance, so the zstat should always be some finite value
            assert np.isfinite(zs[1])
            # pvalue is also nan when variance is 0 and the point tested is equal to the point tested
            assert np.isnan(pv[0])
            # pvalue for second column should be greater than zero since some points are on either side
            # of the tested value
            assert 0 < pv[1] <= 1

            test_point = np.array([2, 1])
            zs = inf.zstat(test_point)
            pv = inf.pvalue(test_point)
            # test value 2 is greater than estimate of 1 and variance is 0, so z score should be -inf
            assert np.isneginf(zs[0])
            # predictions in column 1 have nonzero variance, so the zstat should always be some finite value
            assert np.isfinite(zs[1])
            # pvalue is also nan when variance is 0 and the point tested is equal to the point tested
            assert pv[0] == 0  # pvalue should be zero when test value is greater or less than all samples

            pop = PopulationSummaryResults(np.mean(predictions, axis=0).reshape(1, 2), np.std(
                predictions, axis=0).reshape(1, 2), d_t=1, d_y=2, alpha=0.05, value=0, decimals=3, tol=0.001)
            pop.print()  # verify that we can access all attributes even in degenerate case

    def test_can_summarize(self):
        LinearDML(model_t=LinearRegression(), model_y=LinearRegression()).fit(
            TestInference.Y,
            TestInference.T,
            TestInference.X,
            TestInference.W
        ).summary()

        LinearDRLearner(model_regression=LinearRegression(),
                        model_propensity=LogisticRegression(), fit_cate_intercept=False).fit(
            TestInference.Y,
            TestInference.T > 0,
            TestInference.X,
            TestInference.W,
            inference=BootstrapInference(5)
        ).summary(1)

    class _NoFeatNamesEst:
        def __init__(self, cate_est):
            self.cate_est = clone(cate_est, safe=False)

        def __getattr__(self, name):
            if name != 'cate_feature_names':
                return getattr(self.cate_est, name)
            else:
                return self.__getattribute__(name)
