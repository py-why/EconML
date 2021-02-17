# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
import pytest
import pickle
from sklearn.base import TransformerMixin
from numpy.random import normal, multivariate_normal, binomial
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import PolynomialFeatures
from econml.dr import DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner
from econml.utilities import shape, hstack, vstack, reshape, cross_product
from econml.inference import BootstrapInference, StatsModelsInferenceDiscrete
from contextlib import ExitStack
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
import scipy.special
import econml.tests.utilities  # bugfix for assertWarns


class TestDRLearner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set random seed
        cls.random_state = np.random.RandomState(12345)
        # Generate data
        # DGP constants
        cls.d = 5
        cls.n = 1000
        cls.n_test = 200
        cls.beta = np.array([0.25, -0.38, 1.41, 0.50, -1.22])
        cls.heterogeneity_index = 1
        # Test data
        cls.X_test = cls.random_state.multivariate_normal(
            np.zeros(cls.d),
            np.diag(np.ones(cls.d)),
            cls.n_test)
        # Constant treatment effect and propensity
        cls.const_te_data = TestDRLearner._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestDRLearner._const_te,
            propensity=lambda x: 0.3)
        # Heterogeneous treatment and propensity
        cls.heterogeneous_te_data = TestDRLearner._generate_data(
            cls.n, cls.d, cls._untreated_outcome,
            treatment_effect=TestDRLearner._heterogeneous_te,
            propensity=lambda x: (0.8 if (x[2] > -0.5 and x[2] < 0.5) else 0.2))

    def test_cate_api(self):
        """Test that we correctly implement the CATE API."""
        n = 20

        def make_random(is_discrete, d):
            if d is None:
                return None
            sz = (n, d) if d > 0 else (n,)
            if is_discrete:
                while True:
                    arr = np.random.choice(['a', 'b', 'c'], size=sz)
                    # ensure that we've got at least two of every element
                    _, counts = np.unique(arr, return_counts=True)
                    if len(counts) == 3 and counts.min() > 1:
                        return arr
            else:
                return np.random.normal(size=sz)

        for d_y in [0, 1]:
            is_discrete = True
            for d_t in [0, 1]:
                for d_x in [2, None]:
                    for d_w in [2, None]:
                        W, X, Y, T = [make_random(is_discrete, d)
                                      for is_discrete, d in [(False, d_w),
                                                             (False, d_x),
                                                             (False, d_y),
                                                             (is_discrete, d_t)]]

                        if (X is None) and (W is None):
                            continue
                        d_t_final = 2 if is_discrete else d_t

                        effect_shape = (n,) + ((d_y,) if d_y > 0 else ())
                        effect_summaryframe_shape = (
                            n * (d_y if d_y > 0 else 1), 6)
                        marginal_effect_shape = ((n,) +
                                                 ((d_y,) if d_y > 0 else ()) +
                                                 ((d_t_final,) if d_t_final > 0 else ()))
                        marginal_effect_summaryframe_shape = (n * (d_y if d_y > 0 else 1),
                                                              6 * (d_t_final if d_t_final > 0 else 1))

                        # since T isn't passed to const_marginal_effect, defaults to one row if X is None
                        const_marginal_effect_shape = ((n if d_x else 1,) +
                                                       ((d_y,) if d_y > 0 else ()) +
                                                       ((d_t_final,) if d_t_final > 0 else()))
                        const_marginal_effect_summaryframe_shape = (
                            (n if d_x else 1) * (d_y if d_y > 0 else 1),
                            6 * (d_t_final if d_t_final > 0 else 1))

                        coef_shape = ((d_y,) if d_y > 0 else ()) + (d_x or 1,)
                        coef_summaryframe_shape = ((d_y or 1) * (d_x or 1), 6)

                        intercept_shape = (d_y,) if d_y > 0 else ()
                        intercept_summaryframe_shape = (d_y if d_y > 0 else 1, 6)

                        for est in [LinearDRLearner(model_propensity=LogisticRegression(C=1000, solver='lbfgs',
                                                                                        multi_class='auto')),
                                    DRLearner(model_propensity=LogisticRegression(multi_class='auto'),
                                              model_regression=LinearRegression(),
                                              model_final=StatsModelsLinearRegression(),
                                              multitask_model_final=True)]:

                            # ensure that we can serialize unfit estimator
                            pickle.dumps(est)

                            infs = [None, BootstrapInference(2)]

                            test_linear_attrs = False

                            if isinstance(est, LinearDRLearner):
                                infs.append('auto')
                                test_linear_attrs = True

                            for inf in infs:
                                with self.subTest(d_w=d_w, d_x=d_x, d_y=d_y, d_t=d_t,
                                                  is_discrete=is_discrete, est=est, inf=inf):
                                    est.fit(Y, T, X=X, W=W, inference=inf)

                                    # ensure that we can serialize fit estimator
                                    pickle.dumps(est)

                                    # make sure we can call the marginal_effect and effect methods
                                    const_marg_eff = est.const_marginal_effect(
                                        X)
                                    marg_eff = est.marginal_effect(T, X)
                                    self.assertEqual(
                                        shape(marg_eff), marginal_effect_shape)
                                    self.assertEqual(
                                        shape(const_marg_eff), const_marginal_effect_shape)

                                    np.testing.assert_array_equal(
                                        marg_eff if d_x else marg_eff[0:1], const_marg_eff)

                                    T0 = np.full_like(T, 'a')
                                    eff = est.effect(X, T0=T0, T1=T)
                                    self.assertEqual(shape(eff), effect_shape)
                                    if inf is not None:
                                        T1 = np.full_like(T, 'b')

                                        const_marg_eff_int = est.const_marginal_effect_interval(
                                            X)
                                        marg_eff_int = est.marginal_effect_interval(
                                            T, X)
                                        self.assertEqual(shape(marg_eff_int),
                                                         (2,) + marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_eff_int),
                                                         (2,) + const_marginal_effect_shape)
                                        self.assertEqual(shape(est.effect_interval(X, T0=T0, T1=T)),
                                                         (2,) + effect_shape)

                                        const_marg_effect_inf = est.const_marginal_effect_inference(
                                            X)
                                        effect_inf = est.effect_inference(
                                            X, T0=T0, T1=T1)
                                        marg_effect_inf = est.marginal_effect_inference(
                                            T, X)

                                        # test const marginal inference
                                        self.assertEqual(shape(const_marg_effect_inf.summary_frame()),
                                                         const_marginal_effect_summaryframe_shape)
                                        self.assertEqual(shape(const_marg_effect_inf.point_estimate),
                                                         const_marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_effect_inf.stderr),
                                                         const_marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_effect_inf.var),
                                                         const_marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_effect_inf.pvalue()),
                                                         const_marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_effect_inf.zstat()),
                                                         const_marginal_effect_shape)
                                        self.assertEqual(shape(const_marg_effect_inf.conf_int()),
                                                         (2,) + const_marginal_effect_shape)
                                        np.testing.assert_array_almost_equal(const_marg_effect_inf.conf_int()
                                                                             [0], const_marg_eff_int[0], decimal=5)
                                        const_marg_effect_inf.population_summary()._repr_html_()

                                        # test effect inference
                                        self.assertEqual(shape(effect_inf.summary_frame()),
                                                         effect_summaryframe_shape)
                                        self.assertEqual(shape(effect_inf.point_estimate),
                                                         effect_shape)
                                        self.assertEqual(shape(effect_inf.stderr),
                                                         effect_shape)
                                        self.assertEqual(shape(effect_inf.var),
                                                         effect_shape)
                                        self.assertEqual(shape(effect_inf.pvalue()),
                                                         effect_shape)
                                        self.assertEqual(shape(effect_inf.zstat()),
                                                         effect_shape)
                                        self.assertEqual(shape(effect_inf.conf_int()),
                                                         (2,) + effect_shape)
                                        np.testing.assert_array_almost_equal(effect_inf.conf_int()[0],
                                                                             est.effect_interval(
                                            X, T0=T0, T1=T1)[0],
                                            decimal=5)
                                        effect_inf.population_summary()._repr_html_()

                                        # test marginal effect inference
                                        self.assertEqual(shape(marg_effect_inf.summary_frame()),
                                                         marginal_effect_summaryframe_shape)
                                        self.assertEqual(shape(marg_effect_inf.point_estimate),
                                                         marginal_effect_shape)
                                        self.assertEqual(shape(marg_effect_inf.stderr),
                                                         marginal_effect_shape)
                                        self.assertEqual(shape(marg_effect_inf.var),
                                                         marginal_effect_shape)
                                        self.assertEqual(shape(marg_effect_inf.pvalue()),
                                                         marginal_effect_shape)
                                        self.assertEqual(shape(marg_effect_inf.zstat()),
                                                         marginal_effect_shape)
                                        self.assertEqual(shape(marg_effect_inf.conf_int()),
                                                         (2,) + marginal_effect_shape)
                                        np.testing.assert_array_almost_equal(marg_effect_inf.conf_int()[0],
                                                                             marg_eff_int[0], decimal=5)
                                        marg_effect_inf.population_summary()._repr_html_()

                                        # test coef_ and intercept_ inference
                                        if test_linear_attrs:
                                            if X is not None:
                                                self.assertEqual(shape(est.coef_('b')), coef_shape)
                                                coef_inf = est.coef__inference('b')
                                                self.assertEqual(shape(coef_inf.summary_frame()),
                                                                 coef_summaryframe_shape)
                                                np.testing.assert_array_almost_equal(
                                                    coef_inf.conf_int()[0], est.coef__interval('b')[0])

                                            self.assertEqual(shape(est.intercept_('b')), intercept_shape)
                                            int_inf = est.intercept__inference('b')
                                            self.assertEqual(shape(int_inf.summary_frame()),
                                                             intercept_summaryframe_shape)
                                            np.testing.assert_array_almost_equal(
                                                int_inf.conf_int()[0], est.intercept__interval('b')[0])

                                            # verify we can generate the summary
                                            est.summary('b')

                                    est.score(Y, T, X, W)

                                    # make sure we can call effect with implied scalar treatments, no matter the
                                    # dimensions of T, and also that we warn when there are multiple treatments
                                    if d_t > 1:
                                        cm = self.assertWarns(Warning)
                                    else:
                                        cm = ExitStack()  # ExitStack can be used as a "do nothing" ContextManager
                                    with cm:
                                        effect_shape2 = (
                                            n if d_x else 1,) + ((d_y,) if d_y > 0 else())
                                        eff = est.effect(X, T0='a', T1='b')
                                        self.assertEqual(
                                            shape(eff), effect_shape2)

    def test_can_use_vectors(self):
        """
        TODO Almost identical to DML test, so consider merging
        Test that we can pass vectors for T and Y (not only 2-dimensional arrays).
        """
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(
                                  C=1000, solver='lbfgs', multi_class='auto'),
                              fit_cate_intercept=False,
                              featurizer=FunctionTransformer(validate=True))
        dml.fit(np.array([1, 2, 1, 2]), np.array(
            [1, 2, 1, 2]), X=np.ones((4, 1)))
        self.assertAlmostEqual(dml.coef_(T=2).reshape(())[()], 1)

    def test_can_use_sample_weights(self):
        """
        TODO Almost identical to DML test, so consider merging
        Test that we can pass sample weights to an estimator.
        """
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(
                                  C=1000, solver='lbfgs', multi_class='auto'),
                              featurizer=FunctionTransformer(validate=True))
        dml.fit(np.array([1, 2, 1, 2]), np.array([1, 2, 1, 2]), W=np.ones((4, 1)),
                sample_weight=np.ones((4, )))
        self.assertAlmostEqual(dml.intercept_(T=2), 1)

    def test_discrete_treatments(self):
        """
        TODO Almost identical to DML test, so consider merging
        Test that we can use discrete treatments
        """
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(
                                  C=1000, solver='lbfgs', multi_class='auto'),
                              featurizer=FunctionTransformer(validate=True))
        # create a simple artificial setup where effect of moving from treatment
        #     1 -> 2 is 2,
        #     1 -> 3 is 1, and
        #     2 -> 3 is -1 (necessarily, by composing the previous two effects)
        # Using an uneven number of examples from different classes,
        # and having the treatments in non-lexicographic order,
        # Should rule out some basic issues.
        dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array(
            [3, 2, 1, 2, 3, 1, 1, 1]), X=np.ones((8, 1)))
        np.testing.assert_almost_equal(dml.effect(np.ones((9, 1)),
                                                  T0=np.array(
                                                      [1, 1, 1, 2, 2, 2, 3, 3, 3]),
                                                  T1=np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])),
                                       [0, 2, 1, -2, 0, -1, -1, 1, 0])
        dml.score(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array(
            [3, 2, 1, 2, 3, 1, 1, 1]), np.ones((8, 1)))

    def test_can_custom_splitter(self):
        """
        TODO Almost identical to DML test, so consider merging
        """
        # test that we can fit with a KFold instance
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(
                                  C=1000, solver='lbfgs', multi_class='auto'),
                              cv=KFold(n_splits=3))
        dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array(
            [1, 2, 3, 1, 2, 3]), X=np.ones((6, 1)))
        dml.score(np.array([1, 2, 3, 1, 2, 3]), np.array(
            [1, 2, 3, 1, 2, 3]), np.ones((6, 1)))

        # test that we can fit with a train/test iterable
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(
                                  C=1000, solver='lbfgs', multi_class='auto'),
                              cv=[([0, 1, 2], [3, 4, 5])])
        dml.fit(np.array([1, 2, 3, 1, 2, 3]), np.array(
            [1, 2, 3, 1, 2, 3]), X=np.ones((6, 1)))
        dml.score(np.array([1, 2, 3, 1, 2, 3]), np.array(
            [1, 2, 3, 1, 2, 3]), np.ones((6, 1)))

    def test_can_use_statsmodel_inference(self):
        """
        TODO Almost identical to DML test, so consider merging
        Test that we can use statsmodels to generate confidence intervals
        """
        dml = LinearDRLearner(model_regression=LinearRegression(),
                              model_propensity=LogisticRegression(C=1000, solver='lbfgs', multi_class='auto'))
        dml.fit(np.array([2, 3, 1, 3, 2, 1, 1, 1]), np.array(
            [3, 2, 1, 2, 3, 1, 1, 1]), X=np.ones((8, 1)))
        interval = dml.effect_interval(np.ones((9, 1)),
                                       T0=np.array(
                                           [1, 1, 1, 1, 1, 1, 1, 1, 1]),
                                       T1=np.array(
                                           [2, 2, 3, 2, 2, 3, 2, 2, 3]),
                                       alpha=0.05)
        point = dml.effect(np.ones((9, 1)),
                           T0=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
                           T1=np.array([2, 2, 3, 2, 2, 3, 2, 2, 3]))
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        # for at least some of the examples, the CI should have nonzero width
        assert (lo < hi).any()

        interval = dml.const_marginal_effect_interval(
            np.ones((9, 1)), alpha=0.05)
        point = dml.const_marginal_effect(np.ones((9, 1)))
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        # for at least some of the examples, the CI should have nonzero width
        assert (lo < hi).any()

        interval = dml.coef__interval(T=2, alpha=0.05)
        point = dml.coef_(T=2)
        assert len(interval) == 2
        lo, hi = interval
        assert lo.shape == hi.shape == point.shape
        assert (lo <= point).all()
        assert (point <= hi).all()
        # for at least some of the examples, the CI should have nonzero width
        assert (lo < hi).any()

    def test_drlearner_all_attributes(self):
        import scipy.special
        np.random.seed(123)
        controls = np.random.uniform(-1, 1, size=(5000, 3))
        T = np.random.binomial(2, scipy.special.expit(controls[:, 0]))
        sigma = 0.01
        y = (1 + .5 * controls[:, 0]) * T + controls[:,
                                                     0] + np.random.normal(0, sigma, size=(5000,))
        for X in [controls]:
            for W in [None, controls]:
                for sample_weight in [None, 1 + np.random.randint(10, size=X.shape[0])]:
                    for sample_var in [None, 1 + np.random.randint(10, size=X.shape[0])]:
                        for featurizer in [None, PolynomialFeatures(degree=2, include_bias=False)]:
                            for models in [(GradientBoostingClassifier(), GradientBoostingRegressor(),
                                            RandomForestRegressor(n_estimators=100,
                                                                  max_depth=5, min_samples_leaf=50)),
                                           (GradientBoostingClassifier(), GradientBoostingRegressor(),
                                            RandomForestRegressor(n_estimators=100,
                                                                  max_depth=5, min_samples_leaf=50)),
                                           (LogisticRegression(solver='lbfgs', multi_class='auto'),
                                            LinearRegression(), StatsModelsLinearRegression())]:
                                for multitask_model_final in [False, True]:
                                    if (not isinstance(models, StatsModelsLinearRegression))\
                                            and (sample_var is not None):
                                        continue
                                    with self.subTest(X=X, W=W, sample_weight=sample_weight, sample_var=sample_var,
                                                      featurizer=featurizer, models=models,
                                                      multitask_model_final=multitask_model_final):
                                        est = DRLearner(model_propensity=models[0],
                                                        model_regression=models[1],
                                                        model_final=models[2],
                                                        featurizer=featurizer,
                                                        multitask_model_final=multitask_model_final)
                                        if (X is None) and (W is None):
                                            with pytest.raises(AttributeError) as e_info:
                                                est.fit(y, T, X=X, W=W,
                                                        sample_weight=sample_weight, sample_var=sample_var)
                                            continue
                                        est.fit(
                                            y, T, X=X, W=W, sample_weight=sample_weight, sample_var=sample_var)
                                        np.testing.assert_allclose(est.effect(X[:3], T0=0, T1=1), 1 + .5 * X[:3, 0],
                                                                   rtol=0, atol=.15)
                                        np.testing.assert_allclose(est.const_marginal_effect(X[:3]),
                                                                   np.hstack(
                                                                       [1 + .5 * X[:3, [0]],
                                                                        2 * (1 + .5 * X[:3, [0]])]),
                                                                   rtol=0, atol=.15)
                                        for t in [1, 2]:
                                            np.testing.assert_allclose(est.marginal_effect(t, X[:3]),
                                                                       np.hstack([1 + .5 * X[:3, [0]],
                                                                                  2 * (1 + .5 * X[:3, [0]])]),
                                                                       rtol=0, atol=.15)
                                        assert isinstance(est.score_, float)
                                        assert isinstance(
                                            est.score(y, T, X=X, W=W), float)

                                        feature_names = ['A', 'B', 'C']
                                        out_feat_names = feature_names
                                        if featurizer is not None:
                                            out_feat_names = featurizer.fit(
                                                X).get_feature_names(feature_names)
                                            np.testing.assert_array_equal(
                                                est.featurizer_.n_input_features_, 3)
                                        np.testing.assert_array_equal(est.cate_feature_names(feature_names),
                                                                      out_feat_names)

                                        if isinstance(models[0], GradientBoostingClassifier):
                                            np.testing.assert_array_equal(np.array([mdl.feature_importances_
                                                                                    for mdl
                                                                                    in est.models_regression]).shape,
                                                                          [2, 2 + X.shape[1] +
                                                                           (W.shape[1] if W is not None else 0)])
                                            np.testing.assert_array_equal(np.array([mdl.feature_importances_
                                                                                    for mdl
                                                                                    in est.models_propensity]).shape,
                                                                          [2, X.shape[1] +
                                                                           (W.shape[1] if W is not None else 0)])
                                        else:
                                            np.testing.assert_array_equal(np.array([mdl.coef_
                                                                                    for mdl
                                                                                    in est.models_regression]).shape,
                                                                          [2, 2 + X.shape[1] +
                                                                           (W.shape[1] if W is not None else 0)])
                                            np.testing.assert_array_equal(np.array([mdl.coef_
                                                                                    for mdl
                                                                                    in est.models_propensity]).shape,
                                                                          [2, 3, X.shape[1] +
                                                                           (W.shape[1] if W is not None else 0)])
                                        if multitask_model_final:
                                            if isinstance(models[2], RandomForestRegressor):
                                                np.testing.assert_equal(np.argsort(
                                                    est.multitask_model_cate.feature_importances_)[-1], 0)
                                            else:
                                                true_coef = np.zeros(
                                                    (2, len(out_feat_names)))
                                                true_coef[:, 0] = [.5, 1]
                                                np.testing.assert_allclose(
                                                    est.multitask_model_cate.coef_, true_coef, rtol=0, atol=.15)
                                                np.testing.assert_allclose(
                                                    est.multitask_model_cate.intercept_, [1, 2], rtol=0, atol=.15)
                                        else:
                                            for t in [1, 2]:
                                                if isinstance(models[2], RandomForestRegressor):
                                                    np.testing.assert_equal(np.argsort(
                                                        est.model_cate(T=t).feature_importances_)[-1], 0)
                                                else:
                                                    true_coef = np.zeros(
                                                        len(out_feat_names))
                                                    true_coef[0] = .5 * t
                                                    np.testing.assert_allclose(
                                                        est.model_cate(T=t).coef_, true_coef, rtol=0, atol=.15)
                                                    np.testing.assert_allclose(
                                                        est.model_cate(T=t).intercept_, t, rtol=0, atol=.15)

    def test_drlearner_with_inference_all_attributes(self):
        np.random.seed(123)
        controls = np.random.uniform(-1, 1, size=(10000, 2))
        T = np.random.binomial(2, scipy.special.expit(controls[:, 0]))
        sigma = 0.01
        y = (1 + .5 * controls[:, 0]) * T + controls[:,
                                                     0] + np.random.normal(0, sigma, size=(10000,))
        for X in [None, controls]:
            for W in [None, controls]:
                for sample_weight, sample_var in [(None, None), (np.ones(T.shape[0]), np.zeros(T.shape[0]))]:
                    for featurizer in [None, PolynomialFeatures(degree=2, include_bias=False)]:
                        for model_t, model_y, est_class,\
                                inference in [(GradientBoostingClassifier(), GradientBoostingRegressor(),
                                               ForestDRLearner, 'auto'),
                                              (LogisticRegression(solver='lbfgs', multi_class='auto'),
                                               LinearRegression(), LinearDRLearner, 'auto'),
                                              (LogisticRegression(solver='lbfgs', multi_class='auto'),
                                               LinearRegression(), LinearDRLearner, StatsModelsInferenceDiscrete(
                                                  cov_type='nonrobust')),
                                              (LogisticRegression(solver='lbfgs', multi_class='auto'),
                                               LinearRegression(), SparseLinearDRLearner, 'auto')
                                              ]:
                            with self.subTest(X=X, W=W, sample_weight=sample_weight, sample_var=sample_var,
                                              featurizer=featurizer, model_y=model_y, model_t=model_t,
                                              est_class=est_class, inference=inference):
                                if (X is None) and (est_class == SparseLinearDRLearner):
                                    continue
                                if (X is None) and (est_class == ForestDRLearner):
                                    continue
                                if (featurizer is not None) and (est_class == ForestDRLearner):
                                    continue

                                if est_class == ForestDRLearner:
                                    est = est_class(model_propensity=model_t,
                                                    model_regression=model_y,
                                                    n_estimators=1000)
                                else:
                                    est = est_class(model_propensity=model_t,
                                                    model_regression=model_y,
                                                    featurizer=featurizer)

                                if (X is None) and (W is None):
                                    with pytest.raises(AttributeError) as e_info:
                                        est.fit(
                                            y, T, X=X, W=W, sample_weight=sample_weight, sample_var=sample_var)
                                    continue
                                est.fit(y, T, X=X, W=W, sample_weight=sample_weight,
                                        sample_var=sample_var, inference=inference)
                                if X is not None:
                                    lower, upper = est.effect_interval(
                                        X[:3], T0=0, T1=1)
                                    point = est.effect(X[:3], T0=0, T1=1)
                                    truth = 1 + .5 * X[:3, 0]
                                    TestDRLearner._check_with_interval(
                                        truth, point, lower, upper)
                                    lower, upper = est.const_marginal_effect_interval(
                                        X[:3])
                                    point = est.const_marginal_effect(
                                        X[:3])
                                    truth = np.hstack(
                                        [1 + .5 * X[:3, [0]], 2 * (1 + .5 * X[:3, [0]])])
                                    TestDRLearner._check_with_interval(
                                        truth, point, lower, upper)
                                else:
                                    lower, upper = est.effect_interval(
                                        T0=0, T1=1)
                                    point = est.effect(T0=0, T1=1)
                                    truth = np.array([1])
                                    TestDRLearner._check_with_interval(
                                        truth, point, lower, upper)
                                    lower, upper = est.const_marginal_effect_interval()
                                    point = est.const_marginal_effect()
                                    truth = np.array([[1, 2]])
                                    TestDRLearner._check_with_interval(
                                        truth, point, lower, upper)

                                for t in [1, 2]:
                                    if X is not None:
                                        lower, upper = est.marginal_effect_interval(
                                            t, X[:3])
                                        point = est.marginal_effect(
                                            t, X[:3])
                                        truth = np.hstack(
                                            [1 + .5 * X[:3, [0]], 2 * (1 + .5 * X[:3, [0]])])
                                        TestDRLearner._check_with_interval(
                                            truth, point, lower, upper)
                                    else:
                                        lower, upper = est.marginal_effect_interval(
                                            t)
                                        point = est.marginal_effect(t)
                                        truth = np.array([[1, 2]])
                                        TestDRLearner._check_with_interval(
                                            truth, point, lower, upper)
                                assert isinstance(est.score_, float)
                                assert isinstance(
                                    est.score(y, T, X=X, W=W), float)

                                if X is not None:
                                    feature_names = ['A', 'B']
                                else:
                                    feature_names = []
                                out_feat_names = feature_names
                                if X is not None:
                                    if (featurizer is not None):
                                        out_feat_names = featurizer.fit(
                                            X).get_feature_names(feature_names)
                                        np.testing.assert_array_equal(
                                            est.featurizer_.n_input_features_, 2)
                                    np.testing.assert_array_equal(est.cate_feature_names(feature_names),
                                                                  out_feat_names)

                                if isinstance(model_t, GradientBoostingClassifier):
                                    np.testing.assert_array_equal(np.array([mdl.feature_importances_
                                                                            for mdl
                                                                            in est.models_regression]).shape,
                                                                  [2, 2 + len(feature_names) +
                                                                   (W.shape[1] if W is not None else 0)])
                                    np.testing.assert_array_equal(np.array([mdl.feature_importances_
                                                                            for mdl
                                                                            in est.models_propensity]).shape,
                                                                  [2, len(feature_names) +
                                                                   (W.shape[1] if W is not None else 0)])
                                else:
                                    np.testing.assert_array_equal(np.array([mdl.coef_
                                                                            for mdl
                                                                            in est.models_regression]).shape,
                                                                  [2, 2 + len(feature_names) +
                                                                   (W.shape[1] if W is not None else 0)])
                                    np.testing.assert_array_equal(np.array([mdl.coef_
                                                                            for mdl
                                                                            in est.models_propensity]).shape,
                                                                  [2, 3, len(feature_names) +
                                                                   (W.shape[1] if W is not None else 0)])

                                if isinstance(est, LinearDRLearner) or isinstance(est, SparseLinearDRLearner):
                                    if X is not None:
                                        for t in [1, 2]:
                                            true_coef = np.zeros(
                                                len(out_feat_names))
                                            true_coef[0] = .5 * t
                                            lower, upper = est.model_cate(
                                                T=t).coef__interval()
                                            point = est.model_cate(
                                                T=t).coef_
                                            truth = true_coef
                                            TestDRLearner._check_with_interval(
                                                truth, point, lower, upper)

                                            lower, upper = est.coef__interval(
                                                t)
                                            point = est.coef_(t)
                                            truth = true_coef
                                            TestDRLearner._check_with_interval(
                                                truth, point, lower, upper)
                                            # test coef__inference function works
                                            est.coef__inference(
                                                t).summary_frame()
                                            np.testing.assert_array_almost_equal(
                                                est.coef__inference(t).conf_int()[0], lower, decimal=5)
                                    for t in [1, 2]:
                                        lower, upper = est.model_cate(
                                            T=t).intercept__interval()
                                        point = est.model_cate(
                                            T=t).intercept_
                                        truth = t
                                        TestDRLearner._check_with_interval(
                                            truth, point, lower, upper)

                                        lower, upper = est.intercept__interval(
                                            t)
                                        point = est.intercept_(t)
                                        truth = t
                                        TestDRLearner._check_with_interval(
                                            truth, point, lower, upper)
                                        # test intercept__inference function works
                                        est.intercept__inference(
                                            t).summary_frame()
                                        np.testing.assert_array_almost_equal(
                                            est.intercept__inference(t).conf_int()[0], lower, decimal=5)
                                        # test summary function works
                                        est.summary(t)

                                if isinstance(est, ForestDRLearner):
                                    for t in [1, 2]:
                                        np.testing.assert_array_equal(est.feature_importances_(t).shape,
                                                                      [X.shape[1]])

    @staticmethod
    def _check_with_interval(truth, point, lower, upper):
        np.testing.assert_allclose(point, truth, rtol=0, atol=.2)
        np.testing.assert_array_less(lower - 0.05, truth)
        np.testing.assert_array_less(truth, upper + 0.05)

    def test_DRLearner(self):
        """Tests whether the DRLearner can accurately estimate constant and
           heterogeneous treatment effects.
        """
        DR_learner = DRLearner(model_regression=LinearRegression(),
                               model_final=LinearRegression())
        # Test inputs
        # self._test_inputs(DR_learner)
        # Test constant treatment effect
        self._test_te(DR_learner, tol=0.5, te_type="const")
        # Test heterogeneous treatment effect
        outcome_model = Pipeline(
            [('poly', PolynomialFeatures()), ('model', LinearRegression())])
        DR_learner = DRLearner(model_regression=outcome_model,
                               model_final=LinearRegression())
        self._test_te(DR_learner, tol=0.5, te_type="heterogeneous")
        # Test heterogenous treatment effect for W =/= None
        self._test_with_W(DR_learner, tol=0.5)

    def test_sparse(self):
        """SparseDRLearner test with a sparse DGP"""
        # Sparse DGP
        np.random.seed(123)
        n_x = 50
        n_nonzero = 1
        n_w = 5
        n = 2000
        # Treatment effect coef
        a = np.zeros(n_x)
        nonzero_idx = np.random.choice(n_x, size=n_nonzero, replace=False)
        a[nonzero_idx] = 1
        # Other coefs
        b = np.zeros(n_x + n_w)
        g = np.zeros(n_x + n_w)
        b_nonzero = np.random.choice(n_x + n_w, size=n_nonzero, replace=False)
        g_nonzero = np.random.choice(n_x + n_w, size=n_nonzero, replace=False)
        b[b_nonzero] = 1
        g[g_nonzero] = 1
        # Features and controls
        x = np.random.normal(size=(n, n_x))
        w = np.random.normal(size=(n, n_w))
        xw = np.hstack([x, w])
        T = np.random.binomial(1, scipy.special.expit(xw @ b))
        err_Y = np.random.normal(size=n, scale=0.5)
        Y = T * (x @ a) + xw @ g + err_Y
        # Test sparse estimator
        # --> test coef_, intercept_
        sparse_dml = SparseLinearDRLearner(featurizer=FunctionTransformer())
        sparse_dml.fit(Y, T, X=x, W=w)
        np.testing.assert_allclose(a, sparse_dml.coef_(T=1), atol=2e-1)
        np.testing.assert_allclose(sparse_dml.intercept_(T=1), 0, atol=2e-1)
        # --> test treatment effects
        # Restrict x_test to vectors of norm < 1
        x_test = np.random.uniform(size=(10, n_x))
        true_eff = (x_test @ a)
        eff = sparse_dml.effect(x_test, T0=0, T1=1)
        np.testing.assert_allclose(true_eff, eff, atol=0.5)
        # --> check inference
        y_lower, y_upper = sparse_dml.effect_interval(x_test, T0=0, T1=1)
        in_CI = ((y_lower < true_eff) & (true_eff < y_upper))
        # Check that a majority of true effects lie in the 5-95% CI
        self.assertGreater(in_CI.mean(), 0.8)

    def test_groups(self):
        groups = [1, 2, 3, 4, 5, 6] * 10
        t = [1, 2, 3] * 20
        y = groups
        w = np.random.normal(size=(60, 1))
        est = LinearDRLearner()
        with pytest.raises(Exception):  # can't pass groups without a compatible n_split
            est.fit(y, t, W=w, groups=groups)

        # test outer grouping
        # NOTE: we should ideally use a stratified split with grouping, but sklearn doesn't have one yet
        est = LinearDRLearner(model_propensity=LogisticRegression(),
                              model_regression=LinearRegression(), cv=GroupKFold(2))
        est.fit(y, t, W=w, groups=groups)

        # test nested grouping
        class NestedModel(LassoCV):
            def __init__(self, eps=1e-3, n_alphas=100, alphas=None, fit_intercept=True,
                         precompute='auto', max_iter=1000, tol=1e-4, normalize=False,
                         copy_X=True, cv=None, verbose=False, n_jobs=None,
                         positive=False, random_state=None, selection='cyclic'):

                super().__init__(
                    eps=eps, n_alphas=n_alphas, alphas=alphas,
                    fit_intercept=fit_intercept, normalize=normalize,
                    precompute=precompute, max_iter=max_iter, tol=tol, copy_X=copy_X,
                    cv=cv, verbose=verbose, n_jobs=n_jobs, positive=positive,
                    random_state=random_state, selection=selection)

            def fit(self, X, y):
                # ensure that the grouping has worked correctly and we get all 10 copies of the items in
                # whichever groups we saw
                (yvals, cts) = np.unique(y, return_counts=True)
                for (yval, ct) in zip(yvals, cts):
                    if ct != 10:
                        raise Exception("Grouping failed; received {0} copies of {1} instead of 10".format(ct, yval))
                return super().fit(X, y)

        # test nested grouping
        est = LinearDRLearner(model_propensity=LogisticRegression(),
                              model_regression=NestedModel(cv=2), cv=GroupKFold(2))
        est.fit(y, t, W=w, groups=groups)

        # by default, we use 5 split cross-validation for our T and Y models
        # but we don't have enough groups here to split both the outer and inner samples with grouping
        # TODO: does this imply we should change some defaults to make this more likely to succeed?
        est = LinearDRLearner(cv=GroupKFold(2))
        with pytest.raises(Exception):
            est.fit(y, t, W=w, groups=groups)

    def _test_te(self, learner_instance, tol, te_type="const"):
        if te_type not in ["const", "heterogeneous"]:
            raise ValueError(
                "Type of treatment effect must be 'const' or 'heterogeneous'.")
        X, T, Y = getattr(
            TestDRLearner, "{te_type}_te_data".format(te_type=te_type))
        te_func = getattr(
            TestDRLearner, "_{te_type}_te".format(te_type=te_type))
        # Fit learner and get the effect
        learner_instance.fit(Y, T, X=X)
        te_hat = learner_instance.effect(TestDRLearner.X_test)
        # Get the true treatment effect
        te = np.apply_along_axis(te_func, 1, TestDRLearner.X_test)
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)

    def _test_with_W(self, learner_instance, tol):
        # Only for heterogeneous TE
        X, T, Y = TestDRLearner.heterogeneous_te_data
        # Fit learner on X and W and get the effect
        learner_instance.fit(
            Y, T, X=X[:, [TestDRLearner.heterogeneity_index]], W=X)
        te_hat = learner_instance.effect(
            TestDRLearner.X_test[:, [TestDRLearner.heterogeneity_index]])
        # Get the true treatment effect
        te = np.apply_along_axis(
            TestDRLearner._heterogeneous_te, 1, TestDRLearner.X_test)
        # Compute treatment effect residuals (absolute)
        te_res = np.abs(te - te_hat)
        # Check that at least 90% of predictions are within tolerance interval
        self.assertGreaterEqual(np.mean(te_res < tol), 0.90)

    def _test_inputs(self, learner_instance):
        X, T, Y = TestDRLearner.const_te_data
        # Check that one can pass in regular lists
        learner_instance.fit(list(Y), list(T), X=list(X))
        learner_instance.effect(list(TestDRLearner.X_test))
        # Check that it fails correctly if lists of different shape are passed in
        self.assertRaises(ValueError, learner_instance.fit,
                          Y, T, X[:TestDRLearner.n // 2])
        self.assertRaises(ValueError, learner_instance.fit,
                          Y[:TestDRLearner.n // 2], T, X)
        # Check that it fails when T contains values other than 0 and 1
        self.assertRaises(ValueError, learner_instance.fit, Y, T + 1, X)
        # Check that it works when T, Y have shape (n, 1)
        self.assertWarns(DataConversionWarning,
                         learner_instance.fit, Y.reshape(-1,
                                                         1), T.reshape(-1, 1), X
                         )

    @classmethod
    def _untreated_outcome(cls, x):
        return np.dot(x, cls.beta) + cls.random_state.normal(0, 1)

    @classmethod
    def _const_te(cls, x):
        return 2

    @classmethod
    def _heterogeneous_te(cls, x):
        return x[cls.heterogeneity_index]

    @classmethod
    def _generate_data(cls, n, d, untreated_outcome, treatment_effect, propensity):
        """Generates population data for given untreated_outcome, treatment_effect and propensity functions.

        Parameters
        ----------
            n (int): population size
            d (int): number of covariates
            untreated_outcome (func): untreated outcome conditional on covariates
            treatment_effect (func): treatment effect conditional on covariates
            propensity (func): probability of treatment conditional on covariates
        """
        # Generate covariates
        X = cls.random_state.multivariate_normal(
            np.zeros(d), np.diag(np.ones(d)), n)
        # Generate treatment
        T = np.apply_along_axis(
            lambda x: cls.random_state.binomial(1, propensity(x), 1)[0], 1, X)
        # Calculate outcome
        Y0 = np.apply_along_axis(lambda x: untreated_outcome(x), 1, X)
        treat_effect = np.apply_along_axis(lambda x: treatment_effect(x), 1, X)
        Y = Y0 + treat_effect * T
        return (X, T, Y)
