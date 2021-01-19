import unittest
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from econml.dml import (DML, LinearDML, SparseLinearDML, KernelDML, NonParamDML, ForestDML)
from econml.dr import (DRLearner, LinearDRLearner, SparseLinearDRLearner, ForestDRLearner)
from econml.iv.dml import (DMLATEIV, ProjectedDMLATEIV, DMLIV, NonParamDMLIV)
from econml.iv.dr import (IntentToTreatDRIV, LinearIntentToTreatDRIV)
from econml.sklearn_extensions.linear_model import (DebiasedLasso, WeightedLasso,
                                                    StatsModelsRLM, StatsModelsLinearRegression)
from econml.inference import NormalInferenceResults, BootstrapInference


class TestRefit(unittest.TestCase):

    def _get_data(self):
        X = np.random.choice(np.arange(5), size=(500, 3))
        y = np.random.normal(size=(500,))
        T = np.random.binomial(1, .5, size=(500,))
        W = np.random.normal(size=(500, 2))
        return y, T, X, W

    def test_dml(self):
        """Test setting attributes and refitting"""
        y, T, X, W = self._get_data()

        dml = DML(model_y=LinearRegression(),
                  model_t=LinearRegression(),
                  model_final=StatsModelsLinearRegression(fit_intercept=False),
                  linear_first_stages=False,
                  random_state=123)
        dml.fit(y, T, X=X, W=W)
        with pytest.raises(Exception):
            dml.refit_final()
        dml.fit(y, T, X=X, W=W, cache_values=True)
        dml.model_final = StatsModelsRLM(fit_intercept=False)
        dml.refit_final()
        assert isinstance(dml.model_cate, StatsModelsRLM)
        np.testing.assert_array_equal(dml.model_cate.coef_[1:].flatten(), dml.coef_.flatten())
        lb, ub = dml.model_cate.coef__interval(alpha=0.01)
        lbt, ubt = dml.coef__interval(alpha=0.01)
        np.testing.assert_array_equal(lb[1:].flatten(), lbt.flatten())
        np.testing.assert_array_equal(ub[1:].flatten(), ubt.flatten())
        intcpt = dml.intercept_
        dml.fit_cate_intercept = False
        np.testing.assert_equal(dml.intercept_, intcpt)
        dml.refit_final()
        np.testing.assert_array_equal(dml.model_cate.coef_.flatten(), dml.coef_.flatten())
        lb, ub = dml.model_cate.coef__interval(alpha=0.01)
        lbt, ubt = dml.coef__interval(alpha=0.01)
        np.testing.assert_array_equal(lb.flatten(), lbt.flatten())
        np.testing.assert_array_equal(ub.flatten(), ubt.flatten())
        with pytest.raises(AttributeError):
            dml.intercept_
        with pytest.raises(AttributeError):
            dml.intercept__interval()
        dml.model_final = DebiasedLasso(fit_intercept=False)
        dml.refit_final()
        assert isinstance(dml.model_cate, DebiasedLasso)
        dml.featurizer = PolynomialFeatures(degree=2, include_bias=False)
        dml.model_final = StatsModelsLinearRegression(fit_intercept=False)
        dml.refit_final()
        assert isinstance(dml.featurizer_, PolynomialFeatures)
        dml.fit_cate_intercept = True
        dml.refit_final()
        assert isinstance(dml.featurizer_, Pipeline)
        np.testing.assert_array_equal(dml.coef_.shape, (X.shape[1]**2))
        np.testing.assert_array_equal(dml.coef__interval()[0].shape, (X.shape[1]**2))
        coefpre = dml.coef_
        coefpreint = dml.coef__interval()
        dml.fit(y, T, X=X, W=W)
        np.testing.assert_array_equal(coefpre, dml.coef_)
        np.testing.assert_array_equal(coefpreint[0], dml.coef__interval()[0])
        dml.discrete_treatment = True
        dml.featurizer = None
        dml.linear_first_stages = True
        dml.model_t = LogisticRegression()
        dml.fit(y, T, X=X, W=W)
        newdml = DML(model_y=LinearRegression(),
                     model_t=LogisticRegression(),
                     model_final=StatsModelsLinearRegression(fit_intercept=False),
                     discrete_treatment=True,
                     linear_first_stages=True,
                     random_state=123).fit(y, T, X=X, W=W)
        np.testing.assert_array_equal(dml.coef_, newdml.coef_)
        np.testing.assert_array_equal(dml.coef__interval()[0], newdml.coef__interval()[0])

        ldml = LinearDML(model_y=LinearRegression(),
                         model_t=LinearRegression(),
                         linear_first_stages=False)
        ldml.fit(y, T, X=X, W=W, cache_values=True)
        # can set final model for plain DML, but can't for LinearDML (hardcoded to StatsModelsRegression)
        with pytest.raises(ValueError):
            ldml.model_final = StatsModelsRLM()

        ldml = SparseLinearDML(model_y=LinearRegression(),
                               model_t=LinearRegression(),
                               linear_first_stages=False)
        ldml.fit(y, T, X=X, W=W, cache_values=True)
        # can set final model for plain DML, but can't for LinearDML (hardcoded to StatsModelsRegression)
        with pytest.raises(ValueError):
            ldml.model_final = StatsModelsRLM()
        ldml.alpha = 0.01
        ldml.max_iter = 10
        ldml.tol = 0.01
        ldml.refit_final()
        np.testing.assert_equal(ldml.model_cate.estimators_[0].alpha, 0.01)
        np.testing.assert_equal(ldml.model_cate.estimators_[0].max_iter, 10)
        np.testing.assert_equal(ldml.model_cate.estimators_[0].tol, 0.01)

    def test_nonparam_dml(self):
        y, T, X, W = self._get_data()

        dml = NonParamDML(model_y=LinearRegression(),
                          model_t=LinearRegression(),
                          model_final=WeightedLasso(),
                          random_state=123)
        dml.fit(y, T, X=X, W=W)
        with pytest.raises(Exception):
            dml.refit_final()
        dml.fit(y, T, X=X, W=W, cache_values=True)
        dml.model_final = DebiasedLasso(fit_intercept=False)
        dml.refit_final()
        assert isinstance(dml.model_cate, DebiasedLasso)
        dml.effect_interval(X[:1])
        dml.featurizer = PolynomialFeatures(degree=2, include_bias=False)
        dml.refit_final()
        assert isinstance(dml.featurizer_, PolynomialFeatures)
        dml.effect_interval(X[:1])
        dml.discrete_treatment = True
        dml.featurizer = None
        dml.linear_first_stages = True
        dml.model_t = LogisticRegression()
        dml.model_final = DebiasedLasso()
        dml.fit(y, T, X=X, W=W)
        newdml = NonParamDML(model_y=LinearRegression(),
                             model_t=LogisticRegression(),
                             model_final=DebiasedLasso(),
                             discrete_treatment=True,
                             random_state=123).fit(y, T, X=X, W=W)
        np.testing.assert_array_equal(dml.effect(X[:1]), newdml.effect(X[:1]))
        np.testing.assert_array_equal(dml.effect_interval(X[:1])[0], newdml.effect_interval(X[:1])[0])

    def test_drlearner(self):
        y, T, X, W = self._get_data()

        for est in [LinearDRLearner(random_state=123),
                    SparseLinearDRLearner(random_state=123)]:
            est.fit(y, T, X=X, W=W, cache_values=True)
            np.testing.assert_equal(est.model_regression, 'auto')
            est.model_regression = LinearRegression()
            est.model_propensity = LogisticRegression()
            est.fit(y, T, X=X, W=W, cache_values=True)
            assert isinstance(est.model_regression, LinearRegression)
            with pytest.raises(ValueError):
                est.multitask_model_final = True
            with pytest.raises(ValueError):
                est.model_final = LinearRegression()
            est.min_propensity = .1
            est.mc_iters = 2
            est.featurizer = PolynomialFeatures(degree=2, include_bias=False)
            est.refit_final()
            assert isinstance(est.featurizer_, PolynomialFeatures)
            np.testing.assert_equal(est.mc_iters, 2)
            intcpt = est.intercept_(T=1)
            est.fit_cate_intercept = False
            np.testing.assert_equal(est.intercept_(T=1), intcpt)
            est.refit_final()
            with pytest.raises(AttributeError):
                est.intercept(T=1)
            est.fit(y, T, X=X, W=W, cache_values=False)
            with pytest.raises(AssertionError):
                est.refit_final()

    def test_orthoiv(self):
        y, T, X, W = self._get_data()
        Z = T.copy()
        est = DMLATEIV(model_Y_W=LinearRegression(),
                       model_T_W=LinearRegression(),
                       model_Z_W=LinearRegression(),
                       mc_iters=2)
        est.fit(y, T, W=W, Z=Z, cache_values=True)
        est.refit_final()
        est.model_Y_W = Lasso()
        est.model_T_W = ElasticNet()
        est.model_Z_W = WeightedLasso()
        est.fit(y, T, W=W, Z=Z, cache_values=True)
        assert isinstance(est.models_nuisance_[0]._model_Y_W._model, Lasso)
        assert isinstance(est.models_nuisance_[0]._model_T_W._model, ElasticNet)
        assert isinstance(est.models_nuisance_[0]._model_Z_W._model, WeightedLasso)

        est = ProjectedDMLATEIV(model_Y_W=LinearRegression(),
                                model_T_W=LinearRegression(),
                                model_T_WZ=LinearRegression(),
                                mc_iters=2)
        est.fit(y, T, W=W, Z=Z, cache_values=True)
        est.refit_final()
        est.model_Y_W = Lasso()
        est.model_T_W = ElasticNet()
        est.model_T_WZ = WeightedLasso()
        est.fit(y, T, W=W, Z=Z, cache_values=True)
        assert isinstance(est.models_nuisance_[0]._model_Y_W._model, Lasso)
        assert isinstance(est.models_nuisance_[0]._model_T_W._model, ElasticNet)
        assert isinstance(est.models_nuisance_[0]._model_T_WZ._model, WeightedLasso)

        est = DMLIV(model_Y_X=LinearRegression(),
                    model_T_X=LinearRegression(),
                    model_T_XZ=LinearRegression(),
                    model_final=LinearRegression(fit_intercept=False),
                    mc_iters=2)
        est.fit(y, T, X=X, Z=Z, cache_values=True)
        np.testing.assert_equal(len(est.coef_), X.shape[1])
        est.featurizer = PolynomialFeatures(degree=2, include_bias=False)
        est.refit_final()
        np.testing.assert_equal(len(est.coef_), X.shape[1]**2)
        est.intercept_
        est.fit_cate_intercept = False
        est.intercept_
        est.refit_final()
        with pytest.raises(AttributeError):
            est.intercept_
        est.model_Y_X = Lasso()
        est.model_T_X = ElasticNet()
        est.model_T_XZ = WeightedLasso()
        est.fit(y, T, X=X, Z=Z, cache_values=True)
        assert isinstance(est.models_Y_X[0], Lasso)
        assert isinstance(est.models_T_X[0], ElasticNet)
        assert isinstance(est.models_T_XZ[0], WeightedLasso)

        est = DMLIV(model_Y_X=LinearRegression(),
                    model_T_X=LinearRegression(),
                    model_T_XZ=LinearRegression(),
                    model_final=LinearRegression(fit_intercept=False),
                    mc_iters=2)
        est.fit(y, T, X=X, Z=Z, cache_values=True)
        np.testing.assert_equal(len(est.coef_), X.shape[1])
        est.featurizer = PolynomialFeatures(degree=2, include_bias=False)
        est.refit_final()
        np.testing.assert_equal(len(est.coef_), X.shape[1]**2)
        est.intercept_
        est.fit_cate_intercept = False
        est.intercept_
        est.refit_final()
        with pytest.raises(AttributeError):
            est.intercept_
        est.model_Y_X = Lasso()
        est.model_T_X = ElasticNet()
        est.model_T_XZ = WeightedLasso()
        est.fit(y, T, X=X, Z=Z, cache_values=True)
        assert isinstance(est.models_nuisance_[0]._model_Y_X._model, Lasso)
        assert isinstance(est.models_nuisance_[0]._model_T_X._model, ElasticNet)
        assert isinstance(est.models_nuisance_[0]._model_T_XZ._model, WeightedLasso)

        est = NonParamDMLIV(model_Y_X=LinearRegression(),
                            model_T_X=LinearRegression(),
                            model_T_XZ=LinearRegression(),
                            model_final=LinearRegression(fit_intercept=True),
                            mc_iters=2)
        est.fit(y, T, X=X, Z=Z, cache_values=True)
        est.featurizer = PolynomialFeatures(degree=2, include_bias=False)
        est.model_final = WeightedLasso()
        est.refit_final()
        assert isinstance(est.model_cate, WeightedLasso)
        assert isinstance(est.featurizer_, PolynomialFeatures)

        est = IntentToTreatDRIV(model_Y_X=LinearRegression(), model_T_XZ=LogisticRegression(),
                                flexible_model_effect=LinearRegression())
        est.fit(y, T, X=X, W=W, Z=Z, cache_values=True)
        assert est.model_final is None
        assert isinstance(est.model_final_, LinearRegression)
        est.flexible_model_effect = Lasso()
        est.refit_final()
        assert est.model_final is None
        assert isinstance(est.model_final_, Lasso)
        est.model_final = Lasso()
        est.refit_final()
        assert isinstance(est.model_final, Lasso)
        assert isinstance(est.model_final_, Lasso)
        assert isinstance(est.models_nuisance_[0]._prel_model_effect.model_final_, LinearRegression)
        est.fit(y, T, X=X, W=W, Z=Z, cache_values=True)
        assert isinstance(est.models_nuisance_[0]._prel_model_effect.model_final_, Lasso)

        est = LinearIntentToTreatDRIV(model_Y_X=LinearRegression(), model_T_XZ=LogisticRegression(),
                                      flexible_model_effect=LinearRegression())
        est.fit(y, T, X=X, W=W, Z=Z, cache_values=True)
        est.fit_cate_intercept = False
        est.intercept_
        est.intercept__interval()
        est.refit_final()
        with pytest.raises(AttributeError):
            est.intercept_
        with pytest.raises(AttributeError):
            est.intercept__interval()
        with pytest.raises(ValueError):
            est.model_final = LinearRegression()
        est.flexible_model_effect = Lasso()
        est.fit(y, T, X=X, W=W, Z=Z, cache_values=True)
        assert isinstance(est.models_nuisance_[0]._prel_model_effect.model_final_, Lasso)

    def test_can_set_discrete_treatment(self):
        X = np.random.choice(np.arange(5), size=(500, 3))
        y = np.random.normal(size=(500,))
        T = np.random.choice(np.arange(3), size=(500, 1))
        W = np.random.normal(size=(500, 2))
        est = LinearDML(model_y=RandomForestRegressor(),
                        model_t=RandomForestClassifier(min_samples_leaf=10),
                        discrete_treatment=True,
                        linear_first_stages=False,
                        cv=3)
        est.fit(y, T, X=X, W=W)
        est.effect(X)
        est.discrete_treatment = False
        est.fit(y, T, X=X, W=W)
        est.effect(X)

    def test_refit_final_inference(self):
        """Test that we can perform inference during refit_final"""
        est = LinearDML(linear_first_stages=False, featurizer=PolynomialFeatures(1, include_bias=False))

        X = np.random.choice(np.arange(5), size=(500, 3))
        y = np.random.normal(size=(500,))
        T = np.random.choice(np.arange(3), size=(500, 2))
        W = np.random.normal(size=(500, 2))

        est.fit(y, T, X=X, W=W, cache_values=True, inference='statsmodels')

        assert isinstance(est.effect_inference(X), NormalInferenceResults)

        with pytest.raises(ValueError):
            est.refit_final(inference=BootstrapInference(2))

    def test_rlearner_residuals(self):
        y, T, X, W = self._get_data()

        dml = DML(model_y=LinearRegression(),
                  model_t=LinearRegression(),
                  cv=1,
                  model_final=StatsModelsLinearRegression(fit_intercept=False),
                  linear_first_stages=False,
                  random_state=123)
        with pytest.raises(AttributeError):
            y_res, T_res, X_res, W_res = dml.residuals_
        dml.fit(y, T, X=X, W=W)
        with pytest.raises(AttributeError):
            y_res, T_res, X_res, W_res = dml.residuals_
        dml.fit(y, T, X=X, W=W, cache_values=True)
        y_res, T_res, X_res, W_res = dml.residuals_
        np.testing.assert_array_equal(X, X_res)
        np.testing.assert_array_equal(W, W_res)
        XW = np.hstack([X, W])
        np.testing.assert_array_equal(y_res, y - LinearRegression().fit(XW, y).predict(XW))
        np.testing.assert_array_equal(T_res, T - LinearRegression().fit(XW, T).predict(XW))
