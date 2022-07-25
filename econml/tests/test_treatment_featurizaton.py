# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pytest
import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML, SparseLinearDML, KernelDML, CausalForestDML
from econml.orf import DMLOrthoForest
# from econml.metalearners import XLearner, TLearner, SLearner, DomainAdaptationLearner
# from econml.dr import DRLearner
# from econml.score import RScorer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

from econml.utilities import jacify_featurizer
from econml.iv.sieve import DPolynomialFeatures


class DGP():
    def __init__(self,
                 n=1000,
                 d_t=1,
                 d_y=1,
                 d_x=5,
                 squeeze_T=False,
                 squeeze_Y=False,
                 nuisance_Y=None,
                 nuisance_T=None,
                 theta=None,
                 y_of_t=None,
                 x_eps=1,
                 y_eps=1,
                 t_eps=1
                 ):
        self.n = n
        self.d_t = d_t
        self.d_y = d_y
        self.d_x = d_x

        self.squeeze_T = squeeze_T
        self.squeeze_Y = squeeze_Y

        self.nuisance_Y = nuisance_Y if nuisance_Y else lambda X: 0
        self.nuisance_T = nuisance_T if nuisance_T else lambda X: 0
        self.theta = theta if theta else lambda X: 1
        self.y_of_t = y_of_t if y_of_t else lambda X: 0

        self.x_eps = x_eps
        self.y_eps = y_eps
        self.t_eps = t_eps

    def gen_Y(self):
        noise = np.random.normal(size=(self.n, self.d_y), scale=self.y_eps)
        self.Y = self.theta(self.X) * self.y_of_t(self.T) + self.nuisance_Y(self.X) + noise
        return self.Y

    def gen_X(self):
        self.X = np.random.normal(size=(self.n, self.d_x), scale=self.x_eps)
        return self.X

    def gen_T(self):
        noise = np.random.normal(size=(self.n, self.d_t), scale=self.t_eps)
        self.T_noise = noise
        self.T = noise + self.nuisance_T(self.X)
        return self.T

    def gen_data(self):
        X = self.gen_X()
        T = self.gen_T()
        Y = self.gen_Y()

        if self.squeeze_T:
            T = T.squeeze()
        if self.squeeze_Y:
            Y = Y.squeeze()

        return Y, T, X


def actual_effect(y_of_t, T0, T1):
    return y_of_t(T1) - y_of_t(T0)


def nuisance_T(X):
    return -0.3 * X[:, [1]]


def nuisance_Y(X):
    return 0.2 * X[:, [0]]


# identity featurization effect functions
def identity_y_of_t(T):
    return T


def identity_actual_marginal(T):
    return np.ones(shape=(T.shape))


def identity_actual_cme():
    return 1


identity_treatment_featurizer = FunctionTransformer(func=lambda x: x)


# polynomial featurization effect functions
def poly_y_of_t(T):
    return 0.5 * T**2


def poly_actual_marginal(t):
    return t


def poly_actual_cme():
    return np.array([0, 0.5])


def poly_func_transform(x):
    x = x.reshape(-1, 1)
    return np.hstack([x, x**2])


polynomial_treatment_featurizer = FunctionTransformer(func=poly_func_transform)


@pytest.mark.dml
class TestTreatmentFeaturization(unittest.TestCase):

    def test_featurization(self):
        identity_config = {
            'DGP_params': {
                'n': 10000,
                'd_t': 1,
                'd_y': 1,
                'd_x': 5,
                'squeeze_T': False,
                'squeeze_Y': False,
                'nuisance_Y': nuisance_Y,
                'nuisance_T': nuisance_T,
                'theta': None,
                'y_of_t': identity_y_of_t,
                'x_eps': 1,
                'y_eps': 1,
                't_eps': 1
            },

            'treatment_featurizer': FunctionTransformer(lambda x: x),
            'actual_marginal': identity_actual_marginal,
            'actual_cme': identity_actual_cme,
            'squeeze_Ts': [False, True],
            'squeeze_Ys': [False, True]
        }

        poly_config = {
            'DGP_params': {
                'n': 10000,
                'd_t': 1,
                'd_y': 1,
                'd_x': 5,
                'squeeze_T': False,
                'squeeze_Y': False,
                'nuisance_Y': nuisance_Y,
                'nuisance_T': nuisance_T,
                'theta': None,
                'y_of_t': poly_y_of_t,
                'x_eps': 1,
                'y_eps': 1,
                't_eps': 1
            },

            'treatment_featurizer': FunctionTransformer(func=poly_func_transform),
            'actual_marginal': poly_actual_marginal,
            'actual_cme': poly_actual_cme,
            'squeeze_Ts': [False, True],
            'squeeze_Ys': [False, True]
        }

        poly_config_scikit = poly_config.copy()
        poly_config['treatment_featurizer'] = PolynomialFeatures(degree=2, include_bias=False)
        poly_config['squeeze_Ts'] = [False]

        configs = [
            identity_config,
            poly_config,
            poly_config_scikit
        ]
        estimators = [
            LinearDML,
            CausalForestDML,
            SparseLinearDML,
            KernelDML
        ]

        for config in configs:
            for squeeze_Y in config['squeeze_Ys']:
                for squeeze_T in config['squeeze_Ts']:
                    config['DGP_params']['squeeze_Y'] = squeeze_Y
                    config['DGP_params']['squeeze_T'] = squeeze_T
                    dgp = DGP(**config['DGP_params'])
                    Y, T, X = dgp.gen_data()
                    feat_T = config['treatment_featurizer'].fit_transform(T)

                    for Est in estimators:
                        est = Est(treatment_featurizer=config['treatment_featurizer'])
                        est.fit(Y, T=T, X=X)

                        #  test that treatment names are assigned for the featurized treatment
                        assert(est.cate_treatment_names() is not None)

                        if hasattr(est, 'summary'):
                            est.summary()

                        # expected shapes
                        expected_eff_shape = (config['DGP_params']['n'],) + Y.shape[1:]
                        expected_cme_shape = (config['DGP_params']['n'],) + Y.shape[1:] + feat_T.shape[1:]
                        expected_me_shape = (config['DGP_params']['n'],) + Y.shape[1:] + T.shape[1:]
                        expected_marginal_ate_shape = expected_me_shape[1:]

                        # check effects
                        eff = est.effect(X=X, T0=5, T1=10)
                        assert(eff.shape == expected_eff_shape)
                        actual_eff = actual_effect(config['DGP_params']['y_of_t'], 5, 10)

                        cme = est.const_marginal_effect(X=X)
                        assert(cme.shape == expected_cme_shape)
                        actual_cme = config['actual_cme']()

                        me = est.marginal_effect(T=T, X=X)
                        assert(me.shape == expected_me_shape)
                        actual_me = config['actual_marginal'](T).reshape(me.shape)

                        # ate
                        m_ate = est.marginal_ate(T, X=X)
                        assert(m_ate.shape == expected_marginal_ate_shape)

                        # loose inference checks
                        if isinstance(est, KernelDML):
                            continue

                        # effect inference
                        eff_inf = est.effect_inference(X=X, T0=5, T1=10)
                        eff_lb, eff_ub = eff_inf.conf_int(alpha=0.01)
                        assert(eff.shape == eff_lb.shape)
                        proportion_in_interval = ((eff_lb < actual_eff) & (actual_eff < eff_ub)).mean()
                        np.testing.assert_array_less(0.50, proportion_in_interval)
                        np.testing.assert_almost_equal(eff, eff_inf.point_estimate)

                        # marginal effect inference
                        me_inf = est.marginal_effect_inference(T, X=X)
                        me_lb, me_ub = me_inf.conf_int(alpha=0.01)
                        assert(me.shape == me_lb.shape)
                        proportion_in_interval = ((me_lb < actual_me) & (actual_me < me_ub)).mean()
                        np.testing.assert_array_less(0.50, proportion_in_interval)
                        np.testing.assert_almost_equal(me, me_inf.point_estimate)

                        # const marginal effect inference
                        cme_inf = est.const_marginal_effect_inference(X=X)
                        cme_lb, cme_ub = cme_inf.conf_int(alpha=0.01)
                        assert(cme.shape == cme_lb.shape)
                        proportion_in_interval = ((cme_lb < actual_cme) & (actual_cme < cme_ub)).mean()
                        np.testing.assert_array_less(0.50, proportion_in_interval)
                        np.testing.assert_almost_equal(cme, cme_inf.point_estimate)

    def test_jac(self):
        def func_transform(x):
            x = x.reshape(-1, 1)
            return np.hstack([x, x**2])

        def calc_expected_jacobian(T):
            jac = DPolynomialFeatures(degree=2, include_bias=False).fit_transform(T)
            return jac

        treatment_featurizers = [
            PolynomialFeatures(degree=2, include_bias=False),
            FunctionTransformer(func=func_transform)
        ]

        n = 10000
        d_t = 1
        T = np.random.normal(size=(n, d_t))

        for treatment_featurizer in treatment_featurizers:
            expected_jac = calc_expected_jacobian(T)
            jacified_featurizer = jacify_featurizer(treatment_featurizer)
            jac_T = jacified_featurizer.jac(T)
            np.testing.assert_almost_equal(jac_T, expected_jac)

    def test_fail_discrete_treatment_and_treatment_featurizer(self):
        class OrthoLearner(_OrthoLearner):
            def _gen_ortho_learner_model_nuisance(self):
                pass

            def _gen_ortho_learner_model_final(self):
                pass

        est_and_params = [
            {
                'estimator': OrthoLearner,
                'params': {
                    'cv': 2,
                    'discrete_treatment': False,
                    'treatment_featurizer': None,
                    'discrete_instrument': False,
                    'categories': 'auto',
                    'random_state': None
                }
            },
            {'estimator': LinearDML, 'params': {}},
            {'estimator': CausalForestDML, 'params': {}},
            {'estimator': SparseLinearDML, 'params': {}},
            {'estimator': KernelDML, 'params': {}},
            {'estimator': DMLOrthoForest, 'params': {}}

        ]

        for est_and_param in est_and_params:
            try:
                params = est_and_param['params']
                params['discrete_treatment'] = True
                params['treatment_featurizer'] = True
                est_and_param['estimator'](**params)
                raise ValueError(
                    'Estimator initializaiton did not fail when passed '
                    'both discrete treatment and treatment featurizer')
            except AssertionError:
                pass
