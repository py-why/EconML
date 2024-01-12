# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.
import pytest
import unittest
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

from econml._ortho_learner import _OrthoLearner
from econml.dml import LinearDML, SparseLinearDML, KernelDML, CausalForestDML, NonParamDML
from econml.iv.dml import OrthoIV, DMLIV, NonParamDMLIV
from econml.iv.dr import DRIV, LinearDRIV, SparseLinearDRIV, ForestDRIV
from econml.orf import DMLOrthoForest
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression, WeightedLassoCVWrapper

from econml.utilities import jacify_featurizer
from econml.iv.sieve import DPolynomialFeatures
from copy import deepcopy


class DGP():
    def __init__(self,
                 n=1000,
                 d_t=1,
                 d_y=1,
                 d_x=5,
                 d_z=None,
                 squeeze_T=False,
                 squeeze_Y=False,
                 nuisance_Y=None,
                 nuisance_T=None,
                 nuisance_TZ=None,
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
        self.d_z = d_z

        self.squeeze_T = squeeze_T
        self.squeeze_Y = squeeze_Y

        self.nuisance_Y = nuisance_Y if nuisance_Y else lambda X: 0
        self.nuisance_T = nuisance_T if nuisance_T else lambda X: 0
        self.nuisance_TZ = nuisance_TZ if nuisance_TZ else lambda X: 0
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
        self.T = noise + self.nuisance_T(self.X) + self.nuisance_TZ(self.Z)
        return self.T

    def gen_Z(self):
        if self.d_z:
            Z_noise = np.random.normal(size=(self.n, self.d_z), loc=3, scale=3)
            self.Z = Z_noise
            return self.Z

        else:
            self.Z = None
            return self.Z

    def gen_data(self):
        X = self.gen_X()
        Z = self.gen_Z()
        T = self.gen_T()
        Y = self.gen_Y()

        if self.squeeze_T:
            T = T.squeeze()
        if self.squeeze_Y:
            Y = Y.squeeze()

        data_dict = {
            'Y': Y,
            'T': T,
            'X': X
        }

        if self.d_z:
            data_dict['Z'] = Z

        return data_dict


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


identity_treatment_featurizer = FunctionTransformer()


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


# 1d polynomial featurization functions
def poly_1d_actual_cme():
    return 0.5


def poly_1d_func_transform(x):
    return x**2


polynomial_1d_treatment_featurizer = FunctionTransformer(func=poly_1d_func_transform)


# 2d-to-1d featurization functions

def sum_y_of_t(T):
    return 0.5 * T.sum(axis=1, keepdims=True)


def sum_actual_cme():
    return 0.5


def sum_actual_marginal(t):
    return np.ones(shape=t.shape) * 0.5


def sum_func_transform(x):
    return x.sum(axis=1, keepdims=True)


sum_treatment_featurizer = FunctionTransformer(func=sum_func_transform)


# 2d-to-1d vector featurization functions
def sum_squeeze_func_transform(x):
    return x.sum(axis=1, keepdims=False)


sum_squeeze_treatment_featurizer = FunctionTransformer(func=sum_squeeze_func_transform)


@pytest.mark.treatment_featurization
class TestTreatmentFeaturization(unittest.TestCase):

    def test_featurization(self):
        # use LassoCV rather than also selecting over RandomForests to save time
        dml_models = {
            "model_t": WeightedLassoCVWrapper(),
            "model_y": WeightedLassoCVWrapper()
        }

        dmliv_models = {
            "model_y_xw": WeightedLassoCVWrapper(),
            "model_t_xw": WeightedLassoCVWrapper(),
            "model_t_xwz": WeightedLassoCVWrapper(),
        }

        driv_models = {
            "model_y_xw": WeightedLassoCVWrapper(),
            "model_t_xw": WeightedLassoCVWrapper(),
            "model_z_xw": WeightedLassoCVWrapper(),
            "model_tz_xw": WeightedLassoCVWrapper(),
        }

        identity_config = {
            'DGP_params': {
                'n': 2000,
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

            'treatment_featurizer': identity_treatment_featurizer,
            'actual_marginal': identity_actual_marginal,
            'actual_cme': identity_actual_cme,
            'squeeze_Ts': [False, True],
            'squeeze_Ys': [False, True],
            'est_dicts': [
                {'class': LinearDML, 'init_args': dml_models},
                {'class': CausalForestDML, 'init_args': dml_models},
                {'class': SparseLinearDML, 'init_args': dml_models},
                {'class': KernelDML, 'init_args': dml_models},
            ]
        }

        poly_config = {
            'DGP_params': {
                'n': 2000,
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

            'treatment_featurizer': polynomial_treatment_featurizer,
            'actual_marginal': poly_actual_marginal,
            'actual_cme': poly_actual_cme,
            'squeeze_Ts': [False, True],
            'squeeze_Ys': [False, True],
            'est_dicts': [
                {'class': LinearDML, 'init_args': dml_models},
                {'class': CausalForestDML, 'init_args': dml_models},
                {'class': SparseLinearDML, 'init_args': dml_models},
                {'class': KernelDML, 'init_args': dml_models},
            ]
        }

        poly_config_scikit = deepcopy(poly_config)
        poly_config_scikit['treatment_featurizer'] = PolynomialFeatures(degree=2, include_bias=False)
        poly_config_scikit['squeeze_Ts'] = [False]

        poly_IV_config = deepcopy(poly_config)
        poly_IV_config['DGP_params']['d_z'] = 1
        poly_IV_config['DGP_params']['nuisance_TZ'] = lambda Z: Z
        poly_IV_config['est_dicts'] = [
            {'class': OrthoIV, 'init_args': {**dmliv_models,
                                             'model_t_xwz': RandomForestRegressor(random_state=1),
                                             'projection': True}},
            {'class': DMLIV, 'init_args': {**dmliv_models,
                                           'model_t_xwz': RandomForestRegressor(random_state=1)}},
        ]

        poly_1d_config = deepcopy(poly_config)
        poly_1d_config['treatment_featurizer'] = polynomial_1d_treatment_featurizer
        poly_1d_config['actual_cme'] = poly_1d_actual_cme
        poly_1d_config['est_dicts'].append({
            'class': NonParamDML,
            'init_args': {
                'model_y': LinearRegression(),
                'model_t': LinearRegression(),
                'model_final': StatsModelsLinearRegression()}})

        poly_1d_IV_config = deepcopy(poly_IV_config)
        poly_1d_IV_config['treatment_featurizer'] = polynomial_1d_treatment_featurizer
        poly_1d_IV_config['actual_cme'] = poly_1d_actual_cme
        poly_1d_IV_config['est_dicts'] = [
            {'class': NonParamDMLIV, 'init_args': {**dmliv_models,
                                                   'model_final': StatsModelsLinearRegression()}},
            {'class': DRIV, 'init_args': {**driv_models,
                                          'fit_cate_intercept': True}},
            {'class': LinearDRIV, 'init_args': driv_models},
            {'class': SparseLinearDRIV, 'init_args': driv_models},
            {'class': ForestDRIV, 'init_args': driv_models},
        ]

        sum_IV_config = {
            'DGP_params': {
                'n': 2000,
                'd_t': 2,
                'd_y': 1,
                'd_x': 5,
                'd_z': 1,
                'squeeze_T': False,
                'squeeze_Y': False,
                'nuisance_Y': nuisance_Y,
                'nuisance_T': nuisance_T,
                'nuisance_TZ': lambda Z: Z,
                'theta': None,
                'y_of_t': sum_y_of_t,
                'x_eps': 1,
                'y_eps': 1,
                't_eps': 1
            },

            'treatment_featurizer': sum_treatment_featurizer,
            'actual_marginal': sum_actual_marginal,
            'actual_cme': sum_actual_cme,
            'squeeze_Ts': [False],
            'squeeze_Ys': [False, True],
            'est_dicts': [
                {'class': NonParamDMLIV, 'init_args': {**dmliv_models,
                                                       'model_final': StatsModelsLinearRegression()}},
                {'class': DRIV, 'init_args': {**driv_models,
                                              'fit_cate_intercept': True}},
                {'class': LinearDRIV, 'init_args': driv_models},
                {'class': SparseLinearDRIV, 'init_args': driv_models},
                {'class': ForestDRIV, 'init_args': driv_models},
            ]
        }

        sum_squeeze_IV_config = deepcopy(sum_IV_config)
        sum_squeeze_IV_config['treatment_featurizer'] = sum_squeeze_treatment_featurizer

        sum_config = deepcopy(sum_IV_config)
        sum_config['DGP_params']['d_z'] = None
        sum_config['DGP_params']['nuisance_TZ'] = None
        sum_config['est_dicts'] = deepcopy(poly_1d_config['est_dicts'])

        sum_squeeze_config = deepcopy(sum_config)
        sum_squeeze_config['treatment_featurizer'] = sum_squeeze_treatment_featurizer

        configs = [
            identity_config,
            poly_config,
            poly_config_scikit,
            poly_IV_config,
            poly_1d_config,
            poly_1d_IV_config,
            sum_IV_config,
            sum_squeeze_IV_config,
            sum_config,
            sum_squeeze_config
        ]

        for config in configs:
            for squeeze_Y in config['squeeze_Ys']:
                for squeeze_T in config['squeeze_Ts']:
                    config['DGP_params']['squeeze_Y'] = squeeze_Y
                    config['DGP_params']['squeeze_T'] = squeeze_T
                    dgp = DGP(**config['DGP_params'])
                    data_dict = dgp.gen_data()
                    Y = data_dict['Y']
                    T = data_dict['T']
                    X = data_dict['X']
                    feat_T = config['treatment_featurizer'].fit_transform(T)

                    data_dict_outside_feat = deepcopy(data_dict)
                    data_dict_outside_feat['T'] = feat_T

                    est_dicts = config['est_dicts']

                    for est_dict in est_dicts:
                        estClass = est_dict['class']
                        init_args = deepcopy(est_dict['init_args'])
                        init_args['treatment_featurizer'] = config['treatment_featurizer']
                        init_args['random_state'] = 1

                        est = estClass(**init_args)
                        est.fit(**data_dict)

                        init_args_outside_feat = deepcopy(est_dict['init_args'])
                        init_args_outside_feat['random_state'] = 1
                        est_outside_feat = estClass(**init_args_outside_feat)
                        est_outside_feat.fit(**data_dict_outside_feat)

                        #  test that treatment names are assigned for the featurized treatment
                        assert (est.cate_treatment_names() is not None)

                        if hasattr(est, 'summary'):
                            est.summary()

                        # expected shapes
                        expected_eff_shape = (config['DGP_params']['n'],) + Y.shape[1:]
                        expected_cme_shape = (config['DGP_params']['n'],) + Y.shape[1:] + feat_T.shape[1:]
                        expected_me_shape = (config['DGP_params']['n'],) + Y.shape[1:] + T.shape[1:]
                        expected_marginal_ate_shape = expected_me_shape[1:]

                        # check effects
                        T0 = np.ones(shape=T.shape) * 5
                        T1 = np.ones(shape=T.shape) * 10
                        eff = est.effect(X=X, T0=T0, T1=T1)
                        assert (eff.shape == expected_eff_shape)
                        outside_feat = config['treatment_featurizer']
                        eff_outside_feat = est_outside_feat.effect(
                            X=X, T0=outside_feat.fit_transform(T0), T1=outside_feat.fit_transform(T1))
                        np.testing.assert_almost_equal(eff, eff_outside_feat)
                        actual_eff = actual_effect(config['DGP_params']['y_of_t'], T0, T1)

                        cme = est.const_marginal_effect(X=X)
                        assert (cme.shape == expected_cme_shape)
                        cme_outside_feat = est_outside_feat.const_marginal_effect(X=X)
                        np.testing.assert_almost_equal(cme, cme_outside_feat)
                        actual_cme = config['actual_cme']()

                        me = est.marginal_effect(T=T, X=X)
                        assert (me.shape == expected_me_shape)
                        actual_me = config['actual_marginal'](T).reshape(me.shape)

                        # ate
                        m_ate = est.marginal_ate(T, X=X)
                        assert (m_ate.shape == expected_marginal_ate_shape)

                        if isinstance(est, (LinearDML, SparseLinearDML, LinearDRIV, SparseLinearDRIV)):
                            d_f_t = feat_T.shape[1] if feat_T.shape[1:] else 1
                            expected_coef_inference_shape = (
                                config['DGP_params']['d_y'] * config['DGP_params']['d_x'] * d_f_t, 6)
                            assert est.coef__inference().summary_frame().shape == expected_coef_inference_shape

                            expected_intercept_inf_shape = (
                                config['DGP_params']['d_y'] * d_f_t, 6)
                            assert est.intercept__inference().summary_frame().shape == expected_intercept_inf_shape

                        # loose inference checks
                        # temporarily skip LinearDRIV and SparseLinearDRIV for weird effect shape reasons
                        if isinstance(est, (KernelDML, LinearDRIV, SparseLinearDRIV)):
                            continue

                        if est._inference is None:
                            continue

                        # effect inference
                        eff_inf = est.effect_inference(X=X, T0=T0, T1=T1)
                        eff_lb, eff_ub = eff_inf.conf_int(alpha=0.01)
                        assert (eff.shape == eff_lb.shape)
                        proportion_in_interval = ((eff_lb < actual_eff) & (actual_eff < eff_ub)).mean()
                        np.testing.assert_array_less(0.50, proportion_in_interval)
                        np.testing.assert_almost_equal(eff, eff_inf.point_estimate)

                        # marginal effect inference
                        me_inf = est.marginal_effect_inference(T, X=X)
                        me_lb, me_ub = me_inf.conf_int(alpha=0.01)
                        assert (me.shape == me_lb.shape)
                        proportion_in_interval = ((me_lb < actual_me) & (actual_me < me_ub)).mean()
                        np.testing.assert_array_less(0.50, proportion_in_interval)
                        np.testing.assert_almost_equal(me, me_inf.point_estimate)

                        # const marginal effect inference
                        cme_inf = est.const_marginal_effect_inference(X=X)
                        cme_lb, cme_ub = cme_inf.conf_int(alpha=0.01)
                        assert (cme.shape == cme_lb.shape)
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
            # fit a dummy estimator first so the featurizer can be fit to the treatment
            dummy_est = LinearDML(treatment_featurizer=treatment_featurizer)
            dummy_est.fit(Y=T, T=T, X=T)
            expected_jac = calc_expected_jacobian(T)
            jac_T = dummy_est.transformer.jac(T)
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
                    'discrete_outcome': False,
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

        dummy_vec = np.random.normal(size=(100, 1))

        for est_and_param in est_and_params:
            params = est_and_param['params']
            params['discrete_treatment'] = True
            params['treatment_featurizer'] = True
            est = est_and_param['estimator'](**params)
            with self.assertRaises(AssertionError, msg='Estimator fit did not fail when passed '
                                   'both discrete treatment and treatment featurizer'):
                est.fit(Y=dummy_vec, T=dummy_vec, X=dummy_vec)

    def test_cate_treatment_names_edge_cases(self):
        Y = np.random.normal(size=(100, 1))
        T = np.random.binomial(n=2, p=0.5, size=(100, 1))
        X = np.random.normal(size=(100, 3))

        # edge case with transformer that only takes a vector treatment
        # so far will always return None for cate_treatment_names
        def weird_func(x):
            assert np.ndim(x) == 1
            return x
        est = LinearDML(treatment_featurizer=FunctionTransformer(weird_func)).fit(Y=Y, T=T.squeeze(), X=X)
        assert est.cate_treatment_names() is None
        assert est.cate_treatment_names(['too', 'many', 'feature_names']) is None

        # assert proper handling of improper feature names passed to certain transformers
        est = LinearDML(discrete_treatment=True).fit(Y=Y, T=T, X=X)
        assert est.cate_treatment_names() == ['T0_1', 'T0_2']
        assert est.cate_treatment_names(['too', 'many', 'feature_names']) is None

        est = LinearDML(treatment_featurizer=PolynomialFeatures(degree=2, include_bias=False)).fit(Y=Y, T=T, X=X)
        assert est.cate_treatment_names() == ['T0', 'T0^2']
        # depending on sklearn version, bad feature names either throws error or only uses first relevant name
        assert est.cate_treatment_names(['too', 'many', 'feature_names']) in [None, ['too', 'too^2']]

    def test_alpha_passthrough(self):
        X = np.random.normal(size=(100, 3))
        T = np.random.normal(size=(100, 1)) + X[:, [0]]
        Y = np.random.normal(size=(100, 1)) + T + X[:, [0]]

        est = LinearDML(model_y=LinearRegression(), model_t=LinearRegression(),
                        treatment_featurizer=FunctionTransformer())
        est.fit(Y=Y, T=T, X=X)

        # ensure alpha is passed
        lb, ub = est.marginal_effect_interval(T, X, alpha=1)
        assert (lb == ub).all()

        lb, ub = est.marginal_effect_interval(T, X)
        assert (lb != ub).all()

        lb1, ub1 = est.marginal_effect_interval(T, X, alpha=0.01)
        lb2, ub2 = est.marginal_effect_interval(T, X, alpha=0.1)

        assert (lb1 < lb2).all() and (ub1 > ub2).all()

    def test_identity_feat_with_cate_api(self):
        from .test_dml import TestDML
        treatment_featurizations = [FunctionTransformer()]
        TestDML()._test_cate_api(treatment_featurizations)
