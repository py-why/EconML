# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import unittest
import pytest
import graphviz
from econml.cate_interpreter import SingleTreeCateInterpreter, SingleTreePolicyInterpreter
from econml.dml import LinearDML

graphviz_works = True
try:
    from graphviz import Graph
    g = Graph()
    g.render()
except Exception:
    graphviz_works = False


@pytest.mark.skipif(not graphviz_works, reason="graphviz must be installed to run CATE interpreter tests")
class TestCateInterpreter(unittest.TestCase):

    # can't easily test output, but can at least test that we can all export_graphviz, render, and plot
    def test_can_use_interpreters(self):
        n = 100
        for t_shape in [(n,), (n, 1)]:
            for y_shape in [(n,), (n, 1)]:
                X = np.random.normal(size=(n, 4))
                T = np.random.binomial(1, 0.5, size=t_shape)
                Y = (T.flatten() * (2 * (X[:, 0] > 0) - 1)).reshape(y_shape)
                est = LinearDML(discrete_treatment=True)
                est.fit(Y, T, X=X)
                for intrp in [SingleTreeCateInterpreter(), SingleTreePolicyInterpreter()]:
                    with self.subTest(t_shape=t_shape, y_shape=y_shape, intrp=intrp):
                        with self.assertRaises(Exception):
                            # prior to calling interpret, can't plot, render, etc.
                            intrp.plot()
                        intrp.interpret(est, X)
                        intrp.plot()
                        intrp.render('tmp.pdf', view=False)
                        intrp.export_graphviz()

    @staticmethod
    def coinflip(p_true=0.5):
        return np.random.random_sample() < p_true

    def test_cate_uncertainty_needs_inference(self):
        n = 100
        X = np.random.normal(size=(n, 4))
        T = np.random.binomial(1, 0.5, size=(n,))
        Y = (2 * (X[:, 0] > 0) - 1) * T.flatten()
        est = LinearDML(discrete_treatment=True)
        est.fit(Y, T, X=X, inference=None)

        # can interpret without uncertainty
        intrp = SingleTreeCateInterpreter()
        intrp.interpret(est, X)

        intrp = SingleTreeCateInterpreter(include_model_uncertainty=True)
        with self.assertRaises(Exception):
            # can't interpret with uncertainty if inference wasn't used during fit
            intrp.interpret(est, X)

        # can interpret with uncertainty if we refit
        est.fit(Y, T, X=X)
        intrp.interpret(est, X)

    def test_can_assign_treatment(self):
        n = 100
        X = np.random.normal(size=(n, 4))
        T = np.random.binomial(1, 0.5, size=(n,))
        Y = (2 * (X[:, 0] > 0) - 1) * T.flatten()
        est = LinearDML(discrete_treatment=True)
        est.fit(Y, T, X=X)

        # can interpret without uncertainty
        intrp = SingleTreePolicyInterpreter()
        with self.assertRaises(Exception):
            # can't treat before interpreting
            intrp.treat(X)

        intrp.interpret(est, X)
        T_policy = intrp.treat(X)
        assert T.shape == T_policy.shape

    def test_random_cate_settings(self):
        """Verify that we can call methods on the CATE interpreter with various combinations of inputs"""
        n = 100
        for _ in range(100):
            t_shape = (n,) if self.coinflip else (n, 1)
            y_shape = (n,) if self.coinflip else (n, 1)
            discrete_t = self.coinflip()
            X = np.random.normal(size=(n, 4))
            X2 = np.random.normal(size=(10, 4))
            T = np.random.binomial(1, 0.5, size=t_shape) if discrete_t else np.random.normal(size=t_shape)
            Y = (T.flatten() * (2 * (X[:, 0] > 0) - 1)).reshape(y_shape)

            est = LinearDML(discrete_treatment=discrete_t)

            fit_kwargs = {}
            cate_init_kwargs = {}
            policy_init_kwargs = {}
            intrp_kwargs = {}
            policy_intrp_kwargs = {}
            common_kwargs = {}
            plot_kwargs = {}
            render_kwargs = {}
            export_kwargs = {}

            if self.coinflip():
                cate_init_kwargs.update(include_model_uncertainty=True)
                policy_init_kwargs.update(risk_level=0.1)
            else:
                fit_kwargs.update(inference=None)

            if self.coinflip():
                cate_init_kwargs.update(uncertainty_level=0.01)

            if self.coinflip():
                policy_init_kwargs.update(risk_seeking=True)

            if self.coinflip():
                policy_intrp_kwargs.update(treatment_names=['control gp', 'treated gp'])

            if self.coinflip(1 / 3):
                policy_intrp_kwargs.update(sample_treatment_costs=0.1)
            elif self.coinflip():
                policy_intrp_kwargs.update(sample_treatment_costs=np.random.normal(size=(10,)))

            if self.coinflip():
                common_kwargs.update(feature_names=['A', 'B', 'C', 'D'])

            if self.coinflip():
                common_kwargs.update(filled=False)

            if self.coinflip():
                common_kwargs.update(rounded=False)

            if self.coinflip():
                common_kwargs.update(precision=1)

            if self.coinflip():
                render_kwargs.update(rotate=True)
                export_kwargs.update(rotate=True)

            if self.coinflip():
                render_kwargs.update(leaves_parallel=False)
                export_kwargs.update(leaves_parallel=False)

            if self.coinflip():
                render_kwargs.update(format='png')

            if self.coinflip():
                export_kwargs.update(out_file='out')

            if self.coinflip(0.95):  # don't launch files most of the time
                render_kwargs.update(view=False)

            with self.subTest(t_shape=t_shape,
                              y_shape=y_shape,
                              discrete_t=discrete_t,
                              fit_kwargs=fit_kwargs,
                              cate_init_kwargs=cate_init_kwargs,
                              policy_init_kwargs=policy_init_kwargs,
                              policy_intrp_kwargs=policy_intrp_kwargs,
                              intrp_kwargs=intrp_kwargs,
                              common_kwargs=common_kwargs,
                              plot_kwargs=plot_kwargs,
                              render_kwargs=render_kwargs,
                              export_kwargs=export_kwargs):
                plot_kwargs.update(common_kwargs)
                render_kwargs.update(common_kwargs)
                export_kwargs.update(common_kwargs)
                policy_intrp_kwargs.update(intrp_kwargs)

                est.fit(Y, T, X=X, **fit_kwargs)

                intrp = SingleTreeCateInterpreter(**cate_init_kwargs)
                intrp.interpret(est, X2, **intrp_kwargs)
                intrp.plot(**plot_kwargs)
                intrp.render('outfile', **render_kwargs)
                intrp.export_graphviz(**export_kwargs)

                intrp = SingleTreePolicyInterpreter(**policy_init_kwargs)
                try:
                    intrp.interpret(est, X2, **policy_intrp_kwargs)
                    intrp.plot(**plot_kwargs)
                    intrp.render('outfile', **render_kwargs)
                    intrp.export_graphviz(**export_kwargs)
                except AttributeError as e:
                    assert str(e).find("samples should") >= 0
