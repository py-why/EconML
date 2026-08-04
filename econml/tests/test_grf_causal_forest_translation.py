import unittest

import numpy as np

from econml.grf import GRFCausalForest, causal_forest


class TestGRFCausalForestTranslation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(123)
        n = 240
        X = rng.normal(size=(n, 4))
        W = rng.binomial(1, 1 / (1 + np.exp(-X[:, 0])))
        tau = 0.5 + X[:, 0]
        Y = tau * W + 0.3 * X[:, 1] + rng.normal(size=n)

        cls.X = X
        cls.W = W
        cls.Y = Y

    def _make_est(self, **kwargs):
        return GRFCausalForest(
            num_trees=52,
            seed=123,
            **kwargs,
        )

    def test_class_api_fit_predict(self):
        est = self._make_est()
        est.fit(self.X, self.Y, self.W)
        pred = est.predict(self.X[:12]).predictions
        self.assertEqual(pred.shape, (12,))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_function_api_matches_original_shape(self):
        forest = causal_forest(self.X, self.Y, self.W, num_trees=52, seed=123)
        pred = forest.predict(self.X[:15]).predictions
        self.assertEqual(pred.shape, (15,))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_predict_without_newdata_returns_oos_training_predictions(self):
        est = self._make_est()
        est.fit(self.X, self.Y, self.W)
        pred = est.predict().predictions
        self.assertEqual(pred.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(pred)))
        np.testing.assert_allclose(pred, est.effect())

    def test_nuisances_are_stored(self):
        est = self._make_est()
        est.fit(self.X, self.Y, self.W)
        self.assertEqual(est.Y_hat_.shape, (self.X.shape[0],))
        self.assertEqual(est.W_hat_.shape, (self.X.shape[0],))
        self.assertTrue(np.all(np.isfinite(est.Y_hat_)))
        self.assertTrue(np.all(np.isfinite(est.W_hat_)))

    def test_user_supplied_nuisances_are_used(self):
        y_hat = np.repeat(np.mean(self.Y), self.X.shape[0])
        w_hat = np.repeat(np.mean(self.W), self.X.shape[0])
        est = self._make_est()
        est.fit(self.X, self.Y, self.W, Y_hat=y_hat, W_hat=w_hat)
        np.testing.assert_allclose(est.Y_hat_, y_hat)
        np.testing.assert_allclose(est.W_hat_, w_hat)

    def test_unsupported_honesty_fraction_raises(self):
        est = self._make_est(honesty_fraction=0.75)
        with self.assertRaises(NotImplementedError):
            est.fit(self.X, self.Y, self.W)


if __name__ == "__main__":
    unittest.main()
