import numpy as np
import scipy
from econml.dml import LinearDML
from econml.inference import BootstrapInference

np.random.seed(123)
X = np.random.normal(size=(1000, 5))
T = np.random.binomial(1, scipy.special.expit(X[:, 0]))
y = (1 + .5*X[:, 0]) * T + X[:, 0] + np.random.normal(size=(1000,))
est = LinearDML(discrete_treatment=True)
est.fit(y, T, X=X, W=None, inference=BootstrapInference(10))

# from sklearn.linear_model import LinearRegression
# from econml._ortho_learner import _OrthoLearner
# class ModelNuisance:
#     def __init__(self, model_t, model_y):
#         self._model_t = model_t
#         self._model_y = model_y
#     def fit(self, Y, T, W=None):
#         self._model_t.fit(W, T)
#         self._model_y.fit(W, Y)
#         return self
#     def predict(self, Y, T, W=None):
#         return Y - self._model_y.predict(W), T - self._model_t.predict(W)
# class ModelFinal:
#     def __init__(self):
#         return
#     def fit(self, Y, T, W=None, nuisances=None):
#         Y_res, T_res = nuisances
#         self.model = LinearRegression(fit_intercept=False).fit(T_res.reshape(-1, 1), Y_res)
#         return self
#     def predict(self, X=None):
#         return self.model.coef_[0]
#     def score(self, Y, T, W=None, nuisances=None):
#         Y_res, T_res = nuisances
#         return np.mean((Y_res - self.model.predict(T_res.reshape(-1, 1)))**2)
# class OrthoLearner(_OrthoLearner):
#     def _gen_ortho_learner_model_nuisance(self):
#         return ModelNuisance(LinearRegression(), LinearRegression())
#     def _gen_ortho_learner_model_final(self):
#         return ModelFinal()
# np.random.seed(123)
# X = np.random.normal(size=(100, 3))
# y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, size=(100,))
# est = OrthoLearner(cv=2, discrete_treatment=False, treatment_featurizer=None,
#                    discrete_instrument=False, categories='auto', random_state=None)
# est.fit(y, X[:, 0], W=X[:, 1:])
