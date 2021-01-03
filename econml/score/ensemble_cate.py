import numpy as np


class EnsembleCateEstimator:

    def __init__(self, *, cate_models, weights):
        self.cate_models = cate_models
        self.weights = weights

    def effect(self, X=None, *, T0=0, T1=1):
        return np.average([mdl.effect(X=X, T0=T0, T1=T1) for mdl in self.cate_models],
                          weights=self.weights, axis=0)

    def marginal_effect(self, T, X=None):
        return np.average([mdl.marginal_effect(T, X=X) for mdl in self.cate_models],
                          weights=self.weights, axis=0)

    def const_marginal_effect(self, X=None):
        return np.average([mdl.const_marginal_effect(X=X) for mdl in self.cate_models],
                          weights=self.weights, axis=0)
