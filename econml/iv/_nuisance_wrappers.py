# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn.base import clone
from ..utilities import (add_intercept, fit_with_groups,
                         hstack, inverse_onehot)
from ..dml.dml import _FinalWrapper as _DMLFinalWrapper

# A cut-down version of the DML first stage wrapper, since we don't need to support linear first stages


class _FirstStageWrapper:
    def __init__(self, model, discrete_target):
        self._model = clone(model, safe=False)
        self._discrete_target = discrete_target

    def _combine(self, X, W, Z, n_samples, fitting=True):
        # output is
        #   * a column of ones if X, W, and Z are all None
        #   * just X or W or Z if both of the others are None
        #   * hstack([arrs]) for whatever subset are not None otherwise

        # ensure Z is 2D
        if Z is not None:
            Z = Z.reshape(n_samples, -1)

        if X is None and W is None and Z is None:
            return np.ones((n_samples, 1))

        arrs = [arr for arr in [X, W, Z] if arr is not None]

        if len(arrs) == 1:
            return arrs[0]
        else:
            return hstack(arrs)

    def fit(self, *, X, W, Target, Z=None, sample_weight=None, groups=None):
        if self._discrete_target:
            # In this case, the Target is the one-hot-encoding of the treatment variable
            # We need to go back to the label representation of the one-hot so as to call
            # the classifier.
            if np.any(np.all(Target == 0, axis=0)) or (not np.any(np.all(Target == 0, axis=1))):
                raise AttributeError("Provided crossfit folds contain training splits that " +
                                     "don't contain all treatments")
            Target = inverse_onehot(Target)

        if sample_weight is not None:
            fit_with_groups(self._model, self._combine(X, W, Z, Target.shape[0]), Target,
                            groups=groups, sample_weight=sample_weight)
        else:
            fit_with_groups(self._model, self._combine(X, W, Z, Target.shape[0]), Target,
                            groups=groups)

    def score(self, *, X, W, Target, Z=None, sample_weight=None):
        if hasattr(self._model, 'score'):
            if self._discrete_target:
                # In this case, the Target is the one-hot-encoding of the treatment variable
                # We need to go back to the label representation of the one-hot so as to call
                # the classifier.
                if np.any(np.all(Target == 0, axis=0)) or (not np.any(np.all(Target == 0, axis=1))):
                    raise AttributeError("Provided crossfit folds contain training splits that " +
                                         "don't contain all treatments")
                Target = inverse_onehot(Target)

            if sample_weight is not None:
                return self._model.score(self._combine(X, W, Z, Target.shape[0]), Target, sample_weight=sample_weight)
            else:
                return self._model.score(self._combine(X, W, Z, Target.shape[0]), Target)
        else:
            return None

    def predict(self, X, W, Z=None):
        arrs = [arr for arr in [X, W, Z] if arr is not None]
        n_samples = arrs[0].shape[0] if arrs else 1
        if self._discrete_target:
            return self._model.predict_proba(self._combine(X, W, Z, n_samples, fitting=False))[:, 1:]
        else:
            return self._model.predict(self._combine(X, W, Z, n_samples, fitting=False))


class _FinalWrapper(_DMLFinalWrapper):
    pass
