# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Double ML IV for Heterogeneous Treatment Effects.

An Double/Orthogonal machine learning approach to estimation of heterogeneous
treatment effect with an endogenous treatment and an instrument. It
implements the DMLIV algorithm from the paper:

Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments
Vasilis Syrgkanis, Victor Lei, Miruna Oprescu, Maggie Hei, Keith Battocchi, Greg Lewis
https://arxiv.org/abs/1905.10176

"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from econml.utilities import hstack
from sklearn.base import clone


class _BaseDMLIV:
    """
    The class _BaseDMLIV implements the base class of the DMLIV
    algorithm for estimating a CATE. It accepts three generic machine
    learning models:
    1) model_Y_X that estimates E[Y | X]
    2) model_T_X that estimates E[T | X]
    3) model_T_XZ that estimates E[T | X, Z]
    These are estimated in a cross-fitting manner for each sample in the training set.
    Then it minimizes the square loss:
    \sum_i (Y_i - E[Y|X_i] - theta(X) * (E[T|X_i, Z_i] - E[T|X_i]))^2
    This loss is minimized by the model_effect class, which is passed as an input.
    In the two children classes {DMLIV, GenericDMLIV}, we implement different strategies of how to invoke
    machine learning algorithms to minimize this final square loss.
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ, model_effect,
                 n_splits=2, binary_instrument=False, binary_treatment=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]. In alt_fit this model is also used
            to predict E[ E[T | X,Z] | X], i.e. regress E[T | X,Z] on X.
        model_T_XZ : model to predict E[T | X, Z]
        model_effect : final model that at fit time takes as input (Y-E[Y|X]), (E[T|X,Z]-E[T|X]) and X
            and supports method .effect(X) that produces the cate at X
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        """
        self.model_T_XZ = [clone(model_T_XZ, safe=False)
                           for _ in range(n_splits)]
        self.model_Y_X = [clone(model_Y_X, safe=False)
                          for _ in range(n_splits)]
        self.model_T_X = [clone(model_T_X, safe=False)
                          for _ in range(n_splits)]
        self.model_effect = model_effect
        self.n_splits = n_splits
        self.binary_instrument = binary_instrument
        self.binary_treatment = binary_treatment
        self.stored_final_data = False

    def fit(self, y, T, X, Z, store_final=False):
        """
        Parameters
        ----------
        y : outcome
        T : treatment (single dimensional)
        X : features/controls
        Z : instrument (single dimensional)
        store_final (bool) : whether to store the estimated nuisance values for
            fitting a different final stage model without the need of refitting
            the nuisance values. Increases memory usage. 
        """
        if len(T.shape) > 1 and T.shape[1] > 1:
            raise AssertionError(
                "Can only accept single dimensional treatment")
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional outcome")
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        if (Z.shape[1] > 1) and self.binary_instrument:
            raise AssertionError(
                "Binary instrument flag is True, but instrument is multi-dimensional")

        T = T.flatten()
        y = y.flatten()

        n_samples = y.shape[0]
        proj_t = np.zeros(n_samples)
        pred_t = np.zeros(n_samples)
        res_y = np.zeros(n_samples)

        if self.n_splits == 1:
            splits = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
        # TODO. Deal with multi-class instrument/treatment
        elif self.binary_instrument or self.binary_treatment:
            group = 2*T*self.binary_treatment + Z.flatten()*self.binary_instrument
            splits = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True).split(X, group)
        else:
            splits = KFold(n_splits=self.n_splits, shuffle=True).split(X)

        for idx, (train, test) in enumerate(splits):
            # Estimate h(Z, X) = E[T | Z, X] in cross-fitting manner
            proj_t[test] = self.model_T_XZ[idx].fit(hstack([X[train], Z[train]]),
                                                    T[train]).predict(hstack([X[test],
                                                                              Z[test]]))
            # Estimate residual Y_res = Y - q(X) = Y - E[Y | X] in cross-fitting manner
            res_y[test] = y[test] - \
                self.model_Y_X[idx].fit(X[train], y[train]).predict(X[test])
            # Estimate p(X) = E[T | X] in cross-fitting manner
            pred_t[test] = self.model_T_X[idx].fit(
                X[train], T[train]).predict(X[test])

        # Estimate theta by minimizing square loss (Y_res - theta(X) * (h(Z, X) - p(X)))^2
        self.model_effect.fit(res_y, (proj_t-pred_t).reshape(-1, 1), X)

        if store_final:
            self.stored_final_data = True
            self.X = X
            self.res_t = (proj_t-pred_t).reshape(-1, 1)
            self.res_y = res_y

        return self

    def effect(self, X):
        """
        Parameters
        ----------
        X : features
        """
        return self.model_effect.predict(X)

    @property
    def coef_(self):
        return self.effect_model.coef_

    @property
    def intercept_(self):
        return self.effect_model.intercept_

    @property
    def effect_model(self):
        return self.model_effect

    @property
    def fitted_nuisances(self):
        return {'model_Y_X': self.model_Y_X,
                'model_T_X': self.model_T_X,
                'model_T_XZ': self.model_T_XZ}


class DMLIV(_BaseDMLIV):
    """
    A child of the _BaseDMLIV class that specifies a particular effect model
    where the treatment effect is linear in some featurization of the variable X
    The features are created by a provided featurizer that supports fit_transform.
    Then an arbitrary model fits on the composite set of features.

    Concretely, it assumes that theta(X)=<theta, phi(X)> for some features phi(X)
    and runs a linear model regression of Y-E[Y|X] on phi(X)*(E[T|X,Z]-E[T|X]).
    The features are created by the featurizer provided by the user. The particular
    linear model regression is also specified by the user (e.g. Lasso, ElasticNet)
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ, model_effect, featurizer,
                 n_splits=2, binary_instrument=False, binary_treatment=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]
        model_T_XZ : model to predict E[T | X, Z]
        model_effect : final linear model for predicting (Y-E[Y|X]) from phi(X) * (E[T|X,Z]-E[T|X])
            Method is incorrect if this model is not linear (e.g. Lasso, ElasticNet, LinearRegression).
        featurizer : object that creates features of X to use for effect model. Must have a method
            fit_transform that is applied on X to create phi(X).
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        """
        class ModelEffectWrapper:
            """
            A wrapper class that takes as input X, T, y and estimates an effect model of the form
            y= theta(X) * T + epsilon
            """

            def __init__(self, model_effect, featurizer):
                """
                Parameters
                ----------
                model_effect : model for CATE. At fit takes as input features(X) * (residual T)
                    and (residual Y). At predict time takes as input features(X)
                featurizer : model to produces features(X) from X
                """
                self.model_effect = model_effect
                self.featurizer = featurizer

            def fit(self, y, T, X):
                """
                Parameters
                ----------
                y : outcome
                T : treatment
                X : features
                """
                self.model_effect.fit(self.featurizer.fit_transform(X) * T, y)
                return self

            def predict(self, X):
                """
                Parameters
                ----------
                X : features
                """
                return self.model_effect.predict(self.featurizer.fit_transform(X))\
                    - self.model_effect.predict(self.featurizer.fit_transform(X)*np.zeros((X.shape[0], 1)))

        super(DMLIV, self).__init__(model_Y_X, model_T_X, model_T_XZ,
                                    ModelEffectWrapper(model_effect, featurizer),
                                    n_splits=n_splits,
                                    binary_instrument=binary_instrument,
                                    binary_treatment=binary_treatment)

    def refit_final(self, model_effect, featurizer):
        """ Refits a different effect model in the final stage of dml with
        a different featurizer and a different linear model. Avoids refitting
        the first stage nuisance functions. To call this method you
        first have to call fit(y, T, X, Z, store_final=True), with the
        store_final flag set to True.
        """
        if not self.stored_final_data:
            raise AttributeError(
                "Estimator is not yet fit with store_data=True")
        self.model_effect.model_effect = model_effect
        self.model_effect.featurizer = featurizer
        self.model_effect.fit(self.res_y, self.res_t, self.X)
        return self

    @property
    def effect_model(self):
        """ Returns the linear model fitted in the final stage on features phi(X)*(E[T|X,Z]-E[T|X])
        """
        return self.model_effect.model_effect


class GenericDMLIV(_BaseDMLIV):
    """
    A child of the _BaseDMLIV class that allows for an arbitrary square loss based ML
    method in the final stage of the DMLIV algorithm. The method has to support
    sample weights and the fit method has to take as input sample_weights (e.g. random forests), i.e.
    fit(X, y, sample_weight=None)
    It achieves this by re-writing the final stage square loss of the DMLIV algorithm as:
        \sum_i (E[T|X_i, Z_i] - E[T|X_i])^2 * ((Y_i - E[Y|X_i])/(E[T|X_i, Z_i] - E[T|X_i]) - theta(X))^2
    Then this can be viewed as a weighted square loss regression, where the target label is
        \tilde{Y}_i = (Y_i - E[Y|X_i])/(E[T|X_i, Z_i] - E[T|X_i])
    and each sample has a weight of
        V(X_i) = (E[T|X_i, Z_i] - E[T|X_i])^2
    Thus we can call any regression model with inputs:
        fit(X, \tilde{Y}_i, sample_weight=V(X_i))
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ, model_effect,
                 n_splits=2, binary_instrument=False, binary_treatment=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]
        model_T_XZ : model to predict E[T | X, Z]
        model_effect : final model for predicting \tilde{Y} from X with sample weights V(X)
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        """
        class ModelEffectWrapper:
            """
            A wrapper class that takes as input X, T, y and estimates an effect model of the form
            y= theta(X) * T + epsilon
            by minimizing the weighted square loss \sum_i T_i^2 (Y_i/T_i - theta(X_i))^2
            To avoid numerical instability the variable T_i is clipped when its absolute value
            is below a small number of 1e-6.
            """

            def __init__(self, model_effect):
                """
                Parameters
                ----------
                model_effect : model for CATE. At fit takes as input X, y=tilde{Y}, sample_weight=V(X).
                    At predict time takes as input X
                """
                self.model_effect = model_effect
                self._T_clip = 1e-6

            def fit(self, y, T, X):
                """
                Parameters
                ----------
                y : outcome
                T : treatment
                X : features
                """
                T_sign = np.sign(T)
                T_sign[T_sign == 0] = 1
                clipped_T = T_sign * np.clip(np.abs(T), self._T_clip, np.inf)
                self.model_effect.fit(
                    X, y/clipped_T.flatten(), sample_weight=(T.flatten())**2)
                return self

            def predict(self, X):
                """
                Parameters
                ----------
                X : features
                """
                return self.model_effect.predict(X)

        super(GenericDMLIV, self).__init__(model_Y_X, model_T_X, model_T_XZ,
                                           ModelEffectWrapper(model_effect),
                                           n_splits=n_splits,
                                           binary_instrument=binary_instrument,
                                           binary_treatment=binary_treatment)

    def refit_final(self, model_effect):
        """ Refits a different effect model in the final stage of dml with
        a different featurizer and a different linear model. Avoids refitting
        the first stage nuisance functions. To call this method you
        first have to call fit(y, T, X, Z, store_final=True), with the
        store_final flag set to True.
        """
        if not self.stored_final_data:
            raise AttributeError(
                "Estimator is not yet fit with store_data=True")
        self.model_effect.model_effect = model_effect
        self.model_effect.fit(self.res_y, self.res_t, self.X)
        return self

    @property
    def effect_model(self):
        """ Returns the effect model provided by the user after being fitted in the final stage
        """
        return self.model_effect.model_effect
