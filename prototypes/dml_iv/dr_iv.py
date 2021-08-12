# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Doubly Robust IV for Heterogeneous Treatment Effects.

An Doubly Robust machine learning approach to estimation of heterogeneous
treatment effect with an endogenous treatment and an instrument.

Implements the DRIV algorithm for estimating CATE with IVs from the paper:

Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments
Vasilis Syrgkanis, Victor Lei, Miruna Oprescu, Maggie Hei, Keith Battocchi, Greg Lewis
https://arxiv.org/abs/1905.10176
"""

import numpy as np
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from econml.utilities import hstack
from sklearn.base import clone
import copy
from itertools import tee


class _BaseDRIV:

    """
    The _BaseDRIV algorithm for estimating CATE with IVs. It is the parent of the
    two public classes {DRIV, ProjectedDRIV}
    """

    def __init__(self, nuisance_models,
                 model_effect,
                 cov_clip=.1,
                 n_splits=3,
                 binary_instrument=False, binary_treatment=False,
                 opt_reweighted=False):
        """
        Parameters
        ----------
        nuisance_models : dictionary of nuisance models, with {'name_of_model' : EstimatorObject, ...}
        model_effect : model to estimate second stage effect model from doubly robust target
        cov_clip : clipping of the covariate for regions with low "overlap",
            so as to reduce variance
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        opt_reweighted : whether to reweight the samples to minimize variance. If True then
            model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
            utilities can be used for any linear model to enable sample_weights). If True then
            assumes the model_effect is flexible enough to fit the true CATE model. Otherwise,
            it method will return a biased projection to the model_effect space, biased
            to give more weight on parts of the feature space where the instrument is strong.
        """
        for n_name, n_model in nuisance_models.items():
            setattr(self, n_name, [clone(n_model, safe=False)
                                   for _ in range(n_splits)])
        self.nuisance_model_names = list(nuisance_models.keys())
        self.model_effect = clone(model_effect, safe=False)
        self.cov_clip = cov_clip
        self.n_splits = n_splits
        self.binary_instrument = binary_instrument
        self.binary_treatment = binary_treatment
        self.opt_reweighted = opt_reweighted
        self.stored_final_data = False

    def _check_inputs(self, y, T, X, Z):
        """ Checks dimension of inputs and reshapes inputs. Implemented at child
        """
        raise("Child class needs to implement this method")

    def _nuisance_estimates(self, y, T, X, Z):
        """ Estimates the nuisance quantities for each sample with cross-fitting.
        Implemented at child
        """
        raise("Child class needs to implement this method")

    def _get_split_enum(self, y, T, X, Z):
        """
        Returns an enumerator over the splits for cross-fitting
        """
        # We do a three way split, as typically a preliminary theta estimator would require
        # many samples. So having 2/3 of the sample to train model_theta seems appropriate.
        if self.n_splits == 1:
            splits = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
        # TODO. Deal with multi-class instrument
        elif self.binary_instrument or self.binary_treatment:
            group = 2 * T * self.binary_treatment + Z.flatten() * self.binary_instrument
            splits = StratifiedKFold(
                n_splits=self.n_splits, shuffle=True).split(X, group)
        else:
            splits = KFold(n_splits=self.n_splits, shuffle=True).split(X)

        return splits

    def fit(self, y, T, X, Z, store_final=False):
        """
        Parameters
        ----------
        y : outcome
        T : treatment (single dimensional)
        X : features/controls
        Z : instrument (single dimensional)
        store_final : whether to store nuisance data that are used in the final for
            refitting the final stage later on
        """
        y, T, X, Z = self._check_inputs(y, T, X, Z)

        prel_theta, res_t, res_y, res_z, cov = self._nuisance_estimates(
            y, T, X, Z)

        # Estimate final model of theta(X) by minimizing the square loss:
        # (prel_theta(X) + (Y_res - prel_theta(X) * T_res) * Z_res / cov[T,Z | X] - theta(X))^2
        # We clip the covariance so that it is bounded away from zero, so as to reduce variance
        # at the expense of some small bias. For points with very small covariance we revert
        # to the model-based preliminary estimate and do not add the correction term.
        cov_sign = np.sign(cov)
        cov_sign[cov_sign == 0] = 1
        clipped_cov = cov_sign * np.clip(np.abs(cov),
                                         self.cov_clip, np.inf)
        theta_dr = prel_theta + \
            (res_y - prel_theta * res_t) * res_z / clipped_cov
        if self.opt_reweighted:
            self.model_effect.fit(X, theta_dr, sample_weight=clipped_cov**2)
        else:
            self.model_effect.fit(X, theta_dr)

        if store_final:
            self.X = X
            self.theta_dr = theta_dr
            self.stored_final_data = True
            self.clipped_cov = clipped_cov

        return self

    def refit_final(self, model_effect, opt_reweighted=None):
        """
        Change the final effect model and refit the final stage.
        Parameters
        ----------
        model_effect : an instance of the new effect model to be fitted in the final stage
        opt_reweighted : whether to weight samples for variance reduction in the final model fitting
        """
        if not self.stored_final_data:
            raise AttributeError(
                "Estimator is not yet fit with store_data=True")
        if opt_reweighted is not None:
            self.opt_reweighted = opt_reweighted
        self.model_effect = model_effect
        if self.opt_reweighted:
            self.model_effect.fit(self.X, self.theta_dr,
                                  sample_weight=self.clipped_cov**2)
        else:
            self.model_effect.fit(self.X, self.theta_dr)
        return self

    def effect(self, X):
        """
        Parameters
        ----------
        X : features
        """
        return self.model_effect.predict(X)

    @property
    def effect_model(self):
        return self.model_effect

    @property
    def fitted_nuisances(self):
        nuisance_dict = {}
        for n_name in self.nuisance_model_names:
            nuisance_dict[n_name] = getattr(self, n_name)
        return nuisance_dict

    @property
    def coef_(self):
        if not hasattr(self.effect_model, 'coef_'):
            raise AttributeError("Effect model is not linear!")
        return self.effect_model.coef_

    @property
    def intercept_(self):
        if not hasattr(self.effect_model, 'intercept_'):
            raise AttributeError("Effect model is not linear!")
        return self.effect_model.intercept_


class DRIV(_BaseDRIV):
    """
    Implements the DRIV algorithm
    """

    def __init__(self, model_Y_X, model_T_X, model_Z_X,
                 prel_model_effect, model_TZ_X,
                 model_effect,
                 cov_clip=.1,
                 n_splits=3,
                 binary_instrument=False, binary_treatment=False,
                 opt_reweighted=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]. In alt_fit, this model is also
            used to predict E[T | X, Z]
        model_Z_X : model to predict E[Z | X]
        model_theta : model that estimates a preliminary version of the CATE
            (e.g. via DMLIV or other method)
        model_TZ_X : model to estimate E[T * Z | X]
        model_effect : model to estimate second stage effect model from doubly robust target
        cov_clip : clipping of the covariate for regions with low "overlap",
            so as to reduce variance
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        opt_reweighted : whether to reweight the samples to minimize variance. If True then
            model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
            utilities can be used for any linear model to enable sample_weights). If True then
            assumes the model_effect is flexible enough to fit the true CATE model. Otherwise,
            it method will return a biased projection to the model_effect space, biased
            to give more weight on parts of the feature space where the instrument is strong.
        """
        nuisance_models = {'prel_model_effect': prel_model_effect,
                           'model_TZ_X': model_TZ_X,
                           'model_T_X': model_T_X,
                           'model_Z_X': model_Z_X,
                           'model_Y_X': model_Y_X}
        super(DRIV, self).__init__(nuisance_models, model_effect,
                                   cov_clip=cov_clip,
                                   n_splits=n_splits,
                                   binary_instrument=binary_instrument, binary_treatment=binary_treatment,
                                   opt_reweighted=opt_reweighted)
        return

    def _check_inputs(self, y, T, X, Z):
        if len(Z.shape) > 1 and Z.shape[1] > 1:
            raise AssertionError(
                "Can only accept single dimensional instrument")
        if len(T.shape) > 1 and T.shape[1] > 1:
            raise AssertionError(
                "Can only accept single dimensional treatment")
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional outcome")
        Z = Z.flatten()
        T = T.flatten()
        y = y.flatten()
        return y, T, X, Z

    def _nuisance_estimates(self, y, T, X, Z):
        n_samples = y.shape[0]
        prel_theta = np.zeros(n_samples)
        res_t = np.zeros(n_samples)
        res_y = np.zeros(n_samples)
        res_z = np.zeros(n_samples)
        cov = np.zeros(n_samples)

        splits = self._get_split_enum(y, T, X, Z)
        for idx, (train, test) in enumerate(splits):
            # Estimate preliminary theta in cross fitting manner
            prel_theta[test] = self.prel_model_effect[idx].fit(
                y[train], T[train], X[train], Z[train]).effect(X[test]).flatten()
            # Estimate p(X) = E[T | X] in cross fitting manner
            self.model_T_X[idx].fit(X[train], T[train])
            pr_t_test = self.model_T_X[idx].predict(X[test])
            # Estimate r(Z) = E[Z | X] in cross fitting manner
            self.model_Z_X[idx].fit(X[train], Z[train])
            pr_z_test = self.model_Z_X[idx].predict(X[test])
            # Calculate residual T_res = T - p(X) and Z_res = Z - r(X)
            res_t[test] = T[test] - pr_t_test
            res_z[test] = Z[test] - pr_z_test
            # Estimate residual Y_res = Y - q(X) = Y - E[Y | X] in cross fitting manner
            res_y[test] = y[test] - \
                self.model_Y_X[idx].fit(X[train], y[train]).predict(X[test])
            # Estimate cov[T, Z | X] = E[(T-p(X))*(Z-r(X)) | X] = E[T*Z | X] - E[T |X]*E[Z | X]
            cov[test] = self.model_TZ_X[idx].fit(
                X[train], T[train] * Z[train]).predict(X[test]) - pr_t_test * pr_z_test

        return prel_theta, res_t, res_y, res_z, cov


class ProjectedDRIV(_BaseDRIV):
    """
    This is a slight variant of DRIV where we use E[T|Z, X] as
    the instrument as opposed to Z. The rest is the same as the normal
    fit.
    """

    def __init__(self, model_Y_X, model_T_X, model_T_XZ,
                 prel_model_effect, model_TZ_X,
                 model_effect,
                 cov_clip=.1,
                 n_splits=3,
                 binary_instrument=False, binary_treatment=False,
                 opt_reweighted=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_X : model to predict E[T | X]. In alt_fit, this model is also
            used to predict E[T | X, Z]
        model_T_XZ : model to predict E[T | X, Z]
        model_theta : model that estimates a preliminary version of the CATE
            (e.g. via DMLIV or other method)
        model_TZ_X : model to estimate cov[T, E[T|X,Z] | X] = E[(T-E[T|X]) * (E[T|X,Z] - E[T|X]) | X].
        model_effect : model to estimate second stage effect model from doubly robust target
        cov_clip : clipping of the covariate for regions with low "overlap",
            so as to reduce variance
        n_splits : number of splits to use in cross-fitting
        binary_instrument : whether to stratify cross-fitting splits by instrument
        binary_treatment : whether to stratify cross-fitting splits by treatment
        opt_reweighted : whether to reweight the samples to minimize variance. If True then
            model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
            utilities can be used for any linear model to enable sample_weights). If True then
            assumes the model_effect is flexible enough to fit the true CATE model. Otherwise,
            it method will return a biased projection to the model_effect space, biased
            to give more weight on parts of the feature space where the instrument is strong.
        """
        nuisance_models = {'prel_model_effect': prel_model_effect,
                           'model_Y_X': model_Y_X,
                           'model_T_X': model_T_X,
                           'model_T_XZ': model_T_XZ,
                           'model_TZ_X': model_TZ_X}
        super(ProjectedDRIV, self).__init__(nuisance_models, model_effect,
                                            cov_clip=cov_clip,
                                            n_splits=n_splits,
                                            binary_instrument=binary_instrument, binary_treatment=binary_treatment,
                                            opt_reweighted=opt_reweighted)
        return

    def _check_inputs(self, y, T, X, Z):

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

        return y, T, X, Z

    def _nuisance_estimates(self, y, T, X, Z):

        n_samples = y.shape[0]
        prel_theta = np.zeros(n_samples)
        res_t = np.zeros(n_samples)
        res_y = np.zeros(n_samples)
        res_z = np.zeros(n_samples)
        cov = np.zeros(n_samples)
        proj_t = np.zeros(n_samples)

        splits = self._get_split_enum(y, T, X, Z)

        # TODO. The solution below is not really a valid cross-fitting
        # as the test data are used to create the proj_t on the train
        # which in the second train-test loop is used to create the nuisance
        # cov on the test data. Hence the T variable of some sample
        # is implicitly correlated with its cov nuisance, through this flow
        # of information. However, this seems a rather weak correlation.
        # The more kosher would be to do an internal nested cv loop for the T_XZ
        # model.
        splits, splits_one = tee(splits)
        # Estimate h(X, Z) = E[T | X, Z] in cross fitting manner
        for idx, (train, test) in enumerate(splits_one):
            self.model_T_XZ[idx].fit(hstack([X[train], Z[train]]), T[train])
            proj_t[test] = self.model_T_XZ[idx].predict(
                hstack([X[test], Z[test]]))

        for idx, (train, test) in enumerate(splits):
            # Estimate preliminary theta in cross fitting manner
            prel_theta[test] = self.prel_model_effect[idx].fit(
                y[train], T[train], X[train], Z[train]).effect(X[test]).flatten()
            # Estimate p(X) = E[T | X] in cross fitting manner
            self.model_T_X[idx].fit(X[train], T[train])
            pr_t_test = self.model_T_X[idx].predict(X[test])
            # Calculate residual T_res = T - p(X) and Z_res = h(Z, X) - p(X)
            res_t[test] = T[test] - pr_t_test
            res_z[test] = proj_t[test] - pr_t_test
            # Estimate residual Y_res = Y - q(X) = Y - E[Y | X] in cross fitting manner
            res_y[test] = y[test] - \
                self.model_Y_X[idx].fit(X[train], y[train]).predict(X[test])
            # Estimate cov[T, E[T|X,Z] | X] = E[T * E[T|X,Z]] - E[T|X]^2
            cov[test] = self.model_TZ_X[idx].fit(
                X[train], T[train] * proj_t[train]).predict(X[test]) - pr_t_test**2

        return prel_theta, res_t, res_y, res_z, cov

##############################################################################
# Classes for the DRIV implementation for the special case of intent-to-treat
# A/B test
##############################################################################


class _IntentToTreatDRIV(_BaseDRIV):
    """
    Helper class for the DRIV algorithm for the intent-to-treat A/B test setting
    """

    def __init__(self, model_Y_X, model_T_XZ,
                 prel_model_effect,
                 model_effect,
                 cov_clip=.1,
                 n_splits=3,
                 opt_reweighted=False):
        """
        """
        nuisance_models = {'model_Y_X': model_Y_X,
                           'model_T_XZ': model_T_XZ,
                           'prel_model_effect': prel_model_effect}
        super(_IntentToTreatDRIV, self).__init__(nuisance_models, model_effect,
                                                 cov_clip=cov_clip,
                                                 n_splits=n_splits,
                                                 binary_instrument=True, binary_treatment=True,
                                                 opt_reweighted=opt_reweighted)
        return

    def _check_inputs(self, y, T, X, Z):
        if len(Z.shape) > 1 and Z.shape[1] > 1:
            raise AssertionError(
                "Can only accept single dimensional instrument")
        if len(T.shape) > 1 and T.shape[1] > 1:
            raise AssertionError(
                "Can only accept single dimensional treatment")
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise AssertionError("Can only accept single dimensional outcome")
        Z = Z.flatten()
        T = T.flatten()
        y = y.flatten()
        return y, T, X, Z

    def _nuisance_estimates(self, y, T, X, Z):
        n_samples = y.shape[0]
        prel_theta = np.zeros(n_samples)
        res_t = np.zeros(n_samples)
        res_y = np.zeros(n_samples)
        delta = np.zeros(n_samples)

        splits = self._get_split_enum(y, T, X, Z)
        for idx, (train, test) in enumerate(splits):
            # Estimate preliminary theta in cross fitting manner
            prel_theta[test] = self.prel_model_effect[idx].fit(
                y[train], T[train], X[train], Z[train]).effect(X[test]).flatten()
            # Estimate p(X) = E[T | X] in cross fitting manner
            self.model_T_XZ[idx].fit(hstack([X[train], Z[train].reshape(-1, 1)]), T[train])
            Z_one = np.ones((Z[test].shape[0], 1))
            Z_zero = np.zeros((Z[test].shape[0], 1))
            pr_t_test_one = self.model_T_XZ[idx].predict(hstack([X[test], Z_one]))
            pr_t_test_zero = self.model_T_XZ[idx].predict(hstack([X[test], Z_zero]))
            delta[test] = (pr_t_test_one - pr_t_test_zero) / 2
            pr_t_test = (pr_t_test_one + pr_t_test_zero) / 2
            res_t[test] = T[test] - pr_t_test
            # Estimate residual Y_res = Y - q(X) = Y - E[Y | X] in cross fitting manner
            res_y[test] = y[test] - \
                self.model_Y_X[idx].fit(X[train], y[train]).predict(X[test])

        return prel_theta, res_t, res_y, 2 * Z - 1, delta


class _DummyCATE:
    """
    A dummy cate effect model that always returns zero effect
    """

    def __init__(self):
        return

    def fit(self, y, T, X, Z):
        return self

    def effect(self, X):
        return np.zeros(X.shape[0])


class IntentToTreatDRIV(_IntentToTreatDRIV):
    """
    Implements the DRIV algorithm for the intent-to-treat A/B test setting
    """

    def __init__(self, model_Y_X, model_T_XZ,
                 flexible_model_effect,
                 final_model_effect=None,
                 cov_clip=.1,
                 n_splits=3,
                 opt_reweighted=False):
        """
        Parameters
        ----------
        model_Y_X : model to predict E[Y | X]
        model_T_XZ : model to predict E[T | X, Z]
        flexible_model_effect : a flexible model for a preliminary version of the CATE, must accept
            sample_weight at fit time.
        final_model_effect : a final model for the CATE and projections. If None, then
            flexible_model_effect is also used as a final model
        cov_clip : clipping of the covariate for regions with low "overlap",
            so as to reduce variance
        n_splits : number of splits to use in cross-fitting
        opt_reweighted : whether to reweight the samples to minimize variance. If True then
            final_model_effect.fit must accept sample_weight as a kw argument (WeightWrapper from
            utilities can be used for any linear model to enable sample_weights). If True then
            assumes the final_model_effect is flexible enough to fit the true CATE model. Otherwise,
            it method will return a biased projection to the model_effect space, biased
            to give more weight on parts of the feature space where the instrument is strong.
        """
        prel_model_effect = _IntentToTreatDRIV(clone(model_Y_X, safe=False),
                                               clone(model_T_XZ, safe=False),
                                               _DummyCATE(),
                                               clone(flexible_model_effect, safe=False),
                                               cov_clip=1e-7, n_splits=1, opt_reweighted=True)
        if final_model_effect is None:
            final_model_effect = clone(flexible_model_effect, safe=False)
        super(IntentToTreatDRIV, self).__init__(model_Y_X, model_T_XZ, prel_model_effect,
                                                final_model_effect,
                                                cov_clip=cov_clip,
                                                n_splits=n_splits,
                                                opt_reweighted=opt_reweighted)
        return
