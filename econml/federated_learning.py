# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from sklearn import clone

from econml.utilities import check_input_arrays
from ._cate_estimator import (LinearCateEstimator, TreatmentExpansionMixin,
                              StatsModelsCateEstimatorMixin, StatsModelsCateEstimatorDiscreteMixin)
from .dml import LinearDML
from .inference import StatsModelsInference, StatsModelsInferenceDiscrete
from .sklearn_extensions.linear_model import StatsModelsLinearRegression
from typing import List

# TODO: This could be extended to also work with our sparse and 2SLS estimators,
#       if we add an aggregate method to them
#       Remember to update the docs if this changes


class FederatedEstimator(TreatmentExpansionMixin, LinearCateEstimator):
    """
    A class for federated learning using LinearDML, LinearDRIV, and LinearDRLearner estimators.

    Parameters
    ----------
    estimators : list of LinearDML, LinearDRIV, or LinearDRLearner
        List of estimators to aggregate (all of the same type), which must already have
        been fit.
    """

    def __init__(self, estimators: List[LinearDML]):
        self.estimators = estimators
        dummy_est = clone(self.estimators[0], safe=False)  # used to extract various attributes later
        infs = [est._inference for est in self.estimators]
        assert (
            all(isinstance(inf, StatsModelsInference) for inf in infs) or
            all(isinstance(inf, StatsModelsInferenceDiscrete) for inf in infs)
        ), "All estimators must use either StatsModelsInference or StatsModelsInferenceDiscrete"
        cov_types = set(inf.cov_type for inf in infs)
        assert len(cov_types) == 1, f"All estimators must use the same covariance type, got {cov_types}"
        if isinstance(infs[0], StatsModelsInference):
            inf = StatsModelsInference(cov_type=cov_types.pop())
            cate_est_type = StatsModelsCateEstimatorMixin
            self.model_final_ = StatsModelsLinearRegression.aggregate([est.model_final_ for est in self.estimators])
            inf.model_final = self.model_final_
            inf.bias_part_of_coef = dummy_est.bias_part_of_coef
        else:
            inf = StatsModelsInferenceDiscrete(cov_type=cov_types.pop())
            cate_est_type = StatsModelsCateEstimatorDiscreteMixin
            self.fitted_models_final = [
                StatsModelsLinearRegression.aggregate(models)
                for models in zip(*[est.fitted_models_final for est in self.estimators])]
            inf.fitted_models_final = self.fitted_models_final

        # mix in the appropriate inference class
        self.__class__ = type("FederatedEstimator", (FederatedEstimator, cate_est_type), {})

        # assign all of the attributes from the dummy estimator that would normally be assigned during fitting
        # TODO: This seems hacky; is there a better abstraction to maintain these?
        #       This should also include bias_part_of_coef, model_final_, and fitted_models_final above
        inf.featurizer = dummy_est.featurizer_ if hasattr(dummy_est, 'featurizer_') else None
        inf._est = self
        self._d_t = inf._d_t = dummy_est._d_t
        self._d_y = inf._d_y = dummy_est._d_y
        self.d_t = inf.d_t = inf._d_t[0] if inf._d_t else 1
        self.d_y = inf.d_y = inf._d_y[0] if inf._d_y else 1
        self._d_t_in = inf._d_t_in = dummy_est._d_t_in
        self.fit_cate_intercept_ = inf.fit_cate_intercept = dummy_est.fit_cate_intercept
        self._inference = inf

        # Assign treatment expansion attributes
        self.transformer = dummy_est.transformer

    # Methods needed to implement the LinearCateEstimator interface

    def const_marginal_effect(self, X=None):
        X, = check_input_arrays(X)
        return self._inference.const_marginal_effect_inference(X).point_estimate

    def fit(self, *args, **kwargs):
        """
        This method should not be called; it is included only for compatibility with the
        CATE estimation APIs
        """
        raise NotImplementedError("FederatedEstimator does not support fit")

    # Methods needed to implement the LinearFinalModelCateEstimatorMixin
    def bias_part_of_coef(self):
        return self._inference.bias_part_of_coef
