# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from econml.dml import LinearDML
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from typing import List


class FederatedEstimator:
    """
    A class for federated learning using LinearDML estimators.

    Parameters
    ----------
    estimators : list of LinearDML
        List of LinearDML estimators to aggregate.

    Attributes
    ----------
    estimators : list of LinearDML
        List of LinearDML estimators provided during initialization.

    model_final_ : StatsModelsLinearRegression
        The aggregated model obtained by aggregating models from `estimators`.
    """
    def __init__(self, estimators: List[LinearDML]):
        self.estimators = estimators
        self.model_final_ = StatsModelsLinearRegression.aggregate([est.model_final_ for est in estimators])
