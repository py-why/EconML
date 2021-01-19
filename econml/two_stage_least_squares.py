# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import econml.iv.sieve as sieve
from .utilities import deprecated


@deprecated("The econml.two_stage_least_squares.HermiteFeatures class has been moved "
            "to econml.iv.sieve.HermiteFeatures; "
            "an upcoming release will remove support for the old name")
class HermiteFeatures(sieve.HermiteFeatures):
    pass


@deprecated("The econml.two_stage_least_squares.DPolynomialFeatures class has been moved "
            "to econml.iv.sieve.DPolynomialFeatures; "
            "an upcoming release will remove support for the old name")
class DPolynomialFeatures(sieve.DPolynomialFeatures):
    pass


@deprecated("The econml.two_stage_least_squares.NonparametricTwoStageLeastSquares class has been moved "
            "to econml.iv.sieve.SieveTSLS; "
            "an upcoming release will remove support for the old name")
class NonparametricTwoStageLeastSquares(sieve.SieveTSLS):
    pass
