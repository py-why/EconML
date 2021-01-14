# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import econml.iv.tsls as tsls
from .utilities import deprecated


@deprecated("The econml.two_stage_least_squares.HermiteFeatures class has been moved "
            "to econml.iv.tsls.HermiteFeatures; "
            "an upcoming release will remove support for the old name")
class HermiteFeatures(tsls.HermiteFeatures):
    pass


@deprecated("The econml.two_stage_least_squares.DPolynomialFeatures class has been moved "
            "to econml.iv.tsls.DPolynomialFeatures; "
            "an upcoming release will remove support for the old name")
class DPolynomialFeatures(tsls.DPolynomialFeatures):
    pass


@deprecated("The econml.two_stage_least_squares.NonparametricTwoStageLeastSquares class has been moved "
            "to econml.iv.tsls.NonparametricTSLS; "
            "an upcoming release will remove support for the old name")
class NonparametricTwoStageLeastSquares(tsls.NonparametricTSLS):
    pass
