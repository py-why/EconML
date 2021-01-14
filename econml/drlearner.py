# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import econml.dr as dr
from .utilities import deprecated


@deprecated("The econml.drlearner.DRLearner class has been moved to econml.dr.DRLearner; "
            "an upcoming release will remove support for the old name")
class DRLearner(dr.DRLearner):
    pass


@deprecated("The econml.drlearner.LinearDRLearner class has been moved to econml.dr.LinearDRLearner; "
            "an upcoming release will remove support for the old name")
class DRLearner(dr.DRLearner):
    pass


@deprecated("The econml.drlearner.SparseLinearDRLearner class has been moved to econml.dr.SparseLinearDRLearner; "
            "an upcoming release will remove support for the old name")
class DRLearner(dr.DRLearner):
    pass


@deprecated("The econml.drlearner.ForestDRLearner class has been moved to econml.dr.ForestDRLearner; "
            "an upcoming release will remove support for the old name")
class DRLearner(dr.DRLearner):
    pass
