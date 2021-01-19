# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import econml.orf as orf
from .utilities import deprecated


@deprecated("The econml.ortho_forest.DMLOrthoForest class has been moved to econml.orf.DMLOrthoForest; "
            "an upcoming release will remove support for the old name")
class DMLOrthoForest(orf.DMLOrthoForest):
    pass


@deprecated("The econml.ortho_forest.DiscreteTreatmentOrthoForest class has been "
            "moved to econml.orf.DROrthoForest; "
            "an upcoming release will remove support for the old name")
class DROrthoForest(orf.DROrthoForest):
    pass


@deprecated("The econml.ortho_forest.ContinuousTreatmentOrthoForest class has been "
            "renamed to econml.orf.DMLOrthoForest; "
            "an upcoming release will remove support for the old name")
class ContinuousTreatmentOrthoForest(orf.DMLOrthoForest):
    pass


@deprecated("The econml.ortho_forest.DiscreteTreatmentOrthoForest class has been "
            "renamed to econml.orf.DROrthoForest; "
            "an upcoming release will remove support for the old name")
class DiscreteTreatmentOrthoForest(orf.DROrthoForest):
    pass
