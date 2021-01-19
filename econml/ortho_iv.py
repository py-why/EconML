# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import econml.iv.dml as dmliv
import econml.iv.dr as driv
from .utilities import deprecated


@deprecated("The econml.ortho_iv.DMLATEIV class has been moved to econml.iv.dml.DMLATEIV; "
            "an upcoming release will remove support for the old name")
class DMLATEIV(dmliv.DMLATEIV):
    pass


@deprecated("The econml.ortho_iv.ProjectedDMLATEIV class has been moved to econml.iv.dml.ProjectedDMLATEIV; "
            "an upcoming release will remove support for the old name")
class ProjectedDMLATEIV(dmliv.ProjectedDMLATEIV):
    pass


@deprecated("The econml.ortho_iv.DMLIV class has been moved to econml.iv.dml.DMLIV; "
            "an upcoming release will remove support for the old name")
class DMLIV(dmliv.DMLIV):
    pass


@deprecated("The econml.ortho_iv.NonParamDMLIV class has been moved to econml.iv.dml.NonParamDMLIV; "
            "an upcoming release will remove support for the old name")
class NonParamDMLIV(dmliv.NonParamDMLIV):
    pass


@deprecated("The econml.ortho_iv.IntentToTreatDRIV class has been moved to econml.iv.dr.IntentToTreatDRIV; "
            "an upcoming release will remove support for the old name")
class IntentToTreatDRIV(driv.IntentToTreatDRIV):
    pass


@deprecated("The econml.ortho_iv.LinearIntentToTreatDRIV class has been moved "
            "to econml.iv.dr.LinearIntentToTreatDRIV; "
            "an upcoming release will remove support for the old name")
class LinearIntentToTreatDRIV(driv.LinearIntentToTreatDRIV):
    pass
