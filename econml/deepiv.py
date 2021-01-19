# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import econml.iv.nnet as nnet
from .utilities import deprecated


@deprecated("The econml.deepiv.DeepIV class has renamed to econml.iv.nnet.DeepIV; "
            "an upcoming release will remove support for the old name")
class DeepIV(nnet.DeepIV):
    pass


@deprecated("The econml.deepiv.DeepIVEstimator class has been renamed to econml.iv.nnet.DeepIV; "
            "an upcoming release will remove support for the old name")
class DeepIVEstimator(nnet.DeepIV):
    pass
