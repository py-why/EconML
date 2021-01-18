# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import econml.inference._bootstrap as bootstrap
from .utilities import deprecated


@deprecated("The econml.bootstrap.BootstrapEstimator class has been moved "
            "to econml.inference._bootstrap.BootstrapEstimator and is no longer part of the public API; "
            "an upcoming release will remove support for the old name and will consider `BootstrapEstimator` "
            "as part of the private API with no guarantee of API consistency across releases. "
            "Instead of wrapping CATE esitmators with the BootstrapEstimator to preduce bootstrap confidence "
            "intervals, consider passing `inference='bootstrap'` or "
            "`inference=econml.inference.BootstrapInference(n_bootstrap_samples=..., bootstrap_type=...)`, "
            "as a keyword argument at the `fit` method of the CATE estimator.")
class BootstrapEstimator(bootstrap.BootstrapEstimator):
    pass
