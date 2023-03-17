# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

"""Double Machine Learning for Dynamic Treatment Effects.

A Double/Orthogonal machine learning approach to estimation of heterogeneous
treatment effect in the dynamic treatment regime. For the theoretical
foundations of these methods see: [dynamicdml]_.

References
----------

.. [dynamicdml] Greg Lewis and Vasilis Syrgkanis.
    Double/Debiased Machine Learning for Dynamic Treatment Effects.
    `<https://arxiv.org/abs/2002.07285>`_, 2021.
"""

from ._dml import DynamicDML

__all__ = ["DynamicDML"]
