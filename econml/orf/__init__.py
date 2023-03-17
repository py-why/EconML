# Copyright (c) PyWhy contributors. All rights reserved.
# Licensed under the MIT License.

""" An implementation of Orthogonal Random Forests [orf]_ and special
case python classes.

References
----------
.. [orf] M. Oprescu, V. Syrgkanis and Z. S. Wu.
    Orthogonal Random Forest for Causal Inference.
    *Proceedings of the 36th International Conference on Machine Learning*, 2019.
    URL http://proceedings.mlr.press/v97/oprescu19a.html.
"""

from ._ortho_forest import DMLOrthoForest, DROrthoForest

__all__ = ["DMLOrthoForest",
           "DROrthoForest"]
