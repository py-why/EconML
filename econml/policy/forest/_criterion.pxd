# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# published under the following license and copyright:
# BSD 3-Clause License
#
# Copyright (c) 2007-2020 The scikit-learn developers.
# All rights reserved.

# See _criterion.pyx for implementation details.

from ...tree._tree cimport DTYPE_t          # Type of X
from ...tree._tree cimport DOUBLE_t         # Type of y, sample_weight
from ...tree._tree cimport SIZE_t           # Type for indices and counters
from ...tree._tree cimport INT32_t          # Signed 32 bit integer
from ...tree._tree cimport UINT32_t         # Unsigned 32 bit integer

from ...tree._criterion cimport RegressionCriterion

cdef class LinearPolicyCriterion(RegressionCriterion):
    """ 
    """
    pass
