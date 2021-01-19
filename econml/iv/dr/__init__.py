# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Orthogonal IV for Heterogeneous Treatment Effects.

A Double/Orthogonal machine learning approach to estimation of heterogeneous
treatment effect with an endogenous treatment and an instrument. It
implements the DMLIV and related algorithms from the paper:

Machine Learning Estimation of Heterogeneous Treatment Effects with Instruments
Vasilis Syrgkanis, Victor Lei, Miruna Oprescu, Maggie Hei, Keith Battocchi, Greg Lewis
https://arxiv.org/abs/1905.10176

"""

from ._dr import IntentToTreatDRIV, LinearIntentToTreatDRIV

__all__ = ["IntentToTreatDRIV",
           "LinearIntentToTreatDRIV"]
