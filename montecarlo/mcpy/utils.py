# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np

def filesafe(str):
    return "".join([c for c in str if c.isalpha() or c.isdigit() or c==' ']).rstrip().replace(' ', '_')
