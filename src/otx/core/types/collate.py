"""OTX Collate Mode Enum"""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

class CollateMode(str, Enum):
    """Collate mode for collate function."""
    Torch = "Torch"
    Numpy = "Numpy"
