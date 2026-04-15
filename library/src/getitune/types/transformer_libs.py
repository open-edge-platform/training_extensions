# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Transform library types used in Geti Tune."""

from __future__ import annotations

from enum import Enum


class TransformLibType(str, Enum):
    """Transform library types used in Geti Tune."""

    TORCHVISION = "TORCHVISION"
