# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX explain type definition."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import torch

FeatureMapType = torch.Tensor | Sequence[torch.Tensor] | dict[str, torch.Tensor]


class TargetExplainGroup(str, Enum):
    """OTX target explain group definition.

    Enum contains the following values:
        IMAGE - This implies that single global saliency map will be generated for input image.
        ALL - This implies that saliency maps will be generated for all possible targets.
        PREDICTIONS - This implies that saliency map will be generated per each prediction.
    """

    IMAGE = "IMAGE"
    ALL = "ALL"
    PREDICTIONS = "PREDICTIONS"
