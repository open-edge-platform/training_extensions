# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom OTX Losses for Object Detection."""

from .atss_loss import ATSSCriterion
from .dfine_loss import DFINECriterion
from .rtdetr_loss import DetrCriterion
from .rtmdet_loss import RTMDetCriterion
from .ssd_loss import SSDCriterion
from .yolox_loss import YOLOXCriterion

__all__ = [
    "ATSSCriterion",
    "DetrCriterion",
    "RTMDetCriterion",
    "SSDCriterion",
    "YOLOXCriterion",
    "DFINECriterion",
]
