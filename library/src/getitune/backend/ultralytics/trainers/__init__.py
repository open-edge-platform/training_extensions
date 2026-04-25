# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Ultralytics trainer subclasses for getitune data bridge."""

from .detection import DetectionTrainer
from .instance_segmentation import SegmentationTrainer
from .xpu_mixin import XPUAwareTrainerMixin

__all__ = [
    "DetectionTrainer",
    "SegmentationTrainer",
    "XPUAwareTrainerMixin",
]
