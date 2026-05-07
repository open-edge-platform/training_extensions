# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Ultralytics trainer subclasses for getitune data bridge."""

from .base import GetiTuneDataBridgeMixin
from .detection import DetectionTrainer
from .instance_segmentation import SegmentationTrainer

__all__ = [
    "DetectionTrainer",
    "GetiTuneDataBridgeMixin",
    "SegmentationTrainer",
]
