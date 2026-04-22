# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom Ultralytics validator subclasses that skip /255 normalization."""

from .detection import DetectionValidator
from .instance_segmentation import SegmentationValidator

__all__ = [
    "DetectionValidator",
    "SegmentationValidator",
]
