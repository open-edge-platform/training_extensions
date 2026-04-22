# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Ultralytics model definitions for getitune."""

from .base import UltralyticsModel
from .detection import UltralyticsDetectionModel
from .instance_segmentation import UltralyticsInstSegModel

__all__ = [
    "UltralyticsDetectionModel",
    "UltralyticsInstSegModel",
    "UltralyticsModel",
]
