# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detection validator for the getitune data bridge."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator

from .base import GetiTuneValidatorMixin


class DetectionValidator(GetiTuneValidatorMixin, _UltralyticsDetectionValidator):
    """Detection validator for the getitune data bridge."""

    _include_masks: bool = False
