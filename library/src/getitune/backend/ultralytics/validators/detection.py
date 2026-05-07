# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detection validator for the getitune data bridge."""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator

from .base import GetiTuneValidatorMixin


class DetectionValidator(GetiTuneValidatorMixin, _UltralyticsDetectionValidator):
    """Detection validator for the getitune data bridge.

    Images are already float32 [0,1] from the DataModule pipeline.
    When ``_datamodule`` is set and called without a trainer, runs
    standalone validation bypassing Ultralytics' YAML data parsing.
    """

    _include_masks: bool = False
