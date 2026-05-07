# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segmentation validator for the getitune data bridge."""

from __future__ import annotations

from ultralytics.models.yolo.segment import SegmentationValidator as _UltralyticsSegmentationValidator

from .base import GetiTuneValidatorMixin


class SegmentationValidator(GetiTuneValidatorMixin, _UltralyticsSegmentationValidator):
    """Instance-segmentation validator for the getitune data bridge.

    Images are already float32 [0,1] from the DataModule pipeline.
    When ``_datamodule`` is set and called without a trainer, runs
    standalone validation bypassing Ultralytics' YAML data parsing.
    """

    _include_masks: bool = True
