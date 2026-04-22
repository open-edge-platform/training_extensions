# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom segmentation trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ultralytics.models.yolo.segment import SegmentationTrainer as _UltralyticsSegmentationTrainer

if TYPE_CHECKING:
    from getitune.data.module import DataModule


class SegmentationTrainer(_UltralyticsSegmentationTrainer):
    """Instance-segmentation trainer with a getitune DataModule slot.

    Stores a reference to a :class:`~getitune.data.module.DataModule` so that
    Phase 2 overrides (``build_dataset``, ``get_dataloader``,
    ``preprocess_batch``) can pull data from getitune's augmentation pipeline.

    When no DataModule is bound (``_datamodule is None``), falls back to
    default Ultralytics data loading (requires a data YAML path).
    """

    _datamodule: DataModule | None = None
