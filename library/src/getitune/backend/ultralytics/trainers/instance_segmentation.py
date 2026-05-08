# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom segmentation trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

from copy import copy
from typing import Any

from ultralytics.models.yolo.segment import SegmentationTrainer as _UltralyticsSegmentationTrainer
from ultralytics.models.yolo.segment import SegmentationValidator as _UltralyticsSegmentationValidator

from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin
from getitune.backend.ultralytics.validators.instance_segmentation import SegmentationValidator

from .base import GetiTuneDataBridgeMixin


class SegmentationTrainer(GetiTuneDataBridgeMixin, XPUAwareTrainerMixin, _UltralyticsSegmentationTrainer):
    """Instance-segmentation trainer that routes data through a getitune DataModule.

    Mirrors :class:`DetectionTrainer` but passes ``include_masks=True``
    to the adapter.  Falls back to default Ultralytics loading otherwise.

    Inherits :class:`XPUAwareTrainerMixin` for Intel XPU device support.
    """

    _include_masks: bool = True

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Use upstream preprocessing unless the DataModule bridge is active."""
        if not self._use_getitune_data:
            return _UltralyticsSegmentationTrainer.preprocess_batch(self, batch)
        return self._move_batch_to_device(batch)

    def set_model_attributes(self) -> None:
        """Set model attributes; disable augmentations and overlap mask when using DataModule."""
        super().set_model_attributes()
        if self._use_getitune_data and hasattr(self.args, "overlap_mask"):
            self.args.overlap_mask = False

    def get_validator(self) -> _UltralyticsSegmentationValidator:
        """Return a custom segmentation validator that handles pre-normalised images."""
        if not self._use_getitune_data:
            return super().get_validator()

        self.loss_names = ["box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"]
        validator = SegmentationValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator._datamodule = self._datamodule  # noqa: SLF001
        return validator
