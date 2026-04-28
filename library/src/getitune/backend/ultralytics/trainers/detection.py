# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom detection trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

import multiprocessing
from copy import copy
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect import DetectionTrainer as _UltralyticsDetectionTrainer
from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import ultralytics_collate_fn
from getitune.backend.ultralytics.trainers.xpu_mixin import XPUAwareTrainerMixin
from getitune.backend.ultralytics.validators.detection import DetectionValidator

if TYPE_CHECKING:
    from getitune.data.module import OTXDataModule as DataModule

_MP_CONTEXT = multiprocessing.get_context("spawn")


class DetectionTrainer(XPUAwareTrainerMixin, _UltralyticsDetectionTrainer):
    """Detection trainer that routes data through a getitune DataModule.

    When ``_datamodule`` is set (via the engine's dynamic subclass),
    data loading uses the getitune pipeline and ``preprocess_batch``
    skips ``/255``.  Falls back to default Ultralytics loading otherwise.

    Inherits :class:`XPUAwareTrainerMixin` for Intel XPU device support.
    """

    _datamodule: DataModule | None = None

    # ------------------------------------------------------------------
    # Data overrides
    # ------------------------------------------------------------------

    def get_dataset(self) -> dict[str, Any]:
        """Build data config dict from DataModule or fall back to YAML."""
        if self._datamodule is None:
            return super().get_dataset()

        li = self._datamodule.label_info
        names = dict(enumerate(li.label_names))
        return {
            "train": "datamodule://train",
            "val": "datamodule://val",
            "nc": li.num_classes,
            "names": names,
            "channels": 3,
        }

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None) -> UltralyticsDatasetAdapter:
        """Return adapter wrapping the appropriate DataModule subset.

        Args:
            img_path: Ignored when DataModule is set.
            mode: ``"train"`` or ``"val"``.
            batch: Batch size (unused by adapter).
        """
        if self._datamodule is None:
            return super().build_dataset(img_path, mode, batch)  # type: ignore[return-value]

        subset_key = "train" if mode == "train" else "val"
        vision_dataset = self._datamodule.subsets[subset_key]
        return UltralyticsDatasetAdapter(vision_dataset, include_masks=False)

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Build a DataLoader from the adapter dataset."""
        if self._datamodule is None:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        nw = self.args.workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=ultralytics_collate_fn,
            pin_memory=True,
            drop_last=mode == "train",
            multiprocessing_context=_MP_CONTEXT if nw > 0 else None,
            persistent_workers=nw > 0,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Use upstream preprocessing unless the DataModule bridge is active."""
        if self._datamodule is None:
            return _UltralyticsDetectionTrainer.preprocess_batch(self, batch)
        return self._move_batch_to_device(batch)

    def set_model_attributes(self) -> None:
        """Set model attributes; disable Ultralytics augmentations when using DataModule."""
        super().set_model_attributes()
        if self._datamodule is not None:
            self._disable_ultralytics_augmentations()

    def set_class_weights(self) -> None:
        """Skip class-weight computation (adapter has no ``labels`` attr)."""
        if self._datamodule is not None:
            return
        super().set_class_weights()

    def plot_training_labels(self) -> None:
        """Skip label plotting (adapter has no ``labels`` attr)."""
        if self._datamodule is not None:
            return
        super().plot_training_labels()

    def get_validator(self) -> _UltralyticsDetectionValidator:
        """Return a custom validator that skips /255."""
        if self._datamodule is None:
            return super().get_validator()

        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        validator = DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator._datamodule = self._datamodule  # noqa: SLF001
        return validator

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _disable_ultralytics_augmentations(self) -> None:
        """Zero out Ultralytics augmentation hyperparams."""
        for attr in (
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "flipud",
            "fliplr",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
        ):
            if hasattr(self.args, attr):
                setattr(self.args, attr, 0.0)
        # Prevent train_loader.reset() call — plain DataLoader has no reset().
        if hasattr(self.args, "close_mosaic"):
            self.args.close_mosaic = 0

    def auto_batch(self) -> int:
        """Skip auto-batch when using DataModule."""
        if self._datamodule is not None:
            return self.batch_size
        return super().auto_batch()
