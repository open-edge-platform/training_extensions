# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Custom detection trainer bridging getitune DataModule to Ultralytics."""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect import DetectionTrainer as _UltralyticsDetectionTrainer
from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import ultralytics_collate_fn
from getitune.backend.ultralytics.validators.detection import DetectionValidator

if TYPE_CHECKING:
    from getitune.data.module import DataModule


class DetectionTrainer(_UltralyticsDetectionTrainer):
    """Detection trainer that can bridge a getitune DataModule.

    When ``_datamodule`` is set (via the engine's dynamic subclass factory),
    all data loading is routed through the getitune pipeline.  Ultralytics'
    own augmentations are disabled and ``preprocess_batch`` skips ``/255``
    because the data is already ``float32 [0, 1]``.

    When ``_datamodule is None``, falls back to default Ultralytics data
    loading (requires a data YAML path).
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
        """Return adapter wrapping the appropriate DataModule subset."""
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
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.workers,
            collate_fn=ultralytics_collate_fn,
            pin_memory=True,
            drop_last=mode == "train",
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensors to device; skip ``/255`` (data is already float32 [0,1])."""
        if self._datamodule is None:
            return super().preprocess_batch(batch)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        # Images are already float32 [0, 1] — do NOT divide by 255.
        return batch

    def set_model_attributes(self) -> None:
        """Set model attributes; disable Ultralytics augmentations when using DataModule."""
        super().set_model_attributes()
        if self._datamodule is not None:
            self._disable_ultralytics_augmentations()

    def set_class_weights(self) -> None:
        """Skip class-weight computation when using DataModule adapter.

        Upstream reads ``self.train_loader.dataset.labels`` which doesn't
        exist on our adapter.  Disabled for v1.
        """
        if self._datamodule is not None:
            return
        super().set_class_weights()

    def plot_training_labels(self) -> None:
        """Skip label plotting when using DataModule adapter.

        Upstream reads ``self.train_loader.dataset.labels`` which doesn't
        exist on our adapter.
        """
        if self._datamodule is not None:
            return
        super().plot_training_labels()

    def get_validator(self) -> _UltralyticsDetectionValidator:
        """Return a custom validator that skips /255."""
        if self._datamodule is None:
            return super().get_validator()

        self.loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        return DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )

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

    def auto_batch(self) -> int:
        """Skip auto-batch when using DataModule (batch size comes from config)."""
        if self._datamodule is not None:
            return self.batch_size
        return super().auto_batch()
