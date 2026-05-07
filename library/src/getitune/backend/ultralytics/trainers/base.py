# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base trainer mixin with shared logic for the getitune DataModule bridge."""

from __future__ import annotations

import multiprocessing
from typing import TYPE_CHECKING, Any

from torch.utils.data import DataLoader

from getitune.backend.ultralytics.data.adapter import UltralyticsDatasetAdapter
from getitune.backend.ultralytics.data.collate import ultralytics_collate_fn

if TYPE_CHECKING:
    from getitune.data.module import DataModule

_MP_CONTEXT = multiprocessing.get_context("spawn")


class GetiTuneDataBridgeMixin:
    """Mixin providing shared DataModule bridge logic for Ultralytics trainers.

    When ``_datamodule`` is set (via the engine's dynamic subclass), data
    loading uses the getitune pipeline and ``preprocess_batch`` skips the
    default ``/255`` normalisation (images are already float32 [0,1]).

    Subclasses must set :attr:`_include_masks` to control whether the adapter
    includes instance masks.
    """

    _datamodule: DataModule | None = None
    _include_masks: bool = False

    @property
    def _use_getitune_data(self) -> bool:
        """Whether the trainer is operating with a getitune DataModule."""
        return self._datamodule is not None

    def get_dataset(self) -> dict[str, Any]:
        """Build data config dict from DataModule or fall back to YAML."""
        if not self._use_getitune_data:
            return super().get_dataset()  # type: ignore[misc]

        li = self._datamodule.label_info  # type: ignore[union-attr]
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
        if not self._use_getitune_data:
            return super().build_dataset(img_path, mode, batch)  # type: ignore[misc]

        subset_key = "train" if mode == "train" else "val"
        vision_dataset = self._datamodule.subsets[subset_key]  # type: ignore[union-attr]
        return UltralyticsDatasetAdapter(vision_dataset, include_masks=self._include_masks)

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ) -> DataLoader:
        """Build a DataLoader from the adapter dataset."""
        if not self._use_getitune_data:
            return super().get_dataloader(dataset_path, batch_size, rank, mode)  # type: ignore[misc]

        dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        nw = self.args.workers  # type: ignore[attr-defined]
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

    def set_model_attributes(self) -> None:
        """Set model attributes; disable Ultralytics augmentations when using DataModule."""
        super().set_model_attributes()  # type: ignore[misc]
        if self._use_getitune_data:
            self._disable_ultralytics_augmentations()

    def set_class_weights(self) -> None:
        """Skip class-weight computation (adapter has no ``labels`` attr)."""
        if self._use_getitune_data:
            return
        super().set_class_weights()  # type: ignore[misc]

    def plot_training_labels(self) -> None:
        """Skip label plotting (adapter has no ``labels`` attr)."""
        if self._use_getitune_data:
            return
        super().plot_training_labels()  # type: ignore[misc]

    def auto_batch(self) -> int:
        """Skip auto-batch when using DataModule."""
        if self._use_getitune_data:
            return self.batch_size  # type: ignore[attr-defined]
        return super().auto_batch()  # type: ignore[misc]

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
            if hasattr(self.args, attr):  # type: ignore[attr-defined]
                setattr(self.args, attr, 0.0)  # type: ignore[attr-defined]
        # Prevent train_loader.reset() call — plain DataLoader has no reset().
        if hasattr(self.args, "close_mosaic"):  # type: ignore[attr-defined]
            self.args.close_mosaic = 0  # type: ignore[attr-defined]
