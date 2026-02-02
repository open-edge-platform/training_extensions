# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""GPU Augmentation Callback for applying Kornia augmentations on GPU."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING

from lightning import Callback

from otx.data.augmentation import GPUAugmentationPipeline
from otx.data.entity.sample import OTXSampleBatch
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer

    from otx.config.data import SubsetConfig


class GPUAugmentationCallback(Callback):
    """Callback to apply GPU augmentations using Kornia.

    This callback applies GPU-accelerated augmentations from the GPUAugmentationPipeline
    to batches during training and optionally during validation/testing.

    Key features:
    - Applies augmentations after batch is transferred to GPU
    - Extracts normalization parameters and updates model's data_input_params
    - Supports separate train/val pipelines with different augmentation configs
    - Automatically handles bboxes, masks, keypoints based on batch content

    Args:
        train_config: SubsetConfig for training augmentations.
        val_config: SubsetConfig for validation augmentations (optional).
        apply_on_val: Whether to apply augmentations during validation.
            Defaults to True (normalization is usually applied on val too).
        apply_on_test: Whether to apply augmentations during testing.
            Defaults to True (normalization is usually applied on test too).

    Example:
        >>> callback = GPUAugmentationCallback(
        ...     train_config=train_subset_config,
        ...     val_config=val_subset_config,
        ... )
        >>> trainer = Trainer(callbacks=[callback])
    """

    _DATA_KEYS_BY_TASK: dict[OTXTaskType, list[str]] = {
        OTXTaskType.MULTI_CLASS_CLS: ["label"],
        OTXTaskType.MULTI_LABEL_CLS: ["label"],
        OTXTaskType.H_LABEL_CLS: ["label"],
        OTXTaskType.DETECTION: ["bbox_xyxy", "label"],
        OTXTaskType.INSTANCE_SEGMENTATION: ["bbox_xyxy", "mask", "label"],
        OTXTaskType.KEYPOINT_DETECTION: ["keypoints", "label"],
        OTXTaskType.SEMANTIC_SEGMENTATION: ["mask"],
    }

    def __init__(
        self,
        train_config: SubsetConfig | None = None,
        val_config: SubsetConfig | None = None,
        test_config: SubsetConfig | None = None,
        apply_on_val: bool = True,
        apply_on_test: bool = True,
    ) -> None:
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config if test_config is not None else val_config
        self.apply_on_val = apply_on_val
        self.apply_on_test = apply_on_test

        self._train_pipeline: GPUAugmentationPipeline | None = None
        self._val_pipeline: GPUAugmentationPipeline | None = None
        self._test_pipeline: GPUAugmentationPipeline | None = None


    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup the GPU augmentation pipelines.

        This is called once when the trainer is setup.
        """
        data_keys = ["input", *self._DATA_KEYS_BY_TASK.get(pl_module.task, [])]
        if self.train_config is not None:
            self._train_pipeline = GPUAugmentationPipeline.from_config(self.train_config, data_keys=data_keys)
            log.info(f"GPU train augmentation pipeline:\n{self._train_pipeline}")

        if self.val_config is not None:
            self._val_pipeline = GPUAugmentationPipeline.from_config(self.val_config, data_keys=data_keys)
            log.info(f"GPU val augmentation pipeline:\n{self._val_pipeline}")

        if self.test_config is not None:
            self._test_pipeline = GPUAugmentationPipeline.from_config(self.test_config, data_keys=data_keys)
            log.info(f"GPU test augmentation pipeline:\n{self._test_pipeline}")

        # Update model's normalization params from GPU pipeline
        self._update_model_normalization(pl_module)

    def _update_model_normalization(self, pl_module: LightningModule) -> None:
        """Update model's data_input_params with normalization from GPU pipeline.

        If normalization is in the GPU pipeline, we need to update the model's
        mean/std so that export and inference use the correct values.
        If both model and pipeline have None, set defaults (0,0,0) and (1,1,1).
        """
        # Since we use mean, std values for model export
        # We derive mean, std from test pipeline as priority
        pipeline = self._test_pipeline or self._train_pipeline

        pipeline_mean = pipeline.mean if pipeline else None
        pipeline_std = pipeline.std if pipeline else None
        model_mean = getattr(pl_module.data_input_params, "mean", None)
        model_std = getattr(pl_module.data_input_params, "std", None)

        # pipeline > model > default
        pl_module.data_input_params.mean = pipeline_mean or model_mean or (0, 0, 0)
        pl_module.data_input_params.std = pipeline_std or model_std or (1, 1, 1)

        # log update
        if any([model_mean != pl_module.data_input_params.mean, model_std != pl_module.data_input_params.std]):
            log.info(f"Updated model mean: {model_mean} -> {pl_module.data_input_params.mean}")
            log.info(f"Updated model std: {model_std} -> {pl_module.data_input_params.std}")

    def _apply_pipeline(
        self,
        pipeline: GPUAugmentationPipeline,
        batch: OTXSampleBatch,
    ) -> None:
        """Apply GPU augmentation pipeline to batch (in-place).

        Automatically determines which data to transform based on batch content.
        Kornia decides whether to modify labels based on the augmentations used.

        Args:
            pipeline: GPUAugmentationPipeline to apply.
            batch: OTXSampleBatch to transform.
        """
        # Move pipeline to same device as batch
        device = batch.images.device if hasattr(batch.images, "device") else None
        if device is not None:
            pipeline = pipeline.to(device)

        # Apply pipeline - returns dict with augmented data
        # Labels are included in data_keys, so Kornia will process them if applicable
        result = pipeline(
            batch.images,
            labels=batch.labels,
            bboxes=batch.bboxes,
            masks=batch.masks,
            keypoints=batch.keypoints,
        )

        # Update batch in-place with augmented data
        batch.images = result["images"]
        if result.get("labels") is not None:
            batch.labels = result["labels"]
        if result.get("bboxes") is not None:
            batch.bboxes = result["bboxes"]
        if result.get("masks") is not None:
            batch.masks = result["masks"]
        if result.get("keypoints") is not None:
            batch.keypoints = result["keypoints"]

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: OTXSampleBatch,
        batch_idx: int,
    ) -> None:
        """Apply GPU augmentations to training batch."""
        if self._train_pipeline is None:
            return

        self._apply_pipeline(self._train_pipeline, batch)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: OTXSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply GPU augmentations to validation batch."""
        if not self.apply_on_val or self._val_pipeline is None:
            return

        self._apply_pipeline(self._val_pipeline, batch)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: OTXSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Apply GPU augmentations to test batch."""
        if not self.apply_on_test or self._test_pipeline is None:
            return

        self._apply_pipeline(self._test_pipeline, batch)
