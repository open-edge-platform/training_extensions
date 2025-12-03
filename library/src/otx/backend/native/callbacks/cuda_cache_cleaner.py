# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""CUDA Cache Cleaner callback for memory management during training."""

from __future__ import annotations

import gc
import logging

import torch
from lightning import Callback, LightningModule, Trainer

from otx.data.entity import OTXDataBatch

logger = logging.getLogger(__name__)


class CUDACacheCleaner(Callback):
    """Callback to periodically clean CUDA cache to reduce memory fragmentation.

    This callback can help reduce memory usage by clearing the CUDA cache at strategic
    points during training. However, use with caution as frequent cache clearing can
    slow down training due to memory reallocation overhead.

    Recommended usage:
    - Set clean_on_validation_end=True (default) - Most beneficial, frees eval memory
    - Set clean_on_epoch_end=True only if experiencing OOM between epochs
    - Avoid clean_on_train_batch_end unless absolutely necessary (performance impact)

    Args:
        clean_on_epoch_end: Clean cache at the end of each training epoch.
            Defaults to False.
        clean_on_validation_end: Clean cache after validation. Defaults to True.
        clean_on_train_batch_end: Clean cache after each training batch.
            WARNING: This significantly slows down training. Defaults to False.
        clean_every_n_epochs: Only clean every N epochs (if epoch cleaning enabled).
            Defaults to 1.
        clean_every_n_batches: Only clean every N batches (if batch cleaning enabled).
            Defaults to 100.
        run_gc: Also run Python garbage collection before clearing cache.
            Defaults to True.
        log_memory: Log memory usage before/after cleaning. Defaults to False.
    """

    def __init__(
        self,
        clean_on_epoch_end: bool = False,
        clean_on_validation_end: bool = True,
        clean_on_train_batch_end: bool = False,
        clean_every_n_epochs: int = 1,
        clean_every_n_batches: int = 100,
        run_gc: bool = True,
        log_memory: bool = False,
    ) -> None:
        super().__init__()
        self.clean_on_epoch_end = clean_on_epoch_end
        self.clean_on_validation_end = clean_on_validation_end
        self.clean_on_train_batch_end = clean_on_train_batch_end
        self.clean_every_n_epochs = clean_every_n_epochs
        self.clean_every_n_batches = clean_every_n_batches
        self.run_gc = run_gc
        self.log_memory = log_memory

    def _clean_cache(self, stage: str) -> None:
        """Clean CUDA cache and optionally run garbage collection.

        Args:
            stage: Description of when cleaning is happening (for logging).
        """
        if not torch.cuda.is_available():
            return

        if self.log_memory:
            before_allocated = torch.cuda.memory_allocated() / 1024**3
            before_reserved = torch.cuda.memory_reserved() / 1024**3

        if self.run_gc:
            gc.collect()

        torch.cuda.empty_cache()

        if self.log_memory:
            after_allocated = torch.cuda.memory_allocated() / 1024**3
            after_reserved = torch.cuda.memory_reserved() / 1024**3
            freed = before_reserved - after_reserved
            logger.info(
                f"[{stage}] CUDA cache cleaned. "
                f"Allocated: {before_allocated:.2f}GB -> {after_allocated:.2f}GB, "
                f"Reserved: {before_reserved:.2f}GB -> {after_reserved:.2f}GB, "
                f"Freed: {freed:.2f}GB"
            )

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Clean cache at the end of training epoch if enabled."""
        if self.clean_on_epoch_end and (trainer.current_epoch + 1) % self.clean_every_n_epochs == 0:
            self._clean_cache(f"epoch_{trainer.current_epoch}_end")

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Clean cache after validation if enabled."""
        if self.clean_on_validation_end:
            self._clean_cache("validation_end")

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OTXDataBatch,
        batch: OTXDataBatch,
        batch_idx: int,
    ) -> None:
        """Clean cache after training batch if enabled (use with caution)."""
        if self.clean_on_train_batch_end and (batch_idx + 1) % self.clean_every_n_batches == 0:
            self._clean_cache(f"batch_{batch_idx}_end")
