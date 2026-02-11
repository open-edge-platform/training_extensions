# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Monitor GPU memory hook."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from lightning.pytorch.callbacks.callback import Callback

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


class GPUMemMonitor(Callback):
    """Monitor GPU memory hook with optional CUDA cache cleanup.

    This callback monitors GPU memory usage and optionally clears CUDA cache
    to reduce memory consumption during training.
    """

    def __init__(self, iter_to_clean: int = 50) -> None:
        """Initialize GPU memory monitor.

        Args:
            iter_to_clean: Clean CUDA cache after a certain number of iterations.
                Set to -1 to disable cache cleaning. Defaults to 50.
        """
        super().__init__()
        self.iter_to_clean = iter_to_clean

    def _get_and_log_device_stats(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Get and log current GPU memory usage.

        Args:
            trainer (Trainer): pl trainer.
            pl_module (LightningModule): pl module.
            batch_size (int): batch size.
        """
        device = trainer.strategy.root_device
        if device.type in ["cpu", "xpu", "mps"]:
            return

        device_stats = trainer.accelerator.get_device_stats(device)
        allocated = device_stats["allocated_bytes.all.current"]
        reserved = device_stats["reserved_bytes.all.current"]
        used_memory = (allocated + reserved) / 1024**3  # convert to GiB
        used_memory = round(used_memory, 2)

        pl_module.log(
            name="gpu_mem",
            value=used_memory,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Log GPU memory usage at the start of every train batch.

        Args:
            trainer (Trainer): pl trainer.
            pl_module (LightningModule): pl module.
            batch (Any): current batch.
            batch_idx (int): current batch index.
        """
        self._get_and_log_device_stats(
            trainer,
            pl_module,
        )

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,  # noqa: ANN401
        batch: Any,  # noqa: ANN401
        batch_idx: int,
    ) -> None:
        """Clean up CUDA cache every 50 train iterations if enabled.

        Args:
            trainer (Trainer): pl trainer.
            pl_module (LightningModule): pl module.
            outputs (Any): batch outputs.
            batch (Any): current batch.
            batch_idx (int): current batch index.
        """
        if torch.cuda.is_available() and self.iter_to_clean != -1 and (batch_idx + 1) % self.iter_to_clean == 0:
            torch.cuda.empty_cache()

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,  # noqa: ANN401
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Log GPU memory usage at the start of every validation batch.

        Args:
            trainer (Trainer): pl trainer.
            pl_module (LightningModule): pl module.
            batch (Any): current batch.
            batch_idx (int): current batch index.
            dataloader_idx (int, optional): dataloader index. Defaults to 0.
        """
        self._get_and_log_device_stats(
            trainer,
            pl_module,
        )
