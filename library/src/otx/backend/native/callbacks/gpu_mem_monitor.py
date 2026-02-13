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
    """Monitor GPU memory hook.

    This callback monitors GPU memory usage and logs it.
    """

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
        allocated = int(device_stats.get("allocated_bytes.all.current", torch.cuda.memory_allocated(device)))
        reserved = int(device_stats.get("reserved_bytes.all.current", torch.cuda.memory_reserved(device)))

        allocated_gib = round(allocated / 1024**3, 2)
        reserved_gib = round(reserved / 1024**3, 2)

        pl_module.log(
            name="gpu_mem_allocated_gib",
            value=allocated_gib,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        pl_module.log(
            name="gpu_mem_reserved_gib",
            value=reserved_gib,
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
