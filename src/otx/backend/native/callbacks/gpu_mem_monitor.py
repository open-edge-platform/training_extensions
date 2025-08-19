# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Monitor GPU memory hook."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lightning.pytorch.callbacks.callback import Callback

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer


class GPUMemMonitor(Callback):
    """Monitor GPU memory hook."""

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
        if device.type in ["cpu", "xpu"]:
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
