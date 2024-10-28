# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX GPU Monitor Hook."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lightning.pytorch.callbacks import DeviceStatsMonitor

if TYPE_CHECKING:
    from lightning import Trainer


def _prefix_metric_keys(
    metrics_dict: dict[str, float],
    prefix: str,
    separator: str,
) -> dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}


class GPUMonitor(DeviceStatsMonitor):
    """Monitor GPU stats.

    # TODO(Eugene): only supports CUDA?

    Args:
        DeviceStatsMonitor (_type_): _description_
    """

    def _get_and_log_device_stats(self, trainer: Trainer, key: str) -> None:
        if not trainer._logger_connector.should_update_logs:
            return

        device = trainer.strategy.root_device
        if device.type == "cpu":
            # cpu stats are disabled
            return

        device_stats = trainer.accelerator.get_device_stats(device)

        _device_stats = {
            "allocated.GB.all.current": device_stats["allocated_bytes.all.current"] / 1024**3,
            "allocated.GB.all.peak": device_stats["allocated_bytes.all.peak"] / 1024**3,
        }

        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(
                _device_stats,
                f"{self.__class__.__qualname__}.{key}",
                separator,
            )
            logger.log_metrics(prefixed_device_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)
