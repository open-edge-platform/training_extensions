# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX engine components."""


from lightning import Trainer
from lightning.pytorch.callbacks import DeviceStatsMonitor


def _prefix_metric_keys(
    metrics_dict: dict[str, float],
    prefix: str,
    separator: str,
) -> dict[str, float]:
    return {prefix + separator + k: v for k, v in metrics_dict.items()}


class GPUMonitor(DeviceStatsMonitor):
    def _get_and_log_device_stats(self, trainer: Trainer, key: str) -> None:
        if not trainer._logger_connector.should_update_logs:
            return

        device = trainer.strategy.root_device
        if device.type == "cpu":
            # cpu stats are disabled
            return

        device_stats = trainer.accelerator.get_device_stats(device)

        _device_stats = {
            "allocated.GB.all.current": device_stats["allocated_bytes.all.current"] / 1024 ** 3,
            "allocated.GB.all.peak": device_stats["allocated_bytes.all.peak"] / 1024 ** 3,
        }

        for logger in trainer.loggers:
            separator = logger.group_separator
            prefixed_device_stats = _prefix_metric_keys(_device_stats, f"{self.__class__.__qualname__}.{key}", separator)
            logger.log_metrics(prefixed_device_stats, step=trainer.fit_loop.epoch_loop._batches_that_stepped)