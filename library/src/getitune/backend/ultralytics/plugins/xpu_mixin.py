# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin that adds XPU device support to Ultralytics trainers.

Overrides device-touching methods (GradScaler, autocast, memory utilities)
so that training runs on a single Intel XPU card via upstream PyTorch.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class XPUAwareTrainerMixin:
    """Mixin adding XPU branches to Ultralytics trainer lifecycle methods.

    Must appear **before** the Ultralytics trainer in the MRO so that
    ``super()`` dispatches correctly::

        class DetectionTrainer(XPUAwareTrainerMixin, _UltralyticsDetectionTrainer):
            ...
    """

    args: Any
    device: torch.device
    scaler: Any

    def train(self) -> None:  # type: ignore[override]
        """Wrap the training loop in XPU autocast when running on an Intel GPU.

        Ultralytics' internal ``autocast("cuda")`` is a no-op for XPU
        tensors, so we enable ``torch.amp.autocast("xpu")`` at a higher
        level.  CUDA and CPU paths are unchanged.
        """
        if getattr(self, "device", None) is not None and self.device.type == "xpu":
            amp_enabled = bool(getattr(self.args, "amp", True))
            logger.info(f"Enabling XPU autocast (bf16, enabled={amp_enabled}) for training")
            with torch.amp.autocast("xpu", enabled=amp_enabled, dtype=torch.bfloat16):
                return super().train()  # type: ignore[misc]
        return super().train()  # type: ignore[misc]

    def _setup_train(self) -> None:  # type: ignore[override]
        """Run base setup, then replace the CUDA GradScaler on XPU.

        The parent creates ``GradScaler("cuda", enabled=self.amp)``.
        On XPU we replace it with a disabled scaler because bf16 does
        not require loss scaling.
        """
        super()._setup_train()  # type: ignore[misc]
        if self.device.type == "xpu":
            self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
            logger.debug("Replaced GradScaler with disabled XPU scaler")

    def _get_memory(self, fraction: bool = False) -> float:  # type: ignore[override]
        """Return reserved GPU memory (GiB) or fraction, with an XPU branch."""
        if self.device.type == "xpu":
            memory = torch.xpu.memory_reserved(self.device)
            if fraction:
                total = torch.xpu.get_device_properties(self.device).total_memory
                return (memory / total) if total > 0 else 0.0
            return memory / 2**30
        return super()._get_memory(fraction)  # type: ignore[misc]

    def _clear_memory(self, threshold: float | None = None) -> None:  # type: ignore[override]
        """Clear GPU caches, with an XPU branch."""
        if self.device.type == "xpu":
            if threshold is not None:
                assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."  # noqa: S101
                if self._get_memory(fraction=True) <= threshold:
                    return
            gc.collect()
            torch.xpu.empty_cache()
            return
        super()._clear_memory(threshold)  # type: ignore[misc]

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move DataModule tensors to the active XPU device."""
        non_blocking = self.device.type == "xpu"
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=non_blocking)
        return batch
