# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mixin that adds XPU device support to Ultralytics trainers.

Ultralytics 8.4.x has no native XPU support — ``select_device("xpu")``
raises ``ValueError``, and ``BaseTrainer`` hardcodes CUDA in GradScaler,
autocast, memory utilities, and OOM handling.  This mixin overrides the
device-touching methods so that training runs on a single Intel XPU card
via upstream PyTorch (no IPEX dependency).

Design notes:

* **bf16-mixed precision** — Intel GPUs use ``bfloat16`` (not ``fp16``).
  ``GradScaler`` is disabled on XPU because bf16 does not need loss scaling.
* **Autocast** — Ultralytics' ``autocast(self.amp)`` helper defaults to
  ``device="cuda"``, which is a no-op for XPU tensors.  We wrap the
  top-level ``train()`` call in ``torch.amp.autocast("xpu", ...)`` so that
  the entire training loop runs under XPU mixed precision.
* **Single-card only** — multi-card / DDP on XPU is explicitly deferred.
* **OOM handling** — Ultralytics catches ``torch.cuda.OutOfMemoryError``
  for auto batch-size retry.  XPU OOM surfaces as ``RuntimeError``.  We
  do *not* override ``_do_train()`` (too large / fragile); XPU OOM will
  propagate as-is.  Acceptable for v1.
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

    # ------------------------------------------------------------------
    # Training entry point — XPU autocast wrapper
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Device setup — GradScaler fix
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Memory utilities
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Preprocessing helper
    # ------------------------------------------------------------------

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensors to device with ``non_blocking`` on CUDA and XPU.

        This helper is for the DataModule bridge path only. Task-specific
        fallback preprocessing must continue to use the upstream Ultralytics
        trainer implementation so that image normalization and multi-scale
        resizing remain unchanged for new model releases.
        """
        non_blocking = self.device.type in ("cuda", "xpu")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=non_blocking)
        return batch
