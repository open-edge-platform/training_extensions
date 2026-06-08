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

    **Mixed precision on XPU** uses pure bf16 model weights
    (``model.bfloat16()``) rather than ``torch.amp.autocast``.  This
    avoids autocast's per-op dispatch overhead — which is significant
    for a 260-layer model with thousands of ops per forward pass — while
    providing native bf16 compute performance on XPU.

    We achieve this by:

    1. Letting ``check_amp`` return ``False`` so Ultralytics' internal
       autocast scope is a no-op (``autocast(False)``).
    2. Converting the model to bf16 via ``model.bfloat16()`` after setup.
       bf16 has the same exponent range as fp32 (8 bits), so BatchNorm
       and other precision-sensitive ops work correctly — the crash was
       specific to fp16 (``model.half()``), not bf16.
    3. Casting input images to bf16 in ``_move_batch_to_device`` to
       match the model dtype.
    4. Keeping ``GradScaler`` disabled — bf16 does not need loss scaling.
    5. Keeping ``ModelEMA`` in fp32 for precise exponential averaging
       (bf16's 7-bit mantissa would lose small EMA updates where
       ``1 - decay ≈ 0.0001``).
    6. Converting the model to fp32 for validation when EMA is not
       available, then back to bf16.
    """

    args: Any
    device: torch.device
    model: Any
    scaler: Any

    def _setup_train(self) -> None:  # type: ignore[override]
        """Run base setup with XPU-specific bf16 configuration.

        On XPU we:

        1. Patch ``check_amp`` to return ``False`` so Ultralytics' internal
           ``autocast(self.amp)`` scope is a no-op (``enabled=False``).
        2. Convert the model to bf16 via ``model.bfloat16()`` for native
           XPU bf16 compute without autocast per-op dispatch overhead.
           ``ModelEMA`` (created by the parent as a ``deepcopy`` of the
           fp32 model) stays fp32 — critical for precise averaging.
        3. Replace the CUDA ``GradScaler`` with a disabled one.
        """
        if self.device.type == "xpu":
            import ultralytics.engine.trainer as _trainer_mod

            _orig_check_amp = _trainer_mod.check_amp
            _trainer_mod.check_amp = lambda *_a, **_kw: False
            try:
                super()._setup_train()  # type: ignore[misc]
            finally:
                _trainer_mod.check_amp = _orig_check_amp

            # Convert model to bf16 for native XPU compute.
            # ModelEMA was created by super() as a deepcopy of the fp32
            # model, so it stays fp32 — important for precise averaging.
            self.model.bfloat16()

            self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
            logger.info("XPU: bf16 training (model.bfloat16), EMA fp32, GradScaler disabled")
        else:
            super()._setup_train()  # type: ignore[misc]

    def validate(self) -> Any:  # type: ignore[override]  # noqa: ANN401
        """Run validation in fp32 on XPU when EMA is not available.

        When ``ModelEMA`` is active (the common case), Ultralytics passes
        ``self.ema.ema`` (fp32) to the validator, so no conversion is needed.
        When EMA is not used, the training model (bf16) would be passed
        directly — we convert it to fp32 for validation precision, then
        convert back to bf16 afterward.
        """
        if getattr(self, "device", None) is not None and self.device.type == "xpu":
            has_ema = hasattr(self, "ema") and self.ema is not None
            if not has_ema:
                self.model.float()
                try:
                    return super().validate()  # type: ignore[misc]
                finally:
                    self.model.bfloat16()
        return super().validate()  # type: ignore[misc]

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
                if not 0 <= threshold <= 1:
                    msg = f"Threshold must be between 0 and 1, got {threshold}"
                    raise ValueError(msg)
                if self._get_memory(fraction=True) <= threshold:
                    return
            gc.collect()
            torch.xpu.empty_cache()
            return
        super()._clear_memory(threshold)  # type: ignore[misc]

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move DataModule tensors to the active device.

        Uses ``non_blocking=True`` for CUDA and XPU to overlap the
        host-to-device transfer with compute.  On XPU, images are cast
        to bf16 to match the model's dtype for native bf16 compute.
        """
        non_blocking = self.device.type in ("cuda", "xpu")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=non_blocking)
        if self.device.type == "xpu":
            batch["img"] = batch["img"].bfloat16()
        return batch
