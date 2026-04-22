# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Detection validator that skips /255 normalization."""

from __future__ import annotations

from typing import Any

import torch
from ultralytics.models.yolo.detect import DetectionValidator as _UltralyticsDetectionValidator


class DetectionValidator(_UltralyticsDetectionValidator):
    """Detection validator for getitune data bridge.

    Overrides ``preprocess()`` to skip the ``/255`` normalization since
    getitune supplies images as ``float32 [0, 1]`` already.
    """

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensors to device; skip ``/255``."""
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        # Honor half-precision flag without dividing by 255.
        if self.args.half:
            batch["img"] = batch["img"].half()
        return batch
