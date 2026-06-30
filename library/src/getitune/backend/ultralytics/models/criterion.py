# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Loss criteria and head utilities for Ultralytics-backed classification models."""

from __future__ import annotations

from typing import Any, Callable, cast

import torch
from torch import nn
from torch.nn import functional
from ultralytics.nn.modules.head import Classify


class MultiLabelClassificationLoss(nn.Module):
    """Binary cross-entropy loss for multi-label classification.

    Mirrors the upstream ``v8ClassificationLoss`` call signature so it can be
    dropped in as ``model.criterion`` for YOLO classification models.
    """

    def forward(
        self,
        preds: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute BCE loss between raw logits and multi-hot targets.

        Args:
            preds: Raw classification logits, or a list/tuple containing them.
            batch: Ultralytics batch dict with ``"cls"`` multi-hot targets.

        Returns:
            Tuple of (loss, detached loss).
        """
        logits = preds[0] if isinstance(preds, (list, tuple)) else preds
        targets = batch["cls"].float()
        loss = functional.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
        return loss, loss.detach()


def _multilabel_classify_forward(module: Classify) -> Callable[[torch.Tensor | list[torch.Tensor]], torch.Tensor]:
    """Return a forward function that emits sigmoid at inference time.

    Training and export modes continue to emit raw logits so that the BCE loss
    and ModelAPI sigmoid post-processing work correctly.
    """

    def forward(x: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = module.linear(module.drop(module.pool(module.conv(x)).flatten(1)))
        if module.training or module.export:
            return x
        return x.sigmoid()

    return forward


def configure_multilabel_classify_head(model: torch.nn.Module) -> None:
    """Replace every ``Classify`` head's inference softmax with sigmoid.

    This is applied both when the model wrapper builds the YOLO object for
    inference/export and when the trainer rebuilds a fresh training model.
    """
    for module in model.modules():
        if isinstance(module, Classify):
            module.forward = cast("Callable[..., torch.Tensor]", _multilabel_classify_forward(module))
