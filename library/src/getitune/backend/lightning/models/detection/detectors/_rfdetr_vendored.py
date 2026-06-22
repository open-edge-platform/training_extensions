# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Parts of this file are vendored from the rfdetr library (Apache-2.0):
#   https://github.com/roboflow/rf-detr
#
# Vendored to avoid pulling rfdetr's ``train`` extra, which includes
# AGPL-licensed ``albumentations`` and a second PyTorch Lightning stack.
# These helpers are small, self-contained, and used only by getitune's
# integration layer.

"""Minimal vendored helpers from rfdetr 1.8.0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

from rfdetr.models.backbone import Joiner

if TYPE_CHECKING:
    from torch import nn


class _OptimizerArgs(Protocol):
    """Minimal interface for the args namespace consumed by ``get_param_dict``."""

    lr: float
    lr_component_decay: float


def get_param_dict(args: _OptimizerArgs, model_without_ddp: nn.Module) -> list[dict[str, Any]]:
    """Build parameter groups with correct lr and weight_decay per layer type.

    (Vendored from ``rfdetr.training.param_groups`` to avoid importing the
    ``rfdetr.training`` package, which triggers train-extra dependencies.)

    Args:
        args: Merged namespace from ``ModelConfig`` + ``TrainConfig`` + defaults.
        model_without_ddp: The LWDETR model (unwrapped from DDP).

    Returns:
        List of parameter-group dicts suitable for ``torch.optim.Optimizer``.
    """
    if not isinstance(model_without_ddp.backbone, Joiner):
        msg = f"Expected backbone to be Joiner, got {type(model_without_ddp.backbone).__name__}"
        raise TypeError(msg)

    backbone = cast("Any", model_without_ddp.backbone[0])
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = "transformer.decoder"
    decoder_params = [p for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad]

    decoder_param_lr_pairs = [{"params": param, "lr": args.lr * args.lr_component_decay} for param in decoder_params]

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters()
        if (n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    other_param_dicts = [{"params": param, "lr": args.lr} for param in other_params]

    return other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs


def compute_multi_scale_scales(
    resolution: int,
    expanded_scales: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
) -> list[int]:
    """Compute multi-scale training resolutions.

    (Vendored from ``rfdetr.datasets.coco`` to avoid importing the
    ``rfdetr.datasets`` package, which depends on ``albumentations``.)

    Returns:
        List of candidate resolutions compatible with the backbone.
    """
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    return [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]
