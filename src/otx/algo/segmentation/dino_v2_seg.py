# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""DinoV2Seg model implementations."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urlparse

from torch.hub import download_url_to_file

from otx.algo.classification.backbones.vision_transformer import VisionTransformer
from otx.algo.segmentation.heads import FCNHead
from otx.algo.segmentation.losses import CrossEntropyLossWithIgnore
from otx.algo.segmentation.segmentors import BaseSegmModel
from otx.core.model.segmentation import OTXSegmentationModel

if TYPE_CHECKING:
    from torch import nn
    from typing_extensions import Self


class DinoV2Seg(OTXSegmentationModel):
    """DinoV2Seg Model."""

    AVAILABLE_MODEL_VERSIONS: ClassVar[list[str]] = [
        "dinov2-small-seg",
    ]
    PRETRAINED_WEIGHTS: ClassVar[dict[str, str]] = {
        "dinov2-small-seg": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    }

    def _build_model(self) -> nn.Module:
        if self.model_version not in self.AVAILABLE_MODEL_VERSIONS:
            msg = f"Model version {self.model_version} is not supported."
            raise ValueError(msg)
        backbone = VisionTransformer(arch=self.model_version, img_size=self.input_size)
        backbone.forward = partial(  # type: ignore[method-assign]
            backbone.get_intermediate_layers,
            n=[8, 9, 10, 11],
            reshape=True,
        )
        decode_head = FCNHead(self.model_version, num_classes=self.num_classes)
        criterion = CrossEntropyLossWithIgnore(ignore_index=self.label_info.ignore_index)  # type: ignore[attr-defined]

        backbone.init_weights()
        if self.model_version in self.PRETRAINED_WEIGHTS:
            print(f"init weight - {self.PRETRAINED_WEIGHTS[self.model_version]}")
            parts = urlparse(self.PRETRAINED_WEIGHTS[self.model_version])
            filename = Path(parts.path).name

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_file = cache_dir / filename
            if not Path.exists(cache_file):
                download_url_to_file(self.PRETRAINED_WEIGHTS[self.model_version], cache_file, "", progress=True)
            backbone.load_pretrained(checkpoint_path=cache_file)

        # freeze backbone
        for _, v in backbone.named_parameters():
            v.requires_grad = False

        return BaseSegmModel(
            backbone=backbone,
            decode_head=decode_head,
            criterion=criterion,
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for DinoV2Seg."""
        return {"model_type": "transformer"}

    def to(self, *args, **kwargs) -> Self:
        """Return a model with specified device."""
        ret = super().to(*args, **kwargs)
        if self.device.type == "xpu":
            msg = f"{type(self).__name__} doesn't support XPU."
            raise RuntimeError(msg)
        return ret
