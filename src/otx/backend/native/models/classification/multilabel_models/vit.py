# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViT model implementation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from torch.hub import download_url_to_file

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.vision_transformer import VisionTransformerBackbone
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.heads import (
    MultiLabelLinearClsHead,
)
from otx.backend.native.models.classification.losses import AsymmetricAngularLossWithIgnore
from otx.backend.native.models.classification.multiclass_models.vit import ForwardExplainMixInForViT
from otx.backend.native.models.classification.multilabel_models.base import (
    OTXMultilabelClsModel,
)
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import MultiLabelClsMetricCallable
from otx.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.metrics import MetricCallable

augreg_url = "https://storage.googleapis.com/vit_models/augreg/"
dinov2_url = "https://dl.fbaipublicfiles.com/dinov2/"
pretrained_urls = {
    "vit-tiny": augreg_url
    + "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-small": augreg_url
    + "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz",
    "vit-base": augreg_url
    + "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "vit-large": augreg_url
    + "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "dinov2-small": dinov2_url + "dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    "dinov2-base": dinov2_url + "dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2-large": dinov2_url + "dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2-giant": dinov2_url + "dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
}


class VisionTransformerMultilabelCls(ForwardExplainMixInForViT, OTXMultilabelClsModel):
    """DeitTiny Model for multi-class classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal[
            "vit-tiny",
            "vit-small",
            "vit-base",
            "vit-large",
            "dinov2-small",
            "dinov2-base",
            "dinov2-large",
            "dinov2-giant",
        ] = "vit-tiny",
        freeze_backbone: bool = False,
        peft: Literal["lora", "dora"] | None = None,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        self.peft = peft

        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            freeze_backbone=freeze_backbone,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        for key in list(state_dict.keys()):
            new_key = key.replace("patch_embed.projection", "patch_embed.proj")
            new_key = new_key.replace("backbone.ln1", "backbone.norm")
            new_key = new_key.replace("ffn.layers.0.0", "mlp.fc1")
            new_key = new_key.replace("ffn.layers.1", "mlp.fc2")
            new_key = new_key.replace("layers", "blocks")
            new_key = new_key.replace("ln", "norm")
            if new_key != key:
                state_dict[new_key] = state_dict.pop(key)
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multiclass", add_prefix)

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        vit_backbone = VisionTransformerBackbone(
            model_name=self.model_name,
            img_size=self.data_input_params.input_size,
            peft=self.peft,
        )
        model = ImageClassifier(
            backbone=vit_backbone,
            neck=None,
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=vit_backbone.embed_dim,
            ),
            loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )

        model.init_weights()
        if self.model_name in pretrained_urls:
            print(f"init weight - {pretrained_urls[self.model_name]}")
            parts = urlparse(pretrained_urls[self.model_name])
            filename = Path(parts.path).name

            cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
            cache_file = cache_dir / filename
            if not Path.exists(cache_file):
                download_url_to_file(pretrained_urls[self.model_name], cache_file, "", progress=True)
            model.backbone.load_pretrained(checkpoint_path=cache_file)
        else:
            warnings.warn(
                "No pretrained weights found for the specified model. Initializing model with random weights.",
                stacklevel=1,
            )

        return model
