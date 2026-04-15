# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViT model implementation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from torch import nn
from torch.hub import download_url_to_file

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.vision_transformer import VisionTransformerBackbone
from otx.backend.native.models.classification.classifier import HLabelClassifier
from otx.backend.native.models.classification.heads import (
    HierarchicalLinearClsHead,
)
from otx.backend.native.models.classification.hlabel_models.base import OTXHlabelClsModel
from otx.backend.native.models.classification.losses import AsymmetricAngularLossWithIgnore
from otx.backend.native.models.classification.multiclass_models.vit import ForwardExplainMixInForViT
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import HLabelClsMetricCallable
from otx.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.metrics import MetricCallable

pretrained_urls = {
    "vit-tiny": (
        "https://storage.geti.intel.com/weights/"
        "Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit-small": (
        "https://storage.geti.intel.com/weights/"
        "S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    ),
    "vit-base": (
        "https://storage.geti.intel.com/weights/"
        "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "vit-large": (
        "https://storage.geti.intel.com/weights/"
        "L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz"
    ),
    "dinov2-small": "https://storage.geti.intel.com/weights/dinov2_vits14_reg4_pretrain.pth",
    "dinov2-base": "https://storage.geti.intel.com/weights/dinov2_vitb14_reg4_pretrain.pth",
    "dinov2-large": "https://storage.geti.intel.com/weights/dinov2_vitl14_reg4_pretrain.pth",
    "dinov2-giant": "https://storage.geti.intel.com/weights/dinov2_vitg14_reg4_pretrain.pth",
}


class VisionTransformerHLabelCls(ForwardExplainMixInForViT, OTXHlabelClsModel):
    """VisionTransformerForHLabelCls is a model designed for hierarchical label classification using ViT architecture.

    Args:
        label_info (HLabelInfo): Information about the hierarchical labels.
        model_name (str): Name of the Vision Transformer model to use.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
        optimizer (OptimizerCallable): Callable for the optimizer.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable): Callable for the learning rate scheduler.
        metric (MetricCallable): Callable for the metric.
        torch_compile (bool): Whether to use torch.compile for the model.
    """

    label_info: HLabelInfo

    def __init__(
        self,
        label_info: HLabelInfo,
        data_input_params: DataInputParams | dict | None = None,
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
        metric: MetricCallable = HLabelClsMetricCallable,
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

    def _create_model(self, head_config: dict | None = None) -> nn.Module:  # type: ignore[override]
        head_config = head_config if head_config is not None else self.label_info.as_head_config_dict()
        if not isinstance(self.label_info, HLabelInfo):
            raise TypeError(self.label_info)
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
        init_cfg = [
            {"std": 0.2, "layer": "Linear", "type": "TruncNormal"},
            {"bias": 0.0, "val": 1.0, "layer": "LayerNorm", "type": "Constant"},
        ]
        vit_backbone = VisionTransformerBackbone(
            model_name=self.model_name,
            img_size=self.data_input_params.input_size,
            peft=self.peft,
        )
        model = HLabelClassifier(
            backbone=vit_backbone,
            neck=None,
            head=HierarchicalLinearClsHead(**head_config, in_channels=vit_backbone.embed_dim),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            init_cfg=init_cfg,
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
