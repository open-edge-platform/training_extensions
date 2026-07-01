# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ViT model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torch import nn

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.vision_transformer import VisionTransformerBackbone
from getitune.backend.lightning.models.classification.classifier import HLabelClassifier
from getitune.backend.lightning.models.classification.heads import (
    HierarchicalLinearClsHead,
)
from getitune.backend.lightning.models.classification.hlabel_models.base import LightningHlabelClsModel
from getitune.backend.lightning.models.classification.losses import AsymmetricAngularLossWithIgnore
from getitune.backend.lightning.models.classification.multiclass_models.vit import ForwardExplainMixInForViT
from getitune.backend.lightning.models.classification.utils.loaders import VisionTransformerLoaderMixin
from getitune.backend.lightning.models.classification.utils.pretrained_urls import VIT_PRETRAINED_URLS
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.metrics.accuracy import HLabelClsMetricCallable
from getitune.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable


class VisionTransformerHLabelCls(VisionTransformerLoaderMixin, ForwardExplainMixInForViT, LightningHlabelClsModel):
    """VisionTransformerForHLabelCls is a model designed for hierarchical label classification using ViT architecture.

    Args:
        label_info (HLabelInfo): Information about the hierarchical labels.
        model_name (str): Name of the Vision Transformer model to use.
        data_input_params (DataInputParams | dict | None, optional): Parameters for the image data preprocessing.
        optimizer (OptimizerCallable): Callable for the optimizer.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable): Callable for the learning rate scheduler.
        metric (MetricCallable): Callable for the metric.
        torch_compile (bool): Whether to use torch.compile for the model.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
    """

    label_info: HLabelInfo
    pretrained_urls = VIT_PRETRAINED_URLS

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
        pretrained: bool = True,
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
            pretrained=pretrained,
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
        return model
