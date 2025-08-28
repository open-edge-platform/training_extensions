# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.torchvision import TorchvisionBackbone
from otx.backend.native.models.classification.classifier import HLabelClassifier
from otx.backend.native.models.classification.heads import (
    HierarchicalLinearClsHead,
)
from otx.backend.native.models.classification.hlabel_models.base import OTXHlabelClsModel
from otx.backend.native.models.classification.losses import AsymmetricAngularLossWithIgnore
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import HLabelClsMetricCallable
from otx.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.metrics import MetricCallable


class TVModelHLabelCls(OTXHlabelClsModel):
    """TVModelForHLabelCls class represents a Torchvision model for hierarchical label classification.

    Args:
        label_info (HLabelInfo): Information about the hierarchical labels.
        backbone (TVModelType): The type of Torchvision backbone model.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The metric callable. Defaults to HLabelClsMetricCallble.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        input_size (tuple[int, int], optional): The input size of the images. Defaults to (224, 224).
    """

    def __init__(
        self,
        label_info: HLabelInfo,
        data_input_params: DataInputParams,
        model_name: str = "efficientnet_v2_s",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
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
        backbone = TorchvisionBackbone(backbone=self.model_name)
        return HLabelClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=HierarchicalLinearClsHead(**head_config, in_channels=backbone.in_features),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )
