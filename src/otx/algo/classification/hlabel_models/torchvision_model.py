# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from otx.algo.classification.backbones.torchvision import TorchvisionBackbone, TVModels
from otx.algo.classification.classifier import HLabelClassifier, ImageClassifier
from otx.algo.classification.heads import (
    HierarchicalCBAMClsHead,
    LinearClsHead,
    MultiLabelLinearClsHead,
)
from otx.algo.classification.losses import AsymmetricAngularLossWithIgnore
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
    MultilabelClsBatchDataEntity,
    MultilabelClsBatchPredEntity,
)
from otx.core.metrics.accuracy import HLabelClsMetricCallable, MultiClassClsMetricCallable, MultiLabelClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, DataInputParams
from otx.core.model.hlabel_classification import OTXHlabelClsModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class TVModelForHLabelCls(OTXHlabelClsModel):
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

    label_info: HLabelInfo

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: TVModels = "efficientnet_v2_s",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:

        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _create_model(self, head_config: dict | None = None) -> nn.Module:
        head_config = head_config if head_config is not None else self.label_info.as_head_config_dict()
        backbone = TorchvisionBackbone(backbone=self.model_name)
        return HLabelClassifier(
            backbone=backbone,
            neck=nn.Identity(),
            head=HierarchicalCBAMClsHead(
                in_channels=backbone.in_features,
                **head_config,
            ),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )
