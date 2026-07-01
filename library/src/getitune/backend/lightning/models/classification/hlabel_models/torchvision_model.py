# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the getitune classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.torchvision import TorchvisionBackbone
from getitune.backend.lightning.models.classification.classifier import HLabelClassifier
from getitune.backend.lightning.models.classification.heads import (
    HierarchicalLinearClsHead,
)
from getitune.backend.lightning.models.classification.hlabel_models.base import LightningHlabelClsModel
from getitune.backend.lightning.models.classification.losses import AsymmetricAngularLossWithIgnore
from getitune.backend.lightning.models.classification.necks.gap import GlobalAveragePooling
from getitune.backend.lightning.models.classification.utils.loaders import TorchvisionLoaderMixin
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.metrics.accuracy import HLabelClsMetricCallable
from getitune.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable


class TVModelHLabelCls(TorchvisionLoaderMixin, LightningHlabelClsModel):
    """TVModelForHLabelCls class represents a Torchvision model for hierarchical label classification.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): The data input parameters
            such as input size and normalization. If None is given,
            default parameters for the specific model will be used.
        model_name (str, optional): Backbone model name for feature extraction. Defaults to "efficientnet_v2_s".
        optimizer (OptimizerCallable, optional): Optimizer for model training. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for model evaluation. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
    """

    def __init__(
        self,
        label_info: HLabelInfo,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "efficientnet_v2_s",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
        torch_compile: bool = False,
        pretrained: bool = True,
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
            pretrained=pretrained,
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
