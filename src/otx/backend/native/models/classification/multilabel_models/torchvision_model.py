# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torchvision model for the OTX classification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.torchvision import TorchvisionBackbone
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.heads import (
    MultiLabelLinearClsHead,
)
from otx.backend.native.models.classification.losses import AsymmetricAngularLossWithIgnore
from otx.backend.native.models.classification.multilabel_models.base import (
    OTXMultilabelClsModel,
)
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import MultiLabelClsMetricCallable
from otx.types.label import LabelInfoTypes

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.metrics import MetricCallable


class TVModelMultilabelCls(OTXMultilabelClsModel):
    """Torchvision model for multilabel classification.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        backbone (TVModelType): Backbone model for feature extraction.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        optimizer (OptimizerCallable, optional): Optimizer for model training. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for model evaluation. Defaults to MultiLabelClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        input_size (tuple[int, int], optional): Input size of the images. Defaults to (224, 224).
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "efficientnet_v2_s",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
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

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        backbone = TorchvisionBackbone(backbone=self.model_name)
        return ImageClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.in_features,
                normalized=True,
            ),
            loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            loss_scale=7.0,
        )

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
