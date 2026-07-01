# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TIMM wrapper model class for getitune."""

from __future__ import annotations

from typing import TYPE_CHECKING

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.timm import TimmBackbone
from getitune.backend.lightning.models.classification.classifier import ImageClassifier
from getitune.backend.lightning.models.classification.heads import MultiLabelLinearClsHead
from getitune.backend.lightning.models.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from getitune.backend.lightning.models.classification.multilabel_models.base import (
    LightningMultilabelClsModel,
)
from getitune.backend.lightning.models.classification.necks.gap import GlobalAveragePooling
from getitune.backend.lightning.models.classification.utils.loaders import TimmLoaderMixin
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.metrics.accuracy import MultiLabelClsMetricCallable
from getitune.types.label import LabelInfoTypes

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from getitune.metrics import MetricCallable


class TimmModelMultilabelCls(TimmLoaderMixin, LightningMultilabelClsModel):
    """TimmModel for multi-label classification task.

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
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "tf_efficientnetv2_s.in21k",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
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

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        backbone = TimmBackbone(model_name=self.model_name)
        return ImageClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=MultiLabelLinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.num_features,
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
