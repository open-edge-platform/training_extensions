# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TIMM wrapper model class for OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.timm import TimmBackbone
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.heads import LinearClsHead
from otx.backend.native.models.classification.multiclass_models.base import (
    OTXMulticlassClsModel,
)
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import MultiClassClsMetricCallable
from otx.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.metrics import MetricCallable


class TimmModelMulticlassCls(OTXMulticlassClsModel):
    """TimmModel for multi-class classification task.

    Args:
        label_info (LabelInfoTypes): The label information for the classification task.
        model_name (str): The name of the model.
            You can find available models at timm.list_models() or timm.list_pretrained().
        input_size (tuple[int, int], optional): Model input size in the order of height and width.
            Defaults to (224, 224).
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        optimizer (OptimizerCallable, optional): The optimizer callable for training the model.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
        metric (MetricCallable, optional): The metric callable for evaluating the model.
            Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.

    Example:
        1. API
            >>> model = TimmModelForMulticlassCls(
            ...     model_name="tf_efficientnetv2_s.in21k",
            ...     label_info=<Number-of-classes>,
            ... )
        2. CLI
            >>> otx train \
            ... --model otx.algo.classification.timm_model.TimmModelForMulticlassCls \
            ... --model.model_name tf_efficientnetv2_s.in21k
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "tf_efficientnetv2_s.in21k",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
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
        backbone = TimmBackbone(model_name=self.model_name)
        return ImageClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=LinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.num_features,
            ),
            loss=nn.CrossEntropyLoss(),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "multiclass", add_prefix)

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
