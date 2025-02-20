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
from otx.core.model.multiclass_classification import (
    OTXMulticlassClsModel,
)
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class TVModelForMulticlassCls(OTXMulticlassClsModel):
    """Torchvision model for multiclass classification.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams): Data input parameters such as input size and normalization.
        model_name (TVModels, optional): Backbone model name for feature extraction. Defaults to "efficientnet_v2_s".
        optimizer (OptimizerCallable, optional): Optimizer for model training. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for model evaluation. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: TVModels = "efficientnet_v2_s",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
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

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        backbone = TorchvisionBackbone(backbone=self.model_name)
        neck = GlobalAveragePooling(dim=2)

        return ImageClassifier(
            backbone=backbone,
            neck=neck,
            head=LinearClsHead(
                num_classes=num_classes,
                in_channels=backbone.in_features,
            ),
            loss=nn.CrossEntropyLoss(),
        )

    def forward_explain(self, inputs: MulticlassClsBatchDataEntity) -> MulticlassClsBatchPredEntity:
        """Model forward explain function."""
        outputs = self.model(images=inputs.stacked_images, mode="explain")

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=outputs["preds"],
            scores=outputs["scores"],
            saliency_map=outputs["saliency_map"],
            feature_vector=outputs["feature_vector"],
        )

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
