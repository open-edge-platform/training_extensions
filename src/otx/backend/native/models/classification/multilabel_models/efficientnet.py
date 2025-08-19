# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNet-B0 model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.efficientnet import EfficientNetBackbone
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.heads import MultiLabelLinearClsHead
from otx.backend.native.models.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from otx.backend.native.models.classification.multilabel_models.base import OTXMultilabelClsModel
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import MultiLabelClsMetricCallable
from otx.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import Tensor, nn

    from otx.metrics import MetricCallable


class EfficientNetMultilabelCls(OTXMultilabelClsModel):
    """EfficientNet Model for multi-label classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal[
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
            "efficientnet_b8",
        ] = "efficientnet_b0",
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
        backbone = EfficientNetBackbone(model_name=self.model_name, input_size=self.data_input_params.input_size)
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

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "multilabel", add_prefix)

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
