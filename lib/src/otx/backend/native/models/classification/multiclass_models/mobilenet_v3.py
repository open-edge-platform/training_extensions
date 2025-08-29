# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MobileNetV3 model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torch import Tensor, nn

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones import MobileNetV3Backbone
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.heads import LinearClsHead
from otx.backend.native.models.classification.multiclass_models.base import OTXMulticlassClsModel
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import MultiClassClsMetricCallable
from otx.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.metrics import MetricCallable


class MobileNetV3MulticlassCls(OTXMulticlassClsModel):
    """MobileNetV3MulticlassCls is a class that represents a MobileNetV3 model for multiclass classification.

    Args:
        label_info (LabelInfoTypes): The label information.
        data_input_params (DataInputParams): The data input parameters such as input size and normalization.
        model_name (Literal["mobilenetv3_large", "mobilenetv3_small"], optional): The model name.
            Defaults to "mobilenetv3_large".
        optimizer (OptimizerCallable, optional): The optimizer callable. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): The metric callable. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal["mobilenetv3_large", "mobilenetv3_small"] = "mobilenetv3_large",
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
        backbone = MobileNetV3Backbone(mode=self.model_name, input_size=self.data_input_params.input_size)
        backbone_out_chennels = MobileNetV3Backbone.MV3_CFG[self.model_name]["out_channels"]
        neck = GlobalAveragePooling(dim=2)

        return ImageClassifier(
            backbone=backbone,
            neck=neck,
            head=LinearClsHead(
                num_classes=num_classes,
                in_channels=backbone_out_chennels,
            ),
            loss=nn.CrossEntropyLoss(),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "multiclass", add_prefix)

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
