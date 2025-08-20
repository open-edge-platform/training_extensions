# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MobileNetV3 model implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones import MobileNetV3Backbone
from otx.backend.native.models.classification.classifier import ImageClassifier
from otx.backend.native.models.classification.heads import MultiLabelNonLinearClsHead
from otx.backend.native.models.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from otx.backend.native.models.classification.multilabel_models.base import OTXMultilabelClsModel
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.metrics.accuracy import MultiLabelClsMetricCallable
from otx.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.metrics import MetricCallable


class MobileNetV3MultilabelCls(OTXMultilabelClsModel):
    """MobileNetV3 Model for multi-class classification task.

    Args:
        label_info (LabelInfoTypes): The label information.
        data_input_params (DataInputParams): The data input parameters such as input size and normalization.
        model_name (str, optional): The model name. Defaults to "mobilenetv3_large".
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
        return ImageClassifier(
            backbone=MobileNetV3Backbone(mode=self.model_name, input_size=self.data_input_params.input_size),
            neck=GlobalAveragePooling(dim=2),
            head=MultiLabelNonLinearClsHead(
                num_classes=num_classes,
                in_channels=MobileNetV3Backbone.MV3_CFG[self.model_name]["out_channels"],
                hid_channels=MobileNetV3Backbone.MV3_CFG[self.model_name]["hid_channels"],
                normalized=True,
                activation=nn.PReLU(),
            ),
            loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
            loss_scale=7.0,
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "multilabel", add_prefix)

    def _customize_inputs(self, inputs: OTXDataBatch) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.images,
            "labels": torch.stack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)

        return OTXPredBatch(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=list(scores),
            labels=list(logits.argmax(-1, keepdim=True).unbind(0)),
        )

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
