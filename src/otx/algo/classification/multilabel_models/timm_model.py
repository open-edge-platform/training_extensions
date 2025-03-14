# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TIMM wrapper model class for OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx.algo.classification.backbones.timm import TimmBackbone
from otx.algo.classification.classifier import ImageClassifier
from otx.algo.classification.heads import MultiLabelLinearClsHead
from otx.algo.classification.losses.asymmetric_angular_loss_with_ignore import AsymmetricAngularLossWithIgnore
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.metrics.accuracy import MultiLabelClsMetricCallable
from otx.core.model.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.multilabel_classification import (
    OTXMultilabelClsModel,
)
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.core.metrics import MetricCallable


class TimmModelMultilabelCls(OTXMultilabelClsModel):
    """TimmModel for multi-label classification task.

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
            Defaults to MultiLabelClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
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

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "multilabel", add_prefix)

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
