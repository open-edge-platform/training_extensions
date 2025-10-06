# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TIMM wrapper model class for OTX."""

from __future__ import annotations

from copy import copy
from math import ceil
from typing import TYPE_CHECKING

from torch import nn

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.timm import TimmBackbone
from otx.backend.native.models.classification.classifier import HLabelClassifier
from otx.backend.native.models.classification.heads import HierarchicalLinearClsHead
from otx.backend.native.models.classification.hlabel_models.base import OTXHlabelClsModel
from otx.backend.native.models.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from otx.backend.native.models.classification.necks.gap import GlobalAveragePooling
from otx.backend.native.models.utils.support_otx_v1 import OTXv1Helper
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.metrics.accuracy import HLabelClsMetricCallable
from otx.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.metrics import MetricCallable


class TimmModelHLabelCls(OTXHlabelClsModel):
    """Timm Model for hierarchical label classification task.

    Args:
        label_info (HLabelInfo): The label information for the classification task.
        model_name (str): The name of the model.
            You can find available models at timm.list_models() or timm.list_pretrained().
        input_size (tuple[int, int], optional): Model input size in the order of height and width.
            Defaults to (224, 224).
        pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
        optimizer (OptimizerCallable, optional): The optimizer callable for training the model.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): The learning rate scheduler callable.
        metric (MetricCallable, optional): The metric callable for evaluating the model.
            Defaults to HLabelClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
    """

    def __init__(
        self,
        label_info: HLabelInfo,
        data_input_params: DataInputParams,
        model_name: str = "tf_efficientnetv2_s.in21k",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
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

    def _create_model(self, head_config: dict | None = None) -> nn.Module:  # type: ignore[override]
        head_config = head_config if head_config is not None else self.label_info.as_head_config_dict()
        backbone = TimmBackbone(model_name=self.model_name)
        copied_head_config = copy(head_config)
        copied_head_config["step_size"] = (
            ceil(self.data_input_params.input_size[0] / 32),
            ceil(self.data_input_params.input_size[1] / 32),
        )
        return HLabelClassifier(
            backbone=backbone,
            neck=GlobalAveragePooling(dim=2),
            head=HierarchicalLinearClsHead(**copied_head_config, in_channels=backbone.num_features),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_v2_ckpt(state_dict, "hlabel", add_prefix)
