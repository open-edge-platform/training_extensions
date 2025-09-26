# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNet-B0 model implementation."""

from __future__ import annotations

from copy import copy
from math import ceil
from typing import TYPE_CHECKING, Literal

from torch import nn

from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.classification.backbones.efficientnet import EfficientNetBackbone
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


class EfficientNetHLabelCls(OTXHlabelClsModel):
    """EfficientNet Model for hierarchical label classification task."""

    def __init__(
        self,
        label_info: HLabelInfo,
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
        if not isinstance(self.label_info, HLabelInfo):
            raise TypeError(self.label_info)

        backbone = EfficientNetBackbone(model_name=self.model_name, input_size=self.data_input_params.input_size)

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
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "hlabel", add_prefix)
