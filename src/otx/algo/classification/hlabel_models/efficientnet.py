# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNet-B0 model implementation."""

from __future__ import annotations

from copy import copy, deepcopy
from math import ceil
from typing import TYPE_CHECKING

from torch import Tensor, nn

from otx.algo.classification.backbones.efficientnet import EFFICIENTNET_VERSION, EfficientNetBackbone
from otx.algo.classification.classifier import HLabelClassifier, ImageClassifier
from otx.algo.classification.heads import HierarchicalCBAMClsHead, LinearClsHead, MultiLabelLinearClsHead
from otx.algo.classification.losses.asymmetric_angular_loss_with_ignore import AsymmetricAngularLossWithIgnore
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.classification.utils import get_classification_layers
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
)
from otx.core.metrics.accuracy import HLabelClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, DataInputParams
from otx.core.model.hlabel_classification import OTXHlabelClsModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class EfficientNetHLabel(OTXHlabelClsModel):
    """EfficientNet Model for hierarchical label classification task."""

    label_info: HLabelInfo

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "efficientnet_b0",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
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

    def _create_model(self, head_config: dict | None = None) -> nn.Module:
        head_config = head_config if head_config is not None else self.label_info.as_head_config_dict()
        if not isinstance(self.label_info, HLabelInfo):
            raise TypeError(self.label_info)

        backbone = EfficientNetBackbone(version=self.model_name, input_size=self.data_input_params.input_size)

        copied_head_config = copy(head_config)
        copied_head_config["step_size"] = (ceil(self.data_input_params.input_size[0] / 32), ceil(self.data_input_params.input_size[1] / 32))

        return HLabelClassifier(
            backbone=backbone,
            neck=nn.Identity(),
            head=HierarchicalCBAMClsHead(
                in_channels=backbone.num_features,
                **copied_head_config,
            ),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_effnet_b0_ckpt(state_dict, "hlabel", add_prefix)
