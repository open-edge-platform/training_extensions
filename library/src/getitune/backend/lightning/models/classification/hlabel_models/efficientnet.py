# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientNet-B0 model implementation."""

from __future__ import annotations

from copy import copy
from math import ceil
from typing import TYPE_CHECKING, Literal

from torch import nn

<<<<<<<< HEAD:library/src/getitune/backend/lightning/models/classification/hlabel_models/efficientnet.py
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.efficientnet import EfficientNetBackbone
from getitune.backend.lightning.models.classification.classifier import HLabelClassifier
from getitune.backend.lightning.models.classification.heads import HierarchicalLinearClsHead
from getitune.backend.lightning.models.classification.hlabel_models.base import LightningHlabelClsModel
from getitune.backend.lightning.models.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from getitune.backend.lightning.models.classification.necks.gap import GlobalAveragePooling
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
========
from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.efficientnet import EfficientNetBackbone
from getitune.backend.lightning.models.classification.classifier import HLabelClassifier
from getitune.backend.lightning.models.classification.heads import HierarchicalLinearClsHead
from getitune.backend.lightning.models.classification.hlabel_models.base import LightningHlabelClsModel
from getitune.backend.lightning.models.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from getitune.backend.lightning.models.classification.necks.gap import GlobalAveragePooling
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
>>>>>>>> develop:library/src/getitune/backend/native/models/classification/hlabel_models/efficientnet.py
from getitune.metrics.accuracy import HLabelClsMetricCallable
from getitune.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable


class EfficientNetHLabelCls(LightningHlabelClsModel):
    """EfficientNet Model for hierarchical label classification task."""

    def __init__(
        self,
        label_info: HLabelInfo,
        data_input_params: DataInputParams | dict | None = None,
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

        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
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
