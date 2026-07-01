# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TIMM wrapper model class for getitune."""

from __future__ import annotations

from copy import copy
from math import ceil
from typing import TYPE_CHECKING

from torch import nn

from getitune.backend.lightning.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from getitune.backend.lightning.models.classification.backbones.timm import TimmBackbone
from getitune.backend.lightning.models.classification.classifier import HLabelClassifier
from getitune.backend.lightning.models.classification.heads import HierarchicalLinearClsHead
from getitune.backend.lightning.models.classification.hlabel_models.base import LightningHlabelClsModel
from getitune.backend.lightning.models.classification.losses.asymmetric_angular_loss_with_ignore import (
    AsymmetricAngularLossWithIgnore,
)
from getitune.backend.lightning.models.classification.necks.gap import GlobalAveragePooling
from getitune.backend.lightning.models.classification.utils.loaders import TimmLoaderMixin
from getitune.backend.lightning.schedulers import LRSchedulerListCallable
from getitune.metrics.accuracy import HLabelClsMetricCallable
from getitune.types.label import HLabelInfo

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from getitune.metrics import MetricCallable


class TimmModelHLabelCls(TimmLoaderMixin, LightningHlabelClsModel):
    """Timm Model for hierarchical label classification task.

    Args:
        label_info (LabelInfoTypes): Information about the labels.
        data_input_params (DataInputParams | dict | None, optional): The data input parameters
            such as input size and normalization. If None is given,
            default parameters for the specific model will be used.
        model_name (str, optional): Backbone model name for feature extraction. Defaults to "efficientnet_v2_s".
        optimizer (OptimizerCallable, optional): Optimizer for model training. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Learning rate scheduler.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for model evaluation. Defaults to MultiClassClsMetricCallable.
        torch_compile (bool, optional): Whether to compile the model using TorchScript. Defaults to False.
        kl_weight: The weight of tree-path KL divergence loss. Defaults to zero, use CrossEntropy only.
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
    """

    def __init__(
        self,
        label_info: HLabelInfo,
        data_input_params: DataInputParams | dict | None = None,
        model_name: str = "tf_efficientnetv2_s.in21k",
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
        torch_compile: bool = False,
        kl_weight: float = 0.0,
        pretrained: bool = True,
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
            kl_weight=kl_weight,
            pretrained=pretrained,
        )

    def _create_model(self, head_config: dict | None = None) -> nn.Module:  # type: ignore[override]
        head_config = head_config if head_config is not None else self.label_info.as_head_config_dict()
        backbone = TimmBackbone(model_name=self.model_name)
        if self.data_input_params.input_size is None:
            msg = "input_size should not be None."
            raise ValueError(msg)
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
