# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""MobileNetV3 model implementation."""

from __future__ import annotations

from copy import copy
from math import ceil
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn

from otx.algo.classification.backbones import MobileNetV3Backbone
from otx.algo.classification.classifier import HLabelClassifier, ImageClassifier
from otx.algo.classification.heads import HierarchicalCBAMClsHead, LinearClsHead, MultiLabelNonLinearClsHead
from otx.algo.classification.losses.asymmetric_angular_loss_with_ignore import AsymmetricAngularLossWithIgnore
from otx.algo.classification.necks.gap import GlobalAveragePooling
from otx.algo.utils.support_otx_v1 import OTXv1Helper
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    HlabelClsBatchDataEntity,
    HlabelClsBatchPredEntity,
)
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import HLabelClsMetricCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, DataInputParams
from otx.core.model.hlabel_classification import OTXHlabelClsModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import HLabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.core.metrics import MetricCallable


class MobileNetV3ForHLabelCls(OTXHlabelClsModel):
    """MobileNetV3 Model for hierarchical label classification task."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "mobilenetv3_large",
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

        copied_head_config = copy(head_config)
        copied_head_config["step_size"] = (ceil(self.data_input_params.input_size[0] / 32), ceil(self.data_input_params.input_size[1] / 32))

        backbone = MobileNetV3Backbone(model_name=self.model_name, input_size=self.data_input_params.input_size)
        return HLabelClassifier(
            backbone=backbone,
            neck=nn.Identity(),
            head=HierarchicalCBAMClsHead(
                in_channels=MobileNetV3Backbone.MV3_CFG[self.model_name]["out_channels"],
                **copied_head_config,
            ),
            multiclass_loss=nn.CrossEntropyLoss(),
            multilabel_loss=AsymmetricAngularLossWithIgnore(gamma_pos=0.0, gamma_neg=1.0, reduction="sum"),
        )

    def load_from_otx_v1_ckpt(self, state_dict: dict, add_prefix: str = "model.") -> dict:
        """Load the previous OTX ckpt according to OTX2.0."""
        return OTXv1Helper.load_cls_mobilenet_v3_ckpt(state_dict, "hlabel", add_prefix)

    def _customize_inputs(self, inputs: HlabelClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.stack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: HlabelClsBatchDataEntity,
    ) -> HlabelClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        # To list, batch-wise
        if isinstance(outputs, dict):
            scores = outputs["scores"]
            labels = outputs["labels"]
        else:
            scores = outputs
            labels = outputs.argmax(-1, keepdim=True)

        return HlabelClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: HlabelClsBatchPredEntity,
        inputs: HlabelClsBatchDataEntity,
    ) -> MetricInput:
        hlabel_info: HLabelInfo = self.label_info  # type: ignore[assignment]

        _labels = torch.stack(preds.labels) if isinstance(preds.labels, list) else preds.labels
        _scores = torch.stack(preds.scores) if isinstance(preds.scores, list) else preds.scores
        if hlabel_info.num_multilabel_classes > 0:
            preds_multiclass = _labels[:, : hlabel_info.num_multiclass_heads]
            preds_multilabel = _scores[:, hlabel_info.num_multiclass_heads :]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = _labels
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }
