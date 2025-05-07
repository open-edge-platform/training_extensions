# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from otx.backend.openvino.models.base import OVModel
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import (
    MultiLabelClsMetricCallable,
)
from otx.data.torch import TorchDataBatch, TorchPredBatch

if TYPE_CHECKING:
    from model_api.models.utils import ClassificationResult

    from otx.core.metrics import MetricCallable


class OVMultilabelClassificationModel(OVModel):
    """Multilabel classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiLabelClsMetricCallable,
        **kwargs,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"multilabel": True, "confidence_threshold": 0.0})
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: TorchDataBatch,
    ) -> TorchPredBatch:
        pred_scores = [
            torch.tensor([top_label.confidence for top_label in out.top_labels], device=self.device) for out in outputs
        ]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return TorchPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=[],
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return TorchPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=[],
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: TorchPredBatch,
        inputs: TorchDataBatch,
    ) -> MetricInput:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }
