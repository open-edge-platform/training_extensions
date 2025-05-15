# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for keypoint detection model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from otx.backend.openvino.models.base import OVModel
from otx.core.metrics import MetricCallable, MetricInput
from otx.core.metrics.pck import PCKMeasureCallable
from otx.core.types.task import OTXTaskType
from otx.data import OTXDataBatch, OTXPredBatch

if TYPE_CHECKING:
    from model_api.models.result import DetectedKeypoints
    from torchmetrics import Metric

    from otx.core.types import PathLike


class OVKeypointDetectionModel(OVModel):
    """Keypoint detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX keypoint detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "keypoint_detection",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = PCKMeasureCallable,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )
        self._task = OTXTaskType.KEYPOINT_DETECTION

    def _customize_outputs(
        self,
        outputs: list[DetectedKeypoints],
        inputs: OTXDataBatch,
    ) -> OTXPredBatch:
        keypoints = []
        scores = []
        # default visibility threshold
        visibility_threshold = 0.5
        for output in outputs:
            kps = torch.as_tensor(output.keypoints)
            score = torch.as_tensor(output.scores)
            visible_keypoints = torch.cat([kps, score.unsqueeze(1) > visibility_threshold], dim=1)
            keypoints.append(visible_keypoints)
            scores.append(score)

        return OTXPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            scores=scores,
            bboxes=[],
            labels=[],
        )

    def compute_metrics(self, meter: Metric) -> dict:
        """Compute metrics for the model."""
        meter.input_size = (self.model.h, self.model.w)
        return super()._compute_metrics(meter)

    def prepare_metric_inputs(  # type: ignore[override]
        self,
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
    ) -> MetricInput:
        """Convert prediction and input entities to a format suitable for metric computation.

        Args:
            preds (OTXPredBatch): The predicted batch entity containing predicted keypoints.
            inputs (OTXDataBatch): The input batch entity containing ground truth keypoints.

        Returns:
            MetricInput: A dictionary contains 'preds' and 'target' keys
            corresponding to the predicted and target keypoints for metric evaluation.
        """
        if inputs.keypoints is None:
            msg = "The input ground truth keypoints are not provided."
            raise ValueError(msg)

        if preds.keypoints is None or preds.scores is None:
            msg = "The predicted keypoints or scores are not provided."
            raise ValueError(msg)

        if len(preds.keypoints) != len(inputs.keypoints):
            msg = "The number of predicted keypoints and ground truth keypoints does not match."
            raise ValueError(msg)

        return {
            "preds": [
                {
                    "keypoints": kpt[:, :2],
                    "scores": score,
                }
                for kpt, score in zip(preds.keypoints, preds.scores)
            ],
            "target": [
                {
                    "keypoints": kpt[:, :2],
                    "keypoints_visible": kpt[:, 2],
                }
                for kpt in inputs.keypoints
            ],
        }
