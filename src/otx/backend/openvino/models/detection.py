# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for detection model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any, Literal

import torch
from model_api.tilers import DetectionTiler
from torchmetrics import Metric
from torchvision import tv_tensors

from otx.backend.openvino.models.base import OVModel
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.metrics import MetricCallable, MetricInput
from otx.core.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.data import TorchDataBatch, TorchPredBatch

if TYPE_CHECKING:
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import DetectionResult


class OVDetectionModel(OVModel):
    """Object detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "SSD",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )

    def _setup_tiler(self) -> None:
        """Setup tiler for tile task."""
        execution_mode = "async" if self.async_inference else "sync"
        # Note: Disable async_inference as tiling has its own sync/async implementation
        self.async_inference = False
        self.model = DetectionTiler(self.model, execution_mode=execution_mode)
        log.info(
            f"Enable tiler with tile size: {self.model.tile_size} \
                and overlap: {self.model.tiles_overlap}",
        )

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Reads model configuration from ModelAPI OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): target adapter to read the config
        """
        if model_adapter.model.has_rt_info(["model_info", "confidence_threshold"]):
            best_confidence_threshold = model_adapter.model.get_rt_info(["model_info", "confidence_threshold"]).value
            self.hparams["best_confidence_threshold"] = float(best_confidence_threshold)
        else:
            msg = (
                "Cannot get best_confidence_threshold from OpenVINO IR's rt_info. "
                "Please check whether this model is trained by OTX or not. "
                "Without this information, it can produce a wrong F1 metric score. "
                "At this time, it will be set as the default value = None."
            )
            log.warning(msg)
            self.hparams["best_confidence_threshold"] = None

    def _customize_outputs(
        self,
        outputs: list[DetectionResult],
        inputs: TorchDataBatch,
    ) -> TorchPredBatch | OTXBatchLossEntity:
        # add label index
        bboxes = []
        scores = []
        labels = []

        # some OMZ model requires to shift labels
        first_label = (
            self.model.model.get_label_name(0)
            if isinstance(self.model, DetectionTiler)
            else self.model.get_label_name(0)
        )

        label_shift = 1 if first_label == "background" else 0
        if label_shift:
            log.warning(f"label_shift: {label_shift}")

        for i, output in enumerate(outputs):
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    data=output.bboxes,
                    format="XYXY",
                    canvas_size=inputs.imgs_info[i].img_shape,  # type: ignore[union-attr, index]
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
            scores.append(torch.tensor(output.scores.reshape(-1), device=self.device))
            labels.append(torch.tensor(output.labels.reshape(-1) - label_shift, device=self.device, dtype=torch.long))

        if outputs and outputs[0].saliency_map.size > 1:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return TorchPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return TorchPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: TorchPredBatch,  # type: ignore[override]
        inputs: TorchDataBatch,  # type: ignore[override]
    ) -> MetricInput:
        return {
            "preds": [
                {
                    "boxes": bboxes.data,
                    "scores": scores,
                    "labels": labels,
                }
                for bboxes, scores, labels in zip(preds.bboxes, preds.scores, preds.labels)  # type: ignore[arg-type]
            ],
            "target": [
                {
                    "boxes": bboxes.data,
                    "labels": labels,
                }
                for bboxes, labels in zip(inputs.bboxes, inputs.labels)  # type: ignore[arg-type]
            ],
        }

    def compute_metrics(self, meter: Metric, key: Literal["val", "test"]) -> None:
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._compute_metrics(meter, key, **compute_kwargs)
