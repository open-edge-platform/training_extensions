# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for detection model entity used in OTX."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import torch
from model_api.tilers import DetectionTiler
from torchvision import tv_tensors

from otx.backend.openvino.models.base import OVModel
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.metrics import MetricCallable, MetricInput
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import DetectionResult
    from torchmetrics import Metric

    from otx.types import PathLike


class OVDetectionModel(OVModel):
    """OVDetectionModel: Object detection model compatible for OpenVINO IR inference.

    This class is designed to work with OpenVINO IR models or models from the Intel OMZ repository.
    It provides compatibility with the OTX testing pipeline for object detection tasks.

        Initialize the OVDetectionModel.

            model_path (PathLike): Path to the OpenVINO IR model.
            model_type (str): Type of the model (default: "SSD").
            async_inference (bool): Whether to use asynchronous inference (default: True).
            max_num_requests (int | None): Maximum number of inference requests (default: None).
            use_throughput_mode (bool): Whether to use throughput mode (default: True).
            model_api_configuration (dict[str, Any] | None): Configuration for the model API (default: None).
            metric (MetricCallable): Metric callable for evaluation (default: MeanAveragePrecisionFMeasureCallable).
            **kwargs: Additional keyword arguments.
        ...

        Setup the tiler for handling tiled inference tasks.

        This method configures the tiler with the appropriate execution mode
        and disables asynchronous inference as tiling has its own sync/async implementation.
        ...

        Extract hyperparameters from the OpenVINO model adapter.

            model_adapter (OpenvinoAdapter): The adapter to extract model configuration from.

        This method reads the confidence threshold from the model's runtime information (rt_info).
        If unavailable, it logs a warning and sets the confidence threshold to None.
        ...

        Customize the outputs of the model to match the expected format.

            outputs (list[DetectionResult]): List of detection results from the model.
            inputs (OTXDataBatch): Input batch containing image and metadata.

            OTXPredBatch: A batch of predictions including bounding boxes, scores, labels,
            and optionally saliency maps and feature vectors.
        ...

        Prepare inputs for metric computation.

            preds (OTXPredBatch): Predicted batch containing bounding boxes, scores, and labels.
            inputs (OTXDataBatch): Input batch containing ground truth bounding boxes and labels.

            MetricInput: A dictionary with 'preds' and 'target' keys containing
            the predicted and ground truth bounding boxes and labels.
        ...

        Compute evaluation metrics for the model.

            metric (Metric): Metric object used for evaluation.

            dict: A dictionary containing computed metric values.
        ...
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "SSD",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
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
        self._task = OTXTaskType.DETECTION

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
        inputs: OTXDataBatch,
    ) -> OTXPredBatch:
        """Customize the outputs of the detection model.

        Args:
            outputs (list[DetectionResult]): A list of detection results containing bounding boxes,
                scores, labels, saliency maps, and feature vectors.
            inputs (OTXDataBatch): A batch of input data containing images and their metadata.

        Returns:
            OTXPredBatch: A batch of predictions containing processed bounding boxes, scores, labels,
            and optionally saliency maps and feature vectors.

        Notes:
            - Adjusts label indices based on whether the first label is "background".
            - Converts bounding boxes to the "XYXY" format and aligns them with the input image shape.
            - Handles optional saliency maps and feature vectors if present in the outputs.
        """
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
                    dtype=torch.float32,
                ),
            )
            scores.append(torch.tensor(output.scores.reshape(-1)))
            labels.append(torch.tensor(output.labels.reshape(-1) - label_shift, dtype=torch.long))

        if outputs and outputs[0].saliency_map.size > 1:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return OTXPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return OTXPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def prepare_metric_inputs(
        self,
        preds: OTXPredBatch,  # type: ignore[override]
        inputs: OTXDataBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Convert prediction and input entities to a format suitable for metric computation.

        Args:
            preds (OTXPredBatch): The predicted batch entity containing predicted bboxes.
            inputs (OTXDataBatch): The input batch entity containing ground truth bboxes.

        Returns:
            MetricInput: A dictionary contains 'preds' and 'target' keys
            corresponding to the predicted and target bboxes for metric evaluation.
        """
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

    def compute_metrics(self, metric: Metric) -> dict:
        """Compute metrics for the model."""
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._compute_metrics(metric, **compute_kwargs)
