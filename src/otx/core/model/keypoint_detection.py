# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for keypoint detection model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.keypoint_detection import KeypointDetBatchDataEntity, KeypointDetBatchPredEntity
from otx.core.metrics import MetricCallable, MetricInput
from otx.core.metrics.pck import PCKMeasureCallable
from otx.core.model.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import DetectedKeypoints


class OTXKeypointDetectionModel(OTXModel):
    """Base class for the keypoint detection models used in OTX.

    label_info (LabelInfoTypes): Information about the labels.
    data_input_params (DataInputParams): Parameters for data input.
    model_name (str, optional): Name of the model. Defaults to "keypoint_detection_model".
    optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
    scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
        Defaults to DefaultSchedulerCallable.
    metric (MetricCallable, optional): Callable for the metric. Defaults to PCKMeasureCallable.
    torch_compile (bool, optional): Whether to use torch compile. Defaults to False.

    Base class for the keypoint detection models used in OTX.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "keypoint_detection_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = PCKMeasureCallable,
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

    def _customize_inputs(self, entity: KeypointDetBatchDataEntity) -> dict[str, Any]:
        """Convert KeypointDetBatchDataEntity into Topdown model's input."""
        inputs: dict[str, Any] = {}

        inputs["inputs"] = entity.images
        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"
        return inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: KeypointDetBatchDataEntity,
    ) -> KeypointDetBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        keypoints = []
        scores = []
        for output in outputs:
            if not isinstance(output, tuple):
                raise TypeError(output)
            keypoints.append(torch.as_tensor(output[0], device=self.device))
            scores.append(torch.as_tensor(output[1], device=self.device))

        return KeypointDetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            scores=scores,
            keypoints_visible=[],
            bboxes=[],
            labels=[],
            bbox_info=[],
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        self._metric.input_size = tuple(self.data_input_params.input_size)

    def _convert_pred_entity_to_compute_metric(  # type: ignore[override]
        self,
        preds: KeypointDetBatchPredEntity,
        inputs: KeypointDetBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": [
                {
                    "keypoints": kpt,
                    "scores": score,
                }
                for kpt, score in zip(preds.keypoints, preds.scores)
            ],
            "target": [
                {
                    "keypoints": kpt,
                    "keypoints_visible": kpt_visible,
                }
                for kpt, kpt_visible in zip(inputs.keypoints, inputs.keypoints_visible)
            ],
        }

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(inputs=image, mode="tensor")

    def get_dummy_input(self, batch_size: int = 1) -> KeypointDetBatchDataEntity:
        """Generates a dummy input, suitable for launching forward() on it.

        Args:
            batch_size (int, optional): number of elements in a dummy input sequence. Defaults to 1.

        Returns:
            KeypointDetBatchDataEntity: An entity containing randomly generated inference data.
        """
        images = torch.rand(self.data_input_params.as_ncwh(batch_size))
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return KeypointDetBatchDataEntity(
            batch_size,
            images,
            infos,
            bboxes=[],
            labels=[],
            bbox_info=[],
            keypoints=[],
            keypoints_visible=[],
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="keypoint_detection",
            task_type="keypoint_detection",
            confidence_threshold=self.hparams.get("best_confidence_threshold", None),
        )


class OVKeypointDetectionModel(OVModel):
    """Keypoint detection model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX keypoint detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "keypoint_detection",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = PCKMeasureCallable,
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

    def _customize_outputs(
        self,
        outputs: list[DetectedKeypoints],
        inputs: KeypointDetBatchDataEntity,
    ) -> KeypointDetBatchPredEntity | OTXBatchLossEntity:
        keypoints = []
        scores = []
        for output in outputs:
            keypoints.append(torch.as_tensor(output.keypoints, device=self.device))
            scores.append(torch.as_tensor(output.scores, device=self.device))

        return KeypointDetBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            keypoints=keypoints,
            scores=scores,
            keypoints_visible=[],
            bboxes=[],
            labels=[],
            bbox_info=[],
        )

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        self._metric.input_size = (self.model.h, self.model.w)

    def _convert_pred_entity_to_compute_metric(  # type: ignore[override]
        self,
        preds: KeypointDetBatchPredEntity,
        inputs: KeypointDetBatchDataEntity,
    ) -> MetricInput:
        return {
            "preds": [
                {
                    "keypoints": kpt,
                    "scores": score,
                }
                for kpt, score in zip(preds.keypoints, preds.scores)
            ],
            "target": [
                {
                    "keypoints": kpt,
                    "keypoints_visible": kpt_visible,
                }
                for kpt, kpt_visible in zip(inputs.keypoints, inputs.keypoints_visible)
            ],
        }
