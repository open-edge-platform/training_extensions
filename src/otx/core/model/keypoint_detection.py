# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for keypoint detection model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.metrics import MetricCallable, MetricInput
from otx.core.metrics.pck import PCKMeasureCallable
from otx.core.model.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes
from otx.data.torch import OTXDataBatch, OTXPredBatch

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

    def _customize_inputs(self, entity: OTXDataBatch) -> dict[str, Any]:
        """Convert TorchDataBatch into Topdown model's input."""
        inputs: dict[str, Any] = {}

        inputs["inputs"] = entity.images
        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"
        return inputs

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                losses[k] = v
            return losses

        keypoints = []
        scores = []
        # default visibility threshold
        visibility_threshold = 0.5
        if inputs.imgs_info is None:
            msg = "The input image information is not provided."
            raise ValueError(msg)
        for i, output in enumerate(outputs):
            if not isinstance(output, tuple):
                raise TypeError(output)
            if inputs.imgs_info[i] is None:
                msg = f"The image information for the image {i} is not provided."
                raise ValueError(msg)
            # scale to the original image size
            orig_h, orig_w = inputs.imgs_info[i].ori_shape  # type: ignore[union-attr]
            kp_scale_h, kp_scale_w = (
                orig_h / self.data_input_params.input_size[0],
                orig_w / self.data_input_params.input_size[1],
            )
            inverted_scale = max(kp_scale_h, kp_scale_w)
            kp_scale_h = kp_scale_w = inverted_scale
            # decode kps
            kps = torch.as_tensor(output[0], device=self.device) * torch.tensor(
                [kp_scale_w, kp_scale_h],
                device=self.device,
            )
            score = torch.as_tensor(output[1], device=self.device)
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

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        self._metric.input_size = tuple(self.data_input_params.input_size)

    def _convert_pred_entity_to_compute_metric(  # type: ignore[override]
        self,
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
    ) -> MetricInput:
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

    def forward_for_tracing(self, image: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(inputs=image, mode="tensor")

    def get_dummy_input(self, batch_size: int = 1) -> OTXDataBatch:  # type: ignore[override]
        """Generates a dummy input, suitable for launching forward() on it.

        Args:
            batch_size (int, optional): number of elements in a dummy input sequence. Defaults to 1.

        Returns:
            TorchDataBatch: An entity containing randomly generated inference data.
        """
        images = torch.rand(self.data_input_params.as_ncwh(batch_size))
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape[:2],
                    ori_shape=img.shape[:2],
                ),
            )

        return OTXDataBatch(
            batch_size,
            images,
            labels=[],
            bboxes=[],
            keypoints=[],
            imgs_info=infos,  # type: ignore[arg-type]
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
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        keypoints = []
        scores = []
        # default visibility threshold
        visibility_threshold = 0.5
        for output in outputs:
            kps = torch.as_tensor(output.keypoints, device=self.device)
            score = torch.as_tensor(output.scores, device=self.device)
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

    def configure_metric(self) -> None:
        """Configure the metric."""
        super().configure_metric()
        self._metric.input_size = (self.model.h, self.model.w)

    def _convert_pred_entity_to_compute_metric(  # type: ignore[override]
        self,
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
    ) -> MetricInput:
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
