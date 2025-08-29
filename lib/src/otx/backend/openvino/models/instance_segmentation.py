# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for instance segmentation model entity used in OTX."""

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any

import torch
from model_api.tilers import InstanceSegmentationTiler
from torchvision import tv_tensors

from otx.backend.openvino.models.base import OVModel
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.data.utils.structures.mask.mask_util import encode_rle, polygon_to_rle
from otx.metrics import MetricInput
from otx.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.types.label import LabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import InstanceSegmentationResult
    from torchmetrics import Metric

    from otx.metrics import MetricCallable
    from otx.types import PathLike


class OVInstanceSegmentationModel(OVModel):
    """Instance segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "MaskRCNN",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        **kwargs,
    ) -> None:
        """Initialize the instance segmentation model.

        Args:
            model_path (PathLike): Path to the OpenVINO IR model.
            model_type (str): Type of the model (default: "MaskRCNN").
            async_inference (bool): Whether to use asynchronous inference (default: True).
            max_num_requests (int | None): Maximum number of inference requests (default: None).
            use_throughput_mode (bool): Whether to use throughput mode (default: True).
            model_api_configuration (dict[str, Any] | None): Model API configuration (default: None).
            metric (MetricCallable): Metric callable for evaluation (default: MaskRLEMeanAPFMeasureCallable).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )
        self._task = OTXTaskType.INSTANCE_SEGMENTATION

    def _setup_tiler(self) -> None:
        """Set up the tiler for tiled inference.

        This method configures the tiler for the instance segmentation task,
        enabling tiled inference with specified tile size and overlap.
        """
        execution_mode = "async" if self.async_inference else "sync"
        self.async_inference = False  # Disable async_inference as tiling has its own implementation
        self.model = InstanceSegmentationTiler(self.model, execution_mode=execution_mode)
        log.info(
            f"Enable tiler with tile size: {self.model.tile_size} \
                and overlap: {self.model.tiles_overlap}",
        )

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Retrieve hyperparameters from the OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): The OpenVINO adapter to read the configuration from.

        This method reads the confidence threshold from the model's runtime information
        and updates the hyperparameters accordingly.
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
        outputs: list[InstanceSegmentationResult],
        inputs: OTXDataBatch,
    ) -> OTXPredBatch:
        """Customize the model outputs for OTX compatibility.

        Args:
            outputs (list[InstanceSegmentationResult]): Model outputs.
            inputs (OTXDataBatch): Input data batch.

        Returns:
            OTXPredBatch: Customized predictions batch.
        """
        bboxes = []
        scores = []
        labels = []
        masks = []
        for output in outputs:
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    data=output.bboxes,
                    format="XYXY",
                    canvas_size=inputs.imgs_info[-1].img_shape,  # type: ignore[union-attr,index]
                    dtype=torch.float32,
                ),
            )
            scores.append(torch.tensor(output.scores.reshape(-1)))
            masks.append(torch.tensor(output.masks))
            labels.append(torch.tensor(output.labels.reshape(-1) - 1, dtype=torch.long))

        if outputs and outputs[0].saliency_map:
            predicted_s_maps = []
            for out in outputs:
                image_map = torch.tensor(
                    [s_map for s_map in out.saliency_map if s_map.size > 0],
                    dtype=torch.float32,
                )
                predicted_s_maps.append(image_map)

            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return OTXPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks if any(mask.numel() > 0 for mask in masks) else None,
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
            masks=masks if any(mask.numel() > 0 for mask in masks) else None,
            labels=labels,
        )

    def prepare_metric_inputs(
        self,
        preds: OTXPredBatch,  # type: ignore[override]
        inputs: OTXDataBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Prepare inputs for metric computation.

        Converts predictions and ground truth to the format required by the metric
        and caches the ground truth for the current batch.

        Args:
            preds (OTXPredBatch): Current batch predictions.
            inputs (OTXDataBatch): Current batch ground-truth inputs.

        Returns:
            MetricInput: Dictionary containing predictions and ground truth.
        """
        target_info = []

        pred_info = [
            {
                "boxes": preds.bboxes[idx].data if preds.bboxes is not None else torch.empty((0, 4)),
                "masks": [encode_rle(mask) for mask in preds.masks[idx].data]
                if preds.masks is not None and len(preds.masks)
                else torch.empty((0,)),
                "scores": preds.scores[idx],  # type: ignore[index]
                "labels": preds.labels[idx],  # type: ignore[index]
            }
            for idx in range(len(preds.labels))  # type: ignore[arg-type]
        ]

        for idx in range(len(inputs.labels)):  # type: ignore[arg-type]
            rles = (
                [encode_rle(mask) for mask in inputs.masks[idx].data]
                if inputs.masks is not None and len(inputs.masks[idx]) > 0
                else polygon_to_rle(inputs.polygons[idx], *inputs.imgs_info[idx].ori_shape)  # type: ignore[index,union-attr]
            )
            target_info.append(
                {
                    "boxes": inputs.bboxes[idx].data if inputs.bboxes is not None else torch.empty((0, 4)),
                    "masks": rles,
                    "labels": inputs.labels[idx],  # type: ignore[index]
                },
            )
        return {"preds": pred_info, "target": target_info}

    def compute_metrics(self, metric: Metric) -> dict:
        """Compute evaluation metrics for the model.

        Args:
            metric (Metric): Metric object to compute the evaluation metrics.

        Returns:
            dict: Computed metrics.
        """
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._compute_metrics(metric, **compute_kwargs)

    def _create_label_info_from_ov_ir(self) -> LabelInfo:
        """Create label information from the OpenVINO IR model.

        Reads label information from the OpenVINO IR model's runtime information
        and constructs a LabelInfo object.

        Returns:
            LabelInfo: Label information extracted from the OpenVINO IR model.
        """
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            ir_label_info = LabelInfo.from_json(serialized)
            if ir_label_info.label_names[0] == "otx_empty_lbl":
                ir_label_info.label_names.pop(0)
                ir_label_info.label_ids.pop(0)
                ir_label_info.label_groups[0].pop(0)
            return ir_label_info

        return super()._create_label_info_from_ov_ir()
