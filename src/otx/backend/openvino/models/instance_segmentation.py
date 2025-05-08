# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for instance segmentation model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

import logging as log
from typing import TYPE_CHECKING, Any, Literal

import torch
from model_api.tilers import InstanceSegmentationTiler
from torchmetrics import Metric
from torchvision import tv_tensors

from otx.backend.openvino.models.base import OVModel
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.metrics import MetricInput
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.types.label import LabelInfo
from otx.core.utils.mask_util import encode_rle, polygon_to_rle
from otx.data.torch import TorchDataBatch, TorchPredBatch
from otx.core.types.task import OTXTaskType

if TYPE_CHECKING:
    from model_api.adapters import OpenvinoAdapter
    from model_api.models.utils import InstanceSegmentationResult

    from otx.core.metrics import MetricCallable


class OVInstanceSegmentationModel(
    OVModel,
):
    """Instance segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX detection model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "MaskRCNN",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
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
        self._task = OTXTaskType.INSTANCE_SEGMENTATION

    def _setup_tiler(self) -> None:
        """Setup tiler for tile task."""
        execution_mode = "async" if self.async_inference else "sync"
        # Note: Disable async_inference as tiling has its own sync/async implementation
        self.async_inference = False
        self.model = InstanceSegmentationTiler(self.model, execution_mode=execution_mode)
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
        outputs: list[InstanceSegmentationResult],
        inputs: TorchDataBatch,
    ) -> TorchPredBatch | OTXBatchLossEntity:
        # add label index
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
                    device=self.device,
                    dtype=torch.float32,
                ),
            )
            # NOTE: OTX 1.5 filter predictions with result_based_confidence_threshold,
            # but OTX 2.0 doesn't have it in configuration.
            scores.append(torch.tensor(output.scores.reshape(-1), device=self.device))
            masks.append(torch.tensor(output.masks, device=self.device))
            labels.append(torch.tensor(output.labels.reshape(-1) - 1, device=self.device, dtype=torch.long))

        if outputs and outputs[0].saliency_map:
            predicted_s_maps = []
            for out in outputs:
                image_map = torch.tensor(
                    [s_map for s_map in out.saliency_map if s_map.size > 0],
                    dtype=torch.float32,
                    device=self.device,
                )
                predicted_s_maps.append(image_map)

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return TorchPredBatch(
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

        return TorchPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks if any(mask.numel() > 0 for mask in masks) else None,
            labels=labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: TorchPredBatch,  # type: ignore[override]
        inputs: TorchDataBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Convert the prediction entity to the format that the metric can compute and cache the ground truth.

        This function will convert mask to RLE format and cache the ground truth for the current batch.

        Args:
            preds (TorchPredBatch): Current batch predictions.
            inputs (TorchDataBatch): Current batch ground-truth inputs.

        Returns:
            dict[str, list[dict[str, Tensor]]]: The converted predictions and ground truth.
        """
        target_info = []

        _bboxes = preds.bboxes if preds.bboxes is not None else None
        _masks = preds.masks if preds.masks is not None else None
        pred_info = [
            {
                "boxes": _bboxes[idx].data if _bboxes is not None else torch.empty((0, 4)),
                "masks": [encode_rle(mask) for mask in _masks[idx].data]
                if _masks is not None and len(_masks)
                else torch.empty((0,)),
                "scores": preds.scores[idx],  # type: ignore[index]
                "labels": preds.labels[idx],  # type: ignore[index]
            }
            for idx in range(len(preds.labels))  # type: ignore[arg-type]
        ]

        _bboxes = inputs.bboxes if inputs.bboxes is not None else None
        _masks = inputs.masks if inputs.masks is not None else None
        for idx in range(len(inputs.labels)):  # type: ignore[arg-type]
            rles = (
                [encode_rle(mask) for mask in _masks[idx].data]
                if _masks is not None and len(_masks[idx]) > 0
                else polygon_to_rle(inputs.polygons[idx], *inputs.imgs_info[idx].ori_shape)  # type: ignore[index,union-attr]
            )
            target_info.append(
                {
                    "boxes": _bboxes[idx].data if _bboxes is not None else torch.empty((0, 4)),
                    "masks": rles,
                    "labels": inputs.labels[idx],  # type: ignore[index]
                },
            )
        return {"preds": pred_info, "target": target_info}

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
        compute_kwargs = {"best_confidence_threshold": best_confidence_threshold}
        return super()._log_metrics(meter, key, **compute_kwargs)

    def _create_label_info_from_ov_ir(self) -> LabelInfo:
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            ir_label_info = LabelInfo.from_json(serialized)
            # workaround to hide extra otx_empty_lbl
            if ir_label_info.label_names[0] == "otx_empty_lbl":
                ir_label_info.label_names.pop(0)
                ir_label_info.label_ids.pop(0)
                ir_label_info.label_groups[0].pop(0)
            return ir_label_info

        return super()._create_label_info_from_ov_ir()
