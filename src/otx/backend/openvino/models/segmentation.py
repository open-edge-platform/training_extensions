# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

import json
import logging as log
from typing import TYPE_CHECKING, Any

import numpy as np
from model_api.tilers import SemanticSegmentationTiler
from torchvision import tv_tensors

from otx.backend.openvino.models.base import OVModel
from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.metrics import MetricInput
from otx.core.metrics.dice import SegmCallable
from otx.core.types.label import SegLabelInfo
from otx.core.types.task import OTXTaskType
from otx.data.torch import TorchDataBatch, TorchPredBatch

if TYPE_CHECKING:
    from model_api.models.utils import ImageResultWithSoftPrediction

    from otx.core.metrics import MetricCallable


class OVSegmentationModel(OVModel):
    """Semantic segmentation model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX segmentation model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Segmentation",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
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
        self._task = OTXTaskType.SEMANTIC_SEGMENTATION

    def _setup_tiler(self) -> None:
        """Setup tiler for tile task."""
        execution_mode = "async" if self.async_inference else "sync"
        # Note: Disable async_inference as tiling has its own sync/async implementation
        self.async_inference = False
        self.model = SemanticSegmentationTiler(self.model, execution_mode=execution_mode)
        log.info(
            f"Enable tiler with tile size: {self.model.tile_size} \
                and overlap: {self.model.tiles_overlap}",
        )

    def _customize_outputs(
        self,
        outputs: list[ImageResultWithSoftPrediction],
        inputs: TorchDataBatch,
    ) -> TorchPredBatch | OTXBatchLossEntity:
        masks = [tv_tensors.Mask(np.expand_dims(mask.resultImage, axis=0), device=self.device) for mask in outputs]
        predicted_f_vectors = (
            [out.feature_vector for out in outputs] if outputs and outputs[0].feature_vector.size != 1 else []
        )
        return TorchPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
            feature_vector=predicted_f_vectors,
        )

    def prepare_metric_inputs(
        self,
        preds: TorchPredBatch,  # type: ignore[override]
        inputs: TorchDataBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Convert prediction and input entities to a format suitable for metric computation.

        Args:
            preds (TorchPredBatch): The predicted segmentation batch entity containing predicted masks.
            inputs (TorchDataBatch): The input segmentation batch entity containing ground truth masks.

        Returns:
            MetricInput: A list of dictionaries where each dictionary contains 'preds' and 'target' keys
            corresponding to the predicted and target masks for metric evaluation.
        """
        if preds.masks is None:
            msg = "The predicted masks are not provided."
            raise ValueError(msg)

        if inputs.masks is None:
            msg = "The input ground truth masks are not provided."
            raise ValueError(msg)

        return [
            {
                "preds": pred_mask,
                "target": target_mask,
            }
            for pred_mask, target_mask in zip(preds.masks, inputs.masks)
        ]

    def _create_label_info_from_ov_ir(self) -> SegLabelInfo:
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            label_info = json.loads(ov_model.get_rt_info(["model_info", "label_info"]).value)
            return SegLabelInfo(**label_info)

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)
