# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from otx.backend.openvino.models.base import OVModel
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.metrics import MetricInput
from otx.metrics.accuracy import (
    HLabelClsMetricCallable,
)
from otx.types.label import HLabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from model_api.models.utils import ClassificationResult

    from otx.metrics import MetricCallable
    from otx.types import PathLike


class OVHlabelClassificationModel(OVModel):
    """Hierarchical classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = HLabelClsMetricCallable,
        **kwargs,
    ) -> None:
        """Initialize the hierarchical classification model.

        Args:
            model_path (PathLike): Path to the OpenVINO IR model.
            model_type (str): Type of the model (default: "Classification").
            async_inference (bool): Whether to enable asynchronous inference (default: True).
            max_num_requests (int | None): Maximum number of inference requests (default: None).
            use_throughput_mode (bool): Whether to use throughput mode (default: True).
            model_api_configuration (dict[str, Any] | None): Configuration for the model API (default: None).
            metric (MetricCallable): Metric callable for evaluation (default: HLabelClsMetricCallable).
            **kwargs: Additional keyword arguments.
        """
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"hierarchical": True, "output_raw_scores": True})
        super().__init__(
            model_path=model_path,
            model_type=model_type,
            async_inference=async_inference,
            max_num_requests=max_num_requests,
            use_throughput_mode=use_throughput_mode,
            model_api_configuration=model_api_configuration,
            metric=metric,
        )
        self._task = OTXTaskType.H_LABEL_CLS

    def _customize_outputs(
        self,
        outputs: list[ClassificationResult],
        inputs: OTXDataBatch,
    ) -> OTXPredBatch:
        """Customize the outputs of the model for hierarchical classification.

        Args:
            outputs (list[ClassificationResult]): List of classification results from the model.
            inputs (OTXDataBatch): Input data batch.

        Returns:
            OTXPredBatch: Customized prediction batch containing labels, scores, and optional saliency maps.
        """
        all_pred_labels = []
        all_pred_scores = []
        for output in outputs:
            logits = output.raw_scores
            predicted_labels = []
            predicted_scores = []
            cls_heads_info = self.model.hierarchical_info["cls_heads_info"]
            for i in range(cls_heads_info["num_multiclass_heads"]):
                logits_begin, logits_end = cls_heads_info["head_idx_to_logits_range"][str(i)]
                head_logits = logits[logits_begin:logits_end]
                j = np.argmax(head_logits)
                predicted_labels.append(j)
                predicted_scores.append(head_logits[j])

            if cls_heads_info["num_multilabel_classes"]:
                logits_begin = cls_heads_info["num_single_label_classes"]
                head_logits = logits[logits_begin:]

                for i in range(head_logits.shape[0]):
                    predicted_scores.append(head_logits[i])
                    if head_logits[i] > self.model.confidence_threshold:
                        predicted_labels.append(1)
                    else:
                        predicted_labels.append(0)

            all_pred_labels.append(torch.tensor(predicted_labels, dtype=torch.long))
            all_pred_scores.append(torch.tensor(predicted_scores))

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return OTXPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=all_pred_scores,
                labels=all_pred_labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return OTXPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=all_pred_scores,
            labels=all_pred_labels,
        )

    def prepare_metric_inputs(
        self,
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
    ) -> MetricInput:
        """Prepare inputs for metric computation.

        Converts predictions and ground truth inputs into a format suitable for metric evaluation.

        Args:
            preds (OTXPredBatch): Predicted batch containing labels and scores.
            inputs (OTXDataBatch): Input batch containing ground truth labels.

        Returns:
            MetricInput: A dictionary with 'preds' and 'target' keys for metric evaluation.
        """
        cls_heads_info = self.model.hierarchical_info["cls_heads_info"]
        num_multilabel_classes = cls_heads_info["num_multilabel_classes"]
        num_multiclass_heads = cls_heads_info["num_multiclass_heads"]
        if num_multilabel_classes > 0:
            preds_multiclass = torch.stack(preds.labels)[:, :num_multiclass_heads]
            preds_multilabel = torch.stack(preds.scores)[:, num_multiclass_heads:]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = torch.stack(preds.labels)
        return {
            "preds": pred_result,
            "target": torch.stack(inputs.labels),
        }

    def _create_label_info_from_ov_ir(self) -> HLabelInfo:
        """Create hierarchical label information from OpenVINO IR.

        Extracts label information from the OpenVINO IR model if available.

        Returns:
            HLabelInfo: Hierarchical label information.

        Raises:
            ValueError: If label information cannot be constructed from the OpenVINO IR.
        """
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            return HLabelInfo.from_json(serialized)

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)
