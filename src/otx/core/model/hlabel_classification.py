# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import (
    HLabelClsMetricCallable,
)
from otx.core.model.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import HLabelInfo, LabelInfo, LabelInfoTypes
from otx.data.torch import TorchDataBatch, TorchPredBatch

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import ClassificationResult
    from torch import nn

    from otx.core.metrics import MetricCallable


class OTXHlabelClsModel(OTXModel):
    """H-label classification models used in OTX.

    Args:
    label_info (HLabelInfo): Information about the hierarchical labels.
    data_input_params (DataInputParams): Parameters for data input.
    model_name (str, optional): Name of the model. Defaults to "hlabel_classification_model".
    optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
    scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
    Defaults to DefaultSchedulerCallable.
    metric (MetricCallable, optional): Callable for the metric. Defaults to HLabelClsMetricCallable.
    torch_compile (bool, optional): Flag to indicate whether to use torch.compile. Defaults to False.
    """

    label_info: HLabelInfo

    def __init__(
        self,
        label_info: HLabelInfo,
        data_input_params: DataInputParams,
        model_name: str = "hlabel_classification_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
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

    @abstractmethod
    def _create_model(self, head_config: dict | None = None) -> nn.Module:  # type: ignore[override]
        """Create a PyTorch model for this class."""

    def _identify_classification_layers(self, prefix: str = "model.") -> list[str]:
        """Simple identification of the classification layers. Used for incremental learning."""
        # identify classification layers
        sample_config = deepcopy(self.label_info.as_head_config_dict())
        sample_config["num_classes"] = 5
        sample_model_dict = self._build_model(head_config=sample_config).state_dict()
        sample_config["num_classes"] = 6
        incremental_model_dict = self._build_model(head_config=sample_config).state_dict()
        # iterate over the model dict and compare the shapes.
        # Add the key to the list if the shapes are different
        return [
            prefix + key
            for key in sample_model_dict
            if sample_model_dict[key].shape != incremental_model_dict[key].shape
        ]

    def _customize_inputs(self, inputs: TorchDataBatch) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.images,
            "labels": torch.vstack(inputs.labels),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: TorchDataBatch,
    ) -> TorchPredBatch | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        # To list, batch-wise
        if isinstance(outputs, dict):
            scores = outputs["scores"]
            labels = outputs["labels"]
        else:
            scores = outputs
            labels = outputs.argmax(-1, keepdim=True)

        if self.explain_mode:
            return TorchPredBatch(
                batch_size=inputs.batch_size,
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=list(labels),
                scores=list(scores),
                saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
                feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
            )

        return TorchPredBatch(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=list(labels),
            scores=list(scores),
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Classification",
            task_type="classification",
            multilabel=False,
            hierarchical=True,
            confidence_threshold=0.5,
            output_raw_scores=True,
        )

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            pad_value=0,
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration=None,
            output_names=["logits", "feature_vector", "saliency_map"] if self.explain_mode else None,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: TorchPredBatch,
        inputs: TorchDataBatch,
    ) -> MetricInput:
        hlabel_info: HLabelInfo = self.label_info  # type: ignore[assignment]

        _labels = torch.stack(preds.labels) if isinstance(preds.labels, list) else preds.labels
        _scores = torch.stack(preds.scores) if isinstance(preds.scores, list) else preds.scores
        if hlabel_info.num_multilabel_classes > 0:
            preds_multiclass = _labels[:, : hlabel_info.num_multiclass_heads]
            preds_multilabel = _scores[:, hlabel_info.num_multiclass_heads :]
            pred_result = torch.cat([preds_multiclass, preds_multilabel], dim=1)
        else:
            pred_result = _labels
        return {
            "preds": pred_result,
            "target": torch.vstack(inputs.labels),
        }

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        if not isinstance(label_info, HLabelInfo):
            raise TypeError(label_info)

        return label_info

    def get_dummy_input(self, batch_size: int = 1) -> TorchDataBatch:  # type: ignore[override]
        """Returns a dummy input for classification OV model."""
        images = torch.stack([torch.rand(3, *self.data_input_params.input_size) for _ in range(batch_size)])
        labels = [torch.LongTensor([0])] * batch_size
        return TorchDataBatch(batch_size=batch_size, images=images, labels=labels)

    def forward_explain(self, inputs: TorchDataBatch) -> TorchPredBatch:
        """Model forward explain function."""
        outputs = self.model(images=inputs.images, mode="explain")

        return TorchPredBatch(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=list(outputs["preds"]),
            scores=list(outputs["scores"]),
            saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
            feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
        )

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")


class OVHlabelClassificationModel(OVModel):
    """Hierarchical classification model compatible for OpenVINO IR inference.

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
        metric: MetricCallable = HLabelClsMetricCallable,
        **kwargs,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"hierarchical": True, "output_raw_scores": True})
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

            all_pred_labels.append(torch.tensor(predicted_labels, dtype=torch.long, device=self.device))
            all_pred_scores.append(torch.tensor(predicted_scores, device=self.device))

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return TorchPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=all_pred_scores,
                labels=all_pred_labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return TorchPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=all_pred_scores,
            labels=all_pred_labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: TorchPredBatch,
        inputs: TorchDataBatch,
    ) -> MetricInput:
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
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            return HLabelInfo.from_json(serialized)

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)
