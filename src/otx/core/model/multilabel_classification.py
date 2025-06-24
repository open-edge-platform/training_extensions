# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import (
    MultiLabelClsMetricCallable,
)
from otx.core.model.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfoTypes
from otx.data.torch import OTXDataBatch, OTXPredBatch

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import ClassificationResult

    from otx.core.metrics import MetricCallable


class OTXMultilabelClsModel(OTXModel):
    """Multilabel classification model used in OTX.

    Args:
    label_info (LabelInfoTypes): Information about the hierarchical labels.
    data_input_params (DataInputParams): Parameters for data input.
    model_name (str, optional): Name of the model. Defaults to "multilabel_classification_model".
    optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
    scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
    Defaults to DefaultSchedulerCallable.
    metric (MetricCallable, optional): Callable for the metric. Defaults to HLabelClsMetricCallable.
    torch_compile (bool, optional): Flag to indicate whether to use torch.compile. Defaults to False.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "multiclass_classification_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiLabelClsMetricCallable,
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

    def _customize_inputs(self, inputs: OTXDataBatch) -> dict[str, Any]:
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
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        if self.explain_mode:
            return OTXPredBatch(
                batch_size=inputs.batch_size,
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=list(outputs["labels"]),
                scores=list(outputs["scores"]),
                saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
                feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
            )

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)

        return OTXPredBatch(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=list(logits.argmax(-1, keepdim=True).unbind(0)),
            scores=list(scores),
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Classification",
            task_type="classification",
            multilabel=True,
            hierarchical=False,
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
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
    ) -> MetricInput:
        return {
            "preds": preds.scores,
            "target": inputs.labels,
        }

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model.forward(image)

    def get_dummy_input(self, batch_size: int = 1) -> OTXDataBatch:  # type: ignore[override]
        """Returns a dummy input for classification model."""
        images = torch.stack([torch.rand(3, *self.data_input_params.input_size) for _ in range(batch_size)])
        labels = [torch.LongTensor([0])] * batch_size
        return OTXDataBatch(batch_size=batch_size, images=images, labels=labels)

    def forward_explain(self, inputs: OTXDataBatch) -> OTXPredBatch:
        """Model forward explain function."""
        outputs = self.model(images=inputs.images, mode="explain")

        return OTXPredBatch(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            labels=list(outputs["preds"]),
            scores=list(outputs["scores"]),
            saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
            feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
        )


class OVMultilabelClassificationModel(OVModel):
    """Multilabel classification model compatible for OpenVINO IR inference.

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
        metric: MetricCallable = MultiLabelClsMetricCallable,
        **kwargs,
    ) -> None:
        model_api_configuration = model_api_configuration if model_api_configuration else {}
        model_api_configuration.update({"multilabel": True, "confidence_threshold": 0.0})
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
        inputs: OTXDataBatch,
    ) -> OTXPredBatch:
        pred_scores = [
            torch.tensor([top_label.confidence for top_label in out.top_labels], device=self.device) for out in outputs
        ]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return OTXPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=[],
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return OTXPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=[],
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
    ) -> MetricInput:
        return {
            "preds": torch.stack(preds.scores),
            "target": torch.stack(inputs.labels),
        }
