# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.classification import (
    MulticlassClsBatchDataEntity,
    MulticlassClsBatchPredEntity,
)
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.accuracy import (
    HLabelClsMetricCallable,
    MultiClassClsMetricCallable,
    MultiLabelClsMetricCallable,
)
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel, DataInputParams
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import HLabelInfo, LabelInfo, LabelInfoTypes

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import ClassificationResult

    from otx.core.metrics import MetricCallable


class OTXMulticlassClsModel(OTXModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity]):
    """Base class for the classification models used in OTX."""

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "multiclass_classification_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MultiClassClsMetricCallable,
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

    def _customize_inputs(self, inputs: MulticlassClsBatchDataEntity) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        return {
            "images": inputs.stacked_images,
            "labels": torch.cat(inputs.labels, dim=0),
            "imgs_info": inputs.imgs_info,
            "mode": mode,
        }

    def _customize_outputs(
        self,
        outputs: Any,  # noqa: ANN401
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return OTXBatchLossEntity(loss=outputs)

        if self.explain_mode:
            return MulticlassClsBatchPredEntity(
                batch_size=inputs.batch_size,
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=outputs["scores"],
                labels=outputs["labels"],
                saliency_map=outputs["saliency_map"],
                feature_vector=outputs["feature_vector"],
            )

        # To list, batch-wise
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs["logits"]
        scores = torch.unbind(logits, 0)
        preds = logits.argmax(-1, keepdim=True).unbind(0)

        return MulticlassClsBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.stacked_images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            labels=preds,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        return super()._export_parameters.wrap(
            model_type="Classification",
            task_type="classification",
            multilabel=False,
            hierarchical=False,
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
        preds: MulticlassClsBatchPredEntity,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }

    def _reset_prediction_layer(self, num_classes: int) -> None:
        return

    def get_dummy_input(self, batch_size: int = 1) -> MulticlassClsBatchDataEntity:
        """Returns a dummy input for classification model."""
        images = [torch.rand(3, *self.data_input_params) for _ in range(batch_size)]
        labels = [torch.LongTensor([0])] * batch_size
        return MulticlassClsBatchDataEntity(batch_size, images, [], labels=labels)

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        return self.model(images=image)


class OVMulticlassClassificationModel(
    OVModel[MulticlassClsBatchDataEntity, MulticlassClsBatchPredEntity],
):
    """Classification model compatible for OpenVINO IR inference.

    It can consume OpenVINO IR model path or model name from Intel OMZ repository
    and create the OTX classification model compatible for OTX testing pipeline.
    """

    def __init__(
        self,
        model_name: str,
        model_type: str = "Classification",
        async_inference: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = False,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = MultiClassClsMetricCallable,
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
        outputs: list[ClassificationResult],
        inputs: MulticlassClsBatchDataEntity,
    ) -> MulticlassClsBatchPredEntity:
        pred_labels = [torch.tensor(out.top_labels[0][0], dtype=torch.long, device=self.device) for out in outputs]
        pred_scores = [torch.tensor(out.top_labels[0][2], device=self.device) for out in outputs]

        if outputs and outputs[0].saliency_map.size != 0:
            # Squeeze dim 4D => 3D, (1, num_classes, H, W) => (num_classes, H, W)
            predicted_s_maps = [out.saliency_map[0] for out in outputs]

            # Squeeze dim 2D => 1D, (1, internal_dim) => (internal_dim)
            predicted_f_vectors = [out.feature_vector[0] for out in outputs]
            return MulticlassClsBatchPredEntity(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=pred_scores,
                labels=pred_labels,
                saliency_map=predicted_s_maps,
                feature_vector=predicted_f_vectors,
            )

        return MulticlassClsBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=pred_scores,
            labels=pred_labels,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: MulticlassClsBatchPredEntity,
        inputs: MulticlassClsBatchDataEntity,
    ) -> MetricInput:
        pred = torch.tensor(preds.labels)
        target = torch.tensor(inputs.labels)
        return {
            "preds": pred,
            "target": target,
        }
