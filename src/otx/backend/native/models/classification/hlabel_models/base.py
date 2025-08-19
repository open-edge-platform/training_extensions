# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for classification model entity used in OTX."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.metrics import MetricInput
from otx.metrics.accuracy import (
    HLabelClsMetricCallable,
)
from otx.types.export import TaskLevelExportParameters
from otx.types.label import HLabelInfo, LabelInfo, LabelInfoTypes
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.metrics import MetricCallable


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
        freeze_backbone: bool = False,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = HLabelClsMetricCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            task=OTXTaskType.H_LABEL_CLS,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

        if freeze_backbone:
            classification_layers = self._identify_classification_layers()
            for name, param in self.named_parameters():
                param.requires_grad = name in classification_layers

    @abstractmethod
    def _create_model(self, head_config: dict | None = None) -> nn.Module:  # type: ignore[override]
        """Create a PyTorch model for this class."""

    def _identify_classification_layers(self, prefix: str = "model.") -> list[str]:
        """Simple identification of the classification layers. Used for incremental learning."""
        # identify classification layers
        sample_config = deepcopy(self.label_info.as_head_config_dict())
        sample_config["num_classes"] = 5
        sample_model_dict = self._create_model(head_config=sample_config).state_dict()
        sample_config["num_classes"] = 6
        incremental_model_dict = self._create_model(head_config=sample_config).state_dict()
        # iterate over the model dict and compare the shapes.
        # Add the key to the list if the shapes are different
        return [
            prefix + key
            for key in sample_model_dict
            if sample_model_dict[key].shape != incremental_model_dict[key].shape
        ]

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

        # To list, batch-wise
        if isinstance(outputs, dict):
            scores = outputs["scores"]
            labels = outputs["labels"]
        else:
            scores = outputs
            labels = outputs.argmax(-1, keepdim=True)

        if self.explain_mode:
            return OTXPredBatch(
                batch_size=inputs.batch_size,
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                labels=list(labels),
                scores=list(scores),
                saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
                feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
            )

        return OTXPredBatch(
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
        preds: OTXPredBatch,
        inputs: OTXDataBatch,
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
        if isinstance(label_info, dict):
            if "label_ids" not in label_info:
                # NOTE: This is for backward compatibility
                label_info["label_ids"] = label_info["label_names"]
            return HLabelInfo(**label_info)
        if isinstance(label_info, HLabelInfo):
            if not hasattr(label_info, "label_ids"):
                # NOTE: This is for backward compatibility
                label_info.label_ids = label_info.label_names
            return label_info

        raise TypeError(label_info)

    def get_dummy_input(self, batch_size: int = 1) -> OTXDataBatch:  # type: ignore[override]
        """Returns a dummy input for classification OV model."""
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

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            return self.model(images=image, mode="explain")

        return self.model(images=image, mode="tensor")
