# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Class definition for detection model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

import copy
import json
import logging as log
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as f
from model_api.tilers import SemanticSegmentationTiler
from torchvision import tv_tensors

from otx.core.config.data import TileConfig
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.tile import OTXTileBatchDataEntity
from otx.core.exporter.base import OTXModelExporter
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import MetricInput
from otx.core.metrics.dice import SegmCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel, OVModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.export import TaskLevelExportParameters
from otx.core.types.label import LabelInfo, LabelInfoTypes, SegLabelInfo
from otx.core.utils.tile_merge import SegmentationTileMerge
from otx.data.torch import OTXDataBatch, OTXPredBatch

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from model_api.models.utils import ImageResultWithSoftPrediction
    from torch import Tensor

    from otx.core.metrics import MetricCallable
    from otx.core.model.base import DataInputParams


class OTXSegmentationModel(OTXModel):
    """Semantic Segmentation model used in OTX.

    Args:
        label_info (LabelInfoTypes): Information about the hierarchical labels.
        data_input_params (DataInputParams): Parameters for data input.
        model_name (str, optional): Name of the model. Defaults to "otx_segmentation_model".
        optimizer (OptimizerCallable, optional): Callable for the optimizer. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Callable for the learning rate scheduler.
        Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Callable for the metric. Defaults to SegmCallable.
        torch_compile (bool, optional): Flag to indicate whether to use torch.compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: str = "otx_segmentation_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = SegmCallable,  # type: ignore[assignment]
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ):
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _customize_inputs(self, entity: OTXDataBatch) -> dict[str, Any]:
        if self.training:
            mode = "loss"
        elif self.explain_mode:
            mode = "explain"
        else:
            mode = "predict"

        masks = torch.vstack(entity.masks).long() if mode == "loss" else None
        return {"inputs": entity.images, "img_metas": entity.imgs_info, "masks": masks, "mode": mode}

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

        preds = outputs["preds"] if self.explain_mode else outputs
        feature_vector = outputs["feature_vector"] if self.explain_mode else None
        masks = [
            tv_tensors.Mask(mask.unsqueeze(0), device=self.device)
            if mask.ndim == 2
            else tv_tensors.Mask(mask, device=self.device)
            for mask in preds
        ]

        return OTXPredBatch(
            batch_size=len(preds),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
            feature_vector=feature_vector,
        )

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        if self.label_info.label_names[0] == "otx_background_lbl":
            # remove otx background label for export
            modified_label_info = copy.deepcopy(self.label_info)
            modified_label_info.label_names.pop(0)
            modified_label_info.label_ids.pop(0)
        else:
            modified_label_info = self.label_info

        return super()._export_parameters.wrap(
            model_type="Segmentation",
            task_type="segmentation",
            return_soft_prediction=True,
            soft_threshold=0.5,
            blur_strength=-1,
            label_info=modified_label_info,
            tile_config=self.tile_config if self.tile_config.enable_tiler else None,
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
            output_names=["preds", "feature_vector"] if self.explain_mode else None,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: OTXPredBatch,  # type: ignore[override]
        inputs: OTXDataBatch,  # type: ignore[override]
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

    @staticmethod
    def _dispatch_label_info(label_info: LabelInfoTypes) -> LabelInfo:
        if isinstance(label_info, int):
            return SegLabelInfo.from_num_classes(num_classes=label_info)
        if isinstance(label_info, Sequence) and all(isinstance(name, str) for name in label_info):
            return SegLabelInfo(
                label_names=label_info,
                label_groups=[label_info],
                label_ids=[str(i) for i in range(len(label_info))],
            )
        if isinstance(label_info, SegLabelInfo):
            return label_info

        raise TypeError(label_info)

    def forward_tiles(self, inputs: OTXTileBatchDataEntity) -> OTXPredBatch:
        """Unpack segmentation tiles.

        Args:
            inputs (TileBatchSegDataEntity): Tile batch data entity.

        Returns:
            TorchPredBatch: Merged semantic segmentation prediction.
        """
        if self.explain_mode:
            msg = "Explain mode is not supported for tiling"
            raise NotImplementedError(msg)

        tile_preds: list[OTXPredBatch] = []
        tile_attrs: list[list[dict[str, int | str]]] = []
        merger = SegmentationTileMerge(
            inputs.imgs_info,
            self.num_classes,
            self.tile_config,
            self.explain_mode,
        )
        for batch_tile_attrs, batch_tile_input in inputs.unbind():
            tile_size = batch_tile_attrs[0]["tile_size"]
            output = self.model(
                inputs=batch_tile_input.images,
                img_metas=batch_tile_input.imgs_info,
                mode="tensor",
            )
            output = self._customize_outputs(
                outputs=f.interpolate(output, size=tile_size, mode="bilinear", align_corners=True),
                inputs=batch_tile_input,
            )
            if isinstance(output, OTXBatchLossEntity):
                msg = "Loss output is not supported for tile merging"
                raise TypeError(msg)
            tile_preds.append(output)
            tile_attrs.append(batch_tile_attrs)
        pred_entities = merger.merge(tile_preds, tile_attrs)

        pred_entity = OTXPredBatch(
            batch_size=inputs.batch_size,
            images=torch.stack([pred_entity.image for pred_entity in pred_entities]),
            imgs_info=[pred_entity.img_info for pred_entity in pred_entities],
            masks=[pred_entity.masks for pred_entity in pred_entities],
            scores=[],
        )
        if self.explain_mode:
            pred_entity.saliency_map = [pred_entity.saliency_map for pred_entity in pred_entities]
            pred_entity.feature_vector = [pred_entity.feature_vector for pred_entity in pred_entities]

        return pred_entity

    def forward_for_tracing(self, image: Tensor) -> Tensor | dict[str, Tensor]:
        """Model forward function used for the model tracing during model exportation."""
        if self.explain_mode:
            outputs = self.model(inputs=image, mode="explain")
            outputs["preds"] = torch.softmax(outputs["preds"], dim=1)
            return outputs

        outputs = self.model(inputs=image, mode="tensor")
        return torch.softmax(outputs, dim=1)

    def forward_explain(self, inputs: OTXDataBatch) -> OTXPredBatch:
        """Model forward explain function."""
        outputs = self.model(inputs=inputs.images, mode="explain")

        return OTXPredBatch(
            batch_size=len(outputs["preds"]),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=outputs["preds"],
            feature_vector=outputs["feature_vector"],
        )

    def get_dummy_input(self, batch_size: int = 1) -> OTXDataBatch:  # type: ignore[override]
        """Returns a dummy input for semantic segmentation model."""
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
        return OTXDataBatch(batch_size, images, imgs_info=infos, masks=[])  # type: ignore[arg-type]


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
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        masks = [tv_tensors.Mask(np.expand_dims(mask.resultImage, axis=0), device=self.device) for mask in outputs]
        predicted_f_vectors = (
            [out.feature_vector for out in outputs] if outputs and outputs[0].feature_vector.size != 1 else []
        )
        return OTXPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=[],
            masks=masks,
            feature_vector=predicted_f_vectors,
        )

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: OTXPredBatch,  # type: ignore[override]
        inputs: OTXDataBatch,  # type: ignore[override]
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
