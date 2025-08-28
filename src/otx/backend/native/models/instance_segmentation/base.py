# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for instance segmentation model entity used in OTX."""

# type: ignore[override]

from __future__ import annotations

import copy
import logging as log
import types
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence

import torch
from torch import Tensor
from torchmetrics import Metric, MetricCollection
from torchvision import tv_tensors
from torchvision.models.detection.image_list import ImageList

from otx.backend.native.models.base import DefaultOptimizerCallable, DefaultSchedulerCallable, OTXModel
from otx.backend.native.models.instance_segmentation.segmentors.maskrcnn_tv import MaskRCNN
from otx.backend.native.models.instance_segmentation.segmentors.two_stage import TwoStageDetector
from otx.backend.native.models.utils.utils import InstanceData, load_checkpoint
from otx.backend.native.schedulers import LRSchedulerListCallable
from otx.backend.native.tools.explain.explain_algo import InstSegExplainAlgo, feature_vector_fn
from otx.backend.native.tools.tile_merge import InstanceSegTileMerge
from otx.config.data import TileConfig
from otx.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.data.entity.tile import OTXTileBatchDataEntity
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.data.entity.utils import stack_batch
from otx.data.utils.structures.mask.mask_util import encode_rle, polygon_to_rle
from otx.metrics import MetricInput
from otx.metrics.fmeasure import FMeasure
from otx.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.types.export import TaskLevelExportParameters
from otx.types.label import LabelInfoTypes
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import nn

    from otx.backend.native.models.base import DataInputParams
    from otx.metrics import MetricCallable


class OTXInstanceSegModel(OTXModel):
    """Base class for the Instance Segmentation models used in OTX.

    NOTE: OTXInstanceSegModel has many duplicate methods to OTXDetectionModel,
    however, it is not a subclass of OTXDetectionModel because it has different
    export parameters and different metric computation. Some refactor could be done
    to reduce the code duplication in the future.

    Args:
        label_info (LabelInfoTypes | int | Sequence): Information about the labels used in the model.
            If `int` is given, label info will be constructed from number of classes,
            if `Sequence` is given, label info will be constructed from the sequence of label names.
        data_input_params (DataInputParams): Parameters for the data input.
        model_name (str, optional): Name of the model. Defaults to "inst_segm_model".
        optimizer (OptimizerCallable, optional): Optimizer for the model. Defaults to DefaultOptimizerCallable.
        scheduler (LRSchedulerCallable | LRSchedulerListCallable, optional): Scheduler for the model.
            Defaults to DefaultSchedulerCallable.
        metric (MetricCallable, optional): Metric for evaluating the model.
            Defaults to MaskRLEMeanAPFMeasureCallable.
        torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
        tile_config (TileConfig, optional): Configuration for tiling. Defaults to TileConfig(enable_tiler=False).
        explain_mode (bool, optional): Whether to enable explainable AI mode. Defaults to False.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes | int | Sequence,
        data_input_params: DataInputParams,
        model_name: str = "inst_segm_model",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            task=OTXTaskType.INSTANCE_SEGMENTATION,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()
        self.model.get_results_from_head = self.get_results_from_head

    def _create_model(self, num_classes: int | None = None) -> nn.Module:
        num_classes = num_classes if num_classes is not None else self.num_classes
        detector = self._build_model(num_classes)
        if hasattr(detector, "init_weights"):
            detector.init_weights()
        if isinstance(self.load_from, dict):
            load_checkpoint(detector, self.load_from[self.model_name], map_location="cpu")
        elif self.load_from is not None:
            load_checkpoint(detector, self.load_from, map_location="cpu")

        return detector

    def _customize_inputs(self, entity: OTXDataBatch) -> dict[str, Any]:
        if isinstance(entity.images, list):
            entity.images, entity.imgs_info = stack_batch(entity.images, entity.imgs_info, pad_size_divisor=32)  # type: ignore[assignment,arg-type]
        inputs: dict[str, Any] = {}

        inputs["entity"] = entity
        inputs["mode"] = "loss" if self.training else "predict"

        return inputs

    def _customize_outputs(
        self,
        outputs: list[InstanceData] | dict,
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for loss_name, loss_value in outputs.items():
                if isinstance(loss_value, Tensor):
                    losses[loss_name] = loss_value
                elif isinstance(loss_value, list):
                    losses[loss_name] = sum(_loss.mean() for _loss in loss_value)
            losses.pop("acc", None)
            return losses

        scores: list[Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        predictions = outputs["predictions"] if isinstance(outputs, dict) else outputs
        for img_info, prediction in zip(inputs.imgs_info, predictions):  # type: ignore[arg-type]
            scores.append(prediction.scores)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    prediction.bboxes,
                    format="XYXY",
                    canvas_size=img_info.ori_shape,  # type: ignore[union-attr]
                ),
            )
            output_masks = tv_tensors.Mask(
                prediction.masks,
                dtype=torch.bool,
            )
            masks.append(output_masks)
            labels.append(prediction.labels)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            saliency_map = outputs["saliency_map"].detach().cpu().numpy()
            feature_vector = outputs["feature_vector"].detach().cpu().numpy()

            return OTXPredBatch(
                batch_size=len(predictions),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                masks=masks,
                labels=labels,
                saliency_map=list(saliency_map),
                feature_vector=list(feature_vector),
            )

        return OTXPredBatch(
            batch_size=len(predictions),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            labels=labels,
        )

    def forward_tiles(self, inputs: OTXTileBatchDataEntity) -> OTXPredBatch:
        """Unpack instance segmentation tiles.

        Args:
            inputs (OTXTileBatchDataEntity): Tile batch data entity.

        Returns:
            TorchPredBatch: Merged instance segmentation prediction.
        """
        tile_preds: list[OTXPredBatch] = []
        tile_attrs: list[list[dict[str, int | str]]] = []
        merger = InstanceSegTileMerge(
            inputs.imgs_info,
            self.num_classes,
            self.tile_config,
            self.explain_mode,
        )
        for batch_tile_attrs, batch_tile_input in inputs.unbind():
            output = self.forward_explain(batch_tile_input) if self.explain_mode else self.forward(batch_tile_input)
            if isinstance(output, OTXBatchLossEntity):
                msg = "Loss output is not supported for tile merging"
                raise TypeError(msg)
            tile_preds.append(output)
            tile_attrs.append(batch_tile_attrs)
        pred_entities = merger.merge(tile_preds, tile_attrs)

        pred_entity = OTXPredBatch(
            batch_size=inputs.batch_size,
            images=[pred_entity.image for pred_entity in pred_entities],
            imgs_info=[pred_entity.img_info for pred_entity in pred_entities],
            scores=[pred_entity.scores for pred_entity in pred_entities],
            bboxes=[pred_entity.bboxes for pred_entity in pred_entities],
            labels=[pred_entity.label for pred_entity in pred_entities],
            masks=[pred_entity.masks for pred_entity in pred_entities],
            polygons=[pred_entity.polygons for pred_entity in pred_entities],  # type: ignore[misc]
        )
        if self.explain_mode:
            pred_entity.saliency_map = [pred_entity.saliency_map for pred_entity in pred_entities]
            pred_entity.feature_vector = [pred_entity.feature_vector for pred_entity in pred_entities]

        return pred_entity

    def forward_for_tracing(self, inputs: Tensor) -> tuple[Tensor, ...]:
        """Forward function for export."""
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)

    @property
    def _export_parameters(self) -> TaskLevelExportParameters:
        """Defines parameters required to export a particular model implementation."""
        modified_label_info = copy.deepcopy(self.label_info)
        # Instance segmentation needs to add empty label to satisfy MAPI wrapper requirements
        modified_label_info.label_names.insert(0, "otx_empty_lbl")
        modified_label_info.label_ids.insert(0, "None")
        modified_label_info.label_groups[0].insert(0, "otx_empty_lbl")

        return super()._export_parameters.wrap(
            model_type="MaskRCNN",
            task_type="instance_segmentation",
            confidence_threshold=self.hparams.get("best_confidence_threshold", 0.05),
            iou_threshold=0.5,
            tile_config=self.tile_config if self.tile_config.enable_tiler else None,
            label_info=modified_label_info,
        )

    def test_step(self, batch: OTXDataBatch, batch_idx: int) -> OTXPredBatch:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        preds = self.forward(inputs=batch)

        if isinstance(preds, OTXBatchLossEntity):
            raise TypeError(preds)

        # 1. Filter outputs by threshold
        preds = self._filter_outputs_by_threshold(preds)
        metric_inputs = self._convert_pred_entity_to_compute_metric(preds, batch)

        # 2. Update metric
        if isinstance(metric_inputs, dict):
            self.metric.update(**metric_inputs)
            return preds

        if isinstance(metric_inputs, list) and all(isinstance(inp, dict) for inp in metric_inputs):
            for inp in metric_inputs:
                self.metric.update(**inp)
            return preds

        raise TypeError(metric_inputs)

    def predict_step(
        self,
        batch: OTXDataBatch | OTXTileBatchDataEntity,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> OTXPredBatch:
        """Step function called during PyTorch Lightning Trainer's predict."""
        if self.explain_mode:
            return self._filter_outputs_by_threshold(self.forward_explain(inputs=batch))  # type: ignore[arg-type]

        outputs = self._filter_outputs_by_threshold(self.forward(inputs=batch))  # type: ignore[arg-type]

        if isinstance(outputs, OTXBatchLossEntity):
            raise TypeError(outputs)

        return outputs

    @property
    def best_confidence_threshold(self) -> float:
        """Best confidence threshold to filter outputs.

        Always returns the current value from hparams, with 0.5 as fallback.
        This ensures the threshold is always up-to-date after validation updates it.
        """
        threshold = self.hparams.get("best_confidence_threshold", None)
        if threshold is None:
            # Only log warning once to avoid spam
            if not getattr(self, "_threshold_warning_logged", False):
                log.warning("There is no predefined best_confidence_threshold, 0.5 will be used as default.")
                self._threshold_warning_logged = True
            return 0.5
        return float(threshold)

    def on_load_checkpoint(self, ckpt: dict[str, Any]) -> None:
        """Load state_dict from checkpoint.

        For instance segmentation, it is needed to update confidence threshold and F1 score information when
        the metric is FMeasure.
        """
        hyper_parameters = ckpt.get("hyper_parameters", {})

        # Load best confidence threshold (legacy and new format)
        if best_confidence_threshold := ckpt.get("confidence_threshold") or hyper_parameters.get(
            "best_confidence_threshold",
            None,
        ):
            self.hparams["best_confidence_threshold"] = best_confidence_threshold
        super().on_load_checkpoint(ckpt)

    def _log_metrics(self, meter: Metric, key: Literal["val", "test"], **compute_kwargs) -> None:
        if key == "val":
            super()._log_metrics(meter, key)

            # NOTE: Only update best_confidence_threshold when we achieve a NEW best F1 score
            fmeasure = None
            if isinstance(meter, MetricCollection) and (fmeasure := getattr(meter, "FMeasure", None)):
                pass  # fmeasure is set
            elif isinstance(meter, FMeasure):
                fmeasure = meter

            if fmeasure is not None and hasattr(fmeasure, "best_confidence_threshold"):
                self.hparams["best_confidence_threshold"] = fmeasure.best_confidence_threshold

        if key == "test":
            # NOTE: Test metric logging should use `best_confidence_threshold` found previously.
            best_confidence_threshold = self.hparams.get("best_confidence_threshold", None)
            compute_kwargs = (
                {"best_confidence_threshold": best_confidence_threshold} if best_confidence_threshold else {}
            )

            super()._log_metrics(meter, key, **compute_kwargs)

    def _filter_outputs_by_threshold(self, outputs: OTXPredBatch) -> OTXPredBatch:
        scores = []
        bboxes = []
        labels = []
        masks = []
        polygons = []

        for i in range(len(outputs.imgs_info)):  # type: ignore[arg-type]
            _scores = outputs.scores[i] if outputs.scores is not None else None
            _bboxes = outputs.bboxes[i] if outputs.bboxes is not None else None
            _masks = outputs.masks[i] if outputs.masks is not None else None
            _polygons = outputs.polygons[i] if outputs.polygons is not None else None
            _labels = outputs.labels[i] if outputs.labels is not None else None

            filtered_idx = torch.where(_scores > self.best_confidence_threshold)
            scores.append(_scores[filtered_idx])
            bboxes.append(_bboxes[filtered_idx])
            labels.append(_labels[filtered_idx])

            if _masks is not None:
                masks.append(_masks[filtered_idx])
            if _polygons is not None:
                polygons.append(_polygons[filtered_idx])

        outputs.scores = scores
        outputs.bboxes = bboxes
        outputs.labels = labels
        outputs.masks = masks
        outputs.polygons = polygons
        return outputs

    def _convert_pred_entity_to_compute_metric(
        self,
        preds: OTXPredBatch,  # type: ignore[override]
        inputs: OTXDataBatch,  # type: ignore[override]
    ) -> MetricInput:
        """Convert the prediction entity to the format that the metric can compute and cache the ground truth.

        This function will convert mask to RLE format and cache the ground truth for the current batch.

        Args:
            preds (TorchPredBatch): Current batch predictions.
            inputs (TorchDataBatch): Current batch ground-truth inputs.

        Returns:
            dict[str, list[dict[str, Tensor]]]: The converted predictions and ground truth.
        """
        pred_info = []
        target_info = []
        for i in range(len(preds.imgs_info)):  # type: ignore[arg-type]
            bboxes = preds.bboxes[i] if preds.bboxes is not None else None
            masks = preds.masks[i] if preds.masks is not None else None
            scores = preds.scores[i] if preds.scores is not None else None
            labels = preds.labels[i] if preds.labels is not None else None

            pred_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": [encode_rle(mask) for mask in masks.data],
                    "scores": scores,
                    "labels": labels,
                },
            )
        for i in range(len(inputs.imgs_info)):  # type: ignore[arg-type]
            imgs_info = inputs.imgs_info[i] if inputs.imgs_info is not None else None
            bboxes = inputs.bboxes[i] if inputs.bboxes is not None else None
            masks = inputs.masks[i] if inputs.masks is not None else None
            polygons = inputs.polygons[i] if inputs.polygons is not None else None
            labels = inputs.labels[i] if inputs.labels is not None else None

            rles = (
                [encode_rle(mask) for mask in masks.data]
                if len(masks)
                else polygon_to_rle(polygons, *imgs_info.ori_shape)  # type: ignore[union-attr,arg-type]
            )
            target_info.append(
                {
                    "boxes": bboxes.data,
                    "masks": rles,
                    "labels": labels,
                },
            )
        return {"preds": pred_info, "target": target_info}

    def get_dummy_input(self, batch_size: int = 1) -> OTXDataBatch:  # type: ignore[override]
        """Returns a dummy input for instance segmentation model."""
        images = [torch.rand(3, *self.data_input_params.input_size) for _ in range(batch_size)]
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return OTXDataBatch(batch_size, images, imgs_info=infos)

    def forward_explain(self, inputs: OTXDataBatch) -> OTXPredBatch:
        """Model forward function."""
        if isinstance(inputs, OTXTileBatchDataEntity):
            return self.forward_tiles(inputs)

        self.model.feature_vector_fn = feature_vector_fn
        self.model.explain_fn = self.get_explain_fn()

        outputs = (
            self._forward_explain_inst_seg(self.model, **self._customize_inputs(inputs))
            if self._customize_inputs != OTXInstanceSegModel._customize_inputs
            else self._forward_explain_inst_seg(self.model, inputs)
        )

        return (
            self._customize_outputs(outputs, inputs)
            if self._customize_outputs != OTXInstanceSegModel._customize_outputs
            else outputs["predictions"]
        )

    @staticmethod
    @torch.no_grad()
    def _forward_explain_inst_seg(
        self: TwoStageDetector,
        entity: OTXDataBatch,
        mode: str = "tensor",  # noqa: ARG004
    ) -> dict[str, Tensor]:
        """Forward func of the BaseDetector instance, which located in is in ExplainableOTXInstanceSegModel().model."""
        x = self.backbone(entity.images) if isinstance(self, MaskRCNN) else self.extract_feat(entity.images)

        feature_vector = self.feature_vector_fn(x)
        predictions = self.get_results_from_head(x, entity)

        if isinstance(predictions, tuple) and isinstance(predictions[0], Tensor):
            # Export case, consists of tensors
            # For OV task saliency map are generated on MAPI side
            saliency_map = torch.empty(1, dtype=torch.uint8)
        elif isinstance(predictions, list) and isinstance(predictions[0], (InstanceData, dict)):
            # Predict case, consists of InstanceData or dict
            saliency_map = self.explain_fn(predictions)
        else:
            msg = f"Unexpected predictions type: {type(predictions)}"
            raise TypeError(msg)

        return {
            "predictions": predictions,
            "feature_vector": feature_vector,
            "saliency_map": saliency_map,
        }

    def get_results_from_head(
        self,
        x: tuple[Tensor],
        entity: OTXDataBatch,
    ) -> tuple[Tensor, Tensor, Tensor] | list[InstanceData] | list[dict[str, Tensor]]:
        """Get the results from the head of the instance segmentation model.

        Args:
            x (tuple[Tensor]): The features from backbone and neck.
            data_samples (OptSampleList | None): A list of data samples.

        Returns:
            tuple[Tensor, Tensor, Tensor] | list[InstanceData]: The predicted results from the head of the model.
            Tuple for the Export case, list for the Predict case.
        """
        from otx.backend.native.models.instance_segmentation.maskrcnn_tv import MaskRCNNTV
        from otx.backend.native.models.instance_segmentation.rtmdet_inst import RTMDetInst

        if isinstance(self, MaskRCNNTV):
            ori_shapes = [img_info.ori_shape for img_info in entity.imgs_info]  # type: ignore[union-attr]
            img_shapes = [img_info.img_shape for img_info in entity.imgs_info]  # type: ignore[union-attr]
            image_list = ImageList(entity.images, img_shapes)
            proposals, _ = self.model.rpn(image_list, x)
            detections, _ = self.model.roi_heads(
                x,
                proposals,
                image_list.image_sizes,
            )
            scale_factors = [
                img_meta.scale_factor if img_meta.scale_factor else (1.0, 1.0)  # type: ignore[union-attr]
                for img_meta in entity.imgs_info  # type: ignore[union-attr]
            ]
            return self.model.postprocess(detections, ori_shapes, scale_factors)

        if isinstance(self, RTMDetInst):
            return self.model.bbox_head.predict(x, entity, rescale=False)
        rpn_results_list = self.model.rpn_head.predict(x, entity, rescale=False)
        return self.model.roi_head.predict(x, rpn_results_list, entity, rescale=True)

    def get_explain_fn(self) -> Callable:
        """Returns explain function."""
        explainer = InstSegExplainAlgo(num_classes=self.num_classes)
        return explainer.func

    @contextmanager
    def export_model_forward_context(self) -> Iterator[None]:
        """A context manager for managing the model's forward function during model exportation.

        It temporarily modifies the model's forward function to generate output sinks
        for explain results during the model graph tracing.
        """
        try:
            self._reset_model_forward()
            yield
        finally:
            self._restore_model_forward()

    def _reset_model_forward(self) -> None:
        if not self.explain_mode:
            return

        self.model.explain_fn = self.get_explain_fn()
        forward_with_explain = self._forward_explain_inst_seg

        self.original_model_forward = self.model.forward

        func_type = types.MethodType
        # Patch method
        self.model.forward = func_type(forward_with_explain, self.model)

    def _restore_model_forward(self) -> None:
        if not self.explain_mode:
            return

        if not self.original_model_forward:
            msg = "Original model forward was not saved."
            raise RuntimeError(msg)

        func_type = types.MethodType
        self.model.forward = func_type(self.original_model_forward, self.model)
        self.original_model_forward = None
