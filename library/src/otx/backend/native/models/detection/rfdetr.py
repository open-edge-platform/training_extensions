# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR model implementations for OTX.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat
from rfdetr.config import (
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSmallConfig,
)
from rfdetr.main import Model, download_pretrain_weights
from rfdetr.models.lwdetr import build_criterion_and_postprocessors

from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.detection.base import OTXDetectionModel
from otx.backend.native.models.detection.detectors import RFDETRDetector
from otx.config.data import TileConfig
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class RFDETR(OTXDetectionModel):
    """OTX Detection model class for RF-DETR.

    RF-DETR (Real-time Fast DETR) is a state-of-the-art object detector from Roboflow
    that combines a DINOv2 backbone with a lightweight DETR decoder for real-time
    object detection.

    This implementation uses the rfdetr Python package for the core model components.

    Args:
        label_info: Information about the labels.
        data_input_params: Parameters for image data preprocessing.
        model_name: Name of the model variant to use.
        optimizer: Callable for the optimizer.
        scheduler: Callable for the learning rate scheduler.
        metric: Callable for the metric.
        multi_scale: Whether to use multi-scale training.
        group_detr: Number of groups for Group DETR training (speeds up training).
        torch_compile: Whether to use torch compile.
        tile_config: Configuration for tiling.

    Note:
        RF-DETR uses different patch sizes for different model variants:
        - nano, small, medium: patch_size=16
        - base, large: patch_size=14
        Input sizes must be compatible with patch_size * num_windows.
    """

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | None = None,
        model_name: Literal[
            "rfdetr_nano",
            "rfdetr_small",
            "rfdetr_base",
            "rfdetr_medium",
            "rfdetr_large",
        ] = "rfdetr_base",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        group_detr: int = 13,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.multi_scale = multi_scale
        self.group_detr = group_detr
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

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:  # type: ignore[override]
        """Default preprocessing parameters for RF-DETR models.

        Uses 0-255 range normalization values (unscaled ImageNet stats).
        """
        imagenet_mean = (123.675, 116.28, 103.53)
        imagenet_std = (58.395, 57.12, 57.375)

        return {
            "rfdetr_nano": DataInputParams(
                input_size=(384, 384),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_small": DataInputParams(
                input_size=(512, 512),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_base": DataInputParams(
                input_size=(560, 560),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_medium": DataInputParams(
                input_size=(576, 576),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_large": DataInputParams(
                input_size=(560, 560),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
        }

    def _create_model(self, num_classes: int | None = None) -> RFDETRDetector:
        """Create RF-DETR model using rfdetr package.

        Args:
            num_classes: Number of classes for detection.

        Returns:
            RFDETRDetector instance.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes

        # Get the appropriate config class
        config_mapping: dict[str, type] = {
            "rfdetr_nano": RFDETRNanoConfig,
            "rfdetr_small": RFDETRSmallConfig,
            "rfdetr_base": RFDETRBaseConfig,
            "rfdetr_medium": RFDETRMediumConfig,
            "rfdetr_large": RFDETRLargeConfig,
        }

        config_class = config_mapping[self.model_name]

        # Create config with our num_classes
        # Note: rfdetr uses num_classes + 1 internally for no-object class
        model_config = config_class(
            num_classes=num_classes,
            group_detr=self.group_detr,
        )

        # Download pretrained weights if needed
        if model_config.pretrain_weights:
            download_pretrain_weights(model_config.pretrain_weights)

        # Create the Model instance which builds LWDETR and loads weights
        rfdetr_model = Model(**model_config.model_dump())

        # Build criterion and postprocessor
        criterion, postprocessor = build_criterion_and_postprocessors(rfdetr_model.args)

        # Define optimizer configuration for different parts of the model
        # Following rfdetr's training configuration:
        # - lr_encoder: 1.5e-4 for backbone
        # - lr: 1e-4 for head (default)
        # - lr_vit_layer_decay: 0.8, lr_component_decay: 0.7
        optimizer_configuration: list[dict[str, Any]] = [
            # Lower LR for backbone, no weight decay for norm layers
            {"params": r"^(?=.*backbone)(?=.*(?:norm|ln)).*$", "weight_decay": 0.0, "lr": 0.00015},
            {"params": r"^(?=.*backbone)(?!.*(?:norm|ln)).*$", "lr": 0.00015},
            # No weight decay for norm layers and biases in encoder/decoder
            {"params": r"^(?=.*(?:encoder|decoder|transformer))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        # Create wrapper
        wrapper = RFDETRDetector(
            lwdetr_model=rfdetr_model.model,  # type: ignore[arg-type]
            criterion=criterion,
            postprocessor=postprocessor,
            optimizer_configuration=optimizer_configuration,
            input_size=model_config.resolution,
            multi_scale=self.multi_scale,
            group_detr=self.group_detr,
        )

        return wrapper

    def _customize_inputs(
        self,
        entity: OTXDataBatch,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        """Customize inputs for RF-DETR model.

        Converts OTX batch format to the format expected by RF-DETR.

        Args:
            entity: OTX data batch.
            pad_size_divisor: Divisor for padding.
            pad_value: Value for padding.

        Returns:
            Dictionary with 'images' and 'targets'.
        """
        targets: list[dict[str, Tensor]] = []

        # Get device from images
        images = entity.images
        device = images.device if isinstance(images, Tensor) else images[0].device

        # Prepare bboxes for the model
        if entity.bboxes is not None and entity.labels is not None:
            for bb, ll in zip(entity.bboxes, entity.labels):
                if len(bb) > 0:
                    # Convert to cxcywh if needed
                    if bb.format == BoundingBoxFormat.XYXY:
                        converted_bboxes = box_convert(bb, in_fmt="xyxy", out_fmt="cxcywh")
                    else:
                        converted_bboxes = bb

                    # Normalize bboxes to [0, 1] range
                    canvas_size = bb.canvas_size  # (H, W)
                    scale = torch.tensor(
                        [canvas_size[1], canvas_size[0], canvas_size[1], canvas_size[0]],
                        device=converted_bboxes.device,
                        dtype=converted_bboxes.dtype,
                    )
                    normalized_bboxes = converted_bboxes / scale

                    targets.append({
                        "boxes": normalized_bboxes,
                        "labels": ll,
                    })
                else:
                    # Empty annotations
                    targets.append({
                        "boxes": torch.zeros((0, 4), device=device),
                        "labels": torch.zeros((0,), dtype=torch.long, device=device),
                    })

        if self.explain_mode:
            return {"entity": entity}

        return {
            "images": images,
            "targets": targets if targets else None,
        }

    def _customize_outputs(  # type: ignore[override]
        self,
        outputs: dict[str, Tensor],
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
        """Customize outputs from RF-DETR model.

        Converts model outputs to OTX format.

        Args:
            outputs: Model outputs (losses during training, predictions during inference).
            inputs: Original OTX data batch.

        Returns:
            OTXBatchLossEntity during training, OTXPredBatch during inference.
        """
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(f"Expected dict outputs during training, got {type(outputs)}")

            losses = OTXBatchLossEntity()

            # Get the weight dict from criterion
            model_wrapper: RFDETRDetector = self.model  # type: ignore[assignment]
            weight_dict: dict[str, float] = model_wrapper.criterion.weight_dict  # type: ignore[assignment]

            # Sum auxiliary losses and apply weight dict
            for k, v in outputs.items():
                if k in weight_dict:
                    losses[k] = v * weight_dict[k]
                elif isinstance(v, Tensor) and v.numel() == 1:
                    # Include unweighted losses like class_error
                    losses[k] = v

            return losses

        # Inference mode
        if inputs.imgs_info is None:
            msg = "imgs_info is required for inference"
            raise ValueError(msg)

        original_sizes: list[tuple[int, int]] = []
        for img_info in inputs.imgs_info:
            if img_info is not None and img_info.ori_shape is not None:
                original_sizes.append((img_info.ori_shape[0], img_info.ori_shape[1]))
            else:
                # Fallback to image shape
                images = inputs.images
                if isinstance(images, list):
                    original_sizes.append((int(images[0].shape[1]), int(images[0].shape[2])))
                else:
                    original_sizes.append((int(images.shape[2]), int(images.shape[3])))

        model_wrapper: RFDETRDetector = self.model  # type: ignore[assignment]
        scores, bboxes, labels = model_wrapper.postprocess(outputs, original_sizes)

        if self.explain_mode:
            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            return OTXPredBatch(
                batch_size=len(scores),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                feature_vector=[fv.unsqueeze(0) for fv in outputs["feature_vector"]],
                saliency_map=[sm.to(torch.float32) for sm in outputs["saliency_map"]],
            )

        return OTXPredBatch(
            batch_size=len(scores),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def forward_for_tracing(
        self,
        inputs: torch.Tensor,
    ) -> tuple[Tensor, Tensor, Tensor] | dict[str, Any]:  # type: ignore[override]
        """Forward function for model export/tracing.

        Args:
            inputs: Input images tensor.

        Returns:
            If explain_mode is False: Tuple of (boxes, labels, scores) tensors.
            If explain_mode is True: Dict with boxes, labels, scores, feature_vector, saliency_map.
        """
        shape = (int(inputs.shape[2]), int(inputs.shape[3]))
        meta_info = {
            "pad_shape": shape,
            "batch_input_shape": shape,
            "img_shape": shape,
            "scale_factor": (1.0, 1.0),
        }
        meta_info_list = [meta_info] * len(inputs)
        model_wrapper: RFDETRDetector = self.model  # type: ignore[assignment]
        return model_wrapper.export(inputs, meta_info_list, explain_mode=self.explain_mode)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:  # type: ignore[override]
        """Configure optimizer and learning rate schedulers.

        Uses parameter groups with different learning rates for backbone vs head,
        following RF-DETR's training configuration.

        Returns:
            Tuple of optimizer list and scheduler config list.
        """
        model_wrapper: RFDETRDetector = self.model  # type: ignore[assignment]
        param_groups = self._get_optim_params(model_wrapper.optimizer_configuration, model_wrapper)
        optimizer = self.optimizer_callable(param_groups)
        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs: list[dict[str, Any]] = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config: dict[str, Any] = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    @staticmethod
    def _get_optim_params(
        cfg: list[dict[str, Any]] | None,
        model: nn.Module,
    ) -> list[dict[str, Any]]:
        """Get optimizer parameters with different learning rates.

        Args:
            cfg: Configuration for parameter groups with regex patterns.
            model: The model to get parameters from.

        Returns:
            List of parameter group dictionaries.
        """
        if cfg is None:
            return [{"params": model.parameters()}]

        cfg = copy.deepcopy(cfg)

        param_groups: list[dict[str, Any]] = []
        visited: set[str] = set()

        for pg in cfg:
            if "params" not in pg:
                msg = f"The 'params' key should be included in the configuration, but got {pg.keys()}"
                raise ValueError(msg)

            pattern = pg["params"]
            matching_params: dict[str, nn.Parameter] = {}

            for name, param in model.named_parameters():
                if param.requires_grad and re.search(pattern, name):
                    matching_params[name] = param

            if matching_params:
                pg_copy = {k: v for k, v in pg.items() if k != "params"}
                pg_copy["params"] = list(matching_params.values())
                param_groups.append(pg_copy)
                visited.update(matching_params.keys())

        # Add remaining parameters with default settings
        remaining_params = {
            name: param
            for name, param in model.named_parameters()
            if param.requires_grad and name not in visited
        }

        if remaining_params:
            param_groups.append({"params": list(remaining_params.values())})

        return param_groups

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object for model export."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=True,  # RF-DETR exports better via ONNX
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {
                    "images": {0: "batch"},
                    "bboxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                },
                "opset_version": 17,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RF-DETR."""
        return {"model_type": "transformer"}
