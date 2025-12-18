# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR model implementations for OTX.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from rfdetr import (
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
from otx.backend.native.models.detection.detectors import RFDETRDetector
from otx.backend.native.models.detection.rtdetr import RTDETR
from otx.config.data import TileConfig
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable

if TYPE_CHECKING:
    import torch
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from torch import Tensor

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class RFDETR(RTDETR):
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

    input_size_multiplier = 8

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
            model_name=model_name,  # type: ignore[arg-type]
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
            # No weight decay for norm layers and biases in encoder/decoder, but exclude backbone
            {
                "params": r"^(?!.*backbone)(?=.*(?:encoder|decoder|transformer))(?=.*(?:norm|bias)).*$",
                "weight_decay": 0.0,
            },
        ]

        # Create wrapper
        return RFDETRDetector(
            lwdetr_model=rfdetr_model.model,  # type: ignore[arg-type]
            criterion=criterion,
            postprocessor=postprocessor,
            optimizer_configuration=optimizer_configuration,
            input_size=model_config.resolution,
            multi_scale=self.multi_scale,
            group_detr=self.group_detr,
        )

    def forward_for_tracing(  # type: ignore[override]
        self,
        inputs: torch.Tensor,
    ) -> tuple[Tensor, Tensor, Tensor] | dict[str, Any]:
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
        return self.model.export(inputs, meta_info_list, explain_mode=self.explain_mode)

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
