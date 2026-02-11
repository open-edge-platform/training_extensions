# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR model implementations for OTX.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.main import populate_args
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from torch.export import Dim
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat

from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.detection.d_fine import DFine
from otx.backend.native.models.detection.detectors.rfdetr import RFDETRDetector
from otx.backend.native.models.detection.utils import limit_batch_objects
from otx.backend.native.models.utils.utils import load_checkpoint
from otx.config.data import TileConfig
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.sample import OTXPredictionBatch, OTXSampleBatch
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable
from otx.types.export import OTXExportFormatType
from otx.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class RFDETR(DFine):
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
        torch_compile: Whether to use torch compile.
        tile_config: Configuration for tiling.

    Note:
        RF-DETR uses different patch sizes for different model variants:
        - nano, small, medium: patch_size=16
        - base, large: patch_size=14
        Input sizes must be compatible with patch_size * num_windows.
    """

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "rfdetr_base": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
        "rfdetr_large": "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
        "rfdetr_nano": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
        "rfdetr_small": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
        "rfdetr_medium": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    }

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
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        max_total_objects_per_batch: int | None = 600,
        gradient_checkpointing: bool = False,
    ) -> None:
        self.multi_scale = multi_scale
        self.max_total_objects_per_batch = max_total_objects_per_batch
        self.gradient_checkpointing = gradient_checkpointing
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,  # type: ignore[arg-type]
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
            multi_scale=multi_scale,
        )

    def _create_model(self, num_classes: int | None = None) -> RFDETRDetector:  # pyrefly: ignore[bad-override]
        """Create RF-DETR model using rfdetr package.

        Creates the RFDETRDetector wrapper which contains:
        - LWDETR model (backbone + decoder)
        - SetCriterion (loss computation)
        - PostProcessor (inference post-processing)
        - Optimizer configuration (from rfdetr args)

        Args:
            num_classes: Number of classes for detection.

        Returns:
            RFDETRDetector model instance.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes

        # Create RF-DETR model and reinitialize detection head for our num_classes
        model_class_mapping = {
            "rfdetr_base": RFDETRBase,
            "rfdetr_large": RFDETRLarge,
            "rfdetr_medium": RFDETRMedium,
            "rfdetr_nano": RFDETRNano,
            "rfdetr_small": RFDETRSmall,
        }
        rfdetr = model_class_mapping[self.model_name](
            num_classes=num_classes, pretrained_weights=None, gradient_checkpointing=self.gradient_checkpointing
        )
        # Get the actual LWDETR model
        lwdetr_model = rfdetr.model.model
        load_checkpoint(
            lwdetr_model,  # pyrefly: ignore[bad-argument-type]
            self._pretrained_weights[self.model_name],
            map_location="cpu",
        )
        # Reinitialize detection head with correct num_classes (this modifies the model in-place)
        rfdetr.model.reinitialize_detection_head(num_classes)
        # Update args for criterion building
        rfdetr.model.args.num_classes = num_classes

        # Build criterion and postprocessor with correct args
        model_cfg = rfdetr.get_model_config().model_dump()
        train_cfg = rfdetr.get_train_config(dataset_dir="").model_dump()
        model_cfg.pop("num_classes")
        if "class_names" in model_cfg:
            model_cfg.pop("class_names")

        for k in train_cfg:
            if k in model_cfg:
                model_cfg.pop(k)

        all_kwargs = {**model_cfg, **train_cfg, "num_classes": num_classes}
        rfdetr_args = populate_args(**all_kwargs)

        criterion, postprocessor = build_criterion_and_postprocessors(rfdetr_args)

        # Override criterion.num_classes to match OTX labels
        criterion.num_classes = num_classes

        # Store rfdetr_args for optimizer configuration
        self.rfdetr_args = rfdetr_args

        # Create RFDETRDetector wrapper
        return RFDETRDetector(
            lwdetr_model=lwdetr_model,  # pyrefly: ignore[bad-argument-type]
            criterion=criterion,
            postprocessor=postprocessor,
            rfdetr_args=self.rfdetr_args,  # pyrefly: ignore[bad-argument-type]
            input_size=self.data_input_params.input_size[0],
            multi_scale=self.multi_scale,
        )

    def _customize_inputs(
        self,
        entity: OTXSampleBatch,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        """Customize inputs for RF-DETR with optional batch-level object limiting.

        Args:
            entity: Input batch data.
            pad_size_divisor: Divisor for padding.
            pad_value: Padding value.

        Returns:
            Dict with 'images' and 'targets' for the model.
        """
        targets: list[dict[str, Any]] = []

        # Prepare bboxes for the model
        if entity.bboxes is not None and entity.labels is not None:
            for bb, ll in zip(entity.bboxes, entity.labels):
                scaled_bboxes = bb
                if len(bb):
                    converted_bboxes = (
                        box_convert(bb, in_fmt="xyxy", out_fmt="cxcywh") if bb.format == BoundingBoxFormat.XYXY else bb
                    )
                    # Normalize the bboxes
                    scaled_bboxes = converted_bboxes / torch.tensor(bb.canvas_size[::-1]).tile(2)[None].to(
                        converted_bboxes.device,
                    )
                h, w = bb.canvas_size
                device = scaled_bboxes.device
                targets.append(
                    {
                        "boxes": scaled_bboxes,
                        "labels": ll,
                        "size": torch.tensor([h, w], device=device),
                        "orig_size": torch.tensor([h, w], device=device),
                    }
                )

        # Apply batch-level object limiting if configured (only during training)
        if self.training and self.max_total_objects_per_batch is not None:
            targets = limit_batch_objects(targets, self.max_total_objects_per_batch)

        if self.explain_mode:
            return {"entity": entity}

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(  # pyrefly: ignore[bad-override]
        self,
        outputs: tuple[torch.Tensor, ...] | dict[str, Any],  # type: ignore[override]
        inputs: OTXSampleBatch,
    ) -> OTXPredictionBatch | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v, torch.tensor(0.0))  # pyrefly: ignore[unsupported-operation]
                elif isinstance(v, torch.Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        image_shapes = [img_info.img_shape for img_info in inputs.imgs_info]  # type: ignore[union-attr]
        scores, bboxes, labels, _ = self.model.postprocess(outputs, image_shapes)  # pyrefly: ignore[not-callable]

        return OTXPredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure optimizer and learning-rate schedulers.

        Uses rfdetr's get_param_dict to create proper parameter groups with
        correct lr and weight_decay settings from rfdetr_args.

        Returns:
            Two lists: optimizer list and lr scheduler config list.
        """
        # Extract default lr and weight_decay from optimizer callable
        dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        dummy_param_groups = [{"params": [dummy]}]
        default_lr = self.optimizer_callable(dummy_param_groups).param_groups[0]["lr"]
        default_weight_decay = self.optimizer_callable(dummy_param_groups).param_groups[0]["weight_decay"]

        # Get parameter groups from rfdetr with correct args
        # Access the LWDETR model inside the wrapper
        self.rfdetr_args.lr = default_lr
        self.rfdetr_args.weight_decay = default_weight_decay
        param_groups = get_param_dict(self.rfdetr_args, self.model.lwdetr)  # pyrefly: ignore[bad-argument-type]

        # Create optimizer and schedulers
        optimizer = self.optimizer_callable(param_groups)
        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    def forward_for_tracing(self, inputs: torch.Tensor) -> dict[str, Any] | tuple[torch.Tensor, ...]:
        """Forward pass used for model tracing/export."""
        return self.model.export(inputs)  # pyrefly: ignore[not-callable]

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export the model to the requested format."""
        self.model.lwdetr.export()  # pyrefly: ignore[missing-attribute]
        if self.explain_mode:
            msg = "Explain mode is not supported for RF-DETR model."
            raise ValueError(msg)
        return super().export(output_dir, base_name, export_format, precision)

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object for model export."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
                "opset_version": 18,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RF-DETR."""
        return {"model_type": "transformer"}

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:  # type: ignore[override]
        """Default preprocessing parameters for RF-DETR models."""
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
