# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR model implementation for Instance Segmentation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from rfdetr import (
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
)
from rfdetr.main import populate_args
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from torch.export import Dim
from torchvision import tv_tensors
from torchvision.ops import box_convert

from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.backend.native.models.detection.detectors.rfdetr import RFDETRDetector
from otx.backend.native.models.detection.utils import limit_batch_objects
from otx.backend.native.models.instance_segmentation.base import OTXInstanceSegModel
from otx.backend.native.models.utils.utils import load_checkpoint
from otx.config.data import TileConfig
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.sample import OTXPredictionBatch, OTXSampleBatch
from otx.metrics.fmeasure import MaskRLEMeanAPFMeasureCallable
from otx.types.export import OTXExportFormatType
from otx.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class RFDETRInst(OTXInstanceSegModel):
    """OTX Instance Segmentation model class for RF-DETR.

    RF-DETR (Real-time Fast DETR) is a state-of-the-art object detector from Roboflow
    that combines a DINOv2 backbone with a lightweight DETR decoder. This implementation
    adds instance segmentation support with a mask prediction head.

    This implementation uses the rfdetr Python package with RFDETRSegPreview for the core model components.

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
        RF-DETR Segmentation uses patch_size=12 with 2 windows for 432x432 input resolution.
    """

    _pretrained_weights: ClassVar[dict[str, str]] = {
        "rfdetr_seg_n": "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
        "rfdetr_seg_s": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
        "rfdetr_seg_m": "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
        "rfdetr_seg_l": "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
        "rfdetr_seg_xl": "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
        "rfdetr_seg_2xl": "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
    }

    _model_class_mapping: ClassVar[dict[str, type]] = {
        "rfdetr_seg_n": RFDETRSegNano,
        "rfdetr_seg_s": RFDETRSegSmall,
        "rfdetr_seg_m": RFDETRSegMedium,
        "rfdetr_seg_l": RFDETRSegLarge,
        "rfdetr_seg_xl": RFDETRSegXLarge,
        "rfdetr_seg_2xl": RFDETRSeg2XLarge,
    }

    input_size_multiplier = 24

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams | None = None,
        model_name: Literal[
            "rfdetr_seg_n",
            "rfdetr_seg_s",
            "rfdetr_seg_m",
            "rfdetr_seg_l",
            "rfdetr_seg_xl",
            "rfdetr_seg_2xl",
        ] = "rfdetr_seg_m",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
        max_total_objects_per_batch: int | None = None,
    ) -> None:
        self.multi_scale = multi_scale
        self.max_total_objects_per_batch = max_total_objects_per_batch
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

    def _create_model(self, num_classes: int | None = None) -> RFDETRDetector:
        """Create RF-DETR Instance Segmentation model using rfdetr package.

        Creates the RFDETRInstDetector wrapper which contains:
        - LWDETR model with segmentation head (backbone + decoder + mask head)
        - SetCriterion (loss computation including mask loss)
        - PostProcessor (inference post-processing with mask outputs)
        - Optimizer configuration (from rfdetr args)

        Args:
            num_classes: Number of classes for detection.

        Returns:
            RFDETRInstDetector model instance.
        """
        num_classes = num_classes if num_classes is not None else self.num_classes

        # Create RF-DETR Segmentation model with segmentation_head=True
        model_class = self._model_class_mapping[self.model_name]
        detector = model_class(pretrain_weights=None, gradient_checkpointing=True)
        # Get the actual LWDETR model with segmentation head
        lwdetr_model = detector.model.model
        load_checkpoint(
            lwdetr_model,  # pyrefly: ignore[bad-argument-type]
            self._pretrained_weights[self.model_name],
            map_location="cpu",
        )
        # Reinitialize detection head for our num_classes
        detector.model.reinitialize_detection_head(num_classes)
        # Update args for criterion building
        detector.model.args.num_classes = num_classes

        # Build criterion and postprocessor with correct args (including segmentation_head=True)
        model_cfg = detector.get_model_config().model_dump()
        train_cfg = detector.get_train_config(dataset_dir="").model_dump()
        model_cfg.pop("num_classes")
        if "class_names" in model_cfg:
            model_cfg.pop("class_names")

        for k in train_cfg:
            if k in model_cfg:
                model_cfg.pop(k)

        all_kwargs = {**model_cfg, **train_cfg, "num_classes": num_classes}
        rfdetr_args = populate_args(**all_kwargs)

        criterion, postprocessor = build_criterion_and_postprocessors(rfdetr_args)
        self.criterion = criterion
        self.postprocessor = postprocessor
        # Override criterion.num_classes to match OTX labels
        criterion.num_classes = num_classes

        # Store rfdetr_args for optimizer configuration
        self.rfdetr_args = rfdetr_args

        return RFDETRDetector(
            lwdetr_model=lwdetr_model,
            criterion=criterion,
            postprocessor=postprocessor,
            rfdetr_args=self.rfdetr_args,  # pyrefly: ignore[bad-argument-type]
            input_size=self.data_input_params.input_size[0],
            multi_scale=self.multi_scale,
        )

    def _customize_inputs(self, entity: OTXSampleBatch) -> dict[str, Any]:
        """Convert OTX batch format to RF-DETR format for instance segmentation.

        Args:
            entity: OTX data batch with images, bboxes, labels, and polygons/masks.

        Returns:
            Dictionary with images and targets formatted for RF-DETR.
        """
        # Prepare inputs - RF-DETR wrapper handles NestedTensor creation
        images = entity.images
        # Convert targets with masks
        targets = []

        # Only build targets when all required annotations are present
        if all(getattr(entity, attr) is not None for attr in ("bboxes", "labels", "masks")):
            for i, (bb, ll, mm) in enumerate(zip(entity.bboxes, entity.labels, entity.masks)):  # type: ignore[arg-type]
                # Determine image size (prefer bounding-box canvas_size, fall back to imgs_info or image tensor)
                if len(bb) > 0 and getattr(bb, "canvas_size", None) is not None:
                    h, w = bb.canvas_size

                    # Convert XYXY to CXCYWH - extract .data to get plain tensor
                    boxes_cxcywh = box_convert(bb.data, in_fmt="xyxy", out_fmt="cxcywh")
                    device = boxes_cxcywh.device
                    boxes_normalized = boxes_cxcywh / torch.tensor([w, h, w, h], device=device, dtype=torch.float32)
                else:
                    # Fallbacks for image size and device
                    if getattr(entity, "imgs_info", None) is not None:
                        h, w = entity.imgs_info[i].img_shape  # type: ignore[union-attr, arg-type, index]
                    else:
                        img = entity.images[i]
                        if isinstance(img, torch.Tensor):
                            _, h, w = img.shape
                        else:
                            h, w = 0, 0

                    device = getattr(entity.images, "device", torch.device("cpu"))
                    boxes_normalized = torch.zeros((0, 4), device=device, dtype=torch.float32)

                target_dict = {
                    "boxes": boxes_normalized,
                    "labels": ll,
                    "masks": mm,
                    "size": torch.tensor([h, w], device=device),
                    "orig_size": torch.tensor([h, w], device=device),
                }

                targets.append(target_dict)

        # Apply batch-level object limiting if configured (only during training)
        if self.training and self.max_total_objects_per_batch is not None:
            targets = limit_batch_objects(targets, self.max_total_objects_per_batch)

        return {
            "images": images,
            "targets": targets,
        }

    def _customize_outputs(  # pyrefly: ignore[bad-override]
        self,
        outputs: dict | tuple,  # type: ignore[override]
        inputs: OTXSampleBatch,
    ) -> OTXPredictionBatch | OTXBatchLossEntity:
        """Convert model outputs to OTX format with masks.

        Args:
            outputs: Model outputs (loss dict during training, predictions during inference).
            inputs: Original OTX data batch.

        Returns:
            OTXPredictionBatch with masks during inference, OTXBatchLossEntity during training.
        """
        if self.training:
            # Training: outputs is loss dictionary from RFDETRInstDetector
            if not isinstance(outputs, dict):
                msg = f"Expected dict during training, got {type(outputs)}"
                raise TypeError(msg)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    losses[k] = v
                elif isinstance(v, list):
                    losses[k] = sum(
                        (_loss.mean() for _loss in v),
                        torch.tensor(0.0),
                    )  # pyrefly: ignore[unsupported-operation]
            return losses

        # Inference: get predictions from the model wrapper
        # The RFDETRInstDetector.postprocess returns (scores, boxes, labels, masks)
        original_sizes = [img_info.img_shape for img_info in inputs.imgs_info]  # type: ignore[union-attr]

        # Model forward returns outputs dict, postprocess it
        scores_list, boxes_list, labels_list, masks_list = self.model.postprocess(  # pyrefly: ignore[not-callable]
            outputs,
            original_sizes,
        )

        # Convert masks to proper format for OTX
        formatted_masks = []
        for masks, img_info in zip(masks_list, inputs.imgs_info):  # type: ignore[union-attr, arg-type]
            if masks is not None and len(masks) > 0:
                # Masks are already in [N, H, W] boolean format from postprocessor
                formatted_masks.append(
                    tv_tensors.Mask(masks, dtype=torch.uint8)  # type: ignore[call-overload]
                )
            else:
                # Empty masks
                formatted_masks.append(
                    tv_tensors.Mask(
                        torch.zeros((0, img_info.img_shape[0], img_info.img_shape[1]), dtype=torch.bool)  # type: ignore[union-attr,call-overload]
                    )
                )

        if self.explain_mode:
            msg = "Explain mode is not supported for RF-DETR model."
            raise ValueError(msg)

        return OTXPredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores_list,
            bboxes=boxes_list,
            labels=labels_list,
            masks=formatted_masks,
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
                "output_names": ["bboxes", "labels", "scores", "masks"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
                "opset_version": 18,
            },
            output_names=["bboxes", "labels", "scores", "masks"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RF-DETR."""
        return {"model_type": "transformer"}

    @property
    def _default_preprocessing_params(self) -> dict[str, DataInputParams]:  # type: ignore[override]
        """Default preprocessing parameters for RF-DETR segmentation models."""
        imagenet_mean = (123.675, 116.28, 103.53)
        imagenet_std = (58.395, 57.12, 57.375)

        return {
            "rfdetr_seg_n": DataInputParams(
                input_size=(312, 312),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_s": DataInputParams(
                input_size=(384, 384),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_m": DataInputParams(
                input_size=(432, 432),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_l": DataInputParams(
                input_size=(504, 504),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_xl": DataInputParams(
                input_size=(624, 624),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
            "rfdetr_seg_2xl": DataInputParams(
                input_size=(768, 768),
                mean=imagenet_mean,
                std=imagenet_std,
            ),
        }
