# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR Mixin for shared logic between Detection and Instance Segmentation models.

This mixin encapsulates common functionality used by both RFDETR (detection) and
RFDETRInst (instance segmentation) models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch
from rfdetr.main import populate_args
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.util.get_param_dicts import get_param_dict
from torchvision import tv_tensors
from torchvision.ops import box_convert

from getitune.backend.lightning.models.detection.detectors.rfdetr import RFDETRDetector
from getitune.backend.lightning.models.detection.utils import limit_batch_objects
from getitune.backend.lightning.models.utils.utils import load_checkpoint
from getitune.data.entity.base import BatchLoss
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from pathlib import Path


class RFDETRMixin:
    """Mixin class providing shared RF-DETR functionality for detection and instance segmentation."""

    _pretrained_weights: ClassVar[dict[str, str]]
    _model_class_mapping: ClassVar[dict[str, type]]

    def _build_rfdetr_model(
        self,
        num_classes: int,
        *,
        gradient_checkpointing: bool = False,
    ) -> RFDETRDetector:
        """Build the core RF-DETR model.

        Handles the full pipeline shared by both detection and instance segmentation:
        1. Instantiate the rfdetr model class
        2. Load pretrained checkpoint
        3. Reinitialize detection head for ``num_classes``
        4. Build criterion and postprocessor
        5. Wrap in ``RFDETRDetector``

        Args:
            num_classes: Number of target classes.
            gradient_checkpointing: Whether to enable gradient checkpointing.

        Returns:
            Configured ``RFDETRDetector`` instance.
        """
        model_class = self._model_class_mapping[self.model_name]  # type: ignore[attr-defined]

        detector = model_class(
            pretrain_weights=None,
            gradient_checkpointing=gradient_checkpointing,
        )
        lwdetr_model = detector.model.model

        load_checkpoint(
            lwdetr_model,  # pyrefly: ignore[bad-argument-type]
            self._pretrained_weights[self.model_name],  # type: ignore[attr-defined]
            map_location="cpu",
        )

        # Reinitialize detection head for our num_classes
        detector.model.reinitialize_detection_head(num_classes)
        detector.model.args.num_classes = num_classes

        # Build criterion and postprocessor
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
        criterion.num_classes = num_classes

        # Store rfdetr_args for optimizer configuration
        self.rfdetr_args = rfdetr_args  # type: ignore[attr-defined]

        return RFDETRDetector(
            lwdetr_model=lwdetr_model,  # pyrefly: ignore[bad-argument-type]
            criterion=criterion,
            postprocessor=postprocessor,
            rfdetr_args=self.rfdetr_args,  # pyrefly: ignore[bad-argument-type]
            input_size=self.data_input_params.input_size[0],  # type: ignore[attr-defined]
            multi_scale=self.multi_scale,  # type: ignore[attr-defined]
        )

    def _customize_inputs(  # pyrefly: ignore[bad-override]
        self,
        entity: SampleBatch,
    ) -> dict[str, Any]:
        """Convert getitune batch format to RF-DETR input format.

        Handles both detection (boxes + labels) and instance segmentation
        (boxes + labels + masks) depending on what the entity contains.

        Args:
            entity: getitune data batch.

        Returns:
            Dict with 'images' and 'targets' for the model.
        """
        targets: list[dict[str, Any]] = []
        has_masks = entity.masks is not None

        required_attrs = ("bboxes", "labels", "masks") if has_masks else ("bboxes", "labels")
        if all(getattr(entity, attr) is not None for attr in required_attrs):
            iterables: tuple = (
                (entity.bboxes, entity.labels, entity.masks) if has_masks else (entity.bboxes, entity.labels)  # type: ignore[assignment]
            )
            for _i, items in enumerate(zip(*iterables)):
                bb, ll = items[0], items[1]
                mm = items[2] if has_masks else None

                if len(bb) > 0 and getattr(bb, "canvas_size", None) is not None:
                    h, w = bb.canvas_size
                    boxes_cxcywh = box_convert(bb.data, in_fmt="xyxy", out_fmt="cxcywh")
                    device = boxes_cxcywh.device
                    scaled_bboxes = boxes_cxcywh / torch.tensor(
                        [w, h, w, h],
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    h, w = entity.images.shape[2:]  # pyrefly: ignore[missing-attribute]
                    device = getattr(entity.images, "device", torch.device("cpu"))
                    scaled_bboxes = torch.zeros((0, 4), device=device, dtype=torch.float32)

                target: dict[str, Any] = {
                    "boxes": scaled_bboxes,
                    "labels": ll,
                    "size": torch.tensor([h, w], device=device),
                    "orig_size": torch.tensor([h, w], device=device),
                }
                if mm is not None:
                    target["masks"] = mm

                targets.append(target)

        # Apply batch-level object limiting if configured (only during training)
        if self.training and self.max_total_objects_per_batch is not None:  # type: ignore[attr-defined]
            targets = limit_batch_objects(targets, self.max_total_objects_per_batch)  # type: ignore[attr-defined]

        if self.explain_mode:  # type: ignore[attr-defined]
            return {"entity": entity}

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(  # pyrefly: ignore[bad-override]
        self,
        outputs: tuple[torch.Tensor, ...] | dict[str, Any],
        inputs: SampleBatch,
    ) -> PredictionBatch | BatchLoss:
        """Convert model outputs to getitune format.

        Handles both detection and instance segmentation outputs.
        When the postprocessor returns masks (4th element is not ``None``),
        they are formatted and included in the prediction batch.

        Args:
            outputs: Model outputs (loss dict during training, predictions during inference).
            inputs: Original getitune data batch.

        Returns:
            ``PredictionBatch`` during inference, ``BatchLoss`` during training.
        """
        if self.training:  # type: ignore[attr-defined]
            if not isinstance(outputs, dict):
                msg = f"Expected dict during training, got {type(outputs)}"
                raise TypeError(msg)

            losses = BatchLoss()
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    losses[k] = v
                elif isinstance(v, list):
                    losses[k] = sum(
                        (_loss.mean() for _loss in v),
                        torch.tensor(0.0),
                    )  # pyrefly: ignore[unsupported-operation]
                else:
                    msg = f"Loss output should be list or torch.Tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        image_shapes = [img_info.ori_shape for img_info in inputs.imgs_info]  # type: ignore[union-attr]
        scores_list, boxes_list, labels_list, masks_list = self.model.postprocess(  # type: ignore[attr-defined]  # pyrefly: ignore[not-callable]
            outputs,
            image_shapes,
        )

        prediction_kwargs: dict[str, Any] = {
            "images": inputs.images,
            "imgs_info": inputs.imgs_info,
            "scores": scores_list,
            "bboxes": boxes_list,
            "labels": labels_list,
        }

        # Include masks when the postprocessor produces them (instance segmentation)
        if masks_list is not None and any(m is not None for m in masks_list):
            formatted_masks = []
            for masks, img_info in zip(masks_list, inputs.imgs_info):  # type: ignore[union-attr, arg-type]
                if masks is not None and len(masks) > 0:
                    formatted_masks.append(
                        tv_tensors.Mask(masks, dtype=torch.uint8),  # type: ignore[call-overload]
                    )
                else:
                    formatted_masks.append(
                        tv_tensors.Mask(
                            torch.zeros(
                                (0, img_info.img_shape[0], img_info.img_shape[1]),  # type: ignore[union-attr]
                                dtype=torch.bool,
                            ),  # type: ignore[call-overload]
                        ),
                    )
            prediction_kwargs["masks"] = formatted_masks

        if self.explain_mode:  # type: ignore[attr-defined]
            msg = "Explain mode is not supported for RF-DETR model."
            raise ValueError(msg)

        return PredictionBatch(**prediction_kwargs)

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure optimizer and learning-rate schedulers.

        Uses rfdetr's ``get_param_dict`` to create proper parameter groups with
        correct lr and weight_decay settings from rfdetr_args.

        Returns:
            Two lists: optimizer list and lr scheduler config list.
        """
        # Extract default lr and weight_decay from optimizer callable
        dummy = torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        dummy_param_groups = [{"params": [dummy]}]
        default_lr = self.optimizer_callable(dummy_param_groups).param_groups[0]["lr"]  # type: ignore[attr-defined]
        default_weight_decay = self.optimizer_callable(dummy_param_groups).param_groups[0]["weight_decay"]  # type: ignore[attr-defined]

        # Get parameter groups from rfdetr with correct args
        self.rfdetr_args.lr = default_lr  # type: ignore[attr-defined]
        self.rfdetr_args.weight_decay = default_weight_decay  # type: ignore[attr-defined]
        param_groups = get_param_dict(self.rfdetr_args, self.model.lwdetr)  # type: ignore[attr-defined]  # pyrefly: ignore[bad-argument-type]

        # Create optimizer and schedulers
        optimizer = self.optimizer_callable(param_groups)  # type: ignore[attr-defined]
        schedulers = self.scheduler_callable(optimizer)  # type: ignore[attr-defined]

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
        return self.model.export(inputs)  # type: ignore[attr-defined]  # pyrefly: ignore[not-callable]

    @staticmethod
    def _restore_forward_methods(module: torch.nn.Module) -> None:
        """Undo ``module.export()`` forward monkey-patching on all sub-modules."""
        for m in module.modules():
            if getattr(m, "_export", False) and hasattr(m, "_forward_origin"):
                m.forward = m._forward_origin  # noqa: SLF001  # pyrefly: ignore[bad-assignment]
                m._export = False  # noqa: SLF001  # pyrefly: ignore[bad-argument-type]

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: ExportFormat,
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export the model to the requested format."""
        if self.explain_mode:  # pyrefly: ignore[missing-attribute]
            msg = "Explain mode is not supported for RF-DETR model."
            raise ValueError(msg)
        lwdetr = self.model.lwdetr  # pyrefly: ignore[missing-attribute]
        lwdetr.export()  # pyrefly: ignore[missing-attribute]
        try:
            return super().export(output_dir, base_name, export_format, precision)  # type: ignore[misc]
        except Exception:
            self._restore_forward_methods(lwdetr)
            raise
        finally:
            self._restore_forward_methods(lwdetr)

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RF-DETR."""
        return {"model_type": "transformer"}
