# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR detector wrapper for Geti Tune integration.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.util.misc import nested_tensor_from_tensor_list
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxes

<<<<<<<< HEAD:library/src/getitune/backend/lightning/models/detection/detectors/rfdetr.py
from getitune.backend.lightning.models.modules.base_module import BaseModule
========
from getitune.backend.native.models.modules.base_module import BaseModule
>>>>>>>> develop:library/src/getitune/backend/native/models/detection/detectors/rfdetr.py

if TYPE_CHECKING:
    from jsonargparse import Namespace


class RFDETRDetector(BaseModule):
    """Wrapper around RF-DETR's LWDETR model for Geti Tune integration.

    This wrapper handles the interface between Geti Tune's training pipeline and
    the rfdetr package's LWDETR model and SetCriterion.

    Args:
        lwdetr_model: The LWDETR model instance from rfdetr package.
        criterion: The SetCriterion loss function from rfdetr package.
        postprocessor: The PostProcess module from rfdetr package.
        input_size: The input resolution of the model.
        multi_scale: Whether to enable multi-scale training.
    """

    def __init__(
        self,
        lwdetr_model: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        rfdetr_args: Namespace,
        input_size: int = 560,
        multi_scale: bool = False,
    ) -> None:
        super().__init__()
        self.lwdetr = lwdetr_model
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.input_size = input_size
        self.rng = np.random.default_rng(42)

        # Store scales for multi-scale training
        self.scales = (
            compute_multi_scale_scales(
                rfdetr_args.resolution, rfdetr_args.expanded_scales, rfdetr_args.patch_size, rfdetr_args.num_windows
            )
            if multi_scale
            else []
        )

    def forward(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Args:
            images: NestedTensor with images and masks from _customize_inputs.
            targets: List of target dictionaries with 'boxes' and 'labels'.

        Returns:
            During training: Loss dictionary.
            During inference: Predictions dictionary with 'pred_logits' and 'pred_boxes'.
        """
        # Multi-scale training - need to handle NestedTensor
        if self.training and self.scales:
            sz = int(self.rng.choice(self.scales))
            images = nn.functional.interpolate(images, size=[sz, sz], mode="bilinear", align_corners=False)

        # Convert to list of tensors if needed
        if isinstance(images, Tensor) and images.dim() == 4:
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = list(images)

        samples = nested_tensor_from_tensor_list(image_list)

        # Forward through model - images is already a NestedTensor
        outputs = self.lwdetr(samples)

        if self.training:
            self.criterion.train()
            if targets is None:
                msg = "targets should not be None"
                raise ValueError(msg)

            loss_dict = self.criterion(outputs, targets)
            weight_dict: dict[str, float] = self.criterion.weight_dict  # pyrefly: ignore[bad-assignment]
            return {k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}

        return outputs

    def postprocess(
        self,
        outputs: dict[str, Tensor],
        original_sizes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[BoundingBoxes], list[Tensor], list[Tensor]]:
        """Post-process model outputs to get final predictions.

        Args:
            outputs: Model outputs with 'pred_logits' and 'pred_boxes'.
            original_sizes: List of original image sizes (H, W).

        Returns:
            Tuple of (scores_list, boxes_list, labels_list).
        """
        target_sizes = torch.tensor(original_sizes, device=outputs["pred_logits"].device)
        results = self.postprocessor(outputs, target_sizes)

        scores_list: list[Tensor] = []
        boxes_list: list[BoundingBoxes] = []
        labels_list: list[Tensor] = []
        masks_list: list[Tensor] = []

        for result, orig_size in zip(results, original_sizes):
            scores_list.append(result["scores"])
            boxes_list.append(
                BoundingBoxes(  # type: ignore[call-overload]
                    result["boxes"],
                    format="xyxy",
                    canvas_size=orig_size,
                ),
            )
            labels_list.append(result["labels"].long())
            if "masks" in result:
                masks_list.append(torch.tensor(result["masks"].squeeze(1), dtype=torch.uint8))

        return scores_list, boxes_list, labels_list, masks_list

    def export(
        self,
        batch_inputs: Tensor,
        num_select: int = 300,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        """Export function for model tracing with mask support.

        Args:
            batch_inputs: Input images tensor.
            num_select: Number of top predictions to select.

        Returns:
            Tuple of (boxes, labels, scores, masks) tensors.
        """
        outputs = self.lwdetr(batch_inputs)
        # outputs may be dict or tuple in export mode
        if isinstance(outputs, dict):
            pred_boxes = outputs["pred_boxes"]
            pred_logits = outputs["pred_logits"]
            pred_masks = outputs.get("pred_masks")
        elif len(outputs) == 3:
            pred_boxes, pred_logits, pred_masks = outputs
        else:
            pred_boxes, pred_logits = outputs
            pred_masks = None
        # Process outputs similar to PostProcess
        scores = torch.sigmoid(pred_logits)
        scores, index = torch.topk(scores.flatten(1), num_select, dim=-1)

        num_classes = pred_logits.shape[-1]
        labels = index % num_classes
        box_index = index // num_classes
        boxes = pred_boxes.gather(
            dim=1,
            index=box_index.unsqueeze(-1).repeat(1, 1, pred_boxes.shape[-1]),
        )
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        # Handle masks
        if pred_masks is not None:
            # pred_masks shape: [B, num_queries, H, W]
            # We need to gather masks for selected indices
            masks = pred_masks.gather(
                dim=1,
                index=box_index.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, pred_masks.shape[-2], pred_masks.shape[-1]),
            )
            # Apply sigmoid to get mask probabilities
            masks = torch.sigmoid(masks)
            return boxes, labels, scores, masks

        return boxes, labels, scores
