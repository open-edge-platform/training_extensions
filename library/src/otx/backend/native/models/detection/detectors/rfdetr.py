# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""RF-DETR detector wrapper for OTX integration.

RF-DETR is a state-of-the-art real-time object detector from Roboflow based on
DINOv2 backbone with a lightweight DETR decoder.
Original implementation: https://github.com/roboflow/rf-detr
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
from rfdetr.util.misc import nested_tensor_from_tensor_list
from torch import Tensor, nn
from torchvision.tv_tensors import BoundingBoxes

from otx.backend.native.models.modules.base_module import BaseModule

if TYPE_CHECKING:
    from otx.types.explain import FeatureMapType


class RFDETRDetector(BaseModule):
    """Wrapper around RF-DETR's LWDETR model for OTX integration.

    This wrapper handles the interface between OTX's training pipeline and
    the rfdetr package's LWDETR model and SetCriterion.

    Args:
        lwdetr_model: The LWDETR model instance from rfdetr package.
        criterion: The SetCriterion loss function from rfdetr package.
        postprocessor: The PostProcess module from rfdetr package.
        optimizer_configuration: Configuration for optimizer parameter groups.
        input_size: The input resolution of the model.
        multi_scale: Whether to enable multi-scale training.
        group_detr: Number of groups for Group DETR training.
    """

    def __init__(
        self,
        lwdetr_model: nn.Module,
        criterion: nn.Module,
        postprocessor: nn.Module,
        optimizer_configuration: list[dict[str, Any]] | None = None,
        input_size: int = 560,
        multi_scale: bool = False,
        group_detr: int = 13,
    ) -> None:
        super().__init__()
        self.lwdetr = lwdetr_model
        self.criterion = criterion
        self.postprocessor = postprocessor
        self.optimizer_configuration = optimizer_configuration
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.group_detr = group_detr

        # Explainability functions (set by high-level OTX model)
        self.feature_vector_fn: Callable[[FeatureMapType], Tensor] | None = None
        self.explain_fn: Callable[[tuple[Tensor, ...]], Tensor] | None = None
        self.rng = np.random.default_rng(42)

        # Store scales for multi-scale training
        self.scales: list[int] = []
        if multi_scale:
            self.scales = self._generate_scales(input_size)

    def _generate_scales(self, input_size: int, base_size_repeat: int = 3) -> list[int]:
        """Generate scales for multi-scale training."""
        scale_repeat = (input_size - int(input_size * 0.75 / 32) * 32) // 32
        scales = [int(input_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
        scales += [input_size] * base_size_repeat
        scales += [int(input_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
        return scales

    def forward(
        self,
        images: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass of the model.

        Args:
            images: Input images tensor of shape [B, C, H, W].
            targets: List of target dictionaries with 'boxes' and 'labels'.

        Returns:
            During training: Loss dictionary.
            During inference: Predictions dictionary with 'pred_logits' and 'pred_boxes'.
        """
        # Multi-scale training
        if self.training and self.multi_scale and self.scales:
            sz = int(self.rng.choice(self.scales))
            images = nn.functional.interpolate(images, size=[sz, sz], mode="bilinear", align_corners=False)

        # Convert to list of tensors if needed
        if isinstance(images, Tensor) and images.dim() == 4:
            image_list = [images[i] for i in range(images.shape[0])]
        else:
            image_list = list(images)

        samples = nested_tensor_from_tensor_list(image_list)

        # Forward through model
        outputs = self.lwdetr(samples, targets)

        if self.training and targets is not None:
            # Compute losses during training
            return self.criterion(outputs, targets)

        return outputs

    def postprocess(
        self,
        outputs: dict[str, Tensor],
        original_sizes: list[tuple[int, int]],
    ) -> tuple[list[Tensor], list[BoundingBoxes], list[Tensor]]:
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

        return scores_list, boxes_list, labels_list

    def export(
        self,
        batch_inputs: Tensor,
        batch_img_metas: list[dict],
        explain_mode: bool = False,
        num_select: int = 300,
    ) -> tuple[Tensor, Tensor, Tensor] | dict[str, Any]:
        """Export function for model tracing.

        Args:
            batch_inputs: Input images tensor.
            batch_img_metas: List of image meta information.
            explain_mode: Whether to include explainability outputs.
            num_select: Number of top predictions to select.

        Returns:
            If explain_mode is False: Tuple of (boxes, labels, scores) tensors.
            If explain_mode is True: Dict with boxes, labels, scores, feature_vector, saliency_map.
        """
        # Enable export mode on LWDETR
        self.lwdetr.export()  # type: ignore[operator]
        outputs = self.lwdetr(batch_inputs)
        # outputs is (pred_boxes, pred_logits, pred_masks) in export mode
        pred_boxes, pred_logits, _ = outputs

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

        if explain_mode:
            # Get backbone features for explainability
            backbone_feats = self._get_backbone_features(batch_inputs)

            # Generate feature vector
            feature_vector: list[Tensor] = []
            if self.feature_vector_fn is not None:
                feature_vector = [self.feature_vector_fn(backbone_feats)]

            # Generate saliency maps
            saliency_map: list[Tensor] = []
            if self.explain_fn is not None:
                # Reshape logits similar to DETR for saliency computation
                raw_logits = self._reshape_logits_for_explain(backbone_feats, pred_logits)
                saliency_map = [self.explain_fn(raw_logits)]

            return {
                "bboxes": boxes,
                "labels": labels,
                "scores": scores,
                "feature_vector": feature_vector,
                "saliency_map": saliency_map,
            }

        return boxes, labels, scores

    def _get_backbone_features(self, batch_inputs: Tensor) -> tuple[Tensor, ...]:
        """Extract backbone features for explainability.

        Args:
            batch_inputs: Input images tensor.

        Returns:
            Tuple of backbone feature tensors.
        """
        # Access the backbone through LWDETR's encoder
        # RF-DETR uses DINOv2 backbone
        encoder = self.lwdetr.encoder  # type: ignore[attr-defined]

        # Get features from the backbone
        # DINOv2 produces a single feature map, but we wrap it in tuple for consistency
        with torch.no_grad():
            features = encoder.backbone(batch_inputs)  # type: ignore[attr-defined]
            if isinstance(features, Tensor):
                return (features,)
            return tuple(features) if not isinstance(features, tuple) else features

    def _reshape_logits_for_explain(
        self,
        backbone_feats: tuple[Tensor, ...],
        raw_logits: Tensor,
    ) -> tuple[Tensor, ...]:
        """Reshape raw logits for saliency map computation.

        Similar to DETR.split_and_reshape_logits but adapted for RF-DETR.

        Args:
            backbone_feats: Tuple of backbone features.
            raw_logits: Raw prediction logits from decoder.

        Returns:
            Tuple of reshaped logits aligned with backbone features.
        """
        # For RF-DETR, we need to handle the logits differently
        # since it uses a different decoder architecture
        # For now, return the raw logits reshaped to match backbone spatial dims
        if len(backbone_feats) == 1:
            feat = backbone_feats[0]
            batch_size = feat.shape[0]
            num_classes = raw_logits.shape[-1]
            h, w = feat.shape[-2], feat.shape[-1]

            # Create a spatial representation of logits
            # Average over queries: [B, num_queries, num_classes] -> [B, num_classes]
            spatial_logits = raw_logits.mean(dim=1)  # [B, num_classes]
            # Expand to spatial dimensions: [B, num_classes] -> [B, num_classes, H, W]
            spatial_logits = spatial_logits.unsqueeze(-1).unsqueeze(-1)  # [B, num_classes, 1, 1]
            spatial_logits = spatial_logits.expand(batch_size, num_classes, h, w)
            return (spatial_logits,)

        # Handle multi-scale features
        results = []
        for feat in backbone_feats:
            batch_size = feat.shape[0]
            num_classes = raw_logits.shape[-1]
            h, w = feat.shape[-2], feat.shape[-1]
            spatial_logits = raw_logits.mean(dim=1)  # [B, num_classes]
            spatial_logits = spatial_logits.unsqueeze(-1).unsqueeze(-1)  # [B, num_classes, 1, 1]
            spatial_logits = spatial_logits.expand(batch_size, num_classes, h, w)
            results.append(spatial_logits)
        return tuple(results)
