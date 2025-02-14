"""Validation functions."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def validate_image(image: torch.Tensor) -> torch.Tensor:
    """Validate the image."""
    if not isinstance(image, torch.Tensor):
            msg = "Image must be a torch tensor"
            raise TypeError(msg)
    if image.ndim != 3:
        msg = "Image must have 3 dimensions"
        raise ValueError(msg)
    if image.shape[0] not in [1, 3]:
        msg = "Image must have 1 or 3 channels"
        raise ValueError(msg)
    if image.dtype != torch.float32:
        msg = "Image must have dtype float32"
        raise ValueError(msg)
    return image

def validate_image_batch(image_batch: torch.Tensor) -> torch.Tensor:
    """Validate the image batch."""
    if not isinstance(image_batch, torch.Tensor):
        msg = "Image batch must be a torch tensor"
        raise TypeError(msg)
    if image_batch.dtype != torch.float32:
            msg = "Image batch must have dtype float32"
            raise ValueError(msg)
    if image_batch.ndim != 4:
        msg = "Image batch must have 4 dimensions"
        raise ValueError(msg)
    if image_batch.shape[1] not in [1, 3]:
        msg = "Image batch must have 1 or 3 channels"
        raise ValueError(msg)
    return image_batch


def validate_label(label: torch.Tensor) -> torch.Tensor:
    """Validate the label."""
    if not isinstance(label, torch.Tensor):
        msg = "Label must be a torch tensor"
        raise TypeError(msg)
    if label.dtype != torch.long:
        msg = "Label must have dtype torch.long"
        raise ValueError(msg)
    if label.ndim > 1:
        msg = "Label must have 0 or 1 dimension"
        raise ValueError(msg)
    return label

def validate_label_batch(label_batch: torch.Tensor) -> torch.Tensor:
    """Validate the label batch."""
    if not isinstance(label_batch, torch.Tensor):
        msg = "Label batch must be a torch tensor"
        raise TypeError(msg)
    if label_batch.dtype != torch.long:
        msg = "Label batch must have dtype torch.long"
        raise ValueError(msg)
    if label_batch.ndim > 2:
        msg = f"Label batch must have shape of (N, 1) or (N,), but got {label_batch.shape}"
        raise ValueError(msg)
    return label_batch

def validate_scores(scores: torch.Tensor) -> torch.Tensor:
    """Validate the scores."""
    if not isinstance(scores, torch.Tensor):
        msg = "Scores must be a torch tensor"
        raise TypeError(msg)
    if scores.dtype != torch.float32:
        msg = "Scores must have dtype torch.float32"
        raise ValueError(msg)
    if scores.ndim != 2:
        msg = "Scores must have 2 dimensions"
        raise ValueError(msg)
    return scores

def validate_scores_batch(scores_batch: torch.Tensor) -> torch.Tensor:
    """Validate the scores batch."""
    if not isinstance(scores_batch, torch.Tensor):
        msg = "Scores batch must be a torch tensor"
        raise TypeError(msg)
    if scores_batch.dtype != torch.float32:
        msg = "Scores batch must have dtype torch.float32"
        raise ValueError(msg)
    if scores_batch.ndim != 2:
        msg = "Scores batch must have 2 dimensions"
        raise ValueError(msg)
    return scores_batch

def validate_feature_vector_and_batch(feature_vector: torch.Tensor) -> torch.Tensor:
    """Validate the feature vector."""
    if not isinstance(feature_vector, torch.Tensor):
        msg = "Feature vector must be a torch tensor"
        raise TypeError(msg)
    if feature_vector.dtype != torch.float32:
        msg = "Feature vector must have dtype torch.float32"
        raise ValueError(msg)
    if feature_vector.ndim != 2:
        msg = "Feature vector must have 2 dimensions"
        raise ValueError(msg)
    return feature_vector

def validate_saliency_map(saliency_map: torch.Tensor) -> torch.Tensor:
    """Validate saliency map."""
    if not isinstance(saliency_map, torch.Tensor):
            msg = "Saliency map must be a torch tensor"
            raise TypeError(msg)
    if saliency_map.dtype != torch.float32:
        msg = "Saliency map must have dtype torch.float32"
        raise ValueError(msg)
    if saliency_map.ndim != 3:
        msg = "Saliency map must have 4 dimensions"
        raise ValueError(msg)
    return saliency_map



def validate_saliency_map_batch(saliency_map_batch: torch.Tensor) -> torch.Tensor:
    """Validate the saliency map batch."""
    if not isinstance(saliency_map_batch, torch.Tensor):
        msg = "Saliency map batch must be a torch tensor"
        raise TypeError(msg)
    if saliency_map_batch.dtype != torch.float32:
        msg = "Saliency map batch must have dtype torch.float32"
        raise ValueError(msg)
    if saliency_map_batch.ndim != 4:
        msg = "Saliency map batch must have 4 dimensions"
        raise ValueError(msg)
    return saliency_map_batch

def validate_masks_batch(masks_batch: torch.Tensor) -> torch.Tensor:
    """Validate the masks batch."""
    if not isinstance(masks_batch, torch.Tensor):
        msg = "Masks batch must be a torch tensor"
        raise TypeError(msg)
    if masks_batch.dtype != torch.bool:
        msg = "Masks batch must have dtype torch.bool"
        raise ValueError(msg)
    if masks_batch.ndim != 4:
        msg = "Masks batch must have 4 dimensions"
        raise ValueError(msg)
    return masks_batch

def validate_boxes_batch(boxes_batch: torch.Tensor) -> torch.Tensor:
    """Validate the boxes batch."""
    if not isinstance(boxes_batch, torch.Tensor):
        msg = "Boxes batch must be a torch tensor"
        raise TypeError(msg)
    if boxes_batch.dtype != torch.float32:
        msg = "Boxes batch must have dtype torch.float32"
        raise ValueError(msg)
    if boxes_batch.ndim != 3:
        msg = "Boxes batch must have 3 dimensions"
        raise ValueError(msg)
    if boxes_batch.shape[1] != 4:
        msg = "Boxes batch must have 4 coordinates"
        raise ValueError(msg)
    return boxes_batch