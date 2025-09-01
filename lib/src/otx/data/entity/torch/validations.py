# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Validation functions."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import torch
from datumaro import Polygon
from torchvision.tv_tensors import BoundingBoxes, Mask

from otx.data.entity.base import ImageInfo


class ValidateItemMixin:
    """Validate item mixin."""

    def __post_init__(self) -> None:
        validators = {
            "image": self._image_validator,
            "label": self._label_validator,
            "scores": self._scores_validator,
            "feature_vector": self._feature_vector_validator,
            "saliency_map": self._saliency_map_validator,
            "masks": self._mask_validator,
            "bboxes": self._boxes_validator,
            "keypoints": self._keypoints_validator,
            "polygons": self._polygons_validator,
            "img_info": self._img_info_validator,
        }
        # TODO(ashwinvaidya17): Revisit this
        for field in fields(self):  # type: ignore[arg-type]
            if field.name not in validators:
                msg = f"Validation for field {field.name} is not implemented"
                raise NotImplementedError(msg)
            if field.name in validators and (value := getattr(self, field.name)) is not None:
                validators[field.name](value)  #

    @staticmethod
    def _image_validator(image: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Validate the image."""
        if not isinstance(image, (torch.Tensor, np.ndarray)):
            msg = f"Image must be a torch tensor or numpy array. Got {type(image)}"
            raise TypeError(msg)
        if image.ndim != 3:
            msg = "Image must have 3 dimensions"
            raise ValueError(msg)
        if image.dtype not in (torch.uint8, torch.float32, np.uint8, np.float32):
            msg = "Image must have dtype float32 or uint8"
            raise ValueError(msg)
        return image

    @staticmethod
    def _label_validator(label: torch.Tensor) -> torch.Tensor:
        """Validate the label."""
        if not isinstance(label, torch.Tensor):
            msg = "Label must be a torch tensor"
            raise TypeError(msg)
        if label.dtype != torch.long:
            msg = f"Label must have dtype torch.long, but got {label.dtype}"
            raise ValueError(msg)
        # detection tasks allow multiple labels so the shape is [B, N]
        if label.ndim > 2:
            msg = "Label must have 0, 1, or 2 dimensions"
            raise ValueError(msg)
        return label

    @staticmethod
    def _scores_validator(scores: torch.Tensor) -> torch.Tensor:
        """Validate the scores."""
        if not isinstance(scores, torch.Tensor):
            msg = "Scores must be a torch tensor"
            raise TypeError(msg)
        if not scores.dtype.is_floating_point:
            msg = f"Scores must have a floating point dtype. Got {scores.dtype}"
            raise ValueError(msg)
        if scores.ndim != 1:
            msg = "Scores must have 1 dimension"
            raise ValueError(msg)
        return scores

    @staticmethod
    def _feature_vector_validator(feature_vector: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Validate the feature vector.

        Numpy is mixed for this round as it is used in OV Classification.
        """
        if not isinstance(feature_vector, (torch.Tensor, np.ndarray)):
            msg = "Feature vector must be a torch tensor or numpy array"
            raise TypeError(msg)
        if isinstance(feature_vector, torch.Tensor) and feature_vector.dtype != torch.float32:
            msg = "Feature vector must have dtype torch.float32"
            raise ValueError(msg)
        if isinstance(feature_vector, np.ndarray) and feature_vector.dtype != np.float32:
            msg = "Feature vector must have dtype np.float32"
            raise ValueError(msg)
        if feature_vector.ndim != 2:
            msg = "Feature vector must have 2 dimensions"
            raise ValueError(msg)
        return feature_vector

    @staticmethod
    def _saliency_map_validator(saliency_map: torch.Tensor) -> torch.Tensor:
        """Validate the saliency map."""
        if not isinstance(saliency_map, torch.Tensor):
            msg = "Saliency map must be a torch tensor"
            raise TypeError(msg)
        # TODO(ashwinvaidya17): use only one dtype. Kept for OV Classification compatibility
        if not (saliency_map.dtype.is_floating_point or saliency_map.dtype == torch.uint8):
            msg = "Saliency map must have dtype torch.float32 or torch.uint8"
            raise ValueError(msg)
        if saliency_map.ndim != 3:
            msg = "Saliency map must have 3 dimensions"
            raise ValueError(msg)
        return saliency_map

    @staticmethod
    def _mask_validator(mask: Mask) -> Mask:
        """Validate the mask."""
        if not isinstance(mask, Mask):
            msg = "Mask must be a torchvision.tv_tensors.Mask"
            raise TypeError(msg)
        return mask

    @staticmethod
    def _boxes_validator(boxes: BoundingBoxes) -> BoundingBoxes:
        """Validate the boxes."""
        if not isinstance(boxes, BoundingBoxes):
            msg = "Boxes must be a torchvision.tv_tensors.BoundingBoxes"
            raise TypeError(msg)
        return boxes

    @staticmethod
    def _keypoints_validator(keypoints: torch.Tensor) -> torch.Tensor:
        """Validate the keypoints."""
        if not isinstance(keypoints, torch.Tensor):
            msg = "Keypoints must be a torch tensor"
            raise TypeError(msg)
        if keypoints.dtype != torch.float32:
            msg = "Keypoints must have dtype torch.float32"
            raise ValueError(msg)
        if keypoints.ndim != 2:
            msg = "Keypoints must have 2 dimensions"
            raise ValueError(msg)
        if keypoints.shape[1] != 3:
            msg = "Keypoints must have 2 coordinates and 1 visibility value"
            raise ValueError(msg)
        if any(keypoints[:, 2] > 1) or any(keypoints[:, 2] < 0):
            msg = "Keypoints visibility must be between 0 and 1"
            raise ValueError(msg)
        return keypoints

    @staticmethod
    def _polygons_validator(polygons: list[Polygon]) -> list[Polygon]:
        """Validate the polygons."""
        if len(polygons) == 0:
            return polygons
        if not isinstance(polygons, list):
            msg = f"Polygons must be a list of datumaro.Polygon. Got {type(polygons)}"
            raise TypeError(msg)
        if not isinstance(polygons[0], Polygon):
            msg = f"Polygons must be a list of datumaro.Polygon. Got {type(polygons[0])}"
            raise TypeError(msg)
        return polygons

    @staticmethod
    def _img_info_validator(img_info: ImageInfo) -> ImageInfo:
        """Validate the image info."""
        if not isinstance(img_info, ImageInfo):
            msg = "Image info must be a otx.data.entity.ImageInfo"
            raise TypeError(msg)
        return img_info


class ValidateBatchMixin:
    """Validate batch mixin."""

    def __post_init__(self) -> None:
        validators = {
            "images": self._images_validator,
            "labels": self._labels_validator,
            "scores": self._scores_validator,
            "feature_vector": self._feature_vectors_validator,
            "saliency_map": self._saliency_maps_validator,
            "masks": self._masks_validator,
            "bboxes": self._boxes_validator,
            "keypoints": self._keypoints_validator,
            "polygons": self._polygons_validator,
            "imgs_info": self._imgs_info_validator,
            "batch_size": self._batch_size_validator,
        }
        # TODO(ashwinvaidya17): Revisit this
        for field in fields(self):  # type: ignore[arg-type]
            if field.name not in validators:
                msg = f"Validation for field {field.name} is not implemented"
                raise NotImplementedError(msg)
            if field.name in validators and (value := getattr(self, field.name)) is not None:
                # TODO(ashwinvaidya17): ignore is needed only for batch_size. Revisit
                validators[field.name](value)  # type: ignore[operator]

    @staticmethod
    def _images_validator(image_batch: torch.Tensor) -> torch.Tensor:
        """Validate the image batch."""
        if not isinstance(image_batch, list) and not isinstance(image_batch, torch.Tensor):
            msg = f"Image batch must be a torch tensor or list of tensors. Got {type(image_batch)}"
            raise TypeError(msg)
        if isinstance(image_batch, torch.Tensor):
            if image_batch.dtype not in (torch.float32, torch.uint8):
                msg = f"Image batch must have dtype float32 or uint8. Found {image_batch.dtype}"
                raise ValueError(msg)
            if image_batch.ndim != 4:
                msg = "Image batch must have 4 dimensions"
                raise ValueError(msg)
            if image_batch.shape[1] not in [1, 3]:
                msg = "Image batch must have 1 or 3 channels"
                raise ValueError(msg)
        else:
            if not all(isinstance(image, torch.Tensor) for image in image_batch):
                msg = "Image batch must be a list of torch tensors"
                raise TypeError(msg)
            dtype = image_batch[0].dtype
            if dtype not in (torch.float32, torch.uint8):
                msg = "Image batch must have dtype float32 or uint8"
                raise ValueError(msg)
            if not all(image.dtype == dtype for image in image_batch):
                msg = f"Not all tensors have the same dtype: expected {dtype}"
                raise TypeError(msg)
            if not all(image.ndim == 3 for image in image_batch):
                msg = "Image batch must have 3 dimensions"
                raise ValueError(msg)
            if not all(image.shape[0] in [1, 3] for image in image_batch):
                msg = "Image batch must have 1 or 3 channels"
                raise ValueError(msg)
        return image_batch

    @staticmethod
    def _labels_validator(label_batch: list[torch.Tensor]) -> list[torch.Tensor]:
        """Validate the label batch."""
        if all(label is None for label in label_batch):
            return []
        if not isinstance(label_batch, list) or not isinstance(label_batch[0], torch.Tensor):
            msg = f"Label batch must be a list of torch tensors. Got {type(label_batch)}"
            raise TypeError(msg)
        # assumes homogeneous data so validation is done only for the first element
        if label_batch[0].dtype != torch.long:
            msg = "Label batch must have dtype torch.long"
            raise ValueError(msg)
        if label_batch[0].ndim > 2:
            msg = f"Label batch must have shape of (N, 1) or (N,), but got {label_batch[0].shape}"
            raise ValueError(msg)
        return label_batch

    @staticmethod
    def _scores_validator(scores_batch: list[torch.Tensor | None]) -> list[torch.Tensor]:
        """Validate the scores batch."""
        if all(score is None for score in scores_batch):
            return []
        if not isinstance(scores_batch, list) or not isinstance(scores_batch[0], torch.Tensor):
            msg = f"Scores batch must be a list of torch tensors. Got {type(scores_batch)}"
            raise TypeError(msg)
        # assumes homogeneous data so validation is done only for the first element
        if not scores_batch[0].dtype.is_floating_point:
            msg = f"Scores batch must have a floating point dtype. Got {scores_batch[0].dtype}"
            raise ValueError(msg)
        if scores_batch[0].ndim > 1:
            msg = "Scores batch must have 1 or 2 dimensions"
            raise ValueError(msg)
        return scores_batch

    @staticmethod
    def _feature_vectors_validator(
        feature_vector_batch: list[torch.Tensor | np.ndarray],
    ) -> list[torch.Tensor | np.ndarray]:
        """Validate the feature vector.

        Numpy is mixed for this round as it is used in OV Classification.
        """
        if not isinstance(feature_vector_batch, list) or not isinstance(
            feature_vector_batch[0],
            (torch.Tensor, np.ndarray),
        ):
            msg = (
                "Feature vector batch must be a list of torch tensors or numpy arrays."
                f" Got {type(feature_vector_batch)}"
            )
            raise TypeError(msg)
        # assumes homogeneous data so validation is done only for the first element
        # TODO(ashwinvaidya17): use only one dtype. Kept for OV Classification compatibility
        if isinstance(feature_vector_batch[0], torch.Tensor) and not feature_vector_batch[0].dtype.is_floating_point:
            msg = f"Feature vector must have a floating point dtype. Got {feature_vector_batch[0].dtype}"
            raise ValueError(msg)
        if isinstance(feature_vector_batch[0], np.ndarray) and feature_vector_batch[0].dtype.kind != "f":
            msg = f"Feature vector must have a floating point dtype. Got {feature_vector_batch[0].dtype}"
            raise ValueError(msg)
        if isinstance(feature_vector_batch[0], torch.Tensor) and feature_vector_batch[0].ndim != 2:
            msg = "Feature vector must have 2 dimensions"
            raise ValueError(msg)
        return feature_vector_batch

    @staticmethod
    def _saliency_maps_validator(
        saliency_map_batch: list[torch.Tensor | np.ndarray | None],
    ) -> list[torch.Tensor | np.ndarray]:
        """Validate the saliency map batch.

        Numpy is mixed for this round as it is used in OV Classification.
        """
        if all(saliency_map is None for saliency_map in saliency_map_batch):
            return []
        return saliency_map_batch

    @staticmethod
    def _masks_validator(masks_batch: list[torch.Tensor | None]) -> list[torch.Tensor]:
        """Validate the masks batch."""
        if all(mask is None for mask in masks_batch):
            return []
        if not isinstance(masks_batch, list) or not isinstance(masks_batch[0], torch.Tensor):
            msg = f"Masks batch must be a list of torch tensors. Got {type(masks_batch)}"
            raise TypeError(msg)
        return masks_batch

    @staticmethod
    def _boxes_validator(boxes_batch: list[BoundingBoxes | None]) -> list[BoundingBoxes]:
        """Validate the boxes batch."""
        if all(box is None for box in boxes_batch):
            return []
        if not isinstance(boxes_batch, list) or not isinstance(boxes_batch[0], torch.Tensor):
            msg = f"Boxes batch must be a list of torch tensors. Got {type(boxes_batch)}"
            raise TypeError(msg)
        # assumes homogeneous data so validation is done only for the first element
        if not boxes_batch[0].dtype.is_floating_point:
            msg = f"Boxes batch must have a floating point dtype. Got {boxes_batch[0].dtype}"
            raise ValueError(msg)
        if boxes_batch[0].ndim != 2:
            msg = "Boxes batch must have 2 dimensions"
            raise ValueError(msg)
        if boxes_batch[0].shape[1] != 4:
            msg = "Boxes batch must have 4 coordinates"
            raise ValueError(msg)
        return boxes_batch

    @staticmethod
    def _keypoints_validator(keypoints_batch: list[torch.Tensor | None]) -> list[torch.Tensor]:
        """Validate the keypoints batch."""
        if all(keypoints is None for keypoints in keypoints_batch):
            return []
        if not isinstance(keypoints_batch, list) or not isinstance(keypoints_batch[0], torch.Tensor):
            msg = f"Keypoints batch must be a list of torch tensors. Got {type(keypoints_batch)}"
            raise TypeError(msg)
        # assumes homogeneous data so validation is done only for the first element
        if keypoints_batch[0].dtype != torch.float32:
            msg = "Keypoints batch must have dtype torch.float32"
            raise ValueError(msg)
        if keypoints_batch[0].ndim != 2:
            msg = "Keypoints batch must have 2 dimensions"
            raise ValueError(msg)
        if keypoints_batch[0].shape[1] != 3:
            msg = "Keypoints batch must have 2 coordinates and 1 visibility value"
            raise ValueError(msg)
        if any(keypoints_batch[0][:, 2] > 1) or any(keypoints_batch[0][:, 2] < 0):
            msg = "Keypoints visibility must be between 0 and 1"
            raise ValueError(msg)
        return keypoints_batch

    @staticmethod
    def _imgs_info_validator(imgs_info_batch: list[ImageInfo | None]) -> list[ImageInfo | None]:
        """Validate the image info batch."""
        if all(img_info is None for img_info in imgs_info_batch):
            return []
        if not isinstance(imgs_info_batch, list) or not isinstance(imgs_info_batch[0], ImageInfo):
            msg = "Image info batch must be a list of otx.data.entity.ImageInfo"
            raise TypeError(msg)
        return imgs_info_batch

    @staticmethod
    def _batch_size_validator(batch_size: int) -> int:
        """Validate the batch size.

        Note:
            This is temporary and batch size should not be part of the batch entity.
        """
        if not isinstance(batch_size, int):
            msg = "Batch size must be an integer"
            raise TypeError(msg)
        return batch_size

    @staticmethod
    def _polygons_validator(polygons_batch: list[list[Polygon] | None]) -> list[list[Polygon] | None]:
        """Validate the polygons batch."""
        if all(polygon is None for polygon in polygons_batch):
            return []
        if not isinstance(polygons_batch, list):
            msg = "Polygons batch must be a list"
            raise TypeError(msg)
        if not isinstance(polygons_batch[0], list):
            msg = "Polygons batch must be a list of list"
            raise TypeError(msg)
        if len(polygons_batch[0]) == 0:
            msg = f"Polygons batch must not be empty. Got {polygons_batch}"
            raise ValueError(msg)
        if not isinstance(polygons_batch[0][0], Polygon):
            msg = "Polygons batch must be a list of list of datumaro.Polygon"
            raise TypeError(msg)
        return polygons_batch
