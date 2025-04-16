"""Numpy Validation functions."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import fields

import numpy as np
from datumaro import Polygon

from otx.core.data.entity.base import ImageInfo


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

        for field in fields(self):  # type: ignore[arg-type]
            if field.name not in validators:
                msg = f"Validation for field {field.name} is not implemented"
                raise NotImplementedError(msg)
            if (value := getattr(self, field.name)) is not None:
                validators[field.name](value)

    @staticmethod
    def _image_validator(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            msg = f"Image must be a numpy array. Got {type(image)}"
            raise TypeError(msg)
        if image.ndim != 3:
            msg = "Image must have 3 dimensions"
            raise ValueError(msg)
        if image.shape[-1] not in [1, 3]:
            msg = "Image must have 1 or 3 channels"
            raise ValueError(msg)
        if image.dtype != np.float32:
            msg = "Image must have dtype float32"
            raise ValueError(msg)
        return image

    @staticmethod
    def _label_validator(label: np.ndarray) -> np.ndarray:
        if not isinstance(label, np.ndarray):
            msg = "Label must be a numpy array"
            raise TypeError(msg)
        if label.dtype != np.int64:
            msg = "Label must have dtype np.int64"
            raise ValueError(msg)
        if label.ndim > 2:
            msg = "Label must have 0, 1, or 2 dimensions"
            raise ValueError(msg)
        return label

    @staticmethod
    def _scores_validator(scores: np.ndarray) -> np.ndarray:
        if not isinstance(scores, np.ndarray):
            msg = "Scores must be a numpy array"
            raise TypeError(msg)
        if not np.issubdtype(scores.dtype, np.floating):
            msg = f"Scores must be floating point. Got {scores.dtype}"
            raise ValueError(msg)
        if scores.ndim != 1:
            msg = "Scores must have 1 dimension"
            raise ValueError(msg)
        return scores

    @staticmethod
    def _feature_vector_validator(feature_vector: np.ndarray) -> np.ndarray:
        if not isinstance(feature_vector, np.ndarray):
            msg = "Feature vector must be a numpy array"
            raise TypeError(msg)
        if feature_vector.dtype != np.float32:
            msg = "Feature vector must have dtype np.float32"
            raise ValueError(msg)
        if feature_vector.ndim != 2:
            msg = "Feature vector must have 2 dimensions"
            raise ValueError(msg)
        return feature_vector

    @staticmethod
    def _saliency_map_validator(saliency_map: np.ndarray) -> np.ndarray:
        if not isinstance(saliency_map, np.ndarray):
            msg = "Saliency map must be a numpy array"
            raise TypeError(msg)
        if saliency_map.dtype not in [np.float32, np.uint8]:
            msg = "Saliency map must have dtype float32 or uint8"
            raise ValueError(msg)
        if saliency_map.ndim != 3:
            msg = "Saliency map must have 3 dimensions"
            raise ValueError(msg)
        return saliency_map

    @staticmethod
    def _mask_validator(mask: np.ndarray) -> np.ndarray:
        """Validate the mask."""
        if not isinstance(mask, np.ndarray):
            msg = "Mask must be a numpy array"
            raise TypeError(msg)
        return mask

    @staticmethod
    def _boxes_validator(boxes: np.ndarray) -> np.ndarray:
        if not isinstance(boxes, np.ndarray):
            msg = "Boxes must be a numpy array"
            raise TypeError(msg)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            msg = "Boxes must be of shape (N, 4)"
            raise ValueError(msg)
        return boxes

    @staticmethod
    def _keypoints_validator(keypoints: np.ndarray) -> np.ndarray:
        if not isinstance(keypoints, np.ndarray):
            msg = "Keypoints must be a numpy array"
            raise TypeError(msg)
        if keypoints.dtype != np.float32:
            msg = "Keypoints must have dtype float32"
            raise ValueError(msg)
        if keypoints.ndim != 2 or keypoints.shape[1] != 3:
            msg = "Keypoints must be 2D with shape (N, 3)"
            raise ValueError(msg)
        if np.any((keypoints[:, 2] < 0) | (keypoints[:, 2] > 1)):
            msg = "Keypoints visibility must be between 0 and 1"
            raise ValueError(msg)
        return keypoints

    @staticmethod
    def _polygons_validator(polygons: list[Polygon]) -> list[Polygon]:
        if len(polygons) == 0:
            return polygons
        if not isinstance(polygons, list) or not isinstance(polygons[0], Polygon):
            msg = "Polygons must be a list of datumaro.Polygon"
            raise TypeError(msg)
        return polygons

    @staticmethod
    def _img_info_validator(img_info: ImageInfo) -> ImageInfo:
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

        for field in fields(self):  # type: ignore[arg-type]
            if field.name not in validators:
                msg = f"Validation for field {field.name} is not implemented"
                raise NotImplementedError(msg)
            if (value := getattr(self, field.name)) is not None:
                validators[field.name](value)

    @staticmethod
    def _images_validator(image_batch: np.ndarray | list[np.ndarray]) -> np.ndarray | list[np.ndarray]:
        if isinstance(image_batch, np.ndarray):
            if image_batch.ndim != 4 or image_batch.shape[-1] not in [1, 3]:
                msg = "Image batch must be 4D and have 1 or 3 channels"
                raise ValueError(msg)
            if image_batch.dtype != np.float32:
                msg = "Image batch must have dtype float32"
                raise ValueError(msg)
        elif isinstance(image_batch, list):
            for img in image_batch:
                if (
                    not isinstance(img, np.ndarray)
                    or img.ndim != 3
                    or img.shape[-1] not in [1, 3]
                    or img.dtype != np.float32
                ):
                    msg = "Each image must be 3D with 1 or 3 channels and dtype float32"
                    raise ValueError(msg)
        else:
            msg = f"Image batch must be a numpy array or list of arrays. Got {type(image_batch)}"
            raise TypeError(msg)
        return image_batch

    @staticmethod
    def _labels_validator(label_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(label is None for label in label_batch):
            return []
        if not isinstance(label_batch[0], np.ndarray):
            msg = "Label batch must contain numpy arrays"
            raise TypeError(msg)
        if label_batch[0].dtype != np.int64 or label_batch[0].ndim > 2:
            msg = "Label batch must contain arrays with dtype int64 and max 2 dimensions"
            raise ValueError(msg)
        return label_batch

    @staticmethod
    def _scores_validator(scores_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(score is None for score in scores_batch):
            return []
        if not isinstance(scores_batch[0], np.ndarray):
            msg = "Scores batch must contain numpy arrays"
            raise TypeError(msg)
        if not np.issubdtype(scores_batch[0].dtype, np.floating):
            msg = "Scores must be float arrays"
            raise ValueError(msg)
        return scores_batch

    @staticmethod
    def _feature_vectors_validator(feature_vector_batch: list[np.ndarray]) -> list[np.ndarray]:
        if not isinstance(feature_vector_batch[0], np.ndarray):
            msg = "Feature vector batch must contain numpy arrays"
            raise TypeError(msg)
        if feature_vector_batch[0].dtype != np.float32:
            msg = "Feature vectors must be float32 numpy arrays"
            raise ValueError(msg)
        return feature_vector_batch

    @staticmethod
    def _saliency_maps_validator(saliency_map_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(m is None for m in saliency_map_batch):
            return []
        if not isinstance(saliency_map_batch[0], np.ndarray):
            msg = "Saliency maps batch must contain numpy arrays"
            raise TypeError(msg)
        if saliency_map_batch[0].dtype not in [np.float32, np.uint8]:
            msg = "Saliency maps must be float32 or uint8 arrays"
            raise ValueError(msg)
        return saliency_map_batch

    @staticmethod
    def _masks_validator(masks_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(mask is None for mask in masks_batch):
            return []
        if not isinstance(masks_batch, list) or not isinstance(masks_batch[0], np.ndarray):
            msg = f"Masks batch must be a list of numpy arrays. Got {type(masks_batch)}"
            raise TypeError(msg)
        return masks_batch

    @staticmethod
    def _boxes_validator(boxes_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(box is None for box in boxes_batch):
            return []
        if not isinstance(boxes_batch[0], np.ndarray):
            msg = "Boxes batch must be a list of numpy arrays"
            raise TypeError(msg)
        if not np.issubdtype(boxes_batch[0].dtype, np.floating):
            msg = "Boxes must be floating point arrays"
            raise ValueError(msg)
        if boxes_batch[0].ndim != 2 or boxes_batch[0].shape[1] != 4:
            msg = "Boxes must have shape (N, 4)"
            raise ValueError(msg)
        return boxes_batch

    @staticmethod
    def _keypoints_validator(keypoints_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(keypoints is None for keypoints in keypoints_batch):
            return []
        if not isinstance(keypoints_batch[0], np.ndarray):
            msg = "Keypoints must be numpy arrays"
            raise TypeError(msg)
        if keypoints_batch[0].ndim != 2 or keypoints_batch[0].shape[1] != 3:
            msg = "Keypoints must have shape (N, 3)"
            raise ValueError(msg)
        if np.any((keypoints_batch[0][:, 2] < 0) | (keypoints_batch[0][:, 2] > 1)):
            msg = "Keypoint visibility must be in [0, 1]"
            raise ValueError(msg)
        return keypoints_batch

    @staticmethod
    def _imgs_info_validator(imgs_info_batch: list[ImageInfo]) -> list[ImageInfo]:
        if not isinstance(imgs_info_batch[0], ImageInfo):
            msg = "Each image info must be ImageInfo"
            raise TypeError(msg)
        return imgs_info_batch

    @staticmethod
    def _batch_size_validator(batch_size: int) -> int:
        if not isinstance(batch_size, int):
            msg = "Batch size must be an int"
            raise TypeError(msg)
        return batch_size

    @staticmethod
    def _polygons_validator(polygons_batch: list[list[Polygon]]) -> list[list[Polygon]]:
        if all(polygon is None for polygon in polygons_batch):
            return []
        if not isinstance(polygons_batch[0], list):
            msg = "Polygons batch must be a list of lists"
            raise TypeError(msg)
        if not isinstance(polygons_batch[0][0], Polygon):
            msg = "Polygons batch must be a list of list of Polygon"
            raise TypeError(msg)
        return polygons_batch
