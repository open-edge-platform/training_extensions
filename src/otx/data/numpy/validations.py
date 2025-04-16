from __future__ import annotations

from dataclasses import fields

import numpy as np
from datumaro import Polygon

from otx.core.data.entity.base import ImageInfo


class ValidateItemMixin:
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
                raise NotImplementedError(f"Validation for field {field.name} is not implemented")
            if (value := getattr(self, field.name)) is not None:
                validators[field.name](value)

    @staticmethod
    def _image_validator(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be a numpy array. Got {type(image)}")
        if image.ndim != 3:
            raise ValueError("Image must have 3 dimensions")
        if image.shape[-1] not in [1, 3]:
            raise ValueError("Image must have 1 or 3 channels")
        if image.dtype != np.float32:
            raise ValueError("Image must have dtype float32")
        return image

    @staticmethod
    def _label_validator(label: np.ndarray) -> np.ndarray:
        if not isinstance(label, np.ndarray):
            raise TypeError("Label must be a numpy array")
        if label.dtype != np.int64:
            raise ValueError("Label must have dtype np.int64")
        if label.ndim > 2:
            raise ValueError("Label must have 0, 1, or 2 dimensions")
        return label

    @staticmethod
    def _scores_validator(scores: np.ndarray) -> np.ndarray:
        if not isinstance(scores, np.ndarray):
            raise TypeError("Scores must be a numpy array")
        if not np.issubdtype(scores.dtype, np.floating):
            raise ValueError(f"Scores must be floating point. Got {scores.dtype}")
        if scores.ndim != 1:
            raise ValueError("Scores must have 1 dimension")
        return scores

    @staticmethod
    def _feature_vector_validator(feature_vector: np.ndarray) -> np.ndarray:
        if not isinstance(feature_vector, np.ndarray):
            raise TypeError("Feature vector must be a numpy array")
        if feature_vector.dtype != np.float32:
            raise ValueError("Feature vector must have dtype np.float32")
        if feature_vector.ndim != 2:
            raise ValueError("Feature vector must have 2 dimensions")
        return feature_vector

    @staticmethod
    def _saliency_map_validator(saliency_map: np.ndarray) -> np.ndarray:
        if not isinstance(saliency_map, np.ndarray):
            raise TypeError("Saliency map must be a numpy array")
        if saliency_map.dtype not in [np.float32, np.uint8]:
            raise ValueError("Saliency map must have dtype float32 or uint8")
        if saliency_map.ndim != 3:
            raise ValueError("Saliency map must have 3 dimensions")
        return saliency_map

    @staticmethod
    def _mask_validator(mask) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a numpy array")
        # optionally add shape or dtype checks here
        return mask

    @staticmethod
    def _boxes_validator(boxes: np.ndarray) -> np.ndarray:
        if not isinstance(boxes, np.ndarray):
            raise TypeError("Boxes must be a numpy array")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("Boxes must be of shape (N, 4)")
        return boxes

    @staticmethod
    def _keypoints_validator(keypoints: np.ndarray) -> np.ndarray:
        if not isinstance(keypoints, np.ndarray):
            raise TypeError("Keypoints must be a numpy array")
        if keypoints.dtype != np.float32:
            raise ValueError("Keypoints must have dtype float32")
        if keypoints.ndim != 2 or keypoints.shape[1] != 3:
            raise ValueError("Keypoints must be 2D with shape (N, 3)")
        if np.any((keypoints[:, 2] < 0) | (keypoints[:, 2] > 1)):
            raise ValueError("Keypoints visibility must be between 0 and 1")
        return keypoints

    @staticmethod
    def _polygons_validator(polygons: list[Polygon]) -> list[Polygon]:
        if len(polygons) == 0:
            return polygons
        if not isinstance(polygons, list) or not isinstance(polygons[0], Polygon):
            raise TypeError("Polygons must be a list of datumaro.Polygon")
        return polygons

    @staticmethod
    def _img_info_validator(img_info: ImageInfo) -> ImageInfo:
        if not isinstance(img_info, ImageInfo):
            raise TypeError("Image info must be a otx.data.entity.ImageInfo")
        return img_info


class ValidateBatchMixin:
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
                raise NotImplementedError(f"Validation for field {field.name} is not implemented")
            if (value := getattr(self, field.name)) is not None:
                validators[field.name](value)

    @staticmethod
    def _images_validator(image_batch: np.ndarray | list[np.ndarray]) -> np.ndarray | list[np.ndarray]:
        if isinstance(image_batch, np.ndarray):
            if image_batch.ndim != 4 or image_batch.shape[-1] not in [1, 3]:
                raise ValueError("Image batch must be 4D and have 1 or 3 channels")
            if image_batch.dtype != np.float32:
                raise ValueError("Image batch must have dtype float32")
        elif isinstance(image_batch, list):
            for img in image_batch:
                if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[-1] not in [1, 3] or img.dtype != np.float32:
                    raise ValueError("Each image must be 3D with 1 or 3 channels and dtype float32")
        else:
            raise TypeError(f"Image batch must be a numpy array or list of arrays. Got {type(image_batch)}")
        return image_batch

    @staticmethod
    def _labels_validator(label_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(label is None for label in label_batch):
            return []
        if not isinstance(label_batch[0], np.ndarray) or label_batch[0].dtype != np.int64 or label_batch[0].ndim > 2:
            raise ValueError("Label batch must contain arrays with dtype int64 and max 2 dimensions")
        return label_batch

    @staticmethod
    def _scores_validator(scores_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(score is None for score in scores_batch):
            return []
        if not isinstance(scores_batch[0], np.ndarray) or not np.issubdtype(scores_batch[0].dtype, np.floating):
            raise ValueError("Scores must be float arrays")
        return scores_batch

    @staticmethod
    def _feature_vectors_validator(feature_vector_batch: list[np.ndarray]) -> list[np.ndarray]:
        if not isinstance(feature_vector_batch[0], np.ndarray) or feature_vector_batch[0].dtype != np.float32:
            raise ValueError("Feature vectors must be float32 numpy arrays")
        return feature_vector_batch

    @staticmethod
    def _saliency_maps_validator(saliency_map_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(m is None for m in saliency_map_batch):
            return []
        if not isinstance(saliency_map_batch[0], np.ndarray) or saliency_map_batch[0].dtype not in [np.float32, np.uint8]:
            raise ValueError("Saliency maps must be float32 or uint8 arrays")
        return saliency_map_batch

    @staticmethod
    def _masks_validator(masks_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(mask is None for mask in masks_batch):
            return []
        if not isinstance(masks_batch, list) or not isinstance(masks_batch[0], np.ndarray):
            msg = f"Masks batch must be a list of torch tensors. Got {type(masks_batch)}"
            raise TypeError(msg)
        return masks_batch  
    
    @staticmethod
    def _boxes_validator(boxes_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(box is None for box in boxes_batch):
            return []
        if not isinstance(boxes_batch[0], np.ndarray) or not np.issubdtype(boxes_batch[0].dtype, np.floating):
            raise ValueError("Boxes must be floating point arrays")
        if boxes_batch[0].ndim != 2 or boxes_batch[0].shape[1] != 4:
            raise ValueError("Boxes must have shape (N, 4)")
        return boxes_batch

    @staticmethod
    def _keypoints_validator(keypoints_batch: list[np.ndarray]) -> list[np.ndarray]:
        if all(keypoints is None for keypoints in keypoints_batch):
            return []
        if not isinstance(keypoints_batch[0], np.ndarray):
            raise TypeError("Keypoints must be numpy arrays")
        if keypoints_batch[0].ndim != 2 or keypoints_batch[0].shape[1] != 3:
            raise ValueError("Keypoints must have shape (N, 3)")
        if np.any((keypoints_batch[0][:, 2] < 0) | (keypoints_batch[0][:, 2] > 1)):
            raise ValueError("Keypoint visibility must be in [0, 1]")
        return keypoints_batch

    @staticmethod
    def _imgs_info_validator(imgs_info_batch: list[ImageInfo]) -> list[ImageInfo]:
        if not isinstance(imgs_info_batch[0], ImageInfo):
            raise TypeError("Each image info must be ImageInfo")
        return imgs_info_batch

    @staticmethod
    def _batch_size_validator(batch_size: int) -> int:
        if not isinstance(batch_size, int):
            raise TypeError("Batch size must be an int")
        return batch_size

    @staticmethod
    def _polygons_validator(polygons_batch: list[list[Polygon]]) -> list[list[Polygon]]:
        if all(polygon is None for polygon in polygons_batch):
            return []
        if not isinstance(polygons_batch[0], list) or not isinstance(polygons_batch[0][0], Polygon):
            raise TypeError("Polygons batch must be a list of list of Polygon")
        return polygons_batch
