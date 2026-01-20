# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for OTX data entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import polars as pl
import torch
import torch.utils._pytree as pytree
from datumaro.experimental.dataset import Sample
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import (
    Subset,
    bbox_field,
    image_field,
    image_info_field,
    instance_mask_field,
    keypoints_field,
    label_field,
    mask_field,
    subset_field,
)
from torchvision import tv_tensors

from otx.data.entity.base import ImageInfo

if TYPE_CHECKING:
    from torchvision.tv_tensors import BoundingBoxes, Mask


def register_pytree_node(cls: type[Sample]) -> type[Sample]:
    """Decorator to register an OTX data entity with PyTorch's PyTree.

    This decorator should be applied to every OTX data entity, as TorchVision V2 transforms
    use the PyTree to flatten and unflatten the data entity during runtime.

    Example:
        `MulticlassClsDataEntity` example ::

            @register_pytree_node
            @dataclass
            class MulticlassClsDataEntity(OTXDataEntity):
                ...
    """

    def flatten_fn(obj: object) -> tuple[list[Any], list[str]]:
        obj_dict = dict(obj.__dict__)

        missing_keys = set(obj.__class__.__annotations__.keys()) - set(obj_dict.keys())
        for key in missing_keys:
            obj_dict[key] = getattr(obj, key)

        return (list(obj_dict.values()), list(obj_dict.keys()))

    def unflatten_fn(values: list[Any], context: list[str]) -> object:
        kwargs = dict(zip(context, values))
        # Extract _img_info to set after construction (since __post_init__ would overwrite it)
        img_info = kwargs.pop("_img_info", None)
        # Remove _transforms as it's not a constructor argument
        kwargs.pop("_transforms", None)
        obj = cls(**kwargs)
        # Restore _img_info if it was present (preserves transformed img_info)
        if img_info is not None:
            obj._img_info = img_info
        return obj

    pytree.register_pytree_node(
        cls,
        flatten_fn=flatten_fn,
        unflatten_fn=unflatten_fn,
    )
    return cls


@register_pytree_node
class OTXSample(Sample):
    """Base class for OTX data samples."""

    image: np.ndarray | torch.Tensor | tv_tensors.Image | Any
    subset: Subset = subset_field()

    @property
    def img_info(self) -> ImageInfo:
        """Get image information for the sample."""
        if self._img_info is None:
            err_msg = "img_info is not set."
            raise ValueError(err_msg)
        return self._img_info

    @img_info.setter
    def img_info(self, value: ImageInfo) -> None:
        self._img_info = value


@register_pytree_node
class ClassificationSample(OTXSample):
    """ClassificationSample is a base class for OTX classification items."""

    subset: Subset = subset_field()

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8())
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class ClassificationMultiLabelSample(OTXSample):
    """ClassificationMultiLabelSample is a base class for OTX multi label classification items."""

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: np.ndarray | torch.Tensor = label_field(pl.UInt8(), multi_label=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class ClassificationHierarchicalSample(OTXSample):
    """ClassificationHierarchicalSample is a base class for OTX hierarchical classification items."""

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: np.ndarray | torch.Tensor = label_field(pl.UInt8(), is_list=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class DetectionSample(OTXSample):
    """DetectionSample is a base class for OTX detection items."""

    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(
        dtype=pl.UInt8(), format="BGR", channels_first=True
    )
    label: torch.Tensor = label_field(pl.UInt8(), is_list=True)
    bboxes: np.ndarray | tv_tensors.BoundingBoxes = bbox_field(dtype=pl.Float32())
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        # Convert bboxes to tv_tensors format
        if isinstance(self.bboxes, np.ndarray):
            self.bboxes = tv_tensors.BoundingBoxes(
                self.bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=shape,
                dtype=torch.float32,
            )

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class SegmentationSample(OTXSample):
    """OTXSample for segmentation tasks."""

    subset: Subset = subset_field()
    image: np.ndarray | tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=False)
    masks: tv_tensors.Mask = mask_field(dtype=pl.UInt8(), channels_first=True, has_channels_dim=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)
        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class InstanceSegmentationSample(OTXSample):
    """OTXSample for instance segmentation tasks."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    bboxes: np.ndarray | tv_tensors.BoundingBoxes = bbox_field(dtype=pl.Float32())
    masks: tv_tensors.Mask = instance_mask_field(dtype=pl.UInt8())
    label: torch.Tensor = label_field(is_list=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        # Convert bboxes to tv_tensors format
        if isinstance(self.bboxes, np.ndarray):
            self.bboxes = tv_tensors.BoundingBoxes(
                self.bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=shape,
                dtype=torch.float32,
            )

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


@register_pytree_node
class KeypointSample(OTXSample):
    """KeypointSample is a base class for OTX keypoint detection items."""

    subset: Subset = subset_field()
    image: tv_tensors.Image | np.ndarray | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8(), is_list=True)
    keypoints: torch.Tensor = keypoints_field()
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        self.img_info = ImageInfo(
            img_idx=0,
            img_shape=shape,
            ori_shape=shape,
        )


def collate_fn(samples: list[OTXSample]) -> OTXSampleBatch:
    """Collate OTXSamples into a batch.

    Args:
        samples: List of OTXSamples to batch

    Returns:
        Batched OTXSampleBatch with stacked tensors
    """
    # Check if all images have the same size. TODO(kprokofi): remove this check once OV IR models are moved.
    if all(sample.image.shape == samples[0].image.shape for sample in samples):
        images = torch.stack([sample.image for sample in samples])
    else:
        # we need this only in case of OV inference, where no resize
        images = [sample.image for sample in samples]

    return OTXSampleBatch(
        batch_size=len(samples),
        images=images,
        labels=[sample.label for sample in samples],
        bboxes=[sample.bboxes for sample in samples],
        keypoints=[sample.keypoints for sample in samples],
        masks=[sample.masks for sample in samples],
        polygons=[sample.polygons for sample in samples],  # type: ignore[misc]
        imgs_info=[sample.img_info for sample in samples],
    )


@dataclass
class OTXSampleBatch:
    """OTX sample batch implementation.

    Attributes:
        batch_size: The number of samples in the batch.
        images: The batch of images as a tensor or list of tensors.
        labels: List of label tensors, optional.
        masks: List of masks, optional.
        bboxes: List of bounding boxes, optional.
        keypoints: List of keypoint tensors, optional.
        polygons: List of polygon arrays, optional.
        imgs_info: Sequence of image information, optional.
    """

    batch_size: int  # TODO(ashwinvaidya17): Remove this
    images: torch.Tensor | list[torch.Tensor]
    labels: list[torch.Tensor] | None = None
    masks: list[Mask] | None = None
    bboxes: list[BoundingBoxes] | None = None
    keypoints: list[torch.Tensor] | None = None
    polygons: list[np.ndarray] | None = None
    imgs_info: Sequence[ImageInfo | None] | None = None  # TODO(ashwinvaidya17): revisit

    def __post_init__(self) -> None:
        """Validate the batch after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the batch fields."""
        self._validate_images(self.images)
        if self.labels is not None:
            self._validate_labels(self.labels)
        if self.bboxes is not None:
            self._validate_bboxes(self.bboxes)
        if self.keypoints is not None:
            self._validate_keypoints(self.keypoints)
        if self.masks is not None:
            self._validate_masks(self.masks)
        if self.polygons is not None:
            self._validate_polygons(self.polygons)
        if self.imgs_info is not None:
            self._validate_imgs_info(self.imgs_info)
        self._validate_batch_size(self.batch_size)

    @staticmethod
    def _validate_images(image_batch: torch.Tensor | list[torch.Tensor]) -> None:
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

    @staticmethod
    def _validate_labels(label_batch: list[torch.Tensor | None]) -> None:
        """Validate the label batch."""
        if all(label is None for label in label_batch):
            return
        first_non_none = next((label for label in label_batch if label is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, torch.Tensor):
            msg = f"Label batch must be a list of torch tensors. Got {type(first_non_none)}"
            raise TypeError(msg)
        if first_non_none.dtype != torch.long:
            msg = "Label batch must have dtype torch.long"
            raise ValueError(msg)
        if first_non_none.ndim > 2:
            msg = f"Label batch must have shape of (N, 1) or (N,), but got {first_non_none.shape}"
            raise ValueError(msg)

    @staticmethod
    def _validate_bboxes(boxes_batch: list[BoundingBoxes | None]) -> None:
        """Validate the bboxes batch."""
        if all(box is None for box in boxes_batch):
            return
        first_non_none = next((box for box in boxes_batch if box is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, torch.Tensor):
            msg = f"Boxes batch must be a list of torch tensors. Got {type(first_non_none)}"
            raise TypeError(msg)
        if not first_non_none.dtype.is_floating_point:
            msg = f"Boxes batch must have a floating point dtype. Got {first_non_none.dtype}"
            raise ValueError(msg)
        if first_non_none.ndim != 2:
            msg = "Boxes batch must have 2 dimensions"
            raise ValueError(msg)
        if first_non_none.shape[1] != 4:
            msg = "Boxes batch must have 4 coordinates"
            raise ValueError(msg)

    @staticmethod
    def _validate_keypoints(keypoints_batch: list[torch.Tensor | None]) -> None:
        """Validate the keypoints batch."""
        if all(keypoints is None for keypoints in keypoints_batch):
            return
        first_non_none = next((kp for kp in keypoints_batch if kp is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, torch.Tensor):
            msg = f"Keypoints batch must be a list of torch tensors. Got {type(first_non_none)}"
            raise TypeError(msg)
        if first_non_none.dtype != torch.float32:
            msg = "Keypoints batch must have dtype torch.float32"
            raise ValueError(msg)
        if first_non_none.ndim != 2:
            msg = "Keypoints batch must have 2 dimensions"
            raise ValueError(msg)
        if first_non_none.shape[1] != 3:
            msg = "Keypoints batch must have 2 coordinates and 1 visibility value"
            raise ValueError(msg)
        if any(first_non_none[:, 2] > 1) or any(first_non_none[:, 2] < 0):
            msg = "Keypoints visibility must be between 0 and 1"
            raise ValueError(msg)

    @staticmethod
    def _validate_masks(masks_batch: list[Mask | None]) -> None:
        """Validate the masks batch."""
        if all(mask is None for mask in masks_batch):
            return
        first_non_none = next((mask for mask in masks_batch if mask is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, torch.Tensor):
            msg = f"Masks batch must be a list of torch tensors. Got {type(first_non_none)}"
            raise TypeError(msg)

    @staticmethod
    def _validate_polygons(polygons_batch: list[np.ndarray | None]) -> None:
        """Validate the polygons batch."""
        if all(polygon is None for polygon in polygons_batch):
            return
        first_non_none = next((poly for poly in polygons_batch if poly is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, np.ndarray):
            msg = "Polygons batch must be a list of np.ndarray of np.ndarray"
            raise TypeError(msg)
        if len(first_non_none) == 0:
            msg = f"Polygons batch must not be empty. Got {polygons_batch}"
            raise ValueError(msg)
        if not isinstance(first_non_none[0], np.ndarray):
            msg = "Polygons batch must be a list of np.ndarray of np.ndarray"
            raise TypeError(msg)

    @staticmethod
    def _validate_imgs_info(imgs_info_batch: Sequence[ImageInfo | None]) -> None:
        """Validate the image info batch."""
        if all(img_info is None for img_info in imgs_info_batch):
            return
        first_non_none = next((info for info in imgs_info_batch if info is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, ImageInfo):
            msg = "Image info batch must be a list of otx.data.entity.ImageInfo"
            raise TypeError(msg)

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        """Validate the batch size."""
        if not isinstance(batch_size, int):
            msg = "Batch size must be an integer"
            raise TypeError(msg)

    def pin_memory(self) -> OTXSampleBatch:
        """Pin memory for member tensor variables."""
        # https://github.com/pytorch/pytorch/issues/116403

        kwargs = {}

        def maybe_pin(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, torch.Tensor):
                return x.pin_memory()
            return x

        def maybe_wrap_tv(x: Any) -> Any:  # noqa: ANN401
            if isinstance(x, tv_tensors.TVTensor):
                return tv_tensors.wrap(x.pin_memory(), like=x)
            return maybe_pin(x)

        # Handle images separately because of tv_tensors wrapping
        if self.images is not None:
            if isinstance(self.images, list):
                kwargs["images"] = [maybe_wrap_tv(img) for img in self.images]
            else:
                kwargs["images"] = maybe_wrap_tv(self.images)

        # Generic handler for all other fields
        for field in ["labels", "bboxes", "keypoints", "masks"]:
            value = getattr(self, field)
            if value is not None:
                kwargs[field] = [maybe_wrap_tv(v) if v is not None else None for v in value]

        return self.wrap(**kwargs)

    def wrap(self, **kwargs) -> OTXSampleBatch:
        """Wrap this dataclass with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be overwritten on top of this dataclass

        Returns:
            Updated dataclass
        """
        updated_kwargs = asdict(self)
        updated_kwargs.update(**kwargs)
        return self.__class__(**updated_kwargs)


@dataclass
class OTXPredictionBatch(OTXSampleBatch):
    """OTX prediction batch implementation.

    Extends OTXSampleBatch with prediction-specific fields.

    Attributes:
        scores: List of score tensors, optional.
        feature_vector: List of feature vector tensors, optional.
        saliency_map: List of saliency map tensors, optional.
    """

    scores: list[torch.Tensor] | None = None
    feature_vector: list[torch.Tensor] | None = None
    saliency_map: list[torch.Tensor] | None = None

    def __post_init__(self) -> None:
        """Validate the prediction batch after initialization."""
        super().__post_init__()
        if self.scores is not None:
            self._validate_scores(self.scores)
        if self.feature_vector is not None:
            self._validate_feature_vectors(self.feature_vector)

    @staticmethod
    def _validate_scores(scores_batch: list[torch.Tensor | None]) -> None:
        """Validate the scores batch."""
        if all(score is None for score in scores_batch):
            return
        first_non_none = next((score for score in scores_batch if score is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, torch.Tensor):
            msg = f"Scores batch must be a list of torch tensors. Got {type(first_non_none)}"
            raise TypeError(msg)
        if not first_non_none.dtype.is_floating_point:
            msg = f"Scores batch must have a floating point dtype. Got {first_non_none.dtype}"
            raise ValueError(msg)
        if first_non_none.ndim > 1:
            msg = "Scores batch must have 1 or 2 dimensions"
            raise ValueError(msg)

    @staticmethod
    def _validate_feature_vectors(
        feature_vector_batch: list[torch.Tensor | np.ndarray | None],
    ) -> None:
        """Validate the feature vector batch.

        Numpy is mixed for this round as it is used in OV Classification.
        """
        first_non_none = next((fv for fv in feature_vector_batch if fv is not None), None)
        if first_non_none is None:
            return
        if not isinstance(first_non_none, (torch.Tensor, np.ndarray)):
            msg = f"Feature vector batch must be a list of torch tensors or numpy arrays. Got {type(first_non_none)}"
            raise TypeError(msg)
        if isinstance(first_non_none, torch.Tensor) and not first_non_none.dtype.is_floating_point:
            msg = f"Feature vector must have a floating point dtype. Got {first_non_none.dtype}"
            raise ValueError(msg)
        if isinstance(first_non_none, np.ndarray) and first_non_none.dtype.kind != "f":
            msg = f"Feature vector must have a floating point dtype. Got {first_non_none.dtype}"
            raise ValueError(msg)
        if isinstance(first_non_none, torch.Tensor) and first_non_none.ndim != 2:
            msg = "Feature vector must have 2 dimensions"
            raise ValueError(msg)

    @property
    def has_xai_outputs(self) -> bool:
        """Check if the batch has XAI outputs.

        Necessary for compatibility with tests.
        """
        # TODO(ashwinvaidya17): the tests should directly refer to saliency map.
        return self.saliency_map is not None and len(self.saliency_map) > 0


@dataclass
class OTXPrediction:
    """OTX prediction data entity for a single sample.

    This is used for storing individual prediction results, e.g., after tile merging.

    Attributes:
        image: The image tensor.
        img_info: Image metadata information.
        label: The predicted label tensor, optional.
        masks: The predicted masks, optional.
        bboxes: The predicted bounding boxes, optional.
        keypoints: The predicted keypoints, optional.
        polygons: The predicted polygons, optional.
        scores: The prediction scores, optional.
        feature_vector: The feature vector for XAI, optional.
        saliency_map: The saliency map for XAI, optional.
    """

    image: torch.Tensor | np.ndarray
    img_info: ImageInfo | None = None
    label: torch.Tensor | None = None
    masks: Mask | None = None
    bboxes: BoundingBoxes | None = None
    keypoints: torch.Tensor | None = None
    polygons: np.ndarray | None = None
    scores: torch.Tensor | None = None
    feature_vector: torch.Tensor | None = None
    saliency_map: torch.Tensor | None = None
