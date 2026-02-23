# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Sample classes for OTX data entities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Sequence

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
from otx.data.entity.validation import (
    validate_bboxes,
    validate_feature_vectors,
    validate_images,
    validate_imgs_info,
    validate_keypoints,
    validate_labels,
    validate_masks,
    validate_scores,
)

if TYPE_CHECKING:
    from torchvision.tv_tensors import BoundingBoxes, Mask


# ---------------------------------------------------------------------------
# Dtype helpers for 16-bit image support
# ---------------------------------------------------------------------------

#: Map from IntensityConfig.storage_dtype strings to Polars dtype instances.
STORAGE_DTYPE_MAP: dict[str, pl.DataType] = {
    "uint8": pl.UInt8(),
    "uint16": pl.UInt16(),
    "int16": pl.Int16(),
    "float32": pl.Float32(),
}


def with_image_dtype(
    sample_cls: type[Sample],
    storage_dtype: str,
) -> type[Sample]:
    """Create a variant of *sample_cls* whose ``image`` field uses *storage_dtype*.

    When ``storage_dtype == "uint8"`` (the default) the original class is
    returned unchanged — zero overhead for the common case.

    For other dtypes a thin **dynamic subclass** is created that overrides the
    ``image`` class-variable with the requested Polars dtype.  The subclass is
    cached so repeated calls with the same arguments return the same class
    object (important for Datumaro schema identity comparisons).

    Args:
        sample_cls: One of the concrete sample classes (e.g.
            :class:`ClassificationSample`, :class:`DetectionSample`).
        storage_dtype: A key from :data:`STORAGE_DTYPE_MAP` — ``"uint8"``,
            ``"uint16"``, ``"int16"``, or ``"float32"``.

    Returns:
        Either *sample_cls* itself (uint8) or a dynamically created subclass
        with the overridden ``image`` field.
    """
    if storage_dtype == "uint8":
        return sample_cls

    pl_dtype = STORAGE_DTYPE_MAP.get(storage_dtype)
    if pl_dtype is None:
        msg = f"Unsupported storage_dtype={storage_dtype!r}. Supported values: {list(STORAGE_DTYPE_MAP)}"
        raise ValueError(msg)

    cache_key = (sample_cls, storage_dtype)
    if cache_key in _SAMPLE_DTYPE_CACHE:
        return _SAMPLE_DTYPE_CACHE[cache_key]

    # Detect the original image_field kwargs by inspecting the existing field
    orig_image_field = sample_cls.__dataclass_fields__["image"]  # type: ignore[attr-defined]
    orig_default = orig_image_field.default
    # The original default is an ImageField dataclass produced by image_field().
    # We need to reconstruct it with the new dtype.
    channels_first = getattr(orig_default, "channels_first", True)
    fmt = getattr(orig_default, "format", "RGB")

    new_image_default = image_field(dtype=pl_dtype, channels_first=channels_first, format=fmt)

    # Create a new dataclass that inherits from sample_cls with overridden image field
    new_cls_name = f"{sample_cls.__name__}_{storage_dtype}"

    # We use dataclass inheritance: the new class just overrides the image field
    new_cls = dataclass(
        type(
            new_cls_name,
            (sample_cls,),
            {
                "__annotations__": {"image": tv_tensors.Image | torch.Tensor},
                "image": new_image_default,
            },
        )
    )

    # Register with pytree so torchvision v2 transforms work
    register_pytree_node(new_cls)

    _SAMPLE_DTYPE_CACHE[cache_key] = new_cls
    return new_cls


#: Cache for dynamically created sample classes to avoid re-creation
_SAMPLE_DTYPE_CACHE: dict[tuple[type, str], type[Sample]] = {}


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

    image: torch.Tensor | tv_tensors.Image
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

    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
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

    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8(), multi_label=True)
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

    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8(), is_list=True)
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

    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), format="RGB", channels_first=True)
    label: torch.Tensor = label_field(pl.UInt8(), is_list=True)
    # Use Union type to allow torch.Tensor from Polars (since tv_tensors.BoundingBoxes
    # conversion is not supported in Datumaro), then convert in __post_init__
    bboxes: tv_tensors.BoundingBoxes | torch.Tensor = bbox_field(dtype=pl.Float32())
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)
        # Ensure bboxes are tv_tensors.BoundingBoxes
        if not isinstance(self.bboxes, tv_tensors.BoundingBoxes):
            # If it's a plain tensor, wrap it
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
    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=False)
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
    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    # Use Union type to allow torch.Tensor from Polars (since tv_tensors.BoundingBoxes
    # conversion is not supported in Datumaro), then convert in __post_init__
    bboxes: tv_tensors.BoundingBoxes | torch.Tensor = bbox_field(dtype=pl.Float32())
    masks: tv_tensors.Mask = instance_mask_field(dtype=pl.UInt8())
    label: torch.Tensor = label_field(dtype=pl.UInt8(), is_list=True)
    dm_image_info: DmImageInfo = image_info_field()

    def __post_init__(self) -> None:
        shape = (self.dm_image_info.height, self.dm_image_info.width)

        # Ensure bboxes are tv_tensors.BoundingBoxes
        if not isinstance(self.bboxes, tv_tensors.BoundingBoxes):
            # If it's a plain tensor, wrap it
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
    image: tv_tensors.Image | torch.Tensor = image_field(dtype=pl.UInt8(), channels_first=True)
    label: torch.Tensor = label_field(dtype=pl.UInt8(), is_list=True)
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
    images = torch.stack([sample.image for sample in samples])

    # Collect optional fields - use getattr to handle samples without these attributes
    labels = [sample.label for sample in samples] if hasattr(samples[0], "label") else None
    bboxes = [sample.bboxes for sample in samples] if hasattr(samples[0], "bboxes") else None
    keypoints = [sample.keypoints for sample in samples] if hasattr(samples[0], "keypoints") else None
    masks = [sample.masks for sample in samples] if hasattr(samples[0], "masks") else None

    return OTXSampleBatch(
        images=images,
        labels=labels,
        bboxes=bboxes,
        keypoints=keypoints,
        masks=masks,
        imgs_info=[sample.img_info for sample in samples],
    )


@dataclass
class OTXSampleBatch:
    """OTX sample batch implementation.

    Attributes:
        images: The batch of images as a BCHW tensor.
        labels: List of label tensors, optional.
        masks: List of masks, optional.
        bboxes: List of bounding boxes, optional.
        keypoints: List of keypoint tensors, optional.
        imgs_info: Sequence of image information, optional.
    """

    images: torch.Tensor
    labels: list[torch.Tensor] | None = None
    masks: list[Mask] | None = None
    bboxes: list[BoundingBoxes] | None = None
    keypoints: list[torch.Tensor] | None = None
    imgs_info: Sequence[ImageInfo | None] | None = None

    @property
    def batch_size(self) -> int:
        """Get the number of samples in the batch."""
        return self.images.shape[0]

    def __post_init__(self) -> None:
        """Validate the batch after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the batch fields."""
        validate_images(self.images)
        if self.labels is not None:
            validate_labels(self.labels)
        if self.bboxes is not None:
            validate_bboxes(self.bboxes)
        if self.keypoints is not None:
            validate_keypoints(self.keypoints)
        if self.masks is not None:
            validate_masks(self.masks)
        if self.imgs_info is not None:
            validate_imgs_info(self.imgs_info)

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
        updated = object.__new__(self.__class__)
        updated.__dict__.update(asdict(self))
        updated.__dict__.update(kwargs)
        return updated


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
            validate_scores(self.scores)
        if self.feature_vector is not None:
            validate_feature_vectors(self.feature_vector)


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
        scores: The prediction scores, optional.
        feature_vector: The feature vector for XAI, optional.
        saliency_map: The saliency map for XAI, optional.
    """

    image: torch.Tensor
    img_info: ImageInfo | None = None
    label: torch.Tensor | None = None
    masks: Mask | None = None
    bboxes: BoundingBoxes | None = None
    keypoints: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    feature_vector: torch.Tensor | None = None
    saliency_map: torch.Tensor | None = None
