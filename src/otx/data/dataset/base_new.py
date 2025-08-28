# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for OTXDataset using new Datumaro experimental Dataset."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Sequence, Union

import numpy as np
import torch
from datumaro.experimental import Dataset
from torch.utils.data import Dataset as TorchDataset
from torchvision import tv_tensors

from otx.data.entity.sample import ClassificationSample
from otx.data.entity.torch.validations import ValidateBatchMixin
from otx.data.transform_libs.torchvision import Compose
from otx.types.image import ImageColorChannel

if TYPE_CHECKING:
    from datumaro import Polygon
    from torchvision.tv_tensors import BoundingBoxes, Mask

    from otx.data.entity.base import ImageInfo

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]


@dataclass
class ClassificationBatch(ValidateBatchMixin):
    """Classification batch implementation based on ClassificationSample."""

    batch_size: int
    images: torch.Tensor | list[torch.Tensor]
    labels: list[torch.Tensor] | None = None
    masks: list[Mask] | None = None
    bboxes: list[BoundingBoxes] | None = None
    keypoints: list[torch.Tensor] | None = None
    polygons: list[list[Polygon]] | None = None
    imgs_info: Sequence[ImageInfo | None] | None = None

    def pin_memory(self) -> "ClassificationBatch":
        """Pin memory for member tensor variables."""
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

    def wrap(self, **kwargs) -> "ClassificationBatch":
        """Wrap this dataclass with the given keyword arguments.

        Args:
            **kwargs: Keyword arguments to be overwritten on top of this dataclass
        Returns:
            Updated dataclass
        """
        updated_kwargs = asdict(self)
        updated_kwargs.update(**kwargs)
        return self.__class__(**updated_kwargs)


class OTXDataset(TorchDataset):
    """Base OTXDataset using new Datumaro experimental Dataset.

    Defines basic logic for OTX datasets.

    Args:
        transforms: Transforms to apply on images
        image_color_channel: Color channel of images
        stack_images: Whether or not to stack images in collate function in OTXBatchData entity.
        sample_type: Type of sample to use for this dataset
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms,
        max_refetch: int = 1000,
        image_color_channel: ImageColorChannel = ImageColorChannel.RGB,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
        sample_type: type[ClassificationSample] = ClassificationSample,
    ) -> None:
        self.transforms = transforms
        self.image_color_channel = image_color_channel
        self.stack_images = stack_images
        self.to_tv_image = to_tv_image
        self.sample_type = sample_type
        self.max_refetch = max_refetch
        self.data_format = data_format

        # TODO: Properly reinit label_info
        self.label_info = dm_subset.label_group

        self.dataset = dm_subset

    def __len__(self) -> int:
        return len(self.dataset)

    def _sample_another_idx(self) -> int:
        return np.random.randint(0, len(self))

    def _apply_transforms(self, entity: ClassificationSample) -> ClassificationSample | None:
        if isinstance(self.transforms, Compose):
            if self.to_tv_image:
                entity.as_tv_image()
            return self.transforms(entity)
        if isinstance(self.transforms, Iterable):
            return self._iterable_transforms(entity)
        if callable(self.transforms):
            return self.transforms(entity)

    def _iterable_transforms(self, item: ClassificationSample) -> ClassificationSample | None:
        if not isinstance(self.transforms, list):
            raise TypeError(item)

        results = item
        for transform in self.transforms:
            results = transform(results)
            # MMCV transform can produce None. Please see
            # https://github.com/open-mmlab/mmengine/blob/26f22ed283ae4ac3a24b756809e5961efe6f9da8/mmengine/dataset/base_dataset.py#L59-L66
            if results is None:
                return None

        return results

    def __getitem__(self, index: int) -> ClassificationSample:
        for _ in range(self.max_refetch):
            results = self._get_item_impl(index)

            if results is not None:
                return results

            index = self._sample_another_idx()

        msg = f"Reach the maximum refetch number ({self.max_refetch})"
        raise RuntimeError(msg)

    def _get_item_impl(self, index: int) -> ClassificationSample | None:
        dm_item = self.dataset[index]
        sample = self.sample_type.from_dm_item(dm_item)
        return self._apply_transforms(sample)

    @property
    def collate_fn(self) -> Callable:
        """Collection function to collect samples into a batch in data loader."""
        def _collate_fn(items: list[ClassificationSample]) -> ClassificationBatch:
            """Collate ClassificationSample items into a ClassificationBatch.
            
            Args:
                items: List of ClassificationSample items to batch
            Returns:
                Batched ClassificationSample items with stacked tensors
            """
            # Check if all images have the same size for stacking
            if all(item.image.shape == items[0].image.shape for item in items):
                images = torch.stack([item.image for item in items])
            else:
                # Keep as list if shapes differ (e.g., for OV inference)
                images = [item.image for item in items]

            return ClassificationBatch(
                batch_size=len(items),
                images=images,
                labels=[item.label for item in items] if items[0].label is not None else None,
                masks=[item.masks for item in items] if any(item.masks is not None for item in items) else None,
                bboxes=[item.bboxes for item in items] if any(item.bboxes is not None for item in items) else None,
                keypoints=[item.keypoints for item in items] if any(item.keypoints is not None for item in items) else None,
                polygons=[item.polygons for item in items] if any(item.polygons is not None for item in items) else None,
                imgs_info=[item.img_info for item in items] if any(item.img_info is not None for item in items) else None,
            )
        
        return _collate_fn
        