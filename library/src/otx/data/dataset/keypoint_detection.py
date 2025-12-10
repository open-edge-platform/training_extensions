# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXKeypointDetectionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Union

import torch
from torchvision.transforms.v2.functional import to_dtype, to_image

from otx.data.entity.sample import KeypointSample
from otx.data.transform_libs.torchvision import Compose
from otx.types import OTXTaskType
from otx.types.label import LabelInfo

from .base import OTXDataset

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXKeypointDetectionDataset(OTXDataset):
    """OTX Dataset for keypoint detection tasks.

    This dataset handles keypoint detection where specific key points (like body joints)
    are detected and localized in images. It processes Datumaro dataset items and
    converts them into OTXDataItem format suitable for keypoint detection training
    and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms | None, optional): Transform operations to apply to the data items.
        max_refetch (int, optional): Maximum number of retries when fetching a data item fails.
        image_color_channel (ImageColorChannel, optional): Color channel format for images (RGB, BGR, etc.).
        stack_images (bool, optional): Whether to stack images in batch processing.
        to_tv_image (bool, optional): Whether to convert images to torchvision format.
        data_format (str, optional): Format of the source data (e.g., "coco", "arrow").

    Example:
        >>> from otx.data.dataset.keypoint_detection import OTXKeypointDetectionDataset
        >>> dataset = OTXKeypointDetectionDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ...     data_format="coco"
        ... )
        >>> item = dataset[0]  # Get first item with keypoints
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        sample_type = KeypointSample
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            sample_type=sample_type,
            transforms=transforms,
            max_refetch=max_refetch,
            stack_images=stack_images,
            to_tv_image=to_tv_image,
            data_format=data_format,
        )
        labels = dm_subset.schema.attributes["label"].categories.labels
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def _get_item_impl(self, index: int) -> KeypointSample | None:
        item = self.dm_subset[index]
        keypoints = item.keypoints
        keypoints[:, 2] = torch.clamp(keypoints[:, 2], max=1)  # OTX represents visibility as 0 or 1
        item.keypoints = keypoints
        # Handle image conversion - to_image only permutes numpy arrays, not tensors
        image = item.image
        if isinstance(image, torch.Tensor) and image.ndim == 3 and image.shape[-1] in (1, 3):
            # Image is in HWC format, convert to CHW
            image = image.permute(2, 0, 1)
        item.image = to_dtype(to_image(image), torch.float32)
        return self._apply_transforms(item)  # type: ignore[return-value]

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The keypoint detection task type.
        """
        return OTXTaskType.KEYPOINT_DETECTION
