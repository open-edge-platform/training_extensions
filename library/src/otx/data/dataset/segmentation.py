# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import SegLabelInfo
from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.entity.sample import SegmentationSample
from otx.types import OTXTaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXSegmentationDataset(OTXDataset):
    """OTX Dataset for semantic segmentation tasks.

    This dataset handles semantic segmentation where each pixel in an image is classified
    into one of multiple classes. It processes Datumaro dataset items and converts them
    into OTXDataItem format suitable for semantic segmentation training and inference.

    Args:
        dm_subset: Datumaro dataset subset containing the data items.
        transforms: Transform operations to apply to the data items.
        max_refetch: Maximum number of retries when fetching a data item fails.
        image_color_channel: Color channel format for images (RGB, BGR, etc.).
        stack_images: Whether to stack images in batch processing.
        to_tv_image: Whether to convert images to torchvision format.
        data_format: Format of the source data (e.g., "cityscapes", "pascal_voc").
        ignore_index: Index value for pixels to be ignored during training.

    Attributes:
        ignore_index: Index value for pixels to be ignored during training.

    Example:
        >>> from otx.data.dataset.segmentation import OTXSegmentationDataset
        >>> dataset = OTXSegmentationDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ...     ignore_index=255
        ... )
        >>> item = dataset[0]  # Get first item with segmentation masks
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        to_tv_image: bool = True,
        ignore_index: int = 255,
        data_format: str = "",
    ) -> None:
        sample_type = SegmentationSample
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
            to_tv_image=to_tv_image,
            data_format=data_format,
            sample_type=sample_type,
        )

        labels = list(dm_subset.schema.attributes["masks"].categories.labels)
        self.label_info = SegLabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The semantic segmentation task type.
        """
        return OTXTaskType.SEMANTIC_SEGMENTATION
