# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import SegLabelInfo
from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.entity.sample import SegmentationSample
from otx.data.entity.utils import with_image_dtype
from otx.types import OTXTaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXSegmentationDataset(OTXDataset):
    """OTX Dataset for semantic segmentation tasks.

    This dataset handles semantic segmentation where each pixel in an image is classified
    into one of multiple classes. It processes Datumaro dataset items and converts them
    into OTXSample format suitable for semantic segmentation training and inference.

    Args:
        dm_subset: Datumaro dataset subset containing the data items.
        transforms: Transform operations to apply to the data items.
        max_refetch: Maximum number of retries when fetching a data item fails.
        ignore_index: Index value for pixels to be ignored during training.
        data_format: Source data format (e.g. "coco", "voc"). Defaults to "".
        storage_dtype: Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".

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
        ignore_index: int = 255,
        data_format: str = "",
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(SegmentationSample, storage_dtype)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
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
