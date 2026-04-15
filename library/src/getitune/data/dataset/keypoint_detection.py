# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXKeypointDetectionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from getitune.data.dataset.base import OTXDataset, Transforms
from getitune.data.entity.sample import KeypointSample
from getitune.data.entity.utils import with_image_dtype
from getitune.types import OTXTaskType
from getitune.types.label import LabelInfo

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXKeypointDetectionDataset(OTXDataset):
    """Geti Tune Dataset for keypoint detection tasks.

    This dataset handles keypoint detection where specific key points (like body joints)
    are detected and localized in images. It processes Datumaro dataset items and
    converts them into OTXSample format suitable for keypoint detection training
    and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms | None, optional): Transform operations to apply to the data items.
        max_refetch (int, optional): Maximum number of retries when fetching a data item fails.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".


    Example:
        >>> from getitune.data.dataset.keypoint_detection import OTXKeypointDetectionDataset
        >>> dataset = OTXKeypointDetectionDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item with keypoints
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(KeypointSample, storage_dtype)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
        )
        labels = dm_subset.schema.attributes["keypoints"].categories.labels
        self.label_info = LabelInfo(
            label_names=list(labels),
            label_groups=[],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def _get_item_impl(self, index: int) -> KeypointSample | None:
        item = self._read_dm_item(index)
        item.keypoints[:, 2] = torch.clamp(item.keypoints[:, 2], max=1)  # Geti Tune represents visibility as 0 or 1
        return self._apply_transforms(item)  # type: ignore[return-value]

    @property
    def task_type(self) -> OTXTaskType:
        """Geti Tune Task Type for the dataset.

        Returns:
            OTXTaskType: The keypoint detection task type.
        """
        return OTXTaskType.KEYPOINT_DETECTION
