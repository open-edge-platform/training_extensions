# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXDetectionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import OTXTaskType
from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.dataset.mixins import DataAugSwitchMixin
from otx.data.entity.sample import DetectionSample
from otx.types.label import LabelInfo

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXDetectionDataset(OTXDataset, DataAugSwitchMixin):
    """OTX Dataset for object detection tasks.

    This dataset handles object detection where each image contains multiple objects with
    bounding box annotations. It processes Datumaro dataset items and converts them into
    OTXDataItem format suitable for object detection training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms | None, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        stack_images (bool): Whether to stack images in batch processing.
        to_tv_image (bool): Whether to convert images to torchvision format.
        data_format (str): Format of the source data (e.g., "coco", "pascal_voc").

    Example:
        >>> from otx.data.dataset.detection import OTXDetectionDataset
        >>> dataset = OTXDetectionDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item with bounding boxes
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
        sample_type = DetectionSample
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

        labels = list(dm_subset.schema.attributes["label"].categories.labels)
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int | str, list[int]] = {}
        for idx in range(len(self)):
            item = self.dm_subset[idx]
            labels = item.label.tolist()
            if use_string_label:
                labels = [self.label_info.label_names[label] for label in labels]
            for label in labels:
                if label not in idx_list_per_classes:
                    idx_list_per_classes[label] = []
                idx_list_per_classes[label].append(idx)
        return idx_list_per_classes

    def _apply_transforms(self, entity: DetectionSample) -> DetectionSample | None:
        if self.has_dynamic_augmentation:
            self._apply_augmentation_switch()
        return super()._apply_transforms(entity)

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The object detection task type.
        """
        return OTXTaskType.DETECTION
