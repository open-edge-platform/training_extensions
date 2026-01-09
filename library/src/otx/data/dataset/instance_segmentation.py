# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXInstanceSegDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import LabelInfo
from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.entity.sample import InstanceSegmentationSample
from otx.types import OTXTaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXInstanceSegDataset(OTXDataset):
    """Dataset class for instance segmentation tasks in OTX.

    This class handles loading images and their masks.

    Args:
        dm_subset (DmDataset): The subset of the dataset to use.
        transforms (Transforms, optional): Data transformations to be applied.
        task_type (OTXTaskType, optional): The task type. Defaults to INSTANCE_SEGMENTATION.
        max_refetch (int, optional): Maximum number of times to refetch data. Defaults to 1000.
        stack_images (bool, optional): Whether to stack images. Defaults to True.
        to_tv_image (bool, optional): Whether to convert images to torchvision format. Defaults to True.
        data_format (str, optional): Data format string. Defaults to "".
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        task_type: OTXTaskType = OTXTaskType.INSTANCE_SEGMENTATION,
        max_refetch: int = 1000,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        sample_type = InstanceSegmentationSample
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
        self._task_type = task_type

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

    @property
    def task_type(self) -> OTXTaskType:
        """OTX Task Type for the dataset.

        Returns:
            OTXTaskType: The instance segmentation task type.
        """
        return self._task_type
