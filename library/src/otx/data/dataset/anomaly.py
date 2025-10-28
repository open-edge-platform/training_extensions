# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datumaro.experimental.categories import LabelCategories, LabelSemantic

from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.entity.sample import AnomalySample
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXAnomalyDataset(OTXDataset):
    """Dataset class for anomaly classification tasks in OTX.

    Handles loading images and corresponding masks for anomaly detection/classification,
    supporting both file-based and in-memory (bytes) images. Provides label mapping,
    mask extraction (from file, polygons, ellipses, or bounding boxes), and applies
    transformations as needed for model training or inference.

    Args:
        task_type (OTXTaskType): The type of anomaly task (e.g., classification, detection).
        dm_subset (DmDataset): Datumaro dataset subset containing the data.
        transforms (Transforms, optional): Transformations to apply to the data.
        max_refetch (int, optional): Maximum number of times to refetch data if needed. Defaults to 1000.
        stack_images (bool, optional): Whether to stack images. Defaults to True.
        to_tv_image (bool, optional): Whether to convert images to TorchVision format. Defaults to True.
        data_format (str, optional): Data format string. Defaults to "".
    """

    def __init__(
        self,
        task_type: OTXTaskType,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        stack_images: bool = True,
        to_tv_image: bool = True,
        data_format: str = "",
    ) -> None:
        self._task_type = task_type
        sample_type = AnomalySample
        categories = {
            "label": LabelCategories(
                labels=("normal", "anomalous"),
                label_semantics={LabelSemantic.NORMAL: "normal", LabelSemantic.ANOMALOUS: "anomalous"},
            )
        }
        dm_subset = dm_subset.convert_to_schema(sample_type, target_categories=categories)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
            stack_images=stack_images,
            to_tv_image=to_tv_image,
            data_format=data_format,
        )
        self.label_info = AnomalyLabelInfo()

    @property
    def task_type(self) -> OTXTaskType | None:
        """OTX Task Type for the dataset. Can be None if no task is defined."""
        return self._task_type
