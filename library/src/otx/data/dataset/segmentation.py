# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import SegLabelInfo
from otx.data.dataset.base import OTXDataset, Transforms
from otx.data.entity.sample import SegmentationSample

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXSegmentationDataset(OTXDataset):
    """OTXDataset class for segmentation task."""

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
            transforms=transforms,  # type: ignore[arg-type]
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
