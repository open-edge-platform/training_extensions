# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import SegLabelInfo
from otx.data.dataset.base_new import OTXDataset
from otx.data.entity.sample import SegmentationSample

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXSegmentationDataset(OTXDataset):
    """OTXDataset class for segmentation task."""

    def __init__(self, dm_subset: Dataset, **kwargs) -> None:
        sample_type = SegmentationSample
        super().__init__(dm_subset=dm_subset, sample_type=sample_type, **kwargs)
        dm_subset = dm_subset.convert_to_schema(sample_type)

        labels = dm_subset.schema.attributes["masks"].categories.labels
        self.label_info = SegLabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )
