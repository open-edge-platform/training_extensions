# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import AnomalyLabelInfo
from otx.data.dataset.base_new import OTXDataset
from otx.data.entity.sample import AnomalySample

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXAnomalyDataset(OTXDataset):
    """OTXDataset class for anomaly task."""

    def __init__(self, dm_subset: Dataset, **kwargs) -> None:
        sample_type = AnomalySample
        super().__init__(dm_subset=dm_subset, sample_type=sample_type, **kwargs)
        dm_subset = dm_subset.convert_to_schema(sample_type)

        self.label_info = AnomalyLabelInfo()
