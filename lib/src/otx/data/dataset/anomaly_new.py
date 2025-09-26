# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXSegmentationDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datumaro.experimental.categories import LabelCategories, LabelSemantic

from otx.data.dataset.base_new import OTXDataset
from otx.data.entity.sample import AnomalySample
from otx.types.label import AnomalyLabelInfo
from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXAnomalyDataset(OTXDataset):
    """OTXDataset class for anomaly task."""

    def __init__(self, task_type: OTXTaskType, dm_subset: Dataset, **kwargs) -> None:
        self.task_type = task_type
        sample_type = AnomalySample
        categories = {
            "label": LabelCategories(
                labels=["normal", "anomalous"],
                label_semantics={LabelSemantic.NORMAL: "normal", LabelSemantic.ANOMALOUS: "anomalous"},
            )
        }
        dm_subset = dm_subset.convert_to_schema(sample_type, target_categories=categories)
        super().__init__(dm_subset=dm_subset, sample_type=sample_type, **kwargs)

        self.label_info = AnomalyLabelInfo()
