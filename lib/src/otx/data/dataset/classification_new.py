# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXClassificationDatasets using new Datumaro experimental Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import LabelInfo
from otx.data.dataset.base_new import OTXDataset
from otx.data.entity.sample import ClassificationSample

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXMulticlassClsDataset(OTXDataset):
    """OTXDataset class for multi-class classification task using new Datumaro experimental Dataset."""

    def __init__(self, dm_subset: Dataset, **kwargs) -> None:
        """Initialize OTXMulticlassClsDataset.

        Args:
            **kwargs: Keyword arguments to pass to OTXDataset
        """
        super().__init__(dm_subset=dm_subset, sample_type=ClassificationSample, **kwargs)

        labels = dm_subset.schema.attributes["label"].categories.labels
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int, list[int]] = {}
        for idx in range(len(self)):
            item = self.dm_subset[idx]
            label_id = item.label.item()
            if use_string_label:
                label_id = self.label_info.label_names[label_id]
            if label_id not in idx_list_per_classes:
                idx_list_per_classes[label_id] = []
            idx_list_per_classes[label_id].append(idx)
        return idx_list_per_classes
