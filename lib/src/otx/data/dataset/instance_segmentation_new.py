# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXInstanceSegDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from otx import LabelInfo
from otx.data.dataset.base_new import OTXDataset
from otx.data.entity.sample import InstanceSegmentationSample

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXInstanceSegDataset(OTXDataset):
    """OTXDataset class for instance segmentation task."""

    def __init__(self, dm_subset: Dataset, **kwargs) -> None:
        sample_type = InstanceSegmentationSample
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(dm_subset=dm_subset, sample_type=sample_type, **kwargs)

        labels = dm_subset.schema.attributes["label"].categories.labels
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int, list[int]]:
        """Get a dictionary with class labels as keys and lists of corresponding sample indices as values."""
        idx_list_per_classes: dict[int, list[int]] = {}

        for idx in range(len(self.dm_subset)):
            sample = self.dm_subset[idx]
            labels = sample.labels

            # Handle multiple labels per image (instance segmentation can have multiple instances)
            if hasattr(labels, "__iter__") and not isinstance(labels, str):
                unique_labels = set(labels)
                for label in unique_labels:
                    label_key = label if use_string_label else int(label)
                    if label_key not in idx_list_per_classes:
                        idx_list_per_classes[label_key] = []
                    idx_list_per_classes[label_key].append(idx)
            else:
                # Single label case
                label_key = labels if use_string_label else int(labels)
                if label_key not in idx_list_per_classes:
                    idx_list_per_classes[label_key] = []
                idx_list_per_classes[label_key].append(idx)

        return idx_list_per_classes
