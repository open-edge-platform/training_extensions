# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXClassificationDatasets using new Datumaro experimental Dataset."""

from __future__ import annotations

from otx.data.dataset.base_new import OTXDataset
from otx.data.entity.sample import ClassificationSample


class OTXMulticlassClsDataset(OTXDataset):
    """OTXDataset class for multi-class classification task using new Datumaro experimental Dataset."""

    def __init__(self, **kwargs) -> None:
        """Initialize OTXMulticlassClsDataset.

        Args:
            **kwargs: Keyword arguments to pass to OTXDataset
        """
        kwargs["sample_type"] = ClassificationSample
        super().__init__(**kwargs)

    def get_idx_list_per_classes(self) -> dict[int, list[int]]:
        """Get index list per class."""
        idx_list_per_classes: dict[int, list[int]] = {}
        for idx in range(len(self)):
            item = self.dm_subset[idx]
            label_id = item.label.item()
            if label_id not in idx_list_per_classes:
                idx_list_per_classes[label_id] = []
            idx_list_per_classes[label_id].append(idx)
        return idx_list_per_classes
