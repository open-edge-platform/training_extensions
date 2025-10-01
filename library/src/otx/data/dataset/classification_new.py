# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXClassificationDatasets using new Datumaro experimental Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional
from torchvision.transforms.v2.functional import to_dtype, to_image

from otx import LabelInfo
from otx.data.dataset.base_new import OTXDataset
from otx.data.entity.sample import ClassificationMultiLabelSample, ClassificationSample

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXMulticlassClsDataset(OTXDataset):
    """OTXDataset class for multi-class classification task using new Datumaro experimental Dataset."""

    def __init__(self, dm_subset: Dataset, **kwargs) -> None:
        """Initialize OTXMulticlassClsDataset.

        Args:
            **kwargs: Keyword arguments to pass to OTXDataset
        """
        sample_type = ClassificationSample
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(dm_subset=dm_subset, sample_type=sample_type, **kwargs)

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


class OTXMultilabelClsDataset(OTXDataset):
    """OTXDataset class for multi-label classification task."""

    def __init__(self, dm_subset: Dataset, **kwargs) -> None:
        sample_type = ClassificationMultiLabelSample
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(dm_subset=dm_subset, sample_type=sample_type, **kwargs)

        labels = dm_subset.schema.attributes["label"].categories.labels
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )
        self.num_classes = len(labels)

    def _get_item_impl(self, index: int) -> ClassificationMultiLabelSample | None:
        item = self.dm_subset[index]
        item.image = to_dtype(to_image(item.image), dtype=torch.float32)
        item.label = self._convert_to_onehot(torch.as_tensor(list(item.label)), ignored_labels=[])
        return self._apply_transforms(item)

    def _convert_to_onehot(self, labels: torch.tensor, ignored_labels: list[int]) -> torch.tensor:
        """Convert label to one-hot vector format."""
        # Torch's one_hot() expects the input to be of type long
        # However, when labels are empty, they are of type float32
        onehot = functional.one_hot(labels.long(), self.num_classes).sum(0).clamp_max_(1)
        if ignored_labels:
            for ignore_label in ignored_labels:
                onehot[ignore_label] = -1
        return onehot

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int, list[int]] = {}
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
