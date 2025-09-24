# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXKeypointDetectionDataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Union

import torch
from torchvision.transforms.v2.functional import to_dtype, to_image

from otx.data.entity.sample import KeypointSample
from otx.data.transform_libs.torchvision import Compose
from otx.types.label import LabelInfo

from .base_new import OTXDataset

Transforms = Union[Compose, Callable, List[Callable], dict[str, Compose | Callable | List[Callable]]]

if TYPE_CHECKING:
    from datumaro.experimental import Dataset


class OTXKeypointDetectionDataset(OTXDataset):
    """OTXDataset class for keypoint detection task."""

    def __init__(self, dm_subset: Dataset, **kwargs) -> None:
        sample_type = KeypointSample
        super().__init__(dm_subset=dm_subset, sample_type=sample_type, **kwargs)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        self.dm_subset = dm_subset
        labels = dm_subset.schema.attributes["label"].categories.labels
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def _get_item_impl(self, index: int) -> KeypointSample | None:
        item = self.dm_subset[index]
        keypoints = item.keypoints
        keypoints[:, 2] = torch.clamp(keypoints[:, 2], max=1)  # OTX represents visibility as 0 or 1
        item.keypoints = keypoints
        item.image = to_dtype(to_image(item.image), torch.float32)
        return self._apply_transforms(item)  # type: ignore[return-value]
