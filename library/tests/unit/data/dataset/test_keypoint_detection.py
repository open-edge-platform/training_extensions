# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of keypoint detection datasets."""

from __future__ import annotations

import pytest
import torch
from datumaro.experimental import Dataset
from datumaro.experimental.categories import KeypointCategories, LabelCategories
from datumaro.experimental.fields import ImageInfo as DmImageInfo
from datumaro.experimental.fields import Subset
from torch import Tensor
from torchvision import tv_tensors
from torchvision.transforms.v2 import Identity, Transform

from getitune.data.dataset.keypoint_detection import OTXKeypointDetectionDataset
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import KeypointSample


class TestOTXKeypointDetectionDataset:
    @pytest.fixture
    def fxt_dm_dataset(self) -> Dataset:
        """Build a small keypoint detection dataset with the new DM API."""
        categories = {
            "label": LabelCategories(labels=("car", "tree", "bug", "person")),
            "keypoints": KeypointCategories(labels=("kp0", "kp1", "kp2", "kp3")),
        }
        ds = Dataset(KeypointSample, categories=categories)  # type: ignore[arg-type]

        num_keypoints = 4
        for i in range(4):
            h, w = 10, 10
            img = tv_tensors.Image(torch.zeros(3, h, w, dtype=torch.uint8))
            # Random keypoints: (num_keypoints, 3) with [x, y, visibility]
            kps = torch.zeros(num_keypoints, 3, dtype=torch.float32)
            kps[:, 0] = torch.arange(num_keypoints, dtype=torch.float32)  # x
            kps[:, 1] = torch.arange(num_keypoints, dtype=torch.float32)  # y
            kps[:, 2] = torch.tensor([0, 1, 2, 1], dtype=torch.float32)  # visibility (2 should be clamped)

            sample = KeypointSample(
                image=img,
                label=torch.tensor([i % 4], dtype=torch.long),
                keypoints=kps,
                dm_image_info=DmImageInfo(width=w, height=h),
                subset=Subset.TRAINING,
            )
            ds.append(sample)
        return ds

    @pytest.fixture
    def fxt_tvt_transforms(self) -> Identity:
        return Identity()

    def test_get_item_impl(
        self,
        fxt_dm_dataset,
        fxt_tvt_transforms: Transform,
    ) -> None:
        dataset = OTXKeypointDetectionDataset(
            fxt_dm_dataset,
            transforms=fxt_tvt_transforms,
        )

        entity = dataset._get_item_impl(0)
        assert hasattr(entity, "image")
        assert isinstance(entity.image, Tensor)
        assert hasattr(entity, "img_info")
        assert isinstance(entity.img_info, ImageInfo)
        assert hasattr(entity, "label")
        assert isinstance(entity.label, Tensor)
        assert hasattr(entity, "keypoints")
        assert isinstance(entity.keypoints, Tensor)
        assert entity.keypoints.shape == (4, 3)
        # visibility channel should be clamped to 1 at max
        visibility = entity.keypoints[:, 2]
        assert visibility.max().item() <= 1
