# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Unit tests of keypoint detection datasets."""

from __future__ import annotations

import pytest
from datumaro import Dataset as DmDataset
from torch import Tensor
from torchvision.transforms.v2 import Identity, Transform

from otx.data.dataset.keypoint_detection import OTXKeypointDetectionDataset
from otx.data.entity.base import ImageInfo


class TestOTXKeypointDetectionDataset:
    @pytest.fixture()
    def fxt_dm_dataset(self) -> DmDataset:
        return DmDataset.import_from("tests/assets/car_tree_bug_keypoint", format="coco_person_keypoints")

    @pytest.fixture()
    def fxt_tvt_transforms(self) -> Identity:
        return Identity()

    @pytest.mark.parametrize("subset", ["train", "val"])
    def test_get_item_impl_subset(
        self,
        fxt_dm_dataset,
        fxt_tvt_transforms: Transform,
        subset: str,
    ) -> None:
        dataset = OTXKeypointDetectionDataset(
            fxt_dm_dataset.get_subset(subset).as_dataset(),
            fxt_tvt_transforms,
        )

        entity = dataset._get_item_impl(0)
        assert hasattr(entity, "image")
        assert isinstance(entity.image, Tensor)
        assert hasattr(entity, "img_info")
        assert isinstance(entity.img_info, ImageInfo)
        assert hasattr(entity, "label")
        assert isinstance(entity.label, Tensor)
        assert hasattr(entity, "bboxes")
        assert hasattr(entity, "keypoints")
        assert isinstance(entity.keypoints, Tensor)
        assert entity.keypoints.shape == (4, 3)
