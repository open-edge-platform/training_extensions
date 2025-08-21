# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Test of custom algo modules of OTX Detection task."""

import pytest
from torchvision.transforms.v2 import Resize

from otx.config.data import SubsetConfig
from otx.data.module import OTXDataModule
from otx.types.task import OTXTaskType


@pytest.fixture()
def fxt_data_module():
    return OTXDataModule(
        task=OTXTaskType.INSTANCE_SEGMENTATION,
        data_format="coco_instances",
        data_root="tests/assets/car_tree_bug",
        input_size=(320, 320),
        train_subset=SubsetConfig(
            batch_size=2,
            subset_name="train",
            transforms=[Resize(320)],
        ),
        val_subset=SubsetConfig(
            batch_size=2,
            subset_name="val",
            transforms=[Resize(320)],
        ),
        test_subset=SubsetConfig(
            batch_size=2,
            subset_name="test",
            transforms=[Resize(320)],
        ),
    )
