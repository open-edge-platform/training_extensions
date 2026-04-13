# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from datumaro.experimental import Dataset

from getitune.data.dataset.classification import (
    HLabelInfo,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


_LABEL_NAMES = ["Non-Rigid", "Rigid", "Rectangle", "Triangle", "Circle", "Lion", "Panda"]


@pytest.fixture
def fxt_mock_classification_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_detection_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_segmentation_dm_subset(mocker: MockerFixture) -> MagicMock:
    mock_dm_subset = mocker.MagicMock(spec=Dataset)
    mock_dm_subset.__len__.return_value = 1
    return mock_dm_subset


@pytest.fixture
def fxt_mock_hlabelinfo():
    mock_dict = MagicMock()
    mock_dict.__getitem__.return_value = (0, 0)
    return HLabelInfo(
        label_names=_LABEL_NAMES,
        label_groups=[["Non-Rigid", "Rigid"], ["Rectangle", "Triangle"], ["Circle"], ["Lion"], ["Panda"]],
        label_ids=_LABEL_NAMES,
        num_multiclass_heads=2,
        num_multilabel_classes=3,
        head_idx_to_logits_range={"0": (0, 2), "1": (2, 4)},
        num_single_label_classes=4,
        class_to_group_idx=mock_dict,
        all_groups=[["Non-Rigid", "Rigid"], ["Rectangle", "Triangle"], ["Circle"], ["Lion"], ["Panda"]],
        label_to_idx={
            "Rigid": 0,
            "Rectangle": 1,
            "Triangle": 2,
            "Non-Rigid": 3,
            "Circle": 4,
            "Lion": 5,
            "Panda": 6,
        },
        label_tree_edges=[
            ["Rectangle", "Rigid"],
            ["Triangle", "Rigid"],
            ["Circle", "Non-Rigid"],
        ],
        empty_multiclass_head_indices=[],
    )
