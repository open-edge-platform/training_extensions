# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datumaro.components.annotation import GroupType
from datumaro.experimental.categories import (
    HierarchicalLabelCategories,
    HierarchicalLabelCategory,
    LabelGroup,
)

from otx.types.label import HLabelInfo, NullLabelInfo, SegLabelInfo


def test_as_json(fxt_label_info):
    serialized = fxt_label_info.to_json()
    deserialized = fxt_label_info.__class__.from_json(serialized)
    assert fxt_label_info == deserialized


def test_seg_label_info():
    # Automatically insert background label at zero index
    assert SegLabelInfo.from_num_classes(3) == SegLabelInfo(
        ["label_0", "label_1", "label_2"],
        ["0", "1", "2"],
        [["label_0", "label_1", "label_2"]],
    )
    assert SegLabelInfo.from_num_classes(1) == SegLabelInfo(
        ["background", "label_0"],
        ["0", "1"],
        [["background", "label_0"]],
    )
    assert SegLabelInfo.from_num_classes(0) == NullLabelInfo()


def test_hlabel_info():
    labels = (
        HierarchicalLabelCategory(name="car", parent="vehicle"),
        HierarchicalLabelCategory(name="truck", parent="vehicle"),
        HierarchicalLabelCategory(name="vehicle"),
        HierarchicalLabelCategory(name="plush toy", parent="plush toy"),
        HierarchicalLabelCategory(name="No class"),
    )
    label_groups = (
        LabelGroup(
            name="Detection labels___vehicle",
            labels=("car", "truck"),
            group_type=GroupType.EXCLUSIVE,
        ),
        LabelGroup(
            name="Detection labels___plush toy",
            labels=("plush toy",),
            group_type=GroupType.EXCLUSIVE,
        ),
        LabelGroup(name="No class", labels=("No class",), group_type=GroupType.RESTRICTED),
    )
    dm_label_categories = HierarchicalLabelCategories(items=labels, label_groups=label_groups)

    hlabel_info = HLabelInfo.from_dm_label_groups(dm_label_categories)

    # check if label info can be normalized on export
    dict_label_info = hlabel_info.as_dict(normalize_label_names=True)
    for lbl in dict_label_info["label_names"]:
        assert " " not in lbl

    # Check if class_to_group_idx and label_to_idx have the same keys
    assert list(hlabel_info.class_to_group_idx.keys()) == list(
        hlabel_info.label_to_idx.keys(),
    ), "class_to_group_idx and label_to_idx keys do not match"
