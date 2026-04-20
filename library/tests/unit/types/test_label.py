# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from datumaro.experimental.categories import (
    GroupType,
    HierarchicalLabelCategories,
    HierarchicalLabelCategory,
    LabelGroup,
)

from getitune.types.label import HLabelInfo, NullLabelInfo, SegLabelInfo


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

    # All classification output labels must be in label_to_idx
    assert set(hlabel_info.class_to_group_idx.keys()).issubset(
        set(hlabel_info.label_to_idx.keys()),
    ), "class_to_group_idx keys must be a subset of label_to_idx keys"

    # All parent labels referenced in label_tree_edges must also be in label_to_idx
    for child, parent in hlabel_info.label_tree_edges:
        assert child in hlabel_info.label_to_idx, f"child '{child}' missing from label_to_idx"
        assert parent in hlabel_info.label_to_idx, f"parent '{parent}' missing from label_to_idx"


def test_hlabel_info_parent_only_nodes():
    """Test hierarchy where parent labels are NOT classification outputs.

    CIFAR-100-style: superclasses like 'aquatic_mammals' only appear as
    parent nodes in the tree, never as members of a classification group.
    """
    labels = (
        HierarchicalLabelCategory(name="aquatic_mammals"),
        HierarchicalLabelCategory(name="fish"),
        HierarchicalLabelCategory(name="beaver", parent="aquatic_mammals"),
        HierarchicalLabelCategory(name="otter", parent="aquatic_mammals"),
        HierarchicalLabelCategory(name="shark", parent="fish"),
        HierarchicalLabelCategory(name="trout", parent="fish"),
    )
    label_groups = (
        LabelGroup(name="aquatic_mammals", labels=("beaver", "otter"), group_type=GroupType.EXCLUSIVE),
        LabelGroup(name="fish", labels=("shark", "trout"), group_type=GroupType.EXCLUSIVE),
    )
    dm_label_categories = HierarchicalLabelCategories(items=labels, label_groups=label_groups)

    hlabel_info = HLabelInfo.from_dm_label_groups(dm_label_categories)

    # Parent-only nodes must be in label_to_idx
    assert "aquatic_mammals" in hlabel_info.label_to_idx
    assert "fish" in hlabel_info.label_to_idx

    # All edge parents/children must be in label_to_idx
    for child, parent in hlabel_info.label_tree_edges:
        assert child in hlabel_info.label_to_idx, f"child '{child}' missing from label_to_idx"
        assert parent in hlabel_info.label_to_idx, f"parent '{parent}' missing from label_to_idx"

    # JSON round-trip must be equal
    serialized = hlabel_info.to_json()
    deserialized = HLabelInfo.from_json(serialized)
    assert hlabel_info == deserialized
