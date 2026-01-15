# Copyright (C) 2023-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Dataclasses for label information."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datumaro.experimental.categories import (
        HierarchicalLabelCategories,
        HierarchicalLabelCategory,
    )

__all__ = [
    "HLabelInfo",
    "LabelInfo",
    "LabelInfoTypes",
    "NullLabelInfo",
    "SegLabelInfo",
]


@dataclass
class LabelInfo:
    """Object to represent label information."""

    label_names: list[str]
    label_ids: list[str]
    label_groups: list[list[str]]

    def __eq__(self, other: object) -> bool:
        """Compare two LabelInfo objects, normalizing list/tuple differences."""
        if not isinstance(other, LabelInfo):
            return NotImplemented
        return (
            list(self.label_names) == list(other.label_names)
            and list(self.label_ids) == list(other.label_ids)
            and [list(g) for g in self.label_groups] == [list(g) for g in other.label_groups]
        )

    @property
    def num_classes(self) -> int:
        """Return number of labels."""
        return len(self.label_names)

    @classmethod
    def from_num_classes(cls, num_classes: int) -> LabelInfo:
        """Create this object from the number of classes.

        Args:
            num_classes: Number of classes

        Returns:
            LabelInfo(
                label_names=["label_0", ...],
                label_groups=[["label_0", ...]]
            )
        """
        if num_classes <= 0:
            return NullLabelInfo()

        label_names = [f"label_{idx}" for idx in range(num_classes)]
        label_ids = [str(i) for i in range(num_classes)]

        return cls(
            label_names=label_names,
            label_groups=[label_names],
            label_ids=label_ids,
        )

    def as_dict(self, normalize_label_names: bool = False) -> dict[str, Any]:
        """Return a dictionary including all params."""
        result = asdict(self)

        if normalize_label_names:

            def normalize_fn(node: str | list | tuple | dict | int) -> str | list | tuple | dict | int:
                """Normalizes the label names stored in various nested structures."""
                if isinstance(node, str):
                    return node.replace(" ", "_")
                if isinstance(node, list):
                    return [normalize_fn(item) for item in node]
                if isinstance(node, tuple):
                    return tuple(normalize_fn(item) for item in node)
                if isinstance(node, dict):
                    return {normalize_fn(key): normalize_fn(value) for key, value in node.items()}
                return node

            for k in result:
                result[k] = normalize_fn(result[k])

        return result

    def to_json(self) -> str:
        """Return JSON serialized string."""
        return json.dumps(self.as_dict())

    @classmethod
    def from_json(cls, serialized: str) -> LabelInfo:
        """Reconstruct it from the JSON serialized string."""
        labels_info = json.loads(serialized)
        if "label_ids" not in labels_info:
            labels_info["label_ids"] = labels_info["label_names"]
        return cls(**labels_info)


@dataclass
class HLabelInfo(LabelInfo):
    """The label information represents the hierarchy.

    All params should be kept since they're also used at the Model API side.

    Args:
        num_multiclass_heads: The number of multiclass heads in the hierarchy.
        num_multilabel_classes: The number of multilabel classes.
                            head indices to (start, end) tuples.
        num_single_label_classes: The number of single label classes.
        class_to_group_idx: Dictionary mapping class names to (head_index, label_index)
                        tuples representing position in hierarchy.
        all_groups: List of all label groups in the hierarchy.
        label_to_idx: Dictionary mapping label names to their global indices.
        empty_multiclass_head_indices: List of head indices that don't include any
                                    labels due to label removal.

    Example:
        Single-selection group information (Multiclass, Exclusive):
        {
            "Shape": ["Rigid", "Non-Rigid"],
            "Rigid": ["Rectangle", "Triangle"],
            "Non-Rigid": ["Circle"]
        }

        Multi-selection group information (Multilabel):
        {
            "Animal": ["Lion", "Panda"]
        }

        Results in HLabelInfo with:
        - num_multiclass_heads: 2 (Shape, Rigid)
        - num_multilabel_classes: 3 (Circle, Lion, Panda)
        - head_to_logits_range: {'0': (0, 2), '1': (2, 4)}
        - class_to_group_idx: {'Non-Rigid': (0, 0), 'Rigid': (0, 1), 'Rectangle': (1, 0), ...}
        - all_groups: [['Non-Rigid', 'Rigid'], ['Rectangle', 'Triangle'], ['Circle'], ...]

    Note:
        If there was only one label in the multiclass group, it will be handled as multilabel (Circle).
        All of the member variables should be considered for the Model API.
        Reference: https://github.com/open-edge-platform/training_extensions/blob/develop/lib/src/otx/algorithms/classification/utils/cls_utils.py#L97
    """

    num_multiclass_heads: int
    num_multilabel_classes: int
    head_idx_to_logits_range: dict[str, tuple[int, int]]
    num_single_label_classes: int
    class_to_group_idx: dict[str, tuple[int, int]]
    all_groups: list[tuple[str, ...]]
    label_to_idx: dict[str, int]
    label_tree_edges: list[list[str]]
    empty_multiclass_head_indices: list[int]

    @classmethod
    def from_dm_label_groups(cls, dm_label_categories: HierarchicalLabelCategories) -> HLabelInfo:
        """Generate HLabelData from the Datumaro LabelCategories.

        Args:
            dm_label_categories (LabelCategories): the label categories of datumaro.
        """

        def get_exclusive_group_info(exclusive_groups: list[tuple[str, ...]]) -> dict[str, Any]:
            """Get exclusive group information."""
            last_logits_pos = 0
            num_single_label_classes = 0
            head_idx_to_logits_range = {}
            class_to_idx = {}

            for i, group in enumerate(exclusive_groups):
                head_idx_to_logits_range[str(i)] = (last_logits_pos, last_logits_pos + len(group))
                last_logits_pos += len(group)
                for j, c in enumerate(group):
                    class_to_idx[c] = (i, j)
                    num_single_label_classes += 1

            return {
                "num_multiclass_heads": len(exclusive_groups),
                "head_idx_to_logits_range": head_idx_to_logits_range,
                "class_to_idx": class_to_idx,
                "num_single_label_classes": num_single_label_classes,
            }

        def get_single_label_group_info(
            single_label_groups: list,
            num_exclusive_groups: int,
        ) -> dict[str, Any]:
            """Get single label group information."""
            class_to_idx = {}

            for i, group in enumerate(single_label_groups):
                class_to_idx[group[0]] = (num_exclusive_groups, i)

            return {
                "num_multilabel_classes": len(single_label_groups),
                "class_to_idx": class_to_idx,
            }

        def merge_class_to_idx(
            exclusive_ctoi: dict[str, tuple[int, int]],
            single_label_ctoi: dict[str, tuple[int, int]],
        ) -> dict[str, tuple[int, int]]:
            """Merge the class_to_idx information from exclusive and single_label groups."""
            class_to_idx: dict[str, tuple[int, int]] = {}
            class_to_idx.update(exclusive_ctoi)
            class_to_idx.update(single_label_ctoi)
            return class_to_idx

        def get_label_tree_edges(dm_label_items: tuple[HierarchicalLabelCategory, ...]) -> list[list[str]]:
            """Get label tree edges information. Each edges represent [child, parent]."""
            return [[item.name, item.parent] for item in dm_label_items if item.parent != ""]

        def convert_labels_if_needed(
            dm_label_categories: HierarchicalLabelCategories,
            label_names: list[str],
        ) -> list[tuple[str, ...]]:
            # Check if the labels need conversion and create name to ID mapping if required
            name_to_id_mapping = None
            for label_group in dm_label_categories.label_groups:
                if label_group.labels and label_group.labels[0] not in label_names:
                    name_to_id_mapping = {
                        category.label_semantics["name"]: category.name for category in dm_label_categories.items
                    }

            # If mapping exists, update the labels
            if name_to_id_mapping:
                for label_group in dm_label_categories.label_groups:
                    object.__setattr__(
                        label_group, "labels", [name_to_id_mapping.get(label, label) for label in label_group.labels]
                    )

            # Retrieve all label groups after conversion
            return [group.labels for group in dm_label_categories.label_groups]

        label_names = [item.name for item in dm_label_categories.items]
        all_groups = convert_labels_if_needed(dm_label_categories, label_names)

        exclusive_groups = [g for g in all_groups if len(g) > 1]
        exclusive_group_info = get_exclusive_group_info(exclusive_groups)
        single_label_groups = [g for g in all_groups if len(g) == 1]
        single_label_group_info = get_single_label_group_info(
            single_label_groups,
            exclusive_group_info["num_multiclass_heads"],
        )

        merged_class_to_idx = merge_class_to_idx(
            exclusive_group_info["class_to_idx"],
            single_label_group_info["class_to_idx"],
        )

        label_to_idx = {lbl: i for i, lbl in enumerate(merged_class_to_idx.keys())}

        return HLabelInfo(
            label_names=label_names,
            label_groups=exclusive_groups + single_label_groups,  # type: ignore[arg-type]
            num_multiclass_heads=exclusive_group_info["num_multiclass_heads"],
            num_multilabel_classes=single_label_group_info["num_multilabel_classes"],
            head_idx_to_logits_range=exclusive_group_info["head_idx_to_logits_range"],
            num_single_label_classes=exclusive_group_info["num_single_label_classes"],
            class_to_group_idx=merged_class_to_idx,
            all_groups=exclusive_groups + single_label_groups,
            label_to_idx=label_to_idx,
            label_tree_edges=get_label_tree_edges(dm_label_categories.items),
            empty_multiclass_head_indices=[],  # consider the label removing case
            label_ids=[str(i) for i in range(len(label_names))],
        )

    def as_head_config_dict(self) -> dict[str, Any]:
        """Return a dictionary including params needed to configure the HLabel MMPretrained head network."""
        return {
            "num_classes": self.num_classes,
            "num_multiclass_heads": self.num_multiclass_heads,
            "num_multilabel_classes": self.num_multilabel_classes,
            "head_idx_to_logits_range": self.head_idx_to_logits_range,
            "num_single_label_classes": self.num_single_label_classes,
            "empty_multiclass_head_indices": self.empty_multiclass_head_indices,
        }

    @classmethod
    def from_json(cls, serialized: str) -> HLabelInfo:
        """Reconstruct it from the JSON serialized string."""
        loaded = json.loads(serialized)
        # List to tuple
        loaded["head_idx_to_logits_range"] = {
            key: tuple(value) for key, value in loaded["head_idx_to_logits_range"].items()
        }
        loaded["class_to_group_idx"] = {key: tuple(value) for key, value in loaded["class_to_group_idx"].items()}
        return cls(**loaded)


@dataclass
class SegLabelInfo(LabelInfo):
    """Meta information of Semantic Segmentation."""

    ignore_index: int = 255

    @classmethod
    def from_num_classes(cls, num_classes: int) -> SegLabelInfo:
        """Create this object from the number of classes.

        Args:
            num_classes: Number of classes

        Returns:
            LabelInfo(
                label_names=["Background", "label_0", ..., "label_{num_classes - 1}"]
                label_groups=[["Background", "label_0", ..., "label_{num_classes - 1}"]]
            )
        """
        if num_classes == 1:
            # binary segmentation
            label_names = ["background", "label_0"]
            return SegLabelInfo(label_names=label_names, label_groups=[label_names], label_ids=["0", "1"])

        return super().from_num_classes(num_classes)  # type: ignore[return-value]


@dataclass
class NullLabelInfo(LabelInfo):
    """Represent no label information."""

    def __init__(self) -> None:
        super().__init__(label_names=[], label_groups=[[]], label_ids=[])

    @classmethod
    def from_json(cls, _: str) -> LabelInfo:
        """Reconstruct it from the JSON serialized string."""
        return cls()


# Dispatching rules:
# 1. label_info: int => LabelInfo.from_num_classes(label_info)
# 2. label_info: list[str] => LabelInfo(label_names=label_info, label_groups=[label_info])
# 3. label_info: LabelInfo => label_info
# See OTXModel._dispatch_label_info() for more details
LabelInfoTypes = LabelInfo | int | list[str]
