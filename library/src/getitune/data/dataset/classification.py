# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Module for OTXClassificationDatasets using new Datumaro experimental Dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional

from getitune import HLabelInfo, LabelInfo
from getitune.data.dataset.base import VisionDataset, Transforms
from getitune.data.entity.sample import (
    ClassificationHierarchicalSample,
    ClassificationMultiLabelSample,
    ClassificationSample,
)
from getitune.data.entity.utils import with_image_dtype
from getitune.types import TaskType

if TYPE_CHECKING:
    from datumaro.experimental import Dataset
    from datumaro.experimental.categories import HierarchicalLabelCategories


class MulticlassClsDataset(VisionDataset):
    """Geti Tune Dataset for multi-class classification tasks.

    This dataset handles single-label classification where each image belongs to exactly one class.
    It processes Datumaro dataset items and converts them into BaseSample format suitable for
    multi-class classification training and inference.

    Args:
        dm_subset (Dataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transformations to apply to the data.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".


    Raises:
        ValueError: If an image has multiple labels (multi-label case).

    Example:
        >>> from getitune.data.dataset.classification import MulticlassClsDataset
        >>> dataset = MulticlassClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(ClassificationSample, storage_dtype)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
        )

        labels = list(dm_subset.schema.attributes["label"].categories.labels)  # type: ignore[missing-attribute]
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int | str, list[int]] = {}
        for idx in range(len(self)):
            item = self.dm_subset[idx]
            label_id = item.label.item()
            if use_string_label:
                label_id = self.label_info.label_names[label_id]
            if label_id not in idx_list_per_classes:
                idx_list_per_classes[label_id] = []
            idx_list_per_classes[label_id].append(idx)
        return idx_list_per_classes

    @property
    def task_type(self) -> TaskType:
        """Geti Tune Task Type for the dataset.

        Returns:
            TaskType: The multi-class classification task type.
        """
        return TaskType.MULTI_CLASS_CLS


class MultilabelClsDataset(VisionDataset):
    """Geti Tune Dataset for multi-label classification tasks.

    This dataset handles multi-label classification where each image can belong to multiple classes
    simultaneously. It processes Datumaro dataset items and converts them into BaseSample format
    with one-hot encoded labels suitable for multi-label classification training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".


    Attributes:
        num_classes (int): Number of classes in the dataset.

    Example:
        >>> from getitune.data.dataset.classification import MultilabelClsDataset
        >>> dataset = MultilabelClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item with one-hot encoded labels
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(ClassificationMultiLabelSample, storage_dtype)
        dm_subset = dm_subset.convert_to_schema(sample_type)
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
        )

        labels = list(dm_subset.schema.attributes["label"].categories.labels)  # type: ignore[missing-attribute]
        self.label_info = LabelInfo(
            label_names=labels,
            label_groups=[labels],
            label_ids=[str(i) for i in range(len(labels))],
        )
        self.num_classes = len(labels)

    def _get_item_impl(self, index: int) -> ClassificationMultiLabelSample | None:
        item = self._read_dm_item(index)
        item.label = self._convert_to_onehot(torch.as_tensor(list(item.label)), ignored_labels=[])
        return self._apply_transforms(item)

    def _convert_to_onehot(self, labels: torch.tensor, ignored_labels: list[int]) -> torch.tensor:
        """Convert label to one-hot vector format.

        Args:
            labels: Input label tensor to convert.
            ignored_labels: List of label indices to ignore.

        Returns:
            torch.tensor: One-hot encoded label tensor where ignored labels are set to -1.
        """
        # Torch's one_hot() expects the input to be of type long
        # However, when labels are empty, they are of type float32
        onehot = functional.one_hot(labels.long(), self.num_classes).sum(0).clamp_max_(1)
        if ignored_labels:
            for ignore_label in ignored_labels:
                onehot[ignore_label] = -1
        return onehot

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int | str, list[int]] = {}
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

    @property
    def task_type(self) -> TaskType:
        """Geti Tune Task Type for the dataset.

        Returns:
            TaskType: The multi-label classification task type.
        """
        return TaskType.MULTI_LABEL_CLS


class HlabelClsDataset(VisionDataset):
    """Geti Tune Dataset for hierarchical label classification tasks.

    This dataset handles hierarchical classification where labels are organized in a tree structure
    with multiple classification heads. It supports both multiclass heads (where one class per head
    is selected) and multilabel heads (where multiple classes can be selected simultaneously).

    The dataset processes Datumaro dataset items and converts them into BaseSample format with
    hierarchical label encoding suitable for H-label classification training and inference.

    Args:
        dm_subset (DmDataset): Datumaro dataset subset containing the data items.
        transforms (Transforms, optional): Transform operations to apply to the data items.
        max_refetch (int): Maximum number of retries when fetching a data item fails.
        storage_dtype (str): Storage dtype for image data (e.g. "uint8", "float32"). Defaults to "uint8".


    Attributes:
        label_info (HLabelInfo): HLabelInfo containing hierarchical label structure information.
        id_to_name_mapping (dict[str, str]): Mapping from label IDs to label names.

    Raises:
        ValueError: If the number of multiclass heads is 0.
        TypeError: If label_info is not of type HLabelInfo.

    Example:
        >>> from getitune.data.dataset.classification import HlabelClsDataset
        >>> dataset = HlabelClsDataset(
        ...     dm_subset=my_dm_subset,
        ...     transforms=my_transforms,
        ... )
        >>> item = dataset[0]  # Get first item with hierarchical labels
    """

    def __init__(
        self,
        dm_subset: Dataset,
        transforms: Transforms | None = None,
        max_refetch: int = 1000,
        storage_dtype: str = "uint8",
    ) -> None:
        sample_type = with_image_dtype(ClassificationHierarchicalSample, storage_dtype)
        label_categories = dm_subset.schema.attributes["label"].categories
        # Pass target categories manually to ensure HierarchicalLabelCategories are not converted to regular Labels.
        target_categories = {"label": label_categories} if label_categories is not None else None
        dm_subset = dm_subset.convert_to_schema(sample_type, target_categories=target_categories)  # type: ignore[arg-type]
        super().__init__(
            dm_subset=dm_subset,
            transforms=transforms,
            max_refetch=max_refetch,
        )
        self.dm_categories: HierarchicalLabelCategories = dm_subset.schema.attributes["label"].categories  # type: ignore[assignment]
        self.label_info = HLabelInfo.from_dm_label_groups(self.dm_categories)

        self.id_to_name_mapping = dict(zip(self.label_info.label_ids, self.label_info.label_names))
        self.id_to_name_mapping[""] = ""

        if self.label_info.num_multiclass_heads == 0:
            msg = "The number of multiclass heads should be larger than 0."
            raise ValueError(msg)

    def _get_item_impl(self, index: int) -> ClassificationHierarchicalSample | None:
        item = self._read_dm_item(index)
        item.label = torch.as_tensor(self._convert_label_to_hlabel_format(list(item.label), []))
        return self._apply_transforms(item)

    def _convert_label_to_hlabel_format(self, label_anns: list[int], ignored_labels: list[int]) -> list[int]:
        """Convert format of the label to the h-label.

        It converts the label format to h-label format.
        Total length of result is sum of number of hierarchy and number of multilabel classes.

        i.e.
        Let's assume that we used the same dataset with example of the definition of HLabelData
        and the original labels are ["Rigid", "Triangle", "Lion"].

        Then, h-label format will be [0, 1, 1, 0].
        The first N-th indices represent the label index of multiclass heads (N=num_multiclass_heads),
        others represent the multilabel labels.

        [Multiclass Heads]
        0-th index = 0 -> ["Rigid"(O), "Non-Rigid"(X)] <- First multiclass head
        1-st index = 1 -> ["Rectangle"(O), "Triangle"(X), "Circle"(X)] <- Second multiclass head

        [Multilabel Head]
        2, 3 indices = [1, 0] -> ["Lion"(O), "Panda"(X)]

        Args:
            label_anns: List of label annotations to convert.
            ignored_labels: List of label indices to ignore.

        Returns:
            list[int]: H-label formatted list where first N indices are multiclass heads
                and remaining indices are multilabel classes.

        Raises:
            TypeError: If label_info is not of type HLabelInfo.
        """
        if not isinstance(self.label_info, HLabelInfo):
            msg = f"The type of label_info should be HLabelInfo, got {type(self.label_info)}."
            raise TypeError(msg)

        num_multiclass_heads = self.label_info.num_multiclass_heads
        num_multilabel_classes = self.label_info.num_multilabel_classes

        class_indices = [0] * (num_multiclass_heads + num_multilabel_classes)
        for i in range(num_multiclass_heads):
            class_indices[i] = -1

        for ann in label_anns:
            ann_name = self.dm_categories.items[ann].name
            group_idx, in_group_idx = self.label_info.class_to_group_idx[ann_name]

            if group_idx < num_multiclass_heads:
                class_indices[group_idx] = in_group_idx
            elif ann not in ignored_labels:
                class_indices[num_multiclass_heads + in_group_idx] = 1
            else:
                class_indices[num_multiclass_heads + in_group_idx] = -1

        return class_indices

    def get_idx_list_per_classes(self, use_string_label: bool = False) -> dict[int | str, list[int]]:
        """Get a dictionary mapping class labels (string or int) to lists of samples.

        Args:
            use_string_label (bool): If True, use string class labels as keys.
                If False, use integer indices as keys.
        """
        idx_list_per_classes: dict[int | str, list[int]] = {}
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

    @property
    def task_type(self) -> TaskType:
        """Geti Tune Task Type for the dataset.

        Returns:
            TaskType: The hierarchical label classification task type.
        """
        return TaskType.H_LABEL_CLS
