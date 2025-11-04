# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pre filtering data for OTX."""

from __future__ import annotations

import secrets
import warnings
from functools import partial
from typing import TYPE_CHECKING

from datumaro.components.annotation import Annotation, AnnotationType, Bbox, Ellipse, Points, Polygon
from datumaro.components.dataset import Dataset as DmDataset

from otx.types.task import OTXTaskType

if TYPE_CHECKING:
    from datumaro.components.dataset_base import DatasetItem


def get_labels(dataset: DmDataset, task: OTXTaskType) -> list[str]:
    """Get the labels from the dataset."""
    # label is funky from arrow dataset
    if task == OTXTaskType.KEYPOINT_DETECTION:
        return dataset.categories()[AnnotationType.points][0].labels
    return dataset.categories()[AnnotationType.label]


def pre_filtering(
    dataset: DmDataset,
    data_format: str,
    unannotated_items_ratio: float,
    task: OTXTaskType,
    ignore_index: int | None = None,
) -> DmDataset:
    """Pre-filtering function to filter the dataset based on certain criteria.

    Args:
        dataset (DmDataset): The input dataset to be filtered.
        data_format (str): The format of the dataset.
        unannotated_items_ratio (float): The ratio of background unannotated items to be used.
            This must be a float between 0 and 1.
        task (OTXTaskType): The task type of the dataset.
        ignore_index (int | None, optional): The index to be used for the ignored label. Defaults to None.

    Returns:
        DmDataset: The filtered dataset.
    """
    used_background_items = set()
    msg = f"There are empty annotation items in train set, Of these, only {unannotated_items_ratio*100}% are used."
    warnings.warn(msg, stacklevel=2)

    labels = get_labels(dataset, task)

    dataset = DmDataset.filter(
        dataset,
        partial(is_valid_anno_for_task, task=task, labels=labels),
        filter_annotations=True,
    )
    if task == OTXTaskType.KEYPOINT_DETECTION:
        return dataset
    dataset = remove_unused_labels(dataset, data_format, ignore_index)
    if unannotated_items_ratio > 0:
        empty_items = [
            item.id for item in dataset if item.subset in ("train", "TRAINING") and len(item.annotations) == 0
        ]

        used_background_items = set(
            secrets.SystemRandom().sample(empty_items, int(len(empty_items) * unannotated_items_ratio)),
        )

    return DmDataset.filter(
        dataset,
        lambda item: not (
            item.subset in ("train", "TRAINING") and len(item.annotations) == 0 and item.id not in used_background_items
        ),
    )


def is_valid_annot(item: DatasetItem, annotation: Annotation, labels: list[str]) -> bool:  # noqa: ARG001
    """Return whether DatasetItem's annotation is valid."""
    if isinstance(annotation, Bbox):
        x1, y1, x2, y2 = annotation.points
        if x1 < x2 and y1 < y2:
            return True
        msg = "There are bounding box which is not `x1 < x2 and y1 < y2`, they will be filtered out before training."
        warnings.warn(msg, stacklevel=2)
        return False
    if isinstance(annotation, Polygon):
        # TODO(JaegukHyun): This process is computationally intensive.
        # We should make pre-filtering user-configurable.
        x_points = [annotation.points[i] for i in range(0, len(annotation.points), 2)]
        y_points = [annotation.points[i + 1] for i in range(0, len(annotation.points), 2)]
        if min(x_points) < max(x_points) and min(y_points) < max(y_points) and annotation.get_area() > 0:
            return True
        msg = "There are invalid polygon, they will be filtered out before training."
        return False
    if isinstance(annotation, Points):
        # For keypoint detection, num of (x, y) points should be equal to num of labels
        if len(annotation.points) == 0:
            msg = "There are invalid points, they will be filtered out before training."
            warnings.warn(msg, stacklevel=2)
            return False
        return len(annotation.points) // 2 == len(labels)

    return True


def is_valid_anno_for_task(
    item: DatasetItem,
    annotation: Annotation,
    task: OTXTaskType,
    labels: list[str],
) -> bool:
    """Return whether DatasetItem's annotation is valid for a specific task.

    Args:
        item (DatasetItem): The item to be checked.
        annotation (Annotation): The annotation to be checked.
        task (OTXTaskType): The task type of the dataset.
        labels (list[str]): The labels of the dataset.

    Returns:
        bool: True if the annotation is valid for the task, False otherwise.
    """
    if task == OTXTaskType.DETECTION:
        return isinstance(annotation, Bbox) and is_valid_annot(item, annotation, labels)

    # Rotated detection is a subset of instance segmentation
    if task in [OTXTaskType.INSTANCE_SEGMENTATION, OTXTaskType.ROTATED_DETECTION]:
        return isinstance(annotation, (Polygon, Bbox, Ellipse)) and is_valid_annot(item, annotation, labels)

    if task == OTXTaskType.KEYPOINT_DETECTION:
        return isinstance(annotation, Points) and is_valid_annot(item, annotation, labels)

    return is_valid_annot(item, annotation, labels)


def remove_unused_labels(
    dataset: DmDataset,
    data_format: str,
    ignore_index: int | None,
) -> DmDataset:
    """Remove unused labels in Datumaro dataset."""
    original_categories: list[str] = dataset.get_label_cat_names()
    used_labels: list[int] = list({ann.label for item in dataset for ann in item.annotations if hasattr(ann, "label")})
    if ignore_index is not None:
        used_labels = list(filter(lambda x: x != ignore_index, used_labels))
    if data_format == "ava":
        used_labels = [0, *used_labels]
    if data_format == "common_semantic_segmentation_with_subset_dirs" and len(original_categories) < len(used_labels):
        msg = (
            "There are labels mismatch in dataset categories and actual categories comes from semantic masks."
            "Please, check `dataset_meta.json` file."
        )
        raise ValueError(msg)
    if len(used_labels) == len(original_categories):
        return dataset
    if data_format == "arrow" and max(used_labels) != len(original_categories) - 1:
        # we assume that empty label is always the last one. If it is not explicitly added to the dataset,
        # (not in the used labels) it will be filtered out.
        mapping = {cat: cat for cat in original_categories[:-1]}
    elif data_format == "arrow":
        # this mean that some other class wasn't annotated, we don't need to filter the object classes
        return dataset
    else:
        mapping = {original_categories[idx]: original_categories[idx] for idx in used_labels}
    msg = "There are unused labels in dataset, they will be filtered out before training."
    warnings.warn(msg, stacklevel=2)
    return dataset.transform("remap_labels", mapping=mapping, default="delete")
