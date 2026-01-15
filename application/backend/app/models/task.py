# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import Field

from .base import BaseEntity
from .label import Label


class TaskType(StrEnum):
    """
    Enumeration of supported machine learning task types.

    Defines the available types of annotation and modeling tasks that can be configured within a project.

    Attributes:
        CLASSIFICATION: Image or object classification task where items are assigned to predefined categories.
        DETECTION: Object detection task that identifies and locates objects within images using bounding boxes.
        INSTANCE_SEGMENTATION: Instance segmentation task that identifies and delineates individual object instances at
            the pixel level.
    """

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"


class Task(BaseEntity):
    """
    Represents a labeling task configuration within a project.

    A task defines the type of annotation work to be performed, the available labels, and whether labels are
    mutually exclusive (relevant only for classification problem).

    Attributes:
        exclusive_labels: Whether only one label can be assigned per item. Defaults to False, allowing multiple labels.
        task_type: The type of task (e.g., classification, detection, instance_segmentation).
        labels: List of available labels for annotation. Defaults to empty list.
    """

    exclusive_labels: bool = False
    task_type: TaskType
    labels: list[Label] = Field(default_factory=list)
