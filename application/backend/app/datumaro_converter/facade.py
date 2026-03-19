# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence

from datumaro.experimental import Dataset

from app.models import DatasetItem, Label, Media, Task, TaskType

from .converters import DatasetConverter
from .domain import SampleMode
from .factories import (
    ClassificationSampleFactory,
    DetectionSampleFactory,
    InstanceSegmentationSampleFactory,
    MultilabelClassificationSampleFactory,
    SampleFactory,
)


def convert_dataset(
    task: Task,
    labels: Sequence[Label],
    get_dataset_items_and_media: Callable[[int, int], list[tuple[DatasetItem, Media]]],
    get_media_path: Callable[[Media], str],
    sample_mode: SampleMode,
) -> Dataset:
    """
    Converts dataset items to Datumaro format based on task type.

    Creates an appropriate sample factory based on the task type and uses it to convert dataset items into Datumaro
    samples. The conversion process fetches dataset items in batches, transforms them using task-specific logic, and
    assembles them into a Datumaro Dataset with proper category labels.

    Args:
        task: The machine learning task defining the type of samples to create
        labels: The sequence of project labels used to create label indices and category mappings for the dataset.
        get_dataset_items_and_media: A callback function that retrieves batches of dataset items with their associated
            media. Takes offset and limit parameters and returns a list of (DatasetItem, Media) tuples.
        get_media_path: A callback function that resolves the file path for a given dataset item's media. Takes a
            Media instance and returns the absolute path string (for video frames, it returns video path).
        sample_mode: The mode of sample creation, which can influence which type of samples are generated from dataset
            items.

    Returns:
        A Datumaro Dataset containing the converted samples with appropriate sample type (ClassificationSample,
        DetectionSample, etc.) and label categories.

    Raises:
        ValueError: If the task type is not supported or if sample creation fails due to invalid data
            (e.g., missing labels, inconsistent confidence scores).

    Example:
        >>> task = Task(task_type=TaskType.DETECTION, exclusive_labels=True)
        >>> labels = [Label(id=uuid4(), name="cat"), Label(id=uuid4(), name="dog")]
        >>> dataset = convert_dataset(
        ...     task=task,
        ...     labels=labels,
        ...     get_dataset_items_and_media=lambda offset, limit: fetch_items(offset, limit),
        ...     get_media_path=lambda item: f"/path/to/images/{item.id}.jpg",
        ...     sample_mode=SampleMode.IMPORT_EXPORT,
        ... )
    """
    factory = _create_factory_for_task(task, labels, sample_mode)
    converter = DatasetConverter(factory, get_dataset_items_and_media, get_media_path)
    return converter.convert()


def _create_factory_for_task(task: Task, labels: Sequence[Label], sample_mode: SampleMode) -> SampleFactory:
    match task.task_type:
        case TaskType.DETECTION:
            return DetectionSampleFactory(labels, sample_mode)
        case TaskType.CLASSIFICATION:
            return (
                ClassificationSampleFactory(labels, sample_mode)
                if task.exclusive_labels
                else MultilabelClassificationSampleFactory(labels, sample_mode)
            )
        case TaskType.INSTANCE_SEGMENTATION:
            return InstanceSegmentationSampleFactory(labels, sample_mode)
        case _:
            raise ValueError(f"Unsupported task type: {task.task_type}")
