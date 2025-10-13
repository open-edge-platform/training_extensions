# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

import numpy as np
import polars as pl
from datumaro.experimental import Dataset, Sample, bbox_field, image_path_field, label_field
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import polygon_field

from app.db.schema import DatasetItemDB
from app.schemas import Label
from app.schemas.dataset_item import DatasetItemAnnotation
from app.schemas.project import ProjectBase, TaskType
from app.schemas.shape import Polygon, Rectangle

logger = logging.getLogger(__name__)

CONVERSION_BATCH_SIZE = 50


class DetectionSample(Sample):
    image: str = image_path_field()
    bboxes: np.ndarray[Any, Any] = bbox_field(dtype=pl.Int32)
    label: np.ndarray[Any, Any] = label_field(dtype=pl.Int32, is_list=True)


class ClassificationSample(Sample):
    image: str = image_path_field()
    label: int = label_field(dtype=pl.Int32, is_list=False)


class MultilabelClassificationSample(Sample):
    image: str = image_path_field()
    label: np.ndarray[Any, Any] = label_field(dtype=pl.Int32, is_list=True)


class InstanceSegmentationSample(Sample):
    image: str = image_path_field()
    polygons: np.ndarray[Any, Any] = polygon_field(dtype=pl.Float32)
    label: np.ndarray[Any, Any] = label_field(dtype=pl.Int32, is_list=True)


def convert_rectangle(r: Rectangle) -> list[int]:
    """
    Convert Rectangle shape coordinates to a list

    Args:
        r: Rectangle shape

    Returns:
        list[int]: list of coordinates in x1y1x2y2 format
    """
    return [r.x, r.y, r.x + r.width, r.y + r.height]


def convert_polygon(p: Polygon) -> list[list[int]]:
    """
    Convert Polygon shape coordinates to a list

    Args:
        p: Polygon shape

    Returns:
        list[list[int]]: list of points coordinates in xy format
    """
    return [[point.x, point.y] for point in p.points]


S = TypeVar("S", bound=Sample)  # Sample type e.g. DetectionSample, ClassificationSample


def _convert_dataset(
    sample_type: type[S],
    project_labels: list[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItemDB]],
    get_image_path: Callable[[DatasetItemDB], str],
    convert_sample: Callable[[DatasetItemDB, str, list[UUID]], S | None],
) -> Dataset[S]:
    dataset: Dataset[S] = Dataset(
        sample_type, categories={"label": LabelCategories(labels=tuple([label.name for label in project_labels]))}
    )
    project_labels_ids = [label.id for label in project_labels]
    offset = 0
    dataset_items: list[DatasetItemDB] = get_dataset_items(offset, CONVERSION_BATCH_SIZE)
    while len(dataset_items) > 0:
        for dataset_item in dataset_items:
            image_path = get_image_path(dataset_item)
            sample = convert_sample(dataset_item, image_path, project_labels_ids)
            if sample:
                dataset.append(sample)
        offset += len(dataset_items)
        dataset_items = get_dataset_items(offset, CONVERSION_BATCH_SIZE)
    return dataset


def convert_detection_dataset(
    project_labels: list[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItemDB]],
    get_image_path: Callable[[DatasetItemDB], str],
) -> Dataset[DetectionSample]:
    """
    Convert detection dataset to Datumaro format

    Args:
        project_labels: List of project labels
        get_dataset_items: Function to get a batch of dataset items
        get_image_path: Function to get image path for a dataset item

    Returns:
        Dataset[DetectionSample]: Datumaro dataset
    """

    def _convert_sample(
        dataset_item: DatasetItemDB, image_path: str, project_labels_ids: list[UUID]
    ) -> DetectionSample | None:
        annotations = [DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data]
        coords = [
            convert_rectangle(annotation.shape)
            for annotation in annotations
            if (isinstance(annotation.shape, Rectangle))
        ]
        try:
            labels_indexes = [
                project_labels_ids.index(annotation.labels[0].id)
                for annotation in annotations
                if len(annotation.labels) == 1
            ]
        except ValueError:
            logger.error("Unable to find one of dataset item %s labels in project", dataset_item.id)
            return None
        return DetectionSample(
            image=image_path,
            bboxes=np.array(coords),
            label=np.array(labels_indexes),
        )

    return _convert_dataset(
        sample_type=DetectionSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_classification_dataset(
    project_labels: list[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItemDB]],
    get_image_path: Callable[[DatasetItemDB], str],
) -> Dataset[ClassificationSample]:
    """
    Convert single class segmentation dataset to Datumaro format

    Args:
        project_labels: List of project labels
        get_dataset_items: Function to get a batch of dataset items
        get_image_path: Function to get image path for a dataset item

    Returns:
        Dataset[ClassificationSample]: Datumaro dataset
    """

    def _convert_sample(
        dataset_item: DatasetItemDB, image_path: str, project_labels_ids: list[UUID]
    ) -> ClassificationSample | None:
        annotation = next(
            DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
        )
        try:
            return ClassificationSample(image=image_path, label=project_labels_ids.index(annotation.labels[0].id))
        except ValueError:
            logger.error("Unable to find one of dataset item %s labels in project", dataset_item.id)
            return None

    return _convert_dataset(
        sample_type=ClassificationSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_multiclass_classification_dataset(
    project_labels: list[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItemDB]],
    get_image_path: Callable[[DatasetItemDB], str],
) -> Dataset[MultilabelClassificationSample]:
    """
    Convert multiclass classification dataset to Datumaro format

    Args:
        project_labels: List of project labels
        get_dataset_items: Function to get a batch of dataset items
        get_image_path: Function to get image path for a dataset item

    Returns:
        Dataset[MultilabelClassificationSample]: Datumaro dataset
    """

    def _convert_sample(
        dataset_item: DatasetItemDB, image_path: str, project_labels_ids: list[UUID]
    ) -> MultilabelClassificationSample | None:
        annotation = next(
            DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
        )
        try:
            labels_indexes = [project_labels_ids.index(label.id) for label in annotation.labels]
        except ValueError:
            logger.error("Unable to find one of dataset item %s labels in project", dataset_item.id)
            return None
        return MultilabelClassificationSample(image=image_path, label=np.array(labels_indexes))

    return _convert_dataset(
        sample_type=MultilabelClassificationSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_instance_segmentation_dataset(
    project_labels: list[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItemDB]],
    get_image_path: Callable[[DatasetItemDB], str],
) -> Dataset[InstanceSegmentationSample]:
    """
    Convert instance segmentation dataset to Datumaro format

    Args:
        project_labels: List of project labels
        get_dataset_items: Function to get a batch of dataset items
        get_image_path: Function to get image path for a dataset item

    Returns:
        Dataset[InstanceSegmentationSample]: Datumaro dataset
    """

    def _convert_sample(
        dataset_item: DatasetItemDB, image_path: str, project_labels_ids: list[UUID]
    ) -> InstanceSegmentationSample | None:
        annotations = [DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data]
        polygons = [
            convert_polygon(annotation.shape) for annotation in annotations if (isinstance(annotation.shape, Polygon))
        ]
        try:
            labels_indexes = [
                project_labels_ids.index(annotation.labels[0].id)
                for annotation in annotations
                if len(annotation.labels) == 1
            ]
        except ValueError:
            logger.error("Unable to find one of dataset item %s labels in project", dataset_item.id)
            return None
        return InstanceSegmentationSample(
            image=image_path, polygons=np.array(polygons, dtype=np.float32), label=np.array(labels_indexes)
        )

    return _convert_dataset(
        sample_type=InstanceSegmentationSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_dataset(
    project: ProjectBase,
    labels: list[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItemDB]],
    get_image_path: Callable[[DatasetItemDB], str],
) -> Dataset:
    """
    Convert project dataset to Datumaro format

    Args:
        project: Project to perform conversion for
        labels: Project labels
        get_dataset_items: Function to get a batch of dataset items
        get_image_path: Function to get image path for a dataset item

    Returns:
        Dataset: Datumaro dataset
    """
    match project.task.task_type:
        case TaskType.DETECTION:
            return convert_detection_dataset(
                project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
            )
        case TaskType.CLASSIFICATION:
            if project.task.exclusive_labels:
                return convert_classification_dataset(
                    project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
                )
            return convert_multiclass_classification_dataset(
                project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
            )
        case TaskType.INSTANCE_SEGMENTATION:
            return convert_instance_segmentation_dataset(
                project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
            )
        case _:
            raise Exception
