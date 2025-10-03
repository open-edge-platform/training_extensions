# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from typing import Any
from uuid import UUID

import numpy as np
import polars as pl
from datumaro.experimental import Dataset, Sample, bbox_field, image_path_field, label_field
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import polygon_field

from app.db.schema import DatasetItemDB, LabelDB, ProjectDB
from app.schemas.dataset_item import DatasetItemAnnotation
from app.schemas.project import TaskType
from app.schemas.shape import Polygon, Rectangle


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

    :param r: Rectangle shape
    :return: list of coordinates in x1y1x2y2 format
    """
    return [r.x, r.y, r.x + r.width, r.y + r.height]


def convert_polygon(p: Polygon) -> list[list[int]]:
    """
    Convert Polygon shape coordinates to a list

    :param p: Polygon shape
    :return: list of points coordinates in xy format
    """
    return [[point.x, point.y] for point in p.points]


def convert_detection_dataset(
    project_labels: list[LabelDB], dataset_items: list[DatasetItemDB], get_image_path: Callable[[DatasetItemDB], str]
) -> Dataset[DetectionSample]:
    """
    Convert detection dataset to Datumaro format

    :param project_labels: list of project labels
    :param dataset_items: project dataset items
    :param get_image_path: function to get image path for a dataset item
    :return: Datumaro dataset
    """
    dataset: Dataset[DetectionSample] = Dataset(
        DetectionSample,
        categories={"label": LabelCategories(labels=tuple([label.name for label in project_labels]))},
    )
    project_labels_ids = [UUID(label.id) for label in project_labels]
    for dataset_item in dataset_items:
        image_path = get_image_path(dataset_item)
        annotations = [DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data]
        coords = [
            convert_rectangle(annotation.shape)
            for annotation in annotations
            if (isinstance(annotation.shape, Rectangle))
        ]
        labels_indexes = [
            project_labels_ids.index(annotation.labels[0].id)
            for annotation in annotations
            if len(annotation.labels) == 1
        ]
        dataset.append(
            DetectionSample(
                image=image_path,
                bboxes=np.array(coords),
                label=np.array(labels_indexes),
            )
        )
    return dataset


def convert_classification_dataset(
    project_labels: list[LabelDB], dataset_items: list[DatasetItemDB], get_image_path: Callable[[DatasetItemDB], str]
) -> Dataset[ClassificationSample]:
    """
    Convert single class segmentation dataset to Datumaro format

    :param project_labels: list of project labels
    :param dataset_items: project dataset items
    :param get_image_path: function to get image path for a dataset item
    :return: Datumaro dataset
    """
    dataset: Dataset[ClassificationSample] = Dataset(
        ClassificationSample,
        categories={"label": LabelCategories(labels=tuple([label.name for label in project_labels]))},
    )
    project_labels_ids = [UUID(label.id) for label in project_labels]
    for dataset_item in dataset_items:
        image_path = get_image_path(dataset_item)
        annotation = next(
            DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
        )
        dataset.append(ClassificationSample(image=image_path, label=project_labels_ids.index(annotation.labels[0].id)))
    return dataset


def convert_multiclass_classification_dataset(
    project_labels: list[LabelDB], dataset_items: list[DatasetItemDB], get_image_path: Callable[[DatasetItemDB], str]
) -> Dataset[MultilabelClassificationSample]:
    """
    Convert multiclass classification dataset to Datumaro format

    :param project_labels: list of project labels
    :param dataset_items: project dataset items
    :param get_image_path: function to get image path for a dataset item
    :return: Datumaro dataset
    """
    dataset: Dataset[MultilabelClassificationSample] = Dataset(
        MultilabelClassificationSample,
        categories={"label": LabelCategories(labels=tuple([label.name for label in project_labels]))},
    )
    project_labels_ids = [UUID(label.id) for label in project_labels]
    for dataset_item in dataset_items:
        image_path = get_image_path(dataset_item)
        annotation = next(
            DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data
        )
        labels_indexes = [project_labels_ids.index(label.id) for label in annotation.labels]
        dataset.append(MultilabelClassificationSample(image=image_path, label=np.array(labels_indexes)))
    return dataset


def convert_instance_segmentation_dataset(
    project_labels: list[LabelDB], dataset_items: list[DatasetItemDB], get_image_path: Callable[[DatasetItemDB], str]
) -> Dataset[InstanceSegmentationSample]:
    """
    Convert instance segmentation dataset to Datumaro format

    :param project_labels: list of project labels
    :param dataset_items: project dataset items
    :param get_image_path: function to get image path for a dataset item
    :return: Datumaro dataset
    """
    dataset: Dataset[InstanceSegmentationSample] = Dataset(
        InstanceSegmentationSample,
        categories={"label": LabelCategories(labels=tuple([label.name for label in project_labels]))},
    )
    project_labels_ids = [UUID(label.id) for label in project_labels]
    for dataset_item in dataset_items:
        image_path = get_image_path(dataset_item)
        annotations = [DatasetItemAnnotation.model_validate(annotation) for annotation in dataset_item.annotation_data]
        polygons = [
            convert_polygon(annotation.shape) for annotation in annotations if (isinstance(annotation.shape, Polygon))
        ]
        labels_indexes = [
            project_labels_ids.index(annotation.labels[0].id)
            for annotation in annotations
            if len(annotation.labels) == 1
        ]
        dataset.append(
            InstanceSegmentationSample(
                image=image_path, polygons=np.array(polygons, dtype=np.float32), label=np.array(labels_indexes)
            )
        )
    return dataset


def convert_dataset(
    project: ProjectDB, dataset_items: list[DatasetItemDB], get_image_path: Callable[[DatasetItemDB], str]
) -> Dataset:
    """
    Convert project dataset to Datumaro format

    :param project: project to perform conversion for
    :param dataset_items: project dataset items
    :param get_image_path: function to get image path for a dataset item
    :return: Datumaro dataset
    """
    match project.task_type:
        case TaskType.DETECTION:
            return convert_detection_dataset(
                project_labels=project.labels, dataset_items=dataset_items, get_image_path=get_image_path
            )
        case TaskType.CLASSIFICATION:
            if project.exclusive_labels:
                return convert_classification_dataset(
                    project_labels=project.labels, dataset_items=dataset_items, get_image_path=get_image_path
                )
            return convert_multiclass_classification_dataset(
                project_labels=project.labels, dataset_items=dataset_items, get_image_path=get_image_path
            )
        case TaskType.INSTANCE_SEGMENTATION:
            return convert_instance_segmentation_dataset(
                project_labels=project.labels, dataset_items=dataset_items, get_image_path=get_image_path
            )
        case _:
            raise Exception
