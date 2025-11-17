# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable, Sequence
from typing import TypeVar
from uuid import UUID

import numpy as np
import polars as pl
from datumaro.experimental import (
    Dataset,
    Sample,
    bbox_field,
    image_info_field,
    image_path_field,
    label_field,
    score_field,
)
from datumaro.experimental.categories import LabelCategories
from datumaro.experimental.fields import ImageInfo, Subset, polygon_field, subset_field
from loguru import logger

from app.models import DatasetItem, DatasetItemSubset, Label, Polygon, Rectangle, TaskType
from app.schemas.project import TaskBase
from app.utils.typing import NDArrayFloat32, NDArrayInt

CONVERSION_BATCH_SIZE = 50


class ClassificationSample(Sample):
    """
    Sample for multiclass classification datasets.

    Attributes:
        image: Path to the image
        image_info: Image information (width, height)
        label: Class label index (0-based)
        confidence: Confidence score for the label. Only for model predictions.
    """

    image: str = image_path_field()
    image_info: ImageInfo = image_info_field()
    label: int = label_field(dtype=pl.Int32(), is_list=False)
    confidence: float | None = score_field(dtype=pl.Float32())
    subset: Subset = subset_field()


class MultilabelClassificationSample(Sample):
    """
    Sample for multilabel classification datasets.

    Attributes:
        image: Path to the image
        image_info: Image information (width, height)
        label: Array of class label indices (0-based)
        confidence: Array of confidence scores for each label. Only for model predictions.
    """

    image: str = image_path_field()
    image_info: ImageInfo = image_info_field()
    label: NDArrayInt = label_field(dtype=pl.Int32(), multi_label=True)
    confidence: NDArrayFloat32 | None = score_field(dtype=pl.Float32(), is_list=True)
    subset: Subset = subset_field()


class DetectionSample(Sample):
    """
    Sample for object detection datasets.

    Attributes:
        image: Path to the image
        image_info: Image information (width, height)
        bboxes: Array of bounding boxes in x1y1x2y2 format
        label: Array of class label indices (0-based) for each bounding box
        confidence: Array of confidence scores for each bounding box. Only for model predictions.
    """

    image: str = image_path_field()
    image_info: ImageInfo = image_info_field()
    bboxes: NDArrayInt = bbox_field(dtype=pl.Int32())
    label: NDArrayInt = label_field(dtype=pl.Int32(), is_list=True)
    confidence: NDArrayFloat32 | None = score_field(dtype=pl.Float32(), is_list=True)
    subset: Subset = subset_field()


class InstanceSegmentationSample(Sample):
    """
    Sample for instance segmentation datasets.

    Attributes:
        image: Path to the image
        image_info: Image information (width, height)
        polygons: Array of polygons, each represented as a list of points in xy format
        label: Array of class label indices (0-based) for each polygon
        confidence: Array of confidence scores for each polygon. Only for model predictions.
    """

    image: str = image_path_field()
    image_info: ImageInfo = image_info_field()
    polygons: NDArrayFloat32 = polygon_field(dtype=pl.Float32())
    label: NDArrayInt = label_field(dtype=pl.Int32(), is_list=True)
    confidence: NDArrayFloat32 | None = score_field(dtype=pl.Float32(), is_list=True)
    subset: Subset = subset_field()


def convert_to_dm_subset(subset: DatasetItemSubset | None) -> Subset | None:
    """
    Convert DatasetItemSubset to Datumaro Subset
    Args:
        subset: DatasetItemSubset

    Returns:
        Subset: Datumaro Subset
    Raises:
        ValueError: If subset type cannot be mapped to Datumaro Subset
    """
    if subset is None:
        return Subset.UNASSIGNED
    match subset:
        case DatasetItemSubset.TRAINING:
            return Subset.TRAINING
        case DatasetItemSubset.VALIDATION:
            return Subset.VALIDATION
        case DatasetItemSubset.TESTING:
            return Subset.TESTING
        case DatasetItemSubset.UNASSIGNED:
            return Subset.UNASSIGNED
        case _:
            raise ValueError(f"Unknown subset type: {subset}")


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
    project_labels: Sequence[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItem]],
    get_image_path: Callable[[DatasetItem], str],
    convert_sample: Callable[[DatasetItem, str, list[UUID]], S | None],
) -> Dataset[S]:
    dataset: Dataset[S] = Dataset(
        sample_type, categories={"label": LabelCategories(labels=tuple([label.name for label in project_labels]))}
    )
    project_labels_ids = [label.id for label in project_labels]
    offset = 0
    dataset_items: list[DatasetItem] = get_dataset_items(offset, CONVERSION_BATCH_SIZE)
    while len(dataset_items) > 0:
        for dataset_item in dataset_items:
            image_path = get_image_path(dataset_item)
            sample = convert_sample(dataset_item, image_path, project_labels_ids)
            if sample:
                dataset.append(sample)
        offset += len(dataset_items)
        dataset_items = get_dataset_items(offset, CONVERSION_BATCH_SIZE)
    return dataset


def convert_classification_dataset(
    project_labels: Sequence[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItem]],
    get_image_path: Callable[[DatasetItem], str],
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
        dataset_item: DatasetItem, image_path: str, project_labels_ids: list[UUID]
    ) -> ClassificationSample | None:
        if dataset_item.annotation_data is None:
            return None
        try:
            annotation = dataset_item.annotation_data[0]  # classification -> only one shape (annotation)
            return ClassificationSample(
                image=image_path,
                image_info=ImageInfo(width=dataset_item.width, height=dataset_item.height),
                label=project_labels_ids.index(annotation.labels[0].id),  # multiclass -> only one label
                confidence=annotation.confidences[0] if annotation.confidences else None,
                subset=convert_to_dm_subset(dataset_item.subset),
            )
        except ValueError:
            logger.error("Unable to find one of dataset item {} labels in project", dataset_item.id)
            return None

    return _convert_dataset(
        sample_type=ClassificationSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_multilabel_classification_dataset(
    project_labels: Sequence[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItem]],
    get_image_path: Callable[[DatasetItem], str],
) -> Dataset[MultilabelClassificationSample]:
    """
    Convert multilabel classification dataset to Datumaro format

    Args:
        project_labels: List of project labels
        get_dataset_items: Function to get a batch of dataset items
        get_image_path: Function to get image path for a dataset item

    Returns:
        Dataset[MultilabelClassificationSample]: Datumaro dataset
    """

    def _convert_sample(
        dataset_item: DatasetItem, image_path: str, project_labels_ids: list[UUID]
    ) -> MultilabelClassificationSample | None:
        if dataset_item.annotation_data is None:
            return None
        try:
            annotation = dataset_item.annotation_data[0]  # classification -> only one shape (annotation)
            labels_indexes = [project_labels_ids.index(label.id) for label in annotation.labels]
        except ValueError:
            logger.error("Unable to find one of dataset item {} labels in project", dataset_item.id)
            return None
        return MultilabelClassificationSample(
            image=image_path,
            image_info=ImageInfo(width=dataset_item.width, height=dataset_item.height),
            label=np.array(labels_indexes),
            confidence=np.array(annotation.confidences) if annotation.confidences else None,
            subset=convert_to_dm_subset(dataset_item.subset),
        )

    return _convert_dataset(
        sample_type=MultilabelClassificationSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_detection_dataset(
    project_labels: Sequence[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItem]],
    get_image_path: Callable[[DatasetItem], str],
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
        dataset_item: DatasetItem, image_path: str, project_labels_ids: list[UUID]
    ) -> DetectionSample | None:
        if dataset_item.annotation_data is None:
            return None
        coords = [
            convert_rectangle(annotation.shape)
            for annotation in dataset_item.annotation_data
            if (isinstance(annotation.shape, Rectangle))
        ]
        try:
            labels_indexes = [
                project_labels_ids.index(annotation.labels[0].id)
                for annotation in dataset_item.annotation_data
                if len(annotation.labels) == 1
            ]
        except ValueError:
            logger.error("Unable to find one of dataset item {} labels in project", dataset_item.id)
            return None
        # Every item must be either a model prediction (with confidence score) or a user annotation (without)
        any_with_confidence = any(annotation.confidences is not None for annotation in dataset_item.annotation_data)
        all_with_confidence = all(annotation.confidences is not None for annotation in dataset_item.annotation_data)
        if any_with_confidence and not all_with_confidence:
            logger.error(
                "Dataset item {} contains shapes with and without confidence scores: {}",
                dataset_item.id,
                dataset_item.annotation_data,
            )
            raise ValueError("Either all or none of the annotations must have confidence scores")
        confidences = (
            [  # list of confidence scores, one per bbox
                annotation.confidences[0]  # type: ignore[index]
                for annotation in dataset_item.annotation_data
            ]
            if all_with_confidence
            else []
        )
        return DetectionSample(
            image=image_path,
            image_info=ImageInfo(width=dataset_item.width, height=dataset_item.height),
            bboxes=np.array(coords),
            label=np.array(labels_indexes),
            confidence=np.array(confidences) if confidences else None,
            subset=convert_to_dm_subset(dataset_item.subset),
        )

    return _convert_dataset(
        sample_type=DetectionSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_instance_segmentation_dataset(
    project_labels: Sequence[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItem]],
    get_image_path: Callable[[DatasetItem], str],
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
        dataset_item: DatasetItem, image_path: str, project_labels_ids: list[UUID]
    ) -> InstanceSegmentationSample | None:
        if dataset_item.annotation_data is None:
            return None
        polygons = [
            convert_polygon(annotation.shape)
            for annotation in dataset_item.annotation_data
            if (isinstance(annotation.shape, Polygon))
        ]
        try:
            labels_indexes = [
                project_labels_ids.index(annotation.labels[0].id)
                for annotation in dataset_item.annotation_data
                if len(annotation.labels) == 1
            ]
        except ValueError:
            logger.error("Unable to find one of dataset item {} labels in project", dataset_item.id)
            return None
        # Every item must be either a model prediction (with confidence score) or a user annotation (without)
        any_with_confidence = any(annotation.confidences is not None for annotation in dataset_item.annotation_data)
        all_with_confidence = all(annotation.confidences is not None for annotation in dataset_item.annotation_data)
        if any_with_confidence and not all_with_confidence:
            logger.error(
                "Dataset item {} contains shapes with and without confidence scores: {}",
                dataset_item.id,
                dataset_item.annotation_data,
            )
            raise ValueError("Either all or none of the annotations must have confidence scores")
        confidences = (
            [  # list of confidence scores, one per polygon
                annotation.confidences[0]  # type: ignore[index]
                for annotation in dataset_item.annotation_data
            ]
            if all_with_confidence
            else []
        )
        return InstanceSegmentationSample(
            image=image_path,
            image_info=ImageInfo(width=dataset_item.width, height=dataset_item.height),
            polygons=np.array(polygons, dtype=np.float32),
            label=np.array(labels_indexes),
            confidence=np.array(confidences) if confidences else None,
            subset=convert_to_dm_subset(dataset_item.subset),
        )

    return _convert_dataset(
        sample_type=InstanceSegmentationSample,
        project_labels=project_labels,
        get_dataset_items=get_dataset_items,
        get_image_path=get_image_path,
        convert_sample=_convert_sample,
    )


def convert_dataset(
    task: TaskBase,
    labels: Sequence[Label],
    get_dataset_items: Callable[[int, int], list[DatasetItem]],
    get_image_path: Callable[[DatasetItem], str],
) -> Dataset:
    """
    Convert project dataset to Datumaro format

    Args:
        task: Task metadata
        labels: Project labels
        get_dataset_items: Function to get a batch of dataset items
        get_image_path: Function to get image path for a dataset item

    Returns:
        Dataset: Datumaro dataset
    """
    match task.task_type:
        case TaskType.DETECTION:
            return convert_detection_dataset(
                project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
            )
        case TaskType.CLASSIFICATION:
            if task.exclusive_labels:
                return convert_classification_dataset(
                    project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
                )
            return convert_multilabel_classification_dataset(
                project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
            )
        case TaskType.INSTANCE_SEGMENTATION:
            return convert_instance_segmentation_dataset(
                project_labels=labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
            )
        case _:
            raise Exception
