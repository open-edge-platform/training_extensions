# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, call
from uuid import UUID, uuid4

import numpy as np
import pytest

from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemFormat,
    DatasetItemSubset,
    FullImage,
    Label,
    LabelReference,
    Point,
    Polygon,
    Rectangle,
    Shape,
)
from app.services.datumaro_converter import (
    ClassificationSample,
    DetectionSample,
    InstanceSegmentationSample,
    MultilabelClassificationSample,
    convert_classification_dataset,
    convert_detection_dataset,
    convert_instance_segmentation_dataset,
    convert_multilabel_classification_dataset,
    convert_polygon,
    convert_rectangle,
)


@pytest.fixture
def fxt_project_labels():
    project_id = uuid4()
    return [
        Label(id=uuid4(), project_id=project_id, name="cat", color="#00FF00", hotkey="c"),
        Label(id=uuid4(), project_id=project_id, name="dog", color="#FF0000", hotkey="d"),
    ]


@pytest.fixture
def fxt_dataset_item():
    def _create_dataset_item(
        project_id: UUID,
        name: str,
        format: DatasetItemFormat = DatasetItemFormat.JPG,
        width: int = 200,
        height: int = 100,
        size: int = 300,
        annotation_data: list[DatasetItemAnnotation] = [],
    ) -> DatasetItem:
        return DatasetItem(
            id=uuid4(),
            project_id=project_id,
            name=name,
            format=format,
            width=width,
            height=height,
            size=size,
            annotation_data=annotation_data,
            subset=DatasetItemSubset.TRAINING,
            subset_assigned_at=None,
            user_reviewed=False,
            source_id=None,
            prediction_model_id=None,
        )

    return _create_dataset_item


@pytest.fixture
def fxt_dataset_item_annotation():
    def _create_dataset_item_annotation(labels: list[str], shape: Shape) -> DatasetItemAnnotation:
        return DatasetItemAnnotation(labels=[LabelReference(id=UUID(label)) for label in labels], shape=shape)

    return _create_dataset_item_annotation


@pytest.fixture
def fxt_detection_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_detection_dataset_item(
        project_id: UUID, name: str, label: str, x: int, y: int, width: int, height: int
    ) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            name=name,
            annotation_data=[
                fxt_dataset_item_annotation(labels=[label], shape=Rectangle(x=x, y=y, width=width, height=height))
            ],
        )

    return _create_detection_dataset_item


@pytest.fixture
def fxt_classification_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_classification_dataset_item(project_id: UUID, name: str, label: str) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            name=name,
            annotation_data=[fxt_dataset_item_annotation(labels=[label], shape=FullImage())],
        )

    return _create_classification_dataset_item


@pytest.fixture
def fxt_multilabel_classification_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_classification_dataset_item(project_id: UUID, name: str, labels: list[str]) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            name=name,
            annotation_data=[fxt_dataset_item_annotation(labels=labels, shape=FullImage())],
        )

    return _create_classification_dataset_item


@pytest.fixture
def fxt_instance_segmentation_dataset_item(fxt_dataset_item, fxt_dataset_item_annotation):
    def _create_instance_segmentation_dataset_item(
        project_id: UUID, name: str, annotations: list[tuple[str, list[list[int]]]]
    ) -> DatasetItem:
        return fxt_dataset_item(
            project_id=project_id,
            name=name,
            annotation_data=[
                fxt_dataset_item_annotation(
                    labels=[label], shape=Polygon(points=[Point(x=p[0], y=p[1]) for p in coords])
                )
                for (label, coords) in annotations
            ],
        )

    return _create_instance_segmentation_dataset_item


def test_convert_rectangle() -> None:
    rectangle = Rectangle(x=10, y=20, width=200, height=100)
    result = convert_rectangle(rectangle)
    assert result == [10, 20, 210, 120]


def test_convert_polygon() -> None:
    polygon = Polygon(points=[Point(x=10, y=20), Point(x=20, y=30), Point(x=30, y=40), Point(x=40, y=50)])
    result = convert_polygon(polygon)
    assert result == [[10, 20], [20, 30], [30, 40], [40, 50]]


def test_convert_detection_dataset(fxt_project_labels, fxt_detection_dataset_item) -> None:
    project_id = uuid4()
    dataset_item_1 = fxt_detection_dataset_item(project_id, "cat", str(fxt_project_labels[0].id), 4, 5, 10, 10)
    dataset_item_2 = fxt_detection_dataset_item(project_id, "dog", str(fxt_project_labels[1].id), 14, 35, 10, 10)
    get_dataset_items = MagicMock(side_effect=[[dataset_item_1, dataset_item_2], []])
    get_image_path = MagicMock(side_effect=["path1", "path2"])

    dataset = convert_detection_dataset(
        project_labels=fxt_project_labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
    )

    assert len(dataset) == 2
    assert (
        isinstance(dataset[0], DetectionSample)
        and dataset[0].image == "path1"
        and np.array_equal(dataset[0].label, np.array([0]))
        and dataset[0].image_info.width == 200
        and dataset[0].image_info.height == 100
    )
    assert (
        isinstance(dataset[1], DetectionSample)
        and dataset[1].image == "path2"
        and np.array_equal(dataset[1].label, np.array([1]))
        and dataset[1].image_info.width == 200
        and dataset[1].image_info.height == 100
    )
    assert get_image_path.call_count == 2
    get_image_path.assert_has_calls(
        [
            call(dataset_item_1),
            call(dataset_item_2),
        ]
    )


def test_convert_multiclass_classification_dataset(fxt_project_labels, fxt_classification_dataset_item) -> None:
    project_id = uuid4()
    dataset_item_1 = fxt_classification_dataset_item(project_id, "cat", str(fxt_project_labels[0].id))
    dataset_item_2 = fxt_classification_dataset_item(project_id, "dog", str(fxt_project_labels[1].id))
    get_dataset_items = MagicMock(side_effect=[[dataset_item_1, dataset_item_2], []])
    get_image_path = MagicMock(side_effect=["path1", "path2"])

    dataset = convert_classification_dataset(
        project_labels=fxt_project_labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
    )

    assert len(dataset) == 2
    assert (
        isinstance(dataset[0], ClassificationSample)
        and dataset[0].image == "path1"
        and dataset[0].label == 0
        and dataset[0].image_info.width == 200
        and dataset[0].image_info.height == 100
    )
    assert (
        isinstance(dataset[1], ClassificationSample)
        and dataset[1].image == "path2"
        and dataset[1].label == 1
        and dataset[1].image_info.width == 200
        and dataset[1].image_info.height == 100
    )
    assert get_image_path.call_count == 2
    get_image_path.assert_has_calls(
        [
            call(dataset_item_1),
            call(dataset_item_2),
        ]
    )


def test_convert_multilabel_classification_dataset_item(
    fxt_project_labels, fxt_multilabel_classification_dataset_item
) -> None:
    project_id = uuid4()
    dataset_item = fxt_multilabel_classification_dataset_item(
        project_id, "1", [str(fxt_project_labels[0].id), str(fxt_project_labels[1].id)]
    )
    get_dataset_items = MagicMock(side_effect=[[dataset_item], []])
    get_image_path = MagicMock(side_effect=["path1"])

    dataset = convert_multilabel_classification_dataset(
        project_labels=fxt_project_labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
    )

    assert len(dataset) == 1
    assert (
        isinstance(dataset[0], MultilabelClassificationSample)
        and dataset[0].image == "path1"
        and np.array_equal(dataset[0].label, np.array([0, 1]))
        and dataset[0].image_info.width == 200
        and dataset[0].image_info.height == 100
    )
    get_image_path.assert_called_once_with(dataset_item)


def test_convert_instance_segmentation_dataset(fxt_project_labels, fxt_instance_segmentation_dataset_item) -> None:
    project_id = uuid4()
    dataset_item_1 = fxt_instance_segmentation_dataset_item(
        project_id,
        "cat",
        [
            (str(fxt_project_labels[0].id), [[0, 0], [10, 0], [10, 10], [0, 10]]),
            (str(fxt_project_labels[0].id), [[20, 20], [30, 20], [30, 30], [20, 30]]),
        ],
    )
    dataset_item_2 = fxt_instance_segmentation_dataset_item(
        project_id,
        "dog",
        [
            (str(fxt_project_labels[1].id), [[4, 6], [14, 6], [14, 16], [4, 16]]),
            (str(fxt_project_labels[1].id), [[49, 20], [59, 20], [49, 30], [59, 30]]),
        ],
    )
    get_dataset_items = MagicMock(side_effect=[[dataset_item_1, dataset_item_2], []])
    get_image_path = MagicMock(side_effect=["path1", "path2"])

    dataset = convert_instance_segmentation_dataset(
        project_labels=fxt_project_labels, get_dataset_items=get_dataset_items, get_image_path=get_image_path
    )

    assert len(dataset) == 2
    assert (
        isinstance(dataset[0], InstanceSegmentationSample)
        and dataset[0].image == "path1"
        and np.array_equal(dataset[0].label, np.array([0, 0]))
        and dataset[0].image_info.width == 200
        and dataset[0].image_info.height == 100
    )
    assert (
        isinstance(dataset[1], InstanceSegmentationSample)
        and dataset[1].image == "path2"
        and np.array_equal(dataset[1].label, np.array([1, 1]))
        and dataset[1].image_info.width == 200
        and dataset[1].image_info.height == 100
    )
    assert get_image_path.call_count == 2
    get_image_path.assert_has_calls(
        [
            call(dataset_item_1),
            call(dataset_item_2),
        ]
    )
