# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from uuid import uuid4

import pytest

from app.db.schema import DatasetItemDB, LabelDB, ProjectDB
from app.schemas.dataset_item import DatasetItemAnnotation
from app.schemas.label import LabelReference
from app.schemas.project import TaskType
from app.schemas.shape import FullImage, Point, Polygon, Rectangle
from app.services import DatasetService
from app.services.dataset_service import AnnotationValidationError


class TestDatasetServiceUnit:
    """Unit tests for DatasetService."""

    def test_validate_annotations_labels(self):
        label_id = uuid4()
        project = ProjectDB(
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
            labels=[LabelDB(id=str(label_id), name="cat", color="#00FF00", hotkey="c")],
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        DatasetService(Path("/tmp"))._validate_annotations_labels(annotations=annotations, project=project)

    def test_validate_annotations_labels_not_found(self):
        label_id = uuid4()
        project = ProjectDB(
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
            labels=[LabelDB(id=str(label_id), name="cat", color="#00FF00", hotkey="c")],
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations_labels(annotations=annotations, project=project)

    def test_validate_annotations_coordinates_rectangle(self):
        dataset_item = DatasetItemDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        DatasetService(Path("/tmp"))._validate_annotations_coordinates(
            annotations=annotations, dataset_item=dataset_item
        )

    @pytest.mark.parametrize(
        "x, y, width, height",
        [
            (1000, 0, 10, 10),
            (0, 1000, 10, 10),
            (0, 0, 1000, 10),
            (0, 0, 10, 1000),
        ],
    )
    def test_validate_annotations_coordinates_invalid_rectangle(self, x, y, width, height):
        dataset_item = DatasetItemDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=x, y=y, width=width, height=height),
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations_coordinates(
                annotations=annotations, dataset_item=dataset_item
            )

    def test_validate_annotations_coordinates_polygon(self):
        dataset_item = DatasetItemDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
            )
        ]
        DatasetService(Path("/tmp"))._validate_annotations_coordinates(
            annotations=annotations, dataset_item=dataset_item
        )

    @pytest.mark.parametrize(
        "x, y",
        [
            (1000, 10),
            (10, 1000),
        ],
    )
    def test_validate_annotations_coordinates_invalid_polygon(self, x, y):
        dataset_item = DatasetItemDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=x, y=y)]),
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations_coordinates(
                annotations=annotations, dataset_item=dataset_item
            )

    def test_validate_annotations_multilabel_classification(self):
        project = ProjectDB(
            name="Test Classification Project", task_type=TaskType.CLASSIFICATION, exclusive_labels=False
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    def test_validate_annotations_multilabel_classification_multi_annotations(self):
        project = ProjectDB(
            name="Test Classification Project",
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            ),
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    @pytest.mark.parametrize(
        "shape",
        [
            Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
        ],
    )
    def test_validate_annotations_multilabel_classification_wrong_shape(self, shape):
        project = ProjectDB(
            name="Test Classification Project",
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=shape,
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    def test_validate_annotations_multiclass_classification_multiple_labels(self):
        project = ProjectDB(
            name="Test Classification Project",
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    def test_validate_annotations_detection(self):
        project = ProjectDB(
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=10, y=10, width=10, height=10),
            ),
        ]
        DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    @pytest.mark.parametrize(
        "shape",
        [
            FullImage(type="full_image"),
            Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
        ],
    )
    def test_validate_annotations_detection_wrong_shape(self, shape):
        project = ProjectDB(
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    def test_validate_annotations_detection_wrong_shape_multiple_labels(self):
        project = ProjectDB(
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=10, y=10, width=10, height=10),
            ),
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    def test_validate_annotations_segmentation(self):
        project = ProjectDB(
            name="Test Instance Segmentation Project",
            task_type=TaskType.INSTANCE_SEGMENTATION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=10, y=10), Point(x=20, y=20)]),
            ),
        ]
        DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    @pytest.mark.parametrize(
        "shape",
        [
            FullImage(type="full_image"),
            Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
        ],
    )
    def test_validate_annotations_segmentation_wrong_shape(self, shape):
        project = ProjectDB(
            name="Test Instance Segmentation Project",
            task_type=TaskType.INSTANCE_SEGMENTATION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=shape,
            ),
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)

    def test_validate_annotations_segmentation_wrong_shape_multiple_labels(self):
        project = ProjectDB(
            name="Test Instance Segmentation Project",
            task_type=TaskType.INSTANCE_SEGMENTATION,
            exclusive_labels=False,
        )
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
            ),
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=10, y=10), Point(x=20, y=20)]),
            ),
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService(Path("/tmp"))._validate_annotations(annotations=annotations, project=project)
