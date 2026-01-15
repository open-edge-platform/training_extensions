# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetItemDB
from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    FullImage,
    Label,
    LabelReference,
    Point,
    Polygon,
    Project,
    Rectangle,
    Task,
    TaskType,
)
from app.repositories import DatasetItemRepository
from app.services import DatasetService, LabelService
from app.services.dataset_service import AnnotationValidationError


class TestDatasetServiceUnit:
    """Unit tests for DatasetService."""

    @pytest.fixture
    def fxt_dataset_service(self, tmp_path):
        db_session = MagicMock(spec=Session)
        label_service = MagicMock(spec=LabelService)
        return DatasetService(
            data_dir=tmp_path,
            label_service=label_service,
            db_session=db_session,
        )

    @pytest.fixture
    def fxt_multiclass_classification_project(self):
        return Project(
            id=uuid4(),
            name="Test Multiclass Classification Project",
            task=Task(
                task_type=TaskType.CLASSIFICATION,
                exclusive_labels=True,
            ),
            active_pipeline=False,
        )

    @pytest.fixture
    def fxt_multilabel_classification_project(self):
        return Project(
            id=uuid4(),
            name="Test Multilabel Classification Project",
            task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=False),
            active_pipeline=False,
        )

    @pytest.fixture
    def fxt_detection_project(self):
        return Project(
            id=uuid4(),
            name="Test Detection Project",
            task=Task(task_type=TaskType.DETECTION),
            active_pipeline=False,
        )

    @pytest.fixture
    def fxt_segmentation_project(self):
        return Project(
            id=uuid4(),
            name="Test Instance Segmentation Project",
            task=Task(task_type=TaskType.INSTANCE_SEGMENTATION),
            active_pipeline=False,
        )

    def test_validate_annotations_labels(self) -> None:
        label_id = uuid4()
        labels = [Label(id=label_id, project_id=uuid4(), name="cat", color="#00FF00", hotkey="c")]
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        DatasetService._validate_annotations_labels(annotations=annotations, labels=labels)

    def test_validate_annotations_labels_not_found(self) -> None:
        label_id = uuid4()
        labels = [Label(id=label_id, project_id=uuid4(), name="cat", color="#00FF00", hotkey="c")]
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotations_labels(annotations=annotations, labels=labels)

    def test_validate_annotations_coordinates_rectangle(self) -> None:
        dataset_item = DatasetItemDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

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
            DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

    def test_validate_annotations_coordinates_polygon(self) -> None:
        dataset_item = DatasetItemDB(name="test", format="jpg", width=100, height=50, size=1024)
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4())],
                shape=Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
            )
        ]
        DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

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
            DatasetService._validate_annotations_coordinates(annotations=annotations, dataset_item=dataset_item)

    def test_validate_annotations_multilabel_classification(self, fxt_multilabel_classification_project) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        DatasetService._validate_annotations(annotations=annotations, project=fxt_multilabel_classification_project)

    def test_validate_annotations_multilabel_classification_multi_annotations(
        self, fxt_multilabel_classification_project
    ) -> None:
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
            DatasetService._validate_annotations(annotations=annotations, project=fxt_multilabel_classification_project)

    @pytest.mark.parametrize(
        "shape",
        [
            Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
        ],
    )
    def test_validate_annotations_multilabel_classification_wrong_shape(
        self, shape, fxt_multilabel_classification_project
    ):
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=shape,
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotations(annotations=annotations, project=fxt_multilabel_classification_project)

    def test_validate_annotations_multiclass_classification_multiple_labels(
        self, fxt_multiclass_classification_project
    ) -> None:
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=uuid4()), LabelReference(id=uuid4())],
                shape=FullImage(type="full_image"),
            )
        ]
        with pytest.raises(AnnotationValidationError):
            DatasetService._validate_annotations(annotations=annotations, project=fxt_multiclass_classification_project)

    def test_validate_annotations_detection(self, fxt_detection_project) -> None:
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
        DatasetService._validate_annotations(annotations=annotations, project=fxt_detection_project)

    @pytest.mark.parametrize(
        "shape",
        [
            FullImage(type="full_image"),
            Polygon(type="polygon", points=[Point(x=0, y=0), Point(x=10, y=10)]),
        ],
    )
    def test_validate_annotations_detection_wrong_shape(self, shape, fxt_detection_project):
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
            DatasetService._validate_annotations(annotations=annotations, project=fxt_detection_project)

    def test_validate_annotations_detection_wrong_shape_multiple_labels(self, fxt_detection_project) -> None:
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
            DatasetService._validate_annotations(annotations=annotations, project=fxt_detection_project)

    def test_validate_annotations_segmentation(self, fxt_segmentation_project) -> None:
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
        DatasetService._validate_annotations(annotations=annotations, project=fxt_segmentation_project)

    @pytest.mark.parametrize(
        "shape",
        [
            FullImage(type="full_image"),
            Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
        ],
    )
    def test_validate_annotations_segmentation_wrong_shape(self, shape, fxt_segmentation_project):
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
            DatasetService._validate_annotations(annotations=annotations, project=fxt_segmentation_project)

    def test_validate_annotations_segmentation_wrong_shape_multiple_labels(self, fxt_segmentation_project) -> None:
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
            DatasetService._validate_annotations(annotations=annotations, project=fxt_segmentation_project)

    def test_set_dataset_item_annotations(self, fxt_dataset_service, fxt_detection_project) -> None:
        dataset_service = fxt_dataset_service
        dataset_item_id = uuid4()
        dataset_item = MagicMock(spec=DatasetItem)
        label_id = uuid4()
        dataset_item_annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]

        with (
            patch.object(DatasetService, "_validate_annotations_labels") as mock_validate_labels,
            patch.object(DatasetService, "_validate_annotations") as mock_validate_annotations,
            patch.object(DatasetService, "_validate_annotations_coordinates") as mock_validate_coordinates,
            patch.object(DatasetService, "get_dataset_item_by_id", return_value=dataset_item),
            patch.object(DatasetItemRepository, "set_annotation_data") as mock_repo_set_annotation_data,
            patch.object(DatasetItemRepository, "set_labels") as mock_repo_set_labels,
        ):
            result = dataset_service.set_dataset_item_annotations(
                project=fxt_detection_project,
                dataset_item_id=dataset_item_id,
                annotations=dataset_item_annotations,
                user_reviewed=True,
            )

        mock_validate_labels.assert_called_once()
        mock_validate_annotations.assert_called_once()
        mock_validate_coordinates.assert_called_once()
        mock_repo_set_annotation_data.assert_called_once_with(
            obj_id=str(dataset_item_id),
            annotation_data=[
                {
                    "shape": {"type": "rectangle", "x": 0, "y": 0, "width": 10, "height": 10},
                    "labels": [{"id": str(label_id)}],
                    "confidences": None,
                }
            ],
            user_reviewed=True,
        )
        mock_repo_set_labels.assert_called_once_with(
            dataset_item_id=str(dataset_item_id),
            label_ids={str(label_id)},
        )
        assert result == dataset_item
