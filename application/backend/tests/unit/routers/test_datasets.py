# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import tempfile
from datetime import datetime
from io import BytesIO
from unittest.mock import ANY, MagicMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi import status

from app.api.dependencies import get_dataset_service
from app.api.schemas.dataset_item import (
    DatasetItemAnnotation,
    DatasetItemAssignSubset,
    DatasetItemSubset,
    DatasetItemView,
    SetDatasetItemAnnotations,
)
from app.main import app
from app.models import DatasetItem, DatasetItemAnnotationStatus, DatasetItemFormat, LabelReference, Rectangle
from app.services import DatasetService, ResourceNotFoundError, ResourceType
from app.services.dataset_service import AnnotationValidationError, DatasetItemFilters, SubsetAlreadyAssignedError


@pytest.fixture
def fxt_dataset_item():
    return DatasetItem(
        id=uuid4(),
        project_id=uuid4(),
        name="test_dataset_item",
        format=DatasetItemFormat.JPG,
        width=1024,
        height=768,
        size=2048,
        annotation_data=None,
        prediction_model_id=uuid4(),
        source_id=uuid4(),
        subset=DatasetItemSubset.UNASSIGNED,
        user_reviewed=False,
        subset_assigned_at=None,
    )


@pytest.fixture
def fxt_dataset_service() -> MagicMock:
    dataset_service = MagicMock(spec=DatasetService)
    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    return dataset_service


def test_convert_dataset_item_to_view(fxt_dataset_item) -> None:
    view = DatasetItemView.model_validate(fxt_dataset_item, from_attributes=True)
    assert view == DatasetItemView(
        id=fxt_dataset_item.id,
        name=fxt_dataset_item.name,
        format=fxt_dataset_item.format,
        width=fxt_dataset_item.width,
        height=fxt_dataset_item.height,
        size=fxt_dataset_item.size,
        source_id=fxt_dataset_item.source_id,
        subset=DatasetItemSubset.UNASSIGNED,
    )


class TestDatasetItemEndpoints:
    def test_create_dataset_item_no_file(self, fxt_get_project, fxt_dataset_item, fxt_dataset_service, fxt_client):
        fxt_dataset_service.create_dataset_item.return_value = fxt_dataset_item

        response = fxt_client.post(f"/api/projects/{uuid4()}/dataset/items")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.create_dataset_item.assert_not_called()

    def test_create_dataset_item_success(self, fxt_get_project, fxt_dataset_item, fxt_dataset_service, fxt_client):
        fxt_dataset_service.create_dataset_item.return_value = fxt_dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/items",
            files={"file": ("test_file.jpg", BytesIO(b"123"), "image/jpeg")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "format": "jpg",
            "height": 768,
            "id": str(fxt_dataset_item.id),
            "name": "test_dataset_item",
            "size": 2048,
            "source_id": str(fxt_dataset_item.source_id),
            "subset": "unassigned",
            "width": 1024,
        }
        fxt_dataset_service.create_dataset_item.assert_called_once_with(
            project=fxt_get_project,
            data=ANY,
            name="test_file",
            format="jpg",
            user_reviewed=True,
        )

    def test_list_dataset_items(self, fxt_get_project, fxt_dataset_item, fxt_dataset_service, fxt_client):
        fxt_dataset_service.count_dataset_items.return_value = 1
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.count_dataset_items.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=None,
            label_ids=None,
            subset=None,
        )
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=DatasetItemFilters(
                limit=10, offset=0, start_date=None, end_date=None, annotation_status=None, label_ids=None, subset=None
            ),
        )

    def test_list_dataset_items_filtering_and_pagination(
        self, fxt_get_project, fxt_dataset_item, fxt_dataset_service, fxt_client
    ):
        fxt_dataset_service.count_dataset_items.return_value = 1
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/items?limit=50&offset=2&start_date=2025-01-09T00:00:00Z&end_date=2025-12-31T23:59:59Z"
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.count_dataset_items.assert_called_once_with(
            project=fxt_get_project,
            start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
            end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")),
            annotation_status=None,
            label_ids=None,
            subset=None,
        )
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=DatasetItemFilters(
                limit=50,
                offset=2,
                start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
                end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")),
                annotation_status=None,
                label_ids=None,
                subset=None,
            ),
        )

    @pytest.mark.parametrize("limit", [1000, 0, -20])
    def test_list_dataset_items_wrong_limit(self, fxt_get_project, fxt_dataset_service, fxt_client, limit):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/items?limit=${limit}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_dataset_items_wrong_offset(self, fxt_get_project, fxt_dataset_service, fxt_client, offset):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/items?offset=${offset}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_dataset_items_wrong_dates(self, fxt_get_project, fxt_dataset_service, fxt_client, offset):
        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/items?start_date=2025-12-31T23:59:59Z&end_date=2025-01-09T00:00:00Z"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize(
        "annotation_status",
        [
            DatasetItemAnnotationStatus.UNANNOTATED,
            DatasetItemAnnotationStatus.REVIEWED,
            DatasetItemAnnotationStatus.TO_REVIEW,
        ],
    )
    def test_list_dataset_items_with_annotation_status(
        self, fxt_get_project, fxt_dataset_item, fxt_dataset_service, fxt_client, annotation_status
    ):
        fxt_dataset_service.count_dataset_items.return_value = 1
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items?annotation_status={annotation_status}")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.count_dataset_items.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=annotation_status,
            label_ids=None,
            subset=None,
        )
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=DatasetItemFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=annotation_status,
                label_ids=None,
                subset=None,
            ),
        )

    @pytest.mark.parametrize("subset", ["unassigned", "training", "validation", "testing"])
    def test_list_dataset_items_with_subset(
        self, fxt_get_project, fxt_dataset_item, fxt_dataset_service, fxt_client, subset
    ):
        fxt_dataset_service.count_dataset_items.return_value = 1
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items?subset={subset}")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.count_dataset_items.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=None,
            label_ids=None,
            subset=subset,
        )
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=DatasetItemFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=None,
                label_ids=None,
                subset=subset,
            ),
        )

    @pytest.mark.parametrize(
        "http_method, http_path, service_method",
        [
            ("get", f"/api/projects/{uuid4()}/dataset/items/invalid-id", "get_dataset_item_by_id"),
            ("get", f"/api/projects/{uuid4()}/dataset/items/invalid-id/binary", "get_dataset_item_binary_path_by_id"),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/items/invalid-id/thumbnail",
                "get_dataset_item_thumbnail_path_by_id",
            ),
            ("delete", f"/api/projects/{uuid4()}/dataset/items/invalid-id", "delete_dataset_item"),
            ("post", f"/api/projects/{uuid4()}/dataset/items/invalid-id/annotations", "set_dataset_item_annotations"),
            ("get", f"/api/projects/{uuid4()}/dataset/items/invalid-id/annotations", "get_dataset_item_by_id"),
            (
                "delete",
                f"/api/projects/{uuid4()}/dataset/items/invalid-id/annotations",
                "delete_dataset_item_annotations",
            ),
            ("patch", f"/api/projects/{uuid4()}/dataset/items/invalid-id/subset", "assign_dataset_item_subset"),
        ],
    )
    def test_invalid_ids(
        self, http_method, http_path, service_method, fxt_get_project, fxt_dataset_service, fxt_client
    ):
        response = getattr(fxt_client, http_method)(http_path)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_dataset_service, service_method).assert_not_called()

    def test_get_dataset_item_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_item_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_success(self, fxt_get_project, fxt_dataset_item, fxt_dataset_service, fxt_client):
        fxt_dataset_service.get_dataset_item_by_id.return_value = fxt_dataset_item

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(fxt_dataset_item.id)}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "format": "jpg",
            "height": 768,
            "id": str(fxt_dataset_item.id),
            "name": "test_dataset_item",
            "size": 2048,
            "source_id": str(fxt_dataset_item.source_id),
            "subset": "unassigned",
            "width": 1024,
        }
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_item_id=fxt_dataset_item.id
        )

    def test_get_dataset_item_binary_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_item_binary_path_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/binary")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_binary_path_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_binary_success(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            fxt_dataset_service.get_dataset_item_binary_path_by_id.return_value = tmp_file.name
            response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/binary")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.get_dataset_item_binary_path_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_thumbnail_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/thumbnail")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.assert_called_once_with(
            project=fxt_get_project, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_thumbnail_success(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.return_value = tmp_file.name
            response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/thumbnail")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.assert_called_once_with(
            project=fxt_get_project, dataset_item_id=dataset_item_id
        )

    def test_delete_dataset_item_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()
        fxt_dataset_service.delete_dataset_item.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.delete_dataset_item.assert_called_once_with(
            project=fxt_get_project, dataset_item_id=dataset_item_id
        )

    def test_delete_dataset_item_success(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_dataset_service.delete_dataset_item.assert_called_once_with(
            project=fxt_get_project, dataset_item_id=dataset_item_id
        )

    def test_set_dataset_item_annotations_success(self, fxt_get_project, fxt_dataset_service, fxt_client):
        label_id = uuid4()
        dataset_item_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations",
            json=SetDatasetItemAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "annotations": [
                {
                    "confidences": None,
                    "labels": [{"id": str(label_id)}],
                    "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                }
            ],
            "prediction_model_id": None,
            "user_reviewed": True,
        }
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=dataset_item_id,
            annotations=annotations,
        )

    def test_set_dataset_item_annotations_label_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        label_id = uuid4()
        dataset_item_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_dataset_service.set_dataset_item_annotations.side_effect = AnnotationValidationError(str(label_id))

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations",
            json=SetDatasetItemAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=dataset_item_id,
            annotations=annotations,
        )

    def test_set_dataset_item_annotations_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        label_id = uuid4()
        dataset_item_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        fxt_dataset_service.set_dataset_item_annotations.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations",
            json=SetDatasetItemAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=dataset_item_id,
            annotations=annotations,
        )

    def test_get_dataset_item_annotations(self, fxt_get_project, fxt_dataset_service, fxt_client):
        label_id = uuid4()
        dataset_item_id = uuid4()
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=[
                DatasetItemAnnotation(
                    labels=[LabelReference(id=label_id)],
                    shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
                )
            ],
            user_reviewed=True,
            prediction_model_id=None,
        )
        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "annotations": [
                {
                    "confidences": None,
                    "labels": [{"id": str(label_id)}],
                    "shape": {"height": 10, "type": "rectangle", "width": 10, "x": 0, "y": 0},
                }
            ],
            "prediction_model_id": None,
            "user_reviewed": True,
        }
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=dataset_item_id,
        )

    def test_get_dataset_item_annotations_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_item_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=dataset_item_id,
        )

    def test_get_dataset_item_annotations_not_annotated(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()
        dataset_item = MagicMock(spec=DatasetItem, annotation_data=None, user_reviewed=False, prediction_model_id=None)
        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=dataset_item_id,
        )

    def test_delete_dataset_item_annotations(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=dataset_item_id,
        )

    def test_delete_dataset_item_annotations_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()
        fxt_dataset_service.delete_dataset_item_annotations.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=dataset_item_id,
        )

    def test_assign_dataset_item_subset(self, fxt_get_project, fxt_dataset_service, fxt_dataset_item, fxt_client):
        dataset_item_id = uuid4()

        fxt_dataset_service.assign_dataset_item_subset.return_value = fxt_dataset_item

        response = fxt_client.patch(
            f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/subset",
            json=DatasetItemAssignSubset(subset=DatasetItemSubset.TRAINING).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "format": "jpg",
            "height": 768,
            "id": str(fxt_dataset_item.id),
            "name": "test_dataset_item",
            "size": 2048,
            "source_id": str(fxt_dataset_item.source_id),
            "subset": "unassigned",
            "width": 1024,
        }
        fxt_dataset_service.assign_dataset_item_subset.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=dataset_item_id,
            subset=DatasetItemSubset.TRAINING,
        )

    def test_assign_dataset_item_subset_not_found(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()

        fxt_dataset_service.assign_dataset_item_subset.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.patch(
            f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/subset",
            json=DatasetItemAssignSubset(subset=DatasetItemSubset.TRAINING).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.assign_dataset_item_subset.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=dataset_item_id,
            subset=DatasetItemSubset.TRAINING,
        )

    def test_assign_dataset_item_subset_already_assigned(self, fxt_get_project, fxt_dataset_service, fxt_client):
        dataset_item_id = uuid4()

        fxt_dataset_service.assign_dataset_item_subset.side_effect = SubsetAlreadyAssignedError

        response = fxt_client.patch(
            f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/subset",
            json=DatasetItemAssignSubset(subset=DatasetItemSubset.TRAINING).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_dataset_service.assign_dataset_item_subset.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=dataset_item_id,
            subset=DatasetItemSubset.TRAINING,
        )

    @pytest.mark.parametrize("subset", ["unassigned", "foobar"])
    def test_assign_dataset_item_subset_invalid_subset(self, fxt_get_project, fxt_dataset_service, fxt_client, subset):
        dataset_item_id = uuid4()

        response = fxt_client.patch(
            f"/api/projects/{str(uuid4())}/dataset/items/{str(dataset_item_id)}/subset",
            json='{"subset": "' + subset + '"}',
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.assign_dataset_item_subset.assert_not_called()
