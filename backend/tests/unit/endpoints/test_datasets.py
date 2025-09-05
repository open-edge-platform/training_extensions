# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import tempfile
from datetime import datetime
from io import BytesIO
from unittest.mock import MagicMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi import status

from app.api.dependencies import get_dataset_service
from app.main import app
from app.schemas import DatasetItem
from app.schemas.dataset_item import DatasetItemFormat, DatasetItemSubset
from app.services import DatasetService, ResourceNotFoundError, ResourceType


@pytest.fixture
def fxt_dataset_item():
    return DatasetItem(
        id=uuid4(),
        name="test_dataset_item",
        format=DatasetItemFormat.JPG,
        width=1024,
        height=768,
        size=2048,
        source_id=uuid4(),
        subset=DatasetItemSubset.UNASSIGNED,
    )


@pytest.fixture
def fxt_dataset_service() -> MagicMock:
    dataset_service = MagicMock(spec=DatasetService)
    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    return dataset_service


class TestDatasetItemEndpoints:
    def test_create_dataset_item_no_file(self, fxt_dataset_item, fxt_dataset_service, fxt_client):
        fxt_dataset_service.create_dataset_item.return_value = fxt_dataset_item

        response = fxt_client.post(f"/api/projects/{uuid4()}/dataset/items")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.create_dataset_item.assert_not_called()

    def test_create_dataset_item_success(self, fxt_dataset_item, fxt_dataset_service, fxt_client):
        fxt_dataset_service.create_dataset_item.return_value = fxt_dataset_item

        response = fxt_client.post(
            f"/api/projects/{uuid4()}/dataset/items", files={"file": ("test_file.jpg", BytesIO(b"123"), "image/jpeg")}
        )

        assert response.status_code == status.HTTP_201_CREATED
        fxt_dataset_service.create_dataset_item.assert_called_once()

    def test_list_dataset_items(self, fxt_dataset_item, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/items")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=project_id, limit=10, offset=0, start_date=None, end_date=None
        )

    def test_list_dataset_items_filtering_and_pagination(self, fxt_dataset_item, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(
            f"/api/projects/{str(project_id)}/dataset/items?limit=50&offset=2&start_date=2025-01-09T00:00:00Z&end_date=2025-12-31T23:59:59Z"
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=project_id,
            limit=50,
            offset=2,
            start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
            end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")),
        )

    @pytest.mark.parametrize("limit", [1000, 0, -20])
    def test_list_dataset_items_wrong_limit(self, fxt_dataset_service, fxt_client, limit):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/items?limit=${limit}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_dataset_items_wrong_offset(self, fxt_dataset_service, fxt_client, offset):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/items?offset=${offset}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_dataset_items_wrong_dates(self, fxt_dataset_service, fxt_client, offset):
        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/items?start_date=2025-12-31T23:59:59Z&end_date=2025-01-09T00:00:00Z"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize(
        "http_method, http_path, service_method",
        [
            ("get", f"/api/projects/{uuid4()}/dataset/items/invalid-id", "get_dataset_item_by_id"),
            ("get", f"/api/projects/invalid-id/dataset/items/{uuid4()}", "get_dataset_item_by_id"),
            ("get", f"/api/projects/{uuid4()}/dataset/items/invalid-id/binary", "get_dataset_item_binary_path_by_id"),
            ("get", f"/api/projects/invalid-id/dataset/items/{uuid4()}/binary", "get_dataset_item_binary_path_by_id"),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/items/invalid-id/thumbnail",
                "get_dataset_item_thumbnail_path_by_id",
            ),
            (
                "get",
                f"/api/projects/invalid-id/dataset/items/{uuid4()}/thumbnail",
                "get_dataset_item_thumbnail_path_by_id",
            ),
            ("delete", f"/api/projects/{uuid4()}/dataset/items/invalid-id", "delete_dataset_item"),
            ("delete", f"/api/projects/invalid-id/dataset/items/{uuid4()}", "delete_dataset_item"),
        ],
    )
    def test_invalid_ids(self, http_method, http_path, service_method, fxt_dataset_service, fxt_client):
        response = getattr(fxt_client, http_method)(http_path)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_dataset_service, service_method).assert_not_called()

    def test_get_dataset_item_not_found(self, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_item_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/items/{str(dataset_item_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=project_id, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_success(self, fxt_dataset_item, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        fxt_dataset_service.get_dataset_item_by_id.return_value = fxt_dataset_item

        response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/items/{str(fxt_dataset_item.id)}")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=project_id, dataset_item_id=fxt_dataset_item.id
        )

    def test_get_dataset_item_binary_not_found(self, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_item_binary_path_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/items/{str(dataset_item_id)}/binary")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_binary_path_by_id.assert_called_once_with(
            project_id=project_id, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_binary_success(self, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        dataset_item_id = uuid4()

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            fxt_dataset_service.get_dataset_item_binary_path_by_id.return_value = tmp_file.name
            response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/items/{str(dataset_item_id)}/binary")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.get_dataset_item_binary_path_by_id.assert_called_once_with(
            project_id=project_id, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_thumbnail_not_found(self, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/items/{str(dataset_item_id)}/thumbnail")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.assert_called_once_with(
            project_id=project_id, dataset_item_id=dataset_item_id
        )

    def test_get_dataset_item_thumbnail_success(self, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        dataset_item_id = uuid4()

        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp_file:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.return_value = tmp_file.name
            response = fxt_client.get(f"/api/projects/{str(project_id)}/dataset/items/{str(dataset_item_id)}/thumbnail")

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.assert_called_once_with(
            project_id=project_id, dataset_item_id=dataset_item_id
        )

    def test_delete_dataset_item_not_found(self, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        dataset_item_id = uuid4()
        fxt_dataset_service.delete_dataset_item.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        response = fxt_client.delete(f"/api/projects/{str(project_id)}/dataset/items/{str(dataset_item_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.delete_dataset_item.assert_called_once_with(
            project_id=project_id, dataset_item_id=dataset_item_id
        )

    def test_delete_dataset_item_success(self, fxt_dataset_service, fxt_client):
        project_id = uuid4()
        dataset_item_id = uuid4()

        response = fxt_client.delete(f"/api/projects/{str(project_id)}/dataset/items/{str(dataset_item_id)}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_dataset_service.delete_dataset_item.assert_called_once_with(
            project_id=project_id, dataset_item_id=dataset_item_id
        )
