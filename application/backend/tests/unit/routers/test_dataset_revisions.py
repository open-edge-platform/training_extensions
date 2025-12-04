# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import get_dataset_service
from app.api.schemas.dataset_item import DatasetItemSubset
from app.main import app
from app.models import DatasetItem, DatasetItemFormat
from app.services import DatasetService, ResourceNotFoundError, ResourceType
from app.services.dataset_service import DatasetItemFilters


@pytest.fixture
def fxt_dataset_revision_id():
    return uuid4()


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
        subset=DatasetItemSubset.TRAINING,
        user_reviewed=False,
        subset_assigned_at=None,
    )


@pytest.fixture
def fxt_dataset_service() -> MagicMock:
    dataset_service = MagicMock(spec=DatasetService)
    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    return dataset_service


@pytest.mark.skip(
    reason="Disable these tests until dataset revisions endpoints are fully implemented: https://github.com/open-edge-platform/training_extensions/issues/5052"
)
class TestDatasetRevisionItemEndpoints:
    def test_list_dataset_revision_items_revision_not_found(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        fxt_dataset_service.get_dataset_revision.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_REVISION, str(fxt_dataset_revision_id)
        )

        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_revision.assert_called_once_with(
            project_id=fxt_get_project.id, revision_id=fxt_dataset_revision_id
        )
        fxt_dataset_service.count_dataset_items.assert_not_called()
        fxt_dataset_service.list_dataset_items.assert_not_called()

    def test_list_dataset_revision_items_success(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_item, fxt_dataset_service, fxt_client
    ):
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)
        fxt_dataset_service.count_dataset_items.return_value = 1
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items"
        )

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["pagination"]["total"] == 1
        assert response_data["pagination"]["count"] == 1
        assert response_data["pagination"]["limit"] == 10
        assert response_data["pagination"]["offset"] == 0
        assert len(response_data["items"]) == 1
        assert response_data["items"][0]["id"] == str(fxt_dataset_item.id)
        assert response_data["items"][0]["name"] == "test_dataset_item"

        fxt_dataset_service.get_dataset_revision.assert_called_once_with(
            project_id=fxt_get_project.id, revision_id=fxt_dataset_revision_id
        )
        fxt_dataset_service.count_dataset_items.assert_called_once_with(
            project=fxt_get_project,
            subset=None,
        )
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=DatasetItemFilters(
                limit=10,
                offset=0,
                subset=None,
            ),
        )

    def test_list_dataset_revision_items_with_pagination(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_item, fxt_dataset_service, fxt_client
    ):
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)
        fxt_dataset_service.count_dataset_items.return_value = 100
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items?limit=50&offset=10"
        )

        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["pagination"]["total"] == 100
        assert response_data["pagination"]["limit"] == 50
        assert response_data["pagination"]["offset"] == 10

        fxt_dataset_service.count_dataset_items.assert_called_once_with(
            project=fxt_get_project,
            subset=None,
        )
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=DatasetItemFilters(
                limit=50,
                offset=10,
                subset=None,
            ),
        )

    @pytest.mark.parametrize("subset", ["unassigned", "training", "validation", "testing"])
    def test_list_dataset_revision_items_with_subset_filter(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_item, fxt_dataset_service, fxt_client, subset
    ):
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)
        fxt_dataset_service.count_dataset_items.return_value = 1
        fxt_dataset_service.list_dataset_items.return_value = [fxt_dataset_item]

        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items?subset={subset}"
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_dataset_service.count_dataset_items.assert_called_once_with(
            project=fxt_get_project,
            subset=subset,
        )
        fxt_dataset_service.list_dataset_items.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=DatasetItemFilters(
                limit=10,
                offset=0,
                subset=subset,
            ),
        )

    @pytest.mark.parametrize("limit", [1000, 0, -20])
    def test_list_dataset_revision_items_invalid_limit(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client, limit
    ):
        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items?limit={limit}"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize("offset", [-1, -20])
    def test_list_dataset_revision_items_invalid_offset(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client, offset
    ):
        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items?offset={offset}"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_dataset_service.list_dataset_items.assert_not_called()

    def test_list_dataset_revision_items_invalid_revision_id(self, fxt_get_project, fxt_dataset_service, fxt_client):
        response = fxt_client.get(f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/invalid-id/items")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.get_dataset_revision.assert_not_called()

    def test_get_dataset_revision_item_success(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_item, fxt_dataset_service, fxt_client
    ):
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)
        fxt_dataset_service.get_dataset_item_by_id.return_value = fxt_dataset_item

        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(fxt_dataset_item.id)}"
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "format": "jpg",
            "height": 768,
            "id": str(fxt_dataset_item.id),
            "name": "test_dataset_item",
            "size": 2048,
            "source_id": str(fxt_dataset_item.source_id),
            "subset": "training",
            "width": 1024,
        }
        fxt_dataset_service.get_dataset_revision.assert_called_once_with(
            project_id=fxt_get_project.id, revision_id=fxt_dataset_revision_id
        )
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_item_id=fxt_dataset_item.id
        )

    def test_get_dataset_revision_item_revision_not_found(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_revision.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_REVISION, str(fxt_dataset_revision_id)
        )

        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(dataset_item_id)}"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_by_id.assert_not_called()

    def test_get_dataset_revision_item_not_found(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)
        fxt_dataset_service.get_dataset_item_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        with pytest.raises(ResourceNotFoundError):
            response = fxt_client.get(
                f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(dataset_item_id)}"
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_dataset_revision_item_invalid_ids(self, fxt_get_project, fxt_dataset_service, fxt_client):
        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/invalid-revision-id/items/invalid-item-id"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.get_dataset_revision.assert_not_called()

    def test_get_dataset_revision_item_binary_success(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"test image data")
            tmp_file.flush()
            tmp_file_path = tmp_file.name

        try:
            fxt_dataset_service.get_dataset_item_binary_path_by_id.return_value = tmp_file_path

            response = fxt_client.get(
                f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(dataset_item_id)}/binary"
            )

            assert response.status_code == status.HTTP_200_OK
            fxt_dataset_service.get_dataset_revision.assert_called_once_with(
                project_id=fxt_get_project.id, revision_id=fxt_dataset_revision_id
            )
            fxt_dataset_service.get_dataset_item_binary_path_by_id.assert_called_once_with(
                project_id=fxt_get_project.id, dataset_item_id=dataset_item_id
            )
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def test_get_dataset_revision_item_binary_revision_not_found(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_revision.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_REVISION, str(fxt_dataset_revision_id)
        )

        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(dataset_item_id)}/binary"
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.get_dataset_item_binary_path_by_id.assert_not_called()

    def test_get_dataset_revision_item_binary_not_found(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)
        fxt_dataset_service.get_dataset_item_binary_path_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        with pytest.raises(ResourceNotFoundError):
            response = fxt_client.get(
                f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(dataset_item_id)}/binary"
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_dataset_revision_item_binary_invalid_ids(self, fxt_get_project, fxt_dataset_service, fxt_client):
        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/invalid-id/items/invalid-item-id/binary"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.get_dataset_revision.assert_not_called()

    def test_get_dataset_revision_item_thumbnail_success(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
            tmp_file.write(b"test thumbnail data")
            tmp_file.flush()
            tmp_file_path = tmp_file.name

        try:
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.return_value = tmp_file_path

            response = fxt_client.get(
                f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(dataset_item_id)}/thumbnail"
            )

            assert response.status_code == status.HTTP_200_OK
            fxt_dataset_service.get_dataset_revision.assert_called_once_with(
                project_id=fxt_get_project.id, revision_id=fxt_dataset_revision_id
            )
            fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.assert_called_once_with(
                project=fxt_get_project, dataset_item_id=dataset_item_id
            )
        finally:
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def test_get_dataset_revision_item_thumbnail_not_found(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        dataset_item_id = uuid4()
        fxt_dataset_service.get_dataset_revision.return_value = MagicMock(id=fxt_dataset_revision_id)
        fxt_dataset_service.get_dataset_item_thumbnail_path_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(dataset_item_id)
        )

        with pytest.raises(ResourceNotFoundError):
            response = fxt_client.get(
                f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}/items/{str(dataset_item_id)}/thumbnail"
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_dataset_revision_item_thumbnail_invalid_ids(self, fxt_get_project, fxt_dataset_service, fxt_client):
        response = fxt_client.get(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/invalid-id/items/invalid-item-id/thumbnail"
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.get_dataset_revision.assert_not_called()

    def test_delete_dataset_revision_files_success(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        response = fxt_client.delete(
            f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}"
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_dataset_service.delete_dataset_revision_files.assert_called_once_with(
            project_id=fxt_get_project.id, revision_id=fxt_dataset_revision_id
        )

    def test_delete_dataset_revision_files_not_found(
        self, fxt_get_project, fxt_dataset_revision_id, fxt_dataset_service, fxt_client
    ):
        fxt_dataset_service.delete_dataset_revision_files.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_REVISION, str(fxt_dataset_revision_id)
        )

        with pytest.raises(ResourceNotFoundError):
            response = fxt_client.delete(
                f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/{str(fxt_dataset_revision_id)}"
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_dataset_revision_files_invalid_id(self, fxt_get_project, fxt_dataset_service, fxt_client):
        response = fxt_client.delete(f"/api/projects/{str(fxt_get_project.id)}/dataset_revisions/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.delete_dataset_revision_files.assert_not_called()
