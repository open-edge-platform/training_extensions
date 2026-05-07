# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi import status

from app.api.dependencies import get_dataset_service
from app.api.schemas.dataset_item import DatasetItemSubset, DatasetItemView
from app.main import app
from app.models import DatasetItem, DatasetItemAnnotationStatus
from app.models.dataset import DatasetStatistics
from app.services import DatasetService, ResourceNotFoundError, ResourceType
from app.services.dataset_service import DatasetItemFilters


@pytest.fixture
def fxt_dataset_item():
    return DatasetItem(
        id=uuid4(),
        project_id=uuid4(),
        annotation_data=None,
        prediction_model_id=uuid4(),
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
        subset=DatasetItemSubset.UNASSIGNED,
        user_reviewed=False,
    )


class TestDatasetItemEndpoints:
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

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_dataset_items_wrong_offset(self, fxt_get_project, fxt_dataset_service, fxt_client, offset):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/items?offset=${offset}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_dataset_items_wrong_dates(self, fxt_get_project, fxt_dataset_service, fxt_client, offset):
        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/items?start_date=2025-12-31T23:59:59Z&end_date=2025-01-09T00:00:00Z"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_dataset_service.list_dataset_items.assert_not_called()

    @pytest.mark.parametrize(
        "annotation_status",
        [
            DatasetItemAnnotationStatus.MISSING_ANNOTATIONS,
            DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
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
            "id": str(fxt_dataset_item.id),
            "user_reviewed": False,
            "subset": "unassigned",
        }
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, dataset_item_id=fxt_dataset_item.id
        )

    def test_get_dataset_statistics(self, fxt_get_project, fxt_dataset_service, fxt_client):
        statistics_dict = {
            "images": 5,
            "videos": 2,
            "video_frames": 10,
            "annotated_images": 3,
            "annotated_videos": 1,
            "annotated_video_frames": 4,
            "instances": 7,
            "instances_per_label": [
                {"label_id": "11111111-1111-1111-1111-111111111111", "instances": 5},
                {"label_id": "22222222-2222-2222-2222-222222222222", "instances": 2},
                {"label_id": None, "instances": 3},
            ],
        }
        # Patch the service to return a DatasetStatistics model with the correct fields
        fxt_dataset_service.get_dataset_statistics.return_value = DatasetStatistics.model_validate(statistics_dict)

        response = fxt_client.get(f"/api/projects/{str(fxt_get_project.id)}/dataset/statistics")

        assert response.status_code == 200
        assert response.json() == {
            "media_counts": {
                "images": 5,
                "videos": 2,
                "video_frames": 10,
            },
            "annotations_counts": {
                "annotated_images": 3,
                "annotated_videos": 1,
                "annotated_video_frames": 4,
                "instances": 7,
                "instances_per_label": [
                    {"label_id": "11111111-1111-1111-1111-111111111111", "instances": 5},
                    {"label_id": "22222222-2222-2222-2222-222222222222", "instances": 2},
                    {"label_id": None, "instances": 3},
                ],
            },
        }
        fxt_dataset_service.get_dataset_statistics.assert_called_once_with(project_id=fxt_get_project.id)
