# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os
import tempfile
from datetime import datetime
from io import BytesIO
from unittest.mock import ANY, MagicMock
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from fastapi import status

from app.api.dependencies import get_dataset_service, get_media_service
from app.api.schemas.media import MediaView, SetMediaAnnotations
from app.main import app
from app.models import (
    DatasetItem,
    DatasetItemAnnotation,
    DatasetItemAnnotationStatus,
    LabelReference,
    Media,
    MediaType,
    Rectangle,
    VideoFrame,
)
from app.models.media import ImageFormat, VideoFormat
from app.services import DatasetService, MediaService, ResourceNotFoundError, ResourceType
from app.services.dataset_service import AnnotationValidationError
from app.services.media_service import MediaFilters


@pytest.fixture
def fxt_image_media():
    return Media(
        id=uuid4(),
        type=MediaType.IMAGE,
        project_id=uuid4(),
        name="test_image",
        format=ImageFormat.JPG,
        width=1024,
        height=768,
        size=2048,
        fps=None,
        frame_count=None,
        source_id=uuid4(),
    )


@pytest.fixture
def fxt_video_media():
    return Media(
        id=uuid4(),
        type=MediaType.VIDEO,
        project_id=uuid4(),
        name="test_video",
        format=VideoFormat.MP4,
        width=1024,
        height=768,
        size=2048,
        fps=25,
        frame_count=1000,
        source_id=uuid4(),
    )


@pytest.fixture
def fxt_media_service() -> MagicMock:
    media_service = MagicMock(spec=MediaService)
    app.dependency_overrides[get_media_service] = lambda: media_service
    return media_service


@pytest.fixture
def fxt_dataset_service() -> MagicMock:
    dataset_service = MagicMock(spec=DatasetService)
    app.dependency_overrides[get_dataset_service] = lambda: dataset_service
    return dataset_service


@pytest.mark.parametrize("media_name", ["fxt_image_media", "fxt_video_media"])
def test_convert_image_to_view(request, media_name) -> None:
    media = request.getfixturevalue(media_name)
    view = MediaView.model_validate(media, from_attributes=True)
    assert view == MediaView(
        id=media.id,
        name=media.name,
        type=media.type,
        format=media.format,
        width=media.width,
        height=media.height,
        size=media.size,
        fps=media.fps,
        frame_count=media.frame_count,
        source_id=media.source_id,
    )


class TestMediaEndpoints:
    def test_create_media_no_file(
        self, fxt_get_project, fxt_image_media, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        response = fxt_client.post(f"/api/projects/{uuid4()}/dataset/media")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.create_image.assert_not_called()
        fxt_media_service.create_video.assert_not_called()
        fxt_dataset_service.create_dataset_item.assert_not_called()

    def test_create_image_success(
        self, fxt_get_project, fxt_image_media, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        fxt_media_service.create_image.return_value = fxt_image_media

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media",
            files={"file": ("test_file.jpg", BytesIO(b"123"), "image/jpeg")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "type": "image",
            "format": "jpg",
            "height": 768,
            "id": str(fxt_image_media.id),
            "name": "test_image",
            "size": 2048,
            "source_id": str(fxt_image_media.source_id),
            "width": 1024,
            "fps": None,
            "frame_count": None,
            "duration": None,
        }
        fxt_media_service.create_image.assert_called_once_with(
            project=fxt_get_project,
            data=ANY,
            name="test_file",
            format="jpg",
        )
        fxt_dataset_service.create_dataset_item.assert_called_once_with(
            project=fxt_get_project,
            media=fxt_image_media,
            user_reviewed=False,
        )

    def test_create_video_success(
        self, fxt_get_project, fxt_video_media, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        fxt_media_service.create_video.return_value = fxt_video_media

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media",
            files={"file": ("test_file.mp4", BytesIO(b"123"), "video/mp4")},
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {
            "type": "video",
            "format": "mp4",
            "height": 768,
            "id": str(fxt_video_media.id),
            "name": "test_video",
            "size": 2048,
            "source_id": str(fxt_video_media.source_id),
            "width": 1024,
            "fps": 25.0,
            "frame_count": 1000,
            "duration": 40,
        }
        fxt_media_service.create_video.assert_called_once_with(
            project=fxt_get_project,
            data=ANY,
            name="test_file",
            format="mp4",
        )
        fxt_dataset_service.create_dataset_item.assert_not_called()

    def test_list_media(self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=None,
            label_ids=None,
            subset=None,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10, offset=0, start_date=None, end_date=None, annotation_status=None, label_ids=None, subset=None
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    def test_list_media_filtering_and_pagination(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media?limit=50&offset=2&start_date=2025-01-09T00:00:00Z&end_date=2025-12-31T23:59:59Z"
        )

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
            end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")),
            annotation_status=None,
            label_ids=None,
            subset=None,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=50,
                offset=2,
                start_date=datetime(2025, 1, 9, 0, 0, 0, tzinfo=ZoneInfo("UTC")),
                end_date=datetime(2025, 12, 31, 23, 59, 59, tzinfo=ZoneInfo("UTC")),
                annotation_status=None,
                label_ids=None,
                subset=None,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    @pytest.mark.parametrize("limit", [1000, 0, -20])
    def test_list_media_wrong_limit(self, fxt_get_project, fxt_media_service, fxt_client, limit):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/media?limit=${limit}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_media_wrong_offset(self, fxt_get_project, fxt_media_service, fxt_client, offset):
        response = fxt_client.get(f"/api/projects/{uuid4()}/dataset/media?offset=${offset}")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    @pytest.mark.parametrize("offset", [-20])
    def test_list_media_wrong_dates(self, fxt_get_project, fxt_media_service, fxt_client, offset):
        response = fxt_client.get(
            f"/api/projects/{str(uuid4())}/dataset/media?start_date=2025-12-31T23:59:59Z&end_date=2025-01-09T00:00:00Z"
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        fxt_media_service.list_media.assert_not_called()

    @pytest.mark.parametrize(
        "annotation_status",
        [
            DatasetItemAnnotationStatus.UNANNOTATED,
            DatasetItemAnnotationStatus.REVIEWED,
            DatasetItemAnnotationStatus.TO_REVIEW,
        ],
    )
    def test_list_media_with_annotation_status(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client, annotation_status
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media?annotation_status={annotation_status}")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=annotation_status,
            label_ids=None,
            subset=None,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=annotation_status,
                label_ids=None,
                subset=None,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    @pytest.mark.parametrize("subset", ["unassigned", "training", "validation", "testing"])
    def test_list_media_with_subset(
        self, fxt_get_project, fxt_image_media, fxt_video_media, fxt_media_service, fxt_client, subset
    ):
        fxt_media_service.count_media.return_value = 2
        fxt_media_service.list_media.return_value = [fxt_image_media, fxt_video_media]

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media?subset={subset}")

        assert response.status_code == status.HTTP_200_OK
        fxt_media_service.count_media.assert_called_once_with(
            project=fxt_get_project,
            start_date=None,
            end_date=None,
            annotation_status=None,
            label_ids=None,
            subset=subset,
            exclude_types=[MediaType.VIDEO_FRAME],
        )
        fxt_media_service.list_media.assert_called_once_with(
            project_id=fxt_get_project.id,
            filters=MediaFilters(
                limit=10,
                offset=0,
                start_date=None,
                end_date=None,
                annotation_status=None,
                label_ids=None,
                subset=subset,
            ),
            exclude_types=[MediaType.VIDEO_FRAME],
        )

    @pytest.mark.parametrize(
        "http_method, http_path, service_name, service_method",
        [
            ("get", f"/api/projects/{uuid4()}/dataset/media/invalid-id", "fxt_media_service", "get_media_by_id"),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/binary",
                "fxt_media_service",
                "get_media_binary_path_by_id",
            ),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/thumbnail",
                "fxt_media_service",
                "generate_media_thumbnail",
            ),
            ("delete", f"/api/projects/{uuid4()}/dataset/media/invalid-id", "fxt_media_service", "delete_media"),
            (
                "post",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/annotations",
                "fxt_dataset_service",
                "set_dataset_item_annotations",
            ),
            (
                "get",
                f"/api/projects/{uuid4()}/dataset/media/invalid-id/annotations",
                "fxt_dataset_service",
                "get_dataset_item_by_id",
            ),
        ],
    )
    def test_invalid_ids(
        self, request, http_method, http_path, service_name, service_method, fxt_get_project, fxt_client
    ):
        service = request.getfixturevalue(service_name)
        response = getattr(fxt_client, http_method)(http_path)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(service, service_method).assert_not_called()

    def test_get_media_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        fxt_media_service.get_media_by_id.side_effect = ResourceNotFoundError(ResourceType.MEDIA, str(media_id))

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)

    @pytest.mark.parametrize("media_name, duration", [("fxt_image_media", None), ("fxt_video_media", 40)])
    def test_get_media_success(self, request, media_name, duration, fxt_get_project, fxt_media_service, fxt_client):
        media = request.getfixturevalue(media_name)
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media.id)}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {
            "type": media.type,
            "format": media.format,
            "height": media.height,
            "id": str(media.id),
            "name": media.name,
            "size": media.size,
            "source_id": str(media.source_id),
            "width": media.width,
            "fps": media.fps,
            "frame_count": media.frame_count,
            "duration": duration,
        }
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media.id)

    def test_get_media_binary_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        fxt_media_service.get_media_binary_path_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(media_id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/binary")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_binary_path_by_id.assert_called_once_with(
            project_id=fxt_get_project.id, media_id=media_id
        )

    def test_get_media_binary_success(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()

        tmp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
                temp_file_path = tmp_file.name
                fxt_media_service.get_media_binary_path_by_id.return_value = temp_file_path
                response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/binary")

            assert response.status_code == status.HTTP_200_OK
            fxt_media_service.get_media_binary_path_by_id.assert_called_once_with(
                project_id=fxt_get_project.id, media_id=media_id
            )
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

    def test_get_media_thumbnail_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        fxt_media_service.generate_media_thumbnail.side_effect = ResourceNotFoundError(
            ResourceType.MEDIA, str(media_id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/thumbnail")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.generate_media_thumbnail.assert_called_once_with(project=fxt_get_project, media_id=media_id)

    def test_get_media_thumbnail_success(self, fxt_get_project, fxt_media_service, fxt_client):
        from PIL import Image

        media_id = uuid4()

        # Create a test PIL Image to return from the mock
        test_image = Image.new("RGB", (64, 64), color="blue")
        fxt_media_service.generate_media_thumbnail.return_value = test_image

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/thumbnail")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "image/jpeg"
        fxt_media_service.generate_media_thumbnail.assert_called_once_with(project=fxt_get_project, media_id=media_id)

    def test_delete_media_not_found(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()
        fxt_media_service.delete_media.side_effect = ResourceNotFoundError(ResourceType.DATASET_ITEM, str(media_id))

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.delete_media.assert_called_once_with(project=fxt_get_project, media_id=media_id)

    def test_delete_media_success(self, fxt_get_project, fxt_media_service, fxt_client):
        media_id = uuid4()

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_media_service.delete_media.assert_called_once_with(project=fxt_get_project, media_id=media_id)

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_set_media_annotations_success(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
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
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media_id,
            annotations=annotations,
            user_reviewed=True,
        )

    def test_set_video_annotations_missing_timestamp(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.set_dataset_item_annotations.assert_not_called()

    def test_set_video_annotations_existing_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        video_frame_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
        )
        fxt_media_service.get_frame_by_video_id_and_timestamp.return_value = video_frame
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations?ts=10",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
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
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_media_service.get_frame_by_video_id_and_timestamp.assert_called_once_with(video_id=media_id, timestamp=10)
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=video_frame_id,
            annotations=annotations,
            user_reviewed=True,
        )

    def test_set_video_annotations_extract_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        video_frame_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media
        fxt_media_service.get_frame_by_video_id_and_timestamp.return_value = None
        video_frame_media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO_FRAME,
        )
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
        )
        fxt_media_service.extract_video_frame.return_value = video_frame_media, video_frame
        dataset_item = MagicMock(
            spec=DatasetItem,
            annotation_data=annotations,
            user_reviewed=True,
            prediction_model_id=None,
        )
        fxt_dataset_service.set_dataset_item_annotations.return_value = dataset_item

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations?ts=10",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
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
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_media_service.get_frame_by_video_id_and_timestamp.assert_called_once_with(video_id=media_id, timestamp=10)
        fxt_media_service.extract_video_frame.assert_called_once_with(
            project=fxt_get_project, video_id=media_id, timestamp=10
        )
        fxt_dataset_service.create_dataset_item.assert_called_once_with(
            project=fxt_get_project,
            media=video_frame_media,
            user_reviewed=False,
        )
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=video_frame_id,
            annotations=annotations,
            user_reviewed=True,
        )

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_set_media_annotations_label_not_found(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.set_dataset_item_annotations.side_effect = AnnotationValidationError(str(label_id))

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media_id,
            annotations=annotations,
            user_reviewed=True,
        )

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_set_media_annotations_not_found(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        annotations = [
            DatasetItemAnnotation(
                labels=[LabelReference(id=label_id)],
                shape=Rectangle(type="rectangle", x=0, y=0, width=10, height=10),
            )
        ]
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.set_dataset_item_annotations.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(media_id)
        )

        response = fxt_client.post(
            f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations",
            json=SetMediaAnnotations(annotations=annotations).model_dump(mode="json"),
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_dataset_service.set_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media_id,
            annotations=annotations,
            user_reviewed=True,
        )

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_get_media_annotations(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media
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

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations")

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
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media_id,
        )

    def test_get_video_annotations_missing_timestamp(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_dataset_service.get_dataset_item_by_id.assert_not_called()

    def test_get_video_annotations_existing_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        video_frame_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
        )
        fxt_media_service.get_frame_by_video_id_and_timestamp.return_value = video_frame
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

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations?ts=10")

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
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_media_service.get_frame_by_video_id_and_timestamp.assert_called_once_with(video_id=media_id, timestamp=10)
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=video_frame_id,
        )

    def test_get_video_annotations_frame_not_found(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        label_id = uuid4()
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media
        fxt_media_service.get_frame_by_video_id_and_timestamp.return_value = None
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

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations?ts=10")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_media_service.get_frame_by_video_id_and_timestamp.assert_called_once_with(video_id=media_id, timestamp=10)
        fxt_dataset_service.get_dataset_item_by_id.assert_not_called()

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_get_media_annotations_not_found(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.get_dataset_item_by_id.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(media_id)
        )

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media_id,
        )

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_get_media_annotations_not_annotated(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media
        dataset_item = MagicMock(spec=DatasetItem, annotation_data=None, user_reviewed=False, prediction_model_id=None)
        fxt_dataset_service.get_dataset_item_by_id.return_value = dataset_item

        response = fxt_client.get(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.get_dataset_item_by_id.assert_called_once_with(
            project_id=fxt_get_project.id,
            dataset_item_id=media_id,
        )

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_delete_media_annotations(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media_id,
        )

    def test_delete_video_annotations_missing_timestamp(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_not_called()

    def test_delete_video_annotations_existing_frame(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        video_frame_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media
        video_frame = MagicMock(
            spec=VideoFrame,
            id=video_frame_id,
        )
        fxt_media_service.get_frame_by_video_id_and_timestamp.return_value = video_frame

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations?ts=10")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=video_frame_id,
        )

    def test_delete_video_annotations_frame_not_found(
        self, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=MediaType.VIDEO,
        )
        fxt_media_service.get_media_by_id.return_value = media
        fxt_media_service.get_frame_by_video_id_and_timestamp.return_value = None

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations?ts=10")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_not_called()

    @pytest.mark.parametrize("media_type", [MediaType.IMAGE, MediaType.VIDEO_FRAME])
    def test_delete_media_annotations_not_found(
        self, media_type, fxt_get_project, fxt_media_service, fxt_dataset_service, fxt_client
    ):
        media_id = uuid4()
        media = MagicMock(
            spec=Media,
            type=media_type,
        )
        fxt_media_service.get_media_by_id.return_value = media
        fxt_dataset_service.delete_dataset_item_annotations.side_effect = ResourceNotFoundError(
            ResourceType.DATASET_ITEM, str(media_id)
        )

        response = fxt_client.delete(f"/api/projects/{str(uuid4())}/dataset/media/{str(media_id)}/annotations")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_media_service.get_media_by_id.assert_called_once_with(project_id=fxt_get_project.id, media_id=media_id)
        fxt_dataset_service.delete_dataset_item_annotations.assert_called_once_with(
            project=fxt_get_project,
            dataset_item_id=media_id,
        )
