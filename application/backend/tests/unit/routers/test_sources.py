# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import yaml
from fastapi import status

from app.api.dependencies import get_source_update_service
from app.main import app
from app.schemas import SourceType
from app.schemas.source import VideoFileSourceConfig, WebcamSourceConfig
from app.services import (
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
    ResourceWithNameAlreadyExistsError,
    SourceUpdateService,
)


@pytest.fixture
def fxt_webcam_source() -> WebcamSourceConfig:
    return WebcamSourceConfig(
        id=uuid4(), source_type=SourceType.WEBCAM, name="Test Webcam Source", device_id=1, codec="YUY2"
    )


@pytest.fixture
def fxt_video_source() -> VideoFileSourceConfig:
    return VideoFileSourceConfig(
        id=uuid4(),
        source_type=SourceType.VIDEO_FILE,
        name="Test Folder Source",
        video_path="/test/video/path.mp4",
    )


@pytest.fixture
def fxt_source_update_service() -> MagicMock:
    source_update_service = MagicMock(spec=SourceUpdateService)
    app.dependency_overrides[get_source_update_service] = lambda: source_update_service
    return source_update_service


class TestSourceEndpoints:
    def test_create_source_success(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        source_id = str(fxt_webcam_source.id)
        fxt_source_update_service.create.return_value = fxt_webcam_source

        response = fxt_client.post("/api/sources", json=fxt_webcam_source.model_dump(exclude={"id"}))

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["id"] == source_id
        assert response.json()["name"] == fxt_webcam_source.name
        fxt_source_update_service.create.assert_called_once()

    def test_create_source_disconnected_fails(self, fxt_source_update_service, fxt_client):
        response = fxt_client.post(
            "/api/sources", json={"source_type": SourceType.DISCONNECTED, "name": "Disconnected Source"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_source_update_service.create.assert_not_called()

    def test_create_source_validation_error(self, fxt_source_update_service, fxt_client):
        response = fxt_client.post("/api/sources", json={"name": ""})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_source_update_service.create.assert_not_called()

    def test_create_source_exists(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        fxt_source_update_service.create.side_effect = ResourceWithNameAlreadyExistsError(
            resource_type=ResourceType.SOURCE, resource_name="New Config"
        )
        response = fxt_client.post("/api/sources", json=fxt_webcam_source.model_dump(exclude={"id"}))

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_source_update_service.create.assert_called_once()

    def test_list_sources(self, fxt_webcam_source, fxt_video_source, fxt_source_update_service, fxt_client, request):
        fxt_source_update_service.list_all.return_value = [fxt_webcam_source, fxt_video_source]

        response = fxt_client.get("/api/sources")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_source_update_service.list_all.assert_called_once()

    def test_get_source_success(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        source_id = str(fxt_webcam_source.id)
        fxt_source_update_service.get_by_id.return_value = fxt_webcam_source

        response = fxt_client.get(f"/api/sources/{source_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == source_id
        fxt_source_update_service.get_by_id.assert_called_once_with(fxt_webcam_source.id)

    def test_get_source_not_found(self, fxt_source_update_service, fxt_client):
        source_id = uuid4()
        fxt_source_update_service.get_by_id.side_effect = ResourceNotFoundError(ResourceType.SOURCE, str(source_id))

        response = fxt_client.get(f"/api/sources/{str(source_id)}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_source_update_service.get_by_id.assert_called_once_with(source_id)

    def test_get_source_invalid_uuid(self, fxt_client):
        response = fxt_client.get("/api/sources/invalid-uuid")
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_update_source_success(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        update_data = {"device_id": 5}
        source_id = str(fxt_webcam_source.id)
        updated_config = fxt_webcam_source.model_copy(update=update_data)
        fxt_source_update_service.get_by_id.return_value = fxt_webcam_source
        fxt_source_update_service.update.return_value = updated_config

        response = fxt_client.patch(f"/api/sources/{source_id}", json=update_data)

        assert response.status_code == status.HTTP_200_OK
        key, value = next(iter(update_data.items()))
        assert response.json()[key] == value
        fxt_source_update_service.update.assert_called_once_with(fxt_webcam_source, update_data)

    def test_update_source_not_found(self, fxt_source_update_service, fxt_client):
        source_id = str(uuid4())
        fxt_source_update_service.update.side_effect = ResourceNotFoundError(ResourceType.SOURCE, source_id)

        response = fxt_client.patch(f"/api/sources/{source_id}", json={"name": "Updated"})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_source_type_forbidden(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        source_id = str(fxt_webcam_source.id)

        response = fxt_client.patch(f"/api/sources/{source_id}", json={"source_type": "folder"})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "_type" in response.json()["detail"]
        fxt_source_update_service.update.assert_not_called()

    def test_delete_source_success(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        source_id = str(fxt_webcam_source.id)
        fxt_source_update_service.delete_by_id.side_effect = None

        response = fxt_client.delete(f"/api/sources/{source_id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_source_update_service.delete_by_id.assert_called_once_with(fxt_webcam_source.id)

    def test_delete_source_invalid_id(self, fxt_source_update_service, fxt_client):
        fxt_source_update_service.delete_by_id.side_effect = None

        response = fxt_client.delete("/api/sources/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_source_update_service.delete_by_id.assert_not_called()

    def test_delete_source_not_found(self, fxt_source_update_service, fxt_client):
        source_id = str(uuid4())
        fxt_source_update_service.delete_by_id.side_effect = ResourceNotFoundError(ResourceType.SOURCE, source_id)

        response = fxt_client.delete(f"/api/sources/{source_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_source_in_use(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        source_id = str(fxt_webcam_source.id)
        err = ResourceInUseError(ResourceType.SOURCE, source_id)
        fxt_source_update_service.delete_by_id.side_effect = err

        response = fxt_client.delete(f"/api/sources/{source_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]

    def test_export_source_success(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        source_id = str(fxt_webcam_source.id)
        fxt_source_update_service.get_by_id.return_value = fxt_webcam_source

        response = fxt_client.post(f"/api/sources/{source_id}:export")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/x-yaml"
        assert f"source_{source_id}.yaml" in response.headers["content-disposition"]
        assert response.text == "codec: YUY2\ndevice_id: 1\nname: Test Webcam Source\nsource_type: webcam\n"
        fxt_source_update_service.get_by_id.assert_called_once_with(fxt_webcam_source.id)

    def test_import_source_success(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        sink_data = fxt_webcam_source.model_dump(exclude={"id"}, mode="json")
        yaml_content = yaml.safe_dump(sink_data)
        fxt_source_update_service.create.return_value = fxt_webcam_source

        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}
        response = fxt_client.post("/api/sources:import", files=files)

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json()["id"] == str(fxt_webcam_source.id)
        fxt_source_update_service.create.assert_called_once()

    def test_import_source_exists(self, fxt_webcam_source, fxt_source_update_service, fxt_client):
        sink_data = fxt_webcam_source.model_dump(exclude={"id"}, mode="json")
        yaml_content = yaml.safe_dump(sink_data)
        fxt_source_update_service.create.side_effect = ResourceWithNameAlreadyExistsError(
            resource_type=ResourceType.SOURCE, resource_name="New Config"
        )

        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}
        response = fxt_client.post("/api/sources:import", files=files)

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_source_update_service.create.assert_called_once()

    def test_import_source_invalid_yaml(self, fxt_client):
        invalid_yaml = "invalid: yaml: content: ["
        files = {"yaml_file": ("test.yaml", io.BytesIO(invalid_yaml.encode()), "application/x-yaml")}

        response = fxt_client.post("/api/sources:import", files=files)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Invalid YAML format" in response.json()["detail"]

    def test_import_disconnected_source_fails(self, fxt_source_update_service, fxt_client):
        config_data = {"source_type": "disconnected", "name": "Test"}
        yaml_content = yaml.safe_dump(config_data)
        files = {"yaml_file": ("test.yaml", io.BytesIO(yaml_content.encode()), "application/x-yaml")}

        response = fxt_client.post("/api/sources:import", files=files)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_source_update_service.create.assert_not_called()
