# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import get_model_service
from app.api.schemas import ModelView
from app.main import app
from app.models import TrainingInfo, TrainingStatus
from app.services import ModelService, ResourceInUseError, ResourceNotFoundError, ResourceType


@pytest.fixture
def fxt_model() -> ModelView:
    return ModelView(
        id=uuid4(),
        architecture="Object_Detection_YOLOX",
        training_info=TrainingInfo(status=TrainingStatus.NOT_STARTED, label_schema_revision={}, configuration={}),  # type: ignore
    )  # type: ignore


@pytest.fixture
def fxt_model_service() -> MagicMock:
    model_service = MagicMock(spec=ModelService)
    app.dependency_overrides[get_model_service] = lambda: model_service
    return model_service


class TestModelEndpoints:
    def test_list_model_success(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        fxt_model_service.list_models.return_value = [fxt_model] * 2

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_model_service.list_models.assert_called_once_with(fxt_get_project.id)

    def test_list_model_project_not_found(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        project_id = uuid4()
        fxt_model_service.list_models.side_effect = ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.list_models.assert_called_once_with(fxt_get_project.id)

    def test_list_model_invalid_id(self, fxt_model, fxt_model_service, fxt_client):
        response = fxt_client.get("/api/projects/invalid-id/models")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.list_models.assert_not_called()

    def test_get_model_success(self, fxt_model, fxt_get_project, fxt_model_service, fxt_client):
        fxt_model_service.get_model.return_value = fxt_model

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_200_OK
        fxt_model_service.get_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=fxt_model.id)

    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_model"),
            ("delete", "delete_model"),
        ],
    )
    def test_model_invalid_ids(self, http_method, service_method, fxt_get_project, fxt_model_service, fxt_client):
        response = getattr(fxt_client, http_method)(f"/api/projects/{fxt_get_project.id}/models/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_model_service, service_method).assert_not_called()

    def test_get_model_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.get_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_delete_model_success(self, fxt_get_project, fxt_model, fxt_model_service, fxt_client):
        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=fxt_model.id)

    def test_delete_model_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.delete_model.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_delete_model_in_use(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        err = ResourceInUseError(ResourceType.MODEL, str(model_id))
        fxt_model_service.delete_model.side_effect = err

        response = fxt_client.delete(f"/api/projects/{fxt_get_project.id}/models/{model_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]
        fxt_model_service.delete_model.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_download_model_binary_success(self, fxt_get_project, fxt_model, fxt_model_service, fxt_client, tmp_path):
        import zipfile
        from io import BytesIO

        # Create mock model files
        model_dir = tmp_path / "models" / str(fxt_model.id)
        model_dir.mkdir(parents=True)
        xml_content = "<xml>model data</xml>"
        bin_content = b"binary model data"
        (model_dir / "model.xml").write_text(xml_content)
        (model_dir / "model.bin").write_bytes(bin_content)

        fxt_model_service.get_model_files_path.return_value = model_dir

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{fxt_model.id}/binary")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/zip"
        assert "content-disposition" in response.headers
        assert f"model-{fxt_model.id}-fp16.zip" in response.headers["content-disposition"]

        # Verify zip file contents
        zip_data = BytesIO(response.content)
        with zipfile.ZipFile(zip_data, "r") as zip_file:
            assert "model.xml" in zip_file.namelist()
            assert "model.bin" in zip_file.namelist()
            assert zip_file.read("model.xml").decode() == xml_content
            assert zip_file.read("model.bin") == bin_content

        fxt_model_service.get_model_files_path.assert_called_once_with(
            project_id=fxt_get_project.id, model_id=fxt_model.id
        )

    def test_download_model_binary_not_found(self, fxt_get_project, fxt_model_service, fxt_client):
        model_id = uuid4()
        fxt_model_service.get_model_files_path.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/{model_id}/binary")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model_files_path.assert_called_once_with(project_id=fxt_get_project.id, model_id=model_id)

    def test_download_model_binary_invalid_id(self, fxt_get_project, fxt_model_service, fxt_client):
        response = fxt_client.get(f"/api/projects/{fxt_get_project.id}/models/invalid-id/binary")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.get_model.assert_not_called()
