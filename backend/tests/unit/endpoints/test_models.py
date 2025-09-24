# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import get_model_service
from app.main import app
from app.schemas.model import Model, TrainingInfo, TrainingStatus
from app.services import ModelService, ResourceInUseError, ResourceNotFoundError, ResourceType


@pytest.fixture
def fxt_model() -> Model:
    return Model(
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
    def test_list_model_success(self, fxt_model, fxt_model_service, fxt_client):
        project_id = uuid4()
        fxt_model_service.list_models.return_value = [fxt_model] * 2

        response = fxt_client.get(f"/api/projects/{project_id}/models")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_model_service.list_models.assert_called_once_with(project_id)

    def test_list_model_not_found(self, fxt_model, fxt_model_service, fxt_client):
        project_id = uuid4()
        fxt_model_service.list_models.side_effect = ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        response = fxt_client.get(f"/api/projects/{project_id}/models")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.list_models.assert_called_once_with(project_id)

    def test_list_model_invalid_id(self, fxt_model, fxt_model_service, fxt_client):
        response = fxt_client.get("/api/projects/invalid-id/models")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_model_service.list_models.assert_not_called()

    def test_get_model_success(self, fxt_model, fxt_model_service, fxt_client):
        project_id = uuid4()
        fxt_model_service.get_model_by_id.return_value = fxt_model

        response = fxt_client.get(f"/api/projects/{project_id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_200_OK
        fxt_model_service.get_model_by_id.assert_called_once_with(project_id=project_id, model_id=fxt_model.id)

    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_model_by_id"),
            ("delete", "delete_model_by_id"),
        ],
    )
    def test_model_invalid_ids(self, http_method, service_method, fxt_model_service, fxt_client):
        response = getattr(fxt_client, http_method)(f"/api/projects/{uuid4()}/models/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_model_service, service_method).assert_not_called()

    def test_get_model_not_found(self, fxt_model_service, fxt_client):
        project_id, model_id = uuid4(), uuid4()
        fxt_model_service.get_model_by_id.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.get(f"/api/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.get_model_by_id.assert_called_once_with(project_id=project_id, model_id=model_id)

    def test_delete_model_success(self, fxt_model, fxt_model_service, fxt_client):
        project_id = uuid4()

        response = fxt_client.delete(f"/api/projects/{project_id}/models/{fxt_model.id}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_model_service.delete_model_by_id.assert_called_once_with(project_id=project_id, model_id=fxt_model.id)

    def test_delete_model_not_found(self, fxt_model_service, fxt_client):
        project_id, model_id = uuid4(), uuid4()
        fxt_model_service.delete_model_by_id.side_effect = ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        response = fxt_client.delete(f"/api/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_model_service.delete_model_by_id.assert_called_once_with(project_id=project_id, model_id=model_id)

    def test_delete_model_in_use(self, fxt_model_service, fxt_client):
        project_id, model_id = uuid4(), uuid4()
        err = ResourceInUseError(ResourceType.MODEL, str(model_id))
        fxt_model_service.delete_model_by_id.side_effect = err

        response = fxt_client.delete(f"/api/projects/{project_id}/models/{model_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]
        fxt_model_service.delete_model_by_id.assert_called_once_with(project_id=project_id, model_id=model_id)
