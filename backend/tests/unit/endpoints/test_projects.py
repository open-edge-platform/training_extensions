# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import get_project_service
from app.main import app
from app.schemas import Project
from app.schemas.project import Label, Task
from app.services import (
    ProjectService,
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
)


@pytest.fixture
def fxt_project() -> Project:
    return Project(
        id=uuid4(),
        name="Test Project",
        task=Task(
            task_type="classification",
            exclusive_labels=True,
            labels=[Label(name="cat"), Label(name="dog")],
        ),
    )


@pytest.fixture
def fxt_project_service() -> MagicMock:
    project_service = MagicMock(spec=ProjectService)
    app.dependency_overrides[get_project_service] = lambda: project_service
    return project_service


class TestProjectEndpoints:
    @pytest.mark.parametrize(
        "http_method, service_method",
        [
            ("get", "get_project_by_id"),
            ("delete", "delete_project_by_id"),
        ],
    )
    def test_project_invalid_ids(self, http_method, service_method, fxt_project_service, fxt_client):
        response = getattr(fxt_client, http_method)("/api/projects/invalid-id")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        getattr(fxt_project_service, service_method).assert_not_called()

    def test_list_project_success(self, fxt_project, fxt_project_service, fxt_client):
        fxt_project_service.list_projects.return_value = [fxt_project] * 2

        response = fxt_client.get("/api/projects")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        fxt_project_service.list_projects.assert_called_once()

    def test_get_project_success(self, fxt_project, fxt_project_service, fxt_client):
        fxt_project_service.get_project_by_id.return_value = fxt_project

        response = fxt_client.get(f"/api/projects/{str(fxt_project.id)}")

        assert response.status_code == status.HTTP_200_OK
        fxt_project_service.get_project_by_id.assert_called_once_with(fxt_project.id)

    def test_create_project_success(self, fxt_project, fxt_project_service, fxt_client):
        fxt_project_service.create_project.return_value = fxt_project

        response = fxt_client.post("/api/projects", json=fxt_project.model_dump(mode="json", exclude={"id"}))

        assert response.status_code == status.HTTP_201_CREATED
        fxt_project_service.create_project.assert_called_once()

    def test_create_project_exists(self, fxt_project_service, fxt_project, fxt_client):
        fxt_project_service.create_project.side_effect = ResourceAlreadyExistsError(
            resource_type=ResourceType.PROJECT, resource_name="New Project"
        )
        response = fxt_client.post("/api/projects", json=fxt_project.model_dump(mode="json", exclude={"id"}))

        assert response.status_code == status.HTTP_409_CONFLICT
        fxt_project_service.create_project.assert_called_once()

    def test_create_project_invalid(self, fxt_project_service, fxt_client):
        response = fxt_client.post("/api/projects", json={"name": "New Pipeline", "attr": "invalid"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        fxt_project_service.create_project.assert_not_called()

    def test_delete_project_success(self, fxt_project, fxt_project_service, fxt_client):
        response = fxt_client.delete(f"/api/projects/{str(fxt_project.id)}")

        assert response.status_code == status.HTTP_204_NO_CONTENT
        fxt_project_service.delete_project_by_id.assert_called_once_with(fxt_project.id)

    def test_delete_project_not_found(self, fxt_project_service, fxt_client):
        project_id = str(uuid4())
        fxt_project_service.delete_project_by_id.side_effect = ResourceNotFoundError(ResourceType.PROJECT, project_id)

        response = fxt_client.delete(f"/api/projects/{project_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_project_in_use(self, fxt_project, fxt_project_service, fxt_client):
        project_id = str(fxt_project.id)
        err = ResourceInUseError(ResourceType.PROJECT, project_id)
        fxt_project_service.delete_project_by_id.side_effect = err

        response = fxt_client.delete(f"/api/projects/{project_id}")

        assert response.status_code == status.HTTP_409_CONFLICT
        assert str(err) == response.json()["detail"]
