# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import status

from app.api.dependencies import get_label_service, get_project_service
from app.main import app
from app.schemas import Label, PatchLabels, Project
from app.schemas.label import LabelToAdd, LabelToEdit, LabelToRemove
from app.schemas.project import Task, TaskType
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
            task_type=TaskType.CLASSIFICATION,
            exclusive_labels=True,
            labels=[Label(name="cat", color="#11AA22", hotkey="s"), Label(name="dog", color="#AA2233", hotkey="d")],
        ),
    )


@pytest.fixture
def fxt_project_service() -> MagicMock:
    project_service = MagicMock(spec=ProjectService)
    app.dependency_overrides[get_project_service] = lambda: project_service
    return project_service


@pytest.fixture
def fxt_label_service():
    with patch("app.services.LabelService") as MockLabelService:
        mock_service = MockLabelService.return_value
        app.dependency_overrides[get_label_service] = lambda: mock_service
        return mock_service


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

    def test_update_labels_success(self, fxt_project, fxt_project_service, fxt_label_service, fxt_client):
        """Test successful label update with add, edit, and remove operations."""
        # Setup
        labels_to_add = [LabelToAdd(name="mouse", color="#0000FF", hotkey="m")]
        labels_to_edit = [LabelToEdit(id=fxt_project.task.labels[0].id, new_name="updated_cat")]  # type: ignore[call-arg]
        labels_to_remove = [LabelToRemove(id=fxt_project.task.labels[1].id)]

        patch_labels = PatchLabels(
            labels_to_add=labels_to_add, labels_to_edit=labels_to_edit, labels_to_remove=labels_to_remove
        )

        expected_labels = [
            Label(id=fxt_project.task.labels[0].id, name="updated_cat"),
            Label(id=uuid4(), name="mouse", color="#0000FF", hotkey="m"),
        ]

        fxt_project_service.get_project_by_id.return_value = fxt_project
        fxt_label_service.update_labels_in_project.return_value = expected_labels

        # Execute
        response = fxt_client.patch(
            f"/api/projects/{str(fxt_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        # Assert
        assert response.status_code == status.HTTP_200_OK
        fxt_project_service.get_project_by_id.assert_called_once_with(fxt_project.id)
        fxt_label_service.update_labels_in_project.assert_called_once()
        assert len(response.json()) == 2

    def test_update_labels_project_not_found(self, fxt_project, fxt_project_service, fxt_label_service, fxt_client):
        """Test updating labels for non-existent project returns 404."""
        project_id = uuid4()
        fxt_project_service.get_project_by_id.side_effect = ResourceNotFoundError(ResourceType.PROJECT, str(project_id))

        response = fxt_client.patch(
            f"/api/projects/{str(project_id)}/labels", json=PatchLabels().model_dump(mode="json")
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        fxt_label_service.update_labels_in_project.assert_not_called()

    def test_update_labels_conflict(self, fxt_project, fxt_project_service, fxt_label_service, fxt_client):
        """Test label update with duplicate attributes returns 409."""
        labels_to_add = [LabelToAdd(name="cat", color="#FF0000", hotkey="c")]
        patch_labels = PatchLabels(labels_to_add=labels_to_add)

        fxt_project_service.get_project_by_id.return_value = fxt_project
        fxt_label_service.update_labels_in_project.side_effect = ResourceAlreadyExistsError(
            ResourceType.LABEL, "", message="Label with the same name or hotkey or color already exists"
        )

        response = fxt_client.patch(
            f"/api/projects/{str(fxt_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        assert "already exists" in response.json()["detail"]

    @pytest.mark.parametrize(
        "patch_labels",
        [
            PatchLabels(labels_to_remove=[LabelToRemove(id=uuid4())]),  # Non-existent label to remove
            PatchLabels(labels_to_edit=[LabelToEdit(id=uuid4(), new_name="updated")]),  # type: ignore[call-arg] # Non-existent label to edit
        ],
    )
    def test_update_labels_remove_edit_nonexistent(
        self, patch_labels, fxt_project, fxt_project_service, fxt_label_service, fxt_client
    ):
        """Test editing or removing labels that don't exist returns 404."""
        fxt_project_service.get_project_by_id.return_value = fxt_project

        response = fxt_client.patch(
            f"/api/projects/{str(fxt_project.id)}/labels", json=patch_labels.model_dump(mode="json")
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "do not exist" in response.json()["detail"]
        fxt_label_service.update_labels_in_project.assert_not_called()

    def test_update_labels_invalid_project_id(self, fxt_project_service, fxt_client):
        """Test update with invalid project ID returns 400."""
        response = fxt_client.patch("/api/projects/invalid-id/labels", json={})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_project_service.get_project_by_id.assert_not_called()
