# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB, ProjectDB
from app.services import ModelService, ResourceNotFoundError, ResourceType
from tests.integration.project_factory import ProjectTestDataFactory


@pytest.fixture(autouse=True)
def setup_project_with_models(
    fxt_db_projects: list[ProjectDB],
    fxt_db_models: list[ModelRevisionDB],
    db_session: Session,
) -> None:
    """Fixture to set up a project with dataset items in the database."""

    (
        ProjectTestDataFactory(db_session)
        .with_project(fxt_db_projects[0])
        .with_pipeline(is_running=True)
        .with_models(fxt_db_models)
        .build()
    )


@pytest.fixture
def fxt_model_service(db_session: Session) -> ModelService:
    """Fixture to create a ModelService instance."""
    return ModelService(db_session=db_session)


class TestModelServiceIntegration:
    """Integration tests for ModelService."""

    def test_list_models(
        self, fxt_project_id: UUID, fxt_db_models: list[ModelRevisionDB], fxt_model_service: ModelService
    ):
        """Test retrieving all models."""

        models = fxt_model_service.list_models(fxt_project_id)

        assert len(models) == len(fxt_db_models)
        model_ids = [str(m.id) for m in models]
        for idx in range(len(model_ids)):
            assert fxt_db_models[idx].id in model_ids

    @pytest.mark.parametrize("model_operation", ["list_models", "get_model_by_id", "delete_model_by_id"])
    def test_model_with_non_existent_project(self, model_operation: str, fxt_model_service: ModelService):
        """Test deleting a model from non-existent project raises error."""
        project_id, model_id = uuid4(), uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            if model_operation == "list_models":
                getattr(fxt_model_service, model_operation)(project_id)
            else:
                getattr(fxt_model_service, model_operation)(project_id, model_id)

        assert excinfo.value.resource_type == ResourceType.PROJECT
        assert excinfo.value.resource_id == str(project_id)

    def test_get_model(self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService):
        """Test retrieving a model by ID."""
        model = fxt_model_service.get_model_by_id(fxt_project_id, fxt_model_id)

        assert model is not None
        assert model.id == fxt_model_id

    @pytest.mark.parametrize("model_operation", ["get_model_by_id", "delete_model_by_id"])
    def test_non_existent_model(self, model_operation, fxt_db_projects, fxt_model_service, db_session):
        """Test retrieving a non-existent model raises error."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        project_id, model_id = UUID(db_project.id), uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            getattr(fxt_model_service, model_operation)(project_id, model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)

    def test_delete_model(
        self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService, db_session: Session
    ):
        """Test deleting a model by ID."""
        fxt_model_service.delete_model_by_id(fxt_project_id, fxt_model_id)

        assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is None
