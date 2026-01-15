# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
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
def fxt_model_service(tmp_path: Path, db_session: Session) -> ModelService:
    """Fixture to create a ModelService instance."""
    return ModelService(data_dir=tmp_path, db_session=db_session)


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

    def test_get_model(self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService):
        """Test retrieving a model by ID."""
        model = fxt_model_service.get_model(fxt_project_id, fxt_model_id)

        assert model is not None
        assert model.id == fxt_model_id

    def test_update_model(self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService):
        """Test updating name of a model by ID."""
        new_model_name = "This is a new model name"
        model_metadata = {"name": new_model_name}
        model_from_get_before_update = fxt_model_service.get_model(fxt_project_id, fxt_model_id)
        model_from_update = fxt_model_service.rename_model(
            project_id=fxt_project_id, model_id=fxt_model_id, model_metadata=model_metadata
        )
        model_from_get_after_update = fxt_model_service.get_model(fxt_project_id, fxt_model_id)

        assert model_from_update.name == new_model_name
        assert model_from_get_before_update.name != new_model_name
        assert model_from_get_after_update.name == new_model_name

    @pytest.mark.parametrize("model_operation", ["get_model", "delete_model"])
    def test_non_existent_model(self, model_operation, fxt_project_id, fxt_db_projects, fxt_model_service, db_session):
        """Test retrieving a non-existent model raises error."""
        model_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            getattr(fxt_model_service, model_operation)(fxt_project_id, model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)

    def test_delete_model_no_files(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that deleting a model removes its database record when no filesystem artifacts exist."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)

        fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

        assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is None
        assert not model_rev_path.exists()

    def test_delete_model_with_files(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that deleting a model removes both its database record and filesystem artifacts."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_rev_path.mkdir(parents=True, exist_ok=True)
        (model_rev_path / "config.yaml").touch()
        (model_rev_path / "training.log").touch()

        fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

        assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is None
        assert not model_rev_path.exists()

    def test_delete_model_with_files_no_permission(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that delete_model fails when filesystem cleanup is denied due to OSError."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_rev_path.mkdir(parents=True, exist_ok=True)
        (model_rev_path / "config.yaml").touch()

        # Make directory read-only to simulate permission error
        model_rev_path.chmod(0o444)

        try:
            with pytest.raises(OSError):
                fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

            assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is not None
            assert model_rev_path.exists()
        finally:
            # Cleanup: restore permissions so pytest can clean up temp directory
            model_rev_path.chmod(0o755)
