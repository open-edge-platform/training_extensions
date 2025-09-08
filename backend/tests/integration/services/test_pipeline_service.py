# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import uuid4

import pytest

from app.db.schema import PipelineDB, ProjectDB
from app.schemas import PipelineStatus
from app.services import PipelineService, ResourceNotFoundError, ResourceType
from app.services.configuration_service import PipelineField


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with (
        patch("app.services.pipeline_service.get_db_session") as mock,
        patch("app.services.base.get_db_session") as mock_base,
    ):
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        mock_base.return_value.__enter__.return_value = db_session
        mock_base.return_value.__exit__.return_value = None
        yield


@pytest.fixture
def fxt_pipeline_service(fxt_active_pipeline_service, fxt_metrics_service, fxt_condition) -> PipelineService:
    """Fixture to create a PipelineService instance with mocked dependencies."""
    return PipelineService(fxt_active_pipeline_service, fxt_metrics_service, fxt_condition)


class TestPipelineServiceIntegration:
    """Integration tests for PipelineService."""

    def test_get_pipeline(self, fxt_pipeline_service, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session):
        """Test retrieving a pipeline by ID."""
        project_id = uuid4()
        project_db = ProjectDB(
            id=str(project_id),
            name="Test Project",
            task_type="detection",
            labels=["cat", "dog"],
        )
        pipeline_db = PipelineDB(project_id=str(project_id))
        pipeline_db.sink = fxt_db_sinks[0]
        pipeline_db.source = fxt_db_sources[0]
        pipeline_db.model = fxt_db_models[0]
        project_db.pipeline = pipeline_db
        db_session.add(project_db)
        db_session.flush()

        pipeline = fxt_pipeline_service.get_pipeline_by_id(project_id)

        assert pipeline is not None
        assert pipeline.project_id == project_id
        assert pipeline.status == PipelineStatus.IDLE
        assert pipeline.sink.name == fxt_db_sinks[0].name
        assert pipeline.source.name == fxt_db_sources[0].name
        assert pipeline.model.name == fxt_db_models[0].name

    def test_get_non_existent_pipeline(self, fxt_pipeline_service):
        """Test retrieving a non-existent pipeline raises error."""
        pipeline_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_pipeline_service.get_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)

    @pytest.mark.parametrize("pipeline_attr", [PipelineField.SINK_ID, PipelineField.SOURCE_ID])
    def test_reconfigure_running_pipeline(
        self,
        pipeline_attr,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_db_models,
        fxt_db_projects,
        fxt_pipeline_service,
        fxt_active_pipeline_service,
        fxt_condition,
        db_session,
    ):
        """Test updating a pipeline by ID."""
        db_project = fxt_db_projects[0]
        db_pipeline = db_project.pipeline
        db_pipeline.is_running = True
        db_pipeline.source = fxt_db_sources[0]
        db_pipeline.sink = fxt_db_sinks[0]
        db_pipeline.model = fxt_db_models[0]
        db_session.add(db_project)
        db_session.add(fxt_db_sources[1])
        db_session.add(fxt_db_sinks[1])
        db_session.flush()

        if pipeline_attr == PipelineField.SINK_ID:
            item_id = fxt_db_sinks[1].id
        else:
            item_id = fxt_db_sources[1].id

        updated = fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {str(pipeline_attr): item_id})

        if pipeline_attr == PipelineField.SINK_ID:
            fxt_active_pipeline_service.reload.assert_called_once()
            fxt_condition.notify_all.assert_not_called()
        else:
            fxt_active_pipeline_service.reload.assert_not_called()
            fxt_condition.notify_all.assert_called_once()
        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        assert str(getattr(updated, pipeline_attr)) == item_id
        assert str(getattr(updated, pipeline_attr)) == getattr(db_updated, pipeline_attr)

    @pytest.mark.parametrize("pipeline_status", [PipelineStatus.IDLE, PipelineStatus.RUNNING])
    def test_enable_disable_pipeline(
        self,
        pipeline_status,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_db_models,
        fxt_db_projects,
        fxt_pipeline_service,
        fxt_active_pipeline_service,
        fxt_condition,
        db_session,
    ):
        """Test activating and deactivating an existing pipeline by ID."""
        db_project = fxt_db_projects[0]
        db_pipeline = db_project.pipeline
        db_pipeline.is_running = not pipeline_status.as_bool
        db_pipeline.source = fxt_db_sources[0]
        db_pipeline.sink = fxt_db_sinks[0]
        db_pipeline.model = fxt_db_models[0]
        db_session.add(db_project)
        db_session.flush()

        fxt_pipeline_service.update_pipeline(db_pipeline.project_id, {"status": pipeline_status})

        fxt_active_pipeline_service.reload.assert_called_once()
        fxt_condition.notify_all.assert_called_once()
        db_updated = db_session.get(PipelineDB, db_pipeline.project_id)
        if pipeline_status == PipelineStatus.RUNNING:
            assert db_updated.is_running
        else:
            assert not db_updated.is_running

    def test_reconfigure_non_existent_pipeline(self, fxt_pipeline_service):
        """Test updating a non-existent pipeline raises error."""
        project_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_pipeline_service.update_pipeline(project_id, {"sink_id": uuid4()})

        assert exc_info.value.resource_type == ResourceType.PIPELINE
        assert exc_info.value.resource_id == str(project_id)
