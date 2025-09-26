# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from app.db.schema import PipelineDB, ProjectDB
from app.schemas import PipelineStatus
from app.schemas.pipeline import FixedRateDataCollectionPolicy
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


@pytest.fixture
def fxt_project_with_pipeline(
    fxt_db_projects, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session
) -> Callable[[bool, list[dict] | None], tuple[ProjectDB, PipelineDB]]:
    """Fixture to create a ProjectDB with an associated PipelineDB."""

    def _create_project_with_pipeline(
        is_running: bool, data_policies: list[dict] | None = None
    ) -> tuple[ProjectDB, PipelineDB]:
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()
        db_pipeline = db_project.pipeline
        db_pipeline.is_running = is_running
        db_pipeline.source = fxt_db_sources[0]
        db_pipeline.sink = fxt_db_sinks[0]
        if data_policies:
            db_pipeline.data_collection_policies = data_policies
        db_model_rev = fxt_db_models[0]
        db_model_rev.project_id = db_project.id
        db_pipeline.model_revision = db_model_rev
        db_session.add_all([db_pipeline, fxt_db_sources[1], fxt_db_sinks[1]])
        db_session.flush()
        return db_project, db_pipeline

    return _create_project_with_pipeline


class TestPipelineServiceIntegration:
    """Integration tests for PipelineService."""

    def test_get_pipeline(self, fxt_pipeline_service, fxt_project_with_pipeline, db_session):
        """Test retrieving a pipeline by ID."""
        _, db_pipeline = fxt_project_with_pipeline(
            is_running=False, data_policies=[{"type": "fixed_rate", "enabled": "true", "rate": 0.1}]
        )

        project_id = UUID(db_pipeline.project_id)
        pipeline = fxt_pipeline_service.get_pipeline_by_id(project_id)

        assert pipeline is not None
        assert pipeline.project_id == project_id
        assert pipeline.status == PipelineStatus.IDLE
        assert pipeline.sink.name == db_pipeline.sink.name
        assert pipeline.source.name == db_pipeline.source.name
        assert str(pipeline.model.id) == db_pipeline.model_revision_id
        assert pipeline.data_collection_policies == [FixedRateDataCollectionPolicy(rate=0.1)]

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
        fxt_project_with_pipeline,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_pipeline_service,
        fxt_active_pipeline_service,
        fxt_condition,
        db_session,
    ):
        """Test updating a pipeline by ID."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=True)

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
        fxt_project_with_pipeline,
        fxt_pipeline_service,
        fxt_active_pipeline_service,
        fxt_condition,
        db_session,
    ):
        """Test activating and deactivating an existing pipeline by ID."""
        _, db_pipeline = fxt_project_with_pipeline(is_running=not pipeline_status.as_bool)

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

    def test_set_pipeline_dataset_collection_policy(
        self, fxt_pipeline_service, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session
    ):
        """Test setting pipeline dataset collection policy."""
        project_id = uuid4()
        project_db = ProjectDB(
            id=str(project_id),
            name="Test Project",
            task_type="detection",
        )
        pipeline_db = PipelineDB(project_id=str(project_id))
        pipeline_db.sink = fxt_db_sinks[0]
        pipeline_db.source = fxt_db_sources[0]
        pipeline_db.model = fxt_db_models[0]
        project_db.pipeline = pipeline_db
        db_session.add(project_db)
        db_session.flush()

        pipeline = fxt_pipeline_service.update_pipeline(
            project_id=project_id,
            partial_config={"data_collection_policies": [FixedRateDataCollectionPolicy(type="fixed_rate", rate=0.1)]},
        )

        assert pipeline is not None
        assert pipeline.data_collection_policies == [FixedRateDataCollectionPolicy(type="fixed_rate", rate=0.1)]

    def test_reset_pipeline_dataset_collection_policy(
        self, fxt_pipeline_service, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session
    ):
        """Test resetting pipeline dataset collection policy."""
        project_id = uuid4()
        project_db = ProjectDB(
            id=str(project_id),
            name="Test Project",
            task_type="detection",
        )
        pipeline_db = PipelineDB(project_id=str(project_id))
        pipeline_db.sink = fxt_db_sinks[0]
        pipeline_db.source = fxt_db_sources[0]
        pipeline_db.model = fxt_db_models[0]
        pipeline_db.data_collection_policies = [{"type": "fixed_rate", "enabled": "true", "rate": 0.1}]
        project_db.pipeline = pipeline_db
        db_session.add(project_db)
        db_session.flush()

        pipeline = fxt_pipeline_service.update_pipeline(
            project_id=project_id, partial_config={"data_collection_policies": []}
        )

        assert pipeline is not None
        assert pipeline.data_collection_policies == []
