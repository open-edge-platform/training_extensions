# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from app.db.schema import PipelineDB
from app.schemas import Pipeline, PipelineStatus
from app.services import (
    PipelineService,
    ResourceAlreadyExistsError,
    ResourceInUseError,
    ResourceNotFoundError,
    ResourceType,
)
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
def fxt_pipeline_service(fxt_active_pipeline_service, fxt_condition) -> PipelineService:
    """Fixture to create a PipelineService instance with mocked dependencies."""
    return PipelineService(fxt_active_pipeline_service, fxt_condition)


class TestPipelineServiceIntegration:
    """Integration tests for PipelineService."""

    def test_list_pipelines(self, fxt_pipeline_service, db_session):
        """Test retrieving all pipelines."""
        db_pipeline1 = PipelineDB(
            id=str(uuid4()),
            name="Test Pipeline 1",
            is_running=False,
        )
        db_pipeline2 = PipelineDB(
            id=str(uuid4()),
            name="Test Pipeline 2",
            is_running=False,
        )
        db_session.add(db_pipeline1)
        db_session.add(db_pipeline2)
        db_session.flush()

        pipelines = fxt_pipeline_service.list_pipelines()

        assert len(pipelines) == 2
        for i, db_pipeline in enumerate([db_pipeline1, db_pipeline2]):
            assert str(pipelines[i].id) == db_pipeline.id

    def test_get_pipeline(self, fxt_pipeline_service, db_session):
        """Test retrieving a pipeline by ID."""
        pipeline_id = uuid4()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=False,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        pipeline = fxt_pipeline_service.get_pipeline_by_id(pipeline_id)

        assert pipeline is not None
        assert pipeline.id == pipeline_id
        assert pipeline.name == pipeline_db.name
        assert pipeline.status == PipelineStatus.from_bool(pipeline_db.is_running)

    def test_get_non_existent_pipeline(self, fxt_pipeline_service):
        """Test retrieving a non-existent pipeline raises error."""
        pipeline_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_pipeline_service.get_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)

    def test_create_pipeline(self, fxt_pipeline_service, db_session):
        """Test creating a new pipeline."""
        pipeline = Pipeline(
            name="Test Pipeline",
            status=PipelineStatus.IDLE,
        )
        created = fxt_pipeline_service.create_pipeline(pipeline)

        db_created = db_session.get(PipelineDB, str(created.id))
        assert db_created.name == "Test Pipeline"
        assert not db_created.is_running

    def test_create_pipeline_non_unique(
        self,
        fxt_pipeline_service,
        db_session,
    ):
        """Test creating a new pipeline with the name that already exists."""
        pipeline_id = uuid4()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=False,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        with pytest.raises(ResourceAlreadyExistsError) as excinfo:
            fxt_pipeline_service.create_pipeline(Pipeline(name="Test Pipeline"))

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == "Test Pipeline"

    def test_create_active_pipeline(
        self,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_db_models,
        fxt_pipeline_service,
        fxt_active_pipeline_service,
        fxt_condition,
        db_session,
    ):
        """Test creating a new active pipeline."""
        db_session.add(fxt_db_sinks[0])
        db_session.add(fxt_db_sources[0])
        db_session.add(fxt_db_models[0])
        db_session.flush()

        pipeline = Pipeline(
            name="Test Pipeline",
            status=PipelineStatus.RUNNING,
            source_id=UUID(fxt_db_sources[0].id),
            sink_id=UUID(fxt_db_sinks[0].id),
            model_id=UUID(fxt_db_models[0].id),
        )
        created = fxt_pipeline_service.create_pipeline(pipeline)

        db_created = db_session.get(PipelineDB, str(created.id))
        fxt_active_pipeline_service.reload.assert_called_once()
        fxt_condition.notify_all.assert_called_once()
        assert db_created.name == "Test Pipeline"
        assert db_created.sink_id
        assert db_created.source_id
        assert db_created.model_id
        assert db_created.is_running

    def test_update_pipeline(self, fxt_pipeline_service, db_session):
        """Test updating a pipeline by ID."""
        pipeline_id = uuid4()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=False,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        updated = fxt_pipeline_service.update_pipeline(pipeline_id, {"name": "New Name"})

        db_updated = db_session.get(PipelineDB, pipeline_db.id)
        assert str(updated.id) == db_updated.id
        assert updated.name == "New Name"
        assert updated.name == db_updated.name

    @pytest.mark.parametrize("pipeline_attr", [PipelineField.SINK_ID, PipelineField.SOURCE_ID])
    def test_update_active_pipeline(
        self,
        pipeline_attr,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_db_models,
        fxt_pipeline_service,
        fxt_active_pipeline_service,
        fxt_condition,
        db_session,
    ):
        """Test updating a pipeline by ID."""
        pipeline_id = uuid4()
        db_session.add(fxt_db_sinks[0])
        db_session.add(fxt_db_sinks[1])
        db_session.add(fxt_db_sources[0])
        db_session.add(fxt_db_sources[1])
        db_session.add(fxt_db_models[0])
        db_session.flush()
        if pipeline_attr == PipelineField.SINK_ID:
            item_id = fxt_db_sinks[1].id
        else:
            item_id = fxt_db_sources[1].id
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=True,
            source_id=fxt_db_sources[0].id,
            sink_id=fxt_db_sinks[0].id,
            model_id=fxt_db_models[0].id,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        updated = fxt_pipeline_service.update_pipeline(pipeline_id, {pipeline_attr: item_id})

        if pipeline_attr == PipelineField.SINK_ID:
            fxt_active_pipeline_service.reload.assert_called_once()
            fxt_condition.notify_all.assert_not_called()
        else:
            fxt_active_pipeline_service.reload.assert_not_called()
            fxt_condition.notify_all.assert_called_once()
        db_updated = db_session.get(PipelineDB, pipeline_db.id)
        assert str(updated.id) == db_updated.id
        assert str(getattr(updated, pipeline_attr)) == item_id
        assert str(getattr(updated, pipeline_attr)) == getattr(db_updated, pipeline_attr)

    @pytest.mark.parametrize("pipeline_status", [PipelineStatus.IDLE, PipelineStatus.RUNNING])
    def test_activate_pipeline(
        self,
        pipeline_status,
        fxt_db_sinks,
        fxt_db_sources,
        fxt_db_models,
        fxt_pipeline_service,
        fxt_active_pipeline_service,
        fxt_condition,
        db_session,
    ):
        """Test activating and deactivating an existing pipeline by ID."""
        pipeline_id = uuid4()
        db_session.add(fxt_db_sinks[0])
        db_session.add(fxt_db_sources[0])
        db_session.add(fxt_db_models[0])
        db_session.flush()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=pipeline_status != PipelineStatus.RUNNING,
            source_id=fxt_db_sources[0].id,
            sink_id=fxt_db_sinks[0].id,
            model_id=fxt_db_models[0].id,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        fxt_pipeline_service.update_pipeline(pipeline_id, {"status": pipeline_status})

        fxt_active_pipeline_service.reload.assert_called_once()
        fxt_condition.notify_all.assert_called_once()
        db_updated = db_session.get(PipelineDB, pipeline_db.id)
        if pipeline_status == PipelineStatus.RUNNING:
            assert db_updated.is_running
        else:
            assert not db_updated.is_running

    def test_update_non_existent_pipeline(self, fxt_pipeline_service):
        """Test updating a non-existent pipeline raises error."""
        pipeline_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_pipeline_service.update_pipeline(pipeline_id, {"name": "New Name"})

        assert exc_info.value.resource_type == ResourceType.PIPELINE
        assert exc_info.value.resource_id == str(pipeline_id)

    def test_delete_pipeline(self, fxt_pipeline_service, db_session):
        """Test deleting a pipeline by ID."""
        pipeline_id = uuid4()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=False,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        fxt_pipeline_service.delete_pipeline_by_id(pipeline_id)

        assert db_session.query(PipelineDB).count() == 0

    def test_delete_non_existent_pipeline(self, fxt_pipeline_service):
        """Test deleting a non-existent pipeline raises error."""
        pipeline_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_pipeline_service.delete_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)

    def test_delete_active_pipeline(
        self, fxt_db_sinks, fxt_db_sources, fxt_db_models, fxt_pipeline_service, db_session
    ):
        """Test deleting an active pipeline raises error."""
        pipeline_id = uuid4()
        db_session.add(fxt_db_sinks[0])
        db_session.add(fxt_db_sources[0])
        db_session.add(fxt_db_models[0])
        db_session.flush()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=True,
            source_id=fxt_db_sources[0].id,
            sink_id=fxt_db_sinks[0].id,
            model_id=fxt_db_models[0].id,
        )
        db_session.add(pipeline_db)
        db_session.flush()
        with pytest.raises(ResourceInUseError) as excinfo:
            fxt_pipeline_service.delete_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)
