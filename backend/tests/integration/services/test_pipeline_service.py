from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.db.schema import PipelineDB
from app.schemas import Pipeline, PipelineStatus
from app.services import PipelineService, ResourceInUseError, ResourceNotFoundError, ResourceType


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


class TestPipelineServiceIntegration:
    """Integration tests for PipelineService."""

    def test_list_pipelines(self, db_session):
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

        pipelines = PipelineService().list_pipelines()

        assert len(pipelines) == 2
        for i, db_pipeline in enumerate([db_pipeline1, db_pipeline2]):
            assert str(pipelines[i].id) == db_pipeline.id

    def test_get_pipeline(self, db_session):
        """Test retrieving a pipeline by ID."""
        pipeline_id = uuid4()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=False,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        pipeline = PipelineService().get_pipeline_by_id(pipeline_id)

        assert pipeline is not None
        assert pipeline.id == pipeline_id
        assert pipeline.name == pipeline.name
        assert pipeline.status == PipelineStatus.from_bool(pipeline_db.is_running)

    def test_get_non_existent_pipeline(self):
        """Test retrieving a non-existent pipeline raises error."""

        pipeline_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            PipelineService().get_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)

    def test_create_pipeline(self, db_session):
        """Test creating a new pipeline."""

        pipeline = Pipeline(
            name="Test Pipeline",
            status=PipelineStatus.IDLE,
        )

        pipeline_service = PipelineService()
        created = pipeline_service.create_pipeline(pipeline)

        db_created = db_session.get(PipelineDB, str(created.id))

        assert db_created.name == "Test Pipeline"
        assert not db_created.is_running

    def test_create_active_pipeline(self, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session):
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
        with (
            patch("app.services.pipeline_service.ActivePipelineService") as mock_active_pipeline_service,
            patch("app.core.Scheduler") as mock_scheduler,
        ):
            mock_active_pipeline_service.return_value.reload.return_value = None
            mock_scheduler.return_value.mp_config_changed_condition = MagicMock(spec=Condition)

            pipeline_service = PipelineService()
            created = pipeline_service.create_pipeline(pipeline)

        db_created = db_session.get(PipelineDB, str(created.id))

        mock_active_pipeline_service.return_value.reload.assert_called_once()
        mock_scheduler.return_value.mp_config_changed_condition.notify_all.assert_called_once()
        assert db_created.name == "Test Pipeline"
        assert db_created.sink_id
        assert db_created.source_id
        assert db_created.model_id
        assert db_created.is_running

    def test_update_pipeline(self, db_session):
        """Test updating a pipeline by ID."""

        pipeline_id = uuid4()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=False,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        pipeline_service = PipelineService()
        updated = pipeline_service.update_pipeline(pipeline_id, {"name": "New Name"})

        db_updated = db_session.get(PipelineDB, pipeline_db.id)

        assert str(updated.id) == db_updated.id
        assert updated.name == "New Name"
        assert updated.name == db_updated.name

    @pytest.mark.parametrize("pipeline_attr", ["sink_id", "source_id"])
    def test_update_active_pipeline(self, pipeline_attr, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session):
        """Test updating a pipeline by ID."""

        pipeline_id = uuid4()
        db_session.add(fxt_db_sinks[0])
        db_session.add(fxt_db_sinks[1])
        db_session.add(fxt_db_sources[0])
        db_session.add(fxt_db_sources[1])
        db_session.add(fxt_db_models[0])
        db_session.flush()
        if pipeline_attr == "sink_id":
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

        with (
            patch("app.services.pipeline_service.ActivePipelineService") as mock_active_pipeline_service,
            patch("app.core.Scheduler") as mock_scheduler,
        ):
            mock_active_pipeline_service.return_value.reload.return_value = None
            mock_scheduler.return_value.mp_config_changed_condition = MagicMock(spec=Condition)

            pipeline_service = PipelineService()
            updated = pipeline_service.update_pipeline(pipeline_id, {pipeline_attr: item_id})

        db_updated = db_session.get(PipelineDB, pipeline_db.id)

        if pipeline_attr == "sink_id":
            mock_active_pipeline_service.return_value.reload.assert_called_once()
            mock_scheduler.return_value.mp_config_changed_condition.notify_all.assert_not_called()
        else:
            mock_active_pipeline_service.return_value.reload.assert_not_called()
            mock_scheduler.return_value.mp_config_changed_condition.notify_all.assert_called_once()
        assert str(updated.id) == db_updated.id
        assert str(getattr(updated, pipeline_attr)) == item_id
        assert str(getattr(updated, pipeline_attr)) == getattr(db_updated, pipeline_attr)

    @pytest.mark.parametrize("pipeline_status", [PipelineStatus.IDLE, PipelineStatus.RUNNING])
    def test_activate_pipeline(self, pipeline_status, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session):
        """Test updating a pipeline by ID."""

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

        with (
            patch("app.services.pipeline_service.ActivePipelineService") as mock_active_pipeline_service,
            patch("app.core.Scheduler") as mock_scheduler,
        ):
            mock_active_pipeline_service.return_value.reload.return_value = None
            mock_scheduler.return_value.mp_config_changed_condition = MagicMock(spec=Condition)

            pipeline_service = PipelineService()
            pipeline_service.update_pipeline(pipeline_id, {"status": pipeline_status})

        db_updated = db_session.get(PipelineDB, pipeline_db.id)

        mock_active_pipeline_service.return_value.reload.assert_called_once()
        mock_scheduler.return_value.mp_config_changed_condition.notify_all.assert_called_once()
        if pipeline_status == PipelineStatus.RUNNING:
            assert db_updated.is_running
        else:
            assert not db_updated.is_running

    def test_update_non_existent_pipeline(self):
        """Test updating a non-existent pipeline raises error."""

        pipeline_id = uuid4()
        pipeline_service = PipelineService()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            pipeline_service.update_pipeline(pipeline_id, {"name": "New Name"})

        assert exc_info.value.resource_type == ResourceType.PIPELINE
        assert exc_info.value.resource_id == str(pipeline_id)

    def test_delete_pipeline(self, db_session):
        """Test deleting a pipeline by ID."""
        pipeline_id = uuid4()
        pipeline_db = PipelineDB(
            id=str(pipeline_id),
            name="Test Pipeline",
            is_running=False,
        )
        db_session.add(pipeline_db)
        db_session.flush()

        PipelineService().delete_pipeline_by_id(pipeline_id)

        assert db_session.query(PipelineDB).count() == 0

    def test_delete_non_existent_pipeline(self):
        """Test deleting a non-existent pipeline raises error."""
        pipeline_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            PipelineService().delete_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)

    def test_delete_active_pipeline(self, fxt_db_sinks, fxt_db_sources, fxt_db_models, db_session):
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
            PipelineService().delete_pipeline_by_id(pipeline_id)

        assert excinfo.value.resource_type == ResourceType.PIPELINE
        assert excinfo.value.resource_id == str(pipeline_id)
