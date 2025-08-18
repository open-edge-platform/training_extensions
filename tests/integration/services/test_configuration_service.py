from multiprocessing.synchronize import Condition
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.db.schema import PipelineDB, SinkDB, SourceDB
from app.services import ConfigurationService, ResourceInUseError, ResourceNotFoundError, ResourceType


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with (
        patch("app.services.configuration_service.get_db_session") as mock,
        patch("app.services.base.get_db_session") as mock_base,
    ):
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        mock_base.return_value.__enter__.return_value = db_session
        mock_base.return_value.__exit__.return_value = None
        yield mock


class TestConfigurationServiceIntegration:
    """Integration tests for ConfigurationService."""

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model",
        [
            (ResourceType.SOURCE, "fxt_source_config", SourceDB),
            (ResourceType.SINK, "fxt_sink_config", SinkDB),
        ],
    )
    def test_create_config(self, resource_type, fixture_name, db_model, request, db_session):
        """Test creating a new resource."""

        config = request.getfixturevalue(fixture_name)
        config_service = ConfigurationService()

        if resource_type == ResourceType.SOURCE:
            config_service.create_source(config)
        else:
            config_service.create_sink(config)

        assert db_session.query(db_model).count() == 1
        created = db_session.query(db_model).one()
        assert created.id == str(config.id)
        assert created.name == config.name
        if resource_type == ResourceType.SOURCE:
            assert created.source_type == config.source_type.value
        else:
            assert created.sink_type == config.sink_type.value

    @pytest.mark.parametrize(
        "fixture_name,db_model,list_method",
        [
            ("fxt_db_sources", SourceDB, "list_sources"),
            ("fxt_db_sinks", SinkDB, "list_sinks"),
        ],
    )
    def test_list_configs(self, fixture_name, db_model, list_method, request, db_session):
        """Test retrieving all resource configurations."""

        db_resources = request.getfixturevalue(fixture_name)

        for db_resource in db_resources:
            db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()

        resources = getattr(config_service, list_method)()

        assert len(resources) == len(db_resources)

        for i, resource in enumerate(resources):
            assert str(resource.id) == db_resources[i].id
            assert resource.name == db_resources[i].name

    @pytest.mark.parametrize(
        "fixture_name,db_model,get_method",
        [
            ("fxt_db_sources", SourceDB, "get_source_by_id"),
            ("fxt_db_sinks", SinkDB, "get_sink_by_id"),
        ],
    )
    def test_get_config(self, fixture_name, db_model, get_method, request, db_session):
        """Test retrieving a config by ID."""

        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()

        resource = getattr(config_service, get_method)(db_resource.id)

        assert resource is not None
        assert str(resource.id) == db_resource.id
        assert resource.name == db_resource.name

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model,update_method,update_data",
        [
            (
                ResourceType.SOURCE,
                "fxt_db_sources",
                SourceDB,
                "update_source",
                {"name": "Updated Source", "video_path": "/new/path"},
            ),
            (
                ResourceType.SINK,
                "fxt_db_sinks",
                SinkDB,
                "update_sink",
                {"name": "Updated Sink", "folder_path": "/new/folder"},
            ),
        ],
    )
    def test_update_resource(
        self, resource_type, fixture_name, db_model, update_method, update_data, request, db_session
    ):
        """Test updating a resource configuration."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()
        updated = getattr(config_service, update_method)(db_resource.id, update_data)

        assert updated.name == update_data["name"]
        assert str(updated.id) == db_resource.id

        # Verify in DB
        db_resource = db_session.get(db_model, db_resource.id)
        assert db_resource.name == update_data["name"]
        if resource_type == ResourceType.SOURCE:
            assert db_resource.config_data["video_path"] == update_data["video_path"]
        else:
            assert db_resource.config_data["folder_path"] == update_data["folder_path"]

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model,update_method,update_data",
        [
            (
                ResourceType.SOURCE,
                "fxt_db_sources",
                SourceDB,
                "update_source",
                {"name": "Updated Source", "video_path": "/new/path"},
            ),
            (
                ResourceType.SINK,
                "fxt_db_sinks",
                SinkDB,
                "update_sink",
                {"name": "Updated Sink", "folder_path": "/new/folder"},
            ),
        ],
    )
    def test_update_resource_notify(
        self, resource_type, fixture_name, db_model, update_method, update_data, request, db_session
    ):
        """Test updating a resource configuration that is a part of active pipeline."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        db_pipeline = PipelineDB(name="Active Pipeline", is_running=True)
        setattr(db_pipeline, f"{resource_type.lower()}_id", db_resource.id)
        db_session.add(db_pipeline)
        db_session.flush()

        with (
            patch("app.services.configuration_service.ActivePipelineService") as mock_active_pipeline_service,
            patch("app.core.Scheduler") as mock_scheduler,
        ):
            mock_active_pipeline_service.return_value.reload.return_value = None
            mock_scheduler.return_value.mp_config_changed_condition = MagicMock(spec=Condition)
            config_service = ConfigurationService()
            updated = getattr(config_service, update_method)(db_resource.id, update_data)

            assert updated.name == update_data["name"]
            assert str(updated.id) == db_resource.id

            # Verify in DB
            db_resource = db_session.get(db_model, db_resource.id)
            assert db_resource.name == update_data["name"]
            if resource_type == ResourceType.SOURCE:
                assert db_resource.config_data["video_path"] == update_data["video_path"]
                mock_scheduler.return_value.mp_config_changed_condition.notify_all.assert_called_once()
                mock_active_pipeline_service.return_value.reload.assert_not_called()
            else:
                assert db_resource.config_data["folder_path"] == update_data["folder_path"]
                mock_scheduler.return_value.mp_config_changed_condition.notify_all.assert_not_called()
                mock_active_pipeline_service.return_value.reload.assert_called_once()

    @pytest.mark.parametrize(
        "resource_type,db_model,update_method",
        [
            (ResourceType.SOURCE, SourceDB, "update_source"),
            (ResourceType.SINK, SinkDB, "update_sink"),
        ],
    )
    def test_update_non_existent_resource(self, resource_type, db_model, update_method):
        """Test updating a non-existent resource raises error."""

        config_service = ConfigurationService()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            config_id = uuid4()
            getattr(config_service, update_method)(config_id, {"name": "New Name"})

        assert exc_info.value.resource_type == resource_type
        assert exc_info.value.resource_id == str(config_id)

    @pytest.mark.parametrize(
        "fixture_name,db_model,delete_method",
        [
            ("fxt_db_sources", SourceDB, "delete_source_by_id"),
            ("fxt_db_sinks", SinkDB, "delete_sink_by_id"),
        ],
    )
    def test_delete_resource(self, fixture_name, db_model, delete_method, request, db_session):
        """Test deleting a resource configuration."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()

        getattr(config_service, delete_method)(db_resource.id)

        assert db_session.query(db_model).count() == 0

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model,pipeline_field,delete_method",
        [
            (ResourceType.SOURCE, "fxt_db_sources", SourceDB, "source_id", "delete_source_by_id"),
            (ResourceType.SINK, "fxt_db_sinks", SinkDB, "sink_id", "delete_sink_by_id"),
        ],
    )
    def test_delete_resource_in_use(
        self,
        resource_type,
        fixture_name,
        db_model,
        pipeline_field,
        delete_method,
        fxt_default_pipeline,
        request,
        db_session,
    ):
        """Test deleting a resource that is in use."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        setattr(fxt_default_pipeline, pipeline_field, db_resource.id)
        db_session.add(fxt_default_pipeline)
        db_session.flush()

        with pytest.raises(ResourceInUseError) as exc_info:
            config_service = ConfigurationService()
            getattr(config_service, delete_method)(db_resource.id)

        assert exc_info.value.resource_type == resource_type
        assert exc_info.value.resource_id == db_resource.id
        assert db_session.query(db_model).count() == 1

    @pytest.mark.parametrize(
        "resource_type,db_model,delete_method",
        [
            (ResourceType.SOURCE, SourceDB, "delete_source_by_id"),
            (ResourceType.SINK, SinkDB, "delete_sink_by_id"),
        ],
    )
    def test_delete_non_existent_resource(self, resource_type, db_model, delete_method):
        """Test deleting a resource configuration that doesn't exist."""

        config_service = ConfigurationService()

        with pytest.raises(ResourceNotFoundError) as exc_info:
            config_id = uuid4()
            getattr(config_service, delete_method)(config_id)

        assert exc_info.value.resource_type == resource_type
        assert exc_info.value.resource_id == str(config_id)
