# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from app.db.schema import PipelineDB, SinkDB, SourceDB
from app.services import ConfigurationService, ResourceInUseError, ResourceNotFoundError, ResourceType
from app.services.base import ResourceWithIdAlreadyExistsError, ResourceWithNameAlreadyExistsError
from app.services.event.event_bus import EventType


@pytest.fixture
def fxt_config_service(fxt_event_bus, fxt_condition, db_session) -> ConfigurationService:
    """Fixture to provide a ConfigurationService instance with mocked dependencies."""
    return ConfigurationService(fxt_event_bus, db_session)


class TestConfigurationServiceIntegration:
    """Integration tests for ConfigurationService."""

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model,create_method",
        [
            (ResourceType.SOURCE, "fxt_webcam_source", SourceDB, "create_source"),
            (ResourceType.SINK, "fxt_mqtt_sink", SinkDB, "create_sink"),
        ],
    )
    def test_create_config(
        self, resource_type, fixture_name, db_model, create_method, fxt_config_service, request, db_session
    ):
        """Test creating a new configuration."""
        config = request.getfixturevalue(fixture_name)
        getattr(fxt_config_service, create_method)(config)

        assert db_session.query(db_model).count() == 1
        created = db_session.query(db_model).one()
        assert created.id == str(config.id)
        assert created.name == config.name
        if resource_type == ResourceType.SOURCE:
            assert created.source_type == config.source_type.value
        else:
            assert created.sink_type == config.sink_type.value

    @pytest.mark.parametrize(
        "resource_type,db_fixture_name,fixture_name,db_model,create_method",
        [
            (ResourceType.SOURCE, "fxt_db_sources", "fxt_webcam_source", SourceDB, "create_source"),
            (ResourceType.SINK, "fxt_db_sinks", "fxt_mqtt_sink", SinkDB, "create_sink"),
        ],
    )
    def test_create_config_non_unique(
        self,
        resource_type,
        db_fixture_name,
        fixture_name,
        db_model,
        create_method,
        fxt_config_service,
        request,
        db_session,
    ):
        """Test creating a new configuration with the name that already exists."""
        db_resources = request.getfixturevalue(db_fixture_name)
        db_session.add(db_resources[0])

        config = request.getfixturevalue(fixture_name)
        config.name = db_resources[0].name  # Set the same name as existing resource

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            getattr(fxt_config_service, create_method)(config)

        assert excinfo.value.resource_type == resource_type
        assert excinfo.value.resource_id == config.name

    @pytest.mark.parametrize(
        "resource_type,db_fixture_name,fixture_name,db_model,create_method",
        [
            (ResourceType.SOURCE, "fxt_db_sources", "fxt_webcam_source", SourceDB, "create_source"),
            (ResourceType.SINK, "fxt_db_sinks", "fxt_mqtt_sink", SinkDB, "create_sink"),
        ],
    )
    def test_create_config_duplicating_id(
        self,
        resource_type,
        db_fixture_name,
        fixture_name,
        db_model,
        create_method,
        fxt_config_service,
        request,
        db_session,
    ):
        """Test creating a new configuration with ID that already exists."""
        db_resources = request.getfixturevalue(db_fixture_name)
        db_session.add(db_resources[0])
        db_session.flush()

        config = request.getfixturevalue(fixture_name)
        config.id = db_resources[0].id  # Set the same ID as existing resource

        with pytest.raises(ResourceWithIdAlreadyExistsError) as excinfo:
            getattr(fxt_config_service, create_method)(config)

        assert excinfo.value.resource_type == resource_type
        assert excinfo.value.resource_id == config.id

    @pytest.mark.parametrize("is_running", [True, False])
    def test_get_active_config(
        self,
        is_running,
        fxt_db_projects,
        fxt_db_sources,
        fxt_db_sinks,
        fxt_config_service,
        db_session,
    ):
        """Test getting active configuration."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        db_source = fxt_db_sources[0]
        db_sink = fxt_db_sinks[0]
        db_session.add_all([db_source, db_sink])
        db_session.flush()

        db_pipeline = PipelineDB(
            project_id=db_project.id, source_id=db_source.id, sink_id=db_sink.id, is_running=is_running
        )
        db_session.add(db_pipeline)
        db_session.flush()

        active_sink = fxt_config_service.get_active_sink()
        active_source = fxt_config_service.get_active_source()

        if is_running:
            assert active_sink is not None and str(active_sink.id) == db_sink.id
            assert active_source is not None and str(active_source.id) == db_source.id
        else:
            assert active_sink is None
            assert active_source is None

    @pytest.mark.parametrize(
        "fixture_name,db_model,list_method",
        [
            ("fxt_db_sources", SourceDB, "list_sources"),
            ("fxt_db_sinks", SinkDB, "list_sinks"),
        ],
    )
    def test_list_configs(self, fixture_name, db_model, list_method, fxt_config_service, request, db_session):
        """Test retrieving all resource configurations."""
        db_resources = request.getfixturevalue(fixture_name)

        for db_resource in db_resources:
            db_session.add(db_resource)

        resources = getattr(fxt_config_service, list_method)()

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
    def test_get_config(self, fixture_name, db_model, get_method, fxt_config_service, request, db_session):
        """Test retrieving a config by ID."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        resource = getattr(fxt_config_service, get_method)(db_resource.id)

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
        self, resource_type, fixture_name, db_model, update_method, update_data, fxt_config_service, request, db_session
    ):
        """Test updating a resource configuration."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        updated = getattr(fxt_config_service, update_method)(db_resource.id, update_data)

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
        "resource_type,fixture_name,db_model,update_method",
        [
            (ResourceType.SOURCE, "fxt_db_sources", SourceDB, "update_source"),
            (ResourceType.SINK, "fxt_db_sinks", SinkDB, "update_sink"),
        ],
    )
    def test_update_resource_non_unique(
        self, resource_type, fixture_name, db_model, update_method, fxt_config_service, request, db_session
    ):
        """Test updating a configuration with the name that already exists."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add_all(db_resources[:2])
        db_session.flush()

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            getattr(fxt_config_service, update_method)(db_resource.id, {"name": db_resources[1].name})

        assert excinfo.value.resource_type == resource_type
        assert excinfo.value.resource_id == db_resources[1].name

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
        self,
        resource_type,
        fixture_name,
        db_model,
        update_method,
        update_data,
        fxt_config_service,
        fxt_event_bus,
        fxt_db_projects,
        request,
        db_session,
    ):
        """Test updating a resource configuration that is a part of active pipeline."""
        db_resources = request.getfixturevalue(fixture_name)
        db_config = db_resources[0]
        db_project = fxt_db_projects[0]
        db_pipeline = PipelineDB(project_id=db_project.id)
        db_pipeline.is_running = True
        setattr(db_pipeline, resource_type.lower(), db_config)
        db_session.add(db_project)
        db_session.flush()
        db_session.add(db_pipeline)
        db_session.flush()

        updated = getattr(fxt_config_service, update_method)(db_config.id, update_data)

        assert updated.name == update_data["name"]
        assert str(updated.id) == db_config.id

        # Verify in DB
        db_resource = db_session.get(db_model, db_config.id)
        assert db_resource.name == update_data["name"]
        if resource_type == ResourceType.SOURCE:
            assert db_resource.config_data["video_path"] == update_data["video_path"]
            fxt_event_bus.emit_event.assert_called_once_with(EventType.SOURCE_CHANGED)
        else:
            assert db_resource.config_data["folder_path"] == update_data["folder_path"]
            fxt_event_bus.emit_event.assert_called_once_with(EventType.SINK_CHANGED)

    @pytest.mark.parametrize(
        "resource_type,db_model,update_method",
        [
            (ResourceType.SOURCE, SourceDB, "update_source"),
            (ResourceType.SINK, SinkDB, "update_sink"),
        ],
    )
    def test_update_non_existent_resource(self, resource_type, db_model, update_method, fxt_config_service):
        """Test updating a non-existent resource raises error."""
        with pytest.raises(ResourceNotFoundError) as exc_info:
            config_id = uuid4()
            getattr(fxt_config_service, update_method)(config_id, {"name": "New Name"})

        assert exc_info.value.resource_type == resource_type
        assert exc_info.value.resource_id == str(config_id)

    @pytest.mark.parametrize(
        "fixture_name,db_model,delete_method",
        [
            ("fxt_db_sources", SourceDB, "delete_source_by_id"),
            ("fxt_db_sinks", SinkDB, "delete_sink_by_id"),
        ],
    )
    def test_delete_resource(self, fixture_name, db_model, delete_method, fxt_config_service, request, db_session):
        """Test deleting a resource configuration."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        getattr(fxt_config_service, delete_method)(db_resource.id)

        assert db_session.query(db_model).count() == 0

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model,delete_method",
        [
            (ResourceType.SOURCE, "fxt_db_sources", SourceDB, "delete_source_by_id"),
            (ResourceType.SINK, "fxt_db_sinks", SinkDB, "delete_sink_by_id"),
        ],
    )
    def test_delete_resource_in_use(
        self,
        resource_type,
        fixture_name,
        db_model,
        delete_method,
        fxt_db_projects,
        fxt_config_service,
        request,
        db_session,
    ):
        """Test deleting a resource that is in use."""
        db_resources = request.getfixturevalue(fixture_name)
        db_config = db_resources[0]
        db_project = fxt_db_projects[0]
        db_pipeline = PipelineDB(project_id=db_project.id)
        db_pipeline.is_running = True
        setattr(db_pipeline, resource_type.lower(), db_config)
        db_session.add(db_project)
        db_session.flush()
        db_session.add(db_pipeline)
        db_session.flush()

        with pytest.raises(ResourceInUseError) as exc_info:
            getattr(fxt_config_service, delete_method)(db_config.id)

        assert exc_info.value.resource_type == resource_type
        assert exc_info.value.resource_id == db_config.id
        assert db_session.query(db_model).count() == 1

    @pytest.mark.parametrize(
        "resource_type,db_model,delete_method",
        [
            (ResourceType.SOURCE, SourceDB, "delete_source_by_id"),
            (ResourceType.SINK, SinkDB, "delete_sink_by_id"),
        ],
    )
    def test_delete_non_existent_resource(self, resource_type, db_model, delete_method, fxt_config_service):
        """Test deleting a resource configuration that doesn't exist."""
        with pytest.raises(ResourceNotFoundError) as exc_info:
            config_id = uuid4()
            getattr(fxt_config_service, delete_method)(config_id)

        assert exc_info.value.resource_type == resource_type
        assert exc_info.value.resource_id == str(config_id)
