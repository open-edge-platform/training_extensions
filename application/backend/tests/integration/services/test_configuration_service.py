# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest

from app.db.schema import PipelineDB, SinkDB, SourceDB
from app.models import SinkAdapter
from app.models.sink import FolderConfig
from app.services import ConfigurationService, ResourceInUseError, ResourceNotFoundError, ResourceType
from app.services.base import ResourceWithIdAlreadyExistsError, ResourceWithNameAlreadyExistsError
from app.services.event.event_bus import EventType


@pytest.fixture
def fxt_config_service(fxt_event_bus, fxt_condition, db_session) -> ConfigurationService:
    """Fixture to provide a ConfigurationService instance with mocked dependencies."""
    return ConfigurationService(fxt_event_bus, db_session)


class TestConfigurationServiceIntegration:
    """Integration tests for ConfigurationService."""

    def test_create_sink(self, fxt_mqtt_sink, fxt_config_service, db_session):
        """Test creating a new sink."""
        fxt_config_service.create_sink(
            name=fxt_mqtt_sink.name,
            sink_type=fxt_mqtt_sink.sink_type,
            rate_limit=fxt_mqtt_sink.rate_limit,
            config_data=fxt_mqtt_sink.config_data,
            output_formats=fxt_mqtt_sink.output_formats,
            sink_id=fxt_mqtt_sink.id,
        )

        assert db_session.query(SinkDB).count() == 1
        created = db_session.query(SinkDB).one()
        assert created.id == str(fxt_mqtt_sink.id)
        assert created.name == fxt_mqtt_sink.name
        assert created.sink_type == fxt_mqtt_sink.sink_type.value
        assert created.rate_limit == fxt_mqtt_sink.rate_limit
        assert created.config_data == fxt_mqtt_sink.config_data.model_dump(mode="json")
        assert created.output_formats == fxt_mqtt_sink.output_formats

    def test_create_source(self, fxt_webcam_source, fxt_config_service, db_session):
        """Test creating a new configuration."""
        fxt_config_service.create_source(fxt_webcam_source)

        assert db_session.query(SourceDB).count() == 1
        created = db_session.query(SourceDB).one()
        assert created.id == str(fxt_webcam_source.id)
        assert created.name == fxt_webcam_source.name
        assert created.source_type == fxt_webcam_source.source_type.value

    def test_create_sink_non_unique(
        self,
        fxt_db_sinks,
        fxt_mqtt_sink,
        fxt_config_service,
        db_session,
    ):
        """Test creating a new sink with the name that already exists."""
        db_session.add(fxt_db_sinks[0])

        fxt_mqtt_sink.name = fxt_db_sinks[0].name  # Set the same name as existing resource

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_config_service.create_sink(
                name=fxt_mqtt_sink.name,
                sink_type=fxt_mqtt_sink.sink_type,
                rate_limit=fxt_mqtt_sink.rate_limit,
                config_data=fxt_mqtt_sink.config_data,
                output_formats=fxt_mqtt_sink.output_formats,
                sink_id=fxt_mqtt_sink.id,
            )

        assert excinfo.value.resource_type == ResourceType.SINK
        assert excinfo.value.resource_id == fxt_mqtt_sink.name

    def test_create_source_non_unique(
        self,
        fxt_db_sources,
        fxt_webcam_source,
        fxt_config_service,
        db_session,
    ):
        """Test creating a new source with the name that already exists."""
        db_session.add(fxt_db_sources[0])

        fxt_webcam_source.name = fxt_db_sources[0].name  # Set the same name as existing resource

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_config_service.create_source(fxt_webcam_source)

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_webcam_source.name

    def test_create_sink_duplicating_id(
        self,
        fxt_db_sinks,
        fxt_mqtt_sink,
        fxt_config_service,
        db_session,
    ):
        """Test creating a new sink with ID that already exists."""
        db_session.add(fxt_db_sinks[0])
        db_session.flush()

        fxt_mqtt_sink.id = UUID(fxt_db_sinks[0].id)  # Set the same ID as existing resource

        with pytest.raises(ResourceWithIdAlreadyExistsError) as excinfo:
            fxt_config_service.create_sink(
                name=fxt_mqtt_sink.name,
                sink_type=fxt_mqtt_sink.sink_type,
                rate_limit=fxt_mqtt_sink.rate_limit,
                config_data=fxt_mqtt_sink.config_data,
                output_formats=fxt_mqtt_sink.output_formats,
                sink_id=fxt_mqtt_sink.id,
            )

        assert excinfo.value.resource_type == ResourceType.SINK
        assert excinfo.value.resource_id == fxt_db_sinks[0].id

    def test_create_source_duplicating_id(
        self,
        fxt_db_sources,
        fxt_webcam_source,
        fxt_config_service,
        db_session,
    ):
        """Test creating a new configuration with ID that already exists."""
        db_session.add(fxt_db_sources[0])
        db_session.flush()

        fxt_webcam_source.id = UUID(fxt_db_sources[0].id)  # Set the same ID as existing resource

        with pytest.raises(ResourceWithIdAlreadyExistsError) as excinfo:
            fxt_config_service.create_source(fxt_webcam_source)

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_db_sources[0].id

    @pytest.mark.parametrize("is_running", [True, False])
    def test_get_active_sink(
        self,
        is_running,
        fxt_db_projects,
        fxt_db_sinks,
        fxt_config_service,
        db_session,
    ):
        """Test getting active configuration."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        db_sink = fxt_db_sinks[0]
        db_session.add(db_sink)
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, source_id=None, sink_id=db_sink.id, is_running=is_running)
        db_session.add(db_pipeline)
        db_session.flush()

        active_sink = fxt_config_service.get_active_sink()
        active_sink_id = fxt_config_service.get_active_sink_id()

        if is_running:
            assert active_sink is not None and str(active_sink.id) == db_sink.id
            assert active_sink_id is not None and str(active_sink_id) == db_sink.id
        else:
            assert active_sink is None
            assert active_sink_id is None

    @pytest.mark.parametrize("is_running", [True, False])
    def test_get_active_source(
        self,
        is_running,
        fxt_db_projects,
        fxt_db_sources,
        fxt_config_service,
        db_session,
    ):
        """Test getting active configuration."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, source_id=db_source.id, sink_id=None, is_running=is_running)
        db_session.add(db_pipeline)
        db_session.flush()

        active_source = fxt_config_service.get_active_source()

        if is_running:
            assert active_source is not None and str(active_source.id) == db_source.id
        else:
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

    def test_update_sink(self, fxt_db_sinks, fxt_config_service, db_session):
        """Test updating a sink."""
        db_sink = fxt_db_sinks[0]
        db_session.add(db_sink)
        db_session.flush()

        updated = fxt_config_service.update_sink(
            sink_id=UUID(db_sink.id),
            new_name="Updated Sink",
            new_rate_limit=db_sink.rate_limit,
            new_config_data=FolderConfig(folder_path="/new/folder"),
            new_output_formats=db_sink.output_formats,
        )

        assert updated.name == "Updated Sink"
        assert str(updated.id) == db_sink.id

        # Verify in DB
        db_sink = db_session.get(SinkDB, db_sink.id)
        assert db_sink.name == "Updated Sink"
        assert db_sink.config_data["folder_path"] == "/new/folder"

    def test_update_source(self, fxt_db_sources, fxt_config_service, db_session):
        """Test updating a source."""
        update_data = {"name": "Updated Source", "video_path": "/new/path"}
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        updated = fxt_config_service.update_source(db_source.id, update_data)

        assert updated.name == update_data["name"]
        assert str(updated.id) == db_source.id

        # Verify in DB
        db_source = db_session.get(SourceDB, db_source.id)
        assert db_source.name == update_data["name"]
        assert db_source.config_data["video_path"] == update_data["video_path"]

    def test_update_sink_non_unique(self, fxt_db_sinks, fxt_config_service, db_session):
        """Test updating a sink with the name that already exists."""
        db_sink = fxt_db_sinks[0]
        db_session.add_all(fxt_db_sinks[:2])
        db_session.flush()

        sink = SinkAdapter.validate_python(db_sink, from_attributes=True)

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_config_service.update_sink(
                sink_id=sink.id,
                new_name=fxt_db_sinks[1].name,
                new_rate_limit=sink.rate_limit,
                new_config_data=sink.config_data,
                new_output_formats=sink.output_formats,
            )

        assert excinfo.value.resource_type == ResourceType.SINK
        assert excinfo.value.resource_id == fxt_db_sinks[1].name

    def test_update_source_non_unique(self, fxt_db_sources, fxt_config_service, db_session):
        """Test updating a source with the name that already exists."""
        db_source = fxt_db_sources[0]
        db_session.add_all(fxt_db_sources[:2])
        db_session.flush()

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_config_service.update_source(db_source.id, {"name": fxt_db_sources[1].name})

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_db_sources[1].name

    def test_update_sink_notify(
        self,
        fxt_db_sinks,
        fxt_config_service,
        fxt_event_bus,
        fxt_db_projects,
        db_session,
    ):
        """Test updating a sink configuration that is a part of active pipeline."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        db_sink = fxt_db_sinks[0]
        db_session.add(db_sink)
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, is_running=True, sink_id=db_sink.id)
        db_session.add(db_pipeline)
        db_session.flush()

        updated = fxt_config_service.update_sink(
            sink_id=UUID(db_sink.id),
            new_name="Updated Sink",
            new_rate_limit=db_sink.rate_limit,
            new_config_data=FolderConfig(folder_path="/new/folder"),
            new_output_formats=db_sink.output_formats,
        )

        assert updated.name == "Updated Sink"
        assert str(updated.id) == db_sink.id

        # Verify in DB
        db_sink = db_session.get(SinkDB, db_sink.id)
        assert db_sink.name == "Updated Sink"
        assert db_sink.config_data["folder_path"] == "/new/folder"
        fxt_event_bus.emit_event.assert_called_once_with(EventType.SINK_CHANGED)

    def test_update_source_notify(
        self,
        fxt_db_sources,
        fxt_config_service,
        fxt_event_bus,
        fxt_db_projects,
        db_session,
    ):
        """Test updating a source configuration that is a part of active pipeline."""
        update_data = {"name": "Updated Source", "video_path": "/new/path"}
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, is_running=True, source_id=db_source.id)
        db_session.add(db_pipeline)
        db_session.flush()

        updated = fxt_config_service.update_source(db_source.id, update_data)

        assert updated.name == update_data["name"]
        assert str(updated.id) == db_source.id

        # Verify in DB
        db_source = db_session.get(SourceDB, db_source.id)
        assert db_source.name == update_data["name"]
        assert db_source.config_data["video_path"] == update_data["video_path"]
        fxt_event_bus.emit_event.assert_called_once_with(EventType.SOURCE_CHANGED)

    def test_update_non_existent_sink(self, fxt_db_sinks, fxt_config_service, db_session):
        """Test updating a non-existent sink raises error."""

        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_id = uuid4()
            fxt_config_service.update_sink(
                sink_id=sink_id,
                new_name="New name",
                new_rate_limit=0.1,
                new_config_data=FolderConfig(folder_path="/new/path"),
                new_output_formats=[],
            )

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.resource_id == str(sink_id)

    def test_update_non_existent_source(self, fxt_config_service, db_session):
        """Test updating a non-existent source raises error."""

        with pytest.raises(ResourceNotFoundError) as exc_info:
            source_id = uuid4()
            fxt_config_service.update_source(source_id, {"name": "New Name"})

        assert exc_info.value.resource_type == ResourceType.SOURCE
        assert exc_info.value.resource_id == str(source_id)

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
