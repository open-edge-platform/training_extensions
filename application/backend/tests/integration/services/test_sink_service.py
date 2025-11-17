# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

import pytest

from app.db.schema import PipelineDB, SinkDB
from app.models import SinkAdapter
from app.models.sink import FolderConfig
from app.services import ResourceInUseError, ResourceType, SinkService
from app.services.base import ResourceWithIdAlreadyExistsError, ResourceWithNameAlreadyExistsError
from app.services.event.event_bus import EventType


@pytest.fixture
def fxt_sink_service(fxt_event_bus, db_session) -> SinkService:
    """Fixture to provide a SinkService instance with mocked dependencies."""
    return SinkService(fxt_event_bus, db_session)


class TestSinkServiceIntegration:
    """Integration tests for SinkService."""

    def test_create_sink(self, fxt_mqtt_sink, fxt_sink_service, db_session):
        """Test creating a new sink."""
        fxt_sink_service.create_sink(
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

    def test_create_sink_non_unique(
        self,
        fxt_db_sinks,
        fxt_mqtt_sink,
        fxt_sink_service,
        db_session,
    ):
        """Test creating a new sink with the name that already exists."""
        db_session.add(fxt_db_sinks[0])

        fxt_mqtt_sink.name = fxt_db_sinks[0].name  # Set the same name as existing resource

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_sink_service.create_sink(
                name=fxt_mqtt_sink.name,
                sink_type=fxt_mqtt_sink.sink_type,
                rate_limit=fxt_mqtt_sink.rate_limit,
                config_data=fxt_mqtt_sink.config_data,
                output_formats=fxt_mqtt_sink.output_formats,
                sink_id=fxt_mqtt_sink.id,
            )

        assert excinfo.value.resource_type == ResourceType.SINK
        assert excinfo.value.resource_id == fxt_mqtt_sink.name

    def test_create_sink_duplicating_id(
        self,
        fxt_db_sinks,
        fxt_mqtt_sink,
        fxt_sink_service,
        db_session,
    ):
        """Test creating a new sink with ID that already exists."""
        db_session.add(fxt_db_sinks[0])
        db_session.flush()

        fxt_mqtt_sink.id = UUID(fxt_db_sinks[0].id)  # Set the same ID as existing resource

        with pytest.raises(ResourceWithIdAlreadyExistsError) as excinfo:
            fxt_sink_service.create_sink(
                name=fxt_mqtt_sink.name,
                sink_type=fxt_mqtt_sink.sink_type,
                rate_limit=fxt_mqtt_sink.rate_limit,
                config_data=fxt_mqtt_sink.config_data,
                output_formats=fxt_mqtt_sink.output_formats,
                sink_id=fxt_mqtt_sink.id,
            )

        assert excinfo.value.resource_type == ResourceType.SINK
        assert excinfo.value.resource_id == fxt_db_sinks[0].id

    @pytest.mark.parametrize("is_running", [True, False])
    def test_get_active_sink(
        self,
        is_running,
        fxt_db_projects,
        fxt_db_sinks,
        fxt_sink_service,
        db_session,
    ):
        """Test getting active sink."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        db_sink = fxt_db_sinks[0]
        db_session.add(db_sink)
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, source_id=None, sink_id=db_sink.id, is_running=is_running)
        db_session.add(db_pipeline)
        db_session.flush()

        active_sink = fxt_sink_service.get_active_sink()
        active_sink_id = fxt_sink_service.get_active_sink_id()

        if is_running:
            assert active_sink is not None and str(active_sink.id) == db_sink.id
            assert active_sink_id is not None and str(active_sink_id) == db_sink.id
        else:
            assert active_sink is None
            assert active_sink_id is None

    def test_list_sinks(self, fxt_db_sinks, fxt_sink_service, db_session):
        """Test retrieving all sinks."""
        db_session.add_all(fxt_db_sinks)

        db_sinks = fxt_sink_service.list_all()

        assert len(db_sinks) == len(fxt_db_sinks)
        for i, sink in enumerate(db_sinks):
            assert str(sink.id) == fxt_db_sinks[i].id
            assert sink.name == fxt_db_sinks[i].name

    def test_get_sink(self, fxt_db_sinks, fxt_sink_service, db_session):
        """Test retrieving a sink by ID."""
        db_sink = fxt_db_sinks[0]
        db_session.add(db_sink)
        db_session.flush()

        sink = fxt_sink_service.get_by_id(UUID(db_sink.id))

        assert sink is not None
        assert str(sink.id) == db_sink.id
        assert sink.name == db_sink.name

    def test_update_sink(self, fxt_db_sinks, fxt_sink_service, db_session):
        """Test updating a sink."""
        db_sink = fxt_db_sinks[0]
        db_session.add(db_sink)
        db_session.flush()

        sink = SinkAdapter.validate_python(db_sink, from_attributes=True)

        updated = fxt_sink_service.update_sink(
            sink=sink,
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

    def test_update_sink_non_unique(self, fxt_db_sinks, fxt_sink_service, db_session):
        """Test updating a sink with the name that already exists."""
        db_sink = fxt_db_sinks[0]
        db_session.add_all(fxt_db_sinks[:2])
        db_session.flush()

        sink = SinkAdapter.validate_python(db_sink, from_attributes=True)

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_sink_service.update_sink(
                sink=sink,
                new_name=fxt_db_sinks[1].name,
                new_rate_limit=sink.rate_limit,
                new_config_data=sink.config_data,
                new_output_formats=sink.output_formats,
            )

        assert excinfo.value.resource_type == ResourceType.SINK
        assert excinfo.value.resource_id == fxt_db_sinks[1].name

    def test_update_sink_notify(
        self,
        fxt_db_sinks,
        fxt_sink_service,
        fxt_event_bus,
        fxt_db_projects,
        db_session,
    ):
        """Test updating a sink that is a part of active pipeline."""
        db_project = fxt_db_projects[0]
        db_sink = fxt_db_sinks[0]
        db_session.add_all([db_project, db_sink])
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, is_running=True, sink_id=db_sink.id)
        db_session.add(db_pipeline)
        db_session.flush()

        sink = SinkAdapter.validate_python(db_sink, from_attributes=True)

        updated = fxt_sink_service.update_sink(
            sink=sink,
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

    def test_delete_sink(self, fxt_db_sinks, fxt_sink_service, db_session):
        """Test deleting a sink."""
        db_sink = fxt_db_sinks[0]
        db_session.add(db_sink)
        db_session.flush()

        sink = SinkAdapter.validate_python(db_sink, from_attributes=True)
        fxt_sink_service.delete_sink(sink)

        assert db_session.query(SinkDB).count() == 0

    def test_delete_resource_in_use(
        self,
        fxt_db_projects,
        fxt_db_sinks,
        fxt_sink_service,
        db_session,
    ):
        """Test deleting a sink that is in use."""
        db_project = fxt_db_projects[0]
        db_sink = fxt_db_sinks[0]
        db_session.add_all([db_project, db_sink])
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, is_running=True, sink_id=db_sink.id)
        db_session.add(db_pipeline)
        db_session.flush()

        sink = SinkAdapter.validate_python(db_sink, from_attributes=True)

        with pytest.raises(ResourceInUseError) as exc_info:
            fxt_sink_service.delete_sink(sink)

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.resource_id == db_sink.id
        assert db_session.query(SinkDB).count() == 1
