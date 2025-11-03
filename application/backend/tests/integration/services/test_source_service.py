# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest

from app.db.schema import PipelineDB, SourceDB
from app.services import ResourceInUseError, ResourceNotFoundError, ResourceType, SourceService
from app.services.base import ResourceWithIdAlreadyExistsError, ResourceWithNameAlreadyExistsError
from app.services.event.event_bus import EventType
from app.services.mappers import SourceMapper


@pytest.fixture
def fxt_source_service(fxt_event_bus, db_session) -> SourceService:
    """Fixture to provide a SourceService instance with mocked dependencies."""
    return SourceService(fxt_event_bus, db_session)


class TestSourceServiceIntegration:
    """Integration tests for ConfigurationService."""

    def test_create_source(self, fxt_webcam_source, fxt_source_service, db_session):
        """Test creating a new configuration."""
        fxt_source_service.create(fxt_webcam_source)

        assert db_session.query(SourceDB).count() == 1
        created = db_session.query(SourceDB).one()
        assert created.id == str(fxt_webcam_source.id)
        assert created.name == fxt_webcam_source.name
        assert created.source_type == fxt_webcam_source.source_type.value

    def test_create_source_non_unique(
        self,
        fxt_db_sources,
        fxt_webcam_source,
        fxt_source_service,
        db_session,
    ):
        """Test creating a new source with the name that already exists."""
        db_session.add(fxt_db_sources[0])

        fxt_webcam_source.name = fxt_db_sources[0].name  # Set the same name as existing resource

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_source_service.create(fxt_webcam_source)

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_webcam_source.name

    def test_create_source_duplicating_id(
        self,
        fxt_db_sources,
        fxt_webcam_source,
        fxt_source_service,
        db_session,
    ):
        """Test creating a new configuration with ID that already exists."""
        db_session.add(fxt_db_sources[0])
        db_session.flush()

        fxt_webcam_source.id = UUID(fxt_db_sources[0].id)  # Set the same ID as existing resource

        with pytest.raises(ResourceWithIdAlreadyExistsError) as excinfo:
            fxt_source_service.create(fxt_webcam_source)

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_db_sources[0].id

    @pytest.mark.parametrize("is_running", [True, False])
    def test_get_active_source(
        self,
        is_running,
        fxt_db_projects,
        fxt_db_sources,
        fxt_source_service,
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

        active_source = fxt_source_service.get_active_source()

        if is_running:
            assert active_source is not None and str(active_source.id) == db_source.id
        else:
            assert active_source is None

    def test_list_sources(self, fxt_db_sources, fxt_source_service, db_session):
        """Test retrieving all sources."""
        db_session.add_all(fxt_db_sources)

        db_sources = fxt_source_service.list_all()

        assert len(db_sources) == len(fxt_db_sources)
        for i, source in enumerate(db_sources):
            assert str(source.id) == fxt_db_sources[i].id
            assert source.name == fxt_db_sources[i].name

    def test_get_source(self, fxt_db_sources, fxt_source_service, db_session):
        """Test retrieving a source by ID."""
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        source = fxt_source_service.get_by_id(UUID(db_source.id))

        assert source is not None
        assert str(source.id) == db_source.id
        assert source.name == db_source.name

    def test_update_source(self, fxt_db_sources, fxt_source_service, db_session):
        """Test updating a source."""
        update_data = {"name": "Updated Source", "video_path": "/new/path"}
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        source = SourceMapper.to_schema(db_source)

        updated = fxt_source_service.update(source, update_data)

        assert updated.name == update_data["name"]
        assert str(updated.id) == db_source.id

        # Verify in DB
        db_source = db_session.get(SourceDB, db_source.id)
        assert db_source.name == update_data["name"]
        assert db_source.config_data["video_path"] == update_data["video_path"]

    def test_update_source_non_unique(self, fxt_db_sources, fxt_source_service, db_session):
        """Test updating a source with the name that already exists."""
        db_source = fxt_db_sources[0]
        db_session.add_all(fxt_db_sources[:2])
        db_session.flush()

        source = SourceMapper.to_schema(db_source)

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_source_service.update(source, {"name": fxt_db_sources[1].name})

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_db_sources[1].name

    def test_update_source_notify(
        self,
        fxt_db_sources,
        fxt_source_service,
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

        source = SourceMapper.to_schema(db_source)

        updated = fxt_source_service.update(source, update_data)

        assert updated.name == update_data["name"]
        assert str(updated.id) == db_source.id

        # Verify in DB
        db_source = db_session.get(SourceDB, db_source.id)
        assert db_source.name == update_data["name"]
        assert db_source.config_data["video_path"] == update_data["video_path"]
        fxt_event_bus.emit_event.assert_called_once_with(EventType.SOURCE_CHANGED)

    def test_delete_source(self, fxt_db_sources, fxt_source_service, db_session):
        """Test deleting a source."""
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        fxt_source_service.delete_by_id(db_source.id)

        assert db_session.query(SourceDB).count() == 0

    def test_delete_source_in_use(
        self,
        fxt_db_projects,
        fxt_db_sources,
        fxt_source_service,
        db_session,
    ):
        """Test deleting a source that is in use."""
        db_project = fxt_db_projects[0]
        db_source = fxt_db_sources[0]
        db_session.add_all([db_project, db_source])
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, is_running=True, source_id=db_source.id)
        db_session.add(db_pipeline)
        db_session.flush()

        with pytest.raises(ResourceInUseError) as exc_info:
            fxt_source_service.delete_by_id(db_source.id)

        assert exc_info.value.resource_type == ResourceType.SOURCE
        assert exc_info.value.resource_id == db_source.id
        assert db_session.query(SourceDB).count() == 1

    def test_delete_non_existent_source(self, fxt_source_service):
        """Test deleting a source that doesn't exist."""
        with pytest.raises(ResourceNotFoundError) as exc_info:
            source_id = uuid4()
            fxt_source_service.delete_by_id(source_id)

        assert exc_info.value.resource_type == ResourceType.SOURCE
        assert exc_info.value.resource_id == str(source_id)
