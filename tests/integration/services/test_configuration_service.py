from unittest.mock import patch

import pytest

from app.db.schema import SourceDB
from app.schemas.source import SourceType
from app.services.configuration_service import ConfigurationService, ResourceInUseError, ResourceType
from app.services.mappers.source_mapper import SourceMapper


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.configuration_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
        yield mock


class TestConfigurationServiceIntegration:
    """Integration tests for ConfigurationService."""

    def test_create_source(self, fxt_source_config, db_session):
        """Test creating a new source."""
        config_service = ConfigurationService()
        config = config_service.create_source(fxt_source_config)

        assert db_session.query(SourceDB).count() == 1
        created = db_session.query(SourceDB).one()
        assert created.id == str(config.id)
        assert created.name == config.name
        assert created.source_type == config.source_type.value

    def test_list_sources(self, fxt_db_sources, db_session):
        """Test retrieving all source configurations."""

        for db_source in fxt_db_sources:
            db_session.add(db_source)
        db_session.flush()

        config_service = ConfigurationService()
        sources = config_service.list_sources()

        assert len(sources) == len(fxt_db_sources)

        for i, source in enumerate(sources):
            assert str(source.id) == fxt_db_sources[i].id
            assert source.name == fxt_db_sources[i].name
            assert source.source_type == SourceType(fxt_db_sources[i].source_type)

    def test_get_source(self, fxt_db_sources, db_session):
        """Test retrieving a source by ID."""

        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        config_service = ConfigurationService()
        source = config_service.get_source_by_id(db_source.id)

        assert source is not None
        assert str(source.id) == db_source.id
        assert source.name == db_source.name
        assert source.source_type == SourceType(db_source.source_type)

    def test_update_source(self, fxt_db_sources, db_session):
        """Test updating a source configuration."""
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        config_service = ConfigurationService()
        source = SourceMapper.to_schema(db_source)
        partial_update = {"name": "Updated Source Name", "video_path": "/new/path/video.mp4"}
        updated_source = config_service.update_source(source, partial_update)

        assert updated_source.name == "Updated Source Name"
        assert updated_source.id == source.id
        assert updated_source.video_path == "/new/path/video.mp4"

        # Verify in DB
        db_source = db_session.query(SourceDB).get(db_source.id)
        assert db_source.name == "Updated Source Name"
        assert db_source.config_data["video_path"] == "/new/path/video.mp4"

    def test_delete_source(self, fxt_db_sources, db_session):
        """Test deleting a source configuration."""
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        config_service = ConfigurationService()
        config_service.delete_source_by_id(db_source.id)

        # Verify deletion
        assert db_session.query(SourceDB).count() == 0

    def test_delete_source_in_use(self, fxt_db_sources, fxt_default_pipeline, db_session):
        """Test deleting a source configuration."""
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()
        fxt_default_pipeline.source_id = db_source.id
        db_session.add(fxt_default_pipeline)
        db_session.flush()

        with pytest.raises(ResourceInUseError) as exc_info:
            config_service = ConfigurationService()
            config_service.delete_source_by_id(db_source.id)

        assert exc_info.value.resource_type == ResourceType.SOURCE
        assert exc_info.value.resource_id == db_source.id
        assert db_session.query(SourceDB).count() == 1
