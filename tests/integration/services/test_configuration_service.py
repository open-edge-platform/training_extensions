from unittest.mock import patch

import pytest

from app.db.schema import SinkDB, SourceDB
from app.services.configuration_service import ConfigurationService, ResourceInUseError, ResourceType
from app.services.mappers import SinkMapper, SourceMapper


@pytest.fixture(autouse=True)
def mock_get_db_session(db_session):
    """Mock the get_db_session to use test database."""
    with patch("app.services.configuration_service.get_db_session") as mock:
        mock.return_value.__enter__.return_value = db_session
        mock.return_value.__exit__.return_value = None
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
        "resource_type,fixture_name,db_model",
        [
            (ResourceType.SOURCE, "fxt_db_sources", SourceDB),
            (ResourceType.SINK, "fxt_db_sinks", SinkDB),
        ],
    )
    def test_list_configs(self, resource_type, fixture_name, db_model, request, db_session):
        """Test retrieving all resource configurations."""

        db_resources = request.getfixturevalue(fixture_name)

        for db_resource in db_resources:
            db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()

        if resource_type == ResourceType.SOURCE:
            resources = config_service.list_sources()
        else:
            resources = config_service.list_sinks()

        assert len(resources) == len(db_resources)

        for i, resource in enumerate(resources):
            assert str(resource.id) == db_resources[i].id
            assert resource.name == db_resources[i].name

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model",
        [
            (ResourceType.SOURCE, "fxt_db_sources", SourceDB),
            (ResourceType.SINK, "fxt_db_sinks", SinkDB),
        ],
    )
    def test_get_config(self, resource_type, fixture_name, db_model, request, db_session):
        """Test retrieving a config by ID."""

        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()

        if resource_type == ResourceType.SOURCE:
            resource = config_service.get_source_by_id(db_resource.id)
        else:
            resource = config_service.get_sink_by_id(db_resource.id)

        assert resource is not None
        assert str(resource.id) == db_resource.id
        assert resource.name == db_resource.name

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model,mapper,update_data",
        [
            (
                ResourceType.SOURCE,
                "fxt_db_sources",
                SourceDB,
                SourceMapper,
                {"name": "Updated Source", "video_path": "/new/path"},
            ),
            (
                ResourceType.SINK,
                "fxt_db_sinks",
                SinkDB,
                SinkMapper,
                {"name": "Updated Sink", "folder_path": "/new/folder"},
            ),
        ],
    )
    def test_update_resource(self, resource_type, fixture_name, db_model, mapper, update_data, request, db_session):
        """Test updating a resource configuration."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()
        resource = mapper.to_schema(db_resource)

        if resource_type == ResourceType.SOURCE:
            updated_resource = config_service.update_source(resource, update_data)
        else:
            updated_resource = config_service.update_sink(resource, update_data)

        assert updated_resource.name == update_data["name"]
        assert updated_resource.id == resource.id

        # Verify in DB
        db_resource = db_session.get(db_model, db_resource.id)
        assert db_resource.name == update_data["name"]
        if resource_type == ResourceType.SOURCE:
            assert db_resource.config_data["video_path"] == update_data["video_path"]
        else:
            assert db_resource.config_data["folder_path"] == update_data["folder_path"]

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model",
        [
            (ResourceType.SOURCE, "fxt_db_sources", SourceDB),
            (ResourceType.SINK, "fxt_db_sinks", SinkDB),
        ],
    )
    def test_delete_resource(self, resource_type, fixture_name, db_model, request, db_session):
        """Test deleting a resource configuration."""
        db_resources = request.getfixturevalue(fixture_name)
        db_resource = db_resources[0]
        db_session.add(db_resource)
        db_session.flush()

        config_service = ConfigurationService()

        if resource_type == ResourceType.SOURCE:
            config_service.delete_source_by_id(db_resource.id)
        else:
            config_service.delete_sink_by_id(db_resource.id)

        assert db_session.query(db_model).count() == 0

    @pytest.mark.parametrize(
        "resource_type,fixture_name,db_model,pipeline_field",
        [
            (ResourceType.SOURCE, "fxt_db_sources", SourceDB, "source_id"),
            (ResourceType.SINK, "fxt_db_sinks", SinkDB, "sink_id"),
        ],
    )
    def test_delete_resource_in_use(
        self, resource_type, fixture_name, db_model, pipeline_field, fxt_default_pipeline, request, db_session
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

            if resource_type == ResourceType.SOURCE:
                config_service.delete_source_by_id(db_resource.id)
            else:
                config_service.delete_sink_by_id(db_resource.id)

        assert exc_info.value.resource_type == resource_type
        assert exc_info.value.resource_id == db_resource.id
        assert db_session.query(db_model).count() == 1
