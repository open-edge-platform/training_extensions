from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.db.schema import SourceDB
from app.schemas.source import (
    ImagesFolderSourceConfig,
    IPCameraSourceConfig,
    SourceType,
    VideoFileSourceConfig,
    WebcamSourceConfig,
)
from app.services.mappers.source_mapper import SourceMapper


class TestSourceMapper:
    """Test cases for SourceMapper."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mapper = SourceMapper()

    @pytest.mark.parametrize(
        "schema_instance, expected_model",
        [
            (
                VideoFileSourceConfig(
                    source_type=SourceType.VIDEO_FILE,
                    video_path="/path/to/video.mp4",
                ),
                SourceDB(source_type=SourceType.VIDEO_FILE.value, config_data={"video_path": "/path/to/video.mp4"}),
            ),
            (
                WebcamSourceConfig(
                    source_type=SourceType.WEBCAM,
                    device_id=1,
                ),
                SourceDB(
                    source_type=SourceType.WEBCAM.value,
                    config_data={
                        "device_id": 1,
                    },
                ),
            ),
            (
                IPCameraSourceConfig(
                    source_type=SourceType.IP_CAMERA,
                    stream_url="rtsp://192.168.1.100:554/stream",
                    auth_required=False,
                ),
                SourceDB(
                    source_type=SourceType.IP_CAMERA.value,
                    config_data={
                        "stream_url": "rtsp://192.168.1.100:554/stream",
                        "auth_required": False,
                    },
                ),
            ),
            (
                ImagesFolderSourceConfig(
                    source_type=SourceType.IMAGES_FOLDER,
                    images_folder_path="/path/to/images",
                    ignore_existing_images=True,
                ),
                SourceDB(
                    source_type=SourceType.IMAGES_FOLDER.value,
                    config_data={
                        "images_folder_path": "/path/to/images",
                        "ignore_existing_images": True,
                    },
                ),
            ),
        ],
    )
    def test_from_schema_valid_source_types(self, schema_instance, expected_model):
        """Test from_schema with valid source types."""
        source_id = str(uuid4())
        result = self.mapper.from_schema(schema_instance, source_id=source_id)

        assert isinstance(result, SourceDB)
        assert result.id == source_id
        assert result.source_type == expected_model.source_type
        assert result.config_data == expected_model.config_data

    def test_from_schema_none_source_raises_error(self):
        """Test from_schema raises ValueError when source is None."""
        with pytest.raises(ValueError, match="Source config cannot be None"):
            self.mapper.from_schema(None)

    def test_from_schema_unsupported_source_type(self):
        """Test from_schema raises ValueError for unsupported source type."""
        mock = MagicMock()
        mock.source_type = "UNSUPPORTED_TYPE"

        with pytest.raises(ValueError, match="Unsupported source type: UNSUPPORTED_TYPE"):
            self.mapper.from_schema(mock)

    @pytest.mark.parametrize(
        "db_instance,expected_schema",
        [
            (
                SourceDB(source_type=SourceType.VIDEO_FILE.value, config_data={"video_path": "/path/to/video.mp4"}),
                VideoFileSourceConfig(
                    source_type=SourceType.VIDEO_FILE,
                    video_path="/path/to/video.mp4",
                ),
            ),
            (
                SourceDB(
                    source_type=SourceType.WEBCAM.value,
                    config_data={
                        "device_id": 1,
                    },
                ),
                WebcamSourceConfig(
                    source_type=SourceType.WEBCAM,
                    device_id=1,
                ),
            ),
            (
                SourceDB(
                    source_type=SourceType.IP_CAMERA.value,
                    config_data={
                        "stream_url": "rtsp://192.168.1.100:554/stream",
                        "auth_required": False,
                    },
                ),
                IPCameraSourceConfig(
                    source_type=SourceType.IP_CAMERA,
                    stream_url="rtsp://192.168.1.100:554/stream",
                    auth_required=False,
                ),
            ),
            (
                SourceDB(
                    source_type=SourceType.IMAGES_FOLDER.value,
                    config_data={
                        "images_folder_path": "/path/to/images",
                        "ignore_existing_images": True,
                    },
                ),
                ImagesFolderSourceConfig(
                    source_type=SourceType.IMAGES_FOLDER,
                    images_folder_path="/path/to/images",
                    ignore_existing_images=True,
                ),
            ),
        ],
    )
    def test_to_schema_valid_source_types(self, db_instance, expected_schema):
        """Test to_schema with valid source types."""
        result = self.mapper.to_schema(db_instance)

        assert result.source_type == expected_schema.source_type
        match result.source_type:
            case SourceType.VIDEO_FILE:
                assert isinstance(result, VideoFileSourceConfig)
                assert result.video_path == expected_schema.video_path
            case SourceType.WEBCAM:
                assert isinstance(result, WebcamSourceConfig)
                assert result.device_id == expected_schema.device_id
            case SourceType.IP_CAMERA:
                assert isinstance(result, IPCameraSourceConfig)
                assert result.stream_url == expected_schema.stream_url
                assert result.auth_required == expected_schema.auth_required

    def test_to_schema_unsupported_source_type(self):
        """Test to_schema raises ValueError for unsupported source type."""
        mock = MagicMock()
        mock.source_type = "UNSUPPORTED_TYPE"

        with pytest.raises(ValueError, match="Unsupported source type: UNSUPPORTED_TYPE"):
            self.mapper.to_schema(mock)
