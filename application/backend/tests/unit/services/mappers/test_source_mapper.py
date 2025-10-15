# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

SOURCE_ID = uuid4()
SUPPORTED_SOURCES_MAPPING = [
    (
        VideoFileSourceConfig(
            source_type=SourceType.VIDEO_FILE,
            id=SOURCE_ID,
            name="Test Video Source",
            video_path="/path/to/video.mp4",
        ),
        SourceDB(
            source_type=SourceType.VIDEO_FILE,
            id=str(SOURCE_ID),
            name="Test Video Source",
            config_data={"video_path": "/path/to/video.mp4"},
        ),
    ),
    (
        WebcamSourceConfig(
            source_type=SourceType.WEBCAM,
            id=SOURCE_ID,
            name="Test Webcam Source",
            device_id=1,
        ),
        SourceDB(
            source_type=SourceType.WEBCAM,
            id=str(SOURCE_ID),
            name="Test Webcam Source",
            config_data={
                "device_id": 1,
            },
        ),
    ),
    (
        IPCameraSourceConfig(
            source_type=SourceType.IP_CAMERA,
            id=SOURCE_ID,
            name="Test IPCamera Source",
            stream_url="rtsp://192.168.1.100:554/stream",
            auth_required=False,
        ),
        SourceDB(
            source_type=SourceType.IP_CAMERA,
            id=str(SOURCE_ID),
            name="Test IPCamera Source",
            config_data={
                "stream_url": "rtsp://192.168.1.100:554/stream",
                "auth_required": False,
            },
        ),
    ),
    (
        ImagesFolderSourceConfig(
            source_type=SourceType.IMAGES_FOLDER,
            id=SOURCE_ID,
            name="Test Images Folder Source",
            images_folder_path="/path/to/images",
            ignore_existing_images=True,
        ),
        SourceDB(
            source_type=SourceType.IMAGES_FOLDER,
            id=str(SOURCE_ID),
            name="Test Images Folder Source",
            config_data={
                "images_folder_path": "/path/to/images",
                "ignore_existing_images": True,
            },
        ),
    ),
]


class TestSourceMapper:
    """Test cases for SourceMapper."""

    @pytest.mark.parametrize("schema_instance, expected_model", SUPPORTED_SOURCES_MAPPING.copy())
    def test_from_schema_valid_source_types(self, schema_instance, expected_model):
        """Test from_schema with valid source types."""
        result = SourceMapper.from_schema(schema_instance)

        assert isinstance(result, SourceDB)
        assert result.id == expected_model.id
        assert result.name == expected_model.name
        assert result.source_type == expected_model.source_type
        assert result.config_data == expected_model.config_data

    def test_from_schema_none_source_raises_error(self):
        """Test from_schema raises ValueError when source is None."""
        with pytest.raises(ValueError, match="Source config cannot be None"):
            SourceMapper.from_schema(None)

    @pytest.mark.parametrize("db_instance,expected_schema", [(v, k) for (k, v) in SUPPORTED_SOURCES_MAPPING.copy()])
    def test_to_schema_valid_source_types(self, db_instance, expected_schema):
        """Test to_schema with valid source types."""
        result = SourceMapper.to_schema(db_instance)

        assert result.id == expected_schema.id
        assert result.name == expected_schema.name
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
