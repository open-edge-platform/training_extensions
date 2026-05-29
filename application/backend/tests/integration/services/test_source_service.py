# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.db.schema import PipelineDB, SourceDB
from app.models import SourceAdapter, SourceType
from app.models.source import (
    ImagesFolderConfig,
    ImagesFolderSourceConfig,
    IPCameraConfig,
    IPCameraSourceConfig,
    USBCameraConfig,
    USBCameraSourceConfig,
    VideoFileConfig,
    VideoFileSourceConfig,
)
from app.services import ResourceInUseError, ResourceType, SourceUpdateService
from app.services.base import ResourceWithIdAlreadyExistsError, ResourceWithNameAlreadyExistsError
from app.services.event.event_bus import EventType


@pytest.fixture
def fxt_source_update_service(fxt_event_bus, db_session) -> SourceUpdateService:
    """Fixture to provide a SourceUpdateService instance with mocked dependencies."""
    return SourceUpdateService(fxt_event_bus, db_session)


class TestSourceUpdateServiceIntegration:
    """Integration tests for ConfigurationService."""

    def test_create_source(self, fxt_usb_camera_source, fxt_source_update_service, db_session):
        """Test creating a new configuration."""
        fxt_source_update_service.create_source(
            name=fxt_usb_camera_source.name,
            source_type=fxt_usb_camera_source.source_type,
            config_data=fxt_usb_camera_source.config_data,
            source_id=fxt_usb_camera_source.id,
        )

        assert db_session.query(SourceDB).count() == 1
        created = db_session.query(SourceDB).one()
        assert created.id == str(fxt_usb_camera_source.id)
        assert created.name == fxt_usb_camera_source.name
        assert created.source_type == fxt_usb_camera_source.source_type.value

    def test_create_source_non_unique(
        self,
        fxt_db_sources,
        fxt_usb_camera_source,
        fxt_source_update_service,
        db_session,
    ):
        """Test creating a new source with the name that already exists."""
        db_session.add(fxt_db_sources[0])

        fxt_usb_camera_source.name = fxt_db_sources[0].name  # Set the same name as existing resource

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_source_update_service.create_source(
                name=fxt_usb_camera_source.name,
                source_type=fxt_usb_camera_source.source_type,
                config_data=fxt_usb_camera_source.config_data,
                source_id=fxt_usb_camera_source.id,
            )

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_usb_camera_source.name

    def test_create_source_duplicating_id(
        self,
        fxt_db_sources,
        fxt_usb_camera_source,
        fxt_source_update_service,
        db_session,
    ):
        """Test creating a new configuration with ID that already exists."""
        db_session.add(fxt_db_sources[0])
        db_session.flush()

        fxt_usb_camera_source.id = UUID(fxt_db_sources[0].id)  # Set the same ID as existing resource

        with pytest.raises(ResourceWithIdAlreadyExistsError) as excinfo:
            fxt_source_update_service.create_source(
                name=fxt_usb_camera_source.name,
                source_type=fxt_usb_camera_source.source_type,
                config_data=fxt_usb_camera_source.config_data,
                source_id=fxt_usb_camera_source.id,
            )

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_db_sources[0].id

    @pytest.mark.parametrize("is_running", [True, False])
    def test_get_active_source(
        self,
        is_running,
        fxt_db_projects,
        fxt_db_sources,
        fxt_source_update_service,
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

        active_source = fxt_source_update_service.get_active_source()

        if is_running:
            assert active_source is not None and str(active_source.id) == db_source.id
        else:
            assert active_source is None

    def test_list_sources(self, fxt_db_sources, fxt_source_update_service, db_session):
        """Test retrieving all sources."""
        db_session.add_all(fxt_db_sources)

        db_sources = fxt_source_update_service.list_all()

        assert len(db_sources) == len(fxt_db_sources)
        for i, source in enumerate(db_sources):
            assert str(source.id) == fxt_db_sources[i].id
            assert source.name == fxt_db_sources[i].name

    def test_get_source(self, fxt_db_sources, fxt_source_update_service, db_session):
        """Test retrieving a source by ID."""
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        source = fxt_source_update_service.get_by_id(UUID(db_source.id))

        assert source is not None
        assert str(source.id) == db_source.id
        assert source.name == db_source.name

    def test_update_source(self, fxt_db_sources, fxt_source_update_service, db_session):
        """Test updating a source."""
        update_data = {"name": "Updated Source", "video_path": "/new/path"}
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        source = SourceAdapter.validate_python(db_source, from_attributes=True)

        updated = fxt_source_update_service.update_source(
            source=source,
            new_name="Updated Source",
            new_config_data=VideoFileConfig(video_path="/new/path"),
        )

        assert updated.name == update_data["name"]
        assert str(updated.id) == db_source.id

        # Verify in DB
        db_source = db_session.get(SourceDB, db_source.id)
        assert db_source.name == update_data["name"]
        assert db_source.config_data["video_path"] == update_data["video_path"]

    def test_update_source_non_unique(self, fxt_db_sources, fxt_source_update_service, db_session):
        """Test updating a source with the name that already exists."""
        db_source = fxt_db_sources[0]
        db_session.add_all(fxt_db_sources[:2])
        db_session.flush()

        source = SourceAdapter.validate_python(db_source, from_attributes=True)

        with pytest.raises(ResourceWithNameAlreadyExistsError) as excinfo:
            fxt_source_update_service.update_source(
                source=source,
                new_name=fxt_db_sources[1].name,
                new_config_data=VideoFileConfig(video_path="/new/path"),
            )

        assert excinfo.value.resource_type == ResourceType.SOURCE
        assert excinfo.value.resource_id == fxt_db_sources[1].name

    def test_update_source_notify(
        self,
        fxt_db_sources,
        fxt_source_update_service,
        fxt_event_bus,
        fxt_db_projects,
        db_session,
    ):
        """Test updating a source configuration that is a part of active pipeline."""
        db_project = fxt_db_projects[0]
        db_session.add(db_project)
        db_session.flush()

        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        db_pipeline = PipelineDB(project_id=db_project.id, is_running=True, source_id=db_source.id)
        db_session.add(db_pipeline)
        db_session.flush()

        source = SourceAdapter.validate_python(db_source, from_attributes=True)

        updated = fxt_source_update_service.update_source(
            source=source,
            new_name="Updated Source",
            new_config_data=VideoFileConfig(video_path="/new/path"),
        )

        assert updated.name == "Updated Source"
        assert str(updated.id) == db_source.id

        # Verify in DB
        db_source = db_session.get(SourceDB, db_source.id)
        assert db_source.name == "Updated Source"
        assert db_source.config_data["video_path"] == "/new/path"
        fxt_event_bus.emit_event.assert_called_once_with(EventType.SOURCE_CHANGED)

    def test_delete_source(self, fxt_db_sources, fxt_source_update_service, db_session):
        """Test deleting a source."""
        db_source = fxt_db_sources[0]
        db_session.add(db_source)
        db_session.flush()

        source = SourceAdapter.validate_python(db_source, from_attributes=True)

        fxt_source_update_service.delete_source(source)

        assert db_session.query(SourceDB).count() == 0

    def test_delete_source_in_use(
        self,
        fxt_db_projects,
        fxt_db_sources,
        fxt_source_update_service,
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

        source = SourceAdapter.validate_python(db_source, from_attributes=True)

        with pytest.raises(ResourceInUseError) as exc_info:
            fxt_source_update_service.delete_source(source)

        assert exc_info.value.resource_type == ResourceType.SOURCE
        assert exc_info.value.resource_id == db_source.id
        assert db_session.query(SourceDB).count() == 1
        # Verify the error message includes the project name and state
        assert "Test Detection Project" in str(exc_info.value)
        assert "running" in str(exc_info.value)

    def test_test_source_video_file_exists(self, fxt_source_update_service, tmp_path):
        """Test test_source with a valid video file path (file exists but may not be a valid video)."""
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"\x00" * 100)

        source = VideoFileSourceConfig(
            id=uuid4(),
            source_type=SourceType.VIDEO_FILE,
            name="Test Video",
            config_data=VideoFileConfig(video_path=str(video_file)),
        )

        result = fxt_source_update_service.test_source(source)

        # File exists but is not a valid video, so cv2 can't open it
        assert result["reachable"] is False
        assert "cannot be opened as video" in result["error"]

    def test_test_source_video_file_not_found(self, fxt_source_update_service):
        """Test test_source with a non-existent video file."""
        source = VideoFileSourceConfig(
            id=uuid4(),
            source_type=SourceType.VIDEO_FILE,
            name="Missing Video",
            config_data=VideoFileConfig(video_path="/nonexistent/path/video.mp4"),
        )

        result = fxt_source_update_service.test_source(source)

        assert result["reachable"] is False
        assert "not found" in result["error"]

    def test_test_source_images_folder_exists(self, fxt_source_update_service, tmp_path):
        """Test test_source with an existing accessible images folder."""
        source = ImagesFolderSourceConfig(
            id=uuid4(),
            source_type=SourceType.IMAGES_FOLDER,
            name="Test Folder",
            config_data=ImagesFolderConfig(images_folder_path=str(tmp_path), ignore_existing_images=False),
        )

        result = fxt_source_update_service.test_source(source)

        assert result["reachable"] is True
        assert "latency_ms" in result

    def test_test_source_images_folder_not_found(self, fxt_source_update_service):
        """Test test_source with a non-existent images folder."""
        source = ImagesFolderSourceConfig(
            id=uuid4(),
            source_type=SourceType.IMAGES_FOLDER,
            name="Missing Folder",
            config_data=ImagesFolderConfig(images_folder_path="/nonexistent/folder", ignore_existing_images=False),
        )

        result = fxt_source_update_service.test_source(source)

        assert result["reachable"] is False
        assert "not found" in result["error"]

    def test_test_source_usb_camera_not_available(self, fxt_source_update_service):
        """Test test_source with a USB camera device that is not available."""
        source = USBCameraSourceConfig(
            id=uuid4(),
            source_type=SourceType.USB_CAMERA,
            name="Unavailable USB Camera",
            config_data=USBCameraConfig(device_id=999),
        )

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("app.services.source_service.cv2.VideoCapture", return_value=mock_cap):
            result = fxt_source_update_service.test_source(source)

        assert result["reachable"] is False
        assert "Cannot open USB camera device" in result["error"]
        mock_cap.release.assert_called_once()

    def test_test_source_usb_camera_reachable(self, fxt_source_update_service):
        """Test test_source with a reachable USB camera (mocked cv2 capture)."""
        source = USBCameraSourceConfig(
            id=uuid4(),
            source_type=SourceType.USB_CAMERA,
            name="Reachable USB Camera",
            config_data=USBCameraConfig(device_id=0),
        )

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        with patch("app.services.source_service.cv2.VideoCapture", return_value=mock_cap):
            result = fxt_source_update_service.test_source(source)

        assert result["reachable"] is True
        assert "latency_ms" in result
        mock_cap.release.assert_called_once()

    def test_test_source_ip_camera_unreachable(self, fxt_source_update_service):
        """Test test_source with an unreachable IP camera stream."""
        source = IPCameraSourceConfig(
            id=uuid4(),
            source_type=SourceType.IP_CAMERA,
            name="Unreachable IP Camera",
            config_data=IPCameraConfig(stream_url="rtsp://192.0.2.1:554/stream", auth_required=False),
        )

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("app.services.source_service.cv2.VideoCapture", return_value=mock_cap):
            result = fxt_source_update_service.test_source(source)

        assert result["reachable"] is False
        assert "Cannot open stream" in result["error"]
        mock_cap.release.assert_called_once()

    def test_test_source_ip_camera_reachable(self, fxt_source_update_service):
        """Test test_source with a reachable IP camera (mocked cv2 capture)."""
        source = IPCameraSourceConfig(
            id=uuid4(),
            source_type=SourceType.IP_CAMERA,
            name="Reachable IP Camera",
            config_data=IPCameraConfig(stream_url="rtsp://192.168.1.100:554/stream", auth_required=False),
        )

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())
        with patch("app.services.source_service.cv2.VideoCapture", return_value=mock_cap):
            result = fxt_source_update_service.test_source(source)

        assert result["reachable"] is True
        assert "latency_ms" in result
        mock_cap.release.assert_called_once()
