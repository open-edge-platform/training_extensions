# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import time
from collections.abc import Generator
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from testcontainers.compose import DockerCompose

from app.entities.ip_camera_stream import IPCameraStream
from app.schemas.source import IPCameraSourceConfig, SourceType


@pytest.fixture
def ip_camera() -> Generator[str]:
    """Start IP camera stream using testcontainers."""
    compose = DockerCompose("tests/integration/fixtures", compose_file_name="docker-compose.test.ip-camera.yaml")
    compose.start()

    # Wait for IP camera service to be ready
    camera_logs = compose.get_logs("ip-camera")

    # Wait for the camera service to start
    timeout = 30
    start_time = time.time()
    while time.time() - start_time < timeout:
        if re.search(r"Stream #0:0: Video: rawvideo", camera_logs[0] if camera_logs else ""):
            break
        time.sleep(1)
        camera_logs = compose.get_logs("ip-camera")
    else:
        raise TimeoutError("IP camera service did not start within timeout")

    camera_port = compose.get_service_port("ip-camera", 8080)

    yield f"http://localhost:{camera_port}/stream"

    compose.stop()


class TestIPCameraStream:
    """Integration tests for IPCameraStream functionality."""

    @pytest.fixture()
    def config(self, ip_camera: str) -> IPCameraSourceConfig:
        return IPCameraSourceConfig(source_type=SourceType.IP_CAMERA, stream_url=ip_camera)

    @pytest.fixture()
    def stream(self, config: IPCameraSourceConfig) -> Generator[IPCameraStream]:
        with IPCameraStream(config) as ip_camera_stream:
            yield ip_camera_stream

    def test_can_read_frames(self, stream: IPCameraStream):
        """Test that IPCameraStream can read multiple consecutive frames."""
        assert stream.is_real_time()

        for _ in range(5):
            data = stream.get_data()
            frame, metadata = data.frame_data, data.source_metadata
            assert frame is not None
            assert frame.shape[2] == 3
            assert frame.dtype == "uint8"

            assert metadata["source_type"] == SourceType.IP_CAMERA.value
            assert metadata["stream_url"] == stream.source

    def test_close(self, config: IPCameraSourceConfig):
        """Test that IPCameraStream can be closed."""
        with IPCameraStream(config) as stream:
            data = stream.get_data()
            assert data.frame_data.shape[2] == 3

    def test_invalid_url(self):
        """Test that IPCameraStream handles invalid URLs with specific error."""
        invalid_url = "http://invalid-url:9999/stream"
        config = IPCameraSourceConfig(source_type=SourceType.IP_CAMERA, stream_url=invalid_url)

        # Should raise exception during initialization
        with pytest.raises(RuntimeError, match=f"Could not open video source: {invalid_url}"):
            IPCameraStream(config)

    def test_reconnection_after_frame_read_failure(self, config: IPCameraSourceConfig):
        """Test that IPCameraStream can reconnect after frame read failures."""
        with IPCameraStream(config) as stream:
            data = stream.get_data()
            assert data.frame_data is not None

            # Mock cv2.VideoCapture.read to simulate failure then success
            mock_cap = MagicMock(spec=cv2.VideoCapture)
            call_count = 0

            def mock_read():
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    return False, None

                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                return True, dummy_frame

            mock_cap.read.side_effect = mock_read
            mock_cap.isOpened.return_value = True

            def _initialize_capture():
                stream.cap = mock_cap

            mock_initialize_capture = MagicMock(side_effect=_initialize_capture)
            stream._initialize_capture = mock_initialize_capture  # type: ignore[method-assign]
            stream.cap = mock_cap

            # This should trigger reconnection logic and succeed
            data = stream.get_data()
            assert data.frame_data is not None
            assert call_count == 3  # Verify reconnection attempts were made
            assert mock_initialize_capture.call_count == 2

    def test_max_retries_exceeded(self, config: IPCameraSourceConfig):
        """Test that IPCameraStream raises exception when max retries are exceeded."""
        with IPCameraStream(config) as stream:
            mock_cap = MagicMock(spec=cv2.VideoCapture)
            mock_cap.read.return_value = (False, None)
            mock_cap.isOpened.return_value = True

            def _initialize_capture():
                stream.cap = mock_cap

            mock_initialize_capture = MagicMock(side_effect=_initialize_capture)
            stream._initialize_capture = mock_initialize_capture  # type: ignore[method-assign]
            stream.cap = mock_cap

            # Should exhaust retries and raise RuntimeError
            with pytest.raises(RuntimeError, match="Failed to capture frame from IP camera after multiple retries"):
                stream.get_data()

                assert mock_initialize_capture.call_count == 3
