# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re
import sys
import time
from collections.abc import Generator
from unittest.mock import MagicMock
from uuid import uuid4

import cv2
import pytest
from testcontainers.compose import DockerCompose

from app.models import IPCameraSourceConfig, SourceType
from app.models.source import IPCameraConfig
from app.stream.ip_camera_stream import IPCameraStream


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


@pytest.mark.skipif(sys.platform == "win32", reason="Docker-based tests not supported on Windows CI")
class TestIPCameraStream:
    """Integration tests for IPCameraStream functionality."""

    @pytest.fixture()
    def config(self, ip_camera: str) -> IPCameraSourceConfig:
        return IPCameraSourceConfig(
            source_type=SourceType.IP_CAMERA,
            id=uuid4(),
            name="Test IP Camera",
            config_data=IPCameraConfig(stream_url=ip_camera),
        )

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
            assert data is not None
            assert data.frame_data.shape[2] == 3

    def test_invalid_url(self):
        """Test that IPCameraStream handles invalid URLs with specific error."""
        invalid_url = "http://invalid-url:9999/stream"
        config = IPCameraSourceConfig(
            source_type=SourceType.IP_CAMERA,
            id=uuid4(),
            name="Test IP Camera",
            config_data=IPCameraConfig(stream_url=invalid_url),
        )

        # Should raise exception during initialization
        with pytest.raises(RuntimeError, match=f"Could not open video source: {invalid_url}"):
            IPCameraStream(config)

    def test_reconnects_after_transient_failure(self, config: IPCameraSourceConfig):
        """_reconnect() should re-open the capture and return True on success."""
        with IPCameraStream(config) as stream:
            # Stop the reader thread so we can drive _reconnect() deterministically.
            stream._stop_reader.set()
            stream._reader_thread.join(timeout=2)
            stream._stop_reader.clear()

            recovered_cap = MagicMock(spec=cv2.VideoCapture)
            recovered_cap.isOpened.return_value = True

            init_calls = {"n": 0}

            def _initialize_capture():
                init_calls["n"] += 1
                stream.cap = recovered_cap

            stream._initialize_capture = MagicMock(side_effect=_initialize_capture)  # type: ignore[method-assign]
            stream.BACKOFF_FACTOR = 0.0

            assert stream._reconnect() is True
            assert init_calls["n"] == 1

    def test_permanent_failure_publishes_error(self, config: IPCameraSourceConfig):
        """After MAX_RECONNECT_ATTEMPTS the reader should give up and publish an error."""
        with IPCameraStream(config) as stream:
            stream._stop_reader.set()
            stream._reader_thread.join(timeout=2)
            stream._stop_reader.clear()

            failing_cap = MagicMock(spec=cv2.VideoCapture)
            failing_cap.isOpened.return_value = False

            def _initialize_capture():
                stream.cap = failing_cap

            stream._initialize_capture = MagicMock(side_effect=_initialize_capture)  # type: ignore[method-assign]
            stream.cap = failing_cap
            stream.BACKOFF_FACTOR = 0.0

            assert stream._reconnect() is False
            assert stream._initialize_capture.call_count == stream.MAX_RECONNECT_ATTEMPTS
            assert isinstance(stream._reader_error, RuntimeError)
            assert "permanently unavailable" in str(stream._reader_error)
