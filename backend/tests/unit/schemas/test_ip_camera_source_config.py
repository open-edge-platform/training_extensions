import os
from unittest.mock import patch

import pytest

from app.schemas.source import IP_CAMERA_PASSWORD, IP_CAMERA_USERNAME, IPCameraSourceConfig, SourceType


class TestIpCameraSourceConfig:
    """Test cases for IpCameraSourceConfig."""

    def test_basic_initialization(self):
        """Test basic config initialization without auth."""
        config = IPCameraSourceConfig(
            source_type=SourceType.IP_CAMERA, stream_url="rtsp://192.168.1.100:554/stream", auth_required=False
        )

        assert config.source_type == SourceType.IP_CAMERA
        assert config.stream_url == "rtsp://192.168.1.100:554/stream"
        assert config.get_configured_stream_url() == "rtsp://192.168.1.100:554/stream"
        assert not config.auth_required

    def test_default_auth_required_value(self):
        """Test that auth_required defaults to False."""
        config = IPCameraSourceConfig(source_type=SourceType.IP_CAMERA, stream_url="rtsp://192.168.1.100:554/stream")

        assert not config.auth_required

    @pytest.mark.parametrize(
        "env_vars,description",
        [
            ({}, "both username and password missing"),
            ({IP_CAMERA_USERNAME: "testuser"}, "password missing"),
            ({IP_CAMERA_PASSWORD: "testpass"}, "username missing"),
            ({IP_CAMERA_USERNAME: "", IP_CAMERA_PASSWORD: "testpass"}, "username is empty"),
            ({IP_CAMERA_USERNAME: "testuser", IP_CAMERA_PASSWORD: ""}, "password is empty"),
        ],
    )
    def test_get_configured_stream_url_invalid_credentials(self, env_vars, description):
        """Test error cases for invalid or missing credentials."""
        with patch.dict(os.environ, env_vars, clear=True):
            config = IPCameraSourceConfig(
                source_type=SourceType.IP_CAMERA, stream_url="rtsp://192.168.1.100:554/stream", auth_required=True
            )

            with pytest.raises(RuntimeError, match="IP camera credentials not provided"):
                config.get_configured_stream_url()

    @patch.dict(os.environ, {IP_CAMERA_USERNAME: "testuser", IP_CAMERA_PASSWORD: "testpass"})
    def test_get_configured_stream_url_with_auth_success(self):
        """Test URL configuration with valid credentials."""
        config = IPCameraSourceConfig(
            source_type=SourceType.IP_CAMERA, stream_url="rtsp://192.168.1.100:554/stream", auth_required=True
        )

        result = config.get_configured_stream_url()
        assert result == "rtsp://testuser:testpass@192.168.1.100:554/stream"

    @patch.dict(os.environ, {IP_CAMERA_USERNAME: "testuser", IP_CAMERA_PASSWORD: "testpass"})
    def test_different_url_schemes(self):
        """Test URL configuration with different schemes."""
        test_cases = [
            ("http://192.168.1.100/stream", "http://testuser:testpass@192.168.1.100/stream"),
            ("https://camera.local:443/video", "https://testuser:testpass@camera.local:443/video"),
            ("rtsp://10.0.0.1:554/live", "rtsp://testuser:testpass@10.0.0.1:554/live"),
        ]

        for input_url, expected_url in test_cases:
            config = IPCameraSourceConfig(source_type=SourceType.IP_CAMERA, stream_url=input_url, auth_required=True)

            result = config.get_configured_stream_url()
            assert result == expected_url

    @patch.dict(os.environ, {IP_CAMERA_USERNAME: "testuser", IP_CAMERA_PASSWORD: "testpass"})
    def test_url_with_path_and_query_params(self):
        """Test URL configuration preserves path and query parameters."""
        config = IPCameraSourceConfig(
            source_type=SourceType.IP_CAMERA,
            stream_url="rtsp://192.168.1.100:554/live/stream1?quality=high&format=h264",
            auth_required=True,
        )

        result = config.get_configured_stream_url()
        expected = "rtsp://testuser:testpass@192.168.1.100:554/live/stream1?quality=high&format=h264"
        assert result == expected
