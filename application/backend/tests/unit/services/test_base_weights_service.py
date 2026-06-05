# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from app.services import BaseWeightsService


@pytest.fixture()
def fxt_pretrained_weights_dir(tmp_path: Path) -> Generator[Path]:
    """Set up a temporary data directory for tests."""
    pretrained_weights_dir = tmp_path / "pretrained_weights"
    pretrained_weights_dir.mkdir(parents=True, exist_ok=True)
    yield pretrained_weights_dir


@pytest.fixture
def fxt_base_weights_service(fxt_pretrained_weights_dir: Path) -> BaseWeightsService:
    """Create a BaseWeightsService instance for testing."""
    return BaseWeightsService(fxt_pretrained_weights_dir.parent)


class TestBaseWeightsServiceRetryLogic:
    """Test cases for the retry and proxy fallback logic of BaseWeightsService."""

    def _make_mock_session(self) -> MagicMock:
        """Create a mock session that can be used as a context manager."""
        session = MagicMock()
        session.__enter__ = MagicMock(return_value=session)
        session.__exit__ = MagicMock(return_value=False)
        return session

    def test_check_disk_space_falls_back_when_proxy_fails(self, fxt_base_weights_service):
        """Test that disk space check falls back to direct connection when proxy fails."""
        proxy_session = self._make_mock_session()
        proxy_session.head.side_effect = requests.RequestException("proxy error")
        mock_response = MagicMock()
        mock_response.headers.get.return_value = str(100 * 1024 * 1024)
        direct_session = self._make_mock_session()
        direct_session.head.return_value = mock_response
        with (
            patch.object(
                fxt_base_weights_service,
                "_build_retry_session",
                side_effect=[proxy_session, direct_session],
            ),
            patch("app.services.base_weights_service.shutil.disk_usage") as mock_disk_usage,
        ):
            mock_disk_usage.return_value = MagicMock(free=10 * 1024**3)
            fxt_base_weights_service._check_disk_space("https://example.com/weights.pth")
        assert proxy_session.head.called
        assert direct_session.head.called

    def test_check_disk_space_both_connections_fail_uses_default_size(self, fxt_base_weights_service):
        """Test that disk space check uses the default 500 MB when both proxy and direct connections fail."""
        failing_session = self._make_mock_session()
        failing_session.head.side_effect = requests.RequestException("connection error")
        with (
            patch.object(
                fxt_base_weights_service,
                "_build_retry_session",
                side_effect=[failing_session, failing_session],
            ),
            patch("app.services.base_weights_service.shutil.disk_usage") as mock_disk_usage,
        ):
            mock_disk_usage.return_value = MagicMock(free=10 * 1024**3)
            fxt_base_weights_service._check_disk_space("https://example.com/weights.pth")
        mock_disk_usage.assert_called_once()

    def test_download_weights_falls_back_when_proxy_fails(self, fxt_base_weights_service):
        """Test that download falls back to direct connection when proxy download fails."""
        proxy_session = self._make_mock_session()
        proxy_session.get.side_effect = requests.RequestException("proxy error")
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.iter_content.return_value = [b"fake weights data"]
        direct_session = self._make_mock_session()
        direct_session.get.return_value = mock_response
        local_path = fxt_base_weights_service.pretrained_weights_dir / "detection" / "test_weights.pth"
        with (
            patch.object(fxt_base_weights_service, "_check_disk_space"),
            patch.object(
                fxt_base_weights_service,
                "_build_retry_session",
                side_effect=[proxy_session, direct_session],
            ),
            patch.object(fxt_base_weights_service, "_verify_file_integrity", return_value=True),
        ):
            fxt_base_weights_service._download_weights(
                remote_url="https://example.com/weights.pth",
                local_path=local_path,
                sha_sum="dummy_sha",
            )
        assert proxy_session.get.called
        assert direct_session.get.called

    def test_download_weights_raises_when_both_connections_fail(self, fxt_base_weights_service):
        """Test that download raises RuntimeError when both proxy and direct connections fail."""
        proxy_session = self._make_mock_session()
        proxy_session.get.side_effect = requests.RequestException("proxy error")
        direct_session = self._make_mock_session()
        direct_session.get.side_effect = requests.RequestException("direct error")
        local_path = fxt_base_weights_service.pretrained_weights_dir / "detection" / "test_weights.pth"
        with (
            patch.object(fxt_base_weights_service, "_check_disk_space"),
            patch.object(
                fxt_base_weights_service,
                "_build_retry_session",
                side_effect=[proxy_session, direct_session],
            ),
            pytest.raises(RuntimeError, match="weights.pth"),
        ):
            fxt_base_weights_service._download_weights(
                remote_url="https://example.com/weights.pth",
                local_path=local_path,
                sha_sum="dummy_sha",
            )
