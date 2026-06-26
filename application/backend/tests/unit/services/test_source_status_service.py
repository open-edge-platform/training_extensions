# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from app.models.source import SourceStatus, SourceStatusCode
from app.services.source_status_service import SourceStatusService


class TestSourceStatusService:
    """Test cases for SourceStatusService.get_status"""

    @pytest.fixture
    def fxt_source_status_service(self) -> SourceStatusService:
        return SourceStatusService(source_status_shm=MagicMock(), source_status_lock=MagicMock())

    def test_get_status_matching_source_id(self, fxt_source_status_service: SourceStatusService):
        """Return the status when its source_id matches the requested one."""
        source_id = uuid4()
        status = SourceStatus(code=SourceStatusCode.OK, source_id=source_id, message="streaming")

        with patch("app.services.source_status_service.read_status", return_value=status) as mock_read:
            result = fxt_source_status_service.get_status(source_id)

        assert result is status
        mock_read.assert_called_once_with(
            SourceStatus,
            fxt_source_status_service._source_status_shm,
            fxt_source_status_service._source_status_lock,
        )

    def test_get_status_mismatched_source_id(self, fxt_source_status_service: SourceStatusService):
        """Return None when the latest status belongs to a different source."""
        status = SourceStatus(code=SourceStatusCode.OK, source_id=uuid4())

        with patch("app.services.source_status_service.read_status", return_value=status):
            result = fxt_source_status_service.get_status(uuid4())

        assert result is None

    def test_get_status_no_status_available(self, fxt_source_status_service: SourceStatusService):
        """Return None when no status is available in shared memory."""
        with patch("app.services.source_status_service.read_status", return_value=None):
            result = fxt_source_status_service.get_status(uuid4())

        assert result is None
