# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.models.sink import SinkStatus, SinkStatusCode
from app.services.sink_status_service import SinkStatusService
from app.workers.sink_status_holder import SinkStatusHolder


class TestSinkStatusService:
    """Test cases for SinkStatusService.get_status"""

    @pytest.fixture
    def fxt_sink_status_holder(self) -> MagicMock:
        return MagicMock(spec=SinkStatusHolder)

    @pytest.fixture
    def fxt_sink_status_service(self, fxt_sink_status_holder: MagicMock) -> SinkStatusService:
        return SinkStatusService(sink_status_holder=fxt_sink_status_holder)

    def test_get_status_matching_sink_id(
        self, fxt_sink_status_service: SinkStatusService, fxt_sink_status_holder: MagicMock
    ):
        """Return the status when its sink_id matches the requested one."""
        sink_id = uuid4()
        status = SinkStatus(code=SinkStatusCode.OK, sink_id=sink_id, message="dispatching")
        fxt_sink_status_holder.status = status

        result = fxt_sink_status_service.get_status(sink_id)

        assert result is status

    def test_get_status_mismatched_sink_id(
        self, fxt_sink_status_service: SinkStatusService, fxt_sink_status_holder: MagicMock
    ):
        """Return None when the latest status belongs to a different sink."""
        fxt_sink_status_holder.status = SinkStatus(code=SinkStatusCode.OK, sink_id=uuid4())

        result = fxt_sink_status_service.get_status(uuid4())

        assert result is None

    def test_get_status_no_status_available(
        self, fxt_sink_status_service: SinkStatusService, fxt_sink_status_holder: MagicMock
    ):
        """Return None when no status is available in the holder."""
        fxt_sink_status_holder.status = None

        result = fxt_sink_status_service.get_status(uuid4())

        assert result is None
