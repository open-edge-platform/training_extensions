# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.models.sink import SinkStatus
from app.workers.sink_status_holder import SinkStatusHolder


class SinkStatusService:
    def __init__(self, sink_status_holder: SinkStatusHolder):
        self._sink_status_holder = sink_status_holder

    def get_status(self, sink_id: UUID) -> SinkStatus | None:
        """Return the latest sink status, or None if not available."""
        status = self._sink_status_holder.status
        if status is not None and status.sink_id == sink_id:
            return status
        return None
