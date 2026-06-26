# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Lock
from uuid import UUID

from app.models.source import SourceStatus
from app.workers.shm_status import read_status


class SourceStatusService:
    def __init__(self, source_status_shm: SharedMemory, source_status_lock: Lock):
        self._source_status_shm = source_status_shm
        self._source_status_lock = source_status_lock

    def get_status(self, source_id: UUID) -> SourceStatus | None:
        """Return the latest source status from shared memory, or None if not available."""
        status = read_status(SourceStatus, self._source_status_shm, self._source_status_lock)
        if status is not None and status.source_id == source_id:
            return status
        return None
