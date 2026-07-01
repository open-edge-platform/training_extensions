# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Lock
from uuid import UUID

from app.models.inference import InferenceWorkerStatus
from app.workers.shm_status import read_status


class InferenceStatusService:
    def __init__(self, inference_status_shm: SharedMemory, inference_status_lock: Lock):
        self._inference_status_shm = inference_status_shm
        self._inference_status_lock = inference_status_lock

    def get_status(self, model_id: UUID) -> InferenceWorkerStatus | None:
        """Return the latest inference worker status from shared memory, or None if not available."""
        status = read_status(InferenceWorkerStatus, self._inference_status_shm, self._inference_status_lock)
        if status is not None and status.model_id == model_id:
            return status
        return None
