# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the shared-memory status IPC helpers (write_status / read_status).

These tests use a real SharedMemory block and real multiprocessing locks (instead of mocks) so
that the full serialize -> shared-memory -> deserialize round trip is exercised, including across
a spawned process boundary, which is the actual scenario these helpers are designed for.
"""

import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Lock
from uuid import uuid4

import pytest

from app.models.inference import InferenceWorkerStatus, InferenceWorkerStatusCode
from app.models.sink import SinkStatus, SinkStatusCode
from app.models.source import SourceStatus, SourceStatusCode
from app.workers.shm_status import STATUS_SHM_SIZE, read_status, write_status


def _write_status_in_subprocess(shm_name: str, lock: Lock, status: SourceStatus) -> None:
    """Top-level (picklable) target that attaches to the shared memory by name and writes a status.

    Mirrors how worker processes report their status: they open the scheduler-owned block by name
    and write into it under the shared lock.
    """
    shm = SharedMemory(name=shm_name)
    try:
        write_status(status, shm, lock)
    finally:
        shm.close()


class TestShmStatusIntegration:
    """Integration tests for write_status and read_status against real shared memory."""

    @pytest.fixture
    def mp_ctx(self):
        """Spawn context matching how worker processes are actually started."""
        return multiprocessing.get_context("spawn")

    @pytest.fixture
    def lock(self, mp_ctx) -> Lock:
        return mp_ctx.Lock()

    @pytest.fixture
    def shm(self):
        """A real shared-memory block, unlinked after the test."""
        block = SharedMemory(create=True, size=STATUS_SHM_SIZE)
        try:
            yield block
        finally:
            block.close()
            block.unlink()

    def test_read_status_empty_returns_none(self, shm, lock):
        """A freshly created (zero-length) block yields None."""
        assert read_status(SourceStatus, shm, lock) is None

    def test_write_then_read_round_trip(self, shm, lock):
        """A status written into shared memory is read back as an equal model."""
        status = SourceStatus(code=SourceStatusCode.OK, source_id=uuid4(), message="streaming")

        write_status(status, shm, lock)
        result = read_status(SourceStatus, shm, lock)

        assert result == status

    @pytest.mark.parametrize(
        "model_class, status",
        [
            (SourceStatus, SourceStatus(code=SourceStatusCode.ERROR, source_id=uuid4(), message="boom")),
            (SinkStatus, SinkStatus(code=SinkStatusCode.OK, sink_id=uuid4(), message="dispatching")),
            (
                InferenceWorkerStatus,
                InferenceWorkerStatus(code=InferenceWorkerStatusCode.OK, model_id=uuid4(), message="inferring"),
            ),
        ],
    )
    def test_round_trip_for_each_status_model(self, shm, lock, model_class, status):
        """The helpers work for every status model type that flows through shared memory."""
        write_status(status, shm, lock)

        assert read_status(model_class, shm, lock) == status

    def test_latest_write_overwrites_previous(self, shm, lock):
        """Reading returns the most recently written status."""
        first = SourceStatus(code=SourceStatusCode.OK, source_id=uuid4(), message="first")
        second = SourceStatus(code=SourceStatusCode.FINISHED, source_id=uuid4(), message="second")

        write_status(first, shm, lock)
        write_status(second, shm, lock)

        assert read_status(SourceStatus, shm, lock) == second

    def test_oversized_message_is_truncated(self, shm, lock):
        """A payload larger than the buffer has its message field halved so it still fits and reads back."""
        oversized_message = "x" * (shm.size + 100)
        status = SourceStatus(code=SourceStatusCode.ERROR, source_id=uuid4(), message=oversized_message)

        write_status(status, shm, lock)
        result = read_status(SourceStatus, shm, lock)

        assert result is not None
        assert result.code == SourceStatusCode.ERROR
        assert result.source_id == status.source_id
        assert result.message == oversized_message[: len(oversized_message) // 2]

    def test_cross_process_write_and_read(self, shm, lock, mp_ctx):
        """A status written by a spawned process is read back in the parent process (real IPC path)."""
        status = SourceStatus(code=SourceStatusCode.OK, source_id=uuid4(), message="from-worker")

        process = mp_ctx.Process(target=_write_status_in_subprocess, args=(shm.name, lock, status))
        process.start()
        process.join(timeout=30)

        assert process.exitcode == 0
        assert read_status(SourceStatus, shm, lock) == status
