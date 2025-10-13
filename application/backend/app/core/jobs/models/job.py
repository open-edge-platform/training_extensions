# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from enum import IntEnum
from uuid import UUID, uuid4

from pydantic import BaseModel


class JobStatus(IntEnum):
    PENDING = 0
    RUNNING = 1

    CANCELLING = 5  # intermediate state when a running job is being cancelled

    DONE = 10
    FAILED = 11
    CANCELLED = 12


def now_utc_ts() -> float:
    """Get the current UTC timestamp as a float."""
    return datetime.now(tz=UTC).timestamp()


class Job(BaseModel):
    id: UUID
    status: JobStatus
    submitted_at: float
    started_at: float | None = None
    updated_at: float = now_utc_ts()
    message: str | None = None
    progress: float = 0.0  # percentage of completion
    result: dict | None = None  # result of the job, if completed
    error: str | None = None

    @staticmethod
    def new() -> "Job":
        return Job(
            id=uuid4(),
            status=JobStatus.PENDING,
            submitted_at=now_utc_ts(),
        )

    def start(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = now_utc_ts()
        self.updated_at = now_utc_ts()

    def advance(self, percent: float) -> None:
        self.progress = max(0.0, min(100.0, percent))
        self.updated_at = now_utc_ts()

    def finish(self) -> None:
        self.status = JobStatus.DONE
        self.progress = 100.0
        self.updated_at = now_utc_ts()

    def fail(self, msg: str) -> None:
        self.status = JobStatus.FAILED
        self.updated_at = now_utc_ts()
        self.error = msg

    def cancelling(self) -> None:
        self.status = JobStatus.CANCELLING
        self.updated_at = now_utc_ts()
        self.message = "Job is being cancelled"

    def cancel(self) -> None:
        self.status = JobStatus.CANCELLED
        self.updated_at = now_utc_ts()
        self.message = "Job was cancelled"
