# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any, Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel

from app.core.models import BaseIDModel


class JobStatus(IntEnum):
    PENDING = 0
    RUNNING = 1

    CANCELLING = 5  # intermediate state when a running job is being cancelled

    DONE = 10
    FAILED = 11
    CANCELLED = 12


class JobType(StrEnum):
    TRAIN = "train"
    QUANTIZE = "quantize"
    PREPARE_DATASET_FOR_IMPORT = "prepare_dataset_for_import"
    IMPORT_DATASET_AS_NEW_PROJECT = "import_dataset_as_new_project"
    IMPORT_DATASET_TO_PROJECT = "import_dataset_to_project"
    EXPORT_DATASET = "export_dataset"
    STAGE_DATASET = "stage_dataset"


def now_utc_ts() -> float:
    """Get the current UTC timestamp as a float."""
    return datetime.now(tz=UTC).timestamp()


class JobParams(BaseModel):
    pass


JobParamsT = TypeVar("JobParamsT", bound=JobParams)


class Job(BaseIDModel, Generic[JobParamsT]):
    job_type: JobType
    params: JobParamsT  # parameters of the job to be serialized and send to a runnable context
    status: JobStatus = JobStatus.PENDING
    submitted_at: float = now_utc_ts()
    updated_at: float = now_utc_ts()
    started_at: float | None = None
    message: str | None = None
    progress: float = 0.0  # percentage of completion
    result: dict | None = None  # result of the job, if completed
    error: str | None = None

    @property
    def log_file(self) -> str:
        return f"{self.job_type.lower()}-{self.id}.log"

    def on_complete(self) -> None:
        """Hook called when the job is finished successfully or failed."""

    def start(self) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = now_utc_ts()
        self.updated_at = now_utc_ts()

    def advance(self, percent: float, msg: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        if percent > 0.0:
            self.progress = min(100.0, percent)
        if msg:
            self.message = msg
        if metadata:
            for item in metadata.items():
                setattr(self.params, item[0], item[1])
        self.updated_at = now_utc_ts()

    def finish(self) -> None:
        self.status = JobStatus.DONE
        self.progress = 100.0
        self.updated_at = now_utc_ts()
        self.on_complete()

    def fail(self, msg: str) -> None:
        self.status = JobStatus.FAILED
        self.updated_at = now_utc_ts()
        self.error = msg
        self.on_complete()

    def cancelling(self) -> None:
        self.status = JobStatus.CANCELLING
        self.updated_at = now_utc_ts()
        self.message = "Job is being cancelled"

    def cancel(self) -> None:
        self.status = JobStatus.CANCELLED
        self.updated_at = now_utc_ts()
        self.message = "Job was cancelled"


class ProjectJob(Job, Generic[JobParamsT]):
    project_id: UUID
    params: JobParamsT
