# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, Field

from app.core.jobs.models import Job, JobStatus, JobType

from .dataset_import import ImportDatasetMetadata
from .training import TrainingMetadata

JobMetadata = Annotated[
    TrainingMetadata | ImportDatasetMetadata, Field(..., description="Metadata associated with the job")
]


class JobView(BaseModel):
    """Response schema for job creation."""

    job_id: UUID = Field(..., description="Job identifier")
    job_type: JobType = Field(..., description="Type of the job")
    metadata: JobMetadata = Field(..., description="Metadata associated with the job")
    status: str = Field(..., description="Job status")
    progress: float = Field(..., description="Job progress percentage (0-100)")
    message: str | None = Field(None, description="Additional information about the job status")
    error: str | None = Field(None, description="Error message if the job failed")
    started_at: datetime | None = Field(None, description="Timestamp when the job started")
    finished_at: datetime | None = Field(None, description="Timestamp when the job finished")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "job_type": "train",
                "metadata": {
                    "project": {"id": "7b073838-99d3-42ff-9018-4e901eb047fc"},
                    "model": {
                        "id": "1a2b3c4d-5e6f-7a8b-9c0d-1e2f3a4b5c6d",
                        "architecture": "Custom_Object_Detection_Gen3_ATSS",
                        "parent_revision_id": "ef3983f1-cef0-4ebe-91db-7330f1dd6e27",
                        "dataset_revision_id": "2b073838-99d3-42ff-9018-4e901eb047fc",
                    },
                },
                "status": "done",
                "progress": 100.0,
                "message": "Training completed successfully",
                "error": None,
                "started_at": "2023-10-01T12:00:00Z",
                "finished_at": "2023-10-01T12:30:00Z",
            }
        }
    }

    @staticmethod
    def of(job: Job) -> "JobView":
        match job.job_type:
            case JobType.TRAIN:
                metadata = TrainingMetadata.model_validate(job)
            case (
                JobType.IMPORT_DATASET_AS_NEW_PROJECT
                | JobType.IMPORT_DATASET_TO_PROJECT
                | JobType.PREPARE_DATASET_FOR_IMPORT
            ):
                metadata = ImportDatasetMetadata.model_validate(job)
            case _:
                raise ValueError("Metadata is not defined for this job type")

        return JobView(
            job_id=job.id,
            job_type=job.job_type,
            metadata=metadata,
            status=job.status.name,
            progress=job.progress,
            message=job.message,
            error=job.error,
            started_at=datetime.fromtimestamp(job.started_at, tz=UTC) if job.started_at else None,
            finished_at=datetime.fromtimestamp(job.updated_at, tz=UTC) if job.status >= JobStatus.DONE else None,
        )
