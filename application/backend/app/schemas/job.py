# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.core.jobs.models import Job, JobStatus, JobType, TrainingJob


class TrainingRequestParams(BaseModel):
    """Request schema for training a new model."""

    model_architecture_id: str = Field(..., description="Model architecture identifier")
    parent_model_revision_id: UUID | None = Field(
        None, description="Parent model revision ID for fine-tuning, null for training from scratch"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_architecture_id": "Custom_Object_Detection_Gen3_ATSS",
                "parent_model_revision_id": "ef3983f1-cef0-4ebe-91db-7330f1dd6e27",
            }
        }
    }


class BaseJobRequest(BaseModel):
    job_type: JobType = Field(..., description="Type of the job to be created")
    project_id: UUID = Field(..., description="ID of the project associated with the job")


class TrainingRequest(BaseJobRequest):
    job_type: Literal[JobType.TRAIN]
    parameters: TrainingRequestParams = Field(..., description="Parameters required for the job")

    model_config = {
        "json_schema_extra": {
            "example": {
                "job_type": "train",
                "project_id": "7b073838-99d3-42ff-9018-4e901eb047fc",
                "parameters": {
                    "model_architecture_id": "Custom_Object_Detection_Gen3_ATSS",
                    "parent_model_revision_id": "ef3983f1-cef0-4ebe-91db-7330f1dd6e27",
                },
            }
        }
    }


JobRequest = Annotated[TrainingRequest, Field(discriminator="job_type")]


class ProjectMetadata(BaseModel):
    """Metadata about a project."""

    id: UUID = Field(..., description="Project identifier")


class ModelMetadata(BaseModel):
    """Metadata about a model."""

    id: UUID = Field(..., description="Model identifier")
    architecture: str = Field(..., description="Model architecture identifier")
    parent_revision_id: UUID | None = Field(
        None, description="Parent model revision ID for fine-tuning, null if trained from scratch"
    )
    dataset_revision_id: UUID = Field(..., description="Dataset revision ID used for training")


class TrainingMetadata(BaseModel):
    """Metadata associated with a training job."""

    project: ProjectMetadata = Field(..., description="Project associated with the training job")
    model: ModelMetadata = Field(..., description="Model being trained")

    @staticmethod
    def of(job: TrainingJob) -> "TrainingMetadata":
        return TrainingMetadata(
            project=ProjectMetadata(id=job.project_id),
            model=ModelMetadata(
                id=job.params.model_id,
                architecture=job.params.model_architecture_id,
                parent_revision_id=job.params.parent_model_revision_id,
                dataset_revision_id=UUID("00000000-0000-0000-0000-000000000000"),  # TODO set correct value
            ),
        )


JobMetadata = Annotated[TrainingMetadata, Field(..., description="Metadata associated with the job")]


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
                metadata = TrainingMetadata.of(job)  # type: ignore[arg-type]
            case _:
                raise NotImplementedError("JobView is not implemented for this job type")

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
