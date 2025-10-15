# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class TrainingRequest(BaseModel):
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


class JobType(StrEnum):
    TRAIN = "train"


class JobRequest(BaseModel):
    job_type: JobType = Field(..., description="Type of the job to be created")
    project_id: UUID = Field(..., description="ID of the project associated with the job")
    parameters: TrainingRequest = Field(..., description="Parameters required for the job")

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


class JobResponse(BaseModel):
    """Response schema for job creation."""

    job_id: UUID = Field(..., description="Job identifier")

    model_config = {"json_schema_extra": {"example": {"job_id": "94939cbe-e692-4423-b9d3-5f6d93823be3"}}}
