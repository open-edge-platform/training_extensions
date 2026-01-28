# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from pydantic import BaseModel, Field


class BaseJobRequest(BaseModel):
    project_id: UUID = Field(..., description="ID of the project associated with the job")


class BaseDatasetRequest(BaseModel):
    staged_dataset_id: UUID = Field(..., description="ID of the staged dataset associated with the job")
