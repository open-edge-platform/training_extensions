# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel, Field

from app.api.schemas.dataset import DatasetFilters
from app.core.jobs.models import JobType

from .base import BaseDatasetRequest, BaseJobRequest


class ExportDatasetParams(BaseModel):
    export_format: str = Field(..., description="The format of the exported dataset (e.g. coco)")
    filters: DatasetFilters = Field(default_factory=DatasetFilters, description="Dataset filters to use for export")


class ExportDatasetRequest(BaseJobRequest, BaseDatasetRequest):
    job_type: Literal[JobType.EXPORT_DATASET]

    parameters: ExportDatasetParams = Field(..., description="The configuration for exporting dataset")


class StageDatasetParams(BaseModel):
    filters: DatasetFilters = Field(default_factory=DatasetFilters, description="Dataset filters to use for staging")


class StageDatasetRequest(BaseJobRequest, BaseDatasetRequest):
    job_type: Literal[JobType.STAGE_DATASET]

    parameters: StageDatasetParams = Field(
        default_factory=StageDatasetParams, description="The configuration for staging dataset"
    )
