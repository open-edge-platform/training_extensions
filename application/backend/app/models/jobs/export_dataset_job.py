# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from uuid import UUID

from app.core.jobs.models import JobParams, JobType, ProjectJob
from app.models import DatasetFormat, DatasetItemSubset
from app.models.project import Task


class ExportDatasetJobParams(JobParams):
    dataset_id: UUID | None = None
    project_id: UUID
    task: Task
    export_format: DatasetFormat
    labels: list[str] | None = None
    subsets: list[DatasetItemSubset] | None = None
    include_unannotated: bool = False


class ExportDatasetJob(ProjectJob[ExportDatasetJobParams]):
    job_type: JobType = JobType.EXPORT_DATASET  # pyrefly: ignore[bad-override]
    params: ExportDatasetJobParams
