# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.core.jobs.models import JobParams, JobType, ProjectJob
from app.models.task import Task


class ImportDatasetToProjectJobParams(JobParams):
    staged_dataset_id: UUID
    project_id: UUID
    task: Task
    labels_mapping: dict[str, str | None] | None = None


class ImportDatasetToProjectJob(ProjectJob[ImportDatasetToProjectJobParams]):
    job_type: JobType = JobType.IMPORT_DATASET_TO_PROJECT  # pyrefly: ignore[bad-override]
    params: ImportDatasetToProjectJobParams
