# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.core.jobs.models import Job, JobParams, JobType
from app.models import DatasetItemSubset
from app.models.task import TaskType


class ImportDatasetAsNewProjectJobParams(JobParams):
    staged_dataset_id: UUID
    project_name: str
    task_type: TaskType
    labels: list[str] | None
    subsets: list[DatasetItemSubset] | None
    exclusive_labels: bool = False
    include_unannotated: bool = True
    project_id: UUID | None = None


class ImportDatasetAsNewProjectJob(Job[ImportDatasetAsNewProjectJobParams]):
    job_type: JobType = JobType.IMPORT_DATASET_AS_NEW_PROJECT  # pyrefly: ignore[bad-override]
    params: ImportDatasetAsNewProjectJobParams
