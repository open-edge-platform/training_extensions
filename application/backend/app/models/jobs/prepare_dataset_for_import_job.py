# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID

from app.core.jobs.models import Job, JobParams, JobType


class PrepareDatasetForImportJobParams(JobParams):
    staged_dataset_id: UUID


class PrepareDatasetForImportJob(Job[PrepareDatasetForImportJobParams]):
    job_type: JobType = JobType.PREPARE_DATASET_FOR_IMPORT  # pyrefly: ignore[bad-override]
    params: PrepareDatasetForImportJobParams
