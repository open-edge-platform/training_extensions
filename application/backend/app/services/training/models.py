# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Literal
from uuid import UUID

from app.core.jobs import Job, JobParams, JobType
from app.core.models import TaskType


class TrainingParams(JobParams):
    job_id: UUID | None = None
    model_architecture_id: str
    parent_model_revision_id: UUID | None = None
    task_type: TaskType


class ProjectJob(Job):
    project_id: UUID


class TrainingJob(ProjectJob):
    job_type: Literal[JobType.TRAIN] = JobType.TRAIN
    params: TrainingParams

    def model_post_init(self, _: Any) -> None:
        self.params.job_id = self.id
