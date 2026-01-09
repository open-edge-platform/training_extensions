# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from typing import Any, Generic, Literal
from uuid import UUID, uuid4

from loguru import logger
from pydantic import Field

from app.core.jobs.models import Job, JobParams, JobParamsT, JobType
from app.models import Task
from app.models.system import DeviceInfo


class TrainingJobParams(JobParams):
    job_id: UUID | None = None
    project_id: UUID | None = None
    model_architecture_id: str
    parent_model_revision_id: UUID | None = None
    task: Task
    model_id: UUID = Field(default_factory=uuid4)
    device: DeviceInfo


class ProjectJob(Job, Generic[JobParamsT]):
    project_id: UUID
    params: JobParamsT


class TrainingJob(ProjectJob[TrainingJobParams]):
    job_type: Literal[JobType.TRAIN] = JobType.TRAIN  # pyrefly: ignore[bad-override]
    log_dir: Path
    data_dir: Path
    params: TrainingJobParams

    def model_post_init(self, _: Any) -> None:
        self.params.job_id = self.id
        self.params.project_id = self.project_id

    def on_finish(self) -> None:
        """Copy the training log to the model's directory upon job completion."""
        log_path = self.log_dir / self.log_file
        if not log_path.exists():
            logger.warning(f"Log file {log_path} does not exist")
            return
        new_path = (
            self.data_dir / "projects" / str(self.project_id) / "models" / str(self.params.model_id) / "training.log"
        )
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(log_path, new_path)
