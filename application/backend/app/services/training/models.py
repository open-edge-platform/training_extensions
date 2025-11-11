# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

from loguru import logger

from app.core.jobs import Job, JobParams, JobType
from app.schemas.project import TaskBase


class TrainingParams(JobParams):
    job_id: UUID | None = None
    project_id: UUID | None = None
    model_architecture_id: str
    parent_model_revision_id: UUID | None = None
    task: TaskBase
    model_id: UUID = uuid4()  # Reserve the ID for the model to be created for this training job


class ProjectJob(Job):
    project_id: UUID


class TrainingJob(ProjectJob):
    job_type: Literal[JobType.TRAIN] = JobType.TRAIN
    log_dir: Path
    data_dir: Path
    params: TrainingParams

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
