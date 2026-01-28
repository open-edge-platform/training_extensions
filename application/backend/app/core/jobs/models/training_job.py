# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from loguru import logger
from pydantic import Field

from app.models import Task
from app.models.system import DeviceInfo

from .job import JobParams, JobType, ProjectJob


class TrainingJobParams(JobParams):
    job_id: UUID
    project_id: UUID
    model_architecture_id: str
    parent_model_revision_id: UUID | None = None
    task: Task
    model_id: UUID = Field(default_factory=uuid4)
    device: DeviceInfo


class TrainingJob(ProjectJob[TrainingJobParams]):
    job_type: Literal[JobType.TRAIN] = JobType.TRAIN  # pyrefly: ignore[bad-override]
    log_dir: Path
    data_dir: Path
    params: TrainingJobParams

    def on_complete(self) -> None:
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
