# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from loguru import logger
from pydantic import Field, computed_field

from app.core.jobs.models import JobParams, JobType, ProjectJob
from app.models.project import Task
from app.models.system import DeviceInfo


class TrainingJobParams(JobParams):
    job_id: UUID
    project_id: UUID
    model_architecture_id: str
    model_architecture_name: str
    parent_model_revision_id: UUID | None = None
    task: Task
    model_id: UUID = Field(default_factory=uuid4)
    dataset_revision_id: UUID | None = None
    device: DeviceInfo

    @computed_field  # type: ignore[prop-decorator]
    @property
    def has_model_revision(self) -> bool:
        return self.parent_model_revision_id is not None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        """User-friendly model name derived from architecture name and model ID."""
        return f"{self.model_architecture_name} ({str(self.model_id).split('-')[0]})"


class TrainingJob(ProjectJob[TrainingJobParams]):
    job_type: Literal[JobType.TRAIN] = JobType.TRAIN  # pyrefly: ignore[bad-override]
    log_dir: Path
    data_dir: Path
    params: TrainingJobParams

    def on_complete(self) -> None:
        """Copy the training log and clean up the getitune workspace upon job completion."""
        log_path = self.log_dir / self.log_file
        if not log_path.exists():
            logger.warning(f"Log file {log_path} does not exist")
        else:
            new_path = (
                self.data_dir
                / "projects"
                / str(self.project_id)
                / "models"
                / str(self.params.model_id)
                / "training.log"
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, new_path)

        # Remove the getitune workspace directory (parent of the timestamped subdir)
        # so it never lingers on disk regardless of whether the job succeeded or failed.
        workspace_dir = self.data_dir / f"getitune-workspace-{self.params.model_id}"
        try:
            shutil.rmtree(workspace_dir)
            logger.info(f"Cleaned up getitune workspace directory at {workspace_dir}")
        except FileNotFoundError:
            # Directory was never created or already removed; treat as a no-op.
            pass
        except Exception as cleanup_exc:
            logger.error(f"Failed to clean up getitune workspace directory at {workspace_dir}: {cleanup_exc}")
