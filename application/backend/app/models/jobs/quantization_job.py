# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from loguru import logger
from pydantic import Field

from app.core.jobs.models import JobParams, JobType, ProjectJob


class QuantizationJobParams(JobParams):
    job_id: UUID
    project_id: UUID
    model_id: UUID  # Source model revision to quantize
    model_variant_id: UUID = Field(default_factory=uuid4)
    max_calibration_subset_size: int = Field(default=100, description="Max samples for calibration")
    max_drop: float | None = Field(default=None, description="Max accuracy drop for accuracy-aware quantization")


class QuantizationJob(ProjectJob[QuantizationJobParams]):
    job_type: Literal[JobType.QUANTIZE] = JobType.QUANTIZE  # pyrefly: ignore[bad-override]
    log_dir: Path
    data_dir: Path
    params: QuantizationJobParams

    def on_complete(self) -> None:
        """Copy the quantization log and clean up the getitune workspace upon job completion."""
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
                / "variants"
                / str(self.params.model_variant_id)
                / "quantization.log"
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, new_path)

        # Remove the getitune quantization workspace directory (parent of the timestamped
        # subdir) so it never lingers on disk regardless of whether the job succeeded or failed.
        workspace_dir = self.data_dir / f"getitune-quantize-workspace-{self.params.model_id}"
        try:
            shutil.rmtree(workspace_dir)
            logger.info(f"Cleaned up getitune quantization workspace directory at {workspace_dir}")
        except FileNotFoundError:
            # Directory was never created or already removed; treat as a no-op.
            pass
        except Exception as cleanup_exc:
            logger.error(
                f"Failed to clean up getitune quantization workspace directory at {workspace_dir}: {cleanup_exc}"
            )
