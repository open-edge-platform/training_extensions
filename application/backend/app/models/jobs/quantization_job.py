# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from loguru import logger
from pydantic import Field

from app.core.jobs.models import JobParams, JobType, ProjectJob
from app.models.system import DeviceInfo


class QuantizationJobParams(JobParams):
    job_id: UUID
    project_id: UUID
    model_id: UUID  # Source model revision to quantize
    model_variant_id: UUID = Field(default_factory=uuid4)
    device: DeviceInfo
    max_calibration_subset_size: int = Field(default=100, description="Max samples for calibration")
    max_drop: float | None = Field(default=None, description="Max accuracy drop for accuracy-aware quantization")


class QuantizationJob(ProjectJob[QuantizationJobParams]):
    job_type: Literal[JobType.QUANTIZE] = JobType.QUANTIZE  # pyrefly: ignore[bad-override]
    log_dir: Path
    data_dir: Path
    params: QuantizationJobParams

    def on_complete(self) -> None:
        """Copy the quantization log to the model's directory upon job completion."""
        log_path = self.log_dir / self.log_file
        if not log_path.exists():
            logger.warning(f"Log file {log_path} does not exist")
            return
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
