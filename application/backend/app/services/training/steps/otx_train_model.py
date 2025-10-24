# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time

from app.core.run import ExecutionContext
from app.services.training.base import PipelineContext, TrainingParams, TrainingStep

logger = logging.getLogger(__name__)


class OTXTrainModelStep(TrainingStep):
    def execute(self, ctx: ExecutionContext, params: TrainingParams, _pipeline_ctx: PipelineContext) -> None:
        job_id = params.job_id
        # Simulate training with progress reporting
        step_count = 20
        for i in range(step_count):
            time.sleep(1)
            logger.info("Training step %d/%d for job %s", i + 1, step_count, job_id)
            ctx.report_progress("Model training is in progress", 5.0 * (i + 1))
            ctx.heartbeat()

    def get_name(self) -> str:
        return "Train Model with OTX"
