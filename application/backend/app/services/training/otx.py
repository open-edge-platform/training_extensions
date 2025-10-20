# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from pathlib import Path

from app.core.run import ExecutionContext

from .base import Trainer

logger = logging.getLogger(__name__)


class OTXTrainer(Trainer):
    def __prepare_weights(self) -> Path | None:
        raise NotImplementedError

    def __train(self):
        raise NotImplementedError

    def __evaluate(self):
        raise NotImplementedError

    def run(self, ctx: ExecutionContext) -> None:
        training_params = self.get_training_params(ctx)
        logger.info("Training job started with params: %s", training_params)
        ctx.report_progress("Training job started")
        job_id = training_params.job_id
        # Simulate training with progress reporting
        step_count = 20
        for i in range(step_count):
            time.sleep(1)
            logger.info("Training step %d/%d for job %s", i + 1, step_count, job_id)
            ctx.report_progress("Model training is in progress", 5.0 * (i + 1))
            ctx.heartbeat()

        logger.info("Completed training: %s", job_id)
