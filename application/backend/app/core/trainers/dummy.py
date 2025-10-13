# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time

from .base import Trainer, TrainerContext

logger = logging.getLogger(__name__)


class DummyTrainer(Trainer):
    def run(self, ctx: TrainerContext) -> None:
        job_id = ctx.task.id
        logger.info("Training started. Job ID: %s", job_id)
        # Simulate training with progress reporting
        step_count = 10
        for i in range(step_count):
            time.sleep(1)
            logger.info("Training step %d/%d for job %s", i + 1, step_count, job_id)
            ctx.report_progress(10.0 * (i + 1))
            ctx.heartbeat()

        logger.info("Completed training: %s", job_id)
