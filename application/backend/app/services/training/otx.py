# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from collections.abc import Callable
from pathlib import Path

from app.core.run import ExecutionContext

from .base import Trainer, TrainingParams

logger = logging.getLogger(__name__)


class OTXTrainer(Trainer):
    def _prepare_weights(
        self, data_dir: Path, training_params: TrainingParams, report_fn: Callable[[str], None]
    ) -> Path:
        """
        Prepare weights for training based on training parameters.

        If a parent model revision ID is provided, it fetches the weights from the parent model. Otherwise, it retrieves
        the base weights for the specified model architecture.

        Args:
            data_dir (Path): The base directory for data storage.
            training_params (TrainingParams): The parameters for the training job.
            report_fn (Callable): Function to report progress messages.
        Returns:
            Path: The path to the model weights file.
        Raises:
            ValueError: If project ID is not provided when parent model revision ID is specified.
        """
        report_fn("Preparing weights for training")
        if training_params.parent_model_revision_id is None:
            weights_path = self._weights_service.get_local_weights_path(
                task=training_params.task_type, model_manifest_id=training_params.model_architecture_id
            )
            report_fn("Base weights preparation completed")
            return weights_path
        if training_params.project_id is None:
            raise ValueError("Project ID must be provided for parent model weights preparation")
        weights_path = self._build_model_weights_path(
            data_dir, training_params.project_id, training_params.parent_model_revision_id
        )
        if not weights_path.exists():
            raise FileNotFoundError(f"Parent model weights not found at {weights_path}")
        report_fn("Parent model weights preparation completed")
        return weights_path

    def _train(self):
        raise NotImplementedError

    def _evaluate(self):
        raise NotImplementedError

    def run(self, ctx: ExecutionContext) -> None:
        training_params = self._get_training_params(ctx)
        logger.info("Training job started with params: %s", training_params)
        self._prepare_weights(ctx.data_dir, training_params, ctx.report_progress)
        ctx.report_progress("Started model training")
        job_id = training_params.job_id
        # Simulate training with progress reporting
        step_count = 20
        for i in range(step_count):
            time.sleep(1)
            logger.info("Training step %d/%d for job %s", i + 1, step_count, job_id)
            ctx.report_progress("Model training is in progress", 5.0 * (i + 1))
            ctx.heartbeat()

        logger.info("Completed training: %s", job_id)
