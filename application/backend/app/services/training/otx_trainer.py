# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import UUID

from loguru import logger
from sqlalchemy.orm import Session

from app.core.run import ExecutionContext
from app.services.base_weights_service import BaseWeightsService

from .base import Trainer, step
from .subset_assignment import SplitRatios, SubsetAssigner, SubsetService

MODEL_WEIGHTS_PATH = "model_weights_path"


class OTXTrainer(Trainer):
    """OTX-specific trainer implementation."""

    def __init__(
        self,
        data_dir: Path,
        base_weights_service: BaseWeightsService,
        subset_service: SubsetService,
        subset_assigner: SubsetAssigner,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ):
        super().__init__()
        self._data_dir = data_dir
        self._base_weights_service = base_weights_service
        self._subset_service = subset_service
        self._subset_assigner = subset_assigner
        self._db_session_factory = db_session_factory

    @step("Prepare Model Weights")
    def prepare_weights(self) -> Path:
        """
        Prepare weights for training based on training parameters.

        If a parent model revision ID is provided, it fetches the weights from the parent model.
        Otherwise, it retrieves the base weights for the specified model architecture.
        """
        if self._training_params is None:
            raise ValueError("Training parameters not set")
        parent_model_revision_id = self._training_params.parent_model_revision_id
        task_type = self._training_params.task_type
        model_architecture_id = self._training_params.model_architecture_id
        project_id = self._training_params.project_id
        if parent_model_revision_id is None:
            return self._base_weights_service.get_local_weights_path(
                task=task_type, model_manifest_id=model_architecture_id
            )

        if project_id is None:
            raise ValueError("Project ID must be provided for parent model weights preparation")

        weights_path = self.__build_model_weights_path(self._data_dir, project_id, parent_model_revision_id)
        if not weights_path.exists():
            raise FileNotFoundError(f"Parent model weights not found at {weights_path}")

        return weights_path

    @step("Assign Dataset Subsets")
    def assign_subsets(self) -> None:
        """Assigning subsets to all unassigned dataset items in the project dataset."""
        if self._training_params is None:
            raise ValueError("Training parameters not set")
        project_id = self._training_params.project_id
        self.report_progress("Retrieving unassigned items")
        if project_id is None:
            raise ValueError("Project ID must be provided for subset assignment")

        with self._db_session_factory() as db:
            unassigned_items = self._subset_service.get_unassigned_items_with_labels(project_id, db)

            if not unassigned_items:
                self.report_progress("No unassigned items found")
                return

            self.report_progress(f"Found {len(unassigned_items)} unassigned items")

            # Get current distribution
            current_distribution = self._subset_service.get_subset_distribution(project_id, db)
            logger.info("Current subset distribution: {}", current_distribution)

            # Compute adjusted ratios
            # TODO: Infer target ratios from training params
            target_ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
            adjusted_ratios = current_distribution.compute_adjusted_ratios(target_ratios, len(unassigned_items))

            self.report_progress("Computing optimal subset assignments")
            assignments = self._subset_assigner.assign(unassigned_items, adjusted_ratios)

            # Persist assignments
            self.report_progress("Persisting subset assignments")
            self._subset_service.update_subset_assignments(project_id, assignments, db)

        self.report_progress(f"Successfully assigned {len(assignments)} items to subsets")

    @step("Train Model with OTX")
    def train_model(self) -> None:
        """Execute OTX model training."""
        if self._training_params is None:
            raise ValueError("Training parameters not set")
        # Simulate training with progress reporting
        job_id = self._training_params.job_id
        step_count = 20
        for i in range(step_count):
            time.sleep(1)
            logger.info("Training step {}/{} for job {}", i + 1, step_count, job_id)
            self.report_progress("Model training is in progress", 5.0 * (i + 1))
            self.heartbeat()

    def run(self, ctx: ExecutionContext) -> None:
        self._ctx = ctx
        self._training_params = self._get_training_params(ctx)

        self.prepare_weights()
        self.assign_subsets()
        self.train_model()

    @staticmethod
    def __build_model_weights_path(data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return data_dir / "projects" / str(project_id) / "models" / str(model_id) / "model.pth"
