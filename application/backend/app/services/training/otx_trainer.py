# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from datumaro.experimental import Dataset
from datumaro.experimental.fields import Subset
from loguru import logger
from sqlalchemy.orm import Session

from app.core.run import ExecutionContext
from app.models import DatasetItemAnnotationStatus
from app.schemas.model import TrainingStatus
from app.schemas.project import TaskBase
from app.services import BaseWeightsService, DatasetService, ModelRevisionMetadata, ModelService

from .base import Trainer, step
from .models import TrainingParams
from .subset_assignment import SplitRatios, SubsetAssigner, SubsetService

MODEL_WEIGHTS_PATH = "model_weights_path"


@dataclass(frozen=True)
class DatasetInfo:
    training: Dataset
    validation: Dataset
    testing: Dataset
    revision_id: UUID


class OTXTrainer(Trainer):
    """OTX-specific trainer implementation."""

    def __init__(
        self,
        data_dir: Path,
        base_weights_service: BaseWeightsService,
        subset_service: SubsetService,
        dataset_service: DatasetService,
        model_service: ModelService,
        subset_assigner: SubsetAssigner,
        db_session_factory: Callable[[], AbstractContextManager[Session]],
    ):
        super().__init__()
        self._data_dir = data_dir
        self._base_weights_service = base_weights_service
        self._subset_service = subset_service
        self._dataset_service = dataset_service
        self._model_service = model_service
        self._subset_assigner = subset_assigner
        self._db_session_factory = db_session_factory

    @step("Prepare Model Weights")
    def prepare_weights(self, training_params: TrainingParams) -> Path:
        """
        Prepare weights for training based on training parameters.

        If a parent model revision ID is provided, it fetches the weights from the parent model.
        Otherwise, it retrieves the base weights for the specified model architecture.
        """
        parent_model_revision_id = training_params.parent_model_revision_id
        task = training_params.task
        model_architecture_id = training_params.model_architecture_id
        project_id = training_params.project_id
        if parent_model_revision_id is None:
            return self._base_weights_service.get_local_weights_path(
                task=task.task_type, model_manifest_id=model_architecture_id
            )

        if project_id is None:
            raise ValueError("Project ID must be provided for parent model weights preparation")

        weights_path = self.__build_model_weights_path(self._data_dir, project_id, parent_model_revision_id)
        if not weights_path.exists():
            raise FileNotFoundError(f"Parent model weights not found at {weights_path}")

        return weights_path

    @step("Assign Dataset Subsets")
    def assign_subsets(self, project_id: UUID) -> None:
        """Assigning subsets to all unassigned dataset items in the project dataset."""
        with self._db_session_factory() as db:
            self._subset_service.set_db_session(db)
            self.report_progress("Retrieving unassigned items")
            unassigned_items = self._subset_service.get_unassigned_items_with_labels(project_id)

            if not unassigned_items:
                self.report_progress("No unassigned items found")
                return

            self.report_progress(f"Found {len(unassigned_items)} unassigned items")

            # Get current distribution
            current_distribution = self._subset_service.get_subset_distribution(project_id)
            logger.info("Current subset distribution: {}", current_distribution)

            # Compute adjusted ratios
            # TODO: Infer target ratios from training params
            target_ratios = SplitRatios(train=0.7, val=0.15, test=0.15)
            adjusted_ratios = current_distribution.compute_adjusted_ratios(target_ratios, len(unassigned_items))

            self.report_progress("Computing optimal subset assignments")
            assignments = self._subset_assigner.assign(unassigned_items, adjusted_ratios)

            # Persist assignments
            self.report_progress("Persisting subset assignments")
            self._subset_service.update_subset_assignments(project_id, assignments)

        self.report_progress(f"Successfully assigned {len(assignments)} items to subsets")

    @step("Create Training Dataset")
    def create_training_dataset(self, project_id: UUID, task: TaskBase) -> DatasetInfo:
        """Create datasets for training, validation, and testing."""
        with self._db_session_factory() as db:
            self._dataset_service.set_db_session(db)
            dm_dataset = self._dataset_service.get_dm_dataset(project_id, task, DatasetItemAnnotationStatus.REVIEWED)
            return DatasetInfo(
                training=dm_dataset.filter_by_subset(Subset.TRAINING),
                validation=dm_dataset.filter_by_subset(Subset.VALIDATION),
                testing=dm_dataset.filter_by_subset(Subset.TESTING),
                revision_id=self._dataset_service.save_revision(project_id, dm_dataset),
            )

    @step("Prepare Model Metadata")
    def prepare_model(self, training_params: TrainingParams, dataset_revision_id: UUID) -> None:
        if training_params.project_id is None:
            raise ValueError("Project ID must be provided for model preparation")
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            self._model_service.create_revision(
                ModelRevisionMetadata(
                    model_id=training_params.model_id,
                    project_id=training_params.project_id,
                    architecture_id=training_params.model_architecture_id,
                    parent_revision_id=training_params.parent_model_revision_id,
                    training_configuration=None,  # TODO: to be set when config is added
                    dataset_revision_id=dataset_revision_id,
                    training_status=TrainingStatus.NOT_STARTED,
                )
            )

    @step("Train Model with OTX")
    def train_model(self, training_params: TrainingParams) -> None:
        """Execute OTX model training."""
        # Simulate training with progress reporting
        job_id = training_params.job_id
        step_count = 20
        for i in range(step_count):
            time.sleep(1)
            logger.info("Training step {}/{} for job {}", i + 1, step_count, job_id)
            self.report_progress("Model training is in progress", 5.0 * (i + 1))
            self.heartbeat()

    def run(self, ctx: ExecutionContext) -> None:
        self._ctx = ctx
        training_params = self._get_training_params(ctx)
        project_id = training_params.project_id
        if project_id is None:
            raise ValueError("Project ID must be provided in training parameters")
        task = training_params.task

        self.prepare_weights(training_params)
        self.assign_subsets(project_id)
        dataset_info = self.create_training_dataset(project_id, task)
        self.prepare_model(training_params, dataset_info.revision_id)
        self.train_model(training_params)

    @staticmethod
    def __build_model_weights_path(data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return data_dir / "projects" / str(project_id) / "models" / str(model_id) / "model.pth"
