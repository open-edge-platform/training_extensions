# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast
from uuid import UUID

import polars as pl
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.schema import EvaluationDB, MetricScoreDB, ModelRevisionDB
from app.models import EvaluationResult, ModelRevision, TrainingStatus
from app.models.model_revision import ModelFormat, ModelPrecision
from app.models.training_configuration.configuration import TrainingConfiguration
from app.repositories import EvaluationRepository, LabelRepository, ModelRevisionRepository
from app.services.dataset_revision_service import DatasetRevisionService
from app.supported_models import SupportedModels

from .base import BaseSessionManagedService, ResourceInUseError, ResourceNotFoundError, ResourceType
from .parent_process_guard import parent_process_only

# Mapping of CSV column keys to display names for series metrics.
# Keys not included here will be ignored.
KEY_MAPPING = {
    "epoch": "Epoch",
    "lr-SGD": "Learning rate (SGD)",
    "lr-SGD-1": "Learning rate (SGD-1)",
    "lr-SGD-1-momentum": "Learning rate momentum (SGD-1)",
    "lr-SGD-momentum": "Learning rate momentum (SGD)",
    "step": "Step",
    "train/data_time": "Training data time",
    "train/iter_time": "Training iteration time",
    "train/loss": "Training loss",
    "train/loss_bbox": "Training loss bbox",
    "train/loss_centerness": "Training loss centerness",
    "train/loss_cls": "Training loss cls",
    "train/loss_mask": "Training loss mask",
    "train/loss_obj": "Training loss obj",
    "train/total_loss": "Training total loss",
    "val/accuracy": "Validation accuracy",
    "val/classes": "Validation classes",
    "val/f1-score": "Validation F1 score",
    "val/map": "Validation mAP",
    "val/map_50": "Validation mAP@50",
    "val/map_75": "Validation mAP@75",
    "val/map_large": "Validation mAP large",
    "val/map_medium": "Validation mAP medium",
    "val/map_per_class": "Validation mAP per class",
    "val/map_small": "Validation mAP small",
    "val/mar_1": "Validation mAR@1",
    "val/mar_10": "Validation mAR@10",
    "val/mar_100": "Validation mAR@100",
    "val/mar_100_per_class": "Validation mAR@100 per class",
    "val/mar_large": "Validation mAR large",
    "val/mar_medium": "Validation mAR medium",
    "val/mar_small": "Validation mAR small",
    "validation/data_time": "Validation data time",
    "validation/iter_time": "Validation iteration time",
}

# Columns that should be excluded from metrics parsing
# (used as axis values or are not actual metrics)
BLACKLISTED_KEYS = {"epoch", "step"}


@dataclass(frozen=True)
class ModelRevisionMetadata:
    model_id: UUID
    project_id: UUID
    architecture_id: str
    parent_revision_id: UUID | None
    dataset_revision_id: UUID | None
    training_status: TrainingStatus
    training_configuration: TrainingConfiguration | None = None


class ModelService(BaseSessionManagedService):
    """Service to register and activate models"""

    def __init__(self, data_dir: Path, db_session: Session | None = None) -> None:
        super().__init__(db_session)
        self._projects_dir = data_dir / "projects"

    def get_model(self, project_id: UUID, model_id: UUID) -> ModelRevision:
        """
        Get a model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve.

        Returns:
            ModelRevision: The model revision object containing the model's information.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if not model_rev_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        return ModelRevision.model_validate(model_rev_db)

    def get_model_variants(self, project_id: UUID, model_id: UUID) -> list[dict]:
        """
        Get all variants and their information of a model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve variants for.

        Returns:
            list[dict]: A list of the models variants.
        """
        model_variants = []
        for format in ModelFormat:
            exists, paths = self.get_model_binary_files(project_id=project_id, model_id=model_id, format=format)
            if exists:
                model_size = sum(path.stat().st_size for path in paths)
                model_info = {
                    "format": format.value,
                    "precision": ModelPrecision.FP16 if format != ModelFormat.PYTORCH else ModelPrecision.FP32,
                    "weights_size": model_size,
                }
                model_variants.append(model_info)

        return model_variants

    def get_model_size_in_bytes(self, project_id: UUID, model_id: UUID) -> int:
        """
        Get the total size of the model and all its files in bytes.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve size for.
        Returns:
            int: Total size of the model and all its files in bytes.
        """

        @lru_cache
        def _cached_model_size_in_bytes(_model_path: Path) -> int:
            return sum(f.stat().st_size for f in _model_path.glob("**/*") if f.is_file())

        model_revision = self.get_model(project_id=project_id, model_id=model_id)
        if model_revision.files_deleted:
            return 0

        model_path = self._projects_dir / str(project_id) / "models" / str(model_id)
        return _cached_model_size_in_bytes(_model_path=model_path)

    def rename_model(self, project_id: UUID, model_id: UUID, model_metadata: dict[str, str]) -> ModelRevision:
        """
        Rename a model revision.

        Args:
            project_id (UUID): The unique identifier of the project whose models to get.
            model_id (UUID): The unique identifier of the model to retrieve.
            model_metadata: Dict containing updated model revision name

        Returns:
            ModelRevision: The model revision object containing the model's updated information.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = model_rev_repo.get_by_id(str(model_id))
        if not model_rev_db:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        new_name = model_metadata.get("name")
        if new_name is not None:
            model_rev_db.name = new_name
            model_rev_repo.update(model_rev_db)
        return ModelRevision.model_validate(model_rev_db)

    @parent_process_only
    def delete_model(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete a model.

        Deletes a model revision from the database and deletes the folder from the filesystem
        associated with this model.

        Args:
            project_id (UUID): The unique identifier of the project whose models to delete.
            model_id (UUID): The unique identifier of the model to delete.

        Returns:
            None

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
            ResourceInUseError: If the model cannot be deleted due to integrity constraints
                (e.g., the model is referenced by other entities).
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_to_delete = model_rev_repo.get_by_id(str(model_id))

        try:
            deleted = model_rev_repo.delete(str(model_id))
            if not deleted:
                raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))
        except IntegrityError:
            raise ResourceInUseError(ResourceType.MODEL, str(model_id))

        path = self._projects_dir / str(project_id) / "models" / str(model_id)
        if path.exists():
            shutil.rmtree(path)
            logger.info("Deleted model files at '{}'", path)

        if model_to_delete is not None:
            self._delete_training_dataset_revision_files(deleted_model=model_to_delete)

    @parent_process_only
    def delete_model_files(self, project_id: UUID, model_id: UUID) -> None:
        """
        Delete only the model files from disk, keeping the model revision record in the database and setting its
        files_deleted flag to True.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
        """
        # Mark as deleted in the database
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_rev_db = cast(ModelRevisionDB, model_rev_repo.get_by_id(str(model_id)))
        model_rev_db.files_deleted = True
        model_rev_repo.update(model_rev_db)

        path = self._projects_dir / str(project_id) / "models" / str(model_id)
        if path.exists():
            shutil.rmtree(path)
            logger.info("Deleted model files at '{}'", path)

        self._delete_training_dataset_revision_files(deleted_model=model_rev_db)

    def _delete_training_dataset_revision_files(self, deleted_model: ModelRevisionDB) -> None:
        """
        Delete training dataset revision files if all model revisions linked to it have no files

        Also deletes the dataset revision files if no model is linked to it.

        Args:
            deleted_model: Recently deleted model or model which has its files deleted
        """
        model_rev_repo = ModelRevisionRepository(project_id=deleted_model.project_id, db=self.db_session)
        if (
            deleted_model is not None
            and deleted_model.training_dataset_id is not None
            and all(
                model_db.files_deleted
                for model_db in model_rev_repo.list_all(training_dataset_id=deleted_model.training_dataset_id)
            )
        ):
            DatasetRevisionService(
                data_dir=self._projects_dir.parent, db_session=self._db_session
            ).delete_dataset_revision_files(
                project_id=UUID(deleted_model.project_id), revision_id=UUID(deleted_model.training_dataset_id)
            )

    def list_models(self, project_id: UUID, dataset_revision_id: UUID | None = None) -> list[ModelRevision]:
        """
        Get information about all available model revisions in a project.

        Retrieves a list of all model revisions that belong to the specified project.
        Each model revision is converted to a schema object containing the model's
        metadata and configuration information.

        Args:
            project_id (UUID): The unique identifier of the project whose models to list.
            dataset_revision_id (UUID | None, optional): The unique identifier of the dataset revision to filter on.

        Returns:
            list[ModelRevision]: A list of model revision objects representing all model
                revisions in the project, optionally filtered on dataset revision.
                Returns an empty list if the project has no models.
        """
        model_rev_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        training_dataset_id = str(dataset_revision_id) if dataset_revision_id is not None else None
        return [
            ModelRevision.model_validate(model_rev_db)
            for model_rev_db in model_rev_repo.list_all(training_dataset_id=training_dataset_id)
        ]

    def create_revision(self, metadata: ModelRevisionMetadata) -> None:
        """
        Create and persist a new model revision for the given project metadata.

        Reads the project's label definitions, serializes them into a dict format,
        combines them with the provided metadata into a new model revision record,
        and saves it to the database.

        Args:
            metadata (ModelRevisionMetadata): Metadata used to create the new model revision
                including project id, architecture, optional parent revision id,
                dataset revision id, training status and optional training
                configuration.
        """
        project_id = str(metadata.project_id)
        label_repo = LabelRepository(project_id=project_id, db=self.db_session)
        labels_schema_rev = {"labels": [{"name": label.name, "id": label.id} for label in label_repo.list_all()]}
        arch_name = SupportedModels.get_model_manifest_by_id(metadata.architecture_id).name

        model_revision_repo = ModelRevisionRepository(project_id=project_id, db=self.db_session)
        model_revision_repo.save(
            ModelRevisionDB(
                id=str(metadata.model_id),
                name=f"{arch_name} ({str(metadata.model_id).split('-')[0]})",
                project_id=str(metadata.project_id),
                architecture=metadata.architecture_id,
                parent_revision=str(metadata.parent_revision_id) if metadata.parent_revision_id else None,
                training_status=metadata.training_status,
                training_configuration=metadata.training_configuration.model_dump()
                if metadata.training_configuration
                else {},
                training_dataset_id=str(metadata.dataset_revision_id),
                label_schema_revision=labels_schema_rev,
            )
        )

    def update_revision_status(self, project_id: UUID, model_id: UUID, training_status: TrainingStatus) -> None:
        """
        Updates the training status of a model revision for the given project.

        Args:
            project_id (UUID): Identifier of the project that owns the model revision.
            model_id (UUID): Identifier of the model revision to update.
            training_status (TrainingStatus): New training status to set for the model revision.
        """
        model_revision_repo = ModelRevisionRepository(project_id=str(project_id), db=self.db_session)
        model_revision_repo.update_training_status(obj_id=str(model_id), training_status=training_status)

    def get_model_binary_files(
        self, project_id: UUID, model_id: UUID, format: ModelFormat
    ) -> tuple[bool, tuple[Path, ...]]:
        """
        Get the paths to the model binary files.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.
            format (ModelFormat): The format of the model files to retrieve.

        Returns:
            tuple[bool, tuple[Path, ...]]: A tuple where the first element indicates if the files exist,
                and the second element is a tuple of Paths to the model files.

        Raises:
            ResourceNotFoundError: If the model has been marked as deleted.
            FileNotFoundError: If the model directory does not exist.
        """
        model_revision = self.get_model(project_id=project_id, model_id=model_id)
        if model_revision.files_deleted:
            return False, ()

        model_dir = self._projects_dir / str(project_id) / "models" / str(model_id)
        xml_file = model_dir / "model.xml"
        bin_file = model_dir / "model.bin"
        onnx_file = model_dir / "model.onnx"
        ckpt_file = model_dir / "model.ckpt"

        if format == ModelFormat.OPENVINO and xml_file.exists() and bin_file.exists():
            return True, (xml_file, bin_file)
        if format == ModelFormat.ONNX and onnx_file.exists():
            return True, (onnx_file,)
        if format == ModelFormat.PYTORCH and ckpt_file.exists():
            return True, (ckpt_file,)

        return False, ()

    def save_evaluation_result(self, result: EvaluationResult) -> None:
        evaluation_db = EvaluationDB(
            model_revision_id=str(result.model_revision_id),
            dataset_revision_id=str(result.dataset_revision_id),
            subset=result.subset,
        )
        metrics_db = [MetricScoreDB(metric=item[0], score=item[1]) for item in result.metrics.items()]
        evaluation_db.metric_scores = metrics_db

        evaluation_repo = EvaluationRepository(self.db_session)
        evaluation_repo.save(evaluation_db)

    def get_model_training_metrics(
        self,
        project_id: UUID,
        model_id: UUID,
    ) -> list[dict]:
        """
        Get training metrics from the metrics.csv file for a model.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.

        Returns:
            list[dict]: A list of metric dictionaries

        Raises:
            ResourceNotFoundError: If the model or metrics file is not found.
        """
        # Verify the model exists
        model_revision = self.get_model(project_id=project_id, model_id=model_id)
        if model_revision.files_deleted:
            raise ResourceNotFoundError(ResourceType.MODEL, str(model_id))

        # training + validation metrics are version_0, testing metrics are version_1
        metrics_file = (
            self._projects_dir / str(project_id) / "models" / str(model_id) / "metrics" / "version_0" / "metrics.csv"
        )
        if not metrics_file.exists():
            raise ResourceNotFoundError(ResourceType.MODEL, f"{model_id} (metrics.csv not found)")

        return self._parse_metrics_csv(metrics_file)

    @staticmethod
    def _parse_metrics_csv(metrics_file: Path) -> list[dict]:
        """
        Parse a metrics CSV file and return the formatted metrics.

        For each metric column (excluding 'epoch', 'step', and blacklisted keys):
        1. Filter out rows with null values in the metric column
        2. Determine if the metric is epoch-based or step-based by checking if 'step'
           contains consecutive integers from 1 to N
        3. Build a TrainingMetrics object for that metric

        Args:
            metrics_file (Path): Path to the metrics.csv file.

        Returns:
            list[dict]: A list of formatted metric dictionaries.
        """
        metrics: list[dict] = []

        # Read the CSV file with polars
        df = pl.read_csv(metrics_file)

        # Get all metric columns (exclude 'epoch', 'step', and blacklisted keys)
        metric_columns = [col for col in df.columns if col not in BLACKLISTED_KEYS and col in KEY_MAPPING]

        for col in metric_columns:
            mapped_name = KEY_MAPPING.get(col)

            # Filter to include only 'epoch', 'step', and the current metric column
            # Then filter out rows where the metric column has null values
            metric_df = df.select(["epoch", "step", col]).filter(pl.col(col).is_not_null())

            if metric_df.is_empty():
                logger.debug("Metric '{}' has no non-null values, skipping", col)
                continue

            # Determine if the metric is step-based or epoch-based
            # Step-based: 'step' column contains consecutive integers from 1 to N
            # Epoch-based: 'epoch' column contains consecutive integers from 1 to N
            is_step_based = ModelService._is_step_based(metric_df)

            # Choose the appropriate x-axis column
            x_axis_col = "step" if is_step_based else "epoch"
            x_axis_label = "Step" if is_step_based else "Epoch"

            # Build the points list
            points = [
                {"x": float(row[x_axis_col]), "y": float(row[col]), "type": "point"}
                for row in metric_df.iter_rows(named=True)
            ]

            metric = {
                "header": mapped_name,
                "type": "line",
                "key": mapped_name,
                "value": {
                    "x_axis_label": x_axis_label,
                    "y_axis_label": mapped_name,
                    "line_data": [
                        {
                            "header": mapped_name,
                            "key": mapped_name,
                            "points": points,
                        }
                    ],
                },
            }
            metrics.append(metric)

        return metrics

    @staticmethod
    def _is_step_based(metric_df: pl.DataFrame) -> bool:
        """
        Determine if a metric is step-based by checking if the 'step' column
        contains all consecutive integers from 1 to N without skipping any value.

        Args:
            metric_df (pl.DataFrame): DataFrame to check for step-based or epoch-based metric.
                It is expected to have 'step' and 'epoch' columns, and at least one metric column (with no null values)

        Returns:
            bool: True if the metric is step-based, False if epoch-based.
        """
        # Get the step values and filter out nulls
        step_values = metric_df.select("step")
        epoch_values = metric_df.select("epoch")

        if step_values.is_empty() or epoch_values.is_empty():
            raise ValueError("Malformed metrics data: 'step' or 'epoch' column is missing or empty")

        # Get unique step values and sort them
        unique_steps = step_values["step"].unique()

        # Get the minimum and maximum step values
        min_step = int(unique_steps.min())  # pyrefly: ignore[no-matching-overload]
        max_step = int(unique_steps.max())  # pyrefly: ignore[no-matching-overload]

        # Expected count for consecutive integers from min to max
        expected_count = max_step - min_step + 1

        # If the number of unique steps equals the expected count, they are consecutive and hence step-based
        return len(unique_steps) == expected_count

    def get_logs(self, project_id: UUID, model_id: UUID) -> Path | None:
        """
        Get the training logs for a model revision.

        Args:
            project_id (UUID): The unique identifier of the project.
            model_id (UUID): The unique identifier of the model.

        Returns:
            Path | None: Path to the training log file.

        Raises:
            ResourceNotFoundError: If no model with the given model_id is found.
            ValueError: If the model is in NOT_STARTED or IN_PROGRESS status.
        """
        model_revision = self.get_model(project_id=project_id, model_id=model_id)

        if model_revision.training_info and model_revision.training_info.status in (
            TrainingStatus.NOT_STARTED,
            TrainingStatus.IN_PROGRESS,
        ):
            raise ValueError(
                "Logs are not available for models that have not started or are currently in progress of training"
            )

        log_file = self._projects_dir / str(project_id) / "models" / str(model_id) / "training.log"

        if not log_file.exists():
            return None

        return log_file
