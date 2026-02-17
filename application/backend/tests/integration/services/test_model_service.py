# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetRevisionDB, EvaluationDB, MetricScoreDB, ModelRevisionDB, ProjectDB
from app.models import DatasetItemSubset, EvaluationResult
from app.models.model_revision import ModelFormat, TrainingStatus
from app.services import ModelRevisionMetadata, ModelService, ResourceNotFoundError, ResourceType
from tests.integration.project_factory import ProjectTestDataFactory


@pytest.fixture(autouse=True)
def setup_project_with_models(
    fxt_db_projects: list[ProjectDB],
    fxt_db_models: list[ModelRevisionDB],
    db_session: Session,
) -> None:
    """Fixture to set up a project with dataset items in the database."""

    (
        ProjectTestDataFactory(db_session)
        .with_project(fxt_db_projects[0])
        .with_pipeline(is_running=True)
        .with_models(fxt_db_models)
        .build()
    )


@pytest.fixture
def fxt_model_service(tmp_path: Path, db_session: Session) -> ModelService:
    """Fixture to create a ModelService instance."""
    return ModelService(data_dir=tmp_path, db_session=db_session)


@pytest.fixture
def fxt_model_with_dataset_revision_db(tmp_path: Path, db_session: Session, fxt_project_id: UUID) -> dict[str, Any]:
    """Fixture to create a dataset revision and a model linked to it."""
    dataset_revision_id = uuid4()
    dataset_revision_path = tmp_path / "projects" / str(fxt_project_id) / "dataset_revisions" / str(dataset_revision_id)
    dataset_revision_path.mkdir(parents=True, exist_ok=True)
    (dataset_revision_path / "data.parquet").touch()
    dataset_revision_db = DatasetRevisionDB(
        id=str(dataset_revision_id),
        project_id=str(fxt_project_id),
        name="Test Dataset",
        files_deleted=False,
    )
    db_session.add(dataset_revision_db)
    model_id = uuid4()
    model_db = ModelRevisionDB(
        id=str(model_id),
        name="TestModel",
        project_id=str(fxt_project_id),
        architecture="TestArch",
        training_status="not_started",
        training_configuration={},
        training_dataset_id=str(dataset_revision_id),
        label_schema_revision={},
        files_deleted=False,
    )
    db_session.add(model_db)
    db_session.flush()
    return {
        "dataset_revision_id": dataset_revision_id,
        "dataset_revision_path": dataset_revision_path,
        "model_id": model_id,
        "dataset_revision_db": dataset_revision_db,
        "model_db": model_db,
    }


class TestModelServiceIntegration:
    """Integration tests for ModelService."""

    def test_list_models(
        self, fxt_project_id: UUID, fxt_db_models: list[ModelRevisionDB], fxt_model_service: ModelService
    ):
        """Test retrieving all models."""

        models = fxt_model_service.list_models(fxt_project_id)

        assert len(models) == len(fxt_db_models)
        model_ids = [str(m.id) for m in models]
        for idx in range(len(model_ids)):
            assert fxt_db_models[idx].id in model_ids

    def test_list_models_with_dataset_revision(
        self,
        db_session: Session,
        fxt_project_id: UUID,
        fxt_model_service: ModelService,
    ):
        # Create a dataset revision
        dataset_revision_id = uuid4()
        dataset_revision = DatasetRevisionDB(
            id=str(dataset_revision_id),
            project_id=str(fxt_project_id),
            name="Test Dataset",
            files_deleted=False,
        )
        db_session.add(dataset_revision)
        db_session.flush()

        # Add an extra model that is linked to the dataset revision
        model_id = uuid4()
        model = ModelRevisionDB(
            id=str(model_id),
            name="TestModel",
            project_id=str(fxt_project_id),
            architecture="TestArch",
            parent_revision=None,
            training_status="not_started",
            training_configuration={},
            training_dataset_id=str(dataset_revision_id),
            label_schema_revision={},
        )
        db_session.add(model)
        db_session.flush()

        # Call list_models with dataset_revision_id and without
        dataset_models = fxt_model_service.list_models(fxt_project_id, dataset_revision_id=dataset_revision_id)
        all_models = fxt_model_service.list_models(fxt_project_id)

        # Only the model linked to the dataset revision should be returned
        assert len(all_models) == 3
        assert len(dataset_models) == 1
        assert str(dataset_models[0].id) == str(model_id)

    def test_get_model(self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService):
        """Test retrieving a model by ID."""
        model = fxt_model_service.get_model(fxt_project_id, fxt_model_id)

        assert model is not None
        assert model.id == fxt_model_id

    def test_get_model_with_evaluations(
        self, fxt_model_id: UUID, fxt_project_id: UUID, fxt_model_service: ModelService, db_session: Session
    ):
        # Arrange
        dataset_revision = DatasetRevisionDB(id=str(uuid4()), project_id=str(fxt_project_id), name="test")
        evaluation = EvaluationDB(
            id=str(uuid4()),
            dataset_revision_id=dataset_revision.id,
            model_revision_id=str(fxt_model_id),
            subset=DatasetItemSubset.TESTING,
        )
        metrics = [
            MetricScoreDB(metric="accuracy", score=0.95),
            MetricScoreDB(metric="f1_score", score=0.89),
            MetricScoreDB(metric="precision", score=0.92),
        ]
        evaluation.metric_scores = metrics
        db_session.add(dataset_revision)
        db_session.flush()
        db_session.add(evaluation)
        db_session.flush()

        # Act
        model = fxt_model_service.get_model(fxt_project_id, fxt_model_id)

        # Assert
        evaluations = model.evaluations
        assert len(evaluations) == 1
        evaluation = evaluations[0]
        assert evaluation.model_revision_id == fxt_model_id
        assert evaluation.dataset_revision_id == UUID(dataset_revision.id)
        assert evaluation.subset == DatasetItemSubset.TESTING
        assert evaluation.metrics == {
            "accuracy": 0.95,
            "f1_score": 0.89,
            "precision": 0.92,
        }

    def test_get_model_variants(
        self, tmp_path: Path, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService
    ):
        """Test retrieving model variants."""
        model_vars_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_vars_path.mkdir(parents=True, exist_ok=True)
        (model_vars_path / "model.xml").touch()
        (model_vars_path / "model.bin").touch()
        (model_vars_path / "model.onnx").touch()
        (model_vars_path / "model.ckpt").touch()

        variants = fxt_model_service.get_model_variants(fxt_project_id, fxt_model_id)
        for variant in variants:
            assert variant.get("format") in ["openvino", "onnx", "pytorch"]
            assert variant.get("precision") in ["fp16", "fp32"]
            assert variant.get("weights_size") == 0  # Files are empty, so size is 0

    def test_get_model_size_in_bytes(
        self, tmp_path: Path, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService
    ):
        """Test retrieving total model size in bytes."""
        model_size_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_size_path.mkdir(parents=True, exist_ok=True)
        (model_size_path / "model.xml").write_bytes(b"x" * 100)
        (model_size_path / "model.bin").write_bytes(b"x" * 200)
        (model_size_path / "model.onnx").write_bytes(b"x" * 300)
        (model_size_path / "model.ckpt").write_bytes(b"x" * 400)

        total_size = fxt_model_service.get_model_size_in_bytes(fxt_project_id, fxt_model_id)

        assert total_size == 100 + 200 + 300 + 400

    def test_rename_model(self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService):
        """Test updating name of a model by ID."""
        new_model_name = "This is a new model name"
        model_metadata = {"name": new_model_name}
        model_from_get_before_update = fxt_model_service.get_model(fxt_project_id, fxt_model_id)
        model_from_update = fxt_model_service.rename_model(
            project_id=fxt_project_id, model_id=fxt_model_id, model_metadata=model_metadata
        )
        model_from_get_after_update = fxt_model_service.get_model(fxt_project_id, fxt_model_id)

        assert model_from_update.name == new_model_name
        assert model_from_get_before_update.name != new_model_name
        assert model_from_get_after_update.name == new_model_name

    @pytest.mark.parametrize("model_operation", ["get_model", "delete_model"])
    def test_non_existent_model(self, model_operation, fxt_project_id, fxt_db_projects, fxt_model_service, db_session):
        """Test retrieving a non-existent model raises error."""
        model_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            getattr(fxt_model_service, model_operation)(fxt_project_id, model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)

    def test_delete_model_no_files(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that deleting a model removes its database record when no filesystem artifacts exist."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)

        fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

        assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is None
        assert not model_rev_path.exists()

    def test_delete_model_with_files(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that deleting a model removes both its database record and filesystem artifacts."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_rev_path.mkdir(parents=True, exist_ok=True)
        (model_rev_path / "config.yaml").touch()
        (model_rev_path / "training.log").touch()

        fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

        assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is None
        assert not model_rev_path.exists()

    def test_delete_model_only_files(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that deleting only model files removes only filesystem artifacts."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_rev_path.mkdir(parents=True, exist_ok=True)
        (model_rev_path / "model.xml").touch()
        (model_rev_path / "model.bin").touch()

        fxt_model_service.delete_model_files(project_id=fxt_project_id, model_id=fxt_model_id)

        model_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        assert model_db is not None
        assert model_db.files_deleted is True
        assert not model_rev_path.exists()

    def test_delete_model_with_files_no_permission(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that delete_model fails when filesystem cleanup is denied due to OSError."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_rev_path.mkdir(parents=True, exist_ok=True)
        (model_rev_path / "config.yaml").touch()

        # Make directory read-only to simulate permission error
        model_rev_path.chmod(0o444)

        try:
            with pytest.raises(OSError):
                fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

            assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is not None
            assert model_rev_path.exists()
        finally:
            # Cleanup: restore permissions so pytest can clean up temp directory
            model_rev_path.chmod(0o755)

    def test_delete_model_triggers_dataset_revision_files_deletion(
        self,
        db_session: Session,
        fxt_project_id: UUID,
        fxt_model_service: ModelService,
        fxt_model_with_dataset_revision_db,
    ):
        """Test that deleting a model deletes associated dataset revision files if no models remain."""
        dataset_revision_id = fxt_model_with_dataset_revision_db["dataset_revision_id"]
        dataset_revision_path = fxt_model_with_dataset_revision_db["dataset_revision_path"]
        model_id = fxt_model_with_dataset_revision_db["model_id"]

        fxt_model_service.delete_model(project_id=fxt_project_id, model_id=model_id)

        assert not dataset_revision_path.exists()
        dataset_revision_db = db_session.get(DatasetRevisionDB, str(dataset_revision_id))
        assert dataset_revision_db is not None
        assert dataset_revision_db.files_deleted is True

    def test_delete_model_files_triggers_dataset_revision_files_deletion(
        self,
        db_session: Session,
        fxt_project_id: UUID,
        fxt_model_service: ModelService,
        fxt_model_with_dataset_revision_db,
    ):
        """Test that deleting model files deletes associated dataset revision files if no models remain."""
        dataset_revision_id = fxt_model_with_dataset_revision_db["dataset_revision_id"]
        dataset_revision_path = fxt_model_with_dataset_revision_db["dataset_revision_path"]
        model_id = fxt_model_with_dataset_revision_db["model_id"]

        fxt_model_service.delete_model_files(project_id=fxt_project_id, model_id=model_id)

        assert not dataset_revision_path.exists()
        dataset_revision_db = db_session.get(DatasetRevisionDB, str(dataset_revision_id))
        assert dataset_revision_db is not None
        assert dataset_revision_db.files_deleted is True

    @pytest.mark.parametrize(
        "model_format, expected_files",
        [
            (ModelFormat.OPENVINO, ["model.xml", "model.bin"]),
            (ModelFormat.ONNX, ["model.onnx"]),
            (ModelFormat.PYTORCH, ["model.ckpt"]),
        ],
    )
    def test_get_model_binary_files(
        self,
        model_format,
        expected_files,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
    ):
        """Test retrieving model binary files."""
        model_dir = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_dir.mkdir(parents=True)
        (model_dir / "model.xml").touch()
        (model_dir / "model.bin").touch()
        (model_dir / "model.onnx").touch()
        (model_dir / "model.ckpt").touch()

        files_exist, paths = fxt_model_service.get_model_binary_files(
            project_id=fxt_project_id, model_id=fxt_model_id, format=model_format
        )
        assert files_exist is True
        expected_paths = tuple(model_dir / file for file in expected_files)
        assert paths == expected_paths

    def test_create_revision(
        self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService, db_session: Session
    ):
        """Test creating a new model revision succeeds."""
        dataset_revision_db = DatasetRevisionDB(project_id=str(fxt_project_id), name="Dataset Revision")
        db_session.add(dataset_revision_db)
        db_session.flush()
        model_id = uuid4()
        dataset_revision_id = UUID(dataset_revision_db.id)
        architecture_id = "object-detection-atss-mobilenet-v2"

        fxt_model_service.create_revision(
            ModelRevisionMetadata(
                model_id=model_id,
                project_id=fxt_project_id,
                architecture_id=architecture_id,
                parent_revision_id=fxt_model_id,
                dataset_revision_id=dataset_revision_id,
                training_status=TrainingStatus.IN_PROGRESS,
            )
        )

        model_db = db_session.get(ModelRevisionDB, str(model_id))
        assert model_db is not None
        assert model_db.project_id == str(fxt_project_id)
        assert model_db.parent_revision == str(fxt_model_id)
        assert model_db.training_dataset_id == str(dataset_revision_id)
        assert model_db.architecture == architecture_id
        assert model_db.training_status == TrainingStatus.IN_PROGRESS
        assert model_db.name == f"ATSS-MobileNet-V2 ({str(model_id).split('-')[0]})"

    def test_update_revision(
        self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService, db_session: Session
    ):
        """Test updating an existing model revision succeeds."""
        fxt_model_service.update_revision_status(
            project_id=fxt_project_id,
            model_id=fxt_model_id,
            training_status=TrainingStatus.IN_PROGRESS,
        )

        model_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        assert model_db is not None
        assert model_db.training_status == TrainingStatus.IN_PROGRESS

    def test_save_evaluation_result(
        self, fxt_model_id: UUID, fxt_project_id: UUID, fxt_model_service: ModelService, db_session: Session
    ):
        """Test saving evaluation results to the database succeeds."""
        # Arrange
        dataset_revision = DatasetRevisionDB(project_id=str(fxt_project_id), name="test")
        db_session.add(dataset_revision)
        db_session.flush()

        evaluation_result = EvaluationResult(
            model_revision_id=fxt_model_id,
            dataset_revision_id=UUID(dataset_revision.id),
            subset=DatasetItemSubset.TESTING,
            metrics={"accuracy": 0.95, "f1_score": 0.89, "precision": 0.92},
        )

        # Act
        fxt_model_service.save_evaluation_result(evaluation_result)

        # Assert
        saved_evaluation = db_session.query(EvaluationDB).filter_by(model_revision_id=str(fxt_model_id)).first()

        assert saved_evaluation is not None
        assert saved_evaluation.model_revision_id == str(fxt_model_id)
        assert saved_evaluation.dataset_revision_id == str(dataset_revision.id)
        assert saved_evaluation.subset == DatasetItemSubset.TESTING
        assert len(saved_evaluation.metric_scores) == 3

        metrics_dict = {m.metric: m.score for m in saved_evaluation.metric_scores}
        assert metrics_dict["accuracy"] == 0.95
        assert metrics_dict["f1_score"] == 0.89
        assert metrics_dict["precision"] == 0.92

    def test_get_training_metrics_success(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
    ):
        """Test retrieving training metrics from metrics.csv file."""
        # Create a model directory with a metrics.csv file in the correct path
        metrics_dir = (
            tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id) / "metrics" / "version_0"
        )
        metrics_dir.mkdir(parents=True)

        csv_content = """epoch,step,train/total_loss,val/f1-score
        1,1,0.1,0.95
        2,2,0.2,0.89
        3,3,0.3,0.92
        """.replace(" ", "")  # Remove leading tabs for correct CSV formatting
        (metrics_dir / "metrics.csv").write_text(csv_content)

        metrics = fxt_model_service.get_model_training_metrics(project_id=fxt_project_id, model_id=fxt_model_id)
        # epoch and step are blacklisted, so only train/total_loss and val/f1-score are returned
        assert len(metrics) == 2
        for metric in metrics:
            assert metric["header"] in ["Training total loss", "Validation F1 score"]
            assert metric["type"] == "line"
            assert metric["key"] in ["Training total loss", "Validation F1 score"]
            assert metric.get("value")
            # Check that x_axis_label is set correctly (should be "Step" since steps are consecutive)
            assert metric["value"]["x_axis_label"] == "Step"

    def test_get_training_metrics_epoch_based(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
    ):
        """Test retrieving training metrics when steps are not consecutive (epoch-based)."""
        # Create a model directory with a metrics.csv file in the correct path
        metrics_dir = (
            tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id) / "metrics" / "version_0"
        )
        metrics_dir.mkdir(parents=True)

        # Steps are not consecutive (1, 5, 9) so metric should be epoch-based
        csv_content = """epoch,step,train/total_loss,val/f1-score
        1,1,0.1,0.95
        2,5,0.2,0.89
        3,9,0.3,0.92
        """.replace(" ", "")  # Remove leading tabs for correct CSV formatting
        (metrics_dir / "metrics.csv").write_text(csv_content)

        metrics = fxt_model_service.get_model_training_metrics(project_id=fxt_project_id, model_id=fxt_model_id)
        # epoch and step are blacklisted, so only train/total_loss and val/f1-score are returned
        assert len(metrics) == 2
        for metric in metrics:
            assert metric["header"] in ["Training total loss", "Validation F1 score"]
            assert metric["type"] == "line"
            # Check that x_axis_label is "Epoch" since steps are NOT consecutive
            assert metric["value"]["x_axis_label"] == "Epoch"

    def test_get_training_metrics_file_not_found(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
    ):
        """Test that ResourceNotFoundError is raised when metrics.csv doesn't exist."""
        # Create model directory without metrics.csv (the metrics/version_0 path is expected)
        model_dir = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_dir.mkdir(parents=True)

        with pytest.raises(ResourceNotFoundError) as exc_info:
            fxt_model_service.get_model_training_metrics(project_id=fxt_project_id, model_id=fxt_model_id)

        assert "metrics.csv not found" in str(exc_info.value)

    @pytest.mark.parametrize("training_status", [TrainingStatus.SUCCESSFUL, TrainingStatus.FAILED])
    def test_get_logs_success(
        self,
        training_status: TrainingStatus,
        tmp_path: Path,
        fxt_model_id: UUID,
        fxt_project_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test retrieving training logs for a trained model."""
        # Arrange
        model_rev_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        assert model_rev_db
        model_rev_db.training_status = training_status
        db_session.add(model_rev_db)
        db_session.flush()

        log_file = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id) / "training.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_content = "Training started\nEpoch 1/10\nLoss: 0.5\n"
        log_file.write_text(log_content)

        # Act
        result = fxt_model_service.get_logs(project_id=fxt_project_id, model_id=fxt_model_id)

        # Assert
        assert result is not None
        assert result.exists()
        assert result.read_text() == log_content

    @pytest.mark.parametrize("training_status", [TrainingStatus.NOT_STARTED, TrainingStatus.IN_PROGRESS])
    def test_get_logs_in_status(
        self,
        training_status: TrainingStatus,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test retrieving logs for a model that has not started or in_progress training status."""
        # Arrange
        model_rev_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        assert model_rev_db
        model_rev_db.training_status = training_status
        db_session.add(model_rev_db)
        db_session.flush()

        # Act & Assert
        with pytest.raises(ValueError) as excinfo:
            fxt_model_service.get_logs(project_id=fxt_project_id, model_id=fxt_model_id)

        assert (
            str(excinfo.value)
            == "Logs are not available for models that have not started or are currently in progress of training"
        )

    def test_get_logs_file_not_exists(
        self,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test retrieving logs when the log file does not exist."""
        # Arrange
        model_rev_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        assert model_rev_db
        model_rev_db.training_status = TrainingStatus.SUCCESSFUL
        db_session.add(model_rev_db)
        db_session.flush()

        # Act
        result = fxt_model_service.get_logs(project_id=fxt_project_id, model_id=fxt_model_id)

        # Assert
        assert result is None

    def test_get_logs_non_existent_model(
        self,
        fxt_project_id: UUID,
        fxt_model_service: ModelService,
    ):
        """Test retrieving logs for a non-existent model."""
        # Arrange
        model_id = uuid4()

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_model_service.get_logs(project_id=fxt_project_id, model_id=model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)
