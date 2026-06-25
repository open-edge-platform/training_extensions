# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import (
    DatasetRevisionDB,
    EvaluationDB,
    MetricScoreDB,
    ModelRevisionDB,
    ModelVariantDB,
    PipelineDB,
    ProjectDB,
)
from app.models import DatasetItemSubset, EvaluationResult
from app.models.model_revision import ModelFormat, TrainingStatus
from app.services import ModelRevisionMetadata, ModelService, ResourceInUseError, ResourceNotFoundError, ResourceType
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

    model_id_1 = uuid4()
    model_db_1 = ModelRevisionDB(
        id=str(model_id_1),
        name="TestModel",
        project_id=str(fxt_project_id),
        architecture="TestArch",
        training_status="not_started",
        training_configuration={},
        training_dataset_id=str(dataset_revision_id),
        label_schema_revision={},
        files_deleted=False,
    )
    db_session.add(model_db_1)

    model_id_2 = uuid4()
    model_db_2 = ModelRevisionDB(
        id=str(model_id_2),
        name="TestModel",
        project_id=str(fxt_project_id),
        architecture="TestArch",
        training_status="not_started",
        training_configuration={},
        training_dataset_id=str(dataset_revision_id),
        label_schema_revision={},
        files_deleted=False,
    )
    db_session.add(model_db_2)
    db_session.flush()
    return {
        "dataset_revision_id": dataset_revision_id,
        "dataset_revision_path": dataset_revision_path,
        "dataset_revision_db": dataset_revision_db,
        "model_id_1": model_id_1,
        "model_db_1": model_db_1,
        "model_id_2": model_id_2,
        "model_db_2": model_db_2,
    }


class TestModelServiceIntegration:
    """Integration tests for ModelService."""

    def test_get_model_revision_architecture(
        self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService
    ):
        """Test retrieving the architecture ID of a model revision."""
        architecture = fxt_model_service.get_model_revision_architecture(fxt_project_id, fxt_model_id)

        assert architecture is not None
        assert isinstance(architecture, str)

    def test_get_model_revision_architecture_not_found(self, fxt_project_id: UUID, fxt_model_service: ModelService):
        """Test retrieving architecture for a non-existent model raises error."""
        model_id = uuid4()
        with pytest.raises(ResourceNotFoundError) as excinfo:
            fxt_model_service.get_model_revision_architecture(fxt_project_id, model_id)

        assert excinfo.value.resource_type == ResourceType.MODEL
        assert excinfo.value.resource_id == str(model_id)

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
        db_session.add(dataset_revision)
        db_session.flush()

        # Create a model variant
        variant_id = uuid4()
        variant = ModelVariantDB(
            id=str(variant_id),
            model_revision_id=str(fxt_model_id),
            format="openvino",
            precision="fp16",
        )
        db_session.add(variant)
        db_session.flush()

        evaluation = EvaluationDB(
            id=str(uuid4()),
            model_revision_id=str(fxt_model_id),
            model_variant_id=str(variant_id),
            dataset_revision_id=dataset_revision.id,
            subset=DatasetItemSubset.TESTING,
        )
        metrics = [
            MetricScoreDB(metric="accuracy", score=0.95),
            MetricScoreDB(metric="f1_score", score=0.89),
            MetricScoreDB(metric="precision", score=0.92),
        ]
        evaluation.metric_scores = metrics
        db_session.add(evaluation)
        db_session.flush()

        # Act
        model = fxt_model_service.get_model(fxt_project_id, fxt_model_id)

        # Assert
        assert len(model.variants) == 1
        variant_result = model.variants[0]
        assert len(variant_result.evaluations) == 1
        evaluation_result = variant_result.evaluations[0]
        assert evaluation_result.model_variant_id == variant_id
        assert evaluation_result.dataset_revision_id == UUID(dataset_revision.id)
        assert evaluation_result.subset == DatasetItemSubset.TESTING
        assert evaluation_result.metrics == {
            "accuracy": 0.95,
            "f1_score": 0.89,
            "precision": 0.92,
        }

    def test_get_model_variants(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test retrieving model variants."""
        # Create variant records in the database
        ov_variant_id = uuid4()
        onnx_variant_id = uuid4()
        pt_variant_id = uuid4()

        db_session.add(
            ModelVariantDB(
                id=str(ov_variant_id), model_revision_id=str(fxt_model_id), format="openvino", precision="fp16"
            )
        )
        db_session.add(
            ModelVariantDB(
                id=str(onnx_variant_id), model_revision_id=str(fxt_model_id), format="onnx", precision="fp16"
            )
        )
        db_session.add(
            ModelVariantDB(
                id=str(pt_variant_id), model_revision_id=str(fxt_model_id), format="pytorch", precision="fp32"
            )
        )
        db_session.flush()

        # Create variant directories with files
        for vid, files in [
            (ov_variant_id, ["model.xml", "model.bin"]),
            (onnx_variant_id, ["model.onnx"]),
            (pt_variant_id, ["model.pt"]),
        ]:
            variant_dir = (
                tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id) / "variants" / str(vid)
            )
            variant_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                (variant_dir / f).touch()

        variants = fxt_model_service.get_model_variants(fxt_project_id, fxt_model_id)
        assert len(variants) == 3
        for variant in variants:
            assert variant.format in ["openvino", "onnx", "pytorch"]
            assert variant.precision in ["fp16", "fp32"]
            assert variant.weights_size == 0  # Files are empty, so size is 0

    def test_get_model_size_in_bytes(
        self, tmp_path: Path, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService
    ):
        """Test retrieving total model size in bytes."""
        model_base_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        # Create variant directories with files
        ov_dir = model_base_path / "variants" / str(uuid4())
        ov_dir.mkdir(parents=True, exist_ok=True)
        (ov_dir / "model.xml").write_bytes(b"x" * 100)
        (ov_dir / "model.bin").write_bytes(b"x" * 200)
        onnx_dir = model_base_path / "variants" / str(uuid4())
        onnx_dir.mkdir(parents=True, exist_ok=True)
        (onnx_dir / "model.onnx").write_bytes(b"x" * 300)
        pt_dir = model_base_path / "variants" / str(uuid4())
        pt_dir.mkdir(parents=True, exist_ok=True)
        (pt_dir / "model.pt").write_bytes(b"x" * 400)

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

        with patch("app.services.model_service.shutil.rmtree") as mock_rmtree:
            fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

            mock_rmtree.assert_called_once_with(model_rev_path)
        assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is None

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

        with patch("app.services.model_service.shutil.rmtree") as mock_rmtree:
            fxt_model_service.delete_model_files(project_id=fxt_project_id, model_id=fxt_model_id)

            mock_rmtree.assert_called_once_with(model_rev_path)
        model_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        assert model_db is not None
        assert model_db.files_deleted is True

    def test_delete_model_files_raises_when_model_active_in_pipeline(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that deleting model files raises ResourceInUseError when the model is active in a pipeline."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_rev_path.mkdir(parents=True, exist_ok=True)
        (model_rev_path / "model.xml").touch()
        (model_rev_path / "model.bin").touch()

        # Assign the model to the pipeline so it becomes active
        pipeline = db_session.get(PipelineDB, str(fxt_project_id))
        pipeline.model_revision_id = str(fxt_model_id)  # pyrefly: ignore[missing-attribute]
        db_session.flush()

        with patch("app.services.model_service.shutil.rmtree") as mock_rmtree:
            with pytest.raises(ResourceInUseError):
                fxt_model_service.delete_model_files(project_id=fxt_project_id, model_id=fxt_model_id)

            mock_rmtree.assert_not_called()

        # Verify DB record is unchanged
        model_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        assert model_db is not None
        assert model_db.files_deleted is False

    def test_delete_model_with_files_no_permission(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test that delete_model throws ResourceInUseError when Windows file locks are encountered."""
        model_rev_path = tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id)
        model_rev_path.mkdir(parents=True, exist_ok=True)
        (model_rev_path / "config.yaml").touch()

        err = OSError("Permission denied")
        err.winerror = 32

        with (
            patch("app.services.model_service.shutil.rmtree", side_effect=err),
            pytest.raises(ResourceInUseError),
        ):
            fxt_model_service.delete_model(project_id=fxt_project_id, model_id=fxt_model_id)

        assert db_session.get(ModelRevisionDB, str(fxt_model_id)) is not None
        assert model_rev_path.exists()

    def test_delete_model_triggers_dataset_revision_files_deletion(
        self,
        db_session: Session,
        fxt_project_id: UUID,
        fxt_model_service: ModelService,
        fxt_model_with_dataset_revision_db,
    ):
        """Test that deleting a model deletes associated dataset revision if no models remain."""
        dataset_revision_id = fxt_model_with_dataset_revision_db["dataset_revision_id"]
        dataset_revision_path = fxt_model_with_dataset_revision_db["dataset_revision_path"]
        model_id_1 = fxt_model_with_dataset_revision_db["model_id_1"]
        model_id_2 = fxt_model_with_dataset_revision_db["model_id_2"]

        fxt_model_service.delete_model(project_id=fxt_project_id, model_id=model_id_1)

        assert dataset_revision_path.exists()
        dataset_revision_db = db_session.get(DatasetRevisionDB, str(dataset_revision_id))
        assert dataset_revision_db is not None
        assert dataset_revision_db.files_deleted is False

        fxt_model_service.delete_model(project_id=fxt_project_id, model_id=model_id_2)

        assert not dataset_revision_path.exists()
        dataset_revision_db = db_session.get(DatasetRevisionDB, str(dataset_revision_id))
        assert dataset_revision_db is None

    @pytest.mark.parametrize(
        "model_format, expected_files, precision",
        [
            (ModelFormat.OPENVINO, ["model.xml", "model.bin"], "fp16"),
            (ModelFormat.ONNX, ["model.onnx"], "fp16"),
            (ModelFormat.PYTORCH, ["model.pt"], "fp32"),
        ],
    )
    def test_get_model_binary_files(
        self,
        model_format,
        expected_files,
        precision,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
        db_session: Session,
    ):
        """Test retrieving model binary files."""
        # Create variant record in database
        variant_id = uuid4()
        db_session.add(
            ModelVariantDB(
                id=str(variant_id),
                model_revision_id=str(fxt_model_id),
                format=model_format.value,
                precision=precision,
            )
        )
        db_session.flush()

        # Create variant directory with files
        variant_dir = (
            tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id) / "variants" / str(variant_id)
        )
        variant_dir.mkdir(parents=True)
        for f in expected_files:
            (variant_dir / f).touch()

        files_exist, paths = fxt_model_service.get_model_binary_files(
            project_id=fxt_project_id, model_id=fxt_model_id, model_variant_id=variant_id
        )
        assert files_exist is True
        expected_paths = tuple(variant_dir / file for file in expected_files)
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
                model_name=f"ATSS-MobileNet-V2 ({str(model_id).split('-')[0]})",
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
        started_at = datetime.now(UTC)
        finished_at = started_at + timedelta(hours=1)

        # Test that we can update status, start and finish time
        fxt_model_service.update_revision_status(
            project_id=fxt_project_id,
            model_id=fxt_model_id,
            training_status=TrainingStatus.IN_PROGRESS,
            training_started_at=started_at,
            training_finished_at=finished_at,
        )

        model_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        db_session.refresh(model_db)
        assert model_db is not None
        assert model_db.training_status == TrainingStatus.IN_PROGRESS
        assert model_db.training_started_at == started_at
        assert model_db.training_finished_at == finished_at

        # Test that not providing start and/or finish time won't affect currently set times
        fxt_model_service.update_revision_status(
            project_id=fxt_project_id,
            model_id=fxt_model_id,
            training_status=TrainingStatus.SUCCESSFUL,
        )

        model_db = db_session.get(ModelRevisionDB, str(fxt_model_id))
        db_session.refresh(model_db)
        assert model_db is not None
        assert model_db.training_status == TrainingStatus.SUCCESSFUL
        assert model_db.training_started_at == started_at
        assert model_db.training_finished_at == finished_at

    def test_save_evaluation_result(
        self, fxt_model_id: UUID, fxt_project_id: UUID, fxt_model_service: ModelService, db_session: Session
    ):
        """Test saving evaluation results to the database succeeds."""
        # Arrange
        dataset_revision = DatasetRevisionDB(project_id=str(fxt_project_id), name="test")
        db_session.add(dataset_revision)
        db_session.flush()

        # Create a variant to associate the evaluation with
        variant_id = uuid4()
        variant = ModelVariantDB(
            id=str(variant_id),
            model_revision_id=str(fxt_model_id),
            format="openvino",
            precision="fp16",
        )
        db_session.add(variant)
        db_session.flush()

        evaluation_result = EvaluationResult(
            model_revision_id=fxt_model_id,
            model_variant_id=variant_id,
            dataset_revision_id=UUID(dataset_revision.id),
            subset=DatasetItemSubset.TESTING,
            metrics={"accuracy": 0.95, "f1_score": 0.89, "precision": 0.92},
        )

        # Act
        fxt_model_service.save_evaluation_result(evaluation_result)

        # Assert
        saved_evaluation = db_session.query(EvaluationDB).filter_by(model_variant_id=str(variant_id)).first()

        assert saved_evaluation is not None
        assert saved_evaluation.model_variant_id == str(variant_id)
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
            # Check that x_axis_label is set correctly
            assert metric["value"]["x_axis_label"] in ["Step", "Epoch"]

    def test_get_training_metrics_epoch_step_based(
        self,
        tmp_path: Path,
        fxt_project_id: UUID,
        fxt_model_id: UUID,
        fxt_model_service: ModelService,
    ):
        """Test retrieving training metrics when steps are not consecutive."""
        # Create a model directory with a metrics.csv file in the correct path
        metrics_dir = (
            tmp_path / "projects" / str(fxt_project_id) / "models" / str(fxt_model_id) / "metrics" / "version_0"
        )
        metrics_dir.mkdir(parents=True)

        # Steps are not consecutive (1, 5, 9) so metric should be epoch-based
        csv_content = """epoch,step,train/total_loss,val/f1-score
        ,1,0.1,0.95
        1,5,0.2,0.89
        2,9,0.3,0.92
        """.replace(" ", "")  # Remove leading tabs for correct CSV formatting
        (metrics_dir / "metrics.csv").write_text(csv_content)

        metrics = fxt_model_service.get_model_training_metrics(project_id=fxt_project_id, model_id=fxt_model_id)
        # epoch and step are blacklisted, so only train/total_loss and val/f1-score are returned
        assert len(metrics) == 2
        for metric in metrics:
            assert metric["header"] in ["Training total loss", "Validation F1 score"]
            assert metric["type"] == "line"
            # This test requires the metrics to be correctly identified
            # Even though the steps are not consecutive,
            # the x_axis_label should still be "Step" for train/total_loss and "Epoch" for val/f1-score.
            # This can happen when the user sets log_n_steps = 4 (>1)
            if metric["header"] == "Training total loss":
                assert metric["value"]["x_axis_label"] == "Step"
            elif metric["header"] == "Validation F1 score":
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

    @pytest.mark.parametrize(
        "training_status, as_text",
        [
            (TrainingStatus.SUCCESSFUL, True),
            (TrainingStatus.SUCCESSFUL, False),
            (TrainingStatus.FAILED, True),
            (TrainingStatus.FAILED, False),
        ],
    )
    def test_get_logs_success(
        self,
        training_status: TrainingStatus,
        as_text: bool,
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
        log_content = (
            '{ "text": "Training started\\n", "payload": "abc" }\n'
            '{ "text": "Epoch 1/10\\n", "payload": "dfe"}\n'
            '{ "text": "Epoch 2/10\\n"\n'
        )
        log_file.write_text(log_content)

        # Act
        result = fxt_model_service.get_logs(project_id=fxt_project_id, model_id=fxt_model_id, as_text=as_text)

        # Assert
        assert result is not None
        if as_text:
            assert isinstance(result, Iterator)
            assert list(result) == ["Training started\n", "Epoch 1/10\n", "[MALFORMED LOG LINE]\n"]
        else:
            assert isinstance(result, Path)
            assert result.exists()
            assert result.read_text() == log_content

    @pytest.mark.parametrize("training_status", [TrainingStatus.NOT_STARTED, TrainingStatus.IN_PROGRESS])
    def test_get_logs_not_supported_status(
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
