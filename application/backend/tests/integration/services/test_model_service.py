# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import DatasetRevisionDB, EvaluationDB, MetricScoreDB, ModelRevisionDB, ProjectDB
from app.models import DatasetItemSubset, EvaluationResult
from app.models.model_revision import ModelFormat
from app.services import ModelService, ResourceNotFoundError, ResourceType
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
        request: pytest.FixtureRequest,
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

    def test_update_model(self, fxt_project_id: UUID, fxt_model_id: UUID, fxt_model_service: ModelService):
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

    def test_get_evaluation_results(
        self, fxt_model_id: UUID, fxt_project_id: UUID, fxt_model_service: ModelService, db_session: Session
    ):
        """Test retrieving evaluation results from the database succeeds."""
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
        evaluation_results = fxt_model_service.get_evaluation_results(fxt_model_id)

        # Assert
        assert evaluation_results[0] is not None
        evaluation_result = evaluation_results[0]
        assert evaluation_result.model_revision_id == fxt_model_id
        assert evaluation_result.dataset_revision_id == UUID(dataset_revision.id)
        assert evaluation_result.subset == DatasetItemSubset.TESTING
        assert evaluation_result.metrics == {
            "accuracy": 0.95,
            "f1_score": 0.89,
            "precision": 0.92,
        }
