# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest

from app.db.schema import ModelRevisionDB, ModelVariantDB, PipelineDB, ProjectDB
from app.models.model_revision import ModelFormat, ModelPrecision, TrainingStatus
from app.services.active_model_service import ActiveModelService
from tests.integration.project_factory import ProjectTestDataFactory


@pytest.fixture
def fxt_project(fxt_db_projects, db_session) -> ProjectDB:
    """Fixture to create a project in the database."""
    return ProjectTestDataFactory(db_session).with_project(fxt_db_projects[0]).build()


@pytest.fixture
def fxt_successful_model(fxt_project) -> ModelRevisionDB:
    """Fixture providing a successfully trained model revision."""
    return ModelRevisionDB(
        id=str(uuid4()),
        project_id=fxt_project.id,
        name="YOLOX-S (successful)",
        architecture="object-detection-yolox-s",
        training_status=TrainingStatus.SUCCESSFUL,
        training_configuration={},
        label_schema_revision={},
    )


@pytest.fixture
def fxt_failed_model(fxt_project) -> ModelRevisionDB:
    """Fixture providing a failed model revision."""
    return ModelRevisionDB(
        id=str(uuid4()),
        project_id=fxt_project.id,
        name="YOLOX-S (failed)",
        architecture="object-detection-yolox-s",
        training_status=TrainingStatus.FAILED,
        training_configuration={},
        label_schema_revision={},
    )


@pytest.fixture
def fxt_in_progress_model(fxt_project) -> ModelRevisionDB:
    """Fixture providing an in-progress model revision."""
    return ModelRevisionDB(
        id=str(uuid4()),
        project_id=fxt_project.id,
        name="YOLOX-S (in_progress)",
        architecture="object-detection-yolox-s",
        training_status=TrainingStatus.IN_PROGRESS,
        training_configuration={},
        label_schema_revision={},
    )


@pytest.fixture
def fxt_fp16_openvino_variant(fxt_successful_model) -> ModelVariantDB:
    """Fixture providing an FP16 OpenVINO model variant for the successful model."""
    return ModelVariantDB(
        id=str(uuid4()),
        model_revision_id=fxt_successful_model.id,
        format=ModelFormat.OPENVINO,
        precision=ModelPrecision.FP16,
    )


def _make_db_session_patcher(db_session):
    """Return a context manager that patches get_db_session to yield the test db_session."""

    @contextmanager
    def _patched_get_db_session():
        yield db_session

    return patch("app.services.active_model_service.get_db_session", side_effect=_patched_get_db_session)


class TestActiveModelServiceLoadState:
    """Integration tests for ActiveModelService._load_state."""

    def test_load_state_no_active_pipeline_returns_empty_state(self, db_session, tmp_path):
        """When no pipeline is running, load_state returns an empty ModelActivationState."""
        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path)

        state = service._model_activation_state

        assert state.project_id is None
        assert state.active_model_id is None
        assert state.active_model_variant_id is None
        assert state.available_models == []

    def test_load_state_only_includes_successful_models(
        self,
        fxt_project,
        fxt_successful_model,
        fxt_failed_model,
        fxt_in_progress_model,
        fxt_fp16_openvino_variant,
        db_session,
        tmp_path,
    ):
        """available_models must only contain successfully trained model revisions."""
        # Persist models with different statuses and the active variant
        second_successful = ModelRevisionDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            name="YOLOX-X (successful)",
            architecture="object-detection-yolox-x",
            training_status=TrainingStatus.SUCCESSFUL,
            training_configuration={},
            label_schema_revision={},
        )
        db_session.add_all([fxt_successful_model, second_successful, fxt_failed_model, fxt_in_progress_model])
        db_session.add(fxt_fp16_openvino_variant)
        db_session.flush()

        # Create a running pipeline pointing at the successful model
        pipeline = PipelineDB(
            project_id=fxt_project.id,
            is_running=True,
            model_revision_id=fxt_successful_model.id,
            device="cpu",
        )
        db_session.add(pipeline)
        db_session.flush()

        with _make_db_session_patcher(db_session):
            service = ActiveModelService(data_dir=tmp_path)

        state = service._model_activation_state

        assert len(state.available_models) == 2
        assert set(state.available_models) == {UUID(fxt_successful_model.id), UUID(second_successful.id)}
        assert str(state.active_model_id) == fxt_successful_model.id
        assert str(state.active_model_variant_id) == fxt_fp16_openvino_variant.id
