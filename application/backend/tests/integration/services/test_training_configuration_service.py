# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB, ProjectDB, TrainingConfigurationDB
from app.models.training_configuration.configuration import (
    GlobalDatasetPreparationParameters,
    GlobalParameters,
    SubsetSplit,
    TrainingConfiguration,
)
from app.models.training_configuration.hyperparameters import Hyperparameters
from app.schemas.project import TaskType
from app.services import ResourceNotFoundError
from app.services.training_configuration_service import TrainingConfigurationService


@pytest.fixture
def fxt_training_configuration() -> TrainingConfiguration:
    """Create a mock training configuration."""
    return TrainingConfiguration(
        model_manifest_id="Custom_Image_Classification_EfficientNet-B0",
        global_parameters=GlobalParameters(
            dataset_preparation=GlobalDatasetPreparationParameters(
                subset_split=SubsetSplit(),
            )
        ),
        hyperparameters=Hyperparameters(),
    )


@pytest.fixture
def fxt_training_configuration_service(db_session: Session) -> TrainingConfigurationService:
    return TrainingConfigurationService(db_session)


class TestTrainingConfigurationService:
    def test_get_training_configuration_by_model_revision_id(
        self, fxt_training_configuration, fxt_training_configuration_service, db_session
    ):
        """Test getting training configuration by model revision ID."""
        project = ProjectDB(
            id=str(uuid4()),
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(project)
        model = ModelRevisionDB(
            id=str(uuid4()),
            project_id=project.id,
            architecture="Object_Detection_YOLOv5",
            training_status="running",
            training_configuration=fxt_training_configuration.model_dump(),
            label_schema_revision={},
            files_deleted=False,
        )
        db_session.add(model)
        db_session.flush()

        training_configuration = fxt_training_configuration_service.get_training_configuration(
            project_id=UUID(project.id), model_revision_id=UUID(model.id)
        )
        assert isinstance(training_configuration, TrainingConfiguration)
        assert training_configuration == fxt_training_configuration

    def test_get_training_configuration_by_model_revision_id_not_found(self, fxt_training_configuration_service):
        """Test getting training configuration with non-existent model revision."""
        with pytest.raises(ResourceNotFoundError):
            fxt_training_configuration_service.get_training_configuration(project_id=uuid4(), model_revision_id=uuid4())

    def test_get_training_configuration_by_model_architecture_id_from_db(
        self, fxt_training_configuration, fxt_training_configuration_service, db_session
    ):
        """Test getting training configuration by model architecture ID from database."""
        project = ProjectDB(
            id=str(uuid4()),
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(project)
        training_configuration = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=project.id,
            model_architecture_id="Custom_Object_Detection_YOLOX",
            configuration_data=fxt_training_configuration.model_dump(),
        )
        db_session.add(training_configuration)
        db_session.flush()

        training_configuration = fxt_training_configuration_service.get_training_configuration(
            project_id=UUID(project.id), model_architecture_id="Custom_Object_Detection_YOLOX"
        )
        assert isinstance(training_configuration, TrainingConfiguration)
        assert training_configuration == fxt_training_configuration

    def test_get_training_configuration_by_model_architecture_id_from_manifest(
        self, fxt_training_configuration, fxt_training_configuration_service, db_session
    ):
        """Test getting training configuration by model architecture ID from manifest."""
        training_configuration = fxt_training_configuration_service.get_training_configuration(
            project_id=uuid4(), model_architecture_id="Custom_Object_Detection_YOLOX"
        )
        assert isinstance(training_configuration, TrainingConfiguration)
        assert training_configuration != fxt_training_configuration

    def test_get_training_configuration_default_by_project_type(
        self, fxt_training_configuration, fxt_training_configuration_service, db_session
    ):
        """Test getting general training configuration from default model."""
        project = ProjectDB(
            id=str(uuid4()),
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(project)
        db_session.flush()

        training_configuration = fxt_training_configuration_service.get_training_configuration(
            project_id=UUID(project.id), model_architecture_id=None
        )
        assert isinstance(training_configuration, TrainingConfiguration)
        assert training_configuration != fxt_training_configuration

    def test_get_training_configuration_both_ids_provided_error(self, fxt_training_configuration_service):
        """Test error when both model_architecture_id and model_revision_id are provided."""
        with pytest.raises(ValueError) as exc_info:
            fxt_training_configuration_service.get_training_configuration(
                project_id=uuid4(), model_architecture_id="test_arch", model_revision_id=uuid4()
            )

        assert "Only one of model_architecture_id or model_revision_id should be provided" in str(exc_info.value)

    def test_update_training_configuration_new(
        self, fxt_training_configuration, fxt_training_configuration_service, db_session
    ):
        """Test updating a new training configuration."""
        project = ProjectDB(
            id=str(uuid4()),
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(project)
        db_session.flush()

        training_config_update = {
            "model_manifest_id": "Custom_Image_Classification_EfficientNet-B0",
            "hyperparameters": {
                "dataset_preparation": {"augmentation": {"topdown_affine": {"enable": True, "probability": 0.5}}},
                "training": {"max_epochs": 999},
                "evaluation": {"metric": "new_metric"},
            },
            "global_parameters": {
                "dataset_preparation": {"subset_split": {"training": 70, "validation": 15, "test": 15}}
            },
        }

        training_configuration = fxt_training_configuration_service.update_training_configuration(
            project_id=UUID(project.id), training_config_update=training_config_update, model_architecture_id=None
        )
        assert isinstance(training_configuration, TrainingConfiguration)
        assert training_configuration != fxt_training_configuration
        assert training_configuration.hyperparameters.dataset_preparation.augmentation.topdown_affine.enable is True
        assert training_configuration.hyperparameters.dataset_preparation.augmentation.topdown_affine.probability == 0.5
        assert training_configuration.hyperparameters.training.max_epochs == 999
        assert training_configuration.hyperparameters.evaluation.metric == "new_metric"
        assert training_configuration.global_parameters.dataset_preparation.subset_split.training == 70
        assert training_configuration.global_parameters.dataset_preparation.subset_split.validation == 15
        assert training_configuration.global_parameters.dataset_preparation.subset_split.test == 15

    def test_update_training_configuration_update(
        self, fxt_training_configuration, fxt_training_configuration_service, db_session
    ):
        """Test updating an existing training configuration with new configuration."""
        project = ProjectDB(
            id=str(uuid4()),
            name="Test Detection Project",
            task_type=TaskType.DETECTION,
            exclusive_labels=False,
        )
        db_session.add(project)
        training_configuration = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=project.id,
            model_architecture_id="Custom_Object_Detection_YOLOX",
            configuration_data=fxt_training_configuration.model_dump(),
        )
        db_session.add(training_configuration)
        db_session.flush()

        training_config_update = {
            "model_manifest_id": "Custom_Image_Classification_EfficientNet-B0",
            "hyperparameters": {
                "dataset_preparation": {"augmentation": {"topdown_affine": {"enable": True, "probability": 0.5}}},
                "training": {"max_epochs": 999},
                "evaluation": {"metric": "new_metric"},
            },
            "global_parameters": {
                "dataset_preparation": {"subset_split": {"training": 60, "validation": 10, "test": 30}}
            },
        }

        training_configuration = fxt_training_configuration_service.update_training_configuration(
            project_id=UUID(project.id),
            training_config_update=training_config_update,
            model_architecture_id="Custom_Object_Detection_YOLOX",
        )
        assert isinstance(training_configuration, TrainingConfiguration)
        assert training_configuration != fxt_training_configuration
        assert training_configuration.hyperparameters.dataset_preparation.augmentation.topdown_affine.enable is True
        assert training_configuration.hyperparameters.dataset_preparation.augmentation.topdown_affine.probability == 0.5
        assert training_configuration.hyperparameters.training.max_epochs == 999
        assert training_configuration.hyperparameters.evaluation.metric == "new_metric"
        assert training_configuration.global_parameters.dataset_preparation.subset_split.training == 60
        assert training_configuration.global_parameters.dataset_preparation.subset_split.validation == 10
        assert training_configuration.global_parameters.dataset_preparation.subset_split.test == 30
