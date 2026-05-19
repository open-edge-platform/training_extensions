# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.schema import ModelRevisionDB, ProjectDB, TrainingConfigurationDB
from app.models import TaskType
from app.models.training_configuration.configuration import (
    AlgoLevelParameters,
    TaskLevelParameters,
    TrainingConfiguration,
)
from app.models.training_configuration.dataset_preparation import (
    AlgoLevelDatasetPreparationParameters,
    SubsetSplit,
    TaskLevelDatasetPreparationParameters,
)
from app.models.training_configuration.training import AlgoLevelTrainingParameters, EarlyStopping
from app.services import ResourceNotFoundError
from app.services.training_configuration_service import TrainingConfigurationService


@pytest.fixture
def fxt_project(db_session: Session) -> ProjectDB:
    """Create a test project."""
    project = ProjectDB(
        id=str(uuid4()),
        name="Test Detection Project",
        task_type=TaskType.DETECTION,
        exclusive_labels=False,
    )
    db_session.add(project)
    db_session.flush()
    return project


@pytest.fixture
def fxt_training_configuration() -> TrainingConfiguration:
    """Create a mock training configuration."""
    return TrainingConfiguration(
        task_level_parameters=TaskLevelParameters(
            dataset_preparation=TaskLevelDatasetPreparationParameters(
                subset_split=SubsetSplit(training=65, validation=25, test=10),
            )
        ),
        algo_level_parameters=AlgoLevelParameters(
            dataset_preparation=AlgoLevelDatasetPreparationParameters(),
            training=AlgoLevelTrainingParameters(
                max_epochs=100,
                early_stopping=EarlyStopping(enable=True, patience=5),
                learning_rate=0.001,
                input_size_width=256,
                input_size_height=256,
                allowed_values_input_size=[128, 256, 512],
            ),
        ),
    )


@pytest.fixture
def fxt_training_configuration_service(db_session: Session) -> TrainingConfigurationService:
    return TrainingConfigurationService(db_session)


class TestTrainingConfigurationService:
    """Tests for TrainingConfigurationService."""

    def test_get_by_model_revision(
        self,
        fxt_training_configuration: TrainingConfiguration,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Test retrieving training configuration from an existing model revision."""
        model = ModelRevisionDB(
            id=str(uuid4()),
            name="YOLOX-S (abc123)",
            project_id=fxt_project.id,
            architecture="object-detection-yolox-s",
            training_status="running",
            training_configuration=fxt_training_configuration.model_dump(),
            label_schema_revision={},
            files_deleted=False,
        )
        db_session.add(model)
        db_session.flush()

        result = fxt_training_configuration_service.get_by_model_revision(
            project_id=UUID(fxt_project.id),
            model_revision_id=UUID(model.id),
        )

        assert isinstance(result, TrainingConfiguration)
        assert result == fxt_training_configuration

    def test_get_by_model_revision_with_missing_revision(
        self,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
    ):
        """Test that ResourceNotFoundError is raised when model revision doesn't exist."""
        with pytest.raises(ResourceNotFoundError):
            fxt_training_configuration_service.get_by_model_revision(
                project_id=UUID(fxt_project.id),
                model_revision_id=uuid4(),
            )

    def test_get_default_by_model_architecture(
        self,
        fxt_training_configuration_service: TrainingConfigurationService,
    ):
        """Test retrieving default training configuration for a model architecture."""
        TrainingConfigurationService.get_default_by_model_architecture.cache_clear()

        result = TrainingConfigurationService.get_default_by_model_architecture(
            model_architecture_id="object-detection-yolox-s",
        )

        assert isinstance(result, TrainingConfiguration)
        # Task-level parameters should be the defaults
        default_task_level = TaskLevelParameters()
        assert (
            result.task_level_parameters.dataset_preparation.subset_split.training
            == default_task_level.dataset_preparation.subset_split.training
        )
        assert (
            result.task_level_parameters.dataset_preparation.subset_split.validation
            == default_task_level.dataset_preparation.subset_split.validation
        )
        assert (
            result.task_level_parameters.dataset_preparation.subset_split.test
            == default_task_level.dataset_preparation.subset_split.test
        )
        # Algo-level parameters should be populated from the model manifest
        assert result.algo_level_parameters is not None
        assert result.algo_level_parameters.training.learning_rate == 0.001

        TrainingConfigurationService.get_default_by_model_architecture.cache_clear()

    def test_get_default_by_model_architecture_strips_tiling_when_unsupported(
        self,
        fxt_training_configuration_service: TrainingConfigurationService,
    ):
        """Tiling parameters should be removed from the default config when the architecture doesn't support tiling."""
        TrainingConfigurationService.get_default_by_model_architecture.cache_clear()

        # D-FINE-M does not support tiling (capabilities.tiling: false in its manifest)
        result = TrainingConfigurationService.get_default_by_model_architecture(
            model_architecture_id="object-detection-dfine-m",
        )

        assert result.algo_level_parameters.dataset_preparation.augmentation.tiling is None

        # Sanity check: a model that supports tiling still exposes the tiling parameters
        TrainingConfigurationService.get_default_by_model_architecture.cache_clear()
        yolox_result = TrainingConfigurationService.get_default_by_model_architecture(
            model_architecture_id="object-detection-yolox-s",
        )
        assert yolox_result.algo_level_parameters.dataset_preparation.augmentation.tiling is not None

        TrainingConfigurationService.get_default_by_model_architecture.cache_clear()

    def test_get_by_model_architecture_strips_tiling_when_unsupported(
        self,
        fxt_training_configuration: TrainingConfiguration,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Persisted tiling values should be stripped when the architecture doesn't support tiling."""
        # Persist an algo-level configuration with tiling populated (simulating an older configuration)
        algo_level_data = fxt_training_configuration.algo_level_parameters.model_dump()
        algo_level_data["dataset_preparation"]["augmentation"]["tiling"] = {
            "enable": True,
            "enable_adaptive_tiling": True,
            "tile_size": 400,
            "tile_overlap": 0.2,
        }
        algo_level_config = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            model_architecture_id="object-detection-dfine-m",
            configuration_data=algo_level_data,
        )
        db_session.add(algo_level_config)
        db_session.flush()

        result = fxt_training_configuration_service.get_by_model_architecture(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-dfine-m",
        )

        assert result.algo_level_parameters.dataset_preparation.augmentation.tiling is None

    def test_get_by_model_architecture_with_existing_config(
        self,
        fxt_training_configuration: TrainingConfiguration,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Test retrieving configuration when both task-level and algo-level configs exist in DB."""
        # Create task-level configuration
        task_level_config = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            model_architecture_id=None,
            configuration_data=fxt_training_configuration.task_level_parameters.model_dump(),
        )
        db_session.add(task_level_config)

        # Create algo-level configuration
        algo_level_config = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            model_architecture_id="object-detection-yolox-s",
            configuration_data=fxt_training_configuration.algo_level_parameters.model_dump(),
        )
        db_session.add(algo_level_config)
        db_session.flush()

        result = fxt_training_configuration_service.get_by_model_architecture(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-yolox-s",
        )

        assert isinstance(result, TrainingConfiguration)
        assert result.task_level_parameters.dataset_preparation.subset_split.training == 65
        assert result.algo_level_parameters.training.max_epochs == 100

    def test_get_by_model_architecture_with_default_task_level_config(
        self,
        fxt_training_configuration: TrainingConfiguration,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Test that default task-level parameters are used when not found in DB."""
        # Only create algo-level configuration
        algo_level_config = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            model_architecture_id="object-detection-yolox-s",
            configuration_data=fxt_training_configuration.algo_level_parameters.model_dump(),
        )
        db_session.add(algo_level_config)
        db_session.flush()

        result = fxt_training_configuration_service.get_by_model_architecture(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-yolox-s",
        )

        assert isinstance(result, TrainingConfiguration)
        # Should have default task-level parameters
        default_task_level = TaskLevelParameters()
        assert (
            result.task_level_parameters.dataset_preparation.subset_split.training
            == default_task_level.dataset_preparation.subset_split.training
        )

    def test_get_by_model_architecture_with_default_algo_level_config(
        self,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Test that default algo-level parameters from model manifest are used when not found in DB."""
        # Only create task-level configuration
        task_level_config = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            model_architecture_id=None,
            configuration_data=TaskLevelParameters().model_dump(),
        )
        db_session.add(task_level_config)
        db_session.flush()

        # This should fetch default algo-level parameters from ModelManifestService
        result = fxt_training_configuration_service.get_by_model_architecture(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-yolox-s",
        )

        assert isinstance(result, TrainingConfiguration)
        # Should have default task-level parameters
        default_task_level = TaskLevelParameters()
        assert (
            result.task_level_parameters.dataset_preparation.subset_split.training
            == default_task_level.dataset_preparation.subset_split.training
        )
        # Should have algo-level parameters from the model manifest
        assert result.algo_level_parameters is not None

    def test_update_without_preexisting_configuration(
        self,
        fxt_training_configuration: TrainingConfiguration,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Test creating new task-level and algo-level configurations."""
        fxt_training_configuration_service.update(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-yolox-s",
            training_configuration=fxt_training_configuration,
        )

        # Verify task-level configuration was created
        task_level_config = (
            db_session.query(TrainingConfigurationDB)
            .filter(
                TrainingConfigurationDB.project_id == fxt_project.id,
                TrainingConfigurationDB.model_architecture_id.is_(None),
            )
            .first()
        )
        assert task_level_config is not None
        assert task_level_config.configuration_data["dataset_preparation"]["subset_split"]["training"] == 65

        # Verify algo-level configuration was created
        algo_level_config = (
            db_session.query(TrainingConfigurationDB)
            .filter(
                TrainingConfigurationDB.project_id == fxt_project.id,
                TrainingConfigurationDB.model_architecture_id == "object-detection-yolox-s",
            )
            .first()
        )
        assert algo_level_config is not None
        assert algo_level_config.configuration_data["training"]["max_epochs"] == 100

    def test_update_with_preexisting_configuration(
        self,
        fxt_training_configuration: TrainingConfiguration,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Test updating existing task-level and algo-level configurations."""
        # Create initial configurations
        task_level_config = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            model_architecture_id=None,
            configuration_data={
                "dataset_preparation": {"subset_split": {"training": 50, "validation": 30, "test": 20}}
            },
        )
        db_session.add(task_level_config)

        algo_level_config = TrainingConfigurationDB(
            id=str(uuid4()),
            project_id=fxt_project.id,
            model_architecture_id="object-detection-yolox-s",
            configuration_data={"training": {"max_epochs": 50}},
        )
        db_session.add(algo_level_config)
        db_session.flush()

        # Update with new configuration
        fxt_training_configuration_service.update(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-yolox-s",
            training_configuration=fxt_training_configuration,
        )

        # Refresh from database
        db_session.refresh(task_level_config)
        db_session.refresh(algo_level_config)

        # Verify task-level configuration was updated
        assert task_level_config.configuration_data["dataset_preparation"]["subset_split"]["training"] == 65

        # Verify algo-level configuration was updated
        assert algo_level_config.configuration_data["training"]["max_epochs"] == 100

    def test_update_with_multiple_architectures(
        self,
        fxt_training_configuration: TrainingConfiguration,
        fxt_training_configuration_service: TrainingConfigurationService,
        fxt_project: ProjectDB,
        db_session: Session,
    ):
        """Test that different architectures have separate algo-level configurations."""
        # Create configuration for first architecture
        fxt_training_configuration_service.update(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-yolox-s",
            training_configuration=fxt_training_configuration,
        )

        # Create configuration for second architecture with different values
        second_config = TrainingConfiguration(
            task_level_parameters=fxt_training_configuration.task_level_parameters,
            algo_level_parameters=AlgoLevelParameters(
                dataset_preparation=AlgoLevelDatasetPreparationParameters(),
                training=AlgoLevelTrainingParameters(
                    max_epochs=200,
                    early_stopping=EarlyStopping(enable=True, patience=10),
                    learning_rate=0.0005,
                    input_size_width=200,
                    input_size_height=100,
                    allowed_values_input_size=[100, 200, 400],
                ),
            ),
        )
        fxt_training_configuration_service.update(
            project_id=UUID(fxt_project.id),
            model_architecture_id="object-detection-yolox-l",
            training_configuration=second_config,
        )

        # Verify both algo-level configurations exist with correct values
        yolox_s_config = (
            db_session.query(TrainingConfigurationDB)
            .filter(
                TrainingConfigurationDB.project_id == fxt_project.id,
                TrainingConfigurationDB.model_architecture_id == "object-detection-yolox-s",
            )
            .first()
        )
        assert yolox_s_config is not None
        assert yolox_s_config.configuration_data["training"]["max_epochs"] == 100

        yolox_l_config = (
            db_session.query(TrainingConfigurationDB)
            .filter(
                TrainingConfigurationDB.project_id == fxt_project.id,
                TrainingConfigurationDB.model_architecture_id == "object-detection-yolox-l",
            )
            .first()
        )
        assert yolox_l_config is not None
        assert yolox_l_config.configuration_data["training"]["max_epochs"] == 200

        # Verify only one task-level configuration exists
        task_level_configs = (
            db_session.query(TrainingConfigurationDB)
            .filter(
                TrainingConfigurationDB.project_id == fxt_project.id,
                TrainingConfigurationDB.model_architecture_id.is_(None),
            )
            .all()
        )
        assert len(task_level_configs) == 1
