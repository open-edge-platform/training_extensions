# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from otx.tools.converter import GetiConfigConverter
from otx.types.export import OTXExportFormatType
from otx.types.precision import OTXPrecisionType

from app.core.run import ExecutionContext
from app.models import DatasetItemAnnotationStatus, DatasetItemSubset, Task, TaskType, TrainingStatus
from app.models.training_configuration.configuration import (
    GlobalDatasetPreparationParameters,
    PartialGlobalParameters,
    PartialTrainingConfiguration,
    SubsetSplit,
    TrainingConfiguration,
)
from app.services import (
    DatasetRevisionService,
    DatasetService,
    ModelRevisionMetadata,
    ModelService,
    TrainingConfigurationService,
)
from app.services.base_weights_service import BaseWeightsService
from app.services.training.models import TrainingParams
from app.services.training.otx_trainer import OTXTrainer, TrainingDependencies
from app.services.training.subset_assignment import (
    DatasetItemWithLabels,
    SubsetAssigner,
    SubsetAssignment,
    SubsetDistribution,
    SubsetService,
)


@pytest.fixture
def fxt_weights_service() -> Mock:
    """Create a mock BaseWeightsService."""
    return Mock(spec=BaseWeightsService)


@pytest.fixture
def fxt_subset_service() -> Mock:
    """Mock SubsetService for testing."""
    return Mock(spec=SubsetService)


@pytest.fixture
def fxt_assigner() -> Mock:
    """Mock SubsetAssigner for testing."""
    return Mock(spec=SubsetAssigner)


@pytest.fixture
def fxt_dataset_service() -> Mock:
    """Mock DatasetService for testing."""
    return Mock(spec=DatasetService)


@pytest.fixture
def fxt_dataset_revision_service() -> Mock:
    """Mock DatasetRevisionService for testing."""
    return Mock(spec=DatasetRevisionService)


@pytest.fixture
def fxt_model_service() -> Mock:
    """Mock ModelService for testing."""
    return Mock(spec=ModelService)


@pytest.fixture
def fxt_training_configuration_service() -> Mock:
    """Mock TrainingConfigurationService for testing."""
    return Mock(spec=TrainingConfigurationService)


@pytest.fixture
def fxt_otx_trainer(
    tmp_path: Path,
    fxt_weights_service: Mock,
    fxt_subset_service: Mock,
    fxt_assigner: Mock,
    fxt_dataset_service: Mock,
    fxt_dataset_revision_service: Mock,
    fxt_model_service: Mock,
    fxt_training_configuration_service: Mock,
    fxt_db_session_factory: Callable,
) -> Callable[[], OTXTrainer]:
    """Create an OTXTrainer instance."""

    def create_otx_trainer() -> OTXTrainer:
        otx_trainer = OTXTrainer(
            TrainingDependencies(
                data_dir=tmp_path,
                base_weights_service=fxt_weights_service,
                subset_service=fxt_subset_service,
                subset_assigner=fxt_assigner,
                dataset_service=fxt_dataset_service,
                dataset_revision_service=fxt_dataset_revision_service,
                model_service=fxt_model_service,
                training_configuration_service=fxt_training_configuration_service,
                db_session_factory=fxt_db_session_factory,
            )
        )
        execution_ctx = Mock(spec=ExecutionContext)
        execution_ctx.report = Mock()
        execution_ctx.heartbeat = Mock()
        otx_trainer._ctx = execution_ctx
        return otx_trainer

    return create_otx_trainer


class TestOTXTrainerPrepareWeights:
    """Tests for the OTXTrainer.prepare_weights method."""

    def test_prepare_weights_without_parent_model(
        self,
        fxt_weights_service: Mock,
        fxt_otx_trainer: Callable[[], OTXTrainer],
    ):
        """Test preparing weights when no parent model revision ID is provided."""
        # Arrange
        training_params = TrainingParams(
            model_architecture_id="Object_Detection_YOLOX_S",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=None,
        )
        otx_trainer = fxt_otx_trainer()

        expected_weights_path = Path("/path/to/weights.pth")
        fxt_weights_service.get_local_weights_path.return_value = expected_weights_path

        # Act
        weights_path = otx_trainer.prepare_weights(training_params)

        # Assert
        assert weights_path == expected_weights_path
        fxt_weights_service.get_local_weights_path.assert_called_once_with(
            task=TaskType.DETECTION, model_manifest_id="Object_Detection_YOLOX_S"
        )

    def test_prepare_weights_with_parent_model(
        self,
        tmp_path: Path,
        fxt_otx_trainer: Callable[[], OTXTrainer],
    ):
        """Test preparing weights when parent model revision ID is provided."""
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        training_params = TrainingParams(
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=parent_model_revision_id,
        )
        expected_weights_path = (
            tmp_path / "projects" / str(project_id) / "models" / str(parent_model_revision_id) / "model.pth"
        )
        expected_weights_path.parent.mkdir(parents=True, exist_ok=True)
        expected_weights_path.touch()
        otx_trainer = fxt_otx_trainer()

        # Act
        weights_path = otx_trainer.prepare_weights(training_params)

        # Assert
        assert weights_path == expected_weights_path

    def test_prepare_weights_with_parent_model_no_file_raises_error(
        self,
        tmp_path: Path,
        fxt_otx_trainer: Callable[[], OTXTrainer],
    ):
        """Test that FileNotFoundError is raised when parent model weights file is missing."""
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        training_params = TrainingParams(
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=parent_model_revision_id,
        )
        expected_weights_path = (
            tmp_path / "projects" / str(project_id) / "models" / str(parent_model_revision_id) / "model.pth"
        )
        otx_trainer = fxt_otx_trainer()

        # Act
        with pytest.raises(FileNotFoundError, match=f"Parent model weights not found at {expected_weights_path}"):
            otx_trainer.prepare_weights(training_params)

    def test_prepare_weights_with_parent_model_no_project_id_raises_error(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
    ):
        """Test that ValueError is raised when parent model revision ID is provided without project ID."""
        # Arrange
        training_params = TrainingParams(
            model_architecture_id="Object_Detection_YOLOX_S",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=uuid4(),
            project_id=None,
        )
        otx_trainer = fxt_otx_trainer()

        # Act & Assert
        with pytest.raises(ValueError, match="Project ID must be provided for parent model weights preparation"):
            otx_trainer.prepare_weights(training_params)


class TestOTXTrainerPrepareTrainingConfiguration:
    """Tests for the OTXTrainer.prepare_training_configuration method."""

    def test_prepare_training_configuration(self, fxt_otx_trainer: Callable[[], OTXTrainer]) -> None:
        # Arrange
        project_id = uuid4()
        parent_model_revision_id = uuid4()
        training_params = TrainingParams(
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task=Task(task_type=TaskType.DETECTION),
            parent_model_revision_id=parent_model_revision_id,
        )
        otx_trainer = fxt_otx_trainer()
        mock_training_config = Mock(spec=TrainingConfiguration)
        mock_training_config.model_dump.return_value = {"key1": "value1", "hyperparameters": {"lr": 0.001}}
        otx_trainer._training_configuration_service.get_training_configuration.return_value = mock_training_config  # type: ignore[attr-defined]
        mock_otx_training_config = {"key2": "value2"}
        with patch.object(GetiConfigConverter, "convert", return_value=mock_otx_training_config) as mock_convert:
            # Act
            training_config, otx_training_config = otx_trainer.prepare_training_configuration(
                training_params=training_params, task=Task(task_type=TaskType.DETECTION)
            )

        # Assert
        otx_trainer._training_configuration_service.get_training_configuration.assert_called_once_with(  # type: ignore[attr-defined]
            project_id=training_params.project_id,
            model_architecture_id=training_params.model_architecture_id,
        )
        mock_convert.assert_called_once_with(
            {"key1": "value1", "hyper_parameters": {"lr": 0.001}, "sub_task_type": "DETECTION"}
        )
        assert training_config == mock_training_config
        assert otx_training_config == mock_otx_training_config


class TestOTXTrainerAssignSubsets:
    """Tests for the OTXTrainer.assign_subsets method."""

    def test_assign_subsets_with_unassigned_items(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        fxt_subset_service: Mock,
        fxt_assigner: Mock,
    ):
        """Test assigning subsets when unassigned items exist."""
        # Arrange
        otx_trainer = fxt_otx_trainer()
        project_id = uuid4()
        label_1, label_2, label_3 = uuid4(), uuid4(), uuid4()

        # Create mock unassigned items
        unassigned_items = [
            DatasetItemWithLabels(item_id=uuid4(), labels={label_1, label_2}),
            DatasetItemWithLabels(item_id=uuid4(), labels={label_1}),
            DatasetItemWithLabels(item_id=uuid4(), labels={label_1, label_2, label_3}),
        ]
        fxt_subset_service.get_unassigned_items_with_labels.return_value = unassigned_items

        # Mock configuration
        training_config = PartialTrainingConfiguration(  # type: ignore[call-arg]
            global_parameters=PartialGlobalParameters(
                dataset_preparation=GlobalDatasetPreparationParameters(subset_split=SubsetSplit())
            ),
        )

        # Mock current distribution
        current_distribution = SubsetDistribution(
            counts={
                DatasetItemSubset.TRAINING: 10,
                DatasetItemSubset.VALIDATION: 3,
                DatasetItemSubset.TESTING: 2,
            }
        )
        fxt_subset_service.get_subset_distribution.return_value = current_distribution

        # Mock assignments
        expected_assignments = [
            SubsetAssignment(item_id=unassigned_items[0].item_id, subset=DatasetItemSubset.TRAINING),
            SubsetAssignment(item_id=unassigned_items[1].item_id, subset=DatasetItemSubset.VALIDATION),
            SubsetAssignment(item_id=unassigned_items[2].item_id, subset=DatasetItemSubset.TESTING),
        ]
        fxt_assigner.assign.return_value = expected_assignments

        # Act
        otx_trainer.assign_subsets(training_config=training_config, project_id=project_id)

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id)
        fxt_subset_service.get_subset_distribution.assert_called_once_with(project_id)
        fxt_assigner.assign.assert_called_once()
        fxt_subset_service.update_subset_assignments.assert_called_once_with(project_id, expected_assignments)

    def test_assign_subsets_with_no_unassigned_items(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        fxt_subset_service: Mock,
        fxt_assigner: Mock,
    ):
        """Test assigning subsets when no unassigned items exist."""
        # Arrange
        project_id = uuid4()
        fxt_subset_service.get_unassigned_items_with_labels.return_value = []
        otx_trainer = fxt_otx_trainer()
        training_config = Mock(spec=TrainingConfiguration)

        # Act
        otx_trainer.assign_subsets(training_config=training_config, project_id=project_id)

        # Assert
        fxt_subset_service.get_unassigned_items_with_labels.assert_called_once_with(project_id)
        fxt_subset_service.get_subset_distribution.assert_not_called()
        fxt_assigner.assign.assert_not_called()
        fxt_subset_service.update_subset_assignments.assert_not_called()


class TestOTXTrainerCreateTrainingDataset:
    """Tests for the OTXTrainer.create_training_dataset method."""

    def test_create_training_dataset_success(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        fxt_dataset_service: Mock,
        fxt_dataset_revision_service: Mock,
    ):
        """Test successful creation of training, validation, and testing datasets."""
        # Arrange
        project_id = uuid4()
        task = Task(task_type=TaskType.DETECTION, exclusive_labels=True)
        otx_trainer = fxt_otx_trainer()

        # Mock the Datumaro dataset
        mock_dm_dataset = Mock()
        fxt_dataset_service.get_dm_dataset.return_value = mock_dm_dataset

        # Mock filtered subsets
        mock_training_subset = Mock()
        mock_validation_subset = Mock()
        mock_testing_subset = Mock()

        mock_dm_dataset.filter_by_subset.side_effect = [
            mock_training_subset,
            mock_validation_subset,
            mock_testing_subset,
        ]

        # Mock dataset revision saving
        dataset_revision_id = uuid4()
        fxt_dataset_revision_service.save_revision.return_value = dataset_revision_id

        # Create a training configuration matching the expected structure
        training_config = {
            "data": {
                "input_size": (640, 640),
                "train_subset": {
                    "batch_size": 8,
                    "num_workers": 4,
                    "sampler": {"class_path": "torch.utils.data.RandomSampler"},
                    "transforms": [
                        {"class_path": "torchvision.transforms.v2.RandomHorizontalFlip", "init_args": {"p": 0.5}}
                    ],
                },
                "val_subset": {
                    "batch_size": 4,
                    "num_workers": 2,
                    "sampler": {"class_path": "torch.utils.data.RandomSampler"},
                    "transforms": [],
                },
                "test_subset": {
                    "batch_size": 2,
                    "num_workers": 1,
                    "sampler": {"class_path": "torch.utils.data.RandomSampler"},
                    "transforms": [],
                },
            }
        }

        # Mock TorchVisionTransformLib.generate to return mock transforms
        mock_train_transforms = [Mock()]
        mock_val_transforms = [Mock()]
        mock_test_transforms = [Mock()]

        with patch("app.services.training.otx_trainer.TorchVisionTransformLib.generate") as mock_generate:
            mock_generate.side_effect = [mock_train_transforms, mock_val_transforms, mock_test_transforms]

            # Mock the __get_otx_dataset_class_by_task_type method to return a proper mock class
            mock_dataset_class = Mock()
            mock_dataset_class.__name__ = "OTXDetectionDataset"

            mock_otx_training_dataset = Mock()
            mock_otx_validation_dataset = Mock()
            mock_otx_testing_dataset = Mock()

            mock_dataset_class.side_effect = [
                mock_otx_training_dataset,
                mock_otx_validation_dataset,
                mock_otx_testing_dataset,
            ]

            with patch.object(
                otx_trainer, "_OTXTrainer__get_otx_dataset_class_by_task_type", return_value=mock_dataset_class
            ):
                # Act
                dataset_info = otx_trainer.create_training_dataset(
                    project_id=project_id,
                    task=task,
                    training_config=training_config,
                )

        # Assert
        # Verify get_dm_dataset was called with correct parameters
        fxt_dataset_service.get_dm_dataset.assert_called_once_with(
            project_id=project_id,
            task=task,
            annotation_status=DatasetItemAnnotationStatus.REVIEWED,
        )

        # Verify subsets were filtered for train, val, and test
        assert mock_dm_dataset.filter_by_subset.call_count == 3

        # Verify transforms were generated for each subset
        assert mock_generate.call_count == 3

        # Verify OTXDataset was instantiated three times with correct parameters
        assert mock_dataset_class.call_count == 3

        # Verify first call (training dataset)
        train_call = mock_dataset_class.call_args_list[0]
        assert train_call.kwargs["dm_subset"] == mock_training_subset
        assert train_call.kwargs["transforms"] == mock_train_transforms

        # Verify second call (validation dataset)
        val_call = mock_dataset_class.call_args_list[1]
        assert val_call.kwargs["dm_subset"] == mock_validation_subset
        assert val_call.kwargs["transforms"] == mock_val_transforms

        # Verify third call (testing dataset)
        test_call = mock_dataset_class.call_args_list[2]
        assert test_call.kwargs["dm_subset"] == mock_testing_subset
        assert test_call.kwargs["transforms"] == mock_test_transforms

        # Verify the returned DatasetInfo
        assert dataset_info.otx_training_dataset == mock_otx_training_dataset
        assert dataset_info.otx_validation_dataset == mock_otx_validation_dataset
        assert dataset_info.otx_testing_dataset == mock_otx_testing_dataset
        assert dataset_info.revision_id == dataset_revision_id

        # Verify dataset revision was saved
        fxt_dataset_revision_service.save_revision.assert_called_once_with(
            project_id=project_id,
            dataset=mock_dm_dataset,
        )

        # Verify SubsetConfig objects were created correctly
        assert dataset_info.otx_training_subset_config.batch_size == 8
        assert dataset_info.otx_training_subset_config.num_workers == 4
        assert dataset_info.otx_training_subset_config.transforms == mock_train_transforms

        assert dataset_info.otx_validation_subset_config.batch_size == 4
        assert dataset_info.otx_validation_subset_config.num_workers == 2
        assert dataset_info.otx_validation_subset_config.transforms == mock_val_transforms

        assert dataset_info.otx_testing_subset_config.batch_size == 2
        assert dataset_info.otx_testing_subset_config.num_workers == 1
        assert dataset_info.otx_testing_subset_config.transforms == mock_test_transforms


class TestOTXTrainerPrepareModel:
    """Tests for the OTXTrainer.prepare_model method."""

    def test_prepare_model(
        self,
        tmp_path: Path,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        fxt_model_service: Mock,
    ):
        """Test successful preparation of model metadata."""
        # Arrange
        project_id = uuid4()
        model_id = uuid4()
        model_architecture_id = "Custom_Image_Classification_EfficientNet-B0"
        training_params = TrainingParams(
            model_id=model_id,
            project_id=project_id,
            model_architecture_id=model_architecture_id,
            task=Task(task_type=TaskType.CLASSIFICATION, exclusive_labels=True),
            parent_model_revision_id=None,
        )
        dataset_revision_id = uuid4()
        otx_trainer = fxt_otx_trainer()

        training_config = PartialTrainingConfiguration(model_manifest_id=model_architecture_id)  # type: ignore

        # Act
        otx_trainer.prepare_model(training_params, dataset_revision_id, training_config)

        # Assert
        fxt_model_service.create_revision.assert_called_once_with(
            ModelRevisionMetadata(
                model_id=model_id,
                project_id=project_id,
                architecture_id=model_architecture_id,
                parent_revision_id=None,
                training_status=TrainingStatus.NOT_STARTED,
                dataset_revision_id=dataset_revision_id,
                training_configuration=training_config,
            )
        )


class TestOTXTrainerTrainModel:
    """Tests for the OTXTrainer.train_model method."""

    def test_train_model(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        tmp_path: Path,
    ):
        """Test successful model training."""
        # Arrange
        otx_trainer = fxt_otx_trainer()
        model_id = uuid4()

        # Mock DatasetInfo
        mock_dataset_info = Mock()
        mock_training_dataset = Mock()
        mock_validation_dataset = Mock()
        mock_testing_dataset = Mock()
        mock_training_subset_config = Mock()
        mock_validation_subset_config = Mock()
        mock_testing_subset_config = Mock()

        mock_dataset_info.otx_training_dataset = mock_training_dataset
        mock_dataset_info.otx_validation_dataset = mock_validation_dataset
        mock_dataset_info.otx_testing_dataset = mock_testing_dataset
        mock_dataset_info.otx_training_subset_config = mock_training_subset_config
        mock_dataset_info.otx_validation_subset_config = mock_validation_subset_config
        mock_dataset_info.otx_testing_subset_config = mock_testing_subset_config

        # Mock weights path
        weights_path = tmp_path / "weights.pth"
        weights_path.touch()

        # Create training configuration
        training_config = {
            "model": {
                "class_path": "otx.backend.native.models.detection.yolox.YOLOXModel",
                "init_args": {
                    "model_name": "yolox_tiny",
                },
            },
            "max_epochs": 10,
            "precision": "32",
            "callbacks": [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": "./checkpoints",
                        "filename": "best_checkpoint",
                    },
                }
            ],
        }

        # Mock OTXDataModule
        mock_datamodule = Mock()
        mock_datamodule.label_info.label_names = ["cat", "dog"]
        mock_datamodule.input_size = (640, 640)
        mock_datamodule.input_mean = [0.485, 0.456, 0.406]
        mock_datamodule.input_std = [0.229, 0.224, 0.225]
        mock_datamodule.tile_config = None

        # Mock OTXModel
        mock_otx_model = Mock()

        # Mock OTXEngine
        mock_otx_engine = Mock()
        mock_otx_engine.work_dir = str(tmp_path / f"otx-workspace-{model_id}")
        Path(mock_otx_engine.work_dir).mkdir(parents=True)

        # Create expected checkpoint file
        expected_checkpoint_path = Path(mock_otx_engine.work_dir) / "best_checkpoint.ckpt"
        expected_checkpoint_path.touch()

        with patch("app.services.training.otx_trainer.OTXDataModule.from_otx_datasets") as mock_datamodule_factory:
            mock_datamodule_factory.return_value = mock_datamodule

            with patch("app.services.training.otx_trainer.ArgumentParser") as mock_parser_class:
                mock_parser = Mock()
                mock_parser_class.return_value = mock_parser

                # Mock model instantiation
                mock_model_namespace = Mock()
                mock_model_namespace.get.return_value = mock_otx_model
                mock_parser.instantiate_classes.return_value = mock_model_namespace

                with patch("app.services.training.otx_trainer.OTXEngine") as mock_engine_class:
                    mock_engine_class.return_value = mock_otx_engine

                    # Act
                    trained_model_path, returned_engine = otx_trainer.train_model(
                        training_config=training_config,
                        dataset_info=mock_dataset_info,
                        weights_path=weights_path,
                        model_id=model_id,
                    )

        # Assert
        # Verify OTXDataModule was created correctly
        mock_datamodule_factory.assert_called_once_with(
            train_dataset=mock_training_dataset,
            val_dataset=mock_validation_dataset,
            test_dataset=mock_testing_dataset,
            train_subset=mock_training_subset_config,
            val_subset=mock_validation_subset_config,
            test_subset=mock_testing_subset_config,
        )

        # Verify OTXEngine was initialized
        mock_engine_class.assert_called_once()
        engine_call_kwargs = mock_engine_class.call_args.kwargs
        assert engine_call_kwargs["model"] == mock_otx_model
        assert engine_call_kwargs["data"] == mock_datamodule
        assert engine_call_kwargs["work_dir"] == f"./otx-workspace-{model_id}"

        # Verify training was started
        mock_otx_engine.train.assert_called_once()
        train_call_kwargs = mock_otx_engine.train.call_args.kwargs
        assert train_call_kwargs["max_epochs"] == 10
        assert train_call_kwargs["precision"] == "32"
        assert "callbacks" in train_call_kwargs

        # Verify return values
        assert trained_model_path == expected_checkpoint_path
        assert returned_engine == mock_otx_engine


class TestOTXTrainerEvaluateModel:
    """Tests for the OTXTrainer.evaluate_model method."""

    def test_evaluate_model(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        tmp_path: Path,
    ):
        """Test successful model evaluation on the testing set."""
        # Arrange
        otx_trainer = fxt_otx_trainer()
        mock_otx_engine = Mock()
        mock_datamodule = Mock()
        mock_otx_engine._datamodule = mock_datamodule
        model_checkpoint_path = tmp_path / "best_checkpoint.ckpt"
        model_checkpoint_path.touch()

        # Act
        otx_trainer.evaluate_model(otx_engine=mock_otx_engine, model_checkpoint_path=model_checkpoint_path)

        # Assert
        mock_otx_engine.test.assert_called_once_with(checkpoint=model_checkpoint_path, datamodule=mock_datamodule)


class TestOTXTrainerExportModel:
    """Tests for the OTXTrainer.export_model method."""

    def test_export_model(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        tmp_path: Path,
    ):
        """Test successful model export to OpenVINO format."""
        # Arrange
        otx_trainer = fxt_otx_trainer()
        mock_otx_engine = Mock()
        model_checkpoint_path = tmp_path / "best_checkpoint.ckpt"
        model_checkpoint_path.touch()
        expected_export_path = tmp_path / "exported_model"
        mock_otx_engine.export.return_value = expected_export_path

        # Act
        exported_path = otx_trainer.export_model(
            otx_engine=mock_otx_engine, model_checkpoint_path=model_checkpoint_path
        )

        # Assert
        mock_otx_engine.export.assert_called_once_with(
            checkpoint=model_checkpoint_path,
            export_format=OTXExportFormatType.OPENVINO,
            export_precision=OTXPrecisionType.FP16,
        )
        assert exported_path == expected_export_path


class TestOTXTrainerStoreModelArtifacts:
    """Tests for the OTXTrainer.store_model_artifacts method."""

    def test_store_model_artifacts(
        self,
        fxt_otx_trainer: Callable[[], OTXTrainer],
        tmp_path: Path,
    ):
        """Test successful storing of model artifacts and cleanup."""
        # Arrange
        otx_trainer = fxt_otx_trainer()
        project_id = uuid4()
        model_id = uuid4()

        training_params = TrainingParams(
            model_id=model_id,
            project_id=project_id,
            model_architecture_id="Object_Detection_YOLOX_S",
            task=Task(task_type=TaskType.DETECTION),
        )

        # Create model directory structure
        model_dir = tmp_path / "projects" / str(project_id) / "models" / str(model_id)
        model_dir.mkdir(parents=True)

        # Create OTX work directory with artifacts
        otx_work_dir = tmp_path / f"otx-workspace-{model_id}"
        otx_work_dir.mkdir(parents=True)

        # Create model checkpoint
        trained_model_path = otx_work_dir / "best_checkpoint.ckpt"
        trained_model_path.write_text("checkpoint content")

        # Create exported model files
        exported_model_path = otx_work_dir / "exported_model.pth"
        model_xml_path = exported_model_path.with_suffix(".xml")
        model_bin_path = exported_model_path.with_suffix(".bin")
        model_xml_path.write_text("xml content")
        model_bin_path.write_text("bin content")

        # Create metrics directory
        metrics_dir = otx_work_dir / "csv"
        metrics_dir.mkdir()
        (metrics_dir / "metrics.csv").write_text("epoch,loss\n1,0.5\n")

        # Act
        otx_trainer.store_model_artifacts(
            training_params=training_params,
            otx_work_dir=otx_work_dir,
            trained_model_path=trained_model_path,
            exported_model_path=exported_model_path,
        )

        # Assert
        # Check checkpoint was copied
        assert (model_dir / "model.ckpt").exists()
        assert (model_dir / "model.ckpt").read_text() == "checkpoint content"

        # Check OpenVINO files were copied
        assert (model_dir / "model.xml").exists()
        assert (model_dir / "model.xml").read_text() == "xml content"
        assert (model_dir / "model.bin").exists()
        assert (model_dir / "model.bin").read_text() == "bin content"

        # Check metrics were moved
        assert (model_dir / "metrics").exists()
        assert (model_dir / "metrics" / "metrics.csv").exists()
        assert (model_dir / "metrics" / "metrics.csv").read_text() == "epoch,loss\n1,0.5\n"

        # Check OTX work directory was cleaned up
        assert not otx_work_dir.exists()
