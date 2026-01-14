# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import UUID

import yaml
from datumaro.experimental.fields import Subset
from jsonargparse import ArgumentParser, Namespace
from lightning import Callback
from loguru import logger
from otx.backend.native.engine import OTXEngine
from otx.backend.native.models.base import DataInputParams, OTXModel
from otx.config.data import SamplerConfig, SubsetConfig
from otx.data.dataset import (
    OTXDetectionDataset,
    OTXInstanceSegDataset,
    OTXMulticlassClsDataset,
    OTXMultilabelClsDataset,
)
from otx.data.dataset.base import OTXDataset
from otx.data.module import OTXDataModule
from otx.data.transform_libs.torchvision import TorchVisionTransformLib
from otx.tools.converter import GetiConfigConverter
from otx.types.device import DeviceType as OTXDeviceType
from otx.types.export import OTXExportFormatType
from otx.types.precision import OTXPrecisionType
from otx.types.task import OTXTaskType
from sqlalchemy.orm import Session

from app.core.jobs.models import TrainingJobParams
from app.core.run import ExecutionContext
from app.models import DatasetItemAnnotationStatus, Task, TaskType, TrainingStatus
from app.models.system import DeviceInfo, DeviceType
from app.models.training_configuration.configuration import TrainingConfiguration
from app.services import (
    BaseWeightsService,
    DatasetRevisionService,
    DatasetService,
    ModelRevisionMetadata,
    ModelService,
    TrainingConfigurationService,
)

from .base import Trainer, step
from .subset_assignment import SplitRatios, SubsetAssigner, SubsetService

MODEL_WEIGHTS_PATH = "model_weights_path"


# TODO: Consider adopting some lightweight DI framework
# As the number of constructor dependencies grows and start violating ruff rules, we should evaluate DI frameworks like:
# - dependency-injector (https://python-dependency-injector.ets-labs.org/)
# - injector (https://github.com/python-injector/injector)
# - python-inject (https://github.com/ivankorobkov/python-inject)
@dataclass(frozen=True)
class TrainingDependencies:
    data_dir: Path
    base_weights_service: BaseWeightsService
    subset_service: SubsetService
    dataset_service: DatasetService
    dataset_revision_service: DatasetRevisionService
    model_service: ModelService
    training_configuration_service: TrainingConfigurationService
    subset_assigner: SubsetAssigner
    db_session_factory: Callable[[], AbstractContextManager[Session]]


@dataclass(frozen=True)
class DatasetInfo:
    otx_training_dataset: OTXDataset
    otx_validation_dataset: OTXDataset
    otx_testing_dataset: OTXDataset
    otx_training_subset_config: SubsetConfig
    otx_validation_subset_config: SubsetConfig
    otx_testing_subset_config: SubsetConfig
    revision_id: UUID


class OTXTrainer(Trainer):
    """OTX-specific trainer implementation."""

    def __init__(
        self,
        training_deps: TrainingDependencies,
    ):
        super().__init__()
        self._data_dir = training_deps.data_dir
        self._base_weights_service = training_deps.base_weights_service
        self._subset_service = training_deps.subset_service
        self._dataset_service = training_deps.dataset_service
        self._dataset_revision_service = training_deps.dataset_revision_service
        self._model_service = training_deps.model_service
        self._training_configuration_service = training_deps.training_configuration_service
        self._subset_assigner = training_deps.subset_assigner
        self._db_session_factory = training_deps.db_session_factory

    @step("Prepare Model Weights")
    def prepare_weights(self, training_params: TrainingJobParams) -> Path:
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
    def assign_subsets(self, training_config: TrainingConfiguration, project_id: UUID) -> None:
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
            split_params = training_config.global_parameters.dataset_preparation.subset_split
            target_ratios = SplitRatios(
                train=(split_params.training / 100), val=(split_params.validation / 100), test=(split_params.test / 100)
            )
            adjusted_ratios = current_distribution.compute_adjusted_ratios(target_ratios, len(unassigned_items))
            logger.info("Adjusted subset ratios for unassigned items: {}", adjusted_ratios)

            self.report_progress("Computing optimal subset assignments")
            assignments = self._subset_assigner.assign(unassigned_items, adjusted_ratios)

            # Persist assignments
            self.report_progress("Persisting subset assignments")
            self._subset_service.update_subset_assignments(project_id, assignments)

        self.report_progress(f"Successfully assigned {len(assignments)} items to subsets")

    @step("Prepare Training Configuration")
    def prepare_training_configuration(
        self, training_params: TrainingJobParams, task: Task
    ) -> tuple[TrainingConfiguration, dict]:
        project_id = cast(UUID, training_params.project_id)
        with self._db_session_factory() as db:
            # Load the training configuration from the database
            self._training_configuration_service.set_db_session(db)
            training_config = self._training_configuration_service.get_training_configuration(
                project_id=project_id,
                model_architecture_id=training_params.model_architecture_id,
            )

            # Serialize and persist the configuration in the same YAML format adopted by Geti
            # NOTE: this is a temporary solution to minimize changes in OTX; in the future, after refactoring OTX,
            # we should update this code to build the configuration directly in the format consumed by OTX
            geti_training_config = training_config.model_dump(exclude_none=True)
            geti_training_config["hyper_parameters"] = geti_training_config.pop("hyperparameters")
            geti_training_config["sub_task_type"] = self.__get_otx_task_type_by_task(task)
            geti_config_path = self.__build_model_config_path(self._data_dir, project_id, training_params.model_id)
            geti_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(geti_config_path, "w") as f:
                yaml.dump(geti_training_config, f, default_flow_style=False)
                logger.info("Persisted training configuration at {}", geti_config_path)

            # Convert the configuration to the format adopted by OTX
            converter = GetiConfigConverter()
            otx_training_config = converter.convert(geti_training_config)

            return training_config, otx_training_config

    @step("Create Training Dataset")
    def create_training_dataset(self, project_id: UUID, task: Task, training_config: dict) -> DatasetInfo:
        """Create datasets for training, validation, and testing."""

        def build_subset_config(subset_name: str) -> SubsetConfig:
            subset_cfg_data = training_config["data"][f"{subset_name}_subset"]
            subset_cfg_data["input_size"] = training_config["data"]["input_size"]
            sampler_cfg_data = subset_cfg_data.pop("sampler", {})
            subset_config = SubsetConfig(sampler=SamplerConfig(**sampler_cfg_data), **subset_cfg_data)
            subset_config.transforms = TorchVisionTransformLib.generate(subset_config)
            return subset_config

        with self._db_session_factory() as db:
            self._dataset_service.set_db_session(db)
            self._dataset_revision_service.set_db_session(db)

            # Create a dataset including only the items with user-verified annotations
            logger.info("Creating dataset (dm.Dataset) from the DB items with reviewed annotations")
            dm_dataset = self._dataset_service.get_dm_dataset(
                project_id=project_id, task=task, annotation_status=DatasetItemAnnotationStatus.REVIEWED
            )

            # Extract the subsets (training, validation, testing)
            logger.info("Extracting training, validation, and testing subsets from the dataset")
            dm_training_dataset = dm_dataset.filter_by_subset(Subset.TRAINING)
            dm_validation_dataset = dm_dataset.filter_by_subset(Subset.VALIDATION)
            dm_testing_dataset = dm_dataset.filter_by_subset(Subset.TESTING)

            # Build a SubsetConfig for each subset based on the training configuration.
            # SubsetConfigs define the transformations applied to the subset, as well the parameters for data loaders.
            train_subset_config = build_subset_config("train")
            val_subset_config = build_subset_config("val")
            test_subset_config = build_subset_config("test")

            # Wrap them into OTXDataset instances
            otx_task_type = self.__get_otx_task_type_by_task(task)
            otx_dataset_class = self.__get_otx_dataset_class_by_task_type(otx_task_type)
            logger.info("Preparing {} instances for each subset", otx_dataset_class.__name__)
            otx_training_dataset = otx_dataset_class(
                dm_subset=dm_training_dataset,
                transforms=train_subset_config.transforms,
            )
            otx_validation_dataset = otx_dataset_class(
                dm_subset=dm_validation_dataset,
                transforms=val_subset_config.transforms,
            )
            otx_testing_dataset = otx_dataset_class(
                dm_subset=dm_testing_dataset,
                transforms=test_subset_config.transforms,
            )

            # Store the dataset as a new revision
            logger.info("Saving dataset revision to disk")
            dataset_revision_id = self._dataset_revision_service.save_revision(
                project_id=project_id, dataset=dm_dataset
            )
            logger.info("Dataset revision saved with ID: {}", dataset_revision_id)

            return DatasetInfo(
                otx_training_dataset=otx_training_dataset,
                otx_validation_dataset=otx_validation_dataset,
                otx_testing_dataset=otx_testing_dataset,
                otx_training_subset_config=train_subset_config,
                otx_validation_subset_config=val_subset_config,
                otx_testing_subset_config=test_subset_config,
                revision_id=dataset_revision_id,
            )

    @step("Prepare Model")
    def prepare_model(
        self, training_params: TrainingJobParams, dataset_revision_id: UUID, configuration: TrainingConfiguration
    ) -> None:
        project_id = cast(UUID, training_params.project_id)  # In 'run' we already check that project_id is not None
        logger.info("Preparing model revision (model_id={}, project_id={})", training_params.model_id, project_id)
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            self._model_service.create_revision(
                ModelRevisionMetadata(
                    model_id=training_params.model_id,
                    project_id=project_id,
                    architecture_id=training_params.model_architecture_id,
                    parent_revision_id=training_params.parent_model_revision_id,
                    training_configuration=configuration,
                    dataset_revision_id=dataset_revision_id,
                    training_status=TrainingStatus.NOT_STARTED,
                )
            )

    @step("Train Model")
    def train_model(
        self, training_config: dict, dataset_info: DatasetInfo, weights_path: Path, model_id: UUID, device: DeviceInfo
    ) -> tuple[Path, OTXEngine]:
        """Execute model training."""
        # TODO use weights path to initialize model from pre-downloaded weights
        #  after resolving https://github.com/open-edge-platform/training_extensions/issues/5100
        logger.warning(
            "Argument 'weights_path' (value='{}') is not used in model training yet; "
            "the weights location will be determined internally by OTX",
            weights_path,
        )

        # Build the OTXDataModule
        logger.info("Preparing the OTXDataModule for training (model_id={})", model_id)
        otx_datamodule = OTXDataModule.from_otx_datasets(
            train_dataset=dataset_info.otx_training_dataset,
            val_dataset=dataset_info.otx_validation_dataset,
            test_dataset=dataset_info.otx_testing_dataset,
            train_subset=dataset_info.otx_training_subset_config,
            val_subset=dataset_info.otx_validation_subset_config,
            test_subset=dataset_info.otx_testing_subset_config,
        )

        # Create the OTXModel according to the training configuration
        logger.info("Instantiating the OTXModel for training (model_id={})", model_id)
        model_cfg = training_config["model"]
        model_cfg["init_args"]["label_info"] = otx_datamodule.label_info.label_names
        model_cfg["init_args"]["data_input_params"] = DataInputParams(
            input_size=otx_datamodule.input_size,
            mean=otx_datamodule.input_mean,
            std=otx_datamodule.input_std,
        ).as_dict()
        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(OTXModel, "model", required=False, fail_untyped=False)
        otx_model: OTXModel = model_parser.instantiate_classes(Namespace(model=model_cfg)).get("model")
        if hasattr(otx_model, "tile_config"):
            otx_model.tile_config = otx_datamodule.tile_config

        # Set up the OTXEngine
        logger.info("Initializing the OTXEngine for training (model_id={})", model_id)
        otx_device_type = OTXDeviceType.gpu if device.type is DeviceType.CUDA else OTXDeviceType(device.type)
        otx_engine = OTXEngine(
            model=otx_model,
            data=otx_datamodule,
            work_dir=f"./otx-workspace-{model_id}",
            device=otx_device_type,
        )

        # Set up the callbacks
        callbacks_cfg = training_config.get("callbacks", [])
        for cb_cfg in callbacks_cfg:
            if "init_args" in cb_cfg and "dirpath" in cb_cfg["init_args"]:
                cb_cfg["init_args"]["dirpath"] = otx_engine.work_dir
        parser = ArgumentParser()
        parser.add_argument("--callbacks", type=list[Callback])
        parsed_callbacks_cfg = parser.parse_object({"callbacks": callbacks_cfg})
        callbacks_list = parser.instantiate_classes(parsed_callbacks_cfg).get("callbacks", [])

        # Start training
        logger.info("Starting the training loop (model_id={})", model_id)
        train_kwargs = {"devices": [device.index]} if device.type is not DeviceType.CPU and device.index else {}
        otx_engine.train(
            max_epochs=training_config["max_epochs"],
            precision=training_config["precision"],
            callbacks=callbacks_list,
            **train_kwargs,
        )
        trained_model_path = Path(otx_engine.work_dir) / "best_checkpoint.ckpt"
        logger.info("Model training completed. Trained model saved at {}", trained_model_path)
        return trained_model_path, otx_engine

    @step("Evaluate Model")
    def evaluate_model(self, otx_engine: OTXEngine, model_checkpoint_path: Path) -> None:
        """Evaluate the trained model on the testing set"""
        # TODO evaluate with custom metric and save evaluation results (#4869)
        logger.info("Evaluating the model on the testing set...")
        otx_engine.test(checkpoint=model_checkpoint_path, datamodule=otx_engine._datamodule)

    @step("Export Model to OpenVINO")
    def export_model(self, otx_engine: OTXEngine, model_checkpoint_path: Path) -> Path:
        """Export the trained model to OpenVINO format"""
        logger.info("Exporting the model to OpenVINO format with FP16 precision...")
        exported_model_path = otx_engine.export(
            checkpoint=model_checkpoint_path,
            export_format=OTXExportFormatType.OPENVINO,
            export_precision=OTXPrecisionType.FP16,
        )
        logger.info(f"Model exported to: {exported_model_path}")
        return exported_model_path

    @step("Store Model Artifacts")
    def store_model_artifacts(
        self,
        training_params: TrainingJobParams,
        otx_work_dir: Path,
        trained_model_path: Path,
        exported_model_path: Path,
    ) -> None:
        """Store the selected training artifacts (model binary, metrics, ...) and cleanup the rest"""
        model_dir = self.__base_model_path(
            data_dir=self._data_dir,
            project_id=cast(UUID, training_params.project_id),
            model_id=training_params.model_id,
        )
        # Store the Pytorch model checkpoint and the OpenVINO exported model files
        model_ckpt_source_path = trained_model_path
        model_xml_source_path = exported_model_path.with_suffix(".xml")
        model_bin_source_path = exported_model_path.with_suffix(".bin")
        model_ckpt_dest_path = model_dir / "model.ckpt"
        model_xml_dest_path = model_dir / "model.xml"
        model_bin_dest_path = model_dir / "model.bin"
        shutil.copyfile(model_ckpt_source_path, model_ckpt_dest_path)
        shutil.copyfile(model_xml_source_path, model_xml_dest_path)
        shutil.copyfile(model_bin_source_path, model_bin_dest_path)
        logger.info("Stored model artifacts at {}", model_dir)

        # Store the metrics
        metrics_source_path = otx_work_dir / "csv"
        metrics_dest_path = model_dir / "metrics"
        if metrics_source_path.exists():
            shutil.move(metrics_source_path, metrics_dest_path)
            logger.info("Stored training metrics at {}", metrics_dest_path)

        # Cleanup the OTX work directory
        shutil.rmtree(otx_work_dir)
        logger.info("Cleaned up OTX work directory at {}", otx_work_dir)

    def run(self, ctx: ExecutionContext) -> None:
        self._ctx = ctx
        training_params = self._get_training_params(ctx)
        project_id = training_params.project_id
        if project_id is None:
            raise ValueError("Project ID must be provided in training parameters")
        task = training_params.task

        weights_path = self.prepare_weights(training_params=training_params)
        training_config, otx_training_config = self.prepare_training_configuration(
            training_params=training_params, task=task
        )
        self.assign_subsets(training_config=training_config, project_id=project_id)
        dataset_info = self.create_training_dataset(
            project_id=project_id, task=task, training_config=otx_training_config
        )
        self.prepare_model(
            training_params=training_params, dataset_revision_id=dataset_info.revision_id, configuration=training_config
        )
        trained_model_path, otx_engine = self.train_model(
            training_config=otx_training_config,
            dataset_info=dataset_info,
            weights_path=weights_path,
            model_id=training_params.model_id,
            device=training_params.device,
        )
        self.evaluate_model(otx_engine=otx_engine, model_checkpoint_path=trained_model_path)
        exported_model_path = self.export_model(otx_engine=otx_engine, model_checkpoint_path=trained_model_path)
        self.store_model_artifacts(
            otx_work_dir=Path(otx_engine.work_dir),
            training_params=training_params,
            trained_model_path=trained_model_path,
            exported_model_path=exported_model_path,
        )

    @staticmethod
    def __base_model_path(data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return data_dir / "projects" / str(project_id) / "models" / str(model_id)

    @classmethod
    def __build_model_weights_path(cls, data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return cls.__base_model_path(data_dir, project_id, model_id) / "model.pth"

    @classmethod
    def __build_model_config_path(cls, data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return cls.__base_model_path(data_dir, project_id, model_id) / "config.yaml"

    @classmethod
    def __get_otx_task_type_by_task(cls, task: Task) -> OTXTaskType:
        """Map internal Task to OTXTaskType."""
        match task.task_type:
            case TaskType.CLASSIFICATION:
                if task.exclusive_labels:
                    return OTXTaskType.MULTI_CLASS_CLS
                return OTXTaskType.MULTI_LABEL_CLS
            case TaskType.DETECTION:
                return OTXTaskType.DETECTION
            case TaskType.INSTANCE_SEGMENTATION:
                return OTXTaskType.INSTANCE_SEGMENTATION
            case _:
                raise ValueError(f"Unsupported task type: {task.task_type}")

    @classmethod
    def __get_otx_dataset_class_by_task_type(cls, otx_task_type: OTXTaskType) -> type[OTXDataset]:
        """Get the OTXDataset class corresponding to the given OTXTaskType."""
        otx_task_type_to_class: dict[OTXTaskType, type[OTXDataset]] = {
            OTXTaskType.MULTI_CLASS_CLS: OTXMulticlassClsDataset,
            OTXTaskType.MULTI_LABEL_CLS: OTXMultilabelClsDataset,
            OTXTaskType.DETECTION: OTXDetectionDataset,
            OTXTaskType.INSTANCE_SEGMENTATION: OTXInstanceSegDataset,
        }
        try:
            return otx_task_type_to_class[otx_task_type]
        except KeyError:
            raise ValueError(f"Unsupported OTX task type: {otx_task_type}")
