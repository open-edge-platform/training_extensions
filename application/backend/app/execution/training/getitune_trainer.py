# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import shutil
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import polars as pl
import yaml
from datumaro.experimental import Dataset
from datumaro.experimental.fields import Subset
from getitune import TaskType
from getitune.backend.lightning.models.base import DataInputParams, LightningModel
from getitune.backend.openvino.engine import OVEngine
from getitune.backend.ultralytics.models.base import UltralyticsModel
from getitune.config.data import SamplerConfig, SubsetConfig
from getitune.data.dataset.base import VisionDataset
from getitune.data.entity.utils import detect_storage_dtype
from getitune.data.factory import TransformLibFactory
from getitune.data.module import DataModule
from getitune.engine import create_engine
from getitune.engine.engine import Engine
from getitune.types.device import DeviceType as GetiTuneDeviceType
from getitune.types.export import ExportFormat
from getitune.types.precision import Precision
from jsonargparse import ArgumentParser, Namespace
from lightning import Callback
from loguru import logger
from sqlalchemy.orm import Session

from app.core.jobs.exec.exceptions import CancelledExc
from app.datumaro_converter import SampleMode
from app.execution.base import Execution, ExecutionErr, step
from app.execution.common.geti_config_converter import GetiConfigConverter
from app.execution.common.getitune_converters import (
    convert_metrics,
    get_getitune_dataset_class_by_task_type,
    get_getitune_task_type_by_task,
    get_metric_by_task,
)
from app.models import (
    DatasetItemAnnotationStatus,
    DatasetItemSubset,
    EvaluationResult,
    Task,
    TrainingJobParams,
    TrainingStatus,
)
from app.models.model_revision import ModelFormat, ModelPrecision
from app.models.system import DeviceInfo, DeviceType
from app.models.training_configuration.configuration import TrainingConfiguration
from app.models.training_configuration.dataset_preparation import Filtering
from app.services import (
    BaseWeightsService,
    DatasetRevisionService,
    DatasetService,
    ModelRevisionMetadata,
    ModelService,
    SplitRatios,
    SubsetAssigner,
    SubsetService,
    TrainingConfigurationService,
)

from .progress import TrainingProgressCallback

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
    getitune_training_dataset: VisionDataset
    getitune_validation_dataset: VisionDataset
    getitune_testing_dataset: VisionDataset
    getitune_training_subset_config: SubsetConfig
    getitune_validation_subset_config: SubsetConfig
    getitune_testing_subset_config: SubsetConfig
    revision_id: UUID


@dataclass(frozen=True)
class ExportedModels:
    openvino_model_path: Path
    onnx_model_path: Path


@dataclass(frozen=True)
class ModelVariantDescriptor:
    """Describes a single trained/exported model variant to be evaluated and stored.

    Attributes:
        id: The UUID of the variant record in the database.
        path: Path to the variant's main file (.ckpt for PyTorch, .xml for OpenVINO IR, .onnx for ONNX).
        format: The format of the variant.
    """

    id: UUID
    path: Path
    format: ModelFormat


class GetiTuneTrainer(Execution[TrainingJobParams]):
    """getitune-specific trainer implementation."""

    params_type = TrainingJobParams

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
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            parent_model_revision_id = training_params.parent_model_revision_id
            task = training_params.task
            model_architecture_id = training_params.model_architecture_id
            project_id = training_params.project_id
            if parent_model_revision_id is None:
                return self._base_weights_service.get_local_weights_path(
                    task=task.task_type, model_manifest_id=model_architecture_id
                )

            parent_variants = self._model_service.get_model_variants(
                project_id=project_id, model_id=parent_model_revision_id
            )
            if not parent_variants:
                raise ExecutionErr(
                    "Can't start training - the parent revision has no variants (it may have failed). "
                    "Review the previous revision and retry."
                )
            parent_pytorch_variant = next((v for v in parent_variants if v.format == ModelFormat.PYTORCH), None)
            if parent_pytorch_variant is None:
                raise ExecutionErr(
                    "Can't start training - the parent revision has no PyTorch variant. "
                    "Review the previous revision and retry."
                )
            weights_path = self.__build_model_weights_path(
                self._data_dir, project_id, parent_model_revision_id, parent_pytorch_variant.id
            )
            if not weights_path.exists():
                raise FileNotFoundError(f"Parent model weights not found at {weights_path}")

            return weights_path

    @step("Assign Dataset Subsets")
    def assign_subsets(self, training_config: TrainingConfiguration, project_id: UUID) -> None:
        """Assigning subsets to all unassigned dataset items in the project dataset."""
        with self._db_session_factory() as db:
            self._subset_service.set_db_session(db)
            self.update_message("Retrieving unassigned items")
            unassigned_items = self._subset_service.get_unassigned_items_with_labels(project_id)

            if not unassigned_items:
                self.update_message("No unassigned items found")
                return

            self.update_message(f"Found {len(unassigned_items)} unassigned items")

            # Get target ratios
            split_params = training_config.task_level_parameters.dataset_preparation.subset_split
            target_ratios = SplitRatios(
                train=(split_params.training / 100), val=(split_params.validation / 100), test=(split_params.test / 100)
            )
            logger.info("Target subset ratios for unassigned items: {}", target_ratios)

            self.update_message("Computing optimal subset assignments")
            has_all_subsets_assigned = self._subset_service.has_all_subsets_assigned(project_id)
            assignments = self._subset_assigner.assign(unassigned_items, target_ratios, has_all_subsets_assigned)

            # Persist assignments
            self.update_message("Persisting subset assignments")
            self._subset_service.update_subset_assignments(project_id, assignments)

        self.update_message(f"Successfully assigned {len(assignments)} items to subsets")

    @step("Prepare Training Configuration")
    def prepare_training_configuration(
        self, training_params: TrainingJobParams, task: Task
    ) -> tuple[TrainingConfiguration, dict]:
        project_id = training_params.project_id
        with self._db_session_factory() as db:
            # Load the training configuration from the database
            self._training_configuration_service.set_db_session(db)
            training_config = self._training_configuration_service.get_by_model_architecture(
                project_id=project_id,
                model_architecture_id=training_params.model_architecture_id,
            )

            # Serialize and persist the configuration in the same YAML format adopted by Geti
            # NOTE: this is a temporary solution to minimize changes in getitune;
            # in the future, after refactoring getitune,
            # we should update this code to build the configuration directly in the format consumed by getitune
            geti_training_config = training_config.model_dump(exclude_none=True)
            geti_training_config["hyper_parameters"] = geti_training_config.pop("algo_level_parameters")
            geti_training_config["model_manifest_id"] = training_params.model_architecture_id
            geti_training_config["sub_task_type"] = get_getitune_task_type_by_task(task)
            geti_config_path = self.__build_model_config_path(self._data_dir, project_id, training_params.model_id)
            geti_config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(geti_config_path, "w") as f:
                yaml.dump(geti_training_config, f, default_flow_style=False)
                logger.info("Persisted training configuration at {}", geti_config_path)

            # Convert the configuration to the format adopted by getitune
            converter = GetiConfigConverter()
            getitune_training_config = converter.convert(geti_training_config)

            return training_config, getitune_training_config

    @step("Prepare Training Dataset", 10)
    def prepare_training_dataset(
        self,
        project_id: UUID,
        task: Task,
        getitune_training_config: dict,
        training_config: TrainingConfiguration,
        dataset_revision_id: UUID | None = None,
    ) -> DatasetInfo:
        """
        Prepare datasets for training, validation, and testing.

        If a specific dataset revision ID is provided, it loads that revision from the database.
        Otherwise, it creates a new dataset from the current items in the database with user-verified annotations.
        """

        def build_subset_config(subset_name: str) -> SubsetConfig:
            subset_cfg_data = getitune_training_config["data"][f"{subset_name}_subset"]
            subset_cfg_data["input_size"] = getitune_training_config["data"]["input_size"]
            sampler_cfg_data = subset_cfg_data.pop("sampler", {})
            subset_config = SubsetConfig(sampler=SamplerConfig(**sampler_cfg_data), **subset_cfg_data)
            # pyrefly: ignore[missing-attribute,bad-assignment]
            subset_config.transforms = TransformLibFactory.generate(subset_config)
            return subset_config

        with self._db_session_factory() as db:
            self._dataset_service.set_db_session(db)
            self._dataset_revision_service.set_db_session(db)

            if dataset_revision_id is not None:
                # Load the specified dataset revision from the database
                logger.info("Loading pre-existing dataset revision (ID={}) from the database", dataset_revision_id)
                dm_dataset = self._dataset_revision_service.load_revision(
                    project_id=project_id, dataset_revision_id=dataset_revision_id
                )
            else:
                latest_dataset_revision = self._dataset_revision_service.get_latest_uptodate_dataset_revision(
                    project_id=project_id
                )
                if latest_dataset_revision is not None:
                    dataset_revision_id = latest_dataset_revision.id
                    logger.info(
                        "Loading pre-existing dataset revision (ID={}) from the database, "
                        "as no changes have been detected since last training.",
                        dataset_revision_id,
                    )
                    dm_dataset = self._dataset_revision_service.load_revision(
                        project_id=project_id, dataset_revision_id=dataset_revision_id
                    )

                else:
                    # Create a dataset revision including only the items with user-verified annotations, then save it
                    logger.info("Creating a new dataset revision with user-verified annotated items")
                    dm_dataset = self._dataset_service.get_dm_dataset(
                        project_id=project_id,
                        task=task,
                        annotation_status=DatasetItemAnnotationStatus.WITH_ANNOTATIONS,
                        sample_mode=SampleMode.TRAINING,
                    )
                    dataset_revision_id = self._dataset_revision_service.save_revision(
                        project_id=project_id, dataset=dm_dataset
                    )
                    logger.info("Dataset revision saved with ID: {}", dataset_revision_id)

            # Apply filtering based on min/max annotation objects if enabled
            dm_dataset = self.filter_dataset(dm_dataset=dm_dataset, task=task, training_config=training_config)

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

            # Detect storage dtype and propagate to subset configs.
            storage_dtype = detect_storage_dtype(dm_training_dataset)
            for cfg in (train_subset_config, val_subset_config, test_subset_config):
                cfg.intensity.storage_dtype = storage_dtype

            # Wrap them into VisionDataset instances
            getitune_task_type = get_getitune_task_type_by_task(task)
            getitune_dataset_class = get_getitune_dataset_class_by_task_type(getitune_task_type)
            logger.info("Preparing {} instances for each subset", getitune_dataset_class.__name__)
            getitune_training_dataset = getitune_dataset_class(
                dm_subset=dm_training_dataset,
                transforms=train_subset_config.transforms,  # pyrefly: ignore[missing-attribute,bad-argument-type]
            )
            getitune_validation_dataset = getitune_dataset_class(
                dm_subset=dm_validation_dataset,
                transforms=val_subset_config.transforms,  # pyrefly: ignore[missing-attribute,bad-argument-type]
            )
            getitune_testing_dataset = getitune_dataset_class(
                dm_subset=dm_testing_dataset,
                transforms=test_subset_config.transforms,  # pyrefly: ignore[missing-attribute,bad-argument-type]
            )

            return DatasetInfo(
                getitune_training_dataset=getitune_training_dataset,
                getitune_validation_dataset=getitune_validation_dataset,
                getitune_testing_dataset=getitune_testing_dataset,
                getitune_training_subset_config=train_subset_config,
                getitune_validation_subset_config=val_subset_config,
                getitune_testing_subset_config=test_subset_config,
                revision_id=dataset_revision_id,
            )

    @step("Prepare Model")
    def prepare_model(
        self, training_params: TrainingJobParams, dataset_revision_id: UUID, configuration: TrainingConfiguration
    ) -> None:
        project_id = training_params.project_id
        logger.info("Preparing model revision (model_id={}, project_id={})", training_params.model_id, project_id)
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            self._model_service.create_revision(
                ModelRevisionMetadata(
                    model_id=training_params.model_id,
                    model_name=training_params.model_name,
                    project_id=project_id,
                    architecture_id=training_params.model_architecture_id,
                    parent_revision_id=training_params.parent_model_revision_id,
                    training_configuration=configuration,
                    dataset_revision_id=dataset_revision_id,
                    training_status=TrainingStatus.NOT_STARTED,
                )
            )

    @step("Train Model", 80)
    def train_model(
        self,
        training_config: dict,
        dataset_info: DatasetInfo,
        weights_path: Path,
        model_id: UUID,
        device: DeviceInfo,
        has_parent_revision: bool,
    ) -> tuple[Path, Engine]:
        """Execute model training.

        Instantiates the model from *training_config* with jsonargparse and
        creates the engine through ``create_engine`` from the library.
        Lightning callbacks are built when a ``callbacks`` section is present
        in the config; backends that do not support them simply ignore the
        parameter.
        """
        logger.info("Preparing the DataModule for training (model_id={})", model_id)
        datamodule = DataModule.from_vision_datasets(
            train_dataset=dataset_info.getitune_training_dataset,
            val_dataset=dataset_info.getitune_validation_dataset,
            test_dataset=dataset_info.getitune_testing_dataset,
            train_subset=dataset_info.getitune_training_subset_config,
            val_subset=dataset_info.getitune_validation_subset_config,
            test_subset=dataset_info.getitune_testing_subset_config,
        )

        logger.info("Instantiating model for training (model_id={})", model_id)
        model_cfg = training_config["model"]
        model_cfg["init_args"]["label_info"] = datamodule.label_info.label_names
        model_cfg["init_args"]["data_input_params"] = DataInputParams(
            input_size=cast(tuple[int, int], datamodule.input_size),
            mean=datamodule.input_mean if datamodule.input_mean is not None else (0.0, 0.0, 0.0),
            std=datamodule.input_std if datamodule.input_std is not None else (1.0, 1.0, 1.0),
            intensity_config=datamodule.input_intensity_config,
        ).as_dict()
        logger.info("Initializing engine for training (model_id={})", model_id)
        getitune_device_type = (
            GetiTuneDeviceType.gpu if device.type is DeviceType.CUDA else GetiTuneDeviceType(device.type)
        )
        engine_kwargs: dict[str, Any] = {
            "work_dir": self._data_dir / f"getitune-workspace-{model_id}",
            "device": getitune_device_type,
        }

        model_parser = ArgumentParser()
        model_parser.add_argument("--model", type=LightningModel | UltralyticsModel)
        getitune_model = model_parser.instantiate_classes(Namespace(model=model_cfg)).get("model")

        # Ultralytics models handle their own weight loading (pretrained or
        # parent checkpoint) via load_checkpoint — call it unconditionally.
        # Lightning models expect the checkpoint only for parent-revision
        # training; fresh training uses the model's built-in initialization.
        if hasattr(getitune_model, "load_checkpoint"):
            getitune_model.load_checkpoint(weights_path)
        elif has_parent_revision:
            engine_kwargs["checkpoint"] = weights_path

        if hasattr(getitune_model, "tile_config"):
            getitune_model.tile_config = datamodule.tile_config

        getitune_engine = create_engine(model=getitune_model, data=datamodule, **engine_kwargs)

        callbacks_cfg = training_config.get("callbacks", [])
        for cb_cfg in callbacks_cfg:
            if "init_args" in cb_cfg and "dirpath" in cb_cfg["init_args"]:
                cb_cfg["init_args"]["dirpath"] = getitune_engine.work_dir
        parser = ArgumentParser()
        parser.add_argument("--callbacks", type=list[Callback])
        parsed_callbacks_cfg = parser.parse_object({"callbacks": callbacks_cfg})
        callbacks_list = parser.instantiate_classes(parsed_callbacks_cfg).get("callbacks", [])
        callbacks_list.append(TrainingProgressCallback(self.update_progress, min_p=10, max_p=80))

        logger.info("Starting training loop (model_id={})", model_id)
        train_kwargs: dict[str, Any] = {
            "max_epochs": training_config["max_epochs"],
            "callbacks": callbacks_list,
        }
        if device.type is not DeviceType.CPU and device.index is not None:
            train_kwargs["devices"] = [device.index]
        if "precision" in training_config:
            train_kwargs["precision"] = training_config["precision"]
        # Forward backend-specific training args (e.g. patience for Ultralytics).
        if "training" in training_config:
            train_kwargs.update(training_config["training"])
        getitune_engine.train(**train_kwargs)  # pyrefly: ignore[bad-argument-type]

        trained_model_path = getitune_engine.best_checkpoint
        if trained_model_path is None:
            trained_model_path = Path(getitune_engine.work_dir) / "best_checkpoint.ckpt"
        if not trained_model_path.exists():
            raise FileNotFoundError(f"Trained checkpoint not found at {trained_model_path}")
        logger.info("Model training completed. Trained model saved at {}", trained_model_path)
        return trained_model_path, getitune_engine

    @step("Evaluate Model", 95)
    def evaluate_model(
        self,
        getitune_engine: Engine,
        task: Task,
        model_revision_id: UUID,
        model_variants: list[ModelVariantDescriptor],
        dataset_revision_id: UUID,
    ) -> None:
        """Evaluate the trained model variants on the testing set.

        Each variant is evaluated using its own dedicated engine so that the recorded
        metrics reflect the runtime that will actually serve it. Results are persisted
        against the corresponding model variant id.

        - PyTorch (.ckpt) variants are evaluated with the LightningEngine used for training.
        - OpenVINO (.xml) and ONNX (.onnx) variants are evaluated with OVEngine, which
          natively supports both checkpoint types.
        """
        metric_callable = get_metric_by_task(task)
        ov_work_dir_base = Path(getitune_engine.work_dir)
        datamodule = getitune_engine.datamodule

        for variant in model_variants:
            logger.info("Evaluating the {} model...", variant.format.value)
            match variant.format:
                case ModelFormat.PYTORCH:
                    engine = getitune_engine
                case ModelFormat.OPENVINO:
                    engine = OVEngine(
                        model=variant.path,
                        data=datamodule,
                        work_dir=ov_work_dir_base / "ov_eval",
                    )
                case ModelFormat.ONNX:
                    engine = OVEngine(
                        model=variant.path,
                        data=datamodule,
                        work_dir=ov_work_dir_base / "onnx_eval",
                    )
                case _:
                    raise ExecutionErr(f"Unsupported model variant format for evaluation: {variant.format}")

            metrics = engine.test(metric=metric_callable)
            self._save_evaluation_result(
                metrics=metrics,
                model_revision_id=model_revision_id,
                model_variant_id=variant.id,
                dataset_revision_id=dataset_revision_id,
            )

    def _save_evaluation_result(
        self,
        metrics: dict,
        model_revision_id: UUID,
        model_variant_id: UUID,
        dataset_revision_id: UUID,
    ) -> None:
        """Persist a single EvaluationResult tagged with the given model variant id."""
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            self._model_service.save_evaluation_result(
                EvaluationResult(
                    model_revision_id=model_revision_id,
                    model_variant_id=model_variant_id,
                    dataset_revision_id=dataset_revision_id,
                    subset=DatasetItemSubset.TESTING,
                    metrics=convert_metrics(metrics),
                )
            )

    @step("Export Model")
    def export_model(self, getitune_engine: Engine, model_checkpoint_path: Path) -> ExportedModels:
        """Export the trained model to desired OpenVINO and ONNX formats"""
        logger.info("Exporting the model to OpenVINO format (FP16 precision)...")
        exported_ov_model_path = getitune_engine.export(
            checkpoint=model_checkpoint_path,
            export_format=ExportFormat.OPENVINO,
            export_precision=Precision.FP16,
        )
        logger.info("Model exported to OpenVINO format: {}", exported_ov_model_path)

        logger.info("Exporting the model to ONNX format (FP16 precision)...")
        exported_onnx_model_path = getitune_engine.export(
            checkpoint=model_checkpoint_path,
            export_format=ExportFormat.ONNX,
            export_precision=Precision.FP16,
        )
        logger.info("Model exported to ONNX format: {}", exported_onnx_model_path)
        return ExportedModels(openvino_model_path=exported_ov_model_path, onnx_model_path=exported_onnx_model_path)

    @step("Store Model Artifacts", 100)
    def store_model_artifacts(
        self,
        model_dir: Path,
        getitune_work_dir: Path,
        trained_model_path: Path,
        exported_model_paths: ExportedModels,
        created_variants: dict[ModelFormat, UUID],
    ) -> None:
        """Copy training artifacts into variant directories.

        Each variant's files are stored under model_dir/variants/<variant_id>/model.*

        The getitune workspace itself is removed by ``TrainingJob.on_complete`` after
        the job terminates, so this step does not clean it up.

        Args:
            model_dir: The base model directory.
            getitune_work_dir: The getitune workspace directory containing the source artifacts.
            trained_model_path: Path to the trained checkpoint inside the workspace.
            exported_model_paths: Paths to exported model files inside the workspace.
            created_variants: Mapping of ModelFormat to variant UUID (from create_model_variants).
        """
        variants_dir = model_dir / "variants"
        variants_dir.mkdir(parents=True, exist_ok=True)

        pytorch_variant_dir = variants_dir / str(created_variants[ModelFormat.PYTORCH])
        pytorch_variant_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trained_model_path, pytorch_variant_dir / "model.pt")
        logger.info("Stored PyTorch variant at {}", pytorch_variant_dir)

        # Copy OpenVINO IR files
        openvino_variant_dir = variants_dir / str(created_variants[ModelFormat.OPENVINO])
        openvino_variant_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            exported_model_paths.openvino_model_path.with_suffix(".xml"),
            openvino_variant_dir / "model.xml",
        )
        shutil.copyfile(
            exported_model_paths.openvino_model_path.with_suffix(".bin"),
            openvino_variant_dir / "model.bin",
        )
        logger.info("Stored OpenVINO variant at {}", openvino_variant_dir)

        # Copy ONNX file
        onnx_variant_dir = variants_dir / str(created_variants[ModelFormat.ONNX])
        onnx_variant_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            exported_model_paths.onnx_model_path.with_suffix(".onnx"),
            onnx_variant_dir / "model.onnx",
        )
        logger.info("Stored ONNX variant at {}", onnx_variant_dir)

        # Store the metrics
        metrics_source_path = getitune_work_dir / "csv"
        metrics_dest_path = model_dir / "metrics"
        if metrics_source_path.exists():
            shutil.move(metrics_source_path, metrics_dest_path)
            logger.info("Stored training metrics at {}", metrics_dest_path)
        else:
            results_csv = getitune_work_dir / "train" / "results.csv"
            if results_csv.exists():
                dest_version_dir = metrics_dest_path / "version_0"
                dest_version_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(results_csv, dest_version_dir / "metrics.csv")
                logger.info("Stored training metrics at {}", dest_version_dir)

    def create_model_variants(self, model_revision_id: UUID) -> dict[ModelFormat, UUID]:
        """Create variant records in the database for all exported formats.

        This is called before evaluation so that variant IDs are available for
        associating evaluation results, and before the getitune workspace is cleaned up.

        Args:
            model_revision_id: The ID of the model revision for which to create variants for.

        Returns:
            dict mapping ModelFormat to the created variant UUID.
        """
        created_variants: dict[ModelFormat, UUID] = {}
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            for fmt, precision in [
                (ModelFormat.PYTORCH, ModelPrecision.FP32),
                (ModelFormat.OPENVINO, ModelPrecision.FP16),
                (ModelFormat.ONNX, ModelPrecision.FP16),
            ]:
                variant = self._model_service.create_variant(
                    model_revision_id=model_revision_id,
                    format=fmt,
                    precision=precision,
                )
                created_variants[fmt] = variant.id
                logger.info("Created {} variant record (id={})", fmt.value, variant.id)
        return created_variants

    def execute(self, params: TrainingJobParams) -> None:
        training_start_time = datetime.now(UTC)
        project_id = params.project_id
        task = params.task
        model_dir = self.__base_model_path(
            data_dir=self._data_dir,
            project_id=project_id,
            model_id=params.model_id,
        )

        weights_path = self.prepare_weights(training_params=params)
        training_config, getitune_training_config = self.prepare_training_configuration(
            training_params=params, task=task
        )
        self.assign_subsets(training_config=training_config, project_id=project_id)
        dataset_info = self.prepare_training_dataset(
            project_id=project_id,
            task=task,
            getitune_training_config=getitune_training_config,
            training_config=training_config,
            dataset_revision_id=params.dataset_revision_id,
        )
        self.prepare_model(
            training_params=params, dataset_revision_id=dataset_info.revision_id, configuration=training_config
        )
        try:
            self.__update_model_revision_training_status(
                project_id=project_id,
                model_id=params.model_id,
                status=TrainingStatus.IN_PROGRESS,
                training_started_at=training_start_time,
            )
            trained_model_path, getitune_engine = self.train_model(
                training_config=getitune_training_config,
                dataset_info=dataset_info,
                weights_path=weights_path,
                model_id=params.model_id,
                device=params.device,
                has_parent_revision=params.parent_model_revision_id is not None,
            )
            exported_model_paths = self.export_model(
                getitune_engine=getitune_engine, model_checkpoint_path=trained_model_path
            )

            # Create variant DB records first (no file I/O yet)
            created_variants = self.create_model_variants(model_revision_id=params.model_id)

            # Build descriptors mapping each variant id to its source file path inside the workspace
            model_variants = [
                ModelVariantDescriptor(
                    id=created_variants[ModelFormat.PYTORCH],
                    path=trained_model_path,
                    format=ModelFormat.PYTORCH,
                ),
                ModelVariantDescriptor(
                    id=created_variants[ModelFormat.OPENVINO],
                    path=exported_model_paths.openvino_model_path.with_suffix(".xml"),
                    format=ModelFormat.OPENVINO,
                ),
                ModelVariantDescriptor(
                    id=created_variants[ModelFormat.ONNX],
                    path=exported_model_paths.onnx_model_path.with_suffix(".onnx"),
                    format=ModelFormat.ONNX,
                ),
            ]

            # Evaluate while the workspace and checkpoint still exist
            self.evaluate_model(
                getitune_engine=getitune_engine,
                task=task,
                model_revision_id=params.model_id,
                model_variants=model_variants,
                dataset_revision_id=dataset_info.revision_id,
            )

            # Now copy files into variant dirs and clean up the workspace
            self.store_model_artifacts(
                model_dir=model_dir,
                getitune_work_dir=Path(getitune_engine.work_dir),
                trained_model_path=trained_model_path,
                exported_model_paths=exported_model_paths,
                created_variants=created_variants,
            )
            training_finish_time = datetime.now(UTC)
            self.__update_model_revision_training_status(
                project_id=project_id,
                model_id=params.model_id,
                status=TrainingStatus.SUCCESSFUL,
                training_finished_at=training_finish_time,
            )
        except CancelledExc:
            try:
                self.__delete_model_revision(project_id=project_id, model_id=params.model_id)
            except Exception as cleanup_exc:
                logger.error(
                    "Failed to delete model revision during cancellation (project_id={}, model_id={}): {}",
                    project_id,
                    params.model_id,
                    cleanup_exc,
                )
            raise
        except Exception:
            training_finish_time = datetime.now(UTC)
            self.__update_model_revision_training_status(
                project_id=project_id,
                model_id=params.model_id,
                status=TrainingStatus.FAILED,
                training_finished_at=training_finish_time,
            )
            raise

    def _build_filter_conditions(
        self,
        getitune_task_type: TaskType,
        annotation_field: str,
        filtering_config: Filtering,
    ) -> list:
        """Build polars filter conditions for min/max annotation object counts."""
        filter_conditions = []

        if filtering_config.min_annotation_objects.enable:
            min_value = filtering_config.min_annotation_objects.value
            logger.info("Filtering samples with a minimum of {} annotation objects", min_value)
            if getitune_task_type == TaskType.MULTI_CLASS_CLS:
                # Multiclass label is a scalar - presence means exactly 1 annotation object
                filter_conditions.append(pl.col(annotation_field).is_not_null())
            else:
                filter_conditions.append(pl.col(annotation_field).list.len() >= min_value)

        if filtering_config.max_annotation_objects.enable:
            max_value = filtering_config.max_annotation_objects.value
            logger.info("Filtering samples with a maximum of {} annotation objects", max_value)
            if getitune_task_type != TaskType.MULTI_CLASS_CLS:
                # Max filtering is a no-op for multiclass (always 0 or 1 label)
                filter_conditions.append(pl.col(annotation_field).list.len() <= max_value)

        return filter_conditions

    def filter_dataset(self, dm_dataset: Dataset, task: Task, training_config: TrainingConfiguration) -> Dataset:
        """Filter a dataset based on min/max annotation object counts from the training configuration.

        Args:
            dm_dataset: The dataset to filter.
            task: The task, used to determine which annotation field to count.
            training_config: The training configuration containing the filtering parameters.

        Returns:
            A new Dataset containing only the samples that pass the filter, or the original
            dataset unchanged if filtering is disabled.
        """
        filtering_config = training_config.task_level_parameters.dataset_preparation.filtering
        min_enabled = filtering_config.min_annotation_objects.enable
        max_enabled = filtering_config.max_annotation_objects.enable

        if not (min_enabled or max_enabled):
            return dm_dataset

        logger.info("Applying annotation object filtering to the dataset")

        # Determine the annotation field name based on task type
        getitune_task_type = get_getitune_task_type_by_task(task)
        match getitune_task_type:
            case TaskType.DETECTION:
                annotation_field = "bboxes"
            case TaskType.INSTANCE_SEGMENTATION:
                annotation_field = "polygons"
            case TaskType.MULTI_CLASS_CLS | TaskType.MULTI_LABEL_CLS:
                # For classification tasks, count the label field.
                # For multiclass, label is a single scalar; for multilabel it's a list.
                annotation_field = "label"
            case _:
                logger.warning("Unsupported task type for annotation object filtering: {}", getitune_task_type)
                return dm_dataset

        filter_conditions = self._build_filter_conditions(getitune_task_type, annotation_field, filtering_config)

        if not filter_conditions:
            return dm_dataset

        # Combine all conditions with AND and apply to the dataframe
        combined_filter = filter_conditions[0]
        for condition in filter_conditions[1:]:
            combined_filter = combined_filter & condition

        original_count = len(dm_dataset.df)
        filtered_df = dm_dataset.df.filter(combined_filter)
        filtered_count = len(filtered_df)

        logger.info(
            "Filtered dataset from {} to {} samples ({} samples removed)",
            original_count,
            filtered_count,
            original_count - filtered_count,
        )

        return Dataset.from_dataframe(
            df=filtered_df,
            dtype_or_schema=dm_dataset.dtype,
            schema=dm_dataset.schema,
        )

    @staticmethod
    def __base_model_path(data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return data_dir / "projects" / str(project_id) / "models" / str(model_id)

    @classmethod
    def __build_model_weights_path(cls, data_dir: Path, project_id: UUID, model_id: UUID, model_variant: UUID) -> Path:
        """Get the path to the stored PyTorch checkpoint."""
        model_dir = cls.__base_model_path(data_dir, project_id, model_id)
        variant_dir = model_dir / "variants" / str(model_variant)
        return variant_dir / "model.pt"

    @classmethod
    def __build_model_config_path(cls, data_dir: Path, project_id: UUID, model_id: UUID) -> Path:
        return cls.__base_model_path(data_dir, project_id, model_id) / "config.yaml"

    def __update_model_revision_training_status(
        self,
        project_id: UUID,
        model_id: UUID,
        status: TrainingStatus,
        training_started_at: datetime | None = None,
        training_finished_at: datetime | None = None,
    ):
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            self._model_service.update_revision_status(
                project_id=project_id,
                model_id=model_id,
                training_status=status,
                training_started_at=training_started_at,
                training_finished_at=training_finished_at,
            )

    def __delete_model_revision(self, project_id: UUID, model_id: UUID):
        with self._db_session_factory() as db:
            self._model_service.set_db_session(db)
            self._model_service.delete_model(project_id=project_id, model_id=model_id)
