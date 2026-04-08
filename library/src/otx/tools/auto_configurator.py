# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Auto-Configurator class & util functions for OTX Auto-Configuration."""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

from jsonargparse import ArgumentParser, Namespace

from otx.backend.native.cli.utils import get_otx_root_path, list_models
from otx.backend.native.models.base import DataInputParams, OTXModel
from otx.config.data import SamplerConfig, SubsetConfig, TileConfig
from otx.data.module import OTXDataModule
from otx.types import PathLike
from otx.types.label import LabelInfoTypes
from otx.types.task import OTXTaskType
from otx.utils.utils import can_pass_tile_config, get_model_cls_from_config, should_pass_label_info

if TYPE_CHECKING:
    from otx.backend.openvino.models.base import OVModel


logger = logging.getLogger()
RECIPE_PATH = get_otx_root_path() / "recipe"

DEFAULT_CONFIG_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: RECIPE_PATH / "classification" / "multi_class_cls" / "mobilenet_v3_large.yaml",
    OTXTaskType.MULTI_LABEL_CLS: RECIPE_PATH / "classification" / "multi_label_cls" / "mobilenet_v3_large.yaml",
    OTXTaskType.H_LABEL_CLS: RECIPE_PATH / "classification" / "h_label_cls" / "mobilenet_v3_large.yaml",
    OTXTaskType.DETECTION: RECIPE_PATH / "detection" / "yolox_s.yaml",
    OTXTaskType.ROTATED_DETECTION: RECIPE_PATH / "rotated_detection" / "maskrcnn_r50.yaml",
    OTXTaskType.SEMANTIC_SEGMENTATION: RECIPE_PATH / "semantic_segmentation" / "litehrnet_18.yaml",
    OTXTaskType.INSTANCE_SEGMENTATION: RECIPE_PATH / "instance_segmentation" / "rfdetr_seg_small.yaml",
    OTXTaskType.KEYPOINT_DETECTION: RECIPE_PATH / "keypoint_detection" / "rtmpose_tiny.yaml",
}


OVMODEL_PER_TASK = {
    OTXTaskType.MULTI_CLASS_CLS: "otx.backend.openvino.models.OVMulticlassClassificationModel",
    OTXTaskType.MULTI_LABEL_CLS: "otx.backend.openvino.models.OVMultilabelClassificationModel",
    OTXTaskType.H_LABEL_CLS: "otx.backend.openvino.models.OVHlabelClassificationModel",
    OTXTaskType.DETECTION: "otx.backend.openvino.models.OVDetectionModel",
    OTXTaskType.ROTATED_DETECTION: "otx.backend.openvino.models.OVRotatedDetectionModel",
    OTXTaskType.INSTANCE_SEGMENTATION: "otx.backend.openvino.models.OVInstanceSegmentationModel",
    OTXTaskType.SEMANTIC_SEGMENTATION: "otx.backend.openvino.models.OVSegmentationModel",
    OTXTaskType.KEYPOINT_DETECTION: "otx.backend.openvino.models.OVKeypointDetectionModel",
}


class AutoConfigurator:
    """This Class is used to configure the OTXDataModule, OTXModel, Optimizer, and Scheduler with OTX Default.

    Args:
        data_root (PathLike | None, optional): The root directory for data storage. Defaults to None.
        task (OTXTaskType | None, optional): The task type. If None, the task will be configured based on the model.
            Defaults to None.
        model (PathLike | str | None, optional): Path to the model config file or name of the model to use.
            If None, the task should be provided and the default model for the task will be used.
            Defaults to None.

    Example:
        The following examples show how to use the AutoConfigurator class.

        >>> auto_configurator = AutoConfigurator(
        ...     data_root=<dataset/path>,
        ...     task=<OTXTaskType>,
        ... )

        # If task is None, the task will be configured based on the data root.
        >>> auto_configurator = AutoConfigurator(
        ...     data_root=<dataset/path>,
        ... )
    """

    def __init__(
        self,
        data_root: PathLike | None = None,
        task: OTXTaskType | None = None,
        model: PathLike | str | None = None,
    ) -> None:
        self.data_root = data_root
        self._task = task
        model_config_path: PathLike | None = None
        if model is not None:
            if not str(model).endswith(".yaml"):
                if task is None:
                    msg = "If model is provided as a name, task must be provided to find the model."
                    raise ValueError(msg)
                recipe_list = list_models(task=task, pattern=str(model), return_recipes=True)
                if len(recipe_list) > 1:
                    msg = (
                        "There is more than 1 model match the given name."
                        "It may happen with overlap of the tasks. Using the first one."
                        "To use the specific model, provide model config instead."
                    )
                    logger.warning(msg)
                elif len(recipe_list) == 0:
                    msg = f"Model {model} does not exist."
                    raise FileNotFoundError(msg)
                model_config_path = recipe_list[0]
            else:
                model_config_path = model
            if not Path(model_config_path).exists():
                msg = f"Model config path {model} does not exist."
                raise FileNotFoundError(msg)
        if model_config_path:
            self._config: dict = self._load_default_config(config_path=model_config_path)
            self._task = OTXTaskType(self._config.get("task", task))
        elif task:
            self._config = self._load_default_config(task=task)
        else:
            msg = "Either task or model must be provided."
            raise ValueError(msg)

    @property
    def task(self) -> OTXTaskType:
        """Returns the current task.

        Raises:
            RuntimeError: If there are no ready tasks.

        Returns:
            OTXTaskType | str: The current task.
        """
        if self._task is not None:
            return self._task
        if self._config is not None and "task" in self._config:
            return OTXTaskType(self._config["task"])
        msg = "There are no ready task"
        raise RuntimeError(msg)

    @property
    def config(self) -> dict:
        """Retrieves the configuration for the auto configurator.

        Returns:
            dict: The configuration as a dict object.
        """
        return self._config

    def _load_default_config(self, config_path: PathLike | None = None, task: OTXTaskType | None = None) -> dict:
        """Load the default configuration for the specified model.

        Args:
            model_name (str | None): The name of the model. If provided, the configuration
                file name will be modified to use the specified model.

        Returns:
            dict: The loaded configuration.

        Raises:
            ValueError: If the task doesn't supported for auto-configuration.
        """
        from otx.cli.utils.jsonargparse import get_configuration

        task = task if task is not None else self._task
        if config_path is None:
            if task is None:
                msg = "Either config_path or task must be provided."
                raise ValueError(msg)
            config_path = DEFAULT_CONFIG_PER_TASK[task]

        return get_configuration(config_path)

    def get_datamodule(self, data_root: PathLike | None = None) -> OTXDataModule:
        """Returns an instance of OTXDataModule with the configured data root.

        Returns:
            OTXDataModule | None: An instance of OTXDataModule.
        """
        if data_root is None and self.data_root is None:
            msg = "No data root provided."
            raise ValueError(msg)
        if data_root is not None and not isinstance(data_root, (str, os.PathLike)):
            msg = f"data_root should be of type PathLike, but got {type(data_root)}"
            raise TypeError(msg)

        data_root = data_root if data_root is not None else self.data_root
        self.config["data"]["data_root"] = data_root
        data_config: dict = deepcopy(self.config["data"])
        train_config = data_config.pop("train_subset")
        val_config = data_config.pop("val_subset")
        test_config = data_config.pop("test_subset")
        tile_config = data_config.pop("tile_config", {})

        _ = data_config.pop("__path__", {})  # Remove __path__ key that for CLI
        _ = data_config.pop("config", {})  # Remove config key that for CLI

        return OTXDataModule(
            train_subset=SubsetConfig(sampler=SamplerConfig(**train_config.pop("sampler", {})), **train_config),
            val_subset=SubsetConfig(sampler=SamplerConfig(**val_config.pop("sampler", {})), **val_config),
            test_subset=SubsetConfig(sampler=SamplerConfig(**test_config.pop("sampler", {})), **test_config),
            tile_config=TileConfig(**tile_config),
            **data_config,
        )

    def get_model(
        self,
        model_name: str | None = None,
        label_info: LabelInfoTypes | None = None,
        data_input_params: DataInputParams | dict | None = None,
    ) -> OTXModel:
        """Retrieves the OTXModel instance based on the provided model name and meta information.

        Args:
            model_name (str | None): The name of the model to retrieve. If None, the default model will be used.
            label_info (LabelInfoTypes | None): The meta information about the labels.
                If provided, the number of classes will be updated in the model's configuration.
            data_input_params (DataInputParams | dict | None, optional): The data input parameters
                containing the input size, input mean and std.

        Returns:
            OTXModel: The instantiated OTXModel instance.

        Example:
            The following examples show how to get the OTXModel class.

            # If model_name is None, the default model will be used from task.
            >>> auto_configurator.get_model(
            ...     label_info=<LabelInfo>,
            ... )

            # If model_name is str, the default config file is changed.
            >>> auto_configurator.get_model(
            ...     model_name=<model_name, str>,
            ...     label_info=<LabelInfo>,
            ... )
        """
        # TODO(vinnamki): There are some overlaps with src/otx/cli/cli.py::OTXCLI::instantiate_model
        if model_name is not None:
            self._config = self._load_default_config(model_name)

        skip = set()

        model_config = deepcopy(self.config["model"])

        if data_input_params is not None:
            model_config["init_args"]["data_input_params"] = (
                data_input_params if isinstance(data_input_params, dict) else data_input_params.as_dict()
            )
        elif (datamodule := self.get_datamodule()) is not None:
            # get data_input_params info from datamodule
            if datamodule.input_size is None:
                msg = (
                    "Input size is not specified in the datamodule. Ensure that the datamodule has a valid input size."
                )
                raise ValueError(msg)
            model_config["init_args"]["data_input_params"] = DataInputParams(
                input_size=datamodule.input_size,
                mean=datamodule.input_mean if datamodule.input_mean is not None else (0.0, 0.0, 0.0),
                std=datamodule.input_std if datamodule.input_std is not None else (1.0, 1.0, 1.0),
            ).as_dict()

        model_cls = get_model_cls_from_config(Namespace(model_config))

        if should_pass_label_info(model_cls):
            if label_info is None:
                msg = f"Given model class {model_cls} requires a valid label_info to instantiate."
                raise ValueError(msg)

            model_config["init_args"]["label_info"] = label_info
            skip.add("label_info")

        if can_pass_tile_config(model_cls) and (datamodule := self.get_datamodule()) is not None:
            model_config["init_args"]["tile_config"] = datamodule.tile_config
            skip.add("tile_config")

        model_parser = ArgumentParser()
        model_parser.add_subclass_arguments(
            OTXModel,
            "model",
            skip=skip,
            required=False,
            fail_untyped=False,
        )
        return model_parser.instantiate_classes(Namespace(model=model_config)).get("model")

    def get_ov_model(self, model_name: PathLike, task: OTXTaskType | None = None) -> OVModel:
        """Retrieves the OVModel instance based on the given model name and label information.

        Args:
            model_name (str): The name of the model.
            label_info (LabelInfo): The label information.

        Returns:
            OVModel: The OVModel instance.

        Raises:
            NotImplementedError: If the OVModel for the given task is not supported.
        """
        task = task if task is not None else self.task
        class_path = OVMODEL_PER_TASK.get(task)
        if class_path is None:
            msg = f"{task} doesn't support OVModel."
            raise NotImplementedError(msg)
        class_module, class_name = class_path.rsplit(".", 1)
        module = __import__(class_module, fromlist=[class_name])
        ov_model = getattr(module, class_name)
        return ov_model(
            model_path=model_name,
        )

    def update_ov_subset_pipeline(
        self,
        datamodule: OTXDataModule,
        subset: str = "test",
        task: OTXTaskType | None = None,
        input_size: tuple[int, int] | None = None,
        keep_aspect_ratio: bool = False,
    ) -> OTXDataModule:
        """Returns an OTXDataModule object with OpenVINO subset transforms applied.

        Args:
            datamodule (OTXDataModule): The original OTXDataModule object.
            subset (str, optional): The subset to update. Defaults to "test".
            input_size (tuple[int, int] | None, optional): Model input size (H, W)
                from the OV model metadata.  When provided this overrides
                ``datamodule.input_size`` so that ``$(input_size)`` placeholders
                in the OV recipe augmentations resolve to the correct value.
            keep_aspect_ratio (bool, optional): When ``True``, patches every
                ``otx.data.augmentation.transforms.Resize`` step in the OV
                augmentation lists so that aspect ratio is preserved (matching
                the ``resize_type`` embedded in the IR metadata at export time).
                Defaults to ``False``.

        Returns:
            OTXDataModule: The modified OTXDataModule object with OpenVINO subset transforms applied.
        """
        task = task if task is not None else self._task
        if task is None:
            msg = "Task must be provided to update OpenVINO subset pipeline."
            raise ValueError(msg)
        ov_config_path = DEFAULT_CONFIG_PER_TASK[task].parent / "openvino_model.yaml"
        ov_config = self._load_default_config(config_path=ov_config_path)["data"]
        subset_config = getattr(datamodule, f"{subset}_subset")
        ov_subset = ov_config.get(f"{subset}_subset", ov_config["test_subset"])
        subset_config.batch_size = ov_subset["batch_size"]
        subset_config.augmentations_cpu = ov_subset["augmentations_cpu"]
        subset_config.augmentations_gpu = ov_subset.get("augmentations_gpu", [])
        if keep_aspect_ratio:
            self._patch_resize_keep_aspect_ratio(subset_config.augmentations_cpu)
            self._patch_resize_keep_aspect_ratio(subset_config.augmentations_gpu)
        datamodule.tile_config.enable_tiler = False

        # Resolve input size: prefer model IR metadata, fall back to
        # the training recipe's default, raise if neither is available.
        actual_input_size = input_size or datamodule.input_size
        subset_config.input_size = actual_input_size

        if actual_input_size is None:
            msg = (
                "Cannot determine input_size for the OpenVINO pipeline. "
                "The OV model has dynamic input shapes and the datamodule "
                "does not specify input_size. Please provide input_size "
                "explicitly when calling update_ov_subset_pipeline()."
            )
            raise ValueError(msg)

        msg = (
            f"For OpenVINO IR models, Update the following {subset} \n"
            f"\t augmentations_cpu: {subset_config.augmentations_cpu} \n"
            f"\t batch_size: {subset_config.batch_size} \n"
            "And the tiler is disabled."
        )
        warn(msg, stacklevel=1)

        # If the datamodule was created from pre-constructed datasets (no data_root),
        # rebuild using from_otx_datasets to avoid re-importing from disk.
        # This is useful for the quantization pipeline.
        if not datamodule.data_root and datamodule.subsets:
            datamodule.train_subset.input_size = actual_input_size
            return OTXDataModule.from_otx_datasets(
                train_dataset=datamodule.subsets["train"],
                val_dataset=datamodule.subsets["val"],
                test_dataset=datamodule.subsets.get("test"),
                train_subset=datamodule.train_subset,
                val_subset=datamodule.val_subset,
                test_subset=datamodule.test_subset,
                auto_num_workers=datamodule.auto_num_workers,
                device=datamodule.device,
            )

        return OTXDataModule(
            task=datamodule.task,
            data_root=datamodule.data_root,
            train_subset=datamodule.train_subset,
            val_subset=datamodule.val_subset,
            test_subset=datamodule.test_subset,
            input_size=actual_input_size,
            tile_config=datamodule.tile_config,
            ignore_index=datamodule.ignore_index,
            unannotated_items_ratio=datamodule.unannotated_items_ratio,
            auto_num_workers=datamodule.auto_num_workers,
            device=datamodule.device,
        )

    @staticmethod
    def _patch_resize_keep_aspect_ratio(augmentations: list[dict]) -> None:
        """Set ``keep_aspect_ratio=True`` on every Resize step in an augmentation list.

        The OV recipe templates default to ``keep_aspect_ratio: false``.  When the
        exported model's ``resize_type`` metadata indicates aspect-ratio-preserving
        resize was used during training, this method patches the configs so that
        OV inference preprocessing matches training preprocessing exactly.

        Args:
            augmentations: List of augmentation config dicts, each with
                ``class_path`` and optionally ``init_args`` keys.
        """
        for aug in augmentations:
            if "Resize" in aug.get("class_path", ""):
                init_args = aug.setdefault("init_args", {})
                init_args["keep_aspect_ratio"] = True
