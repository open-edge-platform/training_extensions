# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX engine components."""

from __future__ import annotations

import copy
import csv
import inspect
import logging
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterable, Iterator, Literal
from warnings import warn

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.plugins.precision import MixedPrecision

from otx.core.config.device import DeviceConfig
from otx.core.config.explain import ExplainConfig
from otx.core.data.module import OTXDataModule
from otx.core.model.base import DataInputParams, OTXModel, OVModel
from otx.core.types import PathLike
from otx.core.types.device import DeviceType
from otx.core.types.export import OTXExportFormatType
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType
from otx.core.utils.cache import TrainerArgumentsCache
from otx.utils.device import is_xpu_available
from otx.utils.utils import measure_flops

from otx.engine.utils.auto_configurator import DEFAULT_CONFIG_PER_TASK, AutoConfigurator

if TYPE_CHECKING:
    from lightning import Callback
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
    from pytorch_lightning.trainer.connectors.accelerator_connector import _PRECISION_INPUT

    from otx.core.metrics import MetricCallable


@contextmanager
def override_metric_callable(model: OTXModel, new_metric_callable: MetricCallable | None) -> Iterator[OTXModel]:
    """Override `OTXModel.metric_callable` to change the evaluation metric.

    Args:
        model: Model to override its metric callable
        new_metric_callable: If not None, override the model's one with this. Otherwise, do not override.
    """
    if new_metric_callable is None:
        yield model
        return

    orig_metric_callable = model.metric_callable
    try:
        model.metric_callable = new_metric_callable
        yield model
    finally:
        model.metric_callable = orig_metric_callable


class OVEngine:
    """OV Engine.

    This class defines the Engine for OTX, which governs each step of the OTX workflow.
    """


    def __init__(
        self,
        data_root: PathLike | None = None,
        model: OVModel | str | None = None,
        work_dir: PathLike = "./otx-workspace",
        datamodule: OTXDataModule | None = None,
    ):
        """Initializes the OTX Engine.

        Args:
            data_root (PathLike | None, optional): Root directory for the data. Defaults to None.
            task (OTXTaskType | None, optional): The type of OTX task. Defaults to None.
            work_dir (PathLike, optional): Working directory for the engine. Defaults to "./otx-workspace".
            datamodule (OTXDataModule | None, optional): The data module for the engine. Defaults to None.
            model (OTXModel | str | None, optional): The model for the engine. Defaults to None.
            checkpoint (PathLike | None, optional): Path to the checkpoint file. Defaults to None.
            device (DeviceType, optional): The device type to use. Defaults to DeviceType.auto.
            num_devices (int, optional): The number of devices to use. If it is 2 or more, it will behave as multi-gpu.
            **kwargs: Additional keyword arguments for pl.Trainer.
        """
        self.work_dir = work_dir
        self._auto_configurator = AutoConfigurator(
            data_root=data_root,
            task=datamodule.task if datamodule is not None else None,
            model_name=None if isinstance(model, OVModel) else model,
        )

        self._datamodule: OTXDataModule | None = (
            datamodule if datamodule is not None else self._auto_configurator.get_datamodule()
        )
        self._model: OVModel = (
            model if isinstance(model, OVModel) else self._auto_configurator.get_ov_model(model)
        )
        self.task = self._auto_configurator.task

    # ------------------------------------------------------------------------ #
    # General Engine Entry Points
    # ------------------------------------------------------------------------ #

    def test(
        self,
        checkpoint: PathLike | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        metric: MetricCallable | None = None,
    ) -> dict:
        r"""Run the testing phase of the engine.

        Args:
            checkpoint (PathLike | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module containing the test data.
            metric (MetricCallable | None): If not None, it will override `OTXModel.metric_callable` with the given
                metric callable. It will temporarilly change the evaluation metric for the validation and test.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            dict: Dictionary containing the callback metrics from the trainer.

        Example:
            >>> engine.test(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ... )

        CLI Usage:
            1. To eval model by specifying the work_dir where did the training, run
                ```shell
                >>> otx test --work_dir <WORK_DIR_PATH, str>
                ```
            2. To eval model a specific checkpoint, run
                ```shell
                >>> otx test --work_dir <WORK_DIR_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
            3. Can pick a model.
                ```shell
                >>> otx test \
                ...     --model <CONFIG | CLASS_PATH_OR_NAME> \
                ...     --data_root <DATASET_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            4. To eval with configuration file, run
                ```shell
                >>> otx test --config <CONFIG_PATH, str> --checkpoint <CKPT_PATH, str>
                ```
        """
        model = self.model
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule
        metric = metric if metric is not None else model.metric_callable

        if datamodule is None:
            msg = "Please provide the `data_root` or `datamodule` when creating the Engine, or pass a `datamodule` in OVEngine.test()."
            raise RuntimeError(msg)
        if metric is None:
            msg = "Please provide a `metric` in OVEngine or pass it in OVEngine.test()."
            raise RuntimeError(msg)

        if checkpoint is None and model is None:
            msg = "Please provide either a model or a checkpoint path."
            raise ValueError(msg)
        elif checkpoint is not None and Path(str(checkpoint)).suffix not in [".xml", ".onnx"]:
            msg = "OV Engine supports only OV IR or ONNX checkpoints"
            raise RuntimeError(msg)
        elif checkpoint is not None:
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)

        if self.device.accelerator != "cpu":
            msg = "IR model supports inference only on CPU device. The device is changed automatic."
            warn(msg, stacklevel=1)
            self.device = DeviceType.cpu  # type: ignore[assignment]

        if model.label_info != self.datamodule.label_info:
            msg = (
                "To launch a test pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)

        for data_batch in datamodule.test_dataloader():
            preds = self.model(data_batch)
            metric_inputs = self.model.prepare_metric_inputs(preds, data_batch)
            metric.update(**metric_inputs)

        return self.model.compute_metric(metric)

    def predict(
        self,
        data: OTXDataModule | np.array,
        checkpoint: PathLike | None = None,
        explain: bool = False,
        explain_config: ExplainConfig | None = None,
        **kwargs,
    ) -> list | None:
        r"""Run predictions using the specified model and data.

        Args:
            checkpoint (PathLike | None, optional): The path to the checkpoint file to load the model from.
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module to use for predictions.
            return_predictions (bool | None, optional): Whether to return the predictions or not.
            explain (bool, optional): Whether to dump "saliency_map" and "feature_vector" or not.
            explain_config (ExplainConfig | None, optional): Explain configuration used for saliency map post-processing
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            list | None: The predictions if `return_predictions` is True, otherwise None.

        Example:
            >>> engine.predict(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ...     return_predictions=True,
            ...     explain=True,
            ... )

        CLI Usage:
            1. To predict a model with work_dir, run
                ```shell
                >>> otx predict --work_dir <WORK_DIR_PATH, str>
                ```
            2. To predict a specific model, run
                ```shell
                >>> otx predict \
                ...     --work_dir <WORK_DIR_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            3. To predict with configuration file, run
                ```shell
                >>> otx predict \
                ...     --config <CONFIG_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
        """
        from otx.algo.utils.xai_utils import process_saliency_maps_in_pred_entity, set_crop_padded_map_flag

        model = self.model
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if is_ir_ckpt and not isinstance(model, OVModel):
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)

        # NOTE: Re-initiate datamodule for OVModel as model API supports its own data pipeline.
        if isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")

        if checkpoint is not None and not is_ir_ckpt:
            kwargs_user_input: dict[str, Any] = {}

            model_cls = model.__class__
            model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, **kwargs_user_input)

        if model.label_info != self.datamodule.label_info:
            msg = (
                "To launch a predict pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)


        curr_explain_mode = model.explain_mode
        try:
            model.explain_mode = explain
            model.predict()
        finally:
            model.explain_mode = curr_explain_mode

        if explain:
            if explain_config is None:
                explain_config = ExplainConfig()
            explain_config = set_crop_padded_map_flag(explain_config, datamodule)

            predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config, datamodule.label_info)

        return predict_result

    def optimize(
        self,
        checkpoint: PathLike | None = None,
        datamodule: TRAIN_DATALOADERS | OTXDataModule | None = None,
        max_data_subset_size: int | None = None,
        export_demo_package: bool = False,
    ) -> Path:
        r"""Applies NNCF.PTQ to the underlying models (now works only for OV models).

        PTQ performs int-8 quantization on the input model, so the resulting model
        comes in mixed precision (some operations, however, remain in FP32).

        Args:
            checkpoint (str | Path | None, optional): Checkpoint to optimize. Defaults to None.
            datamodule (TRAIN_DATALOADERS | OTXDataModule | None, optional): The data module to use for optimization.
            max_data_subset_size (int | None): The maximum size of the train subset from `datamodule` that would be
            used for model optimization. If not set, NNCF.PTQ will select subset size according to it's
            default settings.
            export_demo_package (bool): Whether to export demo package with optimized models.
            It outputs zip archive with stand-alone demo package.

        Returns:
            Path: path to the optimized model.

        Example:
            >>> engine.optimize(
            ...     checkpoint=<checkpoint/path>,
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ... )

        CLI Usage:
            1. To optimize a model with IR Model, run
                ```shell
                >>> otx optimize \
                ...     --work_dir <WORK_DIR_PATH, str> \
                ...     --checkpoint <IR_MODEL_WEIGHT_PATH, str>
                ```
            2. To optimize a specific OVModel class with XML, run
                ```shell
                >>> otx optimize \
                ...     --data_root <DATASET_PATH, str> \
                ...     --checkpoint <IR_MODEL_WEIGHT_PATH, str> \
                ...     --model <CONFIG | CLASS_PATH_OR_NAME, OVModel> \
                ...     --model.model_name=<PATH_TO_IR_XML, str>
                ```
        """
        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        optimize_datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if not is_ir_ckpt:
            msg = "Engine.optimize() supports only OV IR or ONNX checkpoints"
            raise RuntimeError(msg)

        model = self.model
        if not isinstance(model, OVModel):
            optimize_datamodule = self._auto_configurator.update_ov_subset_pipeline(
                datamodule=optimize_datamodule,
                subset="train",
            )
            model = self._auto_configurator.get_ov_model(
                model_name=str(checkpoint),
                label_info=optimize_datamodule.label_info,
            )

        ptq_config = {}
        if max_data_subset_size is not None:
            ptq_config["subset_size"] = max_data_subset_size

        if not export_demo_package:
            return model.optimize(
                Path(self.work_dir),
                optimize_datamodule,
                ptq_config,
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_path = model.optimize(Path(tmp_dir), optimize_datamodule, ptq_config)
            return self.export(
                checkpoint=tmp_model_path,
                export_demo_package=True,
            )

    def explain(
        self,
        checkpoint: PathLike | None = None,
        datamodule: EVAL_DATALOADERS | OTXDataModule | None = None,
        explain_config: ExplainConfig | None = None,
        **kwargs,
    ) -> list | None:
        r"""Run XAI using the specified model and data (test subset).

        Args:
            checkpoint (PathLike | None, optional): The path to the checkpoint file to load the model from.
            datamodule (EVAL_DATALOADERS | OTXDataModule | None, optional): The data module to use for predictions.
            explain_config (ExplainConfig | None, optional): Config used to handle saliency maps.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            list: Saliency maps.

        Example:
            >>> engine.explain(
            ...     datamodule=OTXDataModule(),
            ...     checkpoint=<checkpoint/path>,
            ...     explain_config=ExplainConfig(),
            ... )

        CLI Usage:
            1. To run XAI with the torch model in work_dir, run
                ```shell
                >>> otx explain \
                ...     --work_dir <WORK_DIR_PATH, str>
                ```
            2. To run XAI using the specified model (torch or IR), run
                ```shell
                >>> otx explain \
                ...     --work_dir <WORK_DIR_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
            3. To run XAI using the configuration, run
                ```shell
                >>> otx explain \
                ...     --config <CONFIG_PATH> --data_root <DATASET_PATH, str> \
                ...     --checkpoint <CKPT_PATH, str>
                ```
        """
        from otx.algo.utils.xai_utils import (
            process_saliency_maps_in_pred_entity,
            set_crop_padded_map_flag,
        )

        model = self.model

        checkpoint = checkpoint if checkpoint is not None else self.checkpoint
        datamodule = datamodule if datamodule is not None else self.datamodule

        is_ir_ckpt = checkpoint is not None and Path(checkpoint).suffix in [".xml", ".onnx"]
        if is_ir_ckpt and not isinstance(model, OVModel):
            datamodule = self._auto_configurator.update_ov_subset_pipeline(datamodule=datamodule, subset="test")
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint), label_info=datamodule.label_info)

        if checkpoint is not None and not is_ir_ckpt:
            kwargs_user_input: dict[str, Any] = {}

            model_cls = model.__class__
            model = model_cls.load_from_checkpoint(checkpoint_path=checkpoint, **kwargs_user_input)

        if model.label_info != self.datamodule.label_info:
            msg = (
                "To launch a explain pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)

        model.explain_mode = True

        self._build_trainer(**kwargs)

        predict_result = self.trainer.predict(
            model=model,
            datamodule=datamodule,
        )

        if explain_config is None:
            explain_config = ExplainConfig()
        explain_config = set_crop_padded_map_flag(explain_config, datamodule)

        predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config, datamodule.label_info)
        model.explain_mode = False
        return predict_result

    @classmethod
    def from_config(
        cls,
        config_path: PathLike,
        data_root: PathLike | None = None,
        work_dir: PathLike | None = None,
        **kwargs,
    ) -> Engine:
        """Builds the engine from a configuration file.

        Args:
            config_path (PathLike): The configuration file path.
            data_root (PathLike | None): Root directory for the data.
                Defaults to None. If data_root is None, use the data_root from the configuration file.
            work_dir (PathLike | None, optional): Working directory for the engine.
                Defaults to None. If work_dir is None, use the work_dir from the configuration file.
            kwargs: Arguments that can override the engine's arguments.

        Returns:
            Engine: An instance of the Engine class.

        Example:
            >>> engine = Engine.from_config(
            ...     config="config.yaml",
            ... )
        """
        from otx.cli.utils.jsonargparse import get_instantiated_classes

        # For the Engine argument, prepend 'engine.' for CLI parser
        filter_kwargs = ["device", "checkpoint", "task"]
        for key in filter_kwargs:
            if key in kwargs:
                kwargs[f"engine.{key}"] = kwargs.pop(key)
        instantiated_config, train_kwargs = get_instantiated_classes(
            config=config_path,
            data_root=data_root,
            work_dir=work_dir,
            **kwargs,
        )
        engine_kwargs = {**instantiated_config.get("engine", {}), **train_kwargs}

        # Remove any input that is not currently available in Engine and print a warning message.
        set_valid_args = TrainerArgumentsCache.get_trainer_constructor_args().union(
            set(inspect.signature(Engine.__init__).parameters.keys()),
        )
        removed_args = []
        for engine_key in list(engine_kwargs.keys()):
            if engine_key not in set_valid_args:
                engine_kwargs.pop(engine_key)
                removed_args.append(engine_key)
        if removed_args:
            msg = (
                f"Warning: {removed_args} -> not available in Engine constructor. "
                "It will be ignored. Use what need in the right places."
            )
            warn(msg, stacklevel=1)

        if (datamodule := instantiated_config.get("data")) is None:
            msg = "Cannot instantiate datamodule from config."
            raise ValueError(msg)
        if not isinstance(datamodule, OTXDataModule):
            raise TypeError(datamodule)

        if (model := instantiated_config.get("model")) is None:
            msg = "Cannot instantiate model from config."
            raise ValueError(msg)
        if not isinstance(model, OTXModel):
            raise TypeError(model)

        model.label_info = datamodule.label_info

        return cls(
            work_dir=instantiated_config.get("work_dir", work_dir),
            datamodule=datamodule,
            model=model,
            **engine_kwargs,
        )

    # ------------------------------------------------------------------------ #
    # Property and setter functions provided by Engine.
    # ------------------------------------------------------------------------ #

    @property
    def work_dir(self) -> PathLike:
        """Work directory."""
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: PathLike) -> None:
        self._work_dir = work_dir
        self._cache.update(default_root_dir=work_dir)
        self._cache.is_trainer_args_identical = False

    @property
    def device(self) -> DeviceConfig:
        """Device engine uses."""
        return self._device

    @device.setter
    def device(self, device: DeviceType) -> None:
        if is_xpu_available() and device == DeviceType.auto:
            device = DeviceType.xpu
        self._device = DeviceConfig(accelerator=device)
        self._cache.update(accelerator=self._device.accelerator, devices=self._device.devices)
        self._cache.is_trainer_args_identical = False

    @property
    def model(self) -> OTXModel:
        """Returns the model object associated with the engine.

        Returns:
            OTXModel: The OTXModel object.
        """
        return self._model

    @model.setter
    def model(self, model: OTXModel | str) -> None:
        """Sets the model for the engine.

        Args:
            model (OTXModel | str): The model to be set.

        Returns:
            None
        """
        if isinstance(model, str):
            model = self._auto_configurator.get_model(model, label_info=self.datamodule.label_info)
        self._model = model

    @property
    def datamodule(self) -> OTXDataModule:
        """Returns the datamodule object associated with the engine.

        Returns:
            OTXDataModule: The OTXDataModule object.
        """
        if self._datamodule is None:
            msg = "Please include the `data_root` or `datamodule` when creating the Engine."
            raise RuntimeError(msg)
        return self._datamodule
