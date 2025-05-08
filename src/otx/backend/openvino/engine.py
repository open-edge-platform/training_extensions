# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Module for OTX engine components."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from warnings import warn
import defusedxml.ElementTree as ET
import torch

import numpy as np
from rich.progress import Progress

from otx.backend.native.utils.auto_configurator import AutoConfigurator
from otx.core.config.explain import ExplainConfig
from otx.core.data.module import OTXDataModule
from otx.data.torch import TorchDataBatch
from otx.backend.openvino.models import OVModel
from otx.core.types import PathLike

if TYPE_CHECKING:
    from otx.core.metrics import MetricCallable


class OVEngine:
    """OV Engine.

    This class defines the OV Engine for OTX, which governs each step of the OpenVINO validation workflow.
    """

    def __init__(
        self,
        datamodule: OTXDataModule| PathLike | None = None,
        model: OVModel | PathLike | None = None,
        work_dir: PathLike = "./otx-workspace",
    ):
        """Initializes the OTX Engine.

        Args:
            datamodule (OTXDataModule | PathLike | None, optional): The data module or path to the data root.
                If path is provided, Engine will automatically create a datamodule based on the data root and model.
            model (OTXModel | str | None, optional): The OV model for the engine. PathLike object to OpenVINO IR XML file can be provided.
                Defaults to None.
            work_dir (PathLike, optional): Working directory for the engine. Defaults to "./otx-workspace".
        """
        self._work_dir = work_dir
        if isinstance(model, str) and model.endswith(".xml"):
            task = self._derive_task_from_ir(model)
        elif isinstance(model, OVModel):
            task = model.task
        else:
            msg = "Model should be either a valid XML path or an instance of OVModel."
            raise ValueError(msg)

        if datamodule is not None and isinstance(datamodule, OTXDataModule):
            if datamodule.task != task:
                msg = (
                    "The task of the provided datamodule does not match the task derived from the model. "
                    f"datamodule.task={datamodule.task}, model.task={task}"
                )
                raise ValueError(msg)

        self._auto_configurator = AutoConfigurator(
            data_root=datamodule if isinstance(datamodule, PathLike) else None,
            task=task

        )

        self._datamodule: OTXDataModule | None = (
            datamodule if datamodule is not None else self._auto_configurator.get_datamodule()
        )
        self._model: OVModel = model if isinstance(model, OVModel) else self._auto_configurator.get_ov_model(model)

    def _derive_task_from_ir(self, ir_xml: PathLike) -> str:
        """Derives the task from the IR model.

        Args:
            model (PathLike): Path to the IR model.

        Returns:
            str: The derived task.
        """
        tree = ET.parse(ir_xml)
        root = tree.getroot()
        # Find <rt_info>
        rt_info = root.find("rt_info")
        if rt_info is None:
            msg = "No <rt_info> found in the IR model XML file. Please check the model file."
            raise ValueError(msg)

        # Extract values
        task_type = rt_info.find(".//task_type").attrib.get("value")
        multilabel = rt_info.find(".//multilabel").attrib.get("value") == "True"
        hierarchical = rt_info.find(".//hierarchical").attrib.get("value") == "True"
        # Derive task name
        if task_type == "classification":
            if hierarchical:
                task_name = "H_LABEL_CLS"
            elif multilabel:
                task_name = "MULTI_LABEL_CLS"
            else:
                task_name = "MULTI_CLASS_CLS"
        elif task_type == "segmentation":
            task_name = "SEMANTIC_SEGMENTATION"
        elif task_type == "detection":
            task_name = "DETECTION"
        elif task_type == "instance_segmentation":
            task_name = "INSTANCE_SEGMENTATION"
        elif task_type == "keypoint_detection":
            task_name = "KEYPOINT_DETECTION"
        else:
            msg = f"Unsupported task type: {task_type}. Please check the model file."
            raise ValueError(msg)

        return task_name

    def test(
        self,
        checkpoint: PathLike | None = None,
        datamodule: OTXDataModule | None = None,
        metric: MetricCallable | None = None,
    ) -> dict:
        r"""Run the testing phase of the engine.

        Args:
            checkpoint (PathLike | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            datamodule (OTXDataModule | None, optional): The data module containing the test data.
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
        datamodule = datamodule if datamodule is not None else self.datamodule
        metric = metric if metric is not None else self.model.metric_callable(label_info=datamodule.label_info)

        if datamodule is None:
            msg = "Please provide the `data_root` or `datamodule` when creating the Engine, or pass a `datamodule` in OVEngine.test()."
            raise RuntimeError(msg)
        if metric is None:
            msg = "Please provide a `metric` in OVEngine or pass it in OVEngine.test()."
            raise RuntimeError(msg)

        if checkpoint is None and self.model is None:
            msg = "Please provide either a model or a checkpoint path."
            raise ValueError(msg)
        elif checkpoint is not None and Path(str(checkpoint)).suffix not in [".xml", ".onnx"]:
            msg = "OV Engine supports only OV IR or ONNX checkpoints"
            raise RuntimeError(msg)
        elif checkpoint is not None:
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint))
        else:
            model = self.model

        if model.label_info != self.datamodule.label_info:
            msg = (
                "To launch a test pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)

        with Progress() as progress:
            dataloader = datamodule.test_dataloader()
            task = progress.add_task("Testing", total=len(dataloader))
            for data_batch in dataloader:
                preds = self.model(data_batch)
                metric_inputs = self.model.prepare_metric_inputs(preds, data_batch)
                metric.update(**metric_inputs)
                progress.update(task, advance=1)

        return self.model.compute_metrics(metric)

    def predict(
        self,
        data: OTXDataModule | list[np.array],
        checkpoint: PathLike | None = None,
        explain: bool = False,
        explain_config: ExplainConfig | None = None,
    ) -> list | None:
        r"""Run predictions using the specified model and data.

        Args:
            data (OTXDataModule | np.array): The data module or an numpy image to use for predictions.
            checkpoint (PathLike | None, optional): The path to the IR XML file to load the model from.
            explain (bool, optional): Whether to dump "saliency_map" and "feature_vector" or not.
            explain_config (ExplainConfig | None, optional): Explain configuration used for saliency map post-processing

        Returns:
            list: The predictions.
        """
        from otx.algo.utils.xai_utils import process_saliency_maps_in_pred_entity, set_crop_padded_map_flag

        datamodule = self._auto_configurator.get_datamodule(data_root=data) if data is not None else self.datamodule
        model = self._auto_configurator.get_ov_model(model_name=checkpoint) if checkpoint is not None else self.model

        if model.label_info != datamodule.label_info:
            msg = (
                "To launch a predict pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)

        predict_result = []
        if isinstance(data, OTXDataModule):
            dataloader = data.test_dataloader()
            for data_batch in dataloader:
                predict_result.append(model(data_batch))
        elif isinstance(data, list):
            if len(data) == 0:
                msg = "The input data is empty."
                raise ValueError(msg)
            if not isinstance(data[0], np.ndarray):
                msg = "The input data should be a list of numpy arrays."
                raise ValueError(msg)
            customized_inputs = TorchDataBatch(
                batch_size=len(data),
                images=[torch.tensor(im) for im in data], # TODO(@kprokofi): remove torch after numpy support
                imgs_info=[{"img_shape": img.shape} for img in data]
            )
            predict_result.append(model(customized_inputs))

        if explain:
            if explain_config is None:
                explain_config = ExplainConfig()
            explain_config = set_crop_padded_map_flag(explain_config, datamodule)

            predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config, datamodule.label_info)

        return predict_result

    def optimize(
        self,
        checkpoint: PathLike | None = None,
        datamodule: OTXDataModule | None = None,
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
        datamodule: OTXDataModule | None = None,
        explain_config: ExplainConfig | None = None,
        **kwargs,
    ) -> list | None:
        r"""Run XAI using the specified model and data (test subset).

        Args:
            checkpoint (PathLike | None, optional): The path to the checkpoint file to load the model from.
            datamodule (OTXDataModule | None, optional): The data module to use for predictions.
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
            model = self._auto_configurator.get_ov_model(model_name=str(checkpoint))

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

    @property
    def work_dir(self) -> PathLike:
        """Work directory."""
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: PathLike) -> None:
        self._work_dir = work_dir

    @property
    def model(self) -> OVModel:
        """Returns the model object associated with the engine.

        Returns:
            OVModel: The OVModel object.
        """
        return self._model

    @model.setter
    def model(self, model: OVModel | PathLike) -> None:
        """Sets the model for the engine.

        Args:
            model (OVModel | PathLike): The model to be set.

        Returns:
            None
        """
        if isinstance(model, PathLike):
            task = self._derive_task_from_ir(model)
            self._auto_configurator.task = task
            model = self._auto_configurator.get_ov_model(model)

        self._model = model
        self._auto_configurator.task = model.task
        self._auto_configurator._config = None

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

    @datamodule.setter
    def datamodule(self, datamodule: OTXDataModule | PathLike) -> None:
        """Sets the datamodule for the engine.

        Args:
            datamodule (OTXDataModule | PathLike): The datamodule to be set.

        Returns:
            None
        """
        if isinstance(datamodule, OTXDataModule) and datamodule.task != self._model.task:
            msg = (
                "The task of the provided datamodule does not match the task derived from the model. "
                f"datamodule.task={datamodule.task}, model.task={self._model.task}"
            )
            raise ValueError(msg)
        if isinstance(datamodule, PathLike):
            datamodule = self._auto_configurator.get_datamodule(data_root=datamodule)
        self._datamodule = datamodule
