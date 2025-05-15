# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OpenVINO engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import defusedxml.ElementTree as Elet
import numpy as np
import torch
from rich.progress import Progress

from otx.backend.native.utils.auto_configurator import AutoConfigurator
from otx.backend.openvino.models import OVModel
from otx.core.config.explain import ExplainConfig
from otx.core.data.entity.base import ImageInfo
from otx.core.data.module import OTXDataModule
from otx.core.types import OTXTaskType, PathLike
from otx.data.torch import OTXDataBatch

if TYPE_CHECKING:
    from otx.core.metrics import MetricCallable


class OVEngine:
    """OV Engine.

    This class defines the OV Engine for OTX, which governs each step of the OpenVINO validation workflow.
    """

    def __init__(
        self,
        data: OTXDataModule | PathLike | None = None,
        model: OVModel | PathLike | None = None,
        work_dir: PathLike = "./otx-workspace",
    ):
        """Initializes the OTX Engine.

        Args:
            data (OTXDataModule | PathLike | None, optional): The data module or path to the data root.
                If path is provided, Engine will automatically create a datamodule based on the data root and model.
            model (OTXModel | str | None, optional): The OV model for the engine.
                PathLike object to OpenVINO IR XML file can be provided. Defaults to None.
            work_dir (PathLike, optional): Working directory for the engine. Defaults to "./otx-workspace".
        """
        self._work_dir = work_dir
        if isinstance(model, str) and model.endswith(".xml"):
            task: OTXTaskType = self._derive_task_from_ir(model)
        elif isinstance(model, OVModel):
            task = model.task  # type: ignore[assignment]
        else:
            msg = "Model should be either a valid XML path or an instance of OVModel."
            raise ValueError(msg)

        self._auto_configurator = AutoConfigurator(
            data_root=data if isinstance(data, (str, os.PathLike)) else None,
            task=task,
        )

        if isinstance(data, OTXDataModule):
            if data.task != task:
                msg = (
                    "The task of the provided datamodule does not match the task derived from the model. "
                    f"datamodule.task={data.task}, model.task={task}"
                )
                raise ValueError(msg)
            self._datamodule: OTXDataModule | None = data
        else:
            self._datamodule = self._auto_configurator.get_datamodule() if data is not None else None
        if model is not None:
            self._model: OVModel = model if isinstance(model, OVModel) else self._auto_configurator.get_ov_model(model)
        else:
            self._model = None

    def _derive_task_from_ir(self, ir_xml: PathLike) -> OTXTaskType:
        """Derives the task from the IR model.

        Args:
            model (PathLike): Path to the IR model.

        Returns:
            str: The derived task.
        """
        tree = Elet.parse(ir_xml)
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

        return OTXTaskType(task_name)

    def test(
        self,
        data: OTXDataModule | PathLike | None = None,
        checkpoint: PathLike | None = None,
        metric: MetricCallable | None = None,
    ) -> dict:
        r"""Run the testing phase of the engine.

        Args:
            data (OTXDataModule | None, optional): The data to test on. It can be a data module
                or a path to the data root. If a path is provided, the engine will automatically
                create a datamodule based on the data root and model.
            checkpoint (PathLike | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            metric (MetricCallable | None): If not None, it will override `OTXModel.metric_callable` with the given
                metric callable. It will temporarilly change the evaluation metric for the validation and test.
            **kwargs: Additional keyword arguments for pl.Trainer configuration.

        Returns:
            dict: Dictionary containing the metrics after testing.

        """
        if isinstance(data, (str, os.PathLike)):
            datamodule = self._auto_configurator.get_datamodule(data_root=data)
        elif isinstance(data, OTXDataModule):
            datamodule = data
        else:
            datamodule = self.datamodule

        if datamodule is None:
            msg = "Please provide the `data` when creating the Engine, or pass it in OVEngine.test()."
            raise RuntimeError(msg)

        model = self._update_checkpoint(checkpoint)
        metric = metric if metric is not None else model.metric_callable

        if metric is None:
            msg = "Please provide a `metric` when creating a OVModel or pass it in OVEngine.test()."
            raise RuntimeError(msg)

        if model.label_info != datamodule.label_info:
            msg = (
                "To launch a test pipeline, the label information should be same "
                "between the training and testing datasets. "
                "Please check whether you use the same dataset: "
                f"model.label_info={model.label_info}, "
                f"datamodule.label_info={self.datamodule.label_info}"
            )
            raise ValueError(msg)

        metric_callable = metric(datamodule.label_info)
        with Progress() as progress:
            dataloader = datamodule.test_dataloader()
            task = progress.add_task("Testing", total=len(dataloader))
            for data_batch in dataloader:
                preds = model(data_batch)
                metric_inputs = model.prepare_metric_inputs(preds, data_batch)
                metric_callable.update(**metric_inputs)
                progress.update(task, advance=1)

        return model.compute_metrics(metric_callable)

    def predict(
        self,
        data: OTXDataModule | PathLike | list[np.array] | None = None,
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

        model = self._update_checkpoint(checkpoint)
        if isinstance(data, (str, os.PathLike)):
            data = self._auto_configurator.get_datamodule(data_root=data)

        datamodule = data if data is not None else self.datamodule

        predict_result = []
        with Progress() as progress:
            if isinstance(datamodule, OTXDataModule):
                if model.label_info != datamodule.label_info:
                    msg = (
                        "To launch a predict pipeline, the label information should be same "
                        "between the training and testing datasets. "
                        "Please check whether you use the same dataset: "
                        f"model.label_info={model.label_info}, "
                        f"datamodule.label_info={self.datamodule.label_info}"
                    )
                    raise ValueError(msg)
                dataloader = datamodule.test_dataloader()
                task = progress.add_task("Predicting", total=len(dataloader))
                for data_batch in dataloader:
                    predict_result.append(model(data_batch))
                    progress.update(task, advance=1)

            elif isinstance(datamodule, list):
                # list of numpy images will be considered as 1 batch
                task = progress.add_task("Predicting", total=1)
                if len(datamodule) == 0:
                    msg = "The input data is empty."
                    raise ValueError(msg)
                if not isinstance(datamodule[0], np.ndarray):
                    msg = "The input data should be a list of numpy arrays."
                    raise TypeError(msg)
                customized_inputs = OTXDataBatch(
                    batch_size=len(datamodule),
                    images=[
                        torch.tensor(img) for img in datamodule
                    ],  # TODO(@kprokofi): remove torch after numpy support
                    imgs_info=[
                        ImageInfo(img_idx=i, ori_shape=img.shape, img_shape=img.shape)
                        for i, img in enumerate(datamodule)
                    ],
                )
                predict_result.append(model(customized_inputs))
                progress.update(task, advance=1)
            else:
                msg = "The input data should be either a datamodule, valid path to data root or a list of numpy arrays."
                raise TypeError(msg)

        if explain and isinstance(datamodule, OTXDataModule):
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

        Returns:
            Path: path to the optimized model.
        """
        optimize_datamodule = datamodule if datamodule is not None else self.datamodule
        model = self._update_checkpoint(checkpoint)
        optimize_datamodule = self._auto_configurator.update_ov_subset_pipeline(
            datamodule=optimize_datamodule,
            subset="train",
        )

        ptq_config = {}
        if max_data_subset_size is not None:
            ptq_config["subset_size"] = max_data_subset_size

        return model.optimize(
            Path(self.work_dir),
            optimize_datamodule,
            ptq_config,
        )

    def _update_checkpoint(self, checkpoint: PathLike | None) -> OVModel:
        """Update the OVModel with the given checkpoint path.

        Args:
            checkpoint (PathLike): The new IR XML file path.

        Returns:
            None
        """
        if checkpoint is None and self.model is None:
            msg = "Please provide either a model or a checkpoint path."
            raise ValueError(msg)
        if checkpoint is not None and Path(str(checkpoint)).suffix not in [".xml", ".onnx"]:
            msg = "OV Engine supports only OV IR or ONNX checkpoints"
            raise RuntimeError(msg)
        if checkpoint is not None:
            return self._auto_configurator.get_ov_model(model_name=str(checkpoint))

        return self.model  # type: ignore[return-value]

    @property
    def work_dir(self) -> PathLike:
        """Work directory."""
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: PathLike) -> None:
        self._work_dir = work_dir

    @property
    def model(self) -> OVModel | None:
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
        if isinstance(model, (str, os.PathLike)):
            if not str(model).endswith(".xml"):
                msg = f"Model should be a valid XML path. But got: {model}"
                raise ValueError(msg)
            task = self._derive_task_from_ir(model)
            self._model = self._auto_configurator.get_ov_model(model, task)
        elif isinstance(model, OVModel):
            self._model = model
        else:
            msg = "Model should be either a valid XML path or an instance of OVModel."
            raise TypeError(msg)

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
        if isinstance(datamodule, (str, os.PathLike)):
            self._datamodule = self._auto_configurator.get_datamodule(data_root=datamodule)
        elif isinstance(datamodule, OTXDataModule):
            self._datamodule = datamodule
        else:
            msg = "Datamodule should be either a valid path to data root or an instance of OTXDataModule."
            raise TypeError(msg)
