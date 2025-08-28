# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO engine."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import defusedxml.ElementTree as Elet
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
from rich.progress import Progress

from otx.backend.openvino.models import OVModel
from otx.config.explain import ExplainConfig
from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataBatch
from otx.data.module import OTXDataModule
from otx.engine import Engine
from otx.tools.auto_configurator import AutoConfigurator
from otx.types import OTXTaskType, PathLike

if TYPE_CHECKING:
    from otx.metrics import MetricCallable
    from otx.types.types import ANNOTATIONS, DATA, METRICS, MODEL


class OVEngine(Engine):
    """OV Engine.

    This class defines the OV Engine for OTX, which governs each step of the OpenVINO validation workflow.
    """

    def __init__(
        self,
        data: OTXDataModule | PathLike,
        model: OVModel | PathLike,
        work_dir: PathLike = "./otx-workspace",
    ):
        """Initialize the OVEngine.

        Args:
            data (OTXDataModule | PathLike): The data module or path to the data root.
                If a path is provided, the engine will automatically create a datamodule
                based on the data root and model.
            model (OVModel | PathLike): The OV model for the engine.
                A PathLike object to an OpenVINO IR XML file can also be provided. Defaults to None.
            work_dir (PathLike, optional): Working directory for the engine. Defaults to "./otx-workspace".
        """
        self._work_dir = work_dir
        if isinstance(model, (str, os.PathLike)) and Path(model).suffix in [".xml"]:
            task: OTXTaskType | None = self._derive_task_from_ir(model)
        elif isinstance(model, OVModel):
            task = model.task  # type: ignore[assignment]
        else:
            msg = "Please provide a valid OpenVINO model or a path to an OpenVINO IR XML file."
            raise ValueError(msg)
        self._auto_configurator = AutoConfigurator(
            data_root=data if isinstance(data, (str, os.PathLike)) else None,
            task=task,
        )

        if isinstance(data, OTXDataModule):
            if task is not None and data.task != task:
                if "anomaly" in task.value.lower() and "anomaly" in data.task.value.lower():
                    # Anomaly task can be derived from the model, but the datamodule may have a different task type
                    # (e.g., anomaly_classification, anomaly_segmentation, etc.)
                    pass
                else:
                    msg = (
                        "The task of the provided datamodule does not match the task derived from the model. "
                        f"datamodule.task={data.task}, model.task={task}"
                    )
                    raise ValueError(msg)
            self._datamodule: OTXDataModule | None = data
        else:
            self._datamodule = self._auto_configurator.get_datamodule()

        self._model: OVModel = model if isinstance(model, OVModel) else self._auto_configurator.get_ov_model(model)

    def _derive_task_from_ir(self, ir_xml: PathLike) -> OTXTaskType:
        """Derive the task type from the IR model XML file.

        Args:
            ir_xml (PathLike): Path to the IR model XML file.

        Returns:
            OTXTaskType: The derived task type.

        Raises:
            ValueError: If the task type is unsupported or the XML file is invalid.
        """
        task_map = {
            "classification_hcl": OTXTaskType.H_LABEL_CLS,
            "classification_mlc": OTXTaskType.MULTI_LABEL_CLS,
            "classification_mc": OTXTaskType.MULTI_CLASS_CLS,
            "segmentation": OTXTaskType.SEMANTIC_SEGMENTATION,
            "detection": OTXTaskType.DETECTION,
            "instance_segmentation": OTXTaskType.INSTANCE_SEGMENTATION,
            "keypoint_detection": OTXTaskType.KEYPOINT_DETECTION,
            "anomaly": OTXTaskType.ANOMALY,
        }

        tree = Elet.parse(ir_xml)
        root = tree.getroot()
        rt_info = root.find("rt_info")
        if rt_info is None:
            msg = "No <rt_info> found in the IR model XML file. Please check the model file."
            raise ValueError(msg)

        task_type = rt_info.find(".//task_type")
        if task_type is None:
            # check old Anomaly models
            model_info = rt_info.find(".//model_info")
            if model_info:
                model_type = model_info.find(".//model_type")
                if model_type is not None and "anomaly" in model_type.attrib.get("value", "").lower():
                    task_type = "anomaly"
            if task_type is None:
                msg = (
                    "No <task_type> found in the IR model XML file. Please check the model file."
                    "Task cannot be derived from the model."
                )
                raise ValueError(msg)
        else:
            task_type = task_type.attrib.get("value")

        if task_type == "classification":
            if rt_info.find(".//hierarchical").attrib.get("value") == "True":
                otx_task_name = task_type + "_hcl"
            elif rt_info.find(".//multilabel").attrib.get("value") == "True":
                otx_task_name = task_type + "_mlc"
            else:
                otx_task_name = task_type + "_mc"
        else:
            otx_task_name = task_type

        if otx_task_name not in task_map:
            msg = f"Unsupported task type '{otx_task_name}' derived from the IR model XML file."
            raise ValueError(msg)

        return task_map[otx_task_name]

    def train(self, *args, **kwargs) -> METRICS:
        """Train method is not supported for OVEngine."""
        msg = "OVEngine does not support training. Use test or predict methods to evaluate IR model."
        raise NotImplementedError(msg)

    def export(self, *args, **kwargs) -> Path:
        """Export method is not supported for OVEngine."""
        msg = "OVEngine does not support export."
        raise NotImplementedError(msg)

    def test(
        self,
        data: OTXDataModule | PathLike | None = None,
        checkpoint: PathLike | None = None,
        metric: MetricCallable | None = None,
        **kwargs,
    ) -> METRICS:
        """Run the testing phase of the engine.

        Args:
            data (OTXDataModule | PathLike | None, optional): The data to test on. It can be a data module
                or a path to the data root. If a path is provided, the engine will automatically
                create a datamodule based on the data root and model.
            checkpoint (PathLike | None, optional): Path to the checkpoint file to load the model from.
                Defaults to None.
            metric (MetricCallable | None, optional): If provided, overrides `OTXModel.metric_callable` with the given
                metric callable for evaluation.

        Returns:
            METRICS: The computed metrics after testing the model on the provided data.
                (dictionary of metric names and values)

        Raises:
            RuntimeError: If required data or metric is not provided.
            ValueError: If label information between model and datamodule does not match.
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
        metric = metric or model.metric_callable

        datamodule = self._auto_configurator.update_ov_subset_pipeline(
            datamodule=datamodule,
            subset="test",
            task=model.task,
        )

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
                if isinstance(metric_inputs, list):
                    for metric_input in metric_inputs:
                        metric_callable.update(**metric_input)
                else:
                    metric_callable.update(**metric_inputs)
                progress.update(task, advance=1)

        metrics_result = model.compute_metrics(metric_callable)

        self.log_results(metrics_result)

        return metrics_result

    def log_results(self, metrics: METRICS) -> None:
        """Log testing phase results to a CSV file.

        This function behaves similarly to `OTXModel._log_metrics(metrics, key="test")`.
        """
        clean = {}
        for k, v in metrics.items():
            metric_name = f"test/{k}"
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    clean[metric_name] = v.item()
                else:
                    continue  # or flatten/log each value separately
            else:
                clean[metric_name] = v

        logger = CSVLogger(self.work_dir, name="csv/", prefix="")
        logger.log_metrics(clean, step=0)
        logger.finalize("success")

    def predict(
        self,
        data: OTXDataModule | PathLike | list[np.array] | None = None,
        checkpoint: PathLike | None = None,
        explain: bool = False,
        explain_config: ExplainConfig | None = None,
        **kwargs,
    ) -> ANNOTATIONS:
        """Run predictions using the specified model and data.

        Args:
            data (OTXDataModule | PathLike | list[np.array] | None, optional): The data module, path to data root,
                or a list of numpy images to use for predictions.
            checkpoint (PathLike | None, optional): The path to the IR XML file to load the model from.
            explain (bool, optional): Whether to generate "saliency_map" and "feature_vector". Defaults to False.
            explain_config (ExplainConfig | None, optional): Configuration for saliency map post-processing.

        Returns:
            ANNOTATIONS: The predictions made by the model on the provided data.
                (list of OTXPredEntity)

        Raises:
            ValueError: If input data is invalid or label information does not match.
            TypeError: If input data type is unsupported.
        """
        from otx.backend.native.models.utils.xai_utils import process_saliency_maps_in_pred_entity

        model = self._update_checkpoint(checkpoint)
        if isinstance(data, (str, os.PathLike)):
            data = self._auto_configurator.get_datamodule(data_root=data)

        datamodule = data or self.datamodule

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
                datamodule = self._auto_configurator.update_ov_subset_pipeline(
                    datamodule=datamodule,
                    subset="test",
                    task=model.task,
                )
                dataloader = datamodule.test_dataloader()
                task = progress.add_task("Predicting", total=len(dataloader))
                for data_batch in dataloader:
                    predict_result.append(model(data_batch))
                    progress.update(task, advance=1)

            elif isinstance(datamodule, list):
                task = progress.add_task("Predicting", total=1)
                if len(datamodule) == 0:
                    msg = "The input data is empty."
                    raise ValueError(msg)
                if not isinstance(datamodule[0], np.ndarray):
                    msg = "The input data should be a list of numpy arrays."
                    raise TypeError(msg)
                customized_inputs = OTXDataBatch(
                    batch_size=len(datamodule),
                    images=[torch.tensor(img) for img in datamodule],
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
            predict_result = process_saliency_maps_in_pred_entity(predict_result, explain_config, datamodule.label_info)

        return predict_result

    def optimize(
        self,
        checkpoint: PathLike | None = None,
        datamodule: OTXDataModule | None = None,
        max_data_subset_size: int | None = None,
    ) -> Path:
        """Apply Post-Training Quantization (PTQ) to optimize the model.

        PTQ performs int-8 quantization on the input model, resulting in mixed precision.

        Args:
            checkpoint (PathLike | None, optional): Checkpoint to optimize. Defaults to None.
            datamodule (OTXDataModule | None, optional): The data module to use for optimization.
            max_data_subset_size (int | None, optional): Maximum size of the train subset used for optimization.
                Defaults to None.

        Returns:
            Path: Path to the optimized model.
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

    @staticmethod
    def is_supported(model: MODEL, data: DATA) -> bool:
        """Check if the engine is supported for the given model and data."""
        check_model = False
        check_data = False
        if isinstance(model, OVModel):
            check_model = True
        elif isinstance(model, (str, os.PathLike)):
            model_path = Path(model)
            check_model = model_path.suffix in [".xml"]
        if isinstance(data, OTXDataModule):
            check_data = True
        elif isinstance(data, (str, os.PathLike)):
            data_path = Path(data)
            check_data = data_path.is_dir()

        return check_model and check_data

    def _update_checkpoint(self, checkpoint: PathLike | None) -> OVModel:
        """Update the OVModel with the given checkpoint path.

        Args:
            checkpoint (PathLike | None): The new IR XML file path.

        Returns:
            OVModel: The updated OVModel instance.

        Raises:
            ValueError: If no model or checkpoint is provided.
            RuntimeError: If the checkpoint file format is unsupported.
        """
        if checkpoint is None and self.model is None:
            msg = "Please provide either a model or a checkpoint path."
            raise ValueError(msg)
        if checkpoint is not None and Path(str(checkpoint)).suffix not in [".xml"]:
            msg = "OV Engine supports only OV IR checkpoints"
            raise RuntimeError(msg)
        if checkpoint is not None:
            task = self._derive_task_from_ir(checkpoint)
            return self._auto_configurator.get_ov_model(model_name=str(checkpoint), task=task)

        return self.model  # type: ignore[return-value]

    @property
    def work_dir(self) -> PathLike:
        """Get the working directory.

        Returns:
            PathLike: The working directory path.
        """
        return self._work_dir

    @work_dir.setter
    def work_dir(self, work_dir: PathLike) -> None:
        """Set the working directory.

        Args:
            work_dir (PathLike): The new working directory path.
        """
        self._work_dir = work_dir

    @property
    def model(self) -> OVModel:
        """Get the model associated with the engine.

        Returns:
            OVModel: The OVModel object or None if not set.
        """
        return self._model

    @property
    def datamodule(self) -> OTXDataModule:
        """Get the datamodule associated with the engine.

        Returns:
            OTXDataModule: The OTXDataModule object.

        Raises:
            RuntimeError: If the datamodule is not set.
        """
        if self._datamodule is None:
            msg = "Please include the `data_root` or `datamodule` when creating the Engine."
            raise RuntimeError(msg)
        return self._datamodule
