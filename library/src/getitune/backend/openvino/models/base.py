# Copyright (C) 2023-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for base model entity used in getitune."""

from __future__ import annotations

import contextlib
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any

import nncf
import numpy as np
import openvino
import torch
from jsonargparse import ArgumentParser
from model_api.adapters import OpenvinoAdapter, create_core
from model_api.models import ImageModel, Model
from model_api.tilers import Tiler
from torch import Tensor

from getitune.data.entity.base import (
    ImageInfo,
)
from getitune.data.entity.sample import PredictionBatch, SampleBatch
from getitune.metrics import NullMetricCallable
from getitune.types.label import LabelInfo
from getitune.types.task import TaskType

from .utils import get_default_num_async_infer_requests

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from model_api.models.result import Result
    from torchmetrics import Metric, MetricCollection

    from getitune.data.module import DataModule
    from getitune.metrics import MetricCallable, MetricInput
    from getitune.types import PathLike

logger = logging.getLogger()


class OVModel:
    """Base class for the OpenVINO model.

    This is a base class representing interface for interacting with OpenVINO
    Intermediate Representation (IR) models. OVModel can create and validate
    OpenVINO IR model directly from provided path locally or from
    OpenVINO OMZ repository. (Only PyTorch models are supported).
    OVModel supports synchronous as well as asynchronous inference type.

    Args:
        num_classes: Number of classes this model can predict.
    """

    def __init__(
        self,
        model_path: PathLike,
        model_type: str,
        async_inference: bool = True,
        force_cpu: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = NullMetricCallable,
    ) -> None:
        """Initialize the OVModel instance.

        Args:
            model_path (PathLike): Path to the model file.
            model_type (str): Type of the model.
            async_inference (bool): Whether to enable asynchronous inference.
            force_cpu (bool): Whether to force the use of CPU.
            max_num_requests (int | None): Maximum number of inference requests.
            use_throughput_mode (bool): Whether to use throughput mode.
            model_api_configuration (dict[str, Any] | None): Configuration for the Model API.
            metric (MetricCallable): Metric callable for evaluation.
        """
        self.model_type = model_type
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.async_inference = async_inference
        self.num_requests = max_num_requests or get_default_num_async_infer_requests()
        self.use_throughput_mode = use_throughput_mode
        self.model_api_configuration = model_api_configuration or {}
        self.hparams: dict[str, Any] = {}
        self.model = self._create_model()
        self.metric_callable = metric
        self._label_info = self._create_label_info_from_ov_ir()
        self._task: TaskType | None = None
        tile_enabled = False
        with contextlib.suppress(RuntimeError):
            if isinstance(self.model, Model):
                tile_enabled = "tile_size" in self.model.inference_adapter.get_rt_info(["model_info"]).astype(dict)

        if tile_enabled:
            self._setup_tiler()

    def _setup_tiler(self) -> None:
        """Set up the tiler for tile-based tasks."""
        raise NotImplementedError

    @property
    def input_size(self) -> tuple[int, int] | None:
        """Return ``(H, W)`` input size from the underlying ModelAPI model.

        Returns ``None`` when the model uses dynamic shapes (h or w is
        non-positive, which ModelAPI encodes as 0 or -1) or when the
        underlying model attributes are not accessible.
        """
        try:
            base = self.model.model if isinstance(self.model, Tiler) else self.model
            h, w = int(base.h), int(base.w)  # pyrefly: ignore[missing-attribute]
        except (AttributeError, TypeError, ValueError):
            return None
        # ModelAPI uses 0 or -1 for dynamic dimensions; treat any
        # non-positive value as "dynamic" and return None.
        if h > 0 and w > 0:
            return (h, w)
        return None

    @property
    def keep_aspect_ratio(self) -> bool:
        """Return True when the model was exported with aspect-ratio-preserving resize.

        Reads ``resize_type`` from the underlying ModelAPI model parameters,
        which is embedded into the IR metadata at export time by
        ``ModelExporter._extend_model_metadata``.  Returns ``False`` when
        the attribute is not accessible (e.g. for custom / wrapped models).
        """
        _aspect_ratio_resize_types = ("fit_to_window", "fit_to_window_letterbox")

        base = self.model.model if isinstance(self.model, Tiler) else self.model
        resize_type = getattr(getattr(base, "params", None), "resize_type", None)
        return resize_type in _aspect_ratio_resize_types

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Read model configuration from the ModelAPI OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): Target adapter to read the configuration.
        """

    def _resolve_model_type(self, model_adapter: OpenvinoAdapter) -> str:
        """Resolve the ModelAPI wrapper name from IR metadata, falling back to the class default.

        New exports embed ``model_type`` into ``rt_info["model_info"]`` at export
        time (e.g. ``"SSD"`` for Lightning detection, ``"YOLO11"`` for Ultralytics).
        When the metadata is present it is authoritative; otherwise
        ``self.model_type`` (set by the subclass constructor default) is returned
        as a backward-compatible fallback for older IRs that lack the field.

        Args:
            model_adapter (OpenvinoAdapter): The adapter wrapping the loaded IR.

        Returns:
            str: The resolved ModelAPI model type name.
        """
        if model_adapter.model.has_rt_info(["model_info", "model_type"]):
            rt_model_type = model_adapter.model.get_rt_info(["model_info", "model_type"]).value
            if rt_model_type:
                resolved = str(rt_model_type)
                if resolved != self.model_type:
                    logger.info(
                        "Overriding default model_type '%s' with '%s' from IR metadata.",
                        self.model_type,
                        resolved,
                    )
                return resolved
        return self.model_type

    def _create_model(self) -> Model:
        """Create an OpenVINO model using the Model API.

        Returns:
            Model: The created OpenVINO model.
        """
        ov_device = "CPU"
        ie = create_core()
        if not self.force_cpu:
            devices = ie.available_devices
            for device in devices:
                device_name = ie.get_property(device_name=device, property="FULL_DEVICE_NAME")
                if "dGPU" in device_name and "Intel" in device_name:
                    ov_device = device
                    break

        plugin_config = {}
        if self.use_throughput_mode:
            plugin_config["PERFORMANCE_HINT"] = "THROUGHPUT"

        model_adapter = OpenvinoAdapter(
            ie,
            self.model_path,
            device=ov_device,
            max_num_requests=self.num_requests,
            plugin_config=plugin_config,
            model_parameters=self.model_adapter_parameters,
        )

        self._get_hparams_from_adapter(model_adapter)

        model_type = self._resolve_model_type(model_adapter)
        return Model.create_model(model_adapter, model_type=model_type, configuration=self.model_api_configuration)

    def _customize_inputs(self, entity: SampleBatch) -> dict[str, Any]:
        """Customize the input data for the model.

        Args:
            entity (SampleBatch): Input data batch.

        Returns:
            dict[str, Any]: Customized input data.
        """
        images = [np.transpose(im.cpu().numpy(), (1, 2, 0)) for im in entity.images]
        return {"inputs": images}

    def _customize_outputs(
        self,
        outputs: list[Result],
        inputs: SampleBatch,
    ) -> PredictionBatch:
        """Customize the model outputs to getitune format.

        Args:
            outputs (list[Result]): The model outputs.
            inputs (SampleBatch): The input batch entity.

        Returns:
            PredictionBatch: The customized prediction batch entity.
        """
        return PredictionBatch(
            images=inputs.images,
            imgs_info=inputs.imgs_info,
        )

    def forward(self, inputs: SampleBatch, async_inference: bool = True) -> PredictionBatch:
        """Perform forward pass of the model.

        Args:
            inputs (SampleBatch): Input data batch.
            async_inference (bool): Whether to use asynchronous inference.

        Returns:
            PredictionBatch: Model predictions.
        """
        async_inference = async_inference and self.async_inference
        numpy_inputs = self._customize_inputs(inputs)["inputs"]
        outputs = self.model.infer_batch(numpy_inputs) if async_inference else [self.model(im) for im in numpy_inputs]

        return self._customize_outputs(outputs, inputs)

    def optimize(
        self,
        output_dir: Path,
        data_module: DataModule,
        ptq_config: dict[str, Any] | None = None,
        optimized_model_name: str = "optimized_model",
    ) -> Path:
        """Optimize the model using NNCF quantization.

        Args:
            output_dir (Path): Directory to save the optimized model.
            data_module (DataModule): Data module for training data.
            ptq_config (dict[str, Any] | None): PTQ configuration.
            optimized_model_name (str): Name of the optimized model.

        Returns:
            Path: Path to the optimized model.
        """
        output_model_path = output_dir / (optimized_model_name + ".xml")

        def check_if_quantized(model: openvino.Model) -> bool:
            """Check if the OpenVINO model is already quantized.

            Args:
                model (openvino.Model): OpenVINO model.

            Returns:
                bool: True if the model is quantized, False otherwise.
            """
            nodes = model.get_ops()
            return any(op.get_type_name() == "FakeQuantize" for op in nodes)

        ov_model = openvino.Core().read_model(self.model_path)

        if check_if_quantized(ov_model):
            msg = "Model is already optimized by PTQ"
            raise RuntimeError(msg)

        train_dataset = data_module.train_dataloader()

        ptq_config_from_ir = self._read_ptq_config_from_ir(ov_model)
        if ptq_config is not None:
            ptq_config_from_ir.update(ptq_config)
            ptq_config = ptq_config_from_ir
        else:
            ptq_config = ptq_config_from_ir

        quantization_dataset = nncf.Dataset(train_dataset, self.transform_fn)

        if ptq_config.get("max_drop") is not None:
            validation_dataset = nncf.Dataset(data_module.val_dataloader(), self.transform_fn)
            validation_fn = self._create_validation_fn(data_module)
            compressed_model = nncf.quantize_with_accuracy_control(
                model=ov_model,
                calibration_dataset=quantization_dataset,
                validation_dataset=validation_dataset,
                validation_fn=validation_fn,
                **ptq_config,
            )
        else:
            compressed_model = nncf.quantize(
                model=ov_model,
                calibration_dataset=quantization_dataset,
                **ptq_config,
            )

        openvino.save_model(compressed_model, output_model_path)

        return output_model_path

    def _create_validation_fn(
        self, data_module: DataModule
    ) -> Callable[[openvino.CompiledModel, Any], tuple[float, None]]:
        """Create a validation function for accuracy-aware quantization.

        The returned function computes accuracy on the validation set using the
        compiled model provided by NNCF. It is compatible with
        ``nncf.quantize_with_accuracy_control``.

        Args:
            data_module (DataModule): Data module providing the validation dataloader.

        Returns:
            Callable: A function ``(compiled_model, validation_dataset) -> (float, None)``
            where the float is the primary accuracy metric value.
        """

        def _infer_compiled_model(
            compiled_model: openvino.CompiledModel,
            inputs: SampleBatch,
        ) -> PredictionBatch:
            """Run inference using the NNCF-provided compiled model.

            This replaces ``self.forward`` so that accuracy is measured on the
            quantized candidate, not the original FP model.

            Args:
                compiled_model: Compiled OpenVINO model supplied by NNCF.
                inputs: Input data batch.

            Returns:
                PredictionBatch with predictions from the compiled model.
            """
            numpy_inputs = self._customize_inputs(inputs)["inputs"]
            infer_request = compiled_model.create_infer_request()
            outputs: list[Result] = []
            for image in numpy_inputs:
                model_ref: ImageModel = self.model.model if isinstance(self.model, Tiler) else self.model  # type: ignore[assignment]
                resized = model_ref.resize(image, (model_ref.w, model_ref.h))
                resized = model_ref.input_transform(resized)
                input_tensor = model_ref._change_layout(resized)  # noqa: SLF001
                infer_request.infer({0: input_tensor})
                raw_result = {
                    out.get_any_name(): infer_request.get_tensor(out).data.copy() for out in compiled_model.outputs
                }
                result = model_ref.postprocess(raw_result, {"original_shape": image.shape})
                outputs.append(result)
            return self._customize_outputs(outputs, inputs)

        def validation_fn(
            compiled_model: openvino.CompiledModel,
            validation_dataset: nncf.Dataset,  # noqa: ARG001 required by NNCF signature
        ) -> tuple[float, None]:
            """Evaluate the compiled OpenVINO model on the validation dataset.

            Args:
                compiled_model: Compiled OpenVINO model provided by NNCF during
                    accuracy-aware quantization.
                validation_dataset: Validation NNCF dataset (unused, we iterate
                    via the dataloader for proper batching and transforms).

            Returns:
                Tuple of (metric_value, None).
            """
            metric = self.metric_callable(data_module.label_info)

            val_dataloader = data_module.val_dataloader()
            for data_batch in val_dataloader:
                preds = _infer_compiled_model(compiled_model, data_batch)
                metric_inputs = self.prepare_metric_inputs(preds, data_batch)
                if isinstance(metric_inputs, list):
                    for mi in metric_inputs:
                        metric.update(**mi)
                else:
                    metric.update(**metric_inputs)

            results = self.compute_metrics(metric)
            # Take the first scalar metric value as the accuracy indicator
            metric_value = next(iter(results.values()))
            if isinstance(metric_value, torch.Tensor):
                metric_value = metric_value.item()

            return float(metric_value), None

        return validation_fn

    def transform_fn(self, data_batch: SampleBatch) -> np.array:
        """Transform data for PTQ.

        Args:
            data_batch (SampleBatch): Input data batch.

        Returns:
            np.array: Transformed data.
        """
        np_data = self._customize_inputs(data_batch)
        image = np_data["inputs"][0]
        model: ImageModel = self.model.model if isinstance(self.model, Tiler) else self.model  # type: ignore[assignment]
        resized_image = model.resize(image, (model.w, model.h))
        resized_image = model.input_transform(resized_image)
        return model._change_layout(resized_image)  # noqa: SLF001

    def _read_ptq_config_from_ir(self, ov_model: Model) -> dict[str, Any]:
        """Generate PTQ configuration from the OpenVINO model metadata.

        Args:
            ov_model (Model): OpenVINO model.

        Returns:
            dict[str, Any]: PTQ configuration.
        """
        from nncf import IgnoredScope  # type: ignore[attr-defined]
        from nncf.common.quantization.structs import QuantizationPreset  # type: ignore[attr-defined]
        from nncf.parameters import ModelType
        from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

        if "optimization_config" not in ov_model.rt_info["model_info"]:
            return {}

        initial_ptq_config = json.loads(ov_model.rt_info["model_info"]["optimization_config"].value)
        if not initial_ptq_config:
            return {}
        argparser = ArgumentParser()
        if "advanced_parameters" in initial_ptq_config:
            argparser.add_class_arguments(AdvancedQuantizationParameters, "advanced_parameters")
        if "preset" in initial_ptq_config:
            initial_ptq_config["preset"] = QuantizationPreset(initial_ptq_config["preset"])
            argparser.add_argument("--preset", type=QuantizationPreset)
        if "model_type" in initial_ptq_config:
            initial_ptq_config["model_type"] = ModelType(initial_ptq_config["model_type"])
            argparser.add_argument("--model_type", type=ModelType)
        if "ignored_scope" in initial_ptq_config:
            argparser.add_class_arguments(IgnoredScope, "ignored_scope", as_positional=True)

        initial_ptq_config = argparser.parse_object(initial_ptq_config)

        return argparser.instantiate_classes(initial_ptq_config).as_dict()

    def prepare_metric_inputs(
        self,
        preds: PredictionBatch,
        inputs: SampleBatch,
    ) -> MetricInput:
        """Prepare inputs for metric computation.

        Args:
            preds (PredictionBatch): Predicted batch entity.
            inputs (SampleBatch): Input batch entity.

        Returns:
            MetricInput: Dictionary containing predictions and targets.
        """
        raise NotImplementedError

    def compute_metrics(self, metric: Metric | MetricCollection) -> dict:
        """Compute metrics using the provided metric object.

        Args:
            metric (Metric | MetricCollection): Metric object.

        Returns:
            dict: Computed metrics.
        """
        return self._compute_metrics(metric)

    def _compute_metrics(self, metric: Metric | MetricCollection, **compute_kwargs) -> dict:
        """Compute metrics with additional arguments.

        Args:
            metric (Metric | MetricCollection): Metric object.
            **compute_kwargs: Additional arguments for metric computation.

        Returns:
            dict: Computed metrics.
        """
        sig = inspect.signature(metric.compute)
        filtered_kwargs = {key: value for key, value in compute_kwargs.items() if key in sig.parameters}
        if removed_kwargs := set(compute_kwargs.keys()).difference(filtered_kwargs.keys()):
            msg = f"These keyword arguments are removed since they are not in the function signature: {removed_kwargs}"
            logger.debug(msg)

        results: dict[str, Tensor] = metric.compute(**filtered_kwargs)

        if not isinstance(results, dict):
            raise TypeError(results)

        if not results:
            msg = f"{metric} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)
        return results

    @property
    def model_adapter_parameters(self) -> dict:
        """Get model parameters for export.

        Returns:
            dict: Model parameters.
        """
        return {}

    @property
    def label_info(self) -> LabelInfo:
        """Get label information of the model.

        Returns:
            LabelInfo: Label information.
        """
        return self._label_info

    @property
    def task(self) -> TaskType | None:
        """Get the task type of the model.

        Returns:
            TaskType | None: Task type.
        """
        return self._task

    def _create_label_info_from_ov_ir(self) -> LabelInfo:
        """Create label information from the OpenVINO IR.

        Returns:
            LabelInfo: Label information.

        Raises:
            ValueError: If label information cannot be constructed.
        """
        ov_model = self.model.get_model()

        if ov_model.has_rt_info(["model_info", "label_info"]):
            serialized = ov_model.get_rt_info(["model_info", "label_info"]).value
            return LabelInfo.from_json(serialized)

        mapi_model: Model = self.model

        if label_names := getattr(mapi_model, "labels", None):
            msg = (
                'Cannot find "label_info" from OpenVINO IR. '
                "However, we found labels attributes from ModelAPI. "
                "Construct LabelInfo from it."
            )

            logger.warning(msg)
            return LabelInfo(label_names=label_names, label_groups=[label_names], label_ids=[])

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by getitune."
        raise ValueError(msg)

    def get_dummy_input(self, batch_size: int = 1) -> SampleBatch:
        """Generate a dummy input for the model.

        Args:
            batch_size (int): Batch size for the dummy input.

        Returns:
            SampleBatch: Dummy input data.
        """
        images = torch.stack([torch.rand(3, 224, 224) for _ in range(batch_size)])
        img_shape = (224, 224)
        infos = [ImageInfo(img_idx=i, img_shape=img_shape, ori_shape=img_shape) for i in range(batch_size)]
        return SampleBatch(images=images, imgs_info=infos)

    def __call__(self, *args, **kwds):
        """Call the model for inference.

        Args:
            *args: Positional arguments.
            **kwds: Keyword arguments.

        Returns:
            Any: Model output.
        """
        return self.forward(*args, **kwds)
