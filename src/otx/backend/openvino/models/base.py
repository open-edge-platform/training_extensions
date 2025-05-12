# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Class definition for base model entity used in OTX."""

# mypy: disable-error-code="arg-type"

from __future__ import annotations

import contextlib
import inspect
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import openvino
import torch
from jsonargparse import ArgumentParser
from model_api.adapters import OpenvinoAdapter, create_core
from model_api.models import Model
from model_api.tilers import Tiler
from torch import Tensor
from torchmetrics import Metric

from otx.core.data.entity.base import (
    ImageInfo,
    OTXBatchDataEntity,
)
from otx.core.exporter.native import OTXNativeModelExporter
from otx.core.metrics import NullMetricCallable
from otx.core.types.export import OTXExportFormatType
from otx.core.types.label import LabelInfo
from otx.core.types.precision import OTXPrecisionType
from otx.core.types.task import OTXTaskType
from otx.core.utils.build import get_default_num_async_infer_requests
from otx.data.torch import TorchDataBatch, TorchPredBatch

if TYPE_CHECKING:
    from pathlib import Path

    from model_api.adapters import OpenvinoAdapter

    from otx.core.data.module import OTXDataModule
    from otx.core.metrics import MetricCallable

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
        model_path: str,
        model_type: str,
        async_inference: bool = True,
        force_cpu: bool = True,
        max_num_requests: int | None = None,
        use_throughput_mode: bool = True,
        model_api_configuration: dict[str, Any] | None = None,
        metric: MetricCallable = NullMetricCallable,
    ) -> None:
        self.model_type = model_type
        self.model_path = model_path
        self.force_cpu = force_cpu
        self.async_inference = async_inference
        self.num_requests = max_num_requests if max_num_requests is not None else get_default_num_async_infer_requests()
        self.use_throughput_mode = use_throughput_mode
        self.model_api_configuration = model_api_configuration if model_api_configuration is not None else {}
        self.model = self._create_model()
        self.metric_callable = metric
        self._label_info = self._create_label_info_from_ov_ir()
        self._task = None
        tile_enabled = False
        with contextlib.suppress(RuntimeError):
            if isinstance(self.model, Model):
                tile_enabled = "tile_size" in self.model.inference_adapter.get_rt_info(["model_info"]).astype(dict)

        if tile_enabled:
            self._setup_tiler()

    def _setup_tiler(self) -> None:
        """Setup tiler for tile task."""
        raise NotImplementedError

    def _get_hparams_from_adapter(self, model_adapter: OpenvinoAdapter) -> None:
        """Reads model configuration from ModelAPI OpenVINO adapter.

        Args:
            model_adapter (OpenvinoAdapter): target adapter to read the config
        """

    def _create_model(self) -> Model:
        """Create a OV model with help of Model API."""
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

        return Model.create_model(model_adapter, model_type=self.model_type, configuration=self.model_api_configuration)

    def _customize_inputs(self, entity: TorchDataBatch) -> dict[str, Any]:
        # restore original numpy image
        images = [np.transpose(im.cpu().numpy(), (1, 2, 0)) for im in entity.images]
        return {"inputs": images}

    def forward(self, inputs: TorchDataBatch, async_inference: bool = True) -> TorchPredBatch:
        """Model forward function."""
        async_inference = async_inference and self.async_inference
        numpy_inputs = self._customize_inputs(inputs)["inputs"]
        outputs = self.model.infer_batch(numpy_inputs) if async_inference else [self.model(im) for im in numpy_inputs]
        customized_outputs = self._customize_outputs(outputs, inputs)

        return customized_outputs

    def optimize(
        self,
        output_dir: Path,
        data_module: OTXDataModule,
        ptq_config: dict[str, Any] | None = None,
        optimized_model_name: str = "optimized_model",
    ) -> Path:
        """Runs NNCF quantization."""
        import nncf

        output_model_path = output_dir / (optimized_model_name + ".xml")

        def check_if_quantized(model: openvino.Model) -> bool:
            """Checks if OpenVINO model is already quantized."""
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

        quantization_dataset = nncf.Dataset(train_dataset, self.transform_fn)  # type: ignore[attr-defined]

        compressed_model = nncf.quantize(  # type: ignore[attr-defined]
            ov_model,
            quantization_dataset,
            **ptq_config,
        )

        openvino.save_model(compressed_model, output_model_path)

        return output_model_path

    def export(
        self,
        output_dir: Path,
        base_name: str,
        export_format: OTXExportFormatType,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        to_exportable_code: bool = True,
    ) -> Path:
        """Export this model to the specified output directory.

        Args:
            output_dir (Path): directory for saving the exported model
            base_name: (str): base name for the exported model file. Extension is defined by the target export format
            export_format (OTXExportFormatType): format of the output model
            precision (OTXExportPrecisionType): precision of the output model
            to_exportable_code (bool): whether to generate exportable code with demo package.
                OpenVINO model supports only exportable code option.

        Returns:
            Path: path to the exported model.
        """
        if not to_exportable_code:
            msg = "OpenVINO model can be exported only as exportable code with demo package."
            raise RuntimeError(msg)

        # Temporarily unwrap Tiler model if applicable
        original_model = self.model
        if isinstance(original_model, Tiler):
            self.model = original_model.model

        try:
            exported_path = self._exporter.export(
                self.model,
                output_dir,
                base_name,
                export_format,
                precision,
                to_exportable_code,
            )
        finally:
            # Restore the original model
            self.model = original_model

        return exported_path

    def transform_fn(self, data_batch: TorchDataBatch) -> np.array:
        """Data transform function for PTQ."""
        np_data = self._customize_inputs(data_batch)
        image = np_data["inputs"][0]
        # NOTE: Tiler wraps the model, so we need to unwrap it to get the model
        model = self.model.model if isinstance(self.model, Tiler) else self.model
        resized_image = model.resize(image, (model.w, model.h))
        resized_image = model.input_transform(resized_image)
        return model._change_layout(resized_image)  # noqa: SLF001

    def _read_ptq_config_from_ir(self, ov_model: Model) -> dict[str, Any]:
        """Generates the PTQ (Post-Training Quantization) configuration from the meta data of the given OpenVINO model.

        Args:
            ov_model (Model): The OpenVINO model in which the PTQ configuration is embedded.

        Returns:
            dict: The PTQ configuration as a dictionary.
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

    def compute_metrics(self, meter: Metric) -> dict:
        return self._compute_metrics(meter)

    def _compute_metrics(self, meter: Metric, **compute_kwargs) -> dict:
        sig = inspect.signature(meter.compute)
        filtered_kwargs = {key: value for key, value in compute_kwargs.items() if key in sig.parameters}
        if removed_kwargs := set(compute_kwargs.keys()).difference(filtered_kwargs.keys()):
            msg = f"These keyword arguments are removed since they are not in the function signature: {removed_kwargs}"
            logger.debug(msg)

        results: dict[str, Tensor] = meter.compute(**filtered_kwargs)

        if not isinstance(results, dict):
            raise TypeError(results)

        if not results:
            msg = f"{meter} has no data to compute metric or there is an error computing metric"
            raise RuntimeError(msg)
        return results

    @property
    def _exporter(self) -> OTXNativeModelExporter:
        """Exporter of the OVModel for exportable code."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
        )

    @property
    def model_adapter_parameters(self) -> dict:
        """Model parameters for export."""
        return {}

    @property
    def label_info(self) -> LabelInfo:
        """Get this model label information."""
        return self._label_info

    @property
    def task(self) -> OTXTaskType | None:
        """Get task of the model."""
        return self._task

    def _create_label_info_from_ov_ir(self) -> LabelInfo:
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

        msg = "Cannot construct LabelInfo from OpenVINO IR. Please check this model is trained by OTX."
        raise ValueError(msg)

    def get_dummy_input(self, batch_size: int = 1) -> OTXBatchDataEntity:
        """Returns a dummy input for base OV model."""
        # Resize is embedded to the OV model, which means we don't need to know the actual size
        images = [torch.rand(3, 224, 224) for _ in range(batch_size)]
        infos = []
        for i, img in enumerate(images):
            infos.append(
                ImageInfo(
                    img_idx=i,
                    img_shape=img.shape,
                    ori_shape=img.shape,
                ),
            )
        return OTXBatchDataEntity(batch_size=batch_size, images=images, imgs_info=infos)

    def __call__(self, *args, **kwds):
        """Call the model."""
        return self.forward(*args, **kwds)
