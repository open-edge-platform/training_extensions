# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for native model exporter used in getitune."""

from __future__ import annotations

import logging as log
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import onnx
import openvino
import torch

from getitune.backend.lightning.exporter.base import ModelExporter
from getitune.types.export import TaskLevelExportParameters
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from getitune.backend.lightning.models.base import DataInputParams, LightningModel


class LightningModelExporter(ModelExporter):
    """Exporter that uses native torch and OpenVINO conversion tools."""

    def __init__(
        self,
        task_level_export_parameters: TaskLevelExportParameters,
        data_input_params: DataInputParams,
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        via_onnx: bool = False,
        onnx_export_configuration: dict[str, Any] | None = None,
        output_names: list[str] | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        self.via_onnx = via_onnx
        self.onnx_export_configuration = onnx_export_configuration if onnx_export_configuration is not None else {}

        # Sync output_names and input_names from onnx_export_configuration if not explicitly provided
        # This ensures they are used for both ONNX export and direct OpenVINO conversion
        if output_names is None and "output_names" in self.onnx_export_configuration:
            output_names = self.onnx_export_configuration["output_names"]
        if input_names is None and "input_names" in self.onnx_export_configuration:
            input_names = self.onnx_export_configuration["input_names"]

        super().__init__(
            task_level_export_parameters=task_level_export_parameters,
            data_input_params=data_input_params,
            resize_mode=resize_mode,
            pad_value=pad_value,
            swap_rgb=swap_rgb,
            output_names=output_names,
            input_names=input_names,
        )

        if output_names is not None:
            self.onnx_export_configuration.update({"output_names": output_names})

    def to_openvino(
        self,
        model: LightningModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        In this implementation the export is done only via standard OV/ONNX tools.
        """
        input_size = self.data_input_params.as_ncwh()
        dummy_tensor = torch.rand(input_size).to(next(model.parameters()).device)

        if self.via_onnx:
            with tempfile.TemporaryDirectory() as tmpdirname:
                tmp_dir = Path(tmpdirname)

                self.to_onnx(
                    model,
                    tmp_dir,
                    base_model_name,
                    Precision.FP32,
                    False,
                )
                exported_model = openvino.convert_model(
                    tmp_dir / (base_model_name + ".onnx"),
                    input=(openvino.PartialShape(input_size),),
                )
        else:
            exported_model = openvino.convert_model(
                model,
                example_input=dummy_tensor,
                input=(openvino.PartialShape(input_size),),
            )
        exported_model = self._postprocess_openvino_model(exported_model)

        save_path = output_dir / (base_model_name + ".xml")
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == Precision.FP16))
        log.info("Converting to OpenVINO is done.")

        return Path(save_path)

    def to_onnx(
        self,
        model: LightningModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export a PyTorch model to ONNX format with automatic fallback to legacy exporter."""
        dummy_tensor = torch.rand(self.data_input_params.as_ncwh()).to(next(model.parameters()).device)
        save_path = str(output_dir / (base_model_name + ".onnx"))
        self._export_onnx(model, dummy_tensor, save_path)

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")

        return Path(save_path)

    def _export_onnx(self, model: LightningModel, dummy_tensor: torch.Tensor, save_path: str) -> None:
        """Run torch.onnx.export, falling back to legacy TorchScript exporter on dynamo failure.

        The dynamo-based exporter (triggered by ``dynamic_shapes`` in the config)
        can fail due to upstream bugs in onnxscript or PyTorch FX passes.
        When that happens and ``dynamic_shapes`` is present, this method retries
        with the legacy TorchScript exporter by converting ``dynamic_shapes`` to
        ``dynamic_axes``.
        """
        try:
            torch.onnx.export(model, dummy_tensor, save_path, **self.onnx_export_configuration)
        except Exception:
            if "dynamic_shapes" not in self.onnx_export_configuration:
                raise
            log.warning(
                "Dynamo-based ONNX export failed, retrying with legacy TorchScript exporter.",
                exc_info=True,
            )
            legacy_config = self._build_legacy_onnx_config()
            torch.onnx.export(model, dummy_tensor, save_path, **legacy_config)

    def _build_legacy_onnx_config(self) -> dict[str, Any]:
        """Convert dynamo-style onnx_export_configuration to legacy TorchScript style.

        Replaces ``dynamic_shapes`` with equivalent ``dynamic_axes`` and removes
        dynamo-only parameters.
        """
        config = dict(self.onnx_export_configuration)

        dynamic_shapes = config.pop("dynamic_shapes", None)
        if dynamic_shapes is not None and "dynamic_axes" not in config:
            input_names = config.get("input_names", [])
            dynamic_axes: dict[str, dict[int, str]] = {}
            for i, shape_spec in enumerate(dynamic_shapes.values()):
                if i < len(input_names) and isinstance(shape_spec, dict):
                    dynamic_axes[input_names[i]] = {
                        dim_idx: getattr(dim_val, "__name__", str(dim_val)) for dim_idx, dim_val in shape_spec.items()
                    }
            config["dynamic_axes"] = dynamic_axes

        # Remove parameters not supported by the legacy exporter
        config.pop("autograd_inlining", None)
        config["dynamo"] = False

        return config
