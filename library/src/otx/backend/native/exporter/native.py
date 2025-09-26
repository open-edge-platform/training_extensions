# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for native model exporter used in OTX."""

from __future__ import annotations

import logging as log
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import onnx
import openvino
import torch

from otx.backend.native.exporter.base import OTXModelExporter
from otx.types.export import TaskLevelExportParameters
from otx.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from otx.backend.native.models.base import DataInputParams, OTXModel


class OTXNativeModelExporter(OTXModelExporter):
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
        super().__init__(
            task_level_export_parameters=task_level_export_parameters,
            data_input_params=data_input_params,
            resize_mode=resize_mode,
            pad_value=pad_value,
            swap_rgb=swap_rgb,
            output_names=output_names,
            input_names=input_names,
        )
        self.via_onnx = via_onnx
        self.onnx_export_configuration = onnx_export_configuration if onnx_export_configuration is not None else {}
        if output_names is not None:
            self.onnx_export_configuration.update({"output_names": output_names})

    def to_openvino(
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
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
                    OTXPrecisionType.FP32,
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
        openvino.save_model(exported_model, save_path, compress_to_fp16=(precision == OTXPrecisionType.FP16))
        log.info("Converting to OpenVINO is done.")

        return Path(save_path)

    def to_onnx(
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Export the given PyTorch model to ONNX format and save it to the specified output directory.

        Args:
            model (OTXModel): The PyTorch model to be exported.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The base name for the exported model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.
            embed_metadata (bool, optional): Whether to embed metadata in the ONNX model. Defaults to True.

        Returns:
            Path: The path to the saved ONNX model.
        """
        dummy_tensor = torch.rand(self.data_input_params.as_ncwh()).to(next(model.parameters()).device)
        save_path = str(output_dir / (base_model_name + ".onnx"))

        torch.onnx.export(model, dummy_tensor, save_path, **self.onnx_export_configuration)

        onnx_model = onnx.load(save_path)
        onnx_model = self._postprocess_onnx_model(onnx_model, embed_metadata, precision)

        onnx.save(onnx_model, save_path)
        log.info("Converting to ONNX is done.")

        return Path(save_path)
