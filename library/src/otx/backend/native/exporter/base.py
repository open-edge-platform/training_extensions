# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for base model exporter used in OTX."""

from __future__ import annotations

import logging as log
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from otx.types.export import OTXExportFormatType, TaskLevelExportParameters
from otx.types.precision import OTXPrecisionType

if TYPE_CHECKING:
    from pathlib import Path

    import onnx
    import openvino

    from otx.backend.native.models.base import DataInputParams, OTXModel


class OTXModelExporter:
    """Base class for the model exporters used in OTX.

    Args:
        task_level_export_parameters (TaskLevelExportParameters): Collection of export parameters
            which can be defined at a task level.
        data_input_params (DataInputParams): Data input parameters for model preprocessing.
        resize_mode (Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"], optional):
            A resize type for model preprocess. "standard" resizes images without keeping ratio.
            "fit_to_window" resizes images while keeping ratio.
            "fit_to_window_letterbox" resizes images and pads images to fit the size. Defaults to "standard".
        pad_value (int, optional): Padding value. Defaults to 0.
        swap_rgb (bool, optional): Whether to convert the image from BGR to RGB Defaults to False.
        output_names (list[str] | None, optional): Names for model's outputs, which would be
        embedded into resulting model. Note, that order of the output names should be the same,
        as in the target model.
        input_names (list[str] | None, optional): Names for model's inputs, which would be
        embedded into resulting model. Note, that order of the input names should be the same,
        as in the target model.
    """

    def __init__(
        self,
        task_level_export_parameters: TaskLevelExportParameters,
        data_input_params: DataInputParams,
        resize_mode: Literal["crop", "standard", "fit_to_window", "fit_to_window_letterbox"] = "standard",
        pad_value: int = 0,
        swap_rgb: bool = False,
        output_names: list[str] | None = None,
        input_names: list[str] | None = None,
    ) -> None:
        self.data_input_params = data_input_params
        self.resize_mode = resize_mode
        self.pad_value = pad_value
        self.swap_rgb = swap_rgb
        self.task_level_export_parameters = task_level_export_parameters
        self.output_names = output_names
        self.input_names = input_names

    @property
    def metadata(self) -> dict[tuple[str, str], str]:
        """Collection of metadata to be stored in OpenVINO Intermediate Representation or ONNX.

        This metadata is mainly used to support ModelAPI.
        """
        return self.task_level_export_parameters.to_metadata()

    def export(
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        export_format: OTXExportFormatType = OTXExportFormatType.OPENVINO,
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Exports input model to the specified deployable format, such as OpenVINO IR or ONNX.

        Args:
            model (OTXModel): OTXModel to be exported
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            format (OTXExportFormatType): final format of the exported model
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model
        """
        if export_format == OTXExportFormatType.OPENVINO:
            return self.to_openvino(model, output_dir, base_model_name, precision)
        if export_format == OTXExportFormatType.ONNX:
            return self.to_onnx(model, output_dir, base_model_name, precision)

        msg = f"Unsupported export format: {export_format}"
        raise ValueError(msg)

    @abstractmethod
    def to_openvino(
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            model (OTXModel): OTXModel to be exported
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (OTXExportPrecisionType, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model.
        """

    @abstractmethod
    def to_onnx(
        self,
        model: OTXModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: OTXPrecisionType = OTXPrecisionType.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Abstract method for ONNX export.

        Converts the given torch model to ONNX format and saves it to the specified output directory.

        Args:
            model (OTXModel): The input PyTorch model to be converted.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The name of the exported ONNX model. Defaults to "exported_model".
            precision (OTXPrecisionType, optional): The precision type for the exported model.
            Defaults to OTXPrecisionType.FP32.
            embed_metadata (bool, optional): Flag to embed metadata in the exported ONNX model. Defaults to True.

        Returns:
            Path: The file path where the ONNX model is saved.
        """

    @staticmethod
    def _embed_onnx_metadata(onnx_model: onnx.ModelProto, metadata: dict[tuple[str, str], Any]) -> onnx.ModelProto:
        """Embeds metadata to ONNX model."""
        for k, v in metadata.items():
            meta = onnx_model.metadata_props.add()
            attr_path = " ".join(map(str, k))
            meta.key = attr_path.strip()
            meta.value = str(v)

        return onnx_model

    @staticmethod
    def _embed_openvino_ir_metadata(ov_model: openvino.Model, metadata: dict[tuple[str, str], Any]) -> openvino.Model:
        """Embeds metadata to OpenVINO model."""
        for k, data in metadata.items():
            ov_model.set_rt_info(data, list(k))

        return ov_model

    def _extend_model_metadata(self, metadata: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
        """Extends metadata coming from model with preprocessing-specific parameters.

        Model's original metadata has priority over exporter's extra metadata

        Args:
            metadata (dict[tuple[str, str], str]): existing metadata for export

        Returns:
            dict[tuple[str, str] ,str]: updated metadata
        """
        mean_str = " ".join(map(str, self.data_input_params.mean))
        std_str = " ".join(map(str, self.data_input_params.std))

        extra_data = {
            ("model_info", "mean_values"): mean_str.strip(),
            ("model_info", "scale_values"): std_str.strip(),
            ("model_info", "resize_type"): self.resize_mode,
            ("model_info", "pad_value"): str(self.pad_value),
            ("model_info", "reverse_input_channels"): str(self.swap_rgb),
        }
        extra_data.update(metadata)

        return extra_data

    def _postprocess_openvino_model(self, exported_model: openvino.Model) -> openvino.Model:
        if len(exported_model.outputs) == 1 and len(exported_model.outputs[0].get_names()) == 0:
            # workaround for OVC's bug: single output doesn't have a name in OV model
            exported_model.outputs[0].tensor.set_names({"output1"})

        # name assignment process is similar to torch onnx export
        if self.output_names is not None:
            if len(exported_model.outputs) >= len(self.output_names):
                if len(exported_model.outputs) != len(self.output_names):
                    msg = (
                        "Number of model outputs is greater than the number"
                        " of output names to assign. Please check output_names"
                        " argument of the exporter's constructor."
                    )
                    log.warning(msg)

                for i, name in enumerate(self.output_names):
                    traced_names = exported_model.outputs[i].get_names()
                    name_found = False
                    for traced_name in traced_names:
                        if name in traced_name:
                            name_found = True
                            break
                    name_found = name_found and bool(len(traced_names))

                    if not name_found:
                        msg = (
                            f"{name} is not matched with the converted model's traced output names: {traced_names}."
                            " Please check output_names argument of the exporter's constructor."
                        )
                        log.warning(msg)

                    exported_model.outputs[i].tensor.set_names({name})
            else:
                msg = (
                    "Model has less outputs than the number of output names provided: "
                    f"{len(exported_model.outputs)} vs {len(self.output_names)}"
                )
                raise RuntimeError(msg)

        if self.input_names is not None:
            if len(exported_model.inputs) >= len(self.input_names):
                if len(exported_model.inputs) != len(self.input_names):
                    msg = (
                        "Number of model inputs is greater than the number"
                        " of input names to assign. Please check input_names"
                        " argument of the exporter's constructor."
                    )
                    log.warning(msg)

                for i, name in enumerate(self.input_names):
                    traced_names = exported_model.inputs[i].get_names()
                    name_found = False
                    for traced_name in traced_names:
                        if name in traced_name:
                            name_found = True
                            break
                    name_found = name_found and bool(len(traced_names))

                    if not name_found:
                        msg = (
                            f"{name} is not matched with the converted model's traced input names: {traced_names}."
                            " Please check input_names argument of the exporter's constructor."
                        )
                        log.warning(msg)

                    exported_model.inputs[i].tensor.set_names({name})
            else:
                msg = (
                    "Model has less inputs than the number of input names provided: "
                    f"{len(exported_model.inputs)} vs {len(self.input_names)}"
                )
                raise RuntimeError(msg)

        if self.metadata is not None:
            export_metadata = self._extend_model_metadata(self.metadata)
            exported_model = self._embed_openvino_ir_metadata(exported_model, export_metadata)

        return exported_model

    def _postprocess_onnx_model(
        self,
        onnx_model: onnx.ModelProto,
        embed_metadata: bool,
        precision: OTXPrecisionType,
    ) -> onnx.ModelProto:
        if embed_metadata:
            metadata = {} if self.metadata is None else self._extend_model_metadata(self.metadata)
            onnx_model = self._embed_onnx_metadata(onnx_model, metadata)

        if precision == OTXPrecisionType.FP16:
            from onnxconverter_common import float16

            onnx_model = float16.convert_float_to_float16(onnx_model)

        return onnx_model
