# Copyright (C) 2024-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Class definition for base model exporter used in getitune."""

from __future__ import annotations

import logging as log
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Literal

from getitune.types.export import ExportFormat, TaskLevelExportParameters
from getitune.types.precision import Precision

if TYPE_CHECKING:
    from pathlib import Path

    import onnx
    import openvino

    from getitune.backend.lightning.models.base import DataInputParams, LightningModel


class ModelExporter:
    """Base class for the model exporters used in getitune.

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
        model: LightningModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        export_format: ExportFormat = ExportFormat.OPENVINO,
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Exports input model to the specified deployable format, such as OpenVINO IR or ONNX.

        Args:
            model (LightningModel): LightningModel to be exported
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            format (ExportFormat): final format of the exported model
            precision (Precision, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model
        """
        if export_format == ExportFormat.OPENVINO:
            return self.to_openvino(model, output_dir, base_model_name, precision)
        if export_format == ExportFormat.ONNX:
            return self.to_onnx(model, output_dir, base_model_name, precision)

        msg = f"Unsupported export format: {export_format}"
        raise ValueError(msg)

    @abstractmethod
    def to_openvino(
        self,
        model: LightningModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
    ) -> Path:
        """Export to OpenVINO Intermediate Representation format.

        Args:
            model (LightningModel): LightningModel to be exported
            output_dir (Path): path to the directory to store export artifacts
            base_model_name (str, optional): exported model name
            precision (Precision, optional): precision of the exported model's weights

        Returns:
            Path: path to the exported model.
        """

    @abstractmethod
    def to_onnx(
        self,
        model: LightningModel,
        output_dir: Path,
        base_model_name: str = "exported_model",
        precision: Precision = Precision.FP32,
        embed_metadata: bool = True,
    ) -> Path:
        """Abstract method for ONNX export.

        Converts the given torch model to ONNX format and saves it to the specified output directory.

        Args:
            model (LightningModel): The input PyTorch model to be converted.
            output_dir (Path): The directory where the ONNX model will be saved.
            base_model_name (str, optional): The name of the exported ONNX model. Defaults to "exported_model".
            precision (Precision, optional): The precision type for the exported model.
            Defaults to Precision.FP32.
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

        Model's original metadata has priority over exporter's extra metadata.
        When ``data_input_params`` carries an ``intensity_config``, the intensity
        parameters are written to ``("model_info", …)`` keys so that ModelAPI can
        reconstruct the correct intensity preprocessing at inference time.

        Args:
            metadata (dict[tuple[str, str], str]): existing metadata for export

        Returns:
            dict[tuple[str, str] ,str]: updated metadata
        """
        mean_str = " ".join(map(str, self.data_input_params.mean)) if self.data_input_params.mean else ""
        std_str = " ".join(map(str, self.data_input_params.std)) if self.data_input_params.std else ""

        extra_data: dict[tuple[str, str], str] = {
            ("model_info", "mean_values"): mean_str.strip(),
            ("model_info", "scale_values"): std_str.strip(),
            ("model_info", "resize_type"): self.resize_mode,
            ("model_info", "pad_value"): str(self.pad_value),
            ("model_info", "reverse_input_channels"): str(self.swap_rgb),
        }

        # Intensity config metadata
        intensity_cfg = self.data_input_params.intensity_config
        if intensity_cfg is not None:
            # Map storage_dtype to the ModelAPI input_dtype convention
            _dtype_map = {
                "uint8": "u8",
                "uint16": "u16",
                "int16": "i16",
                "float32": "f32",
            }
            try:
                input_dtype = _dtype_map[intensity_cfg.storage_dtype]
            except KeyError as exc:
                msg = f"Unsupported intensity storage_dtype '{intensity_cfg.storage_dtype}'"
                raise ValueError(msg) from exc
            extra_data[("model_info", "input_dtype")] = input_dtype
            extra_data[("model_info", "intensity_mode")] = intensity_cfg.mode

            if intensity_cfg.max_value is not None:
                extra_data[("model_info", "intensity_max_value")] = str(intensity_cfg.max_value)
            if intensity_cfg.window_center is not None:
                extra_data[("model_info", "intensity_window_center")] = str(intensity_cfg.window_center)
            if intensity_cfg.window_width is not None:
                extra_data[("model_info", "intensity_window_width")] = str(intensity_cfg.window_width)

            extra_data[("model_info", "intensity_percentile_low")] = str(intensity_cfg.percentile_low)
            extra_data[("model_info", "intensity_percentile_high")] = str(intensity_cfg.percentile_high)
            extra_data[("model_info", "intensity_scale_factor")] = str(intensity_cfg.scale_factor)
            extra_data[("model_info", "intensity_min_value")] = str(intensity_cfg.min_value)

            if intensity_cfg.repeat_channels > 0:
                extra_data[("model_info", "intensity_repeat_channels")] = str(intensity_cfg.repeat_channels)

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
        precision: Precision,
    ) -> onnx.ModelProto:
        if embed_metadata:
            metadata = {} if self.metadata is None else self._extend_model_metadata(self.metadata)
            onnx_model = self._embed_onnx_metadata(onnx_model, metadata)

        if precision == Precision.FP16:
            onnx_model = _convert_onnx_to_float16(onnx_model)

        return onnx_model


def _convert_onnx_to_float16(onnx_model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert ONNX model to float16, working around onnxconverter_common multi-consumer Cast bug.

    See: https://github.com/open-edge-platform/training_extensions/issues/5439
    """
    from onnxconverter_common import float16

    original_fn = float16.remove_unnecessary_cast_node
    float16.remove_unnecessary_cast_node = lambda graph_proto: None  # noqa: ARG005
    try:
        onnx_model = float16.convert_float_to_float16(onnx_model)
    finally:
        float16.remove_unnecessary_cast_node = original_fn

    _remove_unnecessary_cast_nodes(onnx_model.graph)
    return onnx_model


def _remove_unnecessary_cast_nodes(graph_proto: onnx.GraphProto) -> None:
    """Remove redundant consecutive Cast node pairs, handling multi-consumer Cast nodes."""
    cast_index = _CastNodeIndex(graph_proto)
    remove_candidate = cast_index.find_removable_pairs()
    cast_index.reconnect_and_remove(remove_candidate, graph_proto)


class _CastNodeIndex:
    """Index of Cast nodes and their upstream/downstream relationships."""

    def __init__(self, graph_proto: onnx.GraphProto) -> None:
        self.cast_node_list: list[Any] = []
        self.input_name_to_cast: dict[str, Any] = {}
        self.output_name_to_cast: dict[str, Any] = {}
        self.name_to_node: dict[str, Any] = {}
        self.upstream: dict[str, Any] = {}
        self.downstream: dict[str, Any] = {}

        self._index_cast_nodes(graph_proto)
        self._map_neighbors(graph_proto)
        self._exclude_constant_upstream()

    def _index_cast_nodes(self, graph_proto: onnx.GraphProto) -> None:
        """Index all Cast nodes by their input/output names."""
        for node in graph_proto.node:
            if node.op_type == "Cast":
                self.cast_node_list.append(node)
                self.name_to_node[node.name] = node
                for inp in node.input:
                    self.input_name_to_cast[inp] = node
                for out in node.output:
                    self.output_name_to_cast[out] = node

    def _map_neighbors(self, graph_proto: onnx.GraphProto) -> None:
        """Map each Cast node to its upstream and downstream node(s)."""
        for current_node in graph_proto.node:
            for inp in current_node.input:
                if inp in self.output_name_to_cast:
                    cast_node = self.output_name_to_cast[inp]
                    existing = self.downstream.get(cast_node.name)
                    if existing is None:
                        self.downstream[cast_node.name] = current_node
                    elif isinstance(existing, list):
                        existing.append(current_node)
                    else:
                        self.downstream[cast_node.name] = [existing, current_node]

            for out in current_node.output:
                if out in self.input_name_to_cast:
                    cast_node = self.input_name_to_cast[out]
                    self.upstream[cast_node.name] = current_node

    def _exclude_constant_upstream(self) -> None:
        """Exclude Cast nodes whose upstream is a Constant."""
        for cast_name, up_node in self.upstream.items():
            if up_node.op_type == "Constant":
                cast_node = self.name_to_node[cast_name]
                if cast_node in self.cast_node_list:
                    self.cast_node_list.remove(cast_node)

    def find_removable_pairs(self) -> list[tuple[Any, Any]]:
        """Identify removable Cast16->Cast32 pairs."""
        remove_candidate: list[tuple[Any, Any]] = []
        for cast_name, dn in self.downstream.items():
            cast_node = self.name_to_node[cast_name]
            dn_list = dn if isinstance(dn, list) else [dn]
            for dn_node in dn_list:
                if dn_node.op_type == "Cast" and dn_node in self.cast_node_list and cast_node in self.cast_node_list:
                    first_attr = cast_node.attribute[0].i
                    second_attr = dn_node.attribute[0].i
                    # fp16 (10) -> fp32 (1)  OR  int16 (16) -> int32 (32) style pairs
                    if (first_attr == 10 and second_attr == 1) or (first_attr == 16 and second_attr == 32):
                        remove_candidate.append((cast_node, dn_node))
        return remove_candidate

    def reconnect_and_remove(
        self,
        remove_candidate: list[tuple[Any, Any]],
        graph_proto: onnx.GraphProto,
    ) -> None:
        """Reconnect the graph to bypass each Cast pair and remove them."""
        for first_cast, second_cast in remove_candidate:
            bypass_output = self._compute_bypass_output(first_cast, second_cast)
            if bypass_output is None:
                continue

            dn_raw = self.downstream.get(second_cast.name)
            dn_nodes = dn_raw if isinstance(dn_raw, list) else ([dn_raw] if dn_raw is not None else [])
            for dn_node in dn_nodes:
                for i, inp in enumerate(dn_node.input):
                    if inp in list(second_cast.output):
                        dn_node.input[i] = bypass_output

        for first_cast, second_cast in remove_candidate:
            if first_cast in graph_proto.node:
                graph_proto.node.remove(first_cast)
            if second_cast in graph_proto.node:
                graph_proto.node.remove(second_cast)

    def _compute_bypass_output(self, first_cast: onnx.NodeProto, second_cast: onnx.NodeProto) -> str | None:
        """Return the output name that should replace the cast pair's output."""
        up_node = self.upstream.get(first_cast.name)
        dn_raw = self.downstream.get(second_cast.name)

        if up_node is None and dn_raw is not None:
            return first_cast.input[0]
        if up_node is not None and dn_raw is None:
            msg = "The downstream node of the second cast node should be graph output"
            raise ValueError(msg)
        if up_node is not None:
            for out in up_node.output:
                if out == first_cast.input[0]:
                    return out
        return None
