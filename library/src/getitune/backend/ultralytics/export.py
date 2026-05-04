# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export metadata embedding for Ultralytics models.

After Ultralytics produces a raw OpenVINO IR or ONNX model, this module
injects the ``rt_info["model_info"]`` metadata required by ModelAPI and the
Geti application inference pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import getitune
from getitune.types.label import LabelInfo

if TYPE_CHECKING:
    from pathlib import Path

    from .models.base import UltralyticsModel

logger = logging.getLogger(__name__)

# Default preprocessing parameters for YOLO models.
# YOLO expects: BGR→RGB, divide by 255, letterbox resize with pad=114.
# ModelAPI convention: scale_values are DIVISORS (image / scale).
_YOLO_MEAN = (0.0, 0.0, 0.0)
_YOLO_SCALE = (255.0, 255.0, 255.0)
_YOLO_RESIZE_TYPE = "fit_to_window_letterbox"
_YOLO_PAD_VALUE = 114
_YOLO_REVERSE_INPUT_CHANNELS = True


def build_export_metadata(
    model: UltralyticsModel,
    task_type: str = "detection",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
) -> dict[tuple[str, str], str]:
    """Construct metadata dict for embedding into exported IR/ONNX.

    The returned dict uses the same ``(group, key)`` tuple format as
    :class:`~getitune.types.export.TaskLevelExportParameters.to_metadata`.

    Args:
        model: The Ultralytics model wrapper (provides label_info, model_name).
        task_type: ModelAPI task type string (``"detection"``, ``"instance_segmentation"``).
        confidence_threshold: Default confidence threshold for inference.
        iou_threshold: Default IoU threshold for NMS.

    Returns:
        Dict keyed as ``("model_info", "<key>")`` with string values,
        ready for ``ov_model.set_rt_info(value, list(key))``.
    """
    label_info = model.label_info or LabelInfo(label_names=[], label_ids=[], label_groups=[])
    model_type = model.export_model_type

    # Build labels strings
    all_labels = " ".join(name.replace(" ", "_") for name in label_info.label_names)
    all_label_ids = " ".join(label_info.label_ids) if label_info.label_ids else ""

    # Preprocessing metadata (YOLO standard)
    mean_str = " ".join(str(v) for v in _YOLO_MEAN)
    scale_str = " ".join(str(v) for v in _YOLO_SCALE)

    metadata: dict[tuple[str, str], str] = {
        # Model identification
        ("model_info", "model_type"): model_type,
        ("model_info", "model_name"): model.model_name or "",
        ("model_info", "task_type"): task_type,
        ("model_info", "getitune_version"): getitune.__version__,
        # Label info
        ("model_info", "label_info"): label_info.to_json(),
        ("model_info", "labels"): all_labels.strip(),
        ("model_info", "label_ids"): all_label_ids.strip(),
        # Preprocessing
        ("model_info", "mean_values"): mean_str.strip(),
        ("model_info", "scale_values"): scale_str.strip(),
        ("model_info", "resize_type"): _YOLO_RESIZE_TYPE,
        ("model_info", "pad_value"): str(_YOLO_PAD_VALUE),
        ("model_info", "reverse_input_channels"): str(_YOLO_REVERSE_INPUT_CHANNELS),
        # Postprocessing
        ("model_info", "confidence_threshold"): str(confidence_threshold),
        ("model_info", "iou_threshold"): str(iou_threshold),
        # Optimization (blocked for now)
        ("model_info", "optimization_config"): json.dumps({}),
    }

    return metadata


def embed_openvino_metadata(xml_path: Path, metadata: dict[tuple[str, str], str]) -> Path:
    """Embed metadata into an OpenVINO IR model file.

    Loads the model, writes ``rt_info`` entries, and saves back to the same path.

    Args:
        xml_path: Path to the ``.xml`` file.
        metadata: Metadata dict from :func:`build_export_metadata`.

    Returns:
        The same ``xml_path`` (modified in-place).
    """
    import openvino

    core = openvino.Core()
    ov_model = core.read_model(str(xml_path))

    for key, value in metadata.items():
        ov_model.set_rt_info(value, list(key))

    openvino.save_model(ov_model, str(xml_path))
    logger.info(f"Embedded {len(metadata)} metadata entries into {xml_path}")
    return xml_path


def embed_onnx_metadata(onnx_path: Path, metadata: dict[tuple[str, str], str]) -> Path:
    """Embed metadata into an ONNX model file.

    Uses ONNX ``metadata_props`` with space-separated key format
    (e.g. ``"model_info model_type"``), matching the convention used by
    Lightning's :meth:`ModelExporter._embed_onnx_metadata`.

    Args:
        onnx_path: Path to the ``.onnx`` file.
        metadata: Metadata dict from :func:`build_export_metadata`.

    Returns:
        The same ``onnx_path`` (modified in-place).
    """
    import onnx

    onnx_model = onnx.load(str(onnx_path))

    for key, value in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = " ".join(key)
        meta.value = str(value)

    onnx.save(onnx_model, str(onnx_path))
    logger.info(f"Embedded {len(metadata)} metadata entries into {onnx_path}")
    return onnx_path


def cast_openvino_outputs_to_fp32(xml_path: Path) -> Path:
    """Cast OpenVINO model outputs to f32 in-place.

    Ultralytics FP16 export can produce FP16 outputs, while ModelAPI's YOLO11
    wrapper requires output tensors to be f32. Keeping the internal graph FP16
    and only converting result nodes preserves FP16 export support without a
    custom ModelAPI wrapper.
    """
    import openvino
    from openvino import opset13 as opset

    core = openvino.Core()
    ov_model = core.read_model(str(xml_path))

    changed = False
    for result in ov_model.get_results():
        if result.get_input_element_type(0) == openvino.Type.f32:
            continue
        converted = opset.convert(result.input_value(0), openvino.Type.f32)
        result.input(0).replace_source_output(converted.output(0))
        changed = True

    if changed:
        ov_model.validate_nodes_and_infer_types()
        openvino.save_model(ov_model, str(xml_path))
        logger.info(f"Casted OpenVINO output tensors to f32 in {xml_path}")
    return xml_path


def cast_onnx_outputs_to_fp32(onnx_path: Path) -> Path:
    """Cast ONNX graph outputs to f32 in-place.

    Adds a Cast node after every float16 graph output and rewires that output
    to the cast result. This keeps FP16 internals while exposing f32 outputs to
    ModelAPI's YOLO11 wrapper.
    """
    import onnx
    from onnx import TensorProto, helper

    onnx_model = onnx.load(str(onnx_path))
    graph = onnx_model.graph

    changed = False
    existing_names = {output.name for output in graph.output}
    for output in graph.output:
        tensor_type = output.type.tensor_type
        if tensor_type.elem_type != TensorProto.FLOAT16:
            continue

        original_name = output.name
        cast_output_name = f"{original_name}_fp32"
        suffix = 1
        while cast_output_name in existing_names:
            cast_output_name = f"{original_name}_fp32_{suffix}"
            suffix += 1
        existing_names.add(cast_output_name)

        cast_node = helper.make_node(
            "Cast",
            inputs=[original_name],
            outputs=[cast_output_name],
            name=f"{original_name}_to_fp32",
            to=TensorProto.FLOAT,
        )
        graph.node.append(cast_node)
        output.name = cast_output_name
        tensor_type.elem_type = TensorProto.FLOAT
        changed = True

    if changed:
        onnx.save(onnx_model, str(onnx_path))
        logger.info(f"Casted ONNX output tensors to f32 in {onnx_path}")
    return onnx_path
