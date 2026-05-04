# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics export metadata module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from getitune.backend.ultralytics.export import (
    _YOLO_MEAN,
    _YOLO_PAD_VALUE,
    _YOLO_RESIZE_TYPE,
    _YOLO_SCALE,
    build_export_metadata,
    cast_onnx_outputs_to_fp32,
    cast_openvino_outputs_to_fp32,
    embed_onnx_metadata,
    embed_openvino_metadata,
)
from getitune.types.label import LabelInfo


def _make_mock_model(
    task: str = "detect",
    model_name: str = "yolo26_n",
    label_names: list[str] | None = None,
    label_ids: list[str] | None = None,
) -> MagicMock:
    """Create a mock UltralyticsModel."""
    model = MagicMock()
    model.task = task
    model.model_name = model_name
    if label_names is not None:
        ids = label_ids or [str(i) for i in range(len(label_names))]
        model.label_info = LabelInfo(label_names=label_names, label_ids=ids, label_groups=[label_names])
    else:
        model.label_info = None
    model.export_model_type = "YOLO11"
    model.export_task_type = "detection"
    return model


class TestBuildExportMetadata:
    """Tests for build_export_metadata."""

    def test_basic_metadata_keys(self) -> None:
        """All required metadata keys should be present."""
        model = _make_mock_model(label_names=["cat", "dog"], label_ids=["0", "1"])
        metadata = build_export_metadata(model, task_type="detection")

        expected_keys = [
            ("model_info", "model_type"),
            ("model_info", "model_name"),
            ("model_info", "task_type"),
            ("model_info", "getitune_version"),
            ("model_info", "label_info"),
            ("model_info", "labels"),
            ("model_info", "label_ids"),
            ("model_info", "mean_values"),
            ("model_info", "scale_values"),
            ("model_info", "resize_type"),
            ("model_info", "pad_value"),
            ("model_info", "reverse_input_channels"),
            ("model_info", "confidence_threshold"),
            ("model_info", "iou_threshold"),
            ("model_info", "optimization_config"),
        ]
        for key in expected_keys:
            assert key in metadata, f"Missing key: {key}"

    def test_model_type_from_task(self) -> None:
        """model_type should map from the model's task attribute."""
        model = _make_mock_model(task="detect", label_names=["a"])
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "model_type")] == "YOLO11"

    def test_model_type_unknown_task_defaults_to_yolo11(self) -> None:
        """Unknown task should default to YOLO11."""
        model = _make_mock_model(task="unknown_task", label_names=["a"])
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "model_type")] == "YOLO11"

    def test_model_name_present(self) -> None:
        model = _make_mock_model(model_name="yolo26_s", label_names=["a"])
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "model_name")] == "yolo26_s"

    def test_task_type_forwarded(self) -> None:
        model = _make_mock_model(label_names=["a"])
        metadata = build_export_metadata(model, task_type="instance_segmentation")
        assert metadata[("model_info", "task_type")] == "instance_segmentation"

    def test_labels_space_separated(self) -> None:
        model = _make_mock_model(label_names=["cat", "dog", "bird"], label_ids=["0", "1", "2"])
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "labels")] == "cat dog bird"
        assert metadata[("model_info", "label_ids")] == "0 1 2"

    def test_label_names_with_spaces_get_underscores(self) -> None:
        """Spaces in label names should be replaced with underscores."""
        model = _make_mock_model(label_names=["hot dog", "ice cream"], label_ids=["0", "1"])
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "labels")] == "hot_dog ice_cream"

    def test_no_label_info_uses_empty(self) -> None:
        """With label_info=None, labels should be empty strings."""
        model = _make_mock_model(label_names=None)
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "labels")] == ""
        assert metadata[("model_info", "label_ids")] == ""

    def test_preprocessing_values(self) -> None:
        """Preprocessing metadata should match YOLO standard values."""
        model = _make_mock_model(label_names=["a"])
        metadata = build_export_metadata(model)

        mean_str = " ".join(str(v) for v in _YOLO_MEAN)
        scale_str = " ".join(str(v) for v in _YOLO_SCALE)

        assert metadata[("model_info", "mean_values")] == mean_str
        assert metadata[("model_info", "scale_values")] == scale_str
        assert metadata[("model_info", "resize_type")] == _YOLO_RESIZE_TYPE
        assert metadata[("model_info", "pad_value")] == str(_YOLO_PAD_VALUE)
        assert metadata[("model_info", "reverse_input_channels")] == "True"

    def test_custom_thresholds(self) -> None:
        """Custom confidence and IoU thresholds should be stored."""
        model = _make_mock_model(label_names=["a"])
        metadata = build_export_metadata(model, confidence_threshold=0.5, iou_threshold=0.7)
        assert metadata[("model_info", "confidence_threshold")] == "0.5"
        assert metadata[("model_info", "iou_threshold")] == "0.7"

    def test_default_thresholds(self) -> None:
        """Default thresholds should match YOLO11 ModelAPI defaults."""
        model = _make_mock_model(label_names=["a"])
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "confidence_threshold")] == "0.25"
        assert metadata[("model_info", "iou_threshold")] == "0.7"

    def test_all_values_are_strings(self) -> None:
        """All metadata values should be strings."""
        model = _make_mock_model(label_names=["cat", "dog"], label_ids=["0", "1"])
        metadata = build_export_metadata(model)
        for key, value in metadata.items():
            assert isinstance(value, str), f"Value for {key} is {type(value)}, expected str"

    def test_getitune_version_present(self) -> None:
        """getitune_version should be a non-empty string."""
        import getitune

        model = _make_mock_model(label_names=["a"])
        metadata = build_export_metadata(model)
        assert metadata[("model_info", "getitune_version")] == getitune.__version__


class TestEmbedOpenvinoMetadata:
    """Tests for embed_openvino_metadata."""

    def test_calls_set_rt_info_for_each_entry(self, tmp_path: Path) -> None:
        """Each metadata entry should result in a set_rt_info call."""
        xml_path = tmp_path / "model.xml"
        xml_path.touch()

        metadata = {
            ("model_info", "model_type"): "YOLO11",
            ("model_info", "task_type"): "detection",
        }

        mock_ov_model = MagicMock()
        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        mock_openvino = MagicMock()
        mock_openvino.Core.return_value = mock_core

        with patch.dict("sys.modules", {"openvino": mock_openvino}):
            result = embed_openvino_metadata(xml_path, metadata)

        # Verify rt_info was set for each metadata entry
        assert mock_ov_model.set_rt_info.call_count == 2
        mock_ov_model.set_rt_info.assert_any_call("YOLO11", ["model_info", "model_type"])
        mock_ov_model.set_rt_info.assert_any_call("detection", ["model_info", "task_type"])

        # Verify model was saved back
        mock_openvino.save_model.assert_called_once_with(mock_ov_model, str(xml_path))
        assert result == xml_path


class TestCastOpenvinoOutputsToFp32:
    """Tests for cast_openvino_outputs_to_fp32."""

    def test_converts_non_fp32_outputs(self, tmp_path: Path) -> None:
        """Non-f32 result inputs should be converted before saving."""
        xml_path = tmp_path / "model.xml"
        xml_path.touch()

        mock_result = MagicMock()
        mock_result.get_input_element_type.return_value = "f16"
        mock_result.input_value.return_value = "source_output"
        mock_result.input.return_value = MagicMock()

        mock_ov_model = MagicMock()
        mock_ov_model.get_results.return_value = [mock_result]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        mock_converted = MagicMock()
        mock_converted.output.return_value = "converted_output"

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.Type.f32", "f32"),
            patch("openvino.save_model") as mock_save,
            patch("openvino.opset13.convert", return_value=mock_converted) as mock_convert,
        ):
            result = cast_openvino_outputs_to_fp32(xml_path)

        mock_convert.assert_called_once_with("source_output", "f32")
        mock_result.input.return_value.replace_source_output.assert_called_once_with("converted_output")
        mock_ov_model.validate_nodes_and_infer_types.assert_called_once_with()
        mock_save.assert_called_once_with(mock_ov_model, str(xml_path))
        assert result == xml_path

    def test_skips_existing_fp32_outputs(self, tmp_path: Path) -> None:
        """Existing f32 outputs should not force a save."""
        xml_path = tmp_path / "model.xml"
        xml_path.touch()

        mock_result = MagicMock()
        mock_result.get_input_element_type.return_value = "f32"
        mock_ov_model = MagicMock()
        mock_ov_model.get_results.return_value = [mock_result]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.Type.f32", "f32"),
            patch("openvino.save_model") as mock_save,
            patch("openvino.opset13.convert") as mock_convert,
        ):
            result = cast_openvino_outputs_to_fp32(xml_path)

        mock_convert.assert_not_called()
        mock_save.assert_not_called()
        assert result == xml_path

    def test_returns_same_path(self, tmp_path: Path) -> None:
        """Should return the same xml_path that was passed in."""
        xml_path = tmp_path / "model.xml"
        xml_path.touch()

        mock_ov_model = MagicMock()
        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        mock_openvino = MagicMock()
        mock_openvino.Core.return_value = mock_core

        with patch.dict("sys.modules", {"openvino": mock_openvino}):
            result = embed_openvino_metadata(xml_path, {})

        assert result == xml_path


class TestCastOnnxOutputsToFp32:
    """Tests for cast_onnx_outputs_to_fp32."""

    def test_adds_cast_for_fp16_outputs(self, tmp_path: Path) -> None:
        """Float16 graph outputs should be rewired through a Cast node."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()

        tensor_type = MagicMock()
        tensor_type.elem_type = 10
        output = MagicMock()
        output.name = "output"
        output.type.tensor_type = tensor_type

        graph = MagicMock()
        graph.output = [output]
        graph.node = []
        mock_model = MagicMock()
        mock_model.graph = graph

        mock_tensor_proto = MagicMock()
        mock_tensor_proto.FLOAT16 = 10
        mock_tensor_proto.FLOAT = 1
        mock_helper = MagicMock()
        mock_helper.make_node.return_value = "cast_node"
        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model
        mock_onnx.TensorProto = mock_tensor_proto
        mock_onnx.helper = mock_helper

        with patch.dict(
            "sys.modules",
            {
                "onnx": mock_onnx,
                "onnx.TensorProto": mock_tensor_proto,
                "onnx.helper": mock_helper,
            },
        ):
            result = cast_onnx_outputs_to_fp32(onnx_path)

        mock_helper.make_node.assert_called_once_with(
            "Cast",
            inputs=["output"],
            outputs=["output_fp32"],
            name="output_to_fp32",
            to=1,
        )
        assert graph.node == ["cast_node"]
        assert output.name == "output_fp32"
        assert tensor_type.elem_type == 1
        mock_onnx.save.assert_called_once_with(mock_model, str(onnx_path))
        assert result == onnx_path

    def test_skips_existing_fp32_outputs(self, tmp_path: Path) -> None:
        """Non-float16 graph outputs should not be changed."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()

        tensor_type = MagicMock()
        tensor_type.elem_type = 1
        output = MagicMock()
        output.name = "output"
        output.type.tensor_type = tensor_type

        graph = MagicMock()
        graph.output = [output]
        graph.node = []
        mock_model = MagicMock()
        mock_model.graph = graph

        mock_tensor_proto = MagicMock()
        mock_tensor_proto.FLOAT16 = 10
        mock_tensor_proto.FLOAT = 1
        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_model
        mock_onnx.TensorProto = mock_tensor_proto
        mock_onnx.helper = MagicMock()

        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            result = cast_onnx_outputs_to_fp32(onnx_path)

        assert graph.node == []
        mock_onnx.save.assert_not_called()
        assert result == onnx_path


class TestEmbedOnnxMetadata:
    """Tests for embed_onnx_metadata."""

    def test_adds_metadata_props(self, tmp_path: Path) -> None:
        """Each metadata entry should result in a metadata_props entry."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()

        metadata = {
            ("model_info", "model_type"): "YOLO11",
            ("model_info", "labels"): "cat dog",
        }

        mock_onnx_model = MagicMock()
        props_added = []

        def add_prop() -> MagicMock:
            prop = MagicMock()
            props_added.append(prop)
            return prop

        mock_onnx_model.metadata_props.add = add_prop

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_onnx_model

        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            result = embed_onnx_metadata(onnx_path, metadata)

        # Verify two props were added
        assert len(props_added) == 2

        # Verify keys are space-separated tuples
        keys_set = {p.key for p in props_added}
        assert "model_info model_type" in keys_set
        assert "model_info labels" in keys_set

        # Verify values
        values_map = {p.key: p.value for p in props_added}
        assert values_map["model_info model_type"] == "YOLO11"
        assert values_map["model_info labels"] == "cat dog"

        # Verify saved
        mock_onnx.save.assert_called_once_with(mock_onnx_model, str(onnx_path))
        assert result == onnx_path

    def test_returns_same_path(self, tmp_path: Path) -> None:
        """Should return the same onnx_path that was passed in."""
        onnx_path = tmp_path / "model.onnx"
        onnx_path.touch()

        mock_onnx_model = MagicMock()
        mock_onnx_model.metadata_props.add = MagicMock(return_value=MagicMock())

        mock_onnx = MagicMock()
        mock_onnx.load.return_value = mock_onnx_model

        with patch.dict("sys.modules", {"onnx": mock_onnx}):
            result = embed_onnx_metadata(onnx_path, {})

        assert result == onnx_path
