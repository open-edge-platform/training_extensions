# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics model exporter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.exporter import UltralyticsModelExporter
from getitune.config.data import IntensityConfig
from getitune.types.export import ExportFormat, TaskLevelExportParameters
from getitune.types.label import LabelInfo
from getitune.types.precision import Precision


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


def _data_input_params(imgsz: int = 640) -> DataInputParams:
    return DataInputParams(
        input_size=(imgsz, imgsz),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        intensity_config=IntensityConfig(mode="scale_to_unit", storage_dtype="uint8"),
    )


def _export_parameters() -> TaskLevelExportParameters:
    return TaskLevelExportParameters(
        model_type="YOLO11",
        model_name="yolo26_n",
        task_type="detection",
        label_info=_label_info(),
        optimization_config={},
        confidence_threshold=0.25,
        iou_threshold=0.5,
    )


def _make_exporter(**kwargs) -> UltralyticsModelExporter:
    """Create an exporter with sensible defaults."""
    defaults = {
        "task_level_export_parameters": _export_parameters(),
        "data_input_params": _data_input_params(),
    }
    defaults.update(kwargs)
    return UltralyticsModelExporter(**defaults)


class TestUltralyticsModelExporterInit:
    """Tests for exporter construction and default values."""

    def test_default_resize_mode(self) -> None:
        exporter = _make_exporter()
        assert exporter.resize_mode == "fit_to_window_letterbox"

    def test_default_pad_value(self) -> None:
        exporter = _make_exporter()
        assert exporter.pad_value == 114

    def test_default_swap_rgb(self) -> None:
        exporter = _make_exporter()
        assert exporter.swap_rgb is True

    def test_inherits_from_model_exporter(self) -> None:
        from getitune.backend.lightning.exporter.base import ModelExporter

        exporter = _make_exporter()
        assert isinstance(exporter, ModelExporter)


class TestExporterMetadata:
    """Tests for metadata produced by the inherited metadata pipeline."""

    def test_metadata_includes_model_type(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert ("model_info", "model_type") in metadata
        assert metadata[("model_info", "model_type")] == "YOLO11"

    def test_metadata_includes_task_type(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "task_type")] == "detection"

    def test_metadata_includes_labels(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "labels")] == "cat dog"
        assert metadata[("model_info", "label_ids")] == "0 1"

    def test_metadata_includes_thresholds(self) -> None:
        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "confidence_threshold")] == "0.25"
        assert metadata[("model_info", "iou_threshold")] == "0.5"

    def test_metadata_includes_getitune_version(self) -> None:
        import getitune

        exporter = _make_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "getitune_version")] == getitune.__version__

    def test_extended_metadata_includes_preprocessing(self) -> None:
        """_extend_model_metadata should add mean/scale/resize from DataInputParams."""
        exporter = _make_exporter()
        extended = exporter._extend_model_metadata(exporter.metadata)

        assert extended[("model_info", "mean_values")] == "0.0 0.0 0.0"
        assert extended[("model_info", "scale_values")] == "1.0 1.0 1.0"
        assert extended[("model_info", "resize_type")] == "fit_to_window_letterbox"
        assert extended[("model_info", "pad_value")] == "114"
        assert extended[("model_info", "reverse_input_channels")] == "True"
        assert extended[("model_info", "input_dtype")] == "u8"
        assert extended[("model_info", "intensity_mode")] == "scale_to_unit"
        assert ("model_info", "intensity_max_value") not in extended

    def test_postprocess_openvino_embeds_intensity_metadata(self) -> None:
        """OpenVINO postprocessing should write the IntensityConfig contract into rt_info."""
        exporter = _make_exporter()
        mock_ov_model = MagicMock()
        mock_ov_model.outputs = []
        mock_ov_model.inputs = []

        exporter._postprocess_openvino_model(mock_ov_model)

        rt_info = {tuple(call.args[1]): call.args[0] for call in mock_ov_model.set_rt_info.call_args_list}
        assert rt_info[("model_info", "scale_values")] == "1.0 1.0 1.0"
        assert rt_info[("model_info", "input_dtype")] == "u8"
        assert rt_info[("model_info", "intensity_mode")] == "scale_to_unit"

    def test_all_metadata_values_are_strings(self) -> None:
        exporter = _make_exporter()
        extended = exporter._extend_model_metadata(exporter.metadata)
        for key, value in extended.items():
            assert isinstance(value, str), f"Value for {key} is {type(value)}, expected str"

    def test_custom_thresholds_via_wrap(self) -> None:
        """Threshold overrides via TaskLevelExportParameters.wrap should propagate."""
        params = _export_parameters().wrap(confidence_threshold=0.5, iou_threshold=0.6)
        exporter = _make_exporter(task_level_export_parameters=params)
        metadata = exporter.metadata
        assert metadata[("model_info", "confidence_threshold")] == "0.5"
        assert metadata[("model_info", "iou_threshold")] == "0.6"


class TestToOpenvino:
    """Tests for the to_openvino export path."""

    def test_exports_fp32_with_correct_args(self, tmp_path: Path) -> None:
        """Should call model.export with half=False and end2end=False."""
        exporter = _make_exporter()

        # Setup: Ultralytics produces a directory with a .xml
        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()
        (raw_dir / "model.bin").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model") as mock_save,
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            output_dir = tmp_path / "output"
            result = exporter.to_openvino(mock_yolo, output_dir, "exported_model", Precision.FP32)

        mock_yolo.export.assert_called_once_with(
            format="openvino",
            imgsz=640,
            half=False,
            end2end=False,
            project=str(tmp_path / "output"),
            name="raw_export",
            exist_ok=True,
        )
        mock_save.assert_called_once_with(mock_ov_model, str(output_dir / "exported_model.xml"), compress_to_fp16=False)
        assert result == output_dir / "exported_model.xml"

    def test_fp16_uses_compress_to_fp16(self, tmp_path: Path) -> None:
        """FP16 should use compress_to_fp16=True, NOT half=True."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model") as mock_save,
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            output_dir = tmp_path / "output"
            result = exporter.to_openvino(mock_yolo, output_dir, "exported_model", Precision.FP16)

        # Ultralytics should still get half=False (always FP32 export)
        _, call_kwargs = mock_yolo.export.call_args
        assert call_kwargs["half"] is False

        # OpenVINO save should use compress_to_fp16=True
        mock_save.assert_called_once_with(mock_ov_model, str(output_dir / "exported_model.xml"), compress_to_fp16=True)
        assert result == output_dir / "exported_model.xml"

    def test_postprocess_is_called(self, tmp_path: Path) -> None:
        """_postprocess_openvino_model should be called to embed metadata."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model"),
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model) as mock_pp,
        ):
            exporter.to_openvino(mock_yolo, tmp_path / "out", "model", Precision.FP32)

        mock_pp.assert_called_once_with(mock_ov_model)

    def test_no_xml_raises_file_not_found(self, tmp_path: Path) -> None:
        """If Ultralytics export produces a dir with no .xml, should raise."""
        exporter = _make_exporter()

        empty_dir = tmp_path / "empty_export"
        empty_dir.mkdir()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(empty_dir)

        with pytest.raises(FileNotFoundError, match="No .xml file found"):
            exporter.to_openvino(mock_yolo, tmp_path / "out", "model", Precision.FP32)

    def test_cleanup_removes_raw_export(self, tmp_path: Path) -> None:
        """Raw Ultralytics export directory should be cleaned up."""
        exporter = _make_exporter()

        raw_dir = tmp_path / "raw_openvino"
        raw_dir.mkdir()
        (raw_dir / "model.xml").touch()
        (raw_dir / "model.bin").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_dir)

        mock_ov_model = MagicMock()
        mock_ov_model.inputs = [MagicMock()]
        mock_ov_model.outputs = [MagicMock()]

        mock_core = MagicMock()
        mock_core.read_model.return_value = mock_ov_model

        output_dir = tmp_path / "output"
        with (
            patch("openvino.Core", return_value=mock_core),
            patch("openvino.save_model"),
            patch.object(exporter, "_postprocess_openvino_model", return_value=mock_ov_model),
        ):
            exporter.to_openvino(mock_yolo, output_dir, "exported_model", Precision.FP32)

        # Raw dir should have been cleaned up
        assert not raw_dir.exists()


class TestToOnnx:
    """Tests for the to_onnx export path."""

    def test_exports_fp32_with_correct_args(self, tmp_path: Path) -> None:
        exporter = _make_exporter()

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)

        mock_onnx_model = MagicMock()

        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save") as mock_save,
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model) as mock_pp,
        ):
            output_dir = tmp_path / "output"
            result = exporter.to_onnx(mock_yolo, output_dir, "exported_model", Precision.FP32)

        mock_yolo.export.assert_called_once_with(
            format="onnx",
            imgsz=640,
            half=False,
            end2end=False,
            project=str(tmp_path / "output"),
            name="raw_export",
            exist_ok=True,
        )
        mock_pp.assert_called_once_with(mock_onnx_model, True, Precision.FP32)
        mock_save.assert_called_once_with(mock_onnx_model, str(output_dir / "exported_model.onnx"))
        assert result == output_dir / "exported_model.onnx"

    def test_fp16_passes_precision_to_postprocess(self, tmp_path: Path) -> None:
        exporter = _make_exporter()

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)

        mock_onnx_model = MagicMock()

        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model) as mock_pp,
        ):
            exporter.to_onnx(mock_yolo, tmp_path / "out", "model", Precision.FP16)

        # Ultralytics still exports FP32
        _, call_kwargs = mock_yolo.export.call_args
        assert call_kwargs["half"] is False

        # FP16 conversion is done by _postprocess_onnx_model (inherited)
        mock_pp.assert_called_once_with(mock_onnx_model, True, Precision.FP16)

    def test_raw_onnx_cleaned_up(self, tmp_path: Path) -> None:
        """Raw ONNX file should be removed when it differs from the output."""
        exporter = _make_exporter()

        raw_onnx = tmp_path / "raw_model.onnx"
        raw_onnx.write_bytes(b"fake")

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(raw_onnx)

        mock_onnx_model = MagicMock()

        with (
            patch("onnx.load", return_value=mock_onnx_model),
            patch("onnx.save"),
            patch.object(exporter, "_postprocess_onnx_model", return_value=mock_onnx_model),
        ):
            output_dir = tmp_path / "output"
            exporter.to_onnx(mock_yolo, output_dir, "model", Precision.FP32)

        assert not raw_onnx.exists()


class TestExportDispatch:
    """Tests for the inherited export() dispatch method."""

    def test_openvino_dispatches_to_to_openvino(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        mock_yolo = MagicMock()

        with patch.object(exporter, "to_openvino", return_value=tmp_path / "model.xml") as mock_to_ov:
            result = exporter.export(mock_yolo, tmp_path, "model", ExportFormat.OPENVINO, Precision.FP32)

        mock_to_ov.assert_called_once_with(mock_yolo, tmp_path, "model", Precision.FP32)
        assert result == tmp_path / "model.xml"

    def test_onnx_dispatches_to_to_onnx(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        mock_yolo = MagicMock()

        with patch.object(exporter, "to_onnx", return_value=tmp_path / "model.onnx") as mock_to_onnx:
            result = exporter.export(mock_yolo, tmp_path, "model", ExportFormat.ONNX, Precision.FP32)

        mock_to_onnx.assert_called_once_with(mock_yolo, tmp_path, "model", Precision.FP32)
        assert result == tmp_path / "model.onnx"

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        exporter = _make_exporter()
        mock_yolo = MagicMock()

        bad_format = MagicMock(spec=ExportFormat)
        bad_format.value = "TORCHSCRIPT"

        with pytest.raises(ValueError, match="Unsupported export format"):
            exporter.export(mock_yolo, tmp_path, "model", bad_format, Precision.FP32)


class TestFindXmlInExport:
    """Tests for _find_xml_in_export static method."""

    def test_returns_xml_file_directly(self, tmp_path: Path) -> None:
        xml = tmp_path / "model.xml"
        xml.touch()
        assert UltralyticsModelExporter._find_xml_in_export(xml) == xml

    def test_finds_xml_in_directory(self, tmp_path: Path) -> None:
        (tmp_path / "model.xml").touch()
        result = UltralyticsModelExporter._find_xml_in_export(tmp_path)
        assert result.suffix == ".xml"

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError, match="No .xml file found"):
            UltralyticsModelExporter._find_xml_in_export(empty)


class TestCleanupRawExport:
    """Tests for _cleanup_raw_export static method."""

    def test_removes_different_directory(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "file.txt").touch()
        target = tmp_path / "target"
        target.mkdir()

        UltralyticsModelExporter._cleanup_raw_export(raw, target)
        assert not raw.exists()

    def test_does_not_remove_same_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "output"
        target.mkdir()
        (target / "file.txt").touch()

        UltralyticsModelExporter._cleanup_raw_export(target, target)
        assert target.exists()

    def test_does_not_remove_parent_of_target(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        target = raw / "sub"
        target.mkdir()

        UltralyticsModelExporter._cleanup_raw_export(raw, target)
        assert raw.exists()

    def test_handles_nonexistent_path(self, tmp_path: Path) -> None:
        """Should not raise for nonexistent raw path."""
        UltralyticsModelExporter._cleanup_raw_export(tmp_path / "nonexistent", tmp_path)


class TestYOLO11SegWrapper:
    """Tests for the YOLO11-seg ModelAPI wrapper."""

    def test_registered_with_model_api(self) -> None:
        """YOLO11-seg should be discoverable via Model.available_wrappers()."""
        from model_api.models import Model

        from getitune.backend.ultralytics.exporter import YOLO11Seg  # noqa: F401

        assert "YOLO11-seg" in Model.available_wrappers()

    def test_postprocess_empty_detections(self) -> None:
        """Should return empty InstanceSegmentationResult when no detections pass threshold."""
        import numpy as np

        from getitune.backend.ultralytics.exporter.yolo_seg_wrapper import YOLO11Seg

        wrapper = object.__new__(YOLO11Seg)
        wrapper._det_output_name = "det"
        wrapper._proto_output_name = "proto"
        wrapper._mask_dim = 32
        wrapper._proto_h = 160
        wrapper._proto_w = 160
        wrapper._num_classes = 5
        wrapper.params = MagicMock()
        wrapper.params.confidence_threshold = 0.5
        wrapper.params.resize_type = "fit_to_window_letterbox"
        wrapper.orig_width = 640
        wrapper.orig_height = 640

        # All zeros → no detection above threshold
        outputs = {
            "det": np.zeros((1, 41, 8400), dtype=np.float32),  # 4 + 5 + 32
            "proto": np.zeros((1, 32, 160, 160), dtype=np.float32),
        }
        meta = {"original_shape": (480, 640)}

        result = wrapper.postprocess(outputs, meta)

        assert result.bboxes.shape == (0, 4)
        assert result.masks.shape == (0, 480, 640)
        assert result.scores.shape == (0,)
        assert result.labels.shape == (0,)

    def test_postprocess_with_detections(self) -> None:
        """Should decode masks and return proper InstanceSegmentationResult."""
        import numpy as np

        from getitune.backend.ultralytics.exporter.yolo_seg_wrapper import YOLO11Seg

        wrapper = object.__new__(YOLO11Seg)
        wrapper._det_output_name = "det"
        wrapper._proto_output_name = "proto"
        wrapper._mask_dim = 32
        wrapper._proto_h = 160
        wrapper._proto_w = 160
        wrapper._num_classes = 5
        wrapper.params = MagicMock()
        wrapper.params.confidence_threshold = 0.1
        wrapper.params.resize_type = "fit_to_window_letterbox"
        wrapper.params.iou_threshold = 0.7
        wrapper.params.nms_execute = True
        wrapper.params.nms_max_predictions = 30000
        wrapper.orig_width = 640
        wrapper.orig_height = 640
        wrapper.labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]
        wrapper.get_label_name = lambda i: f"class_{i}"

        # Create a single detection with high confidence
        det = np.zeros((1, 41, 8400), dtype=np.float32)
        # Box at center: xywh = (320, 320, 100, 100) → detection 0
        det[0, 0, 0] = 320.0  # x_center
        det[0, 1, 0] = 320.0  # y_center
        det[0, 2, 0] = 100.0  # width
        det[0, 3, 0] = 100.0  # height
        det[0, 4, 0] = 0.9  # class 0 confidence
        # Mask coefficients: all ones
        det[0, 9:41, 0] = 1.0  # 32 mask coefficients

        # Simple prototypes: uniform → after coeff@proto will be large positive → sigmoid→1 → mask=1
        proto = np.ones((1, 32, 160, 160), dtype=np.float32) * 0.1

        outputs = {"det": det, "proto": proto}
        meta = {"original_shape": (640, 640)}

        result = wrapper.postprocess(outputs, meta)

        assert result.bboxes.shape[0] == 1
        assert result.masks.shape == (1, 640, 640)
        assert result.scores[0] == pytest.approx(0.9, abs=0.01)
        assert result.labels[0] == 1  # 0-indexed + 1 for MaskRCNN convention
        assert result.masks[0].sum() > 0  # mask should have nonzero pixels

    def test_postprocess_non_square_image(self) -> None:
        """Mask decode should handle non-square (letterboxed) images correctly."""
        import numpy as np

        from getitune.backend.ultralytics.exporter.yolo_seg_wrapper import YOLO11Seg

        wrapper = object.__new__(YOLO11Seg)
        wrapper._det_output_name = "det"
        wrapper._proto_output_name = "proto"
        wrapper._mask_dim = 32
        wrapper._proto_h = 160
        wrapper._proto_w = 160
        wrapper._num_classes = 5
        wrapper.params = MagicMock()
        wrapper.params.confidence_threshold = 0.1
        wrapper.params.resize_type = "fit_to_window_letterbox"
        wrapper.params.iou_threshold = 0.7
        wrapper.params.nms_execute = True
        wrapper.params.nms_max_predictions = 30000
        wrapper.orig_width = 640
        wrapper.orig_height = 640
        wrapper.labels = ["class_0", "class_1", "class_2", "class_3", "class_4"]
        wrapper.get_label_name = lambda i: f"class_{i}"

        # Landscape image: 1920x1080 → letterboxed to 640x640 with pad_top=140
        det = np.zeros((1, 41, 8400), dtype=np.float32)
        # Box at center of model input: xywh = (320, 320, 100, 100)
        det[0, 0, 0] = 320.0
        det[0, 1, 0] = 320.0
        det[0, 2, 0] = 100.0
        det[0, 3, 0] = 100.0
        det[0, 4, 0] = 0.9
        det[0, 9:41, 0] = 1.0

        proto = np.ones((1, 32, 160, 160), dtype=np.float32) * 0.1

        outputs = {"det": det, "proto": proto}
        meta = {"original_shape": (1080, 1920)}

        result = wrapper.postprocess(outputs, meta)

        assert result.bboxes.shape[0] == 1
        assert result.masks.shape == (1, 1080, 1920)
        assert result.masks[0].sum() > 0
        # Mask center should be near the center of the original image
        ys, xs = np.where(result.masks[0] > 0)
        center_x = (xs.min() + xs.max()) / 2
        center_y = (ys.min() + ys.max()) / 2
        assert abs(center_x - 960) < 30  # within 30px of image center
        assert abs(center_y - 540) < 30

    def test_model_type_in_is_export_parameters(self) -> None:
        """IS model should report model_type='YOLO11-seg'."""
        from getitune.backend.ultralytics.models.instance_segmentation import UltralyticsInstSegModel
        from getitune.types.label import LabelInfo

        label_info = LabelInfo(label_names=["a"], label_ids=["0"], label_groups=[["a"]])
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=label_info)
        params = model._export_parameters
        assert params.model_type == "YOLO11-seg"
        assert params.task_type == "instance_segmentation"
