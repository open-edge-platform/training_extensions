# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics engine."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torchvision import tv_tensors

from getitune.backend.ultralytics.engine import UltralyticsEngine
from getitune.backend.ultralytics.models import UltralyticsDetectionModel
from getitune.data.entity.base import ImageInfo
from getitune.data.entity.sample import SampleBatch
from getitune.data.module import DataModule
from getitune.types.export import ExportFormat
from getitune.types.label import LabelInfo
from getitune.types.precision import Precision


def _label_info() -> LabelInfo:
    return LabelInfo(
        label_names=["a", "b"],
        label_ids=["0", "1"],
        label_groups=[["a", "b"]],
    )


def _make_engine(tmp_path: Path, mocker) -> tuple[UltralyticsEngine, MagicMock]:
    """Create an engine with a mocked YOLO model for unit testing."""
    model = UltralyticsDetectionModel(label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

    yolo = MagicMock()
    model._yolo = yolo
    return engine, yolo


def test_ultralytics_engine_supports_ultralytics_model_with_datamodule(mocker) -> None:
    model = UltralyticsDetectionModel(label_info=_label_info())
    data = mocker.MagicMock(spec=DataModule)

    assert UltralyticsEngine.is_supported(model, data)


def test_predict_with_datamodule_uses_predict_dataloader(mocker, tmp_path) -> None:
    model = UltralyticsDetectionModel(label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    datamodule.predict_dataloader.return_value = [
        SampleBatch(
            images=tv_tensors.Image(torch.rand(1, 3, 8, 8)),
            imgs_info=[
                ImageInfo(img_idx=0, img_shape=(8, 8), ori_shape=(8, 8))  # pyrefly: ignore[no-matching-overload]
            ],
        )
    ]

    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

    yolo = MagicMock()
    yolo.model = MagicMock()
    yolo.model.to.return_value = yolo.model
    yolo.model.eval.return_value = yolo.model
    yolo.predict.return_value = [
        SimpleNamespace(
            orig_img=np.zeros((8, 8, 3), dtype=np.uint8),
            orig_shape=(8, 8),
            boxes=None,
            masks=None,
        )
    ]
    model._yolo = yolo

    predictions = engine.predict(conf=0.25)

    datamodule.predict_dataloader.assert_called_once_with()
    assert len(predictions) == 1


class TestResolveExportModel:
    """Tests for checkpoint resolution logic in export."""

    def test_explicit_checkpoint_loads_yolo(self, mocker, tmp_path) -> None:
        """Providing a checkpoint path should load a fresh YOLO from that file."""
        engine, _ = _make_engine(tmp_path, mocker)

        ckpt_file = tmp_path / "custom.pt"
        ckpt_file.touch()

        mock_yolo_cls = MagicMock()
        with patch("ultralytics.YOLO", mock_yolo_cls):
            result = engine._resolve_export_model(checkpoint=ckpt_file)

        mock_yolo_cls.assert_called_once_with(str(ckpt_file))
        assert result is mock_yolo_cls.return_value

    def test_explicit_checkpoint_not_found_raises(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            engine._resolve_export_model(checkpoint=tmp_path / "nonexistent.pt")

    def test_auto_discovers_best_pt(self, mocker, tmp_path) -> None:
        """When no checkpoint given, best.pt from train dir should be used."""
        engine, _ = _make_engine(tmp_path, mocker)

        best_pt = tmp_path / "train" / "weights" / "best.pt"
        best_pt.parent.mkdir(parents=True)
        best_pt.touch()

        mock_yolo_cls = MagicMock()
        with patch("ultralytics.YOLO", mock_yolo_cls):
            result = engine._resolve_export_model(checkpoint=None)

        mock_yolo_cls.assert_called_once_with(str(best_pt))
        assert result is mock_yolo_cls.return_value

    def test_fallback_to_current_model(self, mocker, tmp_path) -> None:
        """Without checkpoint or best.pt, the current model is returned."""
        engine, yolo = _make_engine(tmp_path, mocker)

        result = engine._resolve_export_model(checkpoint=None)
        assert result is yolo


class TestExport:
    """Tests for the export() method."""

    def test_unsupported_format_raises(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)

        # Create a fake ExportFormat value that isn't in the map
        bad_format = MagicMock(spec=ExportFormat)
        bad_format.value = "TORCHSCRIPT"

        with pytest.raises(ValueError, match="Unsupported export format"):
            engine.export(export_format=bad_format)

    def test_export_openvino_fp32(self, mocker, tmp_path) -> None:
        """OpenVINO FP32 export should call yolo.export with half=False."""
        engine, _ = _make_engine(tmp_path, mocker)

        # Set up the fake export output: a directory with a .xml file
        export_dir = tmp_path / "fake_export"
        export_dir.mkdir()
        (export_dir / "model.xml").touch()
        (export_dir / "model.bin").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(export_dir)

        with patch.object(engine, "_resolve_export_model", return_value=mock_yolo):
            result = engine.export(
                export_format=ExportFormat.OPENVINO,
                export_precision=Precision.FP32,
            )

        mock_yolo.export.assert_called_once_with(
            format="openvino",
            imgsz=engine._model.imgsz,
            half=False,
        )
        assert result.suffix == ".xml"
        assert result.exists()

    def test_export_openvino_fp16_sets_half(self, mocker, tmp_path) -> None:
        """OpenVINO FP16 export should call yolo.export with half=True."""
        engine, _ = _make_engine(tmp_path, mocker)

        export_dir = tmp_path / "fake_export"
        export_dir.mkdir()
        (export_dir / "model.xml").touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(export_dir)

        with patch.object(engine, "_resolve_export_model", return_value=mock_yolo):
            result = engine.export(
                export_format=ExportFormat.OPENVINO,
                export_precision=Precision.FP16,
            )

        mock_yolo.export.assert_called_once_with(
            format="openvino",
            imgsz=engine._model.imgsz,
            half=True,
        )
        assert result.suffix == ".xml"

    def test_export_onnx_fp32(self, mocker, tmp_path) -> None:
        """ONNX FP32 export should call yolo.export with half=False."""
        engine, _ = _make_engine(tmp_path, mocker)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(onnx_file)

        with patch.object(engine, "_resolve_export_model", return_value=mock_yolo):
            result = engine.export(
                export_format=ExportFormat.ONNX,
                export_precision=Precision.FP32,
            )

        mock_yolo.export.assert_called_once_with(
            format="onnx",
            imgsz=engine._model.imgsz,
            half=False,
        )
        assert result.suffix == ".onnx"

    def test_export_onnx_fp16_sets_half(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(onnx_file)

        with patch.object(engine, "_resolve_export_model", return_value=mock_yolo):
            result = engine.export(
                export_format=ExportFormat.ONNX,
                export_precision=Precision.FP16,
            )

        mock_yolo.export.assert_called_once_with(
            format="onnx",
            imgsz=engine._model.imgsz,
            half=True,
        )
        assert result.suffix == ".onnx"

    def test_export_with_explicit_checkpoint(self, mocker, tmp_path) -> None:
        """Checkpoint arg should be forwarded to _resolve_export_model."""
        engine, _ = _make_engine(tmp_path, mocker)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        ckpt_file = tmp_path / "custom.pt"
        ckpt_file.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(onnx_file)

        with patch.object(engine, "_resolve_export_model", return_value=mock_yolo) as mock_resolve:
            engine.export(
                checkpoint=ckpt_file,
                export_format=ExportFormat.ONNX,
                export_precision=Precision.FP32,
            )

        mock_resolve.assert_called_once_with(ckpt_file)

    def test_export_passes_extra_kwargs(self, mocker, tmp_path) -> None:
        """Extra kwargs should be forwarded to yolo.export()."""
        engine, _ = _make_engine(tmp_path, mocker)

        onnx_file = tmp_path / "model.onnx"
        onnx_file.touch()

        mock_yolo = MagicMock()
        mock_yolo.export.return_value = str(onnx_file)

        with patch.object(engine, "_resolve_export_model", return_value=mock_yolo):
            engine.export(
                export_format=ExportFormat.ONNX,
                export_precision=Precision.FP32,
                simplify=True,
                dynamic=True,
            )

        _, call_kwargs = mock_yolo.export.call_args
        assert call_kwargs["simplify"] is True
        assert call_kwargs["dynamic"] is True


class TestNormalizeOpenvinoExport:
    """Tests for OpenVINO export normalization."""

    def test_copies_directory_to_target(self, mocker, tmp_path) -> None:
        """Export dir should be copied to work_dir/exported_model/."""
        engine, _ = _make_engine(tmp_path, mocker)

        src_dir = tmp_path / "src_export"
        src_dir.mkdir()
        (src_dir / "model.xml").touch()
        (src_dir / "model.bin").touch()

        result = engine._normalize_openvino_export(src_dir)

        target_dir = tmp_path / "exported_model"
        assert target_dir.exists()
        assert result.parent == target_dir
        assert result.suffix == ".xml"
        assert (target_dir / "model.bin").exists()

    def test_no_xml_raises(self, mocker, tmp_path) -> None:
        """Empty export dir with no .xml should raise FileNotFoundError."""
        engine, _ = _make_engine(tmp_path, mocker)

        src_dir = tmp_path / "empty_export"
        src_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No .xml file found"):
            engine._normalize_openvino_export(src_dir)

    def test_already_at_target_no_copy(self, mocker, tmp_path) -> None:
        """If export is already at target_dir, no copy should happen."""
        engine, _ = _make_engine(tmp_path, mocker)

        target_dir = tmp_path / "exported_model"
        target_dir.mkdir()
        (target_dir / "model.xml").touch()

        result = engine._normalize_openvino_export(target_dir)
        assert result == target_dir / "model.xml"

    def test_file_path_returned_as_is(self, mocker, tmp_path) -> None:
        """If export_path is a file (not dir), return it directly."""
        engine, _ = _make_engine(tmp_path, mocker)

        xml_file = tmp_path / "model.xml"
        xml_file.touch()

        result = engine._normalize_openvino_export(xml_file)
        assert result == xml_file


class TestNormalizeOnnxExport:
    """Tests for ONNX export normalization."""

    def test_copies_file_to_target(self, mocker, tmp_path) -> None:
        """Export file should be copied to work_dir/exported_model.onnx."""
        engine, _ = _make_engine(tmp_path, mocker)

        src_file = tmp_path / "src" / "model.onnx"
        src_file.parent.mkdir()
        src_file.write_bytes(b"fake_onnx_data")

        result = engine._normalize_onnx_export(src_file)

        expected = tmp_path / "exported_model.onnx"
        assert result == expected
        assert expected.exists()
        assert expected.read_bytes() == b"fake_onnx_data"

    def test_already_at_target_no_copy(self, mocker, tmp_path) -> None:
        """If file is already at target path, no error."""
        engine, _ = _make_engine(tmp_path, mocker)

        target_file = tmp_path / "exported_model.onnx"
        target_file.write_bytes(b"data")

        result = engine._normalize_onnx_export(target_file)
        assert result == target_file
