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
from getitune.backend.ultralytics.models import UltralyticsDetectionModel, UltralyticsInstSegModel
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


def test_train_args_are_train_only(mocker, tmp_path) -> None:
    model = UltralyticsDetectionModel(label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    engine = UltralyticsEngine(
        model=model,
        data=datamodule,
        work_dir=tmp_path,
        device="cpu",
        train_args={"epochs": 7, "lr0": 0.01},
    )

    yolo = MagicMock()
    model._yolo = yolo

    engine.train()
    _, train_kwargs = yolo.train.call_args
    assert train_kwargs["epochs"] == 7
    assert train_kwargs["lr0"] == 0.01

    with patch.object(engine, "_test_with_datamodule", return_value={}) as test_with_datamodule:
        engine.test()
    test_args, _ = test_with_datamodule.call_args
    assert "epochs" not in test_args[0]
    assert "lr0" not in test_args[0]


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

    def test_prefers_recorded_train_checkpoint(self, mocker, tmp_path) -> None:
        """Recorded checkpoints from train() should win over hardcoded train/best.pt."""
        engine, _ = _make_engine(tmp_path, mocker)

        recorded_ckpt = tmp_path / "custom_run" / "weights" / "best.pt"
        recorded_ckpt.parent.mkdir(parents=True)
        recorded_ckpt.touch()
        engine._record_last_train_checkpoint(recorded_ckpt)

        fallback_best = tmp_path / "train" / "weights" / "best.pt"
        fallback_best.parent.mkdir(parents=True)
        fallback_best.touch()

        mock_yolo_cls = MagicMock()
        with patch("ultralytics.YOLO", mock_yolo_cls):
            result = engine._resolve_export_model(checkpoint=None)

        mock_yolo_cls.assert_called_once_with(str(recorded_ckpt))
        assert result is mock_yolo_cls.return_value

    def test_stale_recorded_checkpoint_falls_back_to_default_best(self, mocker, tmp_path) -> None:
        """Missing recorded checkpoint should not block fallback to work_dir/train/weights/best.pt."""
        engine, _ = _make_engine(tmp_path, mocker)

        stale_ckpt = tmp_path / "custom_run" / "weights" / "best.pt"
        engine._record_last_train_checkpoint(stale_ckpt)
        stale_ckpt.parent.mkdir(parents=True)
        checkpoint_file = tmp_path / ".last_train_checkpoint"
        assert checkpoint_file.exists()

        default_best = tmp_path / "train" / "weights" / "best.pt"
        default_best.parent.mkdir(parents=True)
        default_best.touch()

        mock_yolo_cls = MagicMock()
        with patch("ultralytics.YOLO", mock_yolo_cls):
            result = engine._resolve_export_model(checkpoint=None)

        mock_yolo_cls.assert_called_once_with(str(default_best))
        assert result is mock_yolo_cls.return_value
        assert engine._last_train_checkpoint is None
        assert not checkpoint_file.exists()

    def test_fallback_to_current_model(self, mocker, tmp_path) -> None:
        """Without checkpoint or best.pt, the current model is returned."""
        engine, yolo = _make_engine(tmp_path, mocker)

        result = engine._resolve_export_model(checkpoint=None)
        assert result is yolo


class TestExport:
    """Tests for the export() method."""

    def test_unsupported_format_raises(self, mocker, tmp_path) -> None:
        """Unsupported formats should be rejected by the exporter dispatch."""
        engine, _ = _make_engine(tmp_path, mocker)

        bad_format = MagicMock(spec=ExportFormat)
        bad_format.value = "TORCHSCRIPT"

        mock_exporter = MagicMock()
        mock_exporter.export.side_effect = ValueError("Unsupported export format")

        with (
            patch.object(engine, "_resolve_export_model", return_value=MagicMock()),
            patch.object(engine, "_build_exporter", return_value=mock_exporter),
        ):
            with pytest.raises(ValueError, match="Unsupported export format"):
                engine.export(export_format=bad_format)

    def test_export_delegates_to_exporter(self, mocker, tmp_path) -> None:
        """export() should delegate to exporter.export() with correct args."""
        engine, _ = _make_engine(tmp_path, mocker)

        mock_yolo = MagicMock()
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = tmp_path / "exported_model.xml"

        with (
            patch.object(engine, "_resolve_export_model", return_value=mock_yolo),
            patch.object(engine, "_build_exporter", return_value=mock_exporter),
        ):
            result = engine.export(
                export_format=ExportFormat.OPENVINO,
                export_precision=Precision.FP32,
            )

        mock_exporter.export.assert_called_once_with(
            model=mock_yolo,
            output_dir=engine._work_dir,
            base_model_name="exported_model",
            export_format=ExportFormat.OPENVINO,
            precision=Precision.FP32,
        )
        assert result == tmp_path / "exported_model.xml"

    def test_export_with_explicit_checkpoint(self, mocker, tmp_path) -> None:
        """Checkpoint arg should be forwarded to _resolve_export_model."""
        engine, _ = _make_engine(tmp_path, mocker)
        ckpt_file = tmp_path / "custom.pt"
        ckpt_file.touch()

        mock_yolo = MagicMock()
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = tmp_path / "exported_model.onnx"

        with (
            patch.object(engine, "_resolve_export_model", return_value=mock_yolo) as mock_resolve,
            patch.object(engine, "_build_exporter", return_value=mock_exporter),
        ):
            engine.export(
                checkpoint=ckpt_file,
                export_format=ExportFormat.ONNX,
                export_precision=Precision.FP32,
            )

        mock_resolve.assert_called_once_with(ckpt_file)

    def test_export_passes_configured_thresholds_to_exporter(self, mocker, tmp_path) -> None:
        """Export metadata should use thresholds configured via export_args."""
        model = UltralyticsDetectionModel(label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(
            model=model,
            data=datamodule,
            work_dir=tmp_path,
            device="cpu",
            export_args={"confidence_threshold": 0.4, "iou_threshold": 0.6},
        )

        exporter = engine._build_exporter()
        metadata = exporter.metadata
        assert metadata[("model_info", "confidence_threshold")] == "0.4"
        assert metadata[("model_info", "iou_threshold")] == "0.6"

    def test_instance_segmentation_export_is_blocked(self, mocker, tmp_path) -> None:
        """Segmentation export should fail until a compatible wrapper is validated."""
        model = UltralyticsInstSegModel(label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        with pytest.raises(NotImplementedError, match="instance-segmentation export"):
            engine.export(export_format=ExportFormat.OPENVINO, export_precision=Precision.FP32)

    def test_train_records_actual_trainer_checkpoint(self, mocker, tmp_path) -> None:
        """train() should persist the checkpoint chosen by the underlying trainer."""
        engine, yolo = _make_engine(tmp_path, mocker)

        best_ckpt = tmp_path / "custom_train" / "weights" / "best.pt"
        best_ckpt.parent.mkdir(parents=True)
        best_ckpt.touch()

        trainer = SimpleNamespace(best=best_ckpt, last=tmp_path / "custom_train" / "weights" / "last.pt")
        yolo.trainer = trainer
        yolo.train.return_value = {"fitness": 0.1}

        result = engine.train(name="custom_train")

        assert result == {"ultralytics/fitness": 0.1}
        assert engine._last_train_checkpoint == best_ckpt.resolve()
        assert (tmp_path / ".last_train_checkpoint").read_text(encoding="utf-8").strip() == str(best_ckpt.resolve())

    def test_test_with_datamodule_loads_explicit_checkpoint(self, mocker, tmp_path) -> None:
        """test(checkpoint=...) should validate the requested checkpoint."""
        engine, _ = _make_engine(tmp_path, mocker)
        ckpt_file = tmp_path / "best.pt"
        ckpt_file.touch()

        validator_cls = MagicMock()
        validator = MagicMock()
        validator.return_value = {"metrics/mAP50(B)": 0.5}
        validator_cls.return_value = validator

        loaded_yolo = MagicMock()
        loaded_yolo.model = MagicMock()

        with (
            patch.object(engine, "_make_bound_validator", return_value=validator_cls),
            patch.object(engine, "_resolve_export_model", return_value=loaded_yolo) as resolve,
        ):
            metrics = engine.test(checkpoint=ckpt_file)

        resolve.assert_called_once_with(ckpt_file)
        validator.assert_called_once_with(model=loaded_yolo.model)
        assert metrics == {"val/map_50": 0.5}

    def test_test_with_data_root_loads_explicit_checkpoint(self, tmp_path) -> None:
        """Filesystem validation should use a fresh YOLO model for explicit checkpoint."""
        model = UltralyticsDetectionModel(label_info=_label_info())
        engine = UltralyticsEngine(model=model, data=tmp_path, work_dir=tmp_path / "work", device="cpu")
        ckpt_file = tmp_path / "best.pt"
        ckpt_file.touch()

        loaded_yolo = MagicMock()
        loaded_yolo.val.return_value = {"metrics/mAP50(B)": 0.25}

        with patch.object(engine, "_resolve_export_model", return_value=loaded_yolo) as resolve:
            metrics = engine.test(checkpoint=ckpt_file)

        resolve.assert_called_once_with(ckpt_file)
        loaded_yolo.val.assert_called_once()
        assert metrics == {"val/map_50": 0.25}

    def test_engine_loads_persisted_checkpoint_pointer(self, mocker, tmp_path) -> None:
        """Fresh engine instances should reuse the last recorded training checkpoint."""

        recorded_ckpt = tmp_path / "custom_train" / "weights" / "best.pt"
        recorded_ckpt.parent.mkdir(parents=True)
        recorded_ckpt.touch()
        (tmp_path / ".last_train_checkpoint").write_text(str(recorded_ckpt.resolve()), encoding="utf-8")

        model = UltralyticsDetectionModel(label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        assert engine._last_train_checkpoint == recorded_ckpt.resolve()


class TestBuildExporter:
    """Tests for the _build_exporter helper."""

    def test_returns_ultralytics_exporter(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)
        exporter = engine._build_exporter()

        from getitune.backend.ultralytics.exporter import UltralyticsModelExporter

        assert isinstance(exporter, UltralyticsModelExporter)

    def test_uses_model_data_input_params(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)
        exporter = engine._build_exporter()

        assert exporter.data_input_params.mean == (0.0, 0.0, 0.0)
        assert exporter.data_input_params.std == (255.0, 255.0, 255.0)

    def test_default_yolo_preprocessing_values(self, mocker, tmp_path) -> None:
        engine, _ = _make_engine(tmp_path, mocker)
        exporter = engine._build_exporter()

        assert exporter.resize_mode == "fit_to_window_letterbox"
        assert exporter.pad_value == 114
        assert exporter.swap_rgb is True

    def test_threshold_overrides_from_export_args(self, mocker, tmp_path) -> None:
        model = UltralyticsDetectionModel(label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(
            model=model,
            data=datamodule,
            work_dir=tmp_path,
            device="cpu",
            export_args={"confidence_threshold": 0.4, "iou_threshold": 0.6},
        )
        exporter = engine._build_exporter()
        metadata = exporter.metadata

        assert metadata[("model_info", "confidence_threshold")] == "0.4"
        assert metadata[("model_info", "iou_threshold")] == "0.6"
