# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Ultralytics engine."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
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
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

    yolo = MagicMock()
    model._yolo = yolo
    return engine, yolo


def test_train_args_are_train_only(mocker, tmp_path) -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
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
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    data = mocker.MagicMock(spec=DataModule)

    assert UltralyticsEngine.is_supported(model, data)


def test_engine_propagates_intensity_config_from_datamodule(mocker, tmp_path) -> None:
    """When DataModule has input_intensity_config, engine should propagate it to the model."""
    from getitune.config.data import IntensityConfig

    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    datamodule = mocker.MagicMock(spec=DataModule)
    uint16_cfg = IntensityConfig(mode="scale_to_unit", storage_dtype="uint16")
    datamodule.input_intensity_config = uint16_cfg

    engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

    assert model._intensity_config is uint16_cfg
    assert engine._model.data_input_params.intensity_config is uint16_cfg
    assert engine._model.data_input_params.intensity_config.storage_dtype == "uint16"


def test_engine_default_intensity_config_without_datamodule(tmp_path) -> None:
    """Without DataModule (upstream data path), model should use default uint8 intensity config."""
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
    engine = UltralyticsEngine(model=model, data=tmp_path, work_dir=tmp_path / "work", device="cpu")

    ic = engine._model.data_input_params.intensity_config
    assert ic is not None
    assert ic.mode == "scale_to_unit"
    assert ic.storage_dtype == "uint8"


def test_predict_with_datamodule_uses_predict_dataloader(mocker, tmp_path) -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
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


class TestExportCheckpointResolution:
    """Tests for checkpoint resolution logic in export (now via model.load_checkpoint)."""

    def test_explicit_checkpoint_loads_via_model(self, mocker, tmp_path) -> None:
        """Providing a checkpoint path should call model.load_checkpoint."""
        engine, _ = _make_engine(tmp_path, mocker)

        ckpt_file = tmp_path / "custom.pt"
        ckpt_file.touch()

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export(checkpoint=ckpt_file)

        mock_load.assert_called_once_with(ckpt_file)

    def test_auto_discovers_best_pt(self, mocker, tmp_path) -> None:
        """When no checkpoint given, best.pt from train dir should be used."""
        engine, _ = _make_engine(tmp_path, mocker)

        best_pt = tmp_path / "train" / "weights" / "best.pt"
        best_pt.parent.mkdir(parents=True)
        best_pt.touch()

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export()

        mock_load.assert_called_once_with(best_pt)

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

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export()

        mock_load.assert_called_once_with(recorded_ckpt.resolve())

    def test_no_checkpoint_no_best_pt_does_not_load(self, mocker, tmp_path) -> None:
        """Without checkpoint or best.pt, load_checkpoint should not be called."""
        engine, yolo = _make_engine(tmp_path, mocker)

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml"),
        ):
            engine.export()

        mock_load.assert_not_called()


class TestExport:
    """Tests for the export() method."""

    def test_export_delegates_to_model_export(self, mocker, tmp_path) -> None:
        """export() should delegate to model.export() with correct args."""
        engine, _ = _make_engine(tmp_path, mocker)

        with (
            patch.object(engine._model, "load_checkpoint"),
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.xml") as mock_export,
        ):
            # Create a checkpoint so load_checkpoint gets called
            best_pt = tmp_path / "train" / "weights" / "best.pt"
            best_pt.parent.mkdir(parents=True)
            best_pt.touch()

            result = engine.export(
                export_format=ExportFormat.OPENVINO,
                export_precision=Precision.FP32,
            )

        mock_export.assert_called_once_with(
            output_dir=engine._work_dir,
            base_name="exported_model",
            export_format=ExportFormat.OPENVINO,
            precision=Precision.FP32,
        )
        assert result == tmp_path / "exported_model.xml"

    def test_export_with_explicit_checkpoint(self, mocker, tmp_path) -> None:
        """Checkpoint arg should be forwarded to model.load_checkpoint."""
        engine, _ = _make_engine(tmp_path, mocker)
        ckpt_file = tmp_path / "custom.pt"
        ckpt_file.touch()

        with (
            patch.object(engine._model, "load_checkpoint") as mock_load,
            patch.object(engine._model, "export", return_value=tmp_path / "exported_model.onnx"),
        ):
            engine.export(
                checkpoint=ckpt_file,
                export_format=ExportFormat.ONNX,
                export_precision=Precision.FP32,
            )

        mock_load.assert_called_once_with(ckpt_file)

    def test_instance_segmentation_export_succeeds(self, mocker, tmp_path) -> None:
        """Segmentation export should succeed and produce an IR file."""
        model = UltralyticsInstSegModel(model_name="yolo26n-seg", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        with patch.object(model, "export", return_value=tmp_path / "exported_model.xml") as mock_export:
            result = engine.export(export_format=ExportFormat.OPENVINO, export_precision=Precision.FP32)

        assert result == tmp_path / "exported_model.xml"
        mock_export.assert_called_once()

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

        with (
            patch.object(engine, "_make_bound_validator", return_value=validator_cls),
            patch.object(engine._model, "load_checkpoint") as mock_load,
        ):
            metrics = engine.test(checkpoint=ckpt_file)

        mock_load.assert_called_once_with(ckpt_file)
        assert metrics == {"val/map_50": 0.5}

    def test_test_with_data_root_loads_explicit_checkpoint(self, tmp_path) -> None:
        """Filesystem validation should use a fresh YOLO model for explicit checkpoint."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        engine = UltralyticsEngine(model=model, data=tmp_path, work_dir=tmp_path / "work", device="cpu")
        ckpt_file = tmp_path / "best.pt"
        ckpt_file.touch()

        mock_yolo = MagicMock()
        mock_yolo.val.return_value = {"metrics/mAP50(B)": 0.25}

        with (
            patch.object(model, "load_checkpoint") as mock_load,
            patch.object(type(model), "yolo", new_callable=lambda: property(lambda _: mock_yolo)),
        ):
            metrics = engine.test(checkpoint=ckpt_file)

        mock_load.assert_called_once_with(ckpt_file)
        mock_yolo.val.assert_called_once()
        assert metrics == {"val/map_50": 0.25}

    def test_engine_loads_persisted_checkpoint_pointer(self, mocker, tmp_path) -> None:
        """Fresh engine instances should reuse the last recorded training checkpoint."""

        recorded_ckpt = tmp_path / "custom_train" / "weights" / "best.pt"
        recorded_ckpt.parent.mkdir(parents=True)
        recorded_ckpt.touch()
        (tmp_path / ".last_train_checkpoint").write_text(str(recorded_ckpt.resolve()), encoding="utf-8")

        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        datamodule = mocker.MagicMock(spec=DataModule)
        engine = UltralyticsEngine(model=model, data=datamodule, work_dir=tmp_path, device="cpu")

        assert engine._last_train_checkpoint == recorded_ckpt.resolve()


class TestModelExporter:
    """Tests for the model's _exporter property."""

    def test_returns_ultralytics_exporter(self, mocker, tmp_path) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())

        from getitune.backend.ultralytics.exporter import UltralyticsModelExporter

        assert isinstance(model._exporter, UltralyticsModelExporter)

    def test_uses_model_data_input_params(self, mocker, tmp_path) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        exporter = model._exporter

        assert exporter.data_input_params.mean == (0.0, 0.0, 0.0)
        assert exporter.data_input_params.std == (1.0, 1.0, 1.0)
        assert exporter.data_input_params.intensity_config is not None
        assert exporter.data_input_params.intensity_config.mode == "scale_to_unit"

    def test_default_yolo_preprocessing_values(self, mocker, tmp_path) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        exporter = model._exporter

        assert exporter.resize_mode == "fit_to_window_letterbox"
        assert exporter.pad_value == 114
        assert exporter.swap_rgb is True
