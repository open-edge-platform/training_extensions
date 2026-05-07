# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ultralytics model wrappers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.lightning.models.base import DataInputParams
from getitune.backend.ultralytics.models import UltralyticsDetectionModel
from getitune.backend.ultralytics.models.base import UltralyticsModel
from getitune.types.export import TaskLevelExportParameters
from getitune.types.label import LabelInfo


def _label_info() -> LabelInfo:
    return LabelInfo(label_names=["cat", "dog"], label_ids=["0", "1"], label_groups=[["cat", "dog"]])


def test_model_rejects_checkpoint_name_for_scratch_training() -> None:
    with pytest.raises(ValueError, match="pretrained=False requires a model config"):
        UltralyticsDetectionModel(model_name="yolo26n.pt", pretrained=False)


def test_model_allows_yaml_config_for_scratch_training() -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
    assert model.model_name == "yolo26n.yaml"
    assert model.pretrained is False


def test_default_model_name_is_variant_key() -> None:
    """Default model name should be a variant key matching _pretrained_weights."""
    assert UltralyticsDetectionModel.default_model_name in UltralyticsDetectionModel._pretrained_weights


def test_load_checkpoint_calls_yolo_load(tmp_path: Path) -> None:
    """load_checkpoint should delegate to YOLO.load with the weights path."""
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
    fake_weights = tmp_path / "weights.pt"
    fake_weights.write_bytes(b"fake")

    mock_yolo = MagicMock()
    with patch.object(UltralyticsModel, "_build_yolo", return_value=mock_yolo):
        model.load_checkpoint(fake_weights)

    mock_yolo.load.assert_called_once_with(str(fake_weights))


def test_load_checkpoint_raises_on_missing_file() -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        model.load_checkpoint("/nonexistent/weights.pt")


class TestDataInputParams:
    """Tests for the data_input_params property on UltralyticsModel."""

    def test_returns_data_input_params(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, imgsz=640)
        params = model.data_input_params
        assert isinstance(params, DataInputParams)

    def test_input_size_matches_imgsz(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, imgsz=320)
        params = model.data_input_params
        assert params.input_size == (320, 320)

    def test_mean_is_zero(self) -> None:
        """YOLO expects identity normalization — mean should be (0, 0, 0)."""
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
        assert model.data_input_params.mean == (0.0, 0.0, 0.0)

    def test_std_is_255(self) -> None:
        """YOLO export metadata needs scale_values=255 so ModelAPI divides by 255."""
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
        assert model.data_input_params.std == (255.0, 255.0, 255.0)

    def test_no_intensity_config(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
        assert model.data_input_params.intensity_config is None

    def test_default_imgsz_from_preprocessing_params(self) -> None:
        """When imgsz is not specified, it should come from _default_preprocessing_params."""
        model = UltralyticsDetectionModel(model_name="yolo26n", pretrained=False)
        assert model.imgsz == 640
        assert model.data_input_params.input_size == (640, 640)


class TestExportParameters:
    """Tests for the _export_parameters property on UltralyticsModel."""

    def test_returns_task_level_export_parameters(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        params = model._export_parameters
        assert isinstance(params, TaskLevelExportParameters)

    def test_model_type_is_yolo11(self) -> None:
        """Detection model type should be 'YOLO11' for ModelAPI YOLO adapter."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        assert model._export_parameters.model_type == "YOLO11"

    def test_task_type_is_detection(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        assert model._export_parameters.task_type == "detection"

    def test_model_name_from_model(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False, label_info=_label_info())
        assert model._export_parameters.model_name == "yolo26n.yaml"

    def test_label_info_from_model(self) -> None:
        li = _label_info()
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=li)
        assert model._export_parameters.label_info == li

    def test_none_label_info_uses_empty(self) -> None:
        """When label_info is None, should use an empty LabelInfo."""
        model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
        params = model._export_parameters
        assert params.label_info.label_names == []
        assert params.label_info.label_ids == []

    def test_default_thresholds(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        params = model._export_parameters
        assert params.confidence_threshold == 0.25
        assert params.iou_threshold == 0.7

    def test_optimization_config_empty(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        assert model._export_parameters.optimization_config == {}

    def test_to_metadata_produces_valid_dict(self) -> None:
        """to_metadata should produce a valid metadata dict with all required keys."""
        model = UltralyticsDetectionModel(model_name="yolo26n", label_info=_label_info())
        metadata = model._export_parameters.to_metadata()
        assert ("model_info", "model_type") in metadata
        assert ("model_info", "task_type") in metadata
        assert ("model_info", "labels") in metadata
        assert all(isinstance(v, str) for v in metadata.values())


class TestPretrainedWeights:
    """Tests for the _pretrained_weights pattern."""

    def test_pretrained_weights_defined(self) -> None:
        assert len(UltralyticsDetectionModel._pretrained_weights) > 0

    def test_default_model_in_pretrained_weights(self) -> None:
        assert UltralyticsDetectionModel.default_model_name in UltralyticsDetectionModel._pretrained_weights

    def test_build_yolo_loads_pretrained_when_enabled(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", pretrained=True)
        mock_yolo = MagicMock()
        with patch("getitune.backend.ultralytics.models.base.YOLO", return_value=mock_yolo):
            yolo = model._build_yolo()

        mock_yolo.load.assert_called_once_with(UltralyticsDetectionModel._pretrained_weights["yolo26n"])
        assert yolo is mock_yolo

    def test_build_yolo_skips_pretrained_when_disabled(self) -> None:
        model = UltralyticsDetectionModel(model_name="yolo26n", pretrained=False)
        mock_yolo = MagicMock()
        with patch("getitune.backend.ultralytics.models.base.YOLO", return_value=mock_yolo):
            yolo = model._build_yolo()

        mock_yolo.load.assert_not_called()
        assert yolo is mock_yolo
