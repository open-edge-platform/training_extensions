# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import onnx
import pytest
import torch
from torch.export import Dim

from getitune.backend.lightning.exporter.native import LightningModelExporter
from getitune.backend.lightning.models.base import DataInputParams
from getitune.types.export import TaskLevelExportParameters
from getitune.types.precision import Precision


class TestLightningModelExporter:
    @pytest.fixture
    def exporter(self, mocker):
        return LightningModelExporter(
            task_level_export_parameters=mocker.MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        )

    @pytest.fixture
    def dummy_model(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

    def test_to_openvino_export(self, exporter, dummy_model, tmp_path):
        output_dir = tmp_path / "model_export"
        output_dir.mkdir()

        exported_path = exporter.to_openvino(
            model=dummy_model,
            output_dir=output_dir,
            base_model_name="test_model",
            precision=Precision.FP32,
        )

        assert exported_path.exists()
        assert (output_dir / "test_model.xml").exists()
        assert (output_dir / "test_model.bin").exists()

        exporter.via_onnx = True
        exported_path = exporter.to_openvino(
            model=dummy_model,
            output_dir=output_dir,
            base_model_name="test_model",
            precision=Precision.FP32,
        )

        assert exported_path.exists()
        assert (output_dir / "test_model.xml").exists()
        assert (output_dir / "test_model.bin").exists()

    def test_to_onnx_export(self, exporter, dummy_model, tmp_path):
        output_dir = tmp_path / "onnx_export"
        output_dir.mkdir()

        exported_path = exporter.to_onnx(
            model=dummy_model,
            output_dir=output_dir,
            base_model_name="test_onnx_model",
            precision=Precision.FP32,
        )

        assert exported_path.exists()
        assert (output_dir / "test_onnx_model.onnx").exists()

        onnx_model = onnx.load(str(exported_path))
        onnx.checker.check_model(onnx_model)


def _make_exporter(onnx_config) -> LightningModelExporter:
    """Create a LightningModelExporter with the given ONNX export configuration."""
    return LightningModelExporter(
        task_level_export_parameters=MagicMock(TaskLevelExportParameters),
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        onnx_export_configuration=onnx_config,
    )


class TestBuildLegacyOnnxConfig:
    """Tests for _build_legacy_onnx_config converting dynamo params to legacy."""

    def test_converts_dynamic_shapes_to_dynamic_axes(self):
        exporter = _make_exporter(
            {
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
                "opset_version": 18,
            }
        )
        legacy = exporter._build_legacy_onnx_config()

        assert "dynamic_shapes" not in legacy
        assert legacy["dynamic_axes"] == {"images": {0: "batch"}}
        assert "autograd_inlining" not in legacy
        assert legacy["dynamo"] is False
        assert legacy["opset_version"] == 18
        assert legacy["input_names"] == ["images"]

    def test_preserves_existing_dynamic_axes(self):
        exporter = _make_exporter(
            {
                "input_names": ["image"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "dynamic_axes": {"image": {0: "batch", 2: "height", 3: "width"}},
            }
        )
        legacy = exporter._build_legacy_onnx_config()

        assert legacy["dynamic_axes"] == {"image": {0: "batch", 2: "height", 3: "width"}}

    def test_no_dynamic_shapes_is_noop(self):
        exporter = _make_exporter(
            {
                "input_names": ["image"],
                "opset_version": 18,
            }
        )
        legacy = exporter._build_legacy_onnx_config()

        assert "dynamic_shapes" not in legacy
        assert "dynamic_axes" not in legacy
        assert legacy["dynamo"] is False

    def test_does_not_mutate_original_config(self):
        orig = {
            "input_names": ["images"],
            "dynamic_shapes": {"inputs": {0: Dim("batch")}},
            "autograd_inlining": False,
        }
        exporter = _make_exporter(orig)
        exporter._build_legacy_onnx_config()

        assert "dynamic_shapes" in exporter.onnx_export_configuration
        assert "autograd_inlining" in exporter.onnx_export_configuration

    def test_multiple_inputs(self):
        exporter = _make_exporter(
            {
                "input_names": ["images", "masks"],
                "dynamic_shapes": {
                    "inputs": {0: Dim("batch")},
                    "mask_inputs": {0: Dim("batch"), 1: Dim("num_masks")},
                },
            }
        )
        legacy = exporter._build_legacy_onnx_config()

        assert legacy["dynamic_axes"] == {
            "images": {0: "batch"},
            "masks": {0: "batch", 1: "num_masks"},
        }


class TestExportOnnxFallback:
    """Tests for _export_onnx dynamo-to-legacy fallback."""

    @patch("getitune.backend.lightning.exporter.native.torch.onnx.export")
    def test_no_fallback_when_dynamo_succeeds(self, mock_export, tmp_path):
        exporter = _make_exporter(
            {
                "input_names": ["images"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
            }
        )
        model = MagicMock()
        tensor = torch.rand(1, 3, 224, 224)

        exporter._export_onnx(model, tensor, str(tmp_path / "test.onnx"))

        mock_export.assert_called_once()

    @patch("getitune.backend.lightning.exporter.native.torch.onnx.export")
    def test_fallback_to_legacy_on_dynamo_failure(self, mock_export, tmp_path):
        mock_export.side_effect = [TypeError("'int' object is not subscriptable"), None]

        exporter = _make_exporter(
            {
                "input_names": ["images"],
                "output_names": ["bboxes"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
                "autograd_inlining": False,
                "opset_version": 18,
            }
        )
        model = MagicMock()
        tensor = torch.rand(1, 3, 224, 224)

        exporter._export_onnx(model, tensor, str(tmp_path / "test.onnx"))

        assert mock_export.call_count == 2
        retry_kwargs = mock_export.call_args_list[1][1]
        assert "dynamic_shapes" not in retry_kwargs
        assert retry_kwargs["dynamic_axes"] == {"images": {0: "batch"}}
        assert retry_kwargs["dynamo"] is False
        assert "autograd_inlining" not in retry_kwargs

    @patch("getitune.backend.lightning.exporter.native.torch.onnx.export")
    def test_no_fallback_without_dynamic_shapes(self, mock_export, tmp_path):
        mock_export.side_effect = RuntimeError("some error")

        exporter = _make_exporter(
            {
                "input_names": ["image"],
                "dynamic_axes": {"image": {0: "batch"}},
                "dynamo": False,
            }
        )
        model = MagicMock()
        tensor = torch.rand(1, 3, 224, 224)

        with pytest.raises(RuntimeError, match="some error"):
            exporter._export_onnx(model, tensor, str(tmp_path / "test.onnx"))

        mock_export.assert_called_once()

    @patch("getitune.backend.lightning.exporter.native.torch.onnx.export")
    def test_legacy_fallback_error_propagates(self, mock_export, tmp_path):
        mock_export.side_effect = [
            TypeError("dynamo bug"),
            RuntimeError("legacy also failed"),
        ]

        exporter = _make_exporter(
            {
                "input_names": ["images"],
                "dynamic_shapes": {"inputs": {0: Dim("batch")}},
            }
        )
        model = MagicMock()
        tensor = torch.rand(1, 3, 224, 224)

        with pytest.raises(RuntimeError, match="legacy also failed"):
            exporter._export_onnx(model, tensor, str(tmp_path / "test.onnx"))
