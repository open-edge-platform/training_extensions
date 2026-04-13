# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from onnx import ModelProto
from onnxconverter_common import float16

from otx.backend.native.exporter.base import OTXExportFormatType, OTXModelExporter, OTXPrecisionType
from otx.backend.native.models.base import DataInputParams
from otx.config.data import IntensityConfig
from otx.types.export import TaskLevelExportParameters


class MockModelExporter(OTXModelExporter):
    def to_openvino(self, model, output_dir, base_model_name, precision):
        return output_dir / f"{base_model_name}.xml"

    def to_onnx(self, model, output_dir, base_model_name, precision):
        return output_dir / f"{base_model_name}.onnx"


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def exporter():
    return MockModelExporter(
        task_level_export_parameters=MagicMock(TaskLevelExportParameters),
        data_input_params=DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    )


class TestOTXModelExporter:
    def test_to_openvino(self, mock_model, exporter, tmp_path):
        output_dir = tmp_path
        base_model_name = "test_model"
        precision = OTXPrecisionType.FP32
        result = exporter.export(mock_model, output_dir, base_model_name, OTXExportFormatType.OPENVINO, precision)
        assert result == output_dir / f"{base_model_name}.xml"

    def test_to_onnx(self, mock_model, exporter, tmp_path):
        output_dir = tmp_path
        base_model_name = "test_model"
        precision = OTXPrecisionType.FP32
        result = exporter.export(mock_model, output_dir, base_model_name, OTXExportFormatType.ONNX, precision)
        assert result == output_dir / f"{base_model_name}.onnx"

    def test_export_unsupported_format_raises(self, exporter, mock_model, tmp_path):
        export_format = "unsupported_format"
        with pytest.raises(ValueError, match=f"Unsupported export format: {export_format}"):
            exporter.export(mock_model, tmp_path, export_format=export_format)

    def test_postprocess_openvino_model(self, mock_model, exporter):
        # test output names do not match exporter parameters
        exporter.output_names = ["output1"]
        with pytest.raises(RuntimeError):
            exporter._postprocess_openvino_model(mock_model)
        # test output names match exporter parameters
        exporter.output_names = ["output1", "output2"]
        mock_model.outputs = []
        for output_name in exporter.output_names:
            output = MagicMock()
            output.get_names.return_value = output_name
            mock_model.outputs.append(output)
        processed_model = exporter._postprocess_openvino_model(mock_model)
        # Verify the processed model is returned and the names are set correctly
        assert processed_model is mock_model
        for output, name in zip(processed_model.outputs, exporter.output_names):
            output.tensor.set_names.assert_called_once_with({name})

    def test_embed_metadata_true_precision_fp16(self, exporter):
        onnx_model = ModelProto()
        exporter._embed_onnx_metadata = MagicMock(return_value=onnx_model)
        convert_float_to_float16_mock = MagicMock(return_value=onnx_model)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(float16, "convert_float_to_float16", convert_float_to_float16_mock)
            result = exporter._postprocess_onnx_model(onnx_model, embed_metadata=True, precision=OTXPrecisionType.FP16)
            exporter._embed_onnx_metadata.assert_called_once()
            convert_float_to_float16_mock.assert_called_once_with(onnx_model)
            assert result is onnx_model

    def test_extend_model_metadata_no_intensity(self, exporter):
        """Without intensity_config, metadata should contain only standard preprocessing keys."""
        metadata = exporter._extend_model_metadata({})
        assert ("model_info", "mean_values") in metadata
        assert ("model_info", "scale_values") in metadata
        assert ("model_info", "resize_type") in metadata
        assert ("model_info", "pad_value") in metadata
        assert ("model_info", "reverse_input_channels") in metadata
        # No intensity keys
        assert ("model_info", "input_dtype") not in metadata
        assert ("model_info", "intensity_mode") not in metadata

    def test_extend_model_metadata_with_scale_to_unit_intensity(self):
        """With scale_to_unit IntensityConfig, intensity metadata should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="scale_to_unit",
            max_value=65535.0,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "scale_to_unit"
        assert metadata[("model_info", "intensity_max_value")] == "65535.0"
        assert metadata[("model_info", "intensity_scale_factor")] == "1.0"
        assert metadata[("model_info", "intensity_min_value")] == "0.0"
        assert metadata[("model_info", "intensity_percentile_low")] == "1.0"
        assert metadata[("model_info", "intensity_percentile_high")] == "99.0"
        # No repeat_channels when it's 0
        assert ("model_info", "intensity_repeat_channels") not in metadata
        # window_center and window_width should not be present when None
        assert ("model_info", "intensity_window_center") not in metadata
        assert ("model_info", "intensity_window_width") not in metadata

    def test_extend_model_metadata_with_window_intensity(self):
        """With window IntensityConfig, window parameters should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="window",
            window_center=40.0,
            window_width=400.0,
            repeat_channels=3,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (512, 512),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "window"
        assert metadata[("model_info", "intensity_window_center")] == "40.0"
        assert metadata[("model_info", "intensity_window_width")] == "400.0"
        assert metadata[("model_info", "intensity_repeat_channels")] == "3"

    def test_extend_model_metadata_with_range_scale_intensity(self):
        """With range_scale IntensityConfig (thermal), all range params should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="range_scale",
            scale_factor=0.4,
            min_value=295.15,
            max_value=360.15,
            repeat_channels=3,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (640, 640),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "range_scale"
        assert metadata[("model_info", "intensity_scale_factor")] == "0.4"
        assert metadata[("model_info", "intensity_min_value")] == "295.15"
        assert metadata[("model_info", "intensity_max_value")] == "360.15"
        assert metadata[("model_info", "intensity_repeat_channels")] == "3"

    def test_extend_model_metadata_with_percentile_intensity(self):
        """With percentile IntensityConfig, percentile params should be embedded."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="percentile",
            percentile_low=2.0,
            percentile_high=98.0,
        )
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u16"
        assert metadata[("model_info", "intensity_mode")] == "percentile"
        assert metadata[("model_info", "intensity_percentile_low")] == "2.0"
        assert metadata[("model_info", "intensity_percentile_high")] == "98.0"

    def test_extend_model_metadata_uint8_default_intensity(self):
        """Default IntensityConfig (uint8/scale_to_unit) should map to u8."""
        intensity_cfg = IntensityConfig()  # defaults: storage_dtype="uint8", mode="scale_to_unit"
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})

        assert metadata[("model_info", "input_dtype")] == "u8"
        assert metadata[("model_info", "intensity_mode")] == "scale_to_unit"
        # max_value is None by default, so should NOT be written
        assert ("model_info", "intensity_max_value") not in metadata

    def test_extend_model_metadata_model_metadata_priority(self):
        """Model's original metadata should take priority over exporter's extra metadata."""
        intensity_cfg = IntensityConfig(storage_dtype="uint16", mode="scale_to_unit", max_value=65535.0)
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        # Model metadata overrides
        model_metadata = {("model_info", "intensity_mode"): "custom_mode"}
        metadata = exporter._extend_model_metadata(model_metadata)

        # Model's value should take priority
        assert metadata[("model_info", "intensity_mode")] == "custom_mode"
        # Other intensity keys should still be present
        assert metadata[("model_info", "input_dtype")] == "u16"

    def test_extend_model_metadata_float32_dtype(self):
        """float32 storage_dtype should map to f32."""
        intensity_cfg = IntensityConfig(storage_dtype="float32", mode="scale_to_unit", max_value=1.0)
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        metadata = exporter._extend_model_metadata({})
        assert metadata[("model_info", "input_dtype")] == "f32"

    def test_extend_model_metadata_unsupported_dtype_raises(self):
        """Unsupported storage_dtype must raise ValueError rather than silently embedding a raw value."""
        intensity_cfg = IntensityConfig(storage_dtype="bfloat16", mode="scale_to_unit", max_value=1.0)
        exporter = MockModelExporter(
            task_level_export_parameters=MagicMock(TaskLevelExportParameters),
            data_input_params=DataInputParams(
                (224, 224),
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                intensity_config=intensity_cfg,
            ),
        )
        with pytest.raises(ValueError, match="Unsupported intensity storage_dtype 'bfloat16'"):
            exporter._extend_model_metadata({})


class TestDataInputParams:
    def test_as_dict_without_intensity(self):
        """as_dict without intensity_config should return only input_size, mean, std."""
        params = DataInputParams((224, 224), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        result = params.as_dict()
        assert result == {
            "input_size": (224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }
        assert "intensity_config" not in result

    def test_as_dict_with_intensity(self):
        """as_dict with intensity_config should include intensity_config dict."""
        intensity_cfg = IntensityConfig(
            storage_dtype="uint16",
            mode="window",
            window_center=40.0,
            window_width=400.0,
            repeat_channels=3,
        )
        params = DataInputParams(
            (512, 512),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            intensity_config=intensity_cfg,
        )
        result = params.as_dict()
        assert "intensity_config" in result
        ic = result["intensity_config"]
        assert ic["storage_dtype"] == "uint16"
        assert ic["mode"] == "window"
        assert ic["window_center"] == 40.0
        assert ic["window_width"] == 400.0
        assert ic["repeat_channels"] == 3

    def test_default_intensity_config_is_none(self):
        """Default intensity_config should be None."""
        params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert params.intensity_config is None

    def test_backward_compat_positional_args(self):
        """Existing code using 3 positional args should still work."""
        params = DataInputParams((224, 224), (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        assert params.input_size == (224, 224)
        assert params.mean == (0.0, 0.0, 0.0)
        assert params.std == (1.0, 1.0, 1.0)
        assert params.intensity_config is None
