# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for visual prompting model entity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from model_api.models import SAMVisualPrompter
from torchvision import tv_tensors

from otx.core.data.entity.visual_prompting import (
    VisualPromptingBatchPredEntity,
)
from otx.core.exporter.visual_prompting import OTXVisualPromptingModelExporter
from otx.core.model.visual_prompting import (
    OTXVisualPromptingModel,
    OVVisualPromptingModel,
    _inference_step,
)
from otx.core.types.export import TaskLevelExportParameters


@pytest.fixture()
def otx_visual_prompting_model(mocker) -> OTXVisualPromptingModel:
    mocker.patch.object(OTXVisualPromptingModel, "_create_model")
    model = OTXVisualPromptingModel(label_info=1, input_size=(1024, 1024))
    model.model.image_size = 1024
    return model


def test_inference_step(mocker, otx_visual_prompting_model, fxt_vpm_data_entity) -> None:
    """Test _inference_step."""
    otx_visual_prompting_model.configure_metric()
    mocker.patch.object(otx_visual_prompting_model, "forward", return_value=fxt_vpm_data_entity[2])
    mocker_updates = {}
    for k, v in otx_visual_prompting_model.metric.items():
        mocker_updates[k] = mocker.patch.object(v, "update")

    _inference_step(otx_visual_prompting_model, otx_visual_prompting_model.metric, fxt_vpm_data_entity[1])

    for v in mocker_updates.values():
        v.assert_called()


class TestOTXVisualPromptingModel:
    def test_exporter(self, otx_visual_prompting_model) -> None:
        """Test _exporter."""
        exporter = otx_visual_prompting_model._exporter
        assert isinstance(exporter, OTXVisualPromptingModelExporter)
        assert exporter.input_size == (1, 3, 1024, 1024)
        assert exporter.resize_mode == "fit_to_window"
        assert exporter.mean == (123.675, 116.28, 103.53)
        assert exporter.std == (58.395, 57.12, 57.375)

    def test_export_parameters(self, otx_visual_prompting_model) -> None:
        """Test _export_parameters."""
        export_parameters = otx_visual_prompting_model._export_parameters

        assert isinstance(export_parameters, TaskLevelExportParameters)
        assert export_parameters.model_type == "Visual_Prompting"
        assert export_parameters.task_type == "visual_prompting"

    def test_optimization_config(self, otx_visual_prompting_model) -> None:
        """Test _optimization_config."""
        optimization_config = otx_visual_prompting_model._optimization_config

        assert optimization_config == {
            "model_type": "transformer",
            "advanced_parameters": {
                "activations_range_estimator_params": {
                    "min": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MIN",
                        "quantile_outlier_prob": "1e-4",
                    },
                    "max": {
                        "statistics_type": "QUANTILE",
                        "aggregator_type": "MAX",
                        "quantile_outlier_prob": "1e-4",
                    },
                },
            },
        }

    def test_dummy_input(self, otx_visual_prompting_model):
        batch_size = 2
        batch = otx_visual_prompting_model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size


class TestOVVisualPromptingModel:
    @pytest.fixture()
    def set_ov_visual_prompting_model(self, mocker, tmpdir):
        def ov_visual_prompting_model(for_create_model: bool = False) -> OVVisualPromptingModel:
            if for_create_model:
                mocker.patch("model_api.adapters.create_core")
                mocker.patch("model_api.adapters.get_user_config")
                mocker.patch("model_api.adapters.OpenvinoAdapter")
                mocker.patch("model_api.models.Model.create_model")
            else:
                mocker.patch.object(
                    OVVisualPromptingModel,
                    "_create_model",
                    return_value=SAMVisualPrompter(Mock(), Mock()),
                )
            dirpath = Path(tmpdir)
            (dirpath / "exported_model_image_encoder.xml").touch()
            (dirpath / "exported_model_decoder.xml").touch()
            model_name = str(dirpath / "exported_model_decoder.xml")
            return OVVisualPromptingModel(num_classes=0, model_name=model_name)

        return ov_visual_prompting_model

    def test_create_model(self, set_ov_visual_prompting_model) -> None:
        """Test _create_model."""
        ov_visual_prompting_model = set_ov_visual_prompting_model(for_create_model=True)
        ov_models = ov_visual_prompting_model._create_model()

        assert isinstance(ov_models, SAMVisualPrompter)

    def test_forward(self, mocker, set_ov_visual_prompting_model, fxt_vpm_data_entity) -> None:
        """Test forward."""
        ov_visual_prompting_model = set_ov_visual_prompting_model()
        mocker.patch.object(
            ov_visual_prompting_model.model.encoder,
            "preprocess",
            return_value=(np.zeros((1, 3, 1024, 1024)), {"original_shape": (1024, 1024, 3)}),
        )
        mocker.patch.object(
            ov_visual_prompting_model.model.encoder,
            "infer_sync",
            return_value={"image_embeddings": np.random.random((1, 256, 64, 64))},
        )
        mocker.patch.object(
            ov_visual_prompting_model.model.decoder,
            "preprocess",
            return_value=[
                {
                    "point_coords": np.array([1, 1]).reshape(-1, 1, 2),
                    "point_labels": np.array([1], dtype=np.float32).reshape(-1, 1),
                    "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
                    "has_mask_input": np.zeros((1, 1), dtype=np.float32),
                    "orig_size": np.array([1024, 1024], dtype=np.int64).reshape(-1, 2),
                    "label": 1,
                },
            ],
        )
        mocker.patch.object(
            ov_visual_prompting_model.model.decoder,
            "infer_sync",
            return_value={
                "iou_predictions": 0.0,
                "upscaled_masks": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
            },
        )
        mocker.patch.object(
            ov_visual_prompting_model.model.decoder,
            "postprocess",
            return_value={
                "low_res_masks": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "upscaled_masks": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "hard_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "soft_prediction": np.zeros((1, 1, 1024, 1024), dtype=np.float32),
                "scores": np.zeros((1, 1), dtype=np.float32),
                "iou_predictions": np.zeros((1, 1), dtype=np.float32),
                "labels": np.zeros((1, 1), dtype=np.float32),
            },
        )

        results = ov_visual_prompting_model(fxt_vpm_data_entity[1])

        assert isinstance(results, VisualPromptingBatchPredEntity)
        assert isinstance(results.images, list)
        assert isinstance(results.images[0], tv_tensors.Image)
        assert isinstance(results.masks, list)
        assert isinstance(results.masks[0], tv_tensors.Mask)

    def test_optimize(self, tmpdir, mocker, set_ov_visual_prompting_model) -> None:
        """Test optimize."""
        mocker.patch("openvino.Core.read_model")
        mocker.patch("openvino.save_model")
        mocker.patch("nncf.quantize")

        ov_visual_prompting_model = set_ov_visual_prompting_model()
        fake_data_module = Mock()

        results = ov_visual_prompting_model.optimize(tmpdir, fake_data_module)

        assert "image_encoder" in results
        assert "decoder" in results

    def test_dummy_input(self, set_ov_visual_prompting_model):
        batch_size = 2
        ov_visual_prompting_model = set_ov_visual_prompting_model()
        batch = ov_visual_prompting_model.get_dummy_input(batch_size)
        assert batch.batch_size == batch_size
