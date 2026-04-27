# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ultralytics model wrappers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getitune.backend.ultralytics.models import UltralyticsDetectionModel
from getitune.backend.ultralytics.models.base import UltralyticsModel


def test_model_rejects_checkpoint_name_for_scratch_training() -> None:
    with pytest.raises(ValueError, match="pretrained=False requires a model config"):
        UltralyticsDetectionModel(model_name="yolo26n.pt", pretrained=False)


def test_model_allows_yaml_config_for_scratch_training() -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
    assert model.model_name == "yolo26n.yaml"
    assert model.pretrained is False


def test_default_model_name_is_yaml() -> None:
    """Default model name should be a .yaml config, not a .pt checkpoint."""
    assert UltralyticsDetectionModel.default_model_name.endswith(".yaml")


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
