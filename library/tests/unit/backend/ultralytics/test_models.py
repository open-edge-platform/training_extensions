# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Ultralytics model wrappers."""

from __future__ import annotations

import pytest

from getitune.backend.ultralytics.models import UltralyticsDetectionModel


def test_model_rejects_checkpoint_name_for_scratch_training() -> None:
    with pytest.raises(ValueError, match="pretrained=False requires a model config"):
        UltralyticsDetectionModel(model_name="yolo26n.pt", pretrained=False)


def test_model_allows_yaml_config_for_scratch_training() -> None:
    model = UltralyticsDetectionModel(model_name="yolo26n.yaml", pretrained=False)
    assert model.model_name == "yolo26n.yaml"
    assert model.pretrained is False
