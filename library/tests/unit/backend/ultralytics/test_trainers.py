# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Ultralytics trainer customizations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from getitune.backend.ultralytics.trainers.detection import DetectionTrainer
from getitune.backend.ultralytics.trainers.instance_segmentation import SegmentationTrainer


def test_detection_trainer_fallback_uses_upstream_preprocess() -> None:
    """Non-DataModule detection path must keep Ultralytics preprocessing."""

    trainer = object.__new__(DetectionTrainer)
    trainer._datamodule = None
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.0)

    batch = {"img": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)}

    result = trainer.preprocess_batch(batch)

    assert torch.allclose(result["img"], torch.ones((1, 3, 4, 4), dtype=torch.float32))


def test_segmentation_trainer_fallback_uses_upstream_preprocess() -> None:
    """Non-DataModule segmentation path must keep Ultralytics preprocessing."""

    trainer = object.__new__(SegmentationTrainer)
    trainer._datamodule = None
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.0)

    batch = {"img": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)}

    result = trainer.preprocess_batch(batch)

    assert torch.allclose(result["img"], torch.ones((1, 3, 4, 4), dtype=torch.float32))


def test_detection_trainer_datamodule_path_skips_divide_by_255() -> None:
    """DataModule detection batches must preserve float [0, 1] inputs."""

    trainer = object.__new__(DetectionTrainer)
    trainer._datamodule = MagicMock()
    trainer.device = torch.device("cpu")

    imgs = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    batch = {"img": imgs.clone()}

    result = trainer.preprocess_batch(batch)

    assert torch.allclose(result["img"], imgs)


def test_segmentation_trainer_datamodule_path_skips_divide_by_255() -> None:
    """DataModule segmentation batches must preserve float [0, 1] inputs."""

    trainer = object.__new__(SegmentationTrainer)
    trainer._datamodule = MagicMock()
    trainer.device = torch.device("cpu")

    imgs = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    batch = {"img": imgs.clone()}

    result = trainer.preprocess_batch(batch)

    assert torch.allclose(result["img"], imgs)


def test_move_batch_to_device_uses_non_blocking_for_xpu() -> None:
    """The DataModule helper should use non_blocking transfers on XPU."""

    trainer = object.__new__(DetectionTrainer)
    trainer.device = torch.device("xpu:0")

    tensor = MagicMock(spec=torch.Tensor)
    tensor.to.return_value = tensor
    batch = {"img": tensor, "meta": "value"}

    with patch(
        "getitune.backend.ultralytics.plugins.xpu_mixin.isinstance",
        side_effect=lambda obj, typ: obj is tensor if typ is torch.Tensor else isinstance(obj, typ),
    ):
        result = trainer._move_batch_to_device(batch)

    tensor.to.assert_called_once_with(torch.device("xpu:0"), non_blocking=True)
    assert result["meta"] == "value"
