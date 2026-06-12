# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for Ultralytics trainer customizations."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from getitune.backend.ultralytics.trainers.detection import DetectionTrainer
from getitune.backend.ultralytics.trainers.instance_segmentation import SegmentationTrainer


def test_detection_trainer_fallback_uses_upstream_preprocess() -> None:
    """Non-DataModule detection path must keep Ultralytics preprocessing."""

    trainer = object.__new__(DetectionTrainer)
    trainer._datamodule = None
    trainer._use_getitune_data = False
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.0)

    batch = {"img": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)}

    result = trainer.preprocess_batch(batch)

    assert torch.allclose(result["img"], torch.ones((1, 3, 4, 4), dtype=torch.float32))


def test_segmentation_trainer_fallback_uses_upstream_preprocess() -> None:
    """Non-DataModule segmentation path must keep Ultralytics preprocessing."""

    trainer = object.__new__(SegmentationTrainer)
    trainer._datamodule = None
    trainer._use_getitune_data = False
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.0)

    batch = {"img": torch.full((1, 3, 4, 4), 255, dtype=torch.uint8)}

    result = trainer.preprocess_batch(batch)

    assert torch.allclose(result["img"], torch.ones((1, 3, 4, 4), dtype=torch.float32))


def test_detection_trainer_datamodule_path_skips_divide_by_255() -> None:
    """DataModule detection batches must preserve float [0, 1] inputs."""

    trainer = object.__new__(DetectionTrainer)
    trainer._datamodule = MagicMock()
    trainer._use_getitune_data = True
    trainer.device = torch.device("cpu")

    imgs = torch.rand(2, 3, 8, 8, dtype=torch.float32)
    batch = {"img": imgs.clone()}

    result = trainer.preprocess_batch(batch)

    assert torch.allclose(result["img"], imgs)


def test_segmentation_trainer_datamodule_path_skips_divide_by_255() -> None:
    """DataModule segmentation batches must preserve float [0, 1] inputs."""

    trainer = object.__new__(SegmentationTrainer)
    trainer._datamodule = MagicMock()
    trainer._use_getitune_data = True
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


class TestClearMemory:
    """Tests for the optimized _clear_memory override in GetiTuneBaseTrainer."""

    def _make_trainer(self, *, use_bridge: bool, device_type: str = "cuda") -> DetectionTrainer:
        trainer = object.__new__(DetectionTrainer)
        trainer._datamodule = MagicMock() if use_bridge else None
        trainer._use_getitune_data = use_bridge
        trainer.device = torch.device(device_type if device_type != "cuda" else "cuda:0")
        return trainer

    def test_clear_memory_skips_gc_collect_for_bridge(self) -> None:
        """Bridge path must NOT call gc.collect() (expensive with spawn workers)."""
        trainer = self._make_trainer(use_bridge=True)

        with (
            patch("gc.collect") as mock_gc,
            patch("torch.cuda.empty_cache") as mock_cache,
            patch.object(type(trainer), "_get_memory", return_value=0.8),
        ):
            trainer._clear_memory()

        mock_gc.assert_not_called()
        mock_cache.assert_called_once()

    def test_clear_memory_skips_when_below_threshold(self) -> None:
        """When GPU usage is below threshold, nothing should happen."""
        trainer = self._make_trainer(use_bridge=True)

        with (
            patch("torch.cuda.empty_cache") as mock_cache,
            patch.object(type(trainer), "_get_memory", return_value=0.3),
        ):
            trainer._clear_memory(threshold=0.5)

        mock_cache.assert_not_called()

    def test_clear_memory_clears_when_above_threshold(self) -> None:
        """When GPU usage exceeds threshold, CUDA cache should be cleared."""
        trainer = self._make_trainer(use_bridge=True)

        with (
            patch("torch.cuda.empty_cache") as mock_cache,
            patch.object(type(trainer), "_get_memory", return_value=0.7),
        ):
            trainer._clear_memory(threshold=0.5)

        mock_cache.assert_called_once()

    def test_clear_memory_noop_on_cpu(self) -> None:
        """CPU device should be a no-op."""
        trainer = self._make_trainer(use_bridge=True, device_type="cpu")

        with patch("torch.cuda.empty_cache") as mock_cache:
            trainer._clear_memory()

        mock_cache.assert_not_called()

    def test_clear_memory_fallback_without_bridge(self) -> None:
        """Non-bridge path must delegate to the upstream implementation."""
        trainer = self._make_trainer(use_bridge=False)

        with patch.object(
            DetectionTrainer.__mro__[3],  # BaseTrainer in MRO
            "_clear_memory",
            create=True,
        ) as mock_super:
            trainer._clear_memory(threshold=0.5)

        mock_super.assert_called_once_with(0.5)


class TestProgressCallback:
    """Tests for progress reporting in the Ultralytics bridge."""

    def test_progress_callback_emits_progress(self) -> None:
        """Progress callback should emit linearly interpolated values."""
        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._datamodule = MagicMock()

        progress_values: list[float] = []
        trainer._progress_fn = lambda p: progress_values.append(p)
        trainer._progress_min = 10.0
        trainer._progress_max = 80.0

        # Simulate train_loader with 3 batches and 10 epochs
        trainer.train_loader = list(range(3))
        trainer.epochs = 10

        registered: list = []
        trainer.add_callback = lambda event, fn: registered.append((event, fn))

        trainer._register_progress_callback()

        assert len(registered) == 1
        event, callback = registered[0]
        assert event == "on_train_batch_end"

        # Simulate 5 batch-end calls (total_steps = 30)
        for _ in range(5):
            callback(trainer)

        assert len(progress_values) == 5
        # step 5/30 = 1/6, progress = 10 + (1/6) * 70 ≈ 21.67
        assert pytest.approx(progress_values[4], abs=0.01) == 10.0 + (5 / 30) * 70.0

    def test_progress_callback_noop_when_no_fn(self) -> None:
        """No callback should be registered when _progress_fn is None."""
        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._progress_fn = None

        registered: list = []
        trainer.add_callback = lambda event, fn: registered.append((event, fn))

        trainer._register_progress_callback()

        assert len(registered) == 0
