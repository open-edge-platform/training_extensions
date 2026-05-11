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
    """Tests for the optimized _clear_memory override in GetiTuneDataBridgeMixin."""

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


class TestPeriodicValidation:
    """Tests for the periodic validation/save callback."""

    def test_periodic_val_callback_sets_val_and_save(self) -> None:
        """Callback should set args.val and args.save based on val_interval."""
        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._datamodule = MagicMock()
        trainer._val_interval = 5
        trainer.args = SimpleNamespace(val=True, save=True)
        trainer.callbacks = {}

        registered: list = []
        trainer.add_callback = lambda event, fn: registered.append((event, fn))

        trainer._register_periodic_val_callback()

        assert len(registered) == 1
        event, callback = registered[0]
        assert event == "on_train_epoch_end"

        # Epoch 0 (1st): should NOT validate (1 % 5 != 0)
        trainer.epoch = 0
        callback(trainer)
        assert trainer.args.val is False
        assert trainer.args.save is False

        # Epoch 4 (5th): SHOULD validate (5 % 5 == 0)
        trainer.epoch = 4
        callback(trainer)
        assert trainer.args.val is True
        assert trainer.args.save is True

        # Epoch 5 (6th): should NOT validate (6 % 5 != 0)
        trainer.epoch = 5
        callback(trainer)
        assert trainer.args.val is False
        assert trainer.args.save is False

        # Epoch 9 (10th): SHOULD validate (10 % 5 == 0)
        trainer.epoch = 9
        callback(trainer)
        assert trainer.args.val is True
        assert trainer.args.save is True

    @pytest.mark.parametrize("val_interval", [1, 3, 10])
    def test_periodic_val_respects_custom_interval(self, val_interval: int) -> None:
        """Validation should fire at multiples of val_interval."""
        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._datamodule = MagicMock()
        trainer._val_interval = val_interval
        trainer.args = SimpleNamespace(val=True, save=True)

        registered: list = []
        trainer.add_callback = lambda event, fn: registered.append((event, fn))

        trainer._register_periodic_val_callback()
        _, callback = registered[0]

        val_epochs = []
        for epoch in range(20):
            trainer.epoch = epoch
            callback(trainer)
            if trainer.args.val:
                val_epochs.append(epoch)

        expected = [e for e in range(20) if (e + 1) % val_interval == 0]
        assert val_epochs == expected


class TestCloseMosaic:
    """Tests for the bridge close_mosaic feature."""

    def test_close_mosaic_callback_disables_cached_mosaic(self) -> None:
        """CachedMosaic.prob should be set to 0 at the close_mosaic epoch."""
        from getitune.data.augmentation.pipeline import CPUAugmentationPipeline
        from getitune.data.augmentation.transforms import CachedMosaic

        mosaic = CachedMosaic(img_scale=(640, 640), p=1.0, max_cached_images=4)
        pipeline = CPUAugmentationPipeline([mosaic])

        mock_dataset = MagicMock()
        mock_dataset.transforms = pipeline

        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._datamodule = MagicMock()
        trainer._datamodule.subsets = {"train": mock_dataset}
        trainer._bridge_close_mosaic = 10
        trainer.epochs = 100
        trainer.args = SimpleNamespace(workers=0)

        registered: list = []
        trainer.add_callback = lambda event, fn: registered.append((event, fn))

        trainer._register_close_mosaic_callback()

        assert len(registered) == 1
        event, callback = registered[0]
        assert event == "on_train_epoch_start"

        # Before close_mosaic epoch (89): mosaic should still be active
        trainer.epoch = 89
        callback(trainer)
        assert mosaic.prob == 1.0

        # At close_mosaic epoch (90 = 100 - 10): mosaic should be disabled
        trainer.epoch = 90
        callback(trainer)
        assert mosaic.prob == 0.0

    def test_close_mosaic_noop_when_zero(self) -> None:
        """No callback should be registered when close_mosaic = 0."""
        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._bridge_close_mosaic = 0

        registered: list = []
        trainer.add_callback = lambda event, fn: registered.append((event, fn))

        trainer._register_close_mosaic_callback()

        assert len(registered) == 0

    def test_disable_ultralytics_augmentations_preserves_close_mosaic(self) -> None:
        """_disable_ultralytics_augmentations must save close_mosaic before zeroing it."""
        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._datamodule = MagicMock()
        trainer.args = SimpleNamespace(
            mosaic=1.0,
            mixup=0.0,
            cutmix=0.0,
            copy_paste=0.0,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            flipud=0.0,
            fliplr=0.5,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            close_mosaic=10,
        )

        trainer._disable_ultralytics_augmentations()

        # close_mosaic zeroed for upstream, but preserved in _bridge_close_mosaic
        assert trainer.args.close_mosaic == 0
        assert trainer._bridge_close_mosaic == 10

    def test_close_mosaic_fires_only_once(self) -> None:
        """The callback should fire exactly once even if called for later epochs."""
        from getitune.data.augmentation.pipeline import CPUAugmentationPipeline
        from getitune.data.augmentation.transforms import CachedMosaic

        mosaic = CachedMosaic(img_scale=(640, 640), p=1.0, max_cached_images=4)
        pipeline = CPUAugmentationPipeline([mosaic])

        mock_dataset = MagicMock()
        mock_dataset.transforms = pipeline

        trainer = object.__new__(DetectionTrainer)
        trainer._use_getitune_data = True
        trainer._datamodule = MagicMock()
        trainer._datamodule.subsets = {"train": mock_dataset}
        trainer._bridge_close_mosaic = 5
        trainer.epochs = 50
        trainer.args = SimpleNamespace(workers=0)

        registered: list = []
        trainer.add_callback = lambda event, fn: registered.append((event, fn))
        trainer._register_close_mosaic_callback()
        _, callback = registered[0]

        # Simulate calling for several epochs after close_mosaic
        for epoch in range(44, 50):
            trainer.epoch = epoch
            callback(trainer)

        # Mosaic should be disabled (prob=0) and _disable_bridge_mosaic should not crash
        assert mosaic.prob == 0.0
