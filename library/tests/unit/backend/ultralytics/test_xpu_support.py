# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Phase 3 — XPU device support."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from getitune.backend.ultralytics.engine import UltralyticsEngine
from getitune.types.device import DeviceType

# _resolve_device() tests


class TestResolveDevice:
    """Tests for ``UltralyticsEngine._resolve_device``."""

    def test_auto_prefers_xpu_over_cuda(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=True),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device("auto")
            assert result == torch.device("xpu:0")

    def test_auto_falls_back_to_cuda(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=False),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device("auto")
            assert result == torch.device("cuda:0")

    def test_auto_falls_back_to_cpu(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=False),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=False),
        ):
            result = UltralyticsEngine._resolve_device("auto")
            assert result == torch.device("cpu")

    def test_explicit_xpu_string(self) -> None:
        result = UltralyticsEngine._resolve_device("xpu")
        assert result == torch.device("xpu:0")

    def test_explicit_xpu_with_index(self) -> None:
        result = UltralyticsEngine._resolve_device("xpu:0")
        assert result == torch.device("xpu:0")

    def test_explicit_cuda_string(self) -> None:
        result = UltralyticsEngine._resolve_device("cuda")
        assert result == torch.device("cuda:0")

    def test_bare_integer_maps_to_cuda(self) -> None:
        result = UltralyticsEngine._resolve_device("0")
        assert result == torch.device("cuda:0")

    def test_bare_integer_index_1(self) -> None:
        result = UltralyticsEngine._resolve_device("1")
        assert result == torch.device("cuda:1")

    def test_cpu_string(self) -> None:
        result = UltralyticsEngine._resolve_device("cpu")
        assert result == torch.device("cpu")

    def test_cuda_with_index(self) -> None:
        result = UltralyticsEngine._resolve_device("cuda:1")
        assert result == torch.device("cuda:1")

    # --- DeviceType enum inputs ---

    def test_device_type_xpu(self) -> None:
        result = UltralyticsEngine._resolve_device(DeviceType.xpu)
        assert result == torch.device("xpu:0")

    def test_device_type_gpu(self) -> None:
        result = UltralyticsEngine._resolve_device(DeviceType.gpu)
        assert result == torch.device("cuda:0")

    def test_device_type_cpu(self) -> None:
        result = UltralyticsEngine._resolve_device(DeviceType.cpu)
        assert result == torch.device("cpu")

    def test_device_type_auto_xpu(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=True),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device(DeviceType.auto)
            assert result == torch.device("xpu:0")

    def test_device_type_auto_cuda(self) -> None:
        with (
            patch("getitune.backend.ultralytics.engine.is_xpu_available", return_value=False),
            patch("getitune.backend.ultralytics.engine.torch.cuda.is_available", return_value=True),
        ):
            result = UltralyticsEngine._resolve_device(DeviceType.auto)
            assert result == torch.device("cuda:0")

    def test_returns_torch_device(self) -> None:
        """All return values must be ``torch.device`` instances."""
        result = UltralyticsEngine._resolve_device("cpu")
        assert isinstance(result, torch.device)

    def test_gpu_string_maps_to_cuda(self) -> None:
        result = UltralyticsEngine._resolve_device("gpu")
        assert result == torch.device("cuda:0")


class TestXPUAwareTrainerMixin:
    """Tests for XPU-specific overrides in the trainer mixin."""

    def test_setup_train_replaces_grad_scaler_on_xpu(self) -> None:
        """After _setup_train, GradScaler must be disabled on XPU."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)

            def _setup_train(self) -> None:
                pass  # parent does nothing in test

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._setup_train()

        assert not trainer.scaler.is_enabled()

    def test_setup_train_keeps_grad_scaler_on_cuda(self) -> None:
        """On CUDA, the mixin must not touch the GradScaler."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)

            def _setup_train(self) -> None:
                pass

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._setup_train()

        # Scaler should remain as parent set it (enabled=True)
        assert trainer.scaler.is_enabled()

    def test_get_memory_xpu_returns_float(self) -> None:
        """_get_memory() should return a float on XPU."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")

            def _get_memory(self, fraction: bool = False) -> float:
                return 0.0

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        with patch("torch.xpu.memory_reserved", return_value=1024 * 1024 * 1024):
            result = trainer._get_memory()
            assert isinstance(result, float)
            assert pytest.approx(result, abs=0.01) == 1.0  # 1 GiB

    def test_get_memory_fraction_xpu(self) -> None:
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")

            def _get_memory(self, fraction: bool = False) -> float:
                return 0.0

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        props = MagicMock()
        props.total_memory = 16 * 2**30  # 16 GiB

        with (
            patch("torch.xpu.memory_reserved", return_value=4 * 2**30),
            patch("torch.xpu.get_device_properties", return_value=props),
        ):
            result = trainer._get_memory(fraction=True)
            assert pytest.approx(result, abs=0.01) == 0.25

    def test_clear_memory_xpu_calls_empty_cache(self) -> None:
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")

            def _clear_memory(self, threshold: float | None = None) -> None:
                pass

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        with patch("torch.xpu.empty_cache") as mock_clear:
            trainer._clear_memory()
            mock_clear.assert_called_once()

    def test_clear_memory_cuda_delegates_to_super(self) -> None:
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.super_called = False

            def _clear_memory(self, threshold: float | None = None) -> None:
                self.super_called = True

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer._clear_memory()
        assert trainer.super_called

    def test_move_batch_to_device_uses_non_blocking_on_xpu(self) -> None:
        """The helper should use non_blocking=True on XPU."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()

        tensor = MagicMock(spec=torch.Tensor)
        tensor.to.return_value = tensor
        batch = {"img": tensor, "non_tensor": "value"}

        with patch(
            "getitune.backend.ultralytics.plugins.xpu_mixin.isinstance",
            side_effect=lambda o, t: True if t is torch.Tensor and o is tensor else isinstance(o, t),
        ):
            result = trainer._move_batch_to_device(batch)

        tensor.to.assert_called_once_with(torch.device("xpu:0"), non_blocking=True)
        assert result["non_tensor"] == "value"

    def test_train_wraps_xpu_autocast(self) -> None:
        """On XPU, train() should wrap the call in xpu autocast."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        autocast_entered = False

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("xpu:0")
                self.args = MagicMock()
                self.args.amp = True

            def train(self) -> None:
                nonlocal autocast_entered
                # Check if XPU autocast is active
                autocast_entered = torch.is_autocast_enabled("xpu")

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer.train()
        assert autocast_entered

    def test_train_no_autocast_on_cuda(self) -> None:
        """On CUDA, train() should NOT add extra autocast wrapping."""
        from getitune.backend.ultralytics.plugins.xpu_mixin import XPUAwareTrainerMixin

        class FakeBaseTrainer:
            def __init__(self):
                self.device = torch.device("cuda:0")
                self.args = MagicMock()
                self.args.amp = True
                self.train_called = False

            def train(self) -> None:
                self.train_called = True

        class TestTrainer(XPUAwareTrainerMixin, FakeBaseTrainer):
            pass

        trainer = TestTrainer()
        trainer.train()
        assert trainer.train_called
