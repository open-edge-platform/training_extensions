# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import MagicMock, patch

import pytest

from app.services.system_service import SystemService


class TestSystemService:
    """Test cases for SystemService"""

    @pytest.fixture
    def fxt_system_service(self) -> SystemService:
        return SystemService()

    def test_get_memory_usage(self, fxt_system_service: SystemService):
        """Test getting memory usage"""
        used, total = fxt_system_service.get_memory_usage()

        assert used > 0
        assert total > 0
        assert used <= total

    def test_get_cpu_usage(self, fxt_system_service: SystemService):
        """Test getting CPU usage"""
        cpu_usage = fxt_system_service.get_cpu_usage()

        assert cpu_usage >= 0.0

    def test_get_devices_cpu_only(self, fxt_system_service: SystemService):
        """Test getting devices when only CPU is available"""
        mock_torch = MagicMock()
        # Simulate torch not being available
        mock_torch.xpu.is_available.side_effect = AttributeError()
        mock_torch.cuda.is_available.side_effect = AttributeError()

        with patch.dict(sys.modules, {"torch": mock_torch}):
            devices = fxt_system_service.get_devices()

            assert len(devices) == 1
            assert devices[0].type == "cpu"
            assert devices[0].name == "CPU"
            assert devices[0].memory is None
            assert devices[0].index is None

    def test_get_devices_with_xpu(self, fxt_system_service: SystemService):
        """Test getting devices when Intel XPU is available"""
        mock_torch = MagicMock()
        # Mock XPU device
        mock_dp = MagicMock()
        mock_dp.name = "Intel(R) Graphics [0x7d41]"
        mock_dp.total_memory = 36022263808

        mock_torch.xpu.is_available.return_value = True
        mock_torch.xpu.device_count.return_value = 1
        mock_torch.xpu.get_device_properties.return_value = mock_dp

        # CUDA not available
        mock_torch.cuda.is_available.return_value = False

        with patch.dict(sys.modules, {"torch": mock_torch}):
            devices = fxt_system_service.get_devices()

            assert len(devices) == 2
            assert devices[0].type == "cpu"
            assert devices[1].type == "xpu"
            assert devices[1].name == "Intel(R) Graphics [0x7d41]"
            assert devices[1].memory == 36022263808
            assert devices[1].index == 0

    def test_get_devices_with_cuda(self, fxt_system_service: SystemService):
        """Test getting devices when NVIDIA CUDA is available"""
        mock_torch = MagicMock()
        # XPU not available
        mock_torch.xpu.is_available.return_value = False

        # Mock CUDA device
        mock_dp = MagicMock()
        mock_dp.name = "NVIDIA GeForce RTX 4090"
        mock_dp.total_memory = 25769803776

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_dp

        with patch.dict(sys.modules, {"torch": mock_torch}):
            devices = fxt_system_service.get_devices()

            assert len(devices) == 2
            assert devices[0].type == "cpu"
            assert devices[1].type == "cuda"
            assert devices[1].name == "NVIDIA GeForce RTX 4090"
            assert devices[1].memory == 25769803776
            assert devices[1].index == 0

    def test_get_devices_with_multiple_devices(self, fxt_system_service: SystemService):
        """Test getting devices when multiple GPUs are available"""
        mock_torch = MagicMock()
        # Mock XPU device
        mock_xpu_dp = MagicMock()
        mock_xpu_dp.name = "Intel(R) Graphics [0x7d41]"
        mock_xpu_dp.total_memory = 36022263808

        mock_torch.xpu.is_available.return_value = True
        mock_torch.xpu.device_count.return_value = 1
        mock_torch.xpu.get_device_properties.return_value = mock_xpu_dp

        # Mock CUDA device
        mock_cuda_dp = MagicMock()
        mock_cuda_dp.name = "NVIDIA GeForce RTX 4090"
        mock_cuda_dp.total_memory = 25769803776

        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = mock_cuda_dp

        with patch.dict(sys.modules, {"torch": mock_torch}):
            devices = fxt_system_service.get_devices()

            assert len(devices) == 3
            assert devices[0].type == "cpu"
            assert devices[1].type == "xpu"
            assert devices[2].type == "cuda"

    def test_get_devices_torch_not_available(self, fxt_system_service: SystemService):
        """Test getting devices when torch is not available"""
        # Remove torch from sys.modules to simulate ImportError
        original_torch = sys.modules.get("torch")
        if "torch" in sys.modules:
            del sys.modules["torch"]

        try:
            devices = fxt_system_service.get_devices()

            assert len(devices) == 1
            assert devices[0].type == "cpu"
        finally:
            # Restore torch if it was present
            if original_torch is not None:
                sys.modules["torch"] = original_torch
