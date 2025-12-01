# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import psutil
import torch

from app.schemas.system import DeviceInfo, DeviceType


class SystemService:
    """Service to get system information"""

    def __init__(self) -> None:
        self.process = psutil.Process()

    def get_memory_usage(self) -> tuple[float, float]:
        """
        Get the memory usage of the process

        Returns:
            tuple[float, float]: Used memory in MB and total available memory in MB
        """
        memory_info = psutil.virtual_memory()
        return self.process.memory_info().rss / (1024 * 1024), memory_info.total / (1024 * 1024)

    def get_cpu_usage(self) -> float:
        """
        Get the CPU usage of the process

        Returns:
            float: CPU usage in percentage
        """
        return self.process.cpu_percent(interval=None)

    @staticmethod
    def get_devices() -> list[DeviceInfo]:
        """
        Get available compute devices (CPU, GPUs, ...)

        Returns:
            list[DeviceInfo]: List of available devices
        """
        # CPU is always available
        devices: list[DeviceInfo] = [DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)]

        # Check for Intel XPU devices
        if torch.xpu.is_available():
            for device_idx in range(torch.xpu.device_count()):
                xpu_dp = torch.xpu.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.XPU,
                        name=xpu_dp.name,
                        memory=xpu_dp.total_memory,
                        index=device_idx,
                    )
                )

        # Check for NVIDIA CUDA devices
        if torch.cuda.is_available():
            for device_idx in range(torch.cuda.device_count()):
                cuda_dp = torch.cuda.get_device_properties(device_idx)
                devices.append(
                    DeviceInfo(
                        type=DeviceType.CUDA,
                        name=cuda_dp.name,
                        memory=cuda_dp.total_memory,
                        index=device_idx,
                    )
                )

        return devices
