# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform
import re

import cv2
import psutil
import torch
from cv2_enumerate_cameras import enumerate_cameras

from app.models.system import CameraInfo, DeviceInfo, DeviceType

DEVICE_PATTERN = re.compile(r"^(cpu|xpu|cuda)(-(\d+))?$")
DEFAULT_DEVICE = "cpu"
CV2_BACKENDS = {
    "Windows": cv2.CAP_MSMF,
    "Linux": cv2.CAP_V4L2,
    "Darwin": cv2.CAP_AVFOUNDATION,
}


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

    def get_inference_devices(self) -> list[DeviceInfo]:
        """
        Get available compute devices for inference (CPU, XPU, ...)

        Returns:
            list[DeviceInfo]: List of available devices
        """
        return [device for device in self.get_devices() if device.type != DeviceType.CUDA]

    def get_training_devices(self) -> list[DeviceInfo]:
        """
        Get available compute devices for training (CPUs, XPUs, GPUs, ...)

        Returns:
            list[DeviceInfo]: List of available training devices
        """
        return self.get_devices()  # currently same as get_devices, can be customized later with filters

    def validate_device(self, device_str: str) -> bool:
        """
        Validate if a device string is available on the system.

        Args:
            device_str: Device string in format '<target>[-<index>]' (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            bool: True if the device is available, False otherwise
        """
        device_type, device_index = self._parse_device(device_str)

        # CPU is always available
        if device_type == DeviceType.CPU:
            return True

        # Check if desired device is among available devices
        available_devices = self.get_devices()
        for available_device in available_devices:
            if device_type == available_device.type and device_index == (available_device.index or 0):
                return True

        return False

    def get_device_info(self, device_str: str) -> DeviceInfo:
        """
        Get DeviceInfo for a given device string.

        Args:
            device_str: Device string in format '<target>[-<index>]' (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            DeviceInfo: Information about the specified device
        """
        if not self.validate_device(device_str):
            raise ValueError(f"Device '{device_str}' is not available on the system.")

        device_type, device_index = self._parse_device(device_str)
        if device_type == DeviceType.CPU:
            return DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
        return next(
            device for device in self.get_devices() if device.type == device_type and device.index == device_index
        )

    @staticmethod
    def _parse_device(device_str: str) -> tuple[DeviceType, int]:
        """
        Parse device string into type and index

        Args:
            device_str: Device string in format '<target>[-<index>]' (e.g., 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            tuple[str, int]: Device type and index
        """
        m = DEVICE_PATTERN.match(device_str.lower())
        if not m:
            raise ValueError(f"Invalid device string: {device_str}")

        device_type, _, device_index = m.groups()
        device_index = int(device_index) if device_index is not None else 0
        return DeviceType(device_type.lower()), device_index

    @staticmethod
    def get_camera_devices() -> list[CameraInfo]:
        """
        Get available camera devices.
        Camera names are formatted as "<camera_name> [<index>]".

        Returns:
            list[CameraInfo]: List of available camera devices
        """
        if (backend := CV2_BACKENDS.get(platform.system())) is None:
            raise RuntimeError(f"Unsupported platform: {platform.system()}")

        return [CameraInfo(index=cam.index, name=f"{cam.name} [{cam.index}]") for cam in enumerate_cameras(backend)]
