# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import platform
import re

import cv2
import openvino as ov
import psutil
import torch
from cv2_enumerate_cameras import enumerate_cameras
from loguru import logger

from app.models.system import CameraInfo, DeviceInfo, DeviceType

DEVICE_PATTERN = re.compile(r"^(auto|cpu|xpu|cuda)(-(\d+))?$")
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

    @staticmethod
    def get_inference_devices() -> list[DeviceInfo]:
        """
        Get available compute devices for inference (CPU, XPU, ...).

        Unlike training (which relies on PyTorch), inference is performed via OpenVINO, so devices are listed using
        OpenVINO's `core.available_devices` API.

        OpenVINO returns device names such as 'CPU', 'GPU', 'GPU.0', 'GPU.1', ... Per the OpenVINO documentation,
        when an integrated GPU is present it always takes id 0, and 'GPU' is an alias for 'GPU.0'.

        Returns:
            list[DeviceInfo]: List of available inference devices.
        """
        try:
            core = ov.Core()
            available_devices: list[str] = list(core.available_devices)
        except Exception:
            logger.exception("Failed to query OpenVINO inference devices; falling back to CPU only.")
            return [DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)]

        devices: list[DeviceInfo] = []
        seen_gpu_indices: set[int] = set()
        for ov_device in available_devices:
            try:
                if ov_device == "CPU":
                    devices.append(DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None))
                elif ov_device == "GPU" or ov_device.startswith("GPU."):
                    # 'GPU' aliases 'GPU.0'; deduplicate if both are reported.
                    index = 0 if ov_device == "GPU" else int(ov_device.split(".", 1)[1])
                    if index in seen_gpu_indices:
                        continue
                    seen_gpu_indices.add(index)
                    name = core.get_property(ov_device, "FULL_DEVICE_NAME")
                    try:
                        memory = int(core.get_property(ov_device, "GPU_DEVICE_TOTAL_MEM_SIZE"))
                    except Exception:
                        memory = None
                    devices.append(DeviceInfo(type=DeviceType.XPU, name=str(name), memory=memory, index=index))
                else:
                    # Other OpenVINO devices (e.g., NPU) are not currently mapped to DeviceType.
                    logger.debug("Skipping unsupported OpenVINO inference device: {}", ov_device)
            except Exception:
                logger.exception("Failed to query properties for OpenVINO device '{}'", ov_device)

        # Ensure CPU is always present.
        if not any(d.type == DeviceType.CPU for d in devices):
            devices.insert(0, DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None))

        return devices

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
            device_str: Device string in format '<target>[-<index>]'
                (e.g., 'auto', 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            bool: True if the device is available, False otherwise
        """
        try:
            device_type, device_index = self._parse_device(device_str)
        except ValueError:
            logger.debug("Cannot parse invalid device string: {}", device_str)
            return False

        # CPU is always available
        if device_type in [DeviceType.AUTO, DeviceType.CPU]:
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
            device_str: Device string in format '<target>[-<index>]'
                (e.g., 'auto', 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            DeviceInfo: Information about the specified device
        """
        if not self.validate_device(device_str):
            raise ValueError(f"Device '{device_str}' is not available on the system.")

        device_type, device_index = self._parse_device(device_str)
        if device_type == DeviceType.CPU:
            return DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)
        if device_type == DeviceType.AUTO:
            return DeviceInfo(type=DeviceType.AUTO, name="AUTO", memory=None, index=None)
        return next(
            device for device in self.get_devices() if device.type == device_type and device.index == device_index
        )

    def get_inference_device_info(self, device_str: str) -> DeviceInfo:
        """
        Get DeviceInfo for a given device string, ensuring it's valid for inference.

        Inference device availability is determined via OpenVINO (see `get_inference_devices`).

        Args:
            device_str: Device string in format '<target>[-<index>]'
                (e.g., 'auto', 'cpu', 'xpu', 'xpu-2')

        Returns:
            DeviceInfo: Information about the specified inference device
        """
        try:
            device_type, device_index = self._parse_device(device_str)
        except ValueError as ex:
            raise ValueError(f"Device '{device_str}' is not valid for inference.") from ex

        if device_type == DeviceType.CUDA:
            raise ValueError(f"Device '{device_str}' is not valid for inference (CUDA devices are not supported).")
        if device_type == DeviceType.AUTO:
            return DeviceInfo(type=DeviceType.AUTO, name="AUTO", memory=None, index=None)
        if device_type == DeviceType.CPU:
            return DeviceInfo(type=DeviceType.CPU, name="CPU", memory=None, index=None)

        for available_device in self.get_inference_devices():
            if device_type == available_device.type and device_index == (available_device.index or 0):
                return available_device

        raise ValueError(f"Device '{device_str}' is not available for inference on the system.")

    def get_training_device_info(self, device_str: str) -> DeviceInfo:
        """
        Get DeviceInfo for a given device string, ensuring it's valid for training.

        Args:
            device_str: Device string in format '<target>[-<index>]'
                (e.g., 'auto', 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

        Returns:
            DeviceInfo: Information about the specified training device
        """
        # For training, all devices are currently valid, but we can add custom validation here if needed in the future
        return self.get_device_info(device_str)

    @staticmethod
    def _parse_device(device_str: str) -> tuple[DeviceType, int]:
        """
        Parse device string into type and index

        Args:
            device_str: Device string in format '<target>[-<index>]'
                (e.g., 'auto', 'cpu', 'xpu', 'cuda', 'xpu-2', 'cuda-1')

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
    def supports_int8(device_info: DeviceInfo) -> bool:
        """
        Check if the given device supports INT8 inference using OpenVINO.

        For CPU devices, INT8 is always supported.
        For GPU devices, the check is done via OpenVINO's OPTIMIZATION_CAPABILITIES property.

        Args:
            device_info: The device to check.

        Returns:
            bool: True if the device supports INT8 inference, False otherwise.
        """
        if device_info.type == DeviceType.CPU:
            return True
        if device_info.type == DeviceType.CUDA:
            return False
        try:
            from model_api.adapters import create_core

            core = create_core()
            ov_device = device_info.as_openvino
            capabilities = core.get_property(device_name=ov_device, property="OPTIMIZATION_CAPABILITIES")
            return "INT8" in capabilities
        except Exception:
            logger.exception(
                "Failed to query INT8 support for device '{}' (OpenVINO device '{}'). Assuming not supported.",
                device_info,
                device_info.as_openvino,
            )
            return False

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
