# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import cv2
from loguru import logger

from app.models import SourceType
from app.stream.base_opencv_stream import BaseOpenCVStream

_USB_CAMERA_BACKENDS: dict[str, list[int]] = {
    "Linux": [cv2.CAP_V4L2, cv2.CAP_ANY],
    "Windows": [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY],
    "Darwin": [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY],
}

_BACKEND_NAMES: dict[int, str] = {
    cv2.CAP_MSMF: "MSMF",
    cv2.CAP_DSHOW: "DirectShow",
    cv2.CAP_V4L2: "V4L2",
    cv2.CAP_AVFOUNDATION: "AVFoundation",
    cv2.CAP_ANY: "auto",
}


class USBCameraStream(BaseOpenCVStream):
    """Video stream implementation using USB Camera via OpenCV."""

    def __init__(self, device_id: int = 0, codec: str | None = None) -> None:
        """Initialize USB Camera stream."""
        backends = _USB_CAMERA_BACKENDS.get(platform.system(), [cv2.CAP_ANY])
        backend = self._find_working_backend(device_id, backends)
        super().__init__(
            source=device_id,
            source_type=SourceType.USB_CAMERA,
            codec=codec,
            api_preference=backend,
            device_id=device_id,
        )

    @staticmethod
    def _find_working_backend(device_id: int, backends: list[int]) -> int:
        """Try each backend in order and return the first one that can open the device.

        Args:
            device_id: The camera device ID.
            backends: Ordered list of OpenCV backend constants to try.

        Returns:
            The first backend that successfully opens the device.

        Raises:
            RuntimeError: If no backend can open the device.
        """
        for backend in backends:
            cap = cv2.VideoCapture(device_id, backend)
            if cap.isOpened():
                cap.release()
                logger.info(
                    "USB camera device {} opened with {} backend",
                    device_id,
                    _BACKEND_NAMES.get(backend, str(backend)),
                )
                return backend
            cap.release()
            logger.warning(
                "USB camera device {} could not be opened with {} backend, trying next",
                device_id,
                _BACKEND_NAMES.get(backend, str(backend)),
            )
        raise RuntimeError(
            f"Could not open USB camera device {device_id} with any backend "
            f"({', '.join(_BACKEND_NAMES.get(b, str(b)) for b in backends)})"
        )

    def is_real_time(self) -> bool:
        return True
