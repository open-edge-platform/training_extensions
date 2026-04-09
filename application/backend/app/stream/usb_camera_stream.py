# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import cv2

from app.models import SourceType
from app.stream.base_opencv_stream import BaseOpenCVStream

_USB_CAMERA_BACKENDS: dict[str, int] = {
    "Linux": cv2.CAP_V4L2,
    "Windows": cv2.CAP_MSMF,
    "Darwin": cv2.CAP_AVFOUNDATION,
}


class USBCameraStream(BaseOpenCVStream):
    """Video stream implementation using USB Camera via OpenCV."""

    def __init__(self, device_id: int = 0, codec: str | None = None) -> None:
        """Initialize USB Camera stream."""
        backend = _USB_CAMERA_BACKENDS.get(platform.system(), cv2.CAP_ANY)
        super().__init__(
            source=device_id,
            source_type=SourceType.USB_CAMERA,
            codec=codec,
            api_preference=backend,
            device_id=device_id,
        )

    def is_real_time(self) -> bool:
        return True
