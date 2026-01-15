# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from app.models import SourceType
from app.stream.base_opencv_stream import BaseOpenCVStream


class USBCameraStream(BaseOpenCVStream):
    """Video stream implementation using USB Camera via OpenCV."""

    def __init__(self, device_id: int = 0, codec: str | None = None) -> None:
        """Initialize USB Camera stream."""
        super().__init__(source=device_id, source_type=SourceType.USB_CAMERA, codec=codec, device_id=device_id)

    def is_real_time(self) -> bool:
        return True
