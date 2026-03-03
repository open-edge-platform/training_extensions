# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
import secrets

from PIL import Image


def generate_random_image(width: int = 640, height: int = 480) -> tuple[io.BytesIO, str]:
    """Generate a random test image.

    Returns:
        tuple: (image_buffer, filename, extension)
    """
    # Create random RGB image
    img = Image.new(
        "RGB", (width, height), color=(secrets.randbelow(256), secrets.randbelow(256), secrets.randbelow(256))
    )

    # Save to buffer
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Generate random filename
    filename = f"test_image_{secrets.token_hex(8)}.jpg"

    return buffer, filename
