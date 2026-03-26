# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
from PIL import Image


def crop_to_thumbnail(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    """
    Crop an image to a thumbnail using maximal visible area preservation.

    The image is first scaled according to the side with the least amount of
    rescaling, and then the other side is cropped. This approach ensures that
    a maximal portion of the original image is visible in the final thumbnail.

    Args:
        image: PIL Image object to generate thumbnail for.
        target_height: Target height in pixels to crop the thumbnail to.
        target_width: Target width in pixels to crop the thumbnail to.

    Returns:
        Image.Image: Cropped and resized thumbnail image.

    Note:
        Uses center cropping after scaling to maintain the most important
        visual content from the original image.
    """
    scale_width = target_width / image.width
    scale_height = target_height / image.height
    scaling_factor = max(scale_width, scale_height)
    resized_image = image.resize((int(image.width * scaling_factor), int(image.height * scaling_factor)))
    # cropping
    x1 = (resized_image.width - target_width) / 2
    x2 = x1 + target_width
    y1 = (resized_image.height - target_height) / 2
    y2 = y1 + target_height
    x1 = round(max(x1, 0))
    x2 = round(min(x2, resized_image.width))
    y1 = round(max(y1, 0))
    y2 = round(min(y2, resized_image.height))
    return resized_image.crop((x1, y1, x2, y2))


def convert_to_jpeg_compatible(image: Image.Image) -> Image.Image:
    """Convert an image to a JPEG-compatible mode (RGB, L, or CMYK).

    16-bit modes (I;16, I) require explicit normalization: PIL's built-in
    RGB conversion only preserves the most-significant byte, clipping the
    lower half of the dynamic range and producing washed-out thumbnails.
    We normalize the full value range to [0, 255] before converting to L.
    """
    if image.mode in ("RGB", "L", "CMYK"):
        return image
    if image.mode in ("I;16", "I;16B", "I;16L", "I;16S", "I;16BS", "I"):
        # Normalize the 16-bit (or 32-bit signed int) range to 8-bit
        image = image.convert("I")
        arr = np.array(image, dtype=np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")
    # All other non-JPEG-compatible modes (RGBA, P, F, …)
    return image.convert("RGB")
