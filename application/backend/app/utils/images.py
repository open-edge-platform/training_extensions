# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
from PIL import Image

_HIGH_BIT_DEPTH_MODES = ("I;16", "I;16B", "I;16L", "I;16S", "I;16BS", "I", "F")


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


def is_high_bit_depth_image(image: Image.Image) -> bool:
    """Check if a PIL image is in a high bit depth mode (16-bit, 32-bit int, or float)."""
    return image.mode in _HIGH_BIT_DEPTH_MODES


def normalize_high_bit_depth_image(image: Image.Image) -> Image.Image:
    """Normalize a high bit depth image to 8-bit RGB using min-max scaling.

    This maps the actual pixel value range to [0, 255], making narrow-range
    images (e.g. 16-bit values in [500-1000]) visible instead of appearing black.

    Args:
        image: A PIL Image in a high bit depth mode (16-bit, 32-bit int, or float).

    Returns:
        An 8-bit RGB PIL Image with normalized pixel values.
    """
    arr = np.array(image, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)
    return Image.fromarray(arr.astype(np.uint8), mode="L").convert("RGB")


def convert_to_jpeg_compatible(image: Image.Image) -> Image.Image:
    """Convert an image to a JPEG-compatible mode (RGB, L, or CMYK).

    High bit depth modes (I;16, I, F) require explicit normalization: PIL's built-in
    RGB conversion only preserves the most-significant byte, clipping the
    lower half of the dynamic range and producing washed-out thumbnails.
    We normalize the full value range to [0, 255] before converting to L.
    """
    if image.mode in ("RGB", "L", "CMYK"):
        return image
    if is_high_bit_depth_image(image):
        return normalize_high_bit_depth_image(image)
    # All other non-JPEG-compatible modes (RGBA, P, …)
    return image.convert("RGB")


def needs_display_normalization(image_path: Path) -> bool:
    """Return True if the image at the given path requires normalization for display.

    Opens the image with deferred decoding (no pixel data read) and checks the mode.
    This is cheap enough to call on every request for candidate formats (TIF, PNG).

    Args:
        image_path: Path to the image file on disk.

    Returns:
        True if the image is high bit depth and should be normalized before display.
    """
    with Image.open(image_path) as img:
        return is_high_bit_depth_image(img)


def normalize_image_to_png_bytes(image_path: Path) -> bytes:
    """Normalize a high bit depth image to 8-bit PNG bytes on the fly.

    Opens the image at the given path, applies min-max normalization to map the
    full pixel value range to [0, 255], and returns the result encoded as PNG bytes.

    This function always performs normalization. Call :func:`needs_display_normalization`
    first to guard against unnecessary work for standard 8-bit images.

    Args:
        image_path: Path to the image file on disk.

    Returns:
        PNG-encoded bytes of the normalized 8-bit image.
    """
    from io import BytesIO

    with Image.open(image_path) as img:
        normalized_img = normalize_high_bit_depth_image(img)
        buffer = BytesIO()
        normalized_img.save(buffer, format="PNG")
        return buffer.getvalue()
