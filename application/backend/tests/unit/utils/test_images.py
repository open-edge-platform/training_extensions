# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import io
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from app.utils.images import (
    convert_to_jpeg_compatible,
    crop_to_thumbnail,
    is_high_bit_depth_image,
    needs_display_normalization,
    normalize_high_bit_depth_image,
    normalize_image_to_png_bytes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image_in_mode(mode: str) -> PILImage.Image:
    """Return a small PIL image in the requested mode.

    Modes that PIL can only produce by loading from a file (e.g. "I;16") are
    constructed via an in-memory TIFF round-trip.
    """
    if mode == "I":
        return PILImage.fromarray(np.zeros((4, 4), dtype=np.int32), mode="I")
    if mode == "F":
        return PILImage.fromarray(np.zeros((4, 4), dtype=np.float32), mode="F")
    # "I;16" and variants: round-trip through an in-memory TIFF
    buf = io.BytesIO()
    PILImage.fromarray(np.zeros((4, 4), dtype=np.uint16), mode="I;16").save(buf, format="TIFF")
    buf.seek(0)
    img = PILImage.open(buf)
    img.load()
    return img


def _save_tmp(image: PILImage.Image, tmp_path: Path, filename: str) -> Path:
    path = tmp_path / filename
    image.save(path)
    return path


# ---------------------------------------------------------------------------
# is_high_bit_depth_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["I", "F"])
def test_is_high_bit_depth_image(mode: str) -> None:
    assert is_high_bit_depth_image(_make_image_in_mode(mode)) is True


def test_is_high_bit_depth_image_16bit_tiff(tmp_path: Path) -> None:
    """I;16 mode is only exposed after loading a real TIFF file."""
    path = _save_tmp(PILImage.fromarray(np.zeros((4, 4), dtype=np.uint16), mode="I;16"), tmp_path, "t.tif")
    with PILImage.open(path) as img:
        assert is_high_bit_depth_image(img) is True


@pytest.mark.parametrize("mode", ["RGB", "L", "RGBA", "P", "CMYK"])
def test_is_high_bit_depth_image_standard_modes(mode: str) -> None:
    assert is_high_bit_depth_image(PILImage.new(mode, (4, 4))) is False


# ---------------------------------------------------------------------------
# needs_display_normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image, filename, expected",
    [
        pytest.param(
            PILImage.fromarray(np.ones((8, 8), dtype=np.uint16) * 1000, mode="I;16"),
            "img.tif",
            True,
            id="16bit_tiff",
        ),
        pytest.param(
            PILImage.fromarray(np.ones((8, 8), dtype=np.float32) * 0.5, mode="F"),
            "img_float.tif",
            True,
            id="32bit_float_tiff",
        ),
        pytest.param(
            PILImage.new("RGB", (8, 8), color=(128, 64, 32)),
            "img.png",
            False,
            id="8bit_rgb_png",
        ),
        pytest.param(
            PILImage.new("RGB", (8, 8), color=(10, 20, 30)),
            "img.jpg",
            False,
            id="8bit_jpeg",
        ),
    ],
)
def test_needs_display_normalization(image: PILImage.Image, filename: str, expected: bool, tmp_path: Path) -> None:
    path = _save_tmp(image, tmp_path, filename)
    assert needs_display_normalization(path) is expected


# ---------------------------------------------------------------------------
# normalize_high_bit_depth_image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "image",
    [
        pytest.param(PILImage.fromarray(np.ones((8, 8), dtype=np.uint16) * 1000, mode="I;16"), id="16bit_int"),
        pytest.param(
            PILImage.fromarray(np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8), mode="F"), id="32bit_float"
        ),
    ],
)
def test_normalize_high_bit_depth_image(image: PILImage.Image) -> None:
    result = normalize_high_bit_depth_image(image)
    assert result.mode == "RGB"
    assert result.size == image.size


def test_normalize_high_bit_depth_image_uniform_becomes_black() -> None:
    """A uniform image (lo == hi) must produce all-zero output, not divide-by-zero."""
    img = PILImage.fromarray(np.full((4, 4), fill_value=500, dtype=np.uint16), mode="I;16")
    assert np.array(normalize_high_bit_depth_image(img)).max() == 0


def test_normalize_high_bit_depth_image_maps_full_range() -> None:
    """Min pixel maps to 0 and max pixel maps to 255 after normalization."""
    arr = np.array([[500, 1000]], dtype=np.int32)
    result = normalize_high_bit_depth_image(PILImage.fromarray(arr, mode="I")).convert("L")
    pixels = np.array(result)
    assert pixels.min() == 0
    assert pixels.max() == 255


# ---------------------------------------------------------------------------
# normalize_image_to_png_bytes
# ---------------------------------------------------------------------------


def test_normalize_image_to_png_bytes(tmp_path: Path) -> None:
    path = _save_tmp(PILImage.fromarray(np.ones((8, 8), dtype=np.uint16) * 1000, mode="I;16"), tmp_path, "t.tif")
    result = normalize_image_to_png_bytes(path)
    assert isinstance(result, bytes) and len(result) > 0
    reloaded = PILImage.open(BytesIO(result))
    assert reloaded.format == "PNG"
    assert reloaded.mode == "RGB"


def test_normalize_image_to_png_bytes_narrow_range(tmp_path: Path) -> None:
    """Values in a narrow range (e.g. [500, 1000]) must span the full [0, 255] output."""
    arr = np.array([[500, 1000]], dtype=np.int32)
    path = _save_tmp(PILImage.fromarray(arr, mode="I"), tmp_path, "t.tif")
    pixels = np.array(PILImage.open(BytesIO(normalize_image_to_png_bytes(path))).convert("L"))
    assert pixels.min() == 0
    assert pixels.max() == 255


# ---------------------------------------------------------------------------
# convert_to_jpeg_compatible
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["RGB", "L", "CMYK"])
def test_convert_to_jpeg_compatible_already_compatible(mode: str) -> None:
    img = PILImage.new(mode, (4, 4))
    assert convert_to_jpeg_compatible(img) is img


@pytest.mark.parametrize(
    "mode, expected_mode",
    [
        pytest.param("RGBA", "RGB", id="rgba_to_rgb"),
        pytest.param("P", "RGB", id="palette_to_rgb"),
    ],
)
def test_convert_to_jpeg_compatible_non_jpeg_modes(mode: str, expected_mode: str) -> None:
    assert convert_to_jpeg_compatible(PILImage.new(mode, (4, 4))).mode == expected_mode


def test_convert_to_jpeg_compatible_high_bit_depth() -> None:
    """High bit depth images are normalized to RGB, not just cast."""
    img = PILImage.fromarray(np.full((4, 4), fill_value=1000, dtype=np.uint16), mode="I;16")
    result = convert_to_jpeg_compatible(img)
    assert result.mode == "RGB"
    # Uniform value → all zeros after min-max normalization
    assert np.array(result).max() == 0


# ---------------------------------------------------------------------------
# crop_to_thumbnail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_size, target_h, target_w",
    [
        pytest.param((200, 100), 50, 50, id="landscape_to_square"),
        pytest.param((100, 200), 50, 50, id="portrait_to_square"),
        pytest.param((10, 10), 64, 64, id="upscale_small_image"),
        pytest.param((128, 128), 32, 64, id="asymmetric_target"),
    ],
)
def test_crop_to_thumbnail(src_size: tuple[int, int], target_h: int, target_w: int) -> None:
    result = crop_to_thumbnail(PILImage.new("RGB", src_size), target_height=target_h, target_width=target_w)
    assert result.size == (target_w, target_h)
