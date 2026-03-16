# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import secrets
from enum import StrEnum
from itertools import count as iter_count
from pathlib import Path

import cv2
import numpy as np
from datumaro.experimental import LazyImage, LazyVideoFrame, MediaInfo
from PIL import Image


def generate_random_image(output_path: Path, suffix: str, width: int = 640, height: int = 480) -> Path:
    """Generate a random test image and saves it to the specified path.

    Returns:
        Path: Path to the generated image.
    """
    # Create random RGB image
    random_color = (secrets.randbelow(256), secrets.randbelow(256), secrets.randbelow(256))
    img = Image.new("RGB", (width, height), color=random_color)

    filename = f"{suffix}_test_image_{secrets.token_hex(8)}.jpg"
    img.save(f"{output_path}/{filename}", format="JPEG")

    return output_path / filename


def generate_random_video(
    output_path: Path, width: int = 640, height: int = 480, fps: int = 30, duration: int = 5
) -> Path:
    """
    Generate a random test video.

    Returns:
        Path: Path to the generated video.
    """
    filename = f"test_video_{secrets.token_hex(8)}.mp4"
    video_path = output_path / filename
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for frame_num in range(fps * duration):
        random_color = (secrets.randbelow(256), secrets.randbelow(256), secrets.randbelow(256))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Draw something on each frame
        cv2.putText(frame, f"Frame {frame_num}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, random_color, 2)
        out.write(frame)

    out.release()
    return video_path


class MediaType(StrEnum):
    IMAGE = "image"
    VIDEO_FRAME = "video frame"


class MediaProvider:
    """Knows how to produce a (lazy_media, media_info) pair for a given media type."""

    def __init__(self, media_type: str, tmp_path: Path) -> None:
        self._media_type = media_type
        self._frame_counter = iter_count()
        self._video_path: Path | None = None

        match media_type:
            case MediaType.IMAGE:
                self._media_dir = tmp_path / "images"
                self._media_dir.mkdir(parents=True, exist_ok=True)
            case MediaType.VIDEO_FRAME:
                self._media_dir = tmp_path / "videos"
                self._media_dir.mkdir(parents=True, exist_ok=True)
                self._video_path = generate_random_video(self._media_dir)
            case _:
                raise ValueError(f"Unsupported media type: '{media_type}'")

    def next(self, subset: str) -> tuple[LazyImage | LazyVideoFrame, MediaInfo]:
        """Return the next (lazy_media, media_info) pair."""
        if self._media_type == "image":
            path = generate_random_image(output_path=self._media_dir, suffix=subset)
            return LazyImage(path), MediaInfo(width=640, height=480)

        frame_idx = next(self._frame_counter)
        lazy = LazyVideoFrame(video_path=str(self._video_path), frame_index=frame_idx)
        return lazy, MediaInfo.from_media(lazy)
