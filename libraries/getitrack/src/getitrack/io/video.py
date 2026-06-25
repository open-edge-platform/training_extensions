# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""OpenCV-backed video reader and writer.

`VideoReader` iterates BGR uint8 frames from a video file. `VideoWriter`
writes BGR uint8 frames to a video file, creating parent directories on
demand. Both are context managers and release their OpenCV handles on close.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType

    import numpy as np

_FRAME_CHANNELS = 3


class VideoReader:
    """Sequential frame reader over a video file.

    Iterating yields BGR uint8 arrays of shape ``(height, width, 3)`` in
    decode order. Metadata (fps, frame size, frame count) is exposed as
    properties read from the container header.

    Example:
        >>> with VideoReader("input.mp4") as reader:
        ...     for frame in reader:
        ...         process(frame)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            msg = f"video file not found: {self.path}"
            raise FileNotFoundError(msg)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            msg = f"OpenCV could not open video: {self.path}"
            raise ValueError(msg)

    @property
    def fps(self) -> float:
        """Frames per second reported by the container."""
        return float(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        """Number of frames reported by the container header.

        Some containers report an estimate; the true count is the number
        of frames actually yielded by iteration.
        """
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame

    def close(self) -> None:
        """Release the underlying OpenCV capture handle."""
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class VideoWriter:
    """Video writer for annotated tracking output.

    Frames must be BGR uint8 arrays whose size matches ``frame_size``.
    Parent directories of ``path`` are created automatically.

    Example:
        >>> with VideoWriter("out.mp4", fps=30.0, frame_size=(640, 480)) as writer:
        ...     writer.write(frame)
    """

    def __init__(
        self,
        path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: str = "mp4v",
    ) -> None:
        """Open a video file for writing.

        Args:
            path: Destination video path.
            fps: Output frame rate.
            frame_size: ``(width, height)`` of every frame.
            codec: FourCC codec identifier.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter.fourcc(*codec)
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, frame_size)
        if not self._writer.isOpened():
            msg = f"OpenCV could not open video for writing: {self.path} (codec '{codec}')"
            raise ValueError(msg)
        self.frames_written = 0

    def write(self, frame: np.ndarray) -> None:
        """Append one BGR uint8 frame.

        Args:
            frame: ``(height, width, 3)`` uint8 array matching the
                ``frame_size`` given at construction.

        Raises:
            ValueError: If the frame shape does not match ``frame_size``.
        """
        expected = (self._frame_size[1], self._frame_size[0], _FRAME_CHANNELS)
        if frame.shape != expected:
            msg = f"frame shape {frame.shape} does not match expected {expected}"
            raise ValueError(msg)
        self._writer.write(frame)
        self.frames_written += 1

    def close(self) -> None:
        """Finalise the container and release the OpenCV writer handle."""
        self._writer.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
