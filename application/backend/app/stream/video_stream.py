# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Self

from app.stream.stream_data import StreamData


class VideoStream(ABC):
    """Abstract base class for video stream implementations."""

    @abstractmethod
    def get_data(self) -> StreamData | None:
        """Get the latest frame from the video stream if available.
        Returns:
            np.ndarray: The latest frame as a numpy array or None if no frame is available
        """

    @abstractmethod
    def is_real_time(self) -> bool:
        """Check if the video stream is real-time.
        Returns:
            bool: True if the video stream is real-time, False otherwise
        """

    def is_finished(self) -> bool:
        """Check if the stream has permanently ended and will not produce any further frames.

        Real-time and looping streams never finish; finite sources (e.g. a non-looping
        video file) report True once fully consumed so consumers can stop polling.

        Returns:
            bool: True if the stream is exhausted and should be stopped, False otherwise
        """
        return False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        self.release()

    @abstractmethod
    def release(self) -> None:
        """Release the video stream resources."""

    def __iter__(self) -> Iterator[StreamData | None]:
        while True:
            try:
                yield self.get_data()
            except RuntimeError as exc:
                self.release()
                raise StopIteration from exc
