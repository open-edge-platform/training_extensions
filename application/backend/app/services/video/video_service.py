# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import av
import numpy as np
from loguru import logger

from .interface import IVideoService, VideoMetadata


def _frame_index_from_pts(frame: av.VideoFrame, time_base: float, avg_rate: float) -> int:
    """Compute the zero-based frame index from a decoded frame's PTS."""
    return round(frame.pts * time_base * avg_rate)


def _pts_from_index(index: int, time_base: float, avg_rate: float) -> int:
    """Compute the approximate PTS value for a given frame index."""
    return int(index / (avg_rate * time_base))


def _group_consecutive(sorted_indexes: list[int], gap: int = 8) -> list[list[int]]:
    """Group sorted frame indexes into runs where neighbours are within *gap* frames.

    Frames within a small gap are cheaper to decode sequentially than to seek
    to individually, so they are kept in the same group.

    Args:
        sorted_indexes: Ascending, deduplicated frame indexes.
        gap: Maximum distance between consecutive indexes in a single group.

    Returns:
        List of groups, each a list of ascending frame indexes.
    """
    if not sorted_indexes:
        return []
    groups: list[list[int]] = [[sorted_indexes[0]]]
    for idx in sorted_indexes[1:]:
        if idx - groups[-1][-1] <= gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])
    return groups


def _decode_group(
    container: av.container.InputContainer,
    stream: av.video.stream.VideoStream,
    group: list[int],
    result: dict[int, np.ndarray],
    time_base: float,
    avg_rate: float,
) -> None:
    """Seek once to the first index of *group* and decode forward, collecting all needed frames.

    Args:
        container: Open PyAV input container.
        stream: Video stream to decode.
        group: Ascending list of frame indexes to extract (must be non-empty).
        result: Dictionary to populate with ``{frame_index: rgb_ndarray}``.
        time_base: ``float(stream.time_base)``.
        avg_rate: ``float(stream.average_rate)``.
    """
    target_set = set(group)
    first_target = group[0]
    last_target = group[-1]

    seek_pts = max(_pts_from_index(first_target, time_base, avg_rate) - 1, 0)
    container.seek(seek_pts, stream=stream)

    for frame in container.decode(stream):
        frame_idx = _frame_index_from_pts(frame, time_base, avg_rate)
        if frame_idx < first_target:
            continue
        if frame_idx in target_set:
            result[frame_idx] = frame.to_ndarray(format="rgb24")
            target_set.discard(frame_idx)
            if not target_set:
                return
        if frame_idx > last_target:
            break


class VideoService(IVideoService):
    """Direct video frame extraction using PyAV (ffmpeg) without caching."""

    def __init__(self, av_options: dict[str, str] | None = None) -> None:
        """
        Initialise the video service.

        Args:
            av_options: Optional FFmpeg demuxer options passed to ``av.open()``.
        """
        self._av_options = av_options or {}

    def get_video_metadata(self, video_path: Path) -> VideoMetadata:
        """
        Extract video metadata (dimensions, frame count, FPS) from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            VideoMetadata with width, height, frame_count and fps.

        Raises:
            RuntimeError: If the video cannot be opened or metadata extraction fails.
        """
        try:
            with av.open(str(video_path), options=self._av_options) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 0.0
                frame_count = stream.frames if stream.frames else 0
                width = stream.codec_context.width
                height = stream.codec_context.height
        except Exception as e:
            logger.error(f"Failed getting metadata for video {video_path}", exc_info=e)
            raise RuntimeError("Error occurred while getting video metadata")
        return VideoMetadata(width=width, height=height, frame_count=frame_count, fps=fps)

    def extract_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        """
        Extract a single frame from a video file.

        Args:
            video_path: Path to the video file.
            frame_index: Zero-based index of the frame to extract.

        Returns:
            Extracted frame as a numpy array in RGB format.

        Raises:
            RuntimeError: If the video cannot be opened or the frame cannot be read.
        """
        frames = self.extract_frames(video_path=video_path, frame_indexes=[frame_index])
        return frames[frame_index]

    def extract_frames(self, video_path: Path, frame_indexes: list[int]) -> dict[int, np.ndarray]:
        """
        Extract multiple frames from a video file in a single pass.

        Consecutive or nearby frames are grouped and decoded with a single seek
        to minimise I/O overhead.  Multithreaded codec decoding is enabled.

        Args:
            video_path: Path to the video file.
            frame_indexes: List of zero-based frame indexes to extract.

        Returns:
            Dictionary mapping each requested frame index to its numpy array in RGB format.

        Raises:
            RuntimeError: If the video cannot be opened or any frame cannot be read.
        """
        if not frame_indexes:
            return {}
        sorted_indexes = sorted(set(frame_indexes))
        groups = _group_consecutive(sorted_indexes)
        result: dict[int, np.ndarray] = {}
        try:
            with av.open(str(video_path), options=self._av_options) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                time_base = float(stream.time_base)
                avg_rate = float(stream.average_rate)
                for group in groups:
                    _decode_group(container, stream, group, result, time_base, avg_rate)
        except av.error.FileNotFoundError:
            raise RuntimeError(f"Cannot open video: {video_path}")

        missing = set(sorted_indexes) - result.keys()
        if missing:
            raise RuntimeError(f"Cannot read frames {sorted(missing)} from video: {video_path}")
        return result
