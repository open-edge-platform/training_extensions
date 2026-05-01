# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.codec.context import ThreadType
from loguru import logger
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    width: int = Field(..., description="Video width", ge=0)
    height: int = Field(..., description="Video height", ge=0)
    frame_count: int = Field(..., description="Video frames number", ge=0)
    fps: float = Field(..., description="Video frames per second", ge=0)


def _frame_index_from_pts(frame_pts: int, time_base: Fraction, avg_rate: Fraction) -> int:
    """Compute the zero-based frame index from a decoded frame's PTS."""
    return round(frame_pts * time_base * avg_rate)


def _pts_from_index(index: int, time_base: Fraction, avg_rate: Fraction) -> int:
    """Compute the approximate PTS value for a given frame index."""
    return int(Fraction(index) / (avg_rate * time_base))


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
    time_base: Fraction,
    avg_rate: Fraction,
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

    if first_target == 0:
        # Seek to the very beginning - decode count equals frame index.
        container.seek(0, stream=stream)
        decode_offset = 0
    else:
        seek_pts = max(_pts_from_index(first_target, time_base, avg_rate) - 1, 0)
        container.seek(seek_pts, stream=stream)
        # We don't know exactly which frame the seek landed on, so we
        # calibrate using the *minimum* PTS seen in the first few decoded
        # frames (to handle B-frame reordering).
        decode_offset = None  # will be set on the first frame

    # We rely on a simple decode counter to assign frame indexes.
    # PTS-based indexing is only used once - to figure out which frame
    # the seek landed on (for non-zero seeks).  After that, every call
    # to container.decode() yields the next presentation-order frame, so
    # a plain counter is the most reliable approach, especially for
    # containers (AVI) with non-monotonic or offset PTS values.
    frame_counter = 0

    for frame in container.decode(stream):
        if decode_offset is None:
            # First frame after a non-zero seek: calibrate offset from PTS.
            if frame.pts is not None:
                decode_offset = _frame_index_from_pts(frame.pts, time_base, avg_rate)
            else:
                # No PTS at all - best guess is the seek target itself.
                decode_offset = first_target

        frame_idx = decode_offset + frame_counter
        frame_counter += 1

        if frame_idx < first_target:
            continue
        if frame_idx in target_set:
            result[frame_idx] = frame.to_ndarray(format="rgb24")
            target_set.discard(frame_idx)
            if not target_set:
                return
        if frame_idx > last_target + 16:
            # Allow some overshoot for B-frame reordering before giving up
            break


def get_video_metadata(video_path: Path) -> VideoMetadata:
    """
    Extracts a video metadata

    Args:
        video_path: Video binary file path
    """
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else 0.0
            frame_count = stream.frames if stream.frames else 0
            if frame_count == 0 and fps > 0 and stream.duration and stream.time_base:
                frame_count = round(float(stream.duration * stream.time_base) * fps)
            if frame_count == 0 and fps > 0 and container.duration:
                frame_count = round((container.duration / av.time_base) * fps)
            width = stream.codec_context.width
            height = stream.codec_context.height
    except Exception as e:
        logger.error(f"Failed getting metadata for video {video_path}", exc_info=e)
        raise RuntimeError("Error occurred while getting video metadata")

    return VideoMetadata(
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
    )


def extract_video_frame(
    video_path: Path,
    frame_index: int,
) -> np.ndarray:
    """
    Extracts a video frame.

    Args:
        video_path: Video binary file path
        frame_index: Frame index

    Returns:
        Extracted video frame as numpy array (RGB format)
    """
    frames = extract_video_frames(video_path=video_path, frame_indexes=[frame_index])
    return frames[frame_index]


def extract_video_frames(
    video_path: Path,
    frame_indexes: list[int],
) -> dict[int, np.ndarray]:
    """
    Extracts multiple video frames in a single pass over the video file.
    The video is opened once and frames are read in ascending index order for optimal sequential access.

    Args:
        video_path: Video binary file path
        frame_indexes: List of frame indexes to extract

    Returns:
        Dictionary mapping frame index to the extracted frame as numpy array (RGB format)
    """
    if not frame_indexes:
        return {}
    sorted_indexes = sorted(set(frame_indexes))
    groups = _group_consecutive(sorted_indexes)

    frames: dict[int, np.ndarray] = {}
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = ThreadType.AUTO
            if stream.time_base is None or stream.average_rate is None:
                raise RuntimeError(f"Video stream is missing time_base or average_rate: {video_path}")
            time_base = Fraction(stream.time_base)
            avg_rate = Fraction(stream.average_rate)
            for group in groups:
                _decode_group(container, stream, group, frames, time_base, avg_rate)
    except av.error.FileNotFoundError as exc:
        logger.exception("Cannot open video: {}", video_path)
        raise RuntimeError(f"Cannot open video: {video_path}") from exc
    except (av.error.FFmpegError, IndexError, OSError) as exc:
        logger.exception("Cannot extract frames from video {}: {}", video_path, exc)
        raise RuntimeError(f"Cannot extract frames from video: {video_path}") from exc
    missing = set(sorted_indexes) - frames.keys()
    if missing:
        raise RuntimeError(f"Cannot read frames {sorted(missing)} from video: {video_path}")
    return frames
