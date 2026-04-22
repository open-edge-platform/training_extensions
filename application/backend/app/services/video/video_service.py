# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    width: int = Field(..., description="Video width", ge=0)
    height: int = Field(..., description="Video height", ge=0)
    frame_count: int = Field(..., description="Video frames number", ge=0)
    fps: float = Field(..., description="Video frames per second", ge=0)


def get_video_metadata(video_path: Path) -> VideoMetadata:
    """
    Extracts a video metadata

    Args:
        video_path: Video binary file path
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
    except Exception as e:
        logger.error(f"Failed getting metadata for video {video_path}", exc_info=e)
        raise RuntimeError("Error occurred while getting video metadata")
    finally:
        cap.release()

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

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: dict[int, np.ndarray] = {}
    try:
        sorted_indexes = sorted(set(frame_indexes))
        first_index = sorted_indexes[0]
        last_index = sorted_indexes[-1]
        requested_indexes = set(sorted_indexes)
        if not cap.set(cv2.CAP_PROP_POS_FRAMES, first_index):
            raise RuntimeError(f"Cannot seek to frame at {first_index} index in video: {video_path}")
        current_index = first_index
        while current_index <= last_index:
            read_success, frame = cap.read()
            if not read_success:
                raise RuntimeError(f"Cannot read frame at {current_index} index from video: {video_path}")
            if current_index in requested_indexes:
                frames[current_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_index += 1
        return frames
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Failed extracting video frames from video {video_path}", exc_info=e)
        raise RuntimeError("Error occurred while extracting video frames")
    finally:
        cap.release()
