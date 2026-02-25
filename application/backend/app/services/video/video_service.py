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
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        # Read the frame at the requested position
        read_success, frame = cap.read()
        if not read_success:
            raise RuntimeError(f"Cannot read frame at {frame_index} index from video: {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Failed extracting video frame {frame_index} from video {video_path}", exc_info=e)
        raise RuntimeError("Error occurred while extracting video frame")
    finally:
        cap.release()
